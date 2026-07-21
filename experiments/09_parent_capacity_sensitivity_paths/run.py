"""Run the H4.5 parent-capacity, sensitivity, and path diagnostics.

Purpose:
    Reuse one target-free EFS history per seed for a parent-count sweep,
    local terminal sensitivity, and shared-lambda path continuity.
Dependencies:
    NumPy, Matplotlib, and the parent ``4.5hypothesis`` modules.
Outputs:
    One timestamped directory below ``--output-root``.
Exact command:
    python run.py --output-root results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np


# ``4.5hypothesis`` is not a valid package name because it starts with a
# number and contains a dot. Appending its directory lets this experiment
# reuse the existing EFS equations without copying or changing them.
PARENT_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PARENT_DIRECTORY) not in sys.path:
    sys.path.insert(1, str(PARENT_DIRECTORY))

from data import (  # noqa: E402
    align_source,
    coordinate_scale,
    fit_simplex_weights,
    generate_chaotic_sources,
    hold_out_sources,
    source_distance,
)
from data import self_check as data_self_check  # noqa: E402
from efs import backward_replay, forward_history, passive_forward  # noqa: E402
from efs import self_check as efs_self_check  # noqa: E402
from evaluation import smoke_gate  # noqa: E402
from plotting import plot_all  # noqa: E402


DEFAULT_K_VALUES = (4, 8, 16, 32, 64, 128)
DEFAULT_DELTA_FACTORS = (1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1)
EXPERIMENT_NAMES = ("parent-sweep", "du-sensitivity", "lambda-path")
DEFAULT_EFS = {
    "epsilon": 0.01,
    "exponent_s": 3.0,
    "gamma": 0.001,
    "forward_steps": 2000,
    "beta": 0.1,
    "proximal_steps": 50,
}


def _json_default(value: object) -> object:
    """Convert NumPy scalars and arrays for readable JSON output."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, values: dict[str, object]) -> None:
    """Write one plain JSON object."""
    path.write_text(json.dumps(values, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _append_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Append homogeneous result rows and create the header once."""
    if not rows:
        return
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _new_run_directory(output_root: Path) -> Path:
    """Create one timestamped result directory without overwriting."""
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = output_root / f"h45_parent_capacity_sensitivity_paths_{stamp}"
    directory.mkdir(exist_ok=False)
    return directory


def _standardized_rmse(first: np.ndarray, second: np.ndarray, scale: np.ndarray) -> float:
    """Return complete-array RMSE after per-coordinate standardization."""
    difference = (np.asarray(first) - np.asarray(second)) / scale  # [...,D] / [D] -> [...,D]
    return float(np.sqrt(np.mean(difference * difference)))


def _raw_rmse(first: np.ndarray, second: np.ndarray) -> float:
    """Return unscaled complete-array RMSE."""
    difference = np.asarray(first) - np.asarray(second)
    return float(np.sqrt(np.mean(difference * difference)))


def _unit(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    """Return a unit vector, using ``fallback`` only for a zero norm."""
    norm = float(np.linalg.norm(vector))
    if norm > 1.0e-15:
        return np.asarray(vector, dtype=np.float64) / norm
    if fallback is None:
        raise ValueError("cannot normalize a zero vector")
    fallback_norm = float(np.linalg.norm(fallback))
    if fallback_norm <= 1.0e-15:
        raise ValueError("fallback is also zero")
    return np.asarray(fallback, dtype=np.float64) / fallback_norm


def _lambda_entropy(weights: np.ndarray) -> float:
    """Return Shannon entropy of one simplex vector using natural logarithms."""
    positive = weights[weights > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def _pair_signature(source: np.ndarray) -> np.ndarray:
    """Return all internal particle distances for one source ``[P,D]``."""
    particle_count = source.shape[0]
    values = [
        np.linalg.norm(source[first] - source[second])
        for first in range(particle_count)
        for second in range(first + 1, particle_count)
    ]
    return np.asarray(values, dtype=np.float64)


def _load_efs_parameters(args: argparse.Namespace) -> dict[str, float | int | str]:
    """Use explicit CLI values, then the parent best config, then defaults."""
    loaded: dict[str, object] = {}
    config_path = args.config.resolve()
    if not args.ignore_config and config_path.exists():
        with config_path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"EFS config must contain one JSON object: {config_path}")
        loaded = value

    parameters: dict[str, float | int | str] = {}
    for name, fallback in DEFAULT_EFS.items():
        explicit = getattr(args, name)
        value = explicit if explicit is not None else loaded.get(name, fallback)
        parameters[name] = int(value) if isinstance(fallback, int) else float(value)
    parameters["parameter_source"] = str(config_path) if loaded else "built-in defaults"
    return parameters


def _validate(args: argparse.Namespace, efs: dict[str, float | int | str]) -> None:
    """Reject settings that cannot define the requested arrays."""
    if args.source_count < 8 or args.targets < 1:
        raise ValueError("need at least eight memory sources and one held-out target")
    if not 1 <= args.particles_per_source <= 16:
        raise ValueError("particles per source must be in [1,16] for exact matching")
    if args.dimension < 2:
        raise ValueError("dimension must be at least two")
    if args.workers < 1 or args.workers > 32:
        raise ValueError("workers must be in [1,32]")
    if "lambda-path" in args.experiments and (args.path_pairs < 1 or args.path_pool < 4):
        raise ValueError("path-pairs must be positive and path-pool at least four")
    if "lambda-path" in args.experiments and not 0.0 < args.path_step <= 1.0:
        raise ValueError("path-step must lie in (0,1]")
    if "du-sensitivity" in args.experiments and (
        args.sensitivity_vertices < 1 or args.boundary_neighbors < 2
    ):
        raise ValueError("sensitivity-vertices must be positive and boundary-neighbors at least two")
    if "parent-sweep" in args.experiments and max(args.k_values) > args.source_count:
        raise ValueError("the largest K cannot exceed the memory source count")
    if "du-sensitivity" in args.experiments and any(value <= 0.0 for value in args.delta_factors):
        raise ValueError("all terminal perturbation factors must be positive")
    if "du-sensitivity" in args.experiments and min(args.delta_factors) > 1.0e-2:
        raise ValueError("at least one delta factor must be <= 0.01 for the small-perturbation decision")
    for name in ("epsilon", "exponent_s", "gamma", "beta"):
        if float(efs[name]) <= 0.0:
            raise ValueError(f"{name} must be positive")
    for name in ("forward_steps", "proximal_steps"):
        if int(efs[name]) < 1:
            raise ValueError(f"{name} must be positive")


def _replay_endpoints(
    terminal_sources: np.ndarray,
    history: np.ndarray,
    efs: dict[str, float | int | str],
    workers: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Replay ``[G,P,D]`` candidates and return only arrivals and residuals.

    Threads split the passive candidate axis. Every thread reads the same
    immutable history, and generated particles still never interact.
    """
    group_count = terminal_sources.shape[0]
    arrivals = np.empty_like(terminal_sources)  # shape: [G,P,D]
    max_residual = np.empty(group_count, dtype=np.float64)  # shape: [G]
    worker_count = min(workers, group_count)
    chunks = [chunk for chunk in np.array_split(np.arange(group_count), worker_count) if chunk.size]
    started = time.perf_counter()
    print(f"{label}: replaying {group_count} passive candidates with {worker_count} worker(s)")

    def replay_chunk(item: tuple[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        chunk_number, indices = item
        trajectory, residual, _ = backward_replay(
            terminal_sources[indices],  # advanced indexing [G_chunk,P,D]
            history,
            gamma=float(efs["gamma"]),
            epsilon=float(efs["epsilon"]),
            exponent_s=float(efs["exponent_s"]),
            beta=float(efs["beta"]),
            proximal_steps=int(efs["proximal_steps"]),
            method_names=[label] * indices.size,
            log_every=0,
        )
        # Only frame zero and one residual maximum per candidate are needed by
        # these experiments. Discarding full trajectories limits saved data.
        endpoint = trajectory[0]  # shape: [G_chunk,P,D]
        residual_max = np.max(residual, axis=(0, 2))  # [J,G_chunk,P] -> [G_chunk]
        print(f"  {label} chunk {chunk_number + 1}/{len(chunks)} complete")
        return indices, endpoint, residual_max

    if worker_count == 1:
        results = [replay_chunk((0, chunks[0]))]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(replay_chunk, enumerate(chunks)))
    for indices, endpoint, residual_max in results:
        arrivals[indices] = endpoint
        max_residual[indices] = residual_max
    seconds = time.perf_counter() - started
    return arrivals, max_residual, seconds


def _rank_target_neighbors(
    memory_sources: np.ndarray,
    target_sources: np.ndarray,
    scale: np.ndarray,
    maximum_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank d1 sources once and save target-to-parent particle matching.

    Returns local memory IDs ``[R,Kmax]``, permutations ``[R,Kmax,P]``, and
    standardized source distances ``[R,Kmax]``.
    """
    target_count, particle_count, _ = target_sources.shape
    parent_ids = np.empty((target_count, maximum_k), dtype=np.int64)  # [R,Kmax]
    permutations = np.empty((target_count, maximum_k, particle_count), dtype=np.int64)  # [R,Kmax,P]
    distances = np.empty((target_count, maximum_k), dtype=np.float64)  # [R,Kmax]

    for target_index, target in enumerate(target_sources):
        all_distances = np.empty(memory_sources.shape[0], dtype=np.float64)  # shape: [C]
        all_permutations = np.empty((memory_sources.shape[0], particle_count), dtype=np.int64)  # [C,P]
        for source_id, source in enumerate(memory_sources):
            _, permutation, distance = align_source(target, source, scale)
            all_distances[source_id] = distance
            all_permutations[source_id] = permutation
        order = np.argsort(all_distances)[:maximum_k]  # shape: [Kmax]
        parent_ids[target_index] = order
        permutations[target_index] = all_permutations[order]
        distances[target_index] = all_distances[order]
    return parent_ids, permutations, distances


def _prepare_field(
    seed: int,
    args: argparse.Namespace,
    efs: dict[str, float | int | str],
    directory: Path,
) -> dict[str, object]:
    """Create one target-free memory field and its held-out target witnesses."""
    rng = np.random.default_rng(seed)
    all_sources = generate_chaotic_sources(
        args.source_count + args.targets,
        args.particles_per_source,
        args.dimension,
        args.data_scale,
        rng,
    )  # shape: [C+R,P,D]
    split = hold_out_sources(all_sources, args.targets, rng)
    memory_sources = split["memory_sources"]  # shape: [C,P,D]
    target_sources = split["target_sources"]  # shape: [R,P,D]
    initial_scale = coordinate_scale(memory_sources)  # shape: [D]
    if "parent-sweep" in args.experiments:
        parent_ids, parent_permutations, parent_distances = _rank_target_neighbors(
            memory_sources,
            target_sources,
            initial_scale,
            max(args.k_values),
        )
    else:
        parent_ids = np.empty((args.targets, 0), dtype=np.int64)  # shape: [R,0]
        parent_permutations = np.empty(
            (args.targets, 0, args.particles_per_source), dtype=np.int64
        )  # shape: [R,0,P]
        parent_distances = np.empty((args.targets, 0), dtype=np.float64)  # shape: [R,0]

    initial_plane = memory_sources.reshape(-1, args.dimension)  # [C,P,D] -> [N=C*P,D]
    print(f"seed {seed}: forward EFS on {initial_plane.shape[0]} particles")
    started = time.perf_counter()
    history, forward_diagnostics = forward_history(
        initial_plane,
        gamma=float(efs["gamma"]),
        epsilon=float(efs["epsilon"]),
        exponent_s=float(efs["exponent_s"]),
        steps=int(efs["forward_steps"]),
        log_every=args.log_every,
    )  # shape: [J+1,N,D]
    forward_seconds = time.perf_counter() - started
    passed, failure_reasons, smoke = smoke_gate(initial_plane, history[-1], forward_diagnostics)

    target_trajectory = np.empty(
        (0, args.targets, args.particles_per_source, args.dimension), dtype=np.float64
    )
    if passed:
        target_trajectory = passive_forward(
            target_sources,
            history,
            gamma=float(efs["gamma"]),
            epsilon=float(efs["epsilon"]),
            exponent_s=float(efs["exponent_s"]),
        )  # shape: [J+1,R,P,D]

    # One history file is shared by every experiment for this seed. The NPZ
    # stores all identities and correspondence metadata needed to audit it.
    np.save(directory / f"history_seed{seed}.npy", history)
    np.savez_compressed(
        directory / f"field_seed{seed}.npz",
        memory_sources=memory_sources,
        target_sources=target_sources,
        memory_source_ids=split["memory_source_ids"],
        target_source_ids=split["target_source_ids"],
        target_trajectory=target_trajectory,
        parent_ids=parent_ids,
        parent_permutations=parent_permutations,
        parent_distances=parent_distances,
        initial_scale=initial_scale,
        forward_log_step=np.asarray(forward_diagnostics["log_step"]),
        forward_mean_pair_distance=np.asarray(forward_diagnostics["mean_pair_distance"]),
        forward_rms_radius=np.asarray(forward_diagnostics["rms_radius"]),
    )

    terminal_sources = history[-1].reshape(memory_sources.shape)  # [N,D] -> [C,P,D]
    terminal_scale = np.maximum(np.std(history[-1], axis=0), 1.0e-12)  # shape: [D]
    return {
        "seed": seed,
        "rng": np.random.default_rng(seed + 100_000),
        "history": history,
        "memory_sources": memory_sources,
        "target_sources": target_sources,
        "memory_source_ids": split["memory_source_ids"],
        "target_source_ids": split["target_source_ids"],
        "target_terminal": target_trajectory[-1] if passed else np.empty_like(target_sources),
        "terminal_sources": terminal_sources,
        "initial_scale": initial_scale,
        "terminal_scale": terminal_scale,
        "parent_ids": parent_ids,
        "parent_permutations": parent_permutations,
        "parent_distances": parent_distances,
        "passed": passed,
        "failure_reasons": failure_reasons,
        "smoke": smoke,
        "forward_seconds": forward_seconds,
    }


def _run_parent_sweep(
    context: dict[str, object],
    args: argparse.Namespace,
    efs: dict[str, float | int | str],
    directory: Path,
) -> list[dict[str, object]]:
    """Fit and replay one complete-source lambda for every requested K."""
    seed = int(context["seed"])
    memory_sources = np.asarray(context["memory_sources"])  # shape: [C,P,D]
    terminal_sources = np.asarray(context["terminal_sources"])  # shape: [C,P,D]
    targets = np.asarray(context["target_sources"])  # shape: [R,P,D]
    terminal_targets = np.asarray(context["target_terminal"])  # shape: [R,P,D]
    parent_ids = np.asarray(context["parent_ids"])  # shape: [R,Kmax]
    parent_permutations = np.asarray(context["parent_permutations"])  # [R,Kmax,P]
    initial_scale = np.asarray(context["initial_scale"])  # shape: [D]
    terminal_scale = np.asarray(context["terminal_scale"])  # shape: [D]
    target_source_ids = np.asarray(context["target_source_ids"])  # shape: [R]

    terminal_candidates: list[np.ndarray] = []
    direct_candidates: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    maximum_k = max(args.k_values)
    padded_lambdas: list[np.ndarray] = []

    for target_index, target_terminal in enumerate(terminal_targets):
        for parent_count in args.k_values:
            local_ids = parent_ids[target_index, :parent_count]  # shape: [K]
            permutations = parent_permutations[target_index, :parent_count]  # shape: [K,P]
            matched_terminal = np.empty(
                (parent_count, args.particles_per_source, args.dimension), dtype=np.float64
            )  # shape: [K,P,D]
            matched_initial = np.empty_like(matched_terminal)
            for parent_index, local_id in enumerate(local_ids):
                permutation = permutations[parent_index]  # shape: [P]
                matched_terminal[parent_index] = terminal_sources[local_id][permutation]
                matched_initial[parent_index] = memory_sources[local_id][permutation]

            # Fit one lambda over the complete standardized source:
            #
            #   min ||T - sum_k lambda_k S_k||^2,
            #   lambda_k >= 0, sum_k lambda_k = 1.
            #
            # Flattening [P,D] to [P*D] gives the optimizer one coefficient
            # vector for the entire source, never one vector per particle.
            normalized_parents = matched_terminal / terminal_scale[None, None, :]  # [K,P,D]
            normalized_target = target_terminal / terminal_scale[None, :]  # [P,D]
            fit_started = time.perf_counter()
            weights, fit_error, fit_iterations = fit_simplex_weights(
                normalized_parents.reshape(parent_count, -1),  # [K,P,D] -> [K,P*D]
                normalized_target.reshape(-1),  # [P,D] -> [P*D]
                max_iterations=args.lambda_iterations,
            )
            fit_seconds = time.perf_counter() - fit_started
            terminal_candidate = np.einsum("k,kpd->pd", weights, matched_terminal)  # sum K -> [P,D]
            direct_candidate = np.einsum("k,kpd->pd", weights, matched_initial)  # sum K -> [P,D]
            padded = np.zeros(maximum_k, dtype=np.float64)  # shape: [Kmax]
            padded[:parent_count] = weights

            terminal_candidates.append(terminal_candidate)
            direct_candidates.append(direct_candidate)
            padded_lambdas.append(padded)
            metadata.append({
                "target_index": target_index,
                "target_source_id": int(target_source_ids[target_index]),
                "k": parent_count,
                "local_parent_ids": local_ids.copy(),
                "weights": weights,
                "fit_error": fit_error,
                "fit_iterations": fit_iterations,
                "fit_runtime": fit_seconds,
            })

    terminal_array = np.stack(terminal_candidates)  # shape: [G=R*len(K),P,D]
    direct_array = np.stack(direct_candidates)  # shape: [G,P,D]
    generated, residual, replay_seconds = _replay_endpoints(
        terminal_array, np.asarray(context["history"]), efs, args.workers, "parent-sweep"
    )
    runtime_per_candidate = replay_seconds / generated.shape[0]

    rows: list[dict[str, object]] = []
    for candidate_index, item in enumerate(metadata):
        target_index = int(item["target_index"])
        terminal_fit = _standardized_rmse(
            terminal_array[candidate_index], terminal_targets[target_index], terminal_scale
        )
        replay_error = _standardized_rmse(generated[candidate_index], targets[target_index], initial_scale)
        direct_error = _standardized_rmse(direct_array[candidate_index], targets[target_index], initial_scale)
        weights = np.asarray(item["weights"])
        local_parent_ids = np.asarray(item["local_parent_ids"], dtype=np.int64)
        global_parent_ids = np.asarray(context["memory_source_ids"])[local_parent_ids]
        rows.append({
            "seed": seed,
            "target_index": target_index,
            "target_source_id": item["target_source_id"],
            "k": item["k"],
            "parent_source_ids": ";".join(str(int(value)) for value in global_parent_ids),
            "lambda": ";".join(f"{value:.12g}" for value in weights),
            "terminal_fit_rmse": terminal_fit,
            "terminal_fit_raw_rmse": _raw_rmse(terminal_array[candidate_index], terminal_targets[target_index]),
            "replay_target_rmse": replay_error,
            "replay_target_raw_rmse": _raw_rmse(generated[candidate_index], targets[target_index]),
            "direct_same_lambda_rmse": direct_error,
            "direct_same_lambda_raw_rmse": _raw_rmse(direct_array[candidate_index], targets[target_index]),
            "replay_over_terminal": replay_error / max(terminal_fit, 1.0e-15),
            "replay_over_direct": replay_error / max(direct_error, 1.0e-15),
            "lambda_max": float(np.max(weights)),
            "lambda_entropy": _lambda_entropy(weights),
            "effective_parent_count": float(1.0 / np.sum(weights * weights)),
            "fit_iterations": item["fit_iterations"],
            "fit_runtime": item["fit_runtime"],
            "replay_runtime": runtime_per_candidate,
            "max_replay_residual": float(residual[candidate_index]),
            "finite": int(np.all(np.isfinite(generated[candidate_index]))),
        })

    np.savez_compressed(
        directory / f"parent_sweep_seed{seed}.npz",
        target_source_ids=target_source_ids,
        parent_ids=parent_ids,
        parent_permutations=parent_permutations,
        k=np.asarray([int(item["k"]) for item in metadata]),
        lambdas=np.stack(padded_lambdas),
        terminal_candidates=terminal_array,
        direct_candidates=direct_array,
        generated=generated,
        max_replay_residual=residual,
    )
    return rows


def _sensitivity_directions(
    vertex_id: int,
    initial_plane: np.ndarray,
    terminal_plane: np.ndarray,
    center: np.ndarray,
    initial_scale: np.ndarray,
    rng: np.random.Generator,
    boundary_neighbors: int,
) -> tuple[float, list[tuple[str, np.ndarray, int]]]:
    """Build the five declared perturbation directions for one vertex."""
    terminal_vertex = terminal_plane[vertex_id]  # shape: [D]
    difference = terminal_plane - terminal_vertex[None, :]  # [N,D] - [1,D] -> [N,D]
    squared_distance = np.sum(difference * difference, axis=1)  # shape: [N]
    squared_distance[vertex_id] = np.inf
    order = np.argsort(squared_distance)  # shape: [N]
    nearest_id = int(order[0])
    nearest_distance = math.sqrt(float(squared_distance[nearest_id]))

    branch_pool = order[: min(boundary_neighbors, order.size)]  # nearby in du, shape: [B]
    initial_difference = (
        initial_plane[branch_pool] - initial_plane[vertex_id][None, :]
    ) / initial_scale[None, :]  # [B,D]
    branch_id = int(branch_pool[np.argmax(np.linalg.norm(initial_difference, axis=1))])

    random_direction = _unit(rng.normal(size=terminal_plane.shape[1]))
    nearest_direction = _unit(terminal_plane[nearest_id] - terminal_vertex, random_direction)
    branch_direction = _unit(terminal_plane[branch_id] - terminal_vertex, random_direction)

    local_ids = order[: min(16, order.size)]
    local_offsets = terminal_plane[local_ids] - terminal_vertex[None, :]  # shape: [B,D]
    _, _, right_vectors = np.linalg.svd(local_offsets, full_matrices=False)
    local_span = right_vectors[0]  # first local offset-span direction, shape: [D]
    if float(local_span @ nearest_direction) < 0.0:
        local_span = -local_span

    radial = _unit(terminal_vertex - center, random_direction)
    tangent = local_span - float(local_span @ radial) * radial  # remove radial component, shape: [D]
    if np.linalg.norm(tangent) <= 1.0e-12:
        tangent = random_direction - float(random_direction @ radial) * radial
    tangent = _unit(tangent, nearest_direction)
    return nearest_distance, [
        ("random", random_direction, -1),
        ("nearest", nearest_direction, nearest_id),
        ("du_close_d1_far", branch_direction, branch_id),
        ("local_span", local_span, -1),
        ("radial", radial, -1),
        ("tangential", tangent, -1),
    ]


def _nearest_particle_ids(outputs: np.ndarray, initial_plane: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Return nearest initial particle IDs for query points ``[Q,D]``."""
    nearest = np.empty(outputs.shape[0], dtype=np.int64)  # shape: [Q]
    for start in range(0, outputs.shape[0], 256):
        stop = min(start + 256, outputs.shape[0])
        difference = (
            outputs[start:stop, None, :] - initial_plane[None, :, :]
        ) / scale[None, None, :]  # [B,1,D] - [1,N,D] -> [B,N,D]
        squared = np.sum(difference * difference, axis=2)  # shape: [B,N]
        nearest[start:stop] = np.argmin(squared, axis=1)
    return nearest


def _run_du_sensitivity(
    context: dict[str, object],
    args: argparse.Namespace,
    efs: dict[str, float | int | str],
    directory: Path,
) -> list[dict[str, object]]:
    """Perturb exact terminal vertices and measure local inverse response."""
    seed = int(context["seed"])
    history = np.asarray(context["history"])
    initial_plane = history[0]  # shape: [N,D]
    terminal_plane = history[-1]  # shape: [N,D]
    initial_scale = np.asarray(context["initial_scale"])  # shape: [D]
    rng = np.asarray([seed], dtype=np.int64)  # saved below as explicit provenance
    direction_rng = np.random.default_rng(seed + 200_000)
    vertex_count = min(args.sensitivity_vertices, initial_plane.shape[0])
    vertex_ids = direction_rng.choice(initial_plane.shape[0], size=vertex_count, replace=False)  # [V]
    center = np.mean(terminal_plane, axis=0)  # shape: [D]

    seeds: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    direction_vectors: list[np.ndarray] = []
    for vertex_id in vertex_ids:
        nearest_distance, directions = _sensitivity_directions(
            int(vertex_id),
            initial_plane,
            terminal_plane,
            center,
            initial_scale,
            direction_rng,
            args.boundary_neighbors,
        )
        for direction_name, direction, direction_target in directions:
            for factor in args.delta_factors:
                # The displacement magnitude is a declared fraction of this
                # vertex's nearest-neighbor spacing in the terminal plane.
                displacement = factor * nearest_distance * direction  # scalar * [D] -> [D]
                seeds.append(terminal_plane[vertex_id] + displacement)  # shape: [D]
                direction_vectors.append(direction)
                metadata.append({
                    "vertex_id": int(vertex_id),
                    "direction": direction_name,
                    "direction_target": direction_target,
                    "factor": factor,
                    "nearest_distance": nearest_distance,
                    "terminal_delta": float(np.linalg.norm(displacement)),
                })

    # Replay exact vertices and all perturbations together. Exact replay is a
    # baseline for separating inverse numerical error from local sensitivity.
    exact_count = vertex_ids.size
    terminal_sources = np.concatenate(
        (terminal_plane[vertex_ids], np.stack(seeds)), axis=0
    )[:, None, :]  # [V+Q,D] -> [V+Q,P=1,D]
    arrivals, residual, replay_seconds = _replay_endpoints(
        terminal_sources, history, efs, args.workers, "du-sensitivity"
    )
    exact_arrivals = arrivals[:exact_count, 0]  # shape: [V,D]
    perturbed_arrivals = arrivals[exact_count:, 0]  # shape: [Q,D]
    terminal_seed_array = terminal_sources[exact_count:, 0]  # shape: [Q,D]
    vertex_to_exact = {int(vertex_id): exact_arrivals[index] for index, vertex_id in enumerate(vertex_ids)}
    nearest_after = _nearest_particle_ids(perturbed_arrivals, initial_plane, initial_scale)
    runtime_per_candidate = replay_seconds / terminal_sources.shape[0]

    rows: list[dict[str, object]] = []
    for index, item in enumerate(metadata):
        vertex_id = int(item["vertex_id"])
        original = initial_plane[vertex_id]  # shape: [D]
        output = perturbed_arrivals[index]  # shape: [D]
        exact_output = vertex_to_exact[vertex_id]  # shape: [D]
        terminal_delta = float(item["terminal_delta"])
        output_to_original = float(np.linalg.norm(output - original))
        output_from_exact = float(np.linalg.norm(output - exact_output))
        source_id = vertex_id // args.particles_per_source
        source_particles = np.asarray(context["memory_sources"])[source_id]  # shape: [P,D]
        distance_to_source = float(np.min(np.linalg.norm(source_particles - output[None, :], axis=1)))
        rows.append({
            "seed": seed,
            "vertex_id": vertex_id,
            "source_id": int(np.asarray(context["memory_source_ids"])[source_id]),
            "direction": item["direction"],
            "direction_target_particle_id": item["direction_target"],
            "delta_factor": item["factor"],
            "terminal_nearest_neighbor_distance": item["nearest_distance"],
            "terminal_delta": terminal_delta,
            "output_delta": output_to_original,
            "output_delta_from_exact_replay": output_from_exact,
            "amplification": output_to_original / max(terminal_delta, 1.0e-15),
            "local_amplification": output_from_exact / max(terminal_delta, 1.0e-15),
            "nearest_d1_identity_before": vertex_id,
            "nearest_d1_identity_after": int(nearest_after[index]),
            "identity_changed": int(nearest_after[index] != vertex_id),
            "output_distance_to_original_source": distance_to_source,
            "exact_vertex_error": float(np.linalg.norm(exact_output - original)),
            "max_replay_residual": float(residual[exact_count + index]),
            "replay_runtime": runtime_per_candidate,
            "finite": int(np.all(np.isfinite(output))),
        })

    np.savez_compressed(
        directory / f"du_sensitivity_seed{seed}.npz",
        random_seed=rng,
        vertex_ids=vertex_ids,
        exact_terminal=terminal_plane[vertex_ids],
        exact_arrivals=exact_arrivals,
        terminal_seeds=terminal_seed_array,
        direction_vectors=np.stack(direction_vectors),
        perturbed_arrivals=perturbed_arrivals,
        max_replay_residual=residual,
    )
    return rows


def _select_path_pairs(
    memory_sources: np.ndarray,
    terminal_sources: np.ndarray,
    initial_scale: np.ndarray,
    terminal_scale: np.ndarray,
    rng: np.random.Generator,
    pool_size: int,
    pairs_per_category: int,
) -> list[dict[str, object]]:
    """Select four source-pair categories while fixing matching in d1."""
    pool = np.sort(rng.choice(memory_sources.shape[0], size=min(pool_size, memory_sources.shape[0]), replace=False))
    records: list[dict[str, object]] = []
    for first, second in combinations(pool.tolist(), 2):
        _, permutation_d1, distance_d1 = align_source(
            memory_sources[first], memory_sources[second], initial_scale
        )
        # The d1 permutation defines scientific correspondence. Apply that
        # same identity map in du instead of rematching terminal particles.
        distance_du = _standardized_rmse(
            terminal_sources[first], terminal_sources[second][permutation_d1], terminal_scale
        )
        records.append({
            "first": first,
            "second": second,
            "distance_d1": distance_d1,
            "distance_du": distance_du,
            "permutation_d1": permutation_d1,
        })

    selected: list[dict[str, object]] = []

    def add(category: str, candidates: list[dict[str, object]]) -> None:
        for item in candidates[:pairs_per_category]:
            selected.append({
                **item,
                "category": category,
                "permutation_frame": "d1",
                "permutation": np.asarray(item["permutation_d1"]),
            })

    add("close_d1", sorted(records, key=lambda item: float(item["distance_d1"])))
    add("close_du", sorted(records, key=lambda item: float(item["distance_du"])))

    # Search only within the closest 20 percent in du, then choose the pairs
    # whose d1-matched initial sources are farthest apart. This is the
    # declared branch-boundary stress category.
    du_sorted = sorted(records, key=lambda item: float(item["distance_du"]))
    local_count = max(pairs_per_category, math.ceil(0.20 * len(du_sorted)))
    branch_candidates = sorted(
        du_sorted[:local_count], key=lambda item: float(item["distance_d1"]), reverse=True
    )
    add("close_du_far_d1", branch_candidates)

    ordinary: list[dict[str, object]] = []
    anchors = rng.choice(pool, size=min(pairs_per_category, pool.size), replace=False)
    for anchor in anchors:
        best: dict[str, object] | None = None
        for candidate in range(memory_sources.shape[0]):
            if candidate == anchor:
                continue
            _, permutation, distance_d1 = align_source(
                memory_sources[anchor], memory_sources[candidate], initial_scale
            )
            if best is None or distance_d1 < float(best["distance_d1"]):
                best = {
                    "first": int(anchor),
                    "second": candidate,
                    "distance_d1": distance_d1,
                    "distance_du": _standardized_rmse(
                        terminal_sources[anchor], terminal_sources[candidate][permutation], terminal_scale
                    ),
                    "permutation_d1": permutation,
                }
        if best is not None:
            ordinary.append(best)
    add("ordinary_d1_neighbor", ordinary)
    return selected


def _nearest_source_ids(outputs: np.ndarray, memory_sources: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Return exact permutation-invariant nearest source IDs for ``[G,P,D]``."""
    nearest = np.empty(outputs.shape[0], dtype=np.int64)  # shape: [G]
    for output_index, output in enumerate(outputs):
        distances = np.asarray([
            source_distance(output, source, scale) for source in memory_sources
        ])  # shape: [C]
        nearest[output_index] = int(np.argmin(distances))
        if (output_index + 1) % 100 == 0:
            print(f"  nearest-source evaluation {output_index + 1}/{outputs.shape[0]}")
    return nearest


def _run_lambda_paths(
    context: dict[str, object],
    args: argparse.Namespace,
    efs: dict[str, float | int | str],
    directory: Path,
) -> list[dict[str, object]]:
    """Replay smooth two-parent lambda paths and measure output jumps."""
    seed = int(context["seed"])
    memory_sources = np.asarray(context["memory_sources"])  # shape: [C,P,D]
    terminal_sources = np.asarray(context["terminal_sources"])  # shape: [C,P,D]
    initial_scale = np.asarray(context["initial_scale"])  # shape: [D]
    terminal_scale = np.asarray(context["terminal_scale"])  # shape: [D]
    rng = np.random.default_rng(seed + 300_000)
    pairs = _select_path_pairs(
        memory_sources,
        terminal_sources,
        initial_scale,
        terminal_scale,
        rng,
        args.path_pool,
        args.path_pairs,
    )
    t_values = np.arange(0.0, 1.0 + 0.5 * args.path_step, args.path_step, dtype=np.float64)
    if t_values[-1] < 1.0:
        t_values = np.append(t_values, 1.0)
    t_values[-1] = 1.0

    terminal_paths: list[np.ndarray] = []
    direct_paths: list[np.ndarray] = []
    for pair in pairs:
        first = int(pair["first"])
        second = int(pair["second"])
        permutation = np.asarray(pair["permutation"], dtype=np.int64)  # shape: [P]
        first_terminal = terminal_sources[first]  # shape: [P,D]
        second_terminal = terminal_sources[second][permutation]  # shape: [P,D]
        first_initial = memory_sources[first]  # shape: [P,D]
        second_initial = memory_sources[second][permutation]  # shape: [P,D]

        # lambda(t)=[1-t,t] is broadcast over every particle and coordinate.
        terminal_path = (
            (1.0 - t_values)[:, None, None] * first_terminal[None, :, :]
            + t_values[:, None, None] * second_terminal[None, :, :]
        )  # [T,1,1]*[1,P,D] + [T,1,1]*[1,P,D] -> [T,P,D]
        direct_path = (
            (1.0 - t_values)[:, None, None] * first_initial[None, :, :]
            + t_values[:, None, None] * second_initial[None, :, :]
        )  # shape: [T,P,D]
        terminal_paths.append(terminal_path)
        direct_paths.append(direct_path)

    terminal_array = np.concatenate(terminal_paths, axis=0)  # [B*T,P,D]
    direct_array = np.concatenate(direct_paths, axis=0)  # [B*T,P,D]
    generated, residual, replay_seconds = _replay_endpoints(
        terminal_array, np.asarray(context["history"]), efs, args.workers, "lambda-path"
    )
    nearest_local_ids = _nearest_source_ids(generated, memory_sources, initial_scale)
    runtime_per_candidate = replay_seconds / generated.shape[0]

    rows: list[dict[str, object]] = []
    path_length = t_values.size
    permutations = []
    for path_index, pair in enumerate(pairs):
        start = path_index * path_length
        stop = start + path_length
        output_path = generated[start:stop]  # shape: [T,P,D]
        terminal_path = terminal_array[start:stop]  # shape: [T,P,D]
        permutations.append(np.asarray(pair["permutation"]))
        previous_identity = -1
        previous_signature = _pair_signature(output_path[0])

        for step_index, t_value in enumerate(t_values):
            if step_index == 0:
                terminal_step = 0.0
                output_step = 0.0
                amplification = 0.0
                relation_change = 0.0
                particle_jumps = np.zeros(args.particles_per_source, dtype=np.float64)
            else:
                terminal_difference = terminal_path[step_index] - terminal_path[step_index - 1]  # [P,D]
                output_difference = output_path[step_index] - output_path[step_index - 1]  # [P,D]
                terminal_step = float(np.linalg.norm(terminal_difference))
                output_step = float(np.linalg.norm(output_difference))
                amplification = output_step / max(terminal_step, 1.0e-15)
                signature = _pair_signature(output_path[step_index])
                relation_change = float(np.sqrt(np.mean((signature - previous_signature) ** 2)))
                previous_signature = signature
                particle_jumps = np.linalg.norm(output_difference, axis=1)  # [P,D] -> [P]

            nearest_local = int(nearest_local_ids[start + step_index])
            nearest_global = int(np.asarray(context["memory_source_ids"])[nearest_local])
            rows.append({
                "seed": seed,
                "path_index": path_index,
                "category": pair["category"],
                "permutation_frame": pair["permutation_frame"],
                "first_source_id": int(np.asarray(context["memory_source_ids"])[int(pair["first"])]),
                "second_source_id": int(np.asarray(context["memory_source_ids"])[int(pair["second"])]),
                "particle_permutation": ";".join(str(int(value)) for value in np.asarray(pair["permutation"])),
                "distance_d1": pair["distance_d1"],
                "distance_du": pair["distance_du"],
                "distance_d1_fixed_matching": pair["distance_d1"],
                "t": t_value,
                "lambda": f"{1.0 - t_value:.12g};{t_value:.12g}",
                "terminal_step_norm": terminal_step,
                "output_step_norm": output_step,
                "step_amplification": amplification,
                "nearest_source_identity": nearest_global,
                "nearest_source_changed": int(previous_identity >= 0 and nearest_global != previous_identity),
                "pairwise_relation_change": relation_change,
                "maximum_particle_jump": float(np.max(particle_jumps)),
                "particle_jumps": ";".join(f"{value:.12g}" for value in particle_jumps),
                "terminal_displacement_from_start": float(np.linalg.norm(terminal_path[step_index] - terminal_path[0])),
                "output_displacement_from_start": float(np.linalg.norm(output_path[step_index] - output_path[0])),
                "direct_displacement_from_start": float(np.linalg.norm(direct_array[start + step_index] - direct_array[start])),
                "max_replay_residual": float(residual[start + step_index]),
                "replay_runtime": runtime_per_candidate,
                "finite": int(np.all(np.isfinite(output_path[step_index]))),
            })
            previous_identity = nearest_global

    np.savez_compressed(
        directory / f"lambda_path_seed{seed}.npz",
        pair_local_ids=np.asarray([[int(pair["first"]), int(pair["second"])] for pair in pairs]),
        categories=np.asarray([str(pair["category"]) for pair in pairs]),
        permutation_frames=np.asarray([str(pair["permutation_frame"]) for pair in pairs]),
        permutations=np.stack(permutations),
        t=t_values,
        terminal_paths=np.stack(terminal_paths),
        direct_paths=np.stack(direct_paths),
        generated=generated.reshape(len(pairs), path_length, args.particles_per_source, args.dimension),
        nearest_source_local_ids=nearest_local_ids.reshape(len(pairs), path_length),
        max_replay_residual=residual.reshape(len(pairs), path_length),
    )
    return rows


def _percentile(values: list[float], probability: float) -> float:
    """Return one percentile or NaN for an empty collection."""
    return float(np.quantile(values, probability)) if values else math.nan


def _build_summary(
    parent_rows: list[dict[str, object]],
    sensitivity_rows: list[dict[str, object]],
    path_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Create descriptive grouped metrics and one declared decision flag."""
    summary: dict[str, object] = {}
    if parent_rows:
        by_k: dict[str, object] = {}
        for parent_count in sorted({int(row["k"]) for row in parent_rows}):
            selected = [row for row in parent_rows if int(row["k"]) == parent_count]
            by_k[str(parent_count)] = {
                "trials": len(selected),
                "median_terminal_fit_rmse": float(np.median([float(row["terminal_fit_rmse"]) for row in selected])),
                "median_replay_target_rmse": float(np.median([float(row["replay_target_rmse"]) for row in selected])),
                "median_direct_same_lambda_rmse": float(np.median([float(row["direct_same_lambda_rmse"]) for row in selected])),
                "median_replay_over_direct": float(np.median([float(row["replay_over_direct"]) for row in selected])),
                "median_effective_parent_count": float(np.median([float(row["effective_parent_count"]) for row in selected])),
            }
        summary["parent_sweep"] = by_k

    if sensitivity_rows:
        by_factor: dict[str, object] = {}
        for factor in sorted({float(row["delta_factor"]) for row in sensitivity_rows}):
            selected = [row for row in sensitivity_rows if float(row["delta_factor"]) == factor]
            values = [float(row["local_amplification"]) for row in selected]
            by_factor[f"{factor:.12g}"] = {
                "samples": len(selected),
                "median_local_amplification": _percentile(values, 0.50),
                "p90_local_amplification": _percentile(values, 0.90),
                "p95_local_amplification": _percentile(values, 0.95),
                "maximum_local_amplification": max(values),
                "identity_change_fraction": float(np.mean([int(row["identity_changed"]) for row in selected])),
            }
        summary["du_sensitivity"] = by_factor

    if path_rows:
        by_category: dict[str, object] = {}
        for category in sorted({str(row["category"]) for row in path_rows}):
            selected = [
                row for row in path_rows
                if str(row["category"]) == category and float(row["t"]) > 0.0
            ]
            amplification = [float(row["step_amplification"]) for row in selected]
            by_category[category] = {
                "steps": len(selected),
                "median_step_amplification": _percentile(amplification, 0.50),
                "p95_step_amplification": _percentile(amplification, 0.95),
                "maximum_step_amplification": max(amplification),
                "maximum_particle_jump": max(float(row["maximum_particle_jump"]) for row in selected),
                "nearest_identity_change_fraction": float(
                    np.mean([int(row["nearest_source_changed"]) for row in selected])
                ),
            }
        summary["lambda_path"] = by_category

    thresholds = {
        "substantial_terminal_improvement": 0.25,
        "substantial_replay_improvement": 0.20,
        "good_terminal_rmse": 0.10,
        "maximum_p95_local_amplification": 10.0,
        "maximum_small_delta_identity_change_fraction": 0.05,
        "maximum_p95_path_step_amplification": 10.0,
    }
    summary["heuristic_thresholds"] = thresholds
    if not parent_rows or not sensitivity_rows or not path_rows:
        summary["decision"] = "incomplete: all three required experiments must run"
        return summary

    k_values = sorted({int(row["k"]) for row in parent_rows})
    smallest_k, largest_k = k_values[0], k_values[-1]
    small_rows = [row for row in parent_rows if int(row["k"]) == smallest_k]
    large_rows = [row for row in parent_rows if int(row["k"]) == largest_k]
    terminal_small = float(np.median([float(row["terminal_fit_rmse"]) for row in small_rows]))
    terminal_large = float(np.median([float(row["terminal_fit_rmse"]) for row in large_rows]))
    replay_small = float(np.median([float(row["replay_target_rmse"]) for row in small_rows]))
    replay_large = float(np.median([float(row["replay_target_rmse"]) for row in large_rows]))
    replay_over_direct = float(np.median([float(row["replay_over_direct"]) for row in large_rows]))
    terminal_gain = 1.0 - terminal_large / max(terminal_small, 1.0e-15)
    replay_gain = 1.0 - replay_large / max(replay_small, 1.0e-15)

    local_rows = [row for row in sensitivity_rows if float(row["delta_factor"]) <= 1.0e-2]
    sensitivity_p95 = _percentile([float(row["local_amplification"]) for row in local_rows], 0.95)
    identity_fraction = float(np.mean([int(row["identity_changed"]) for row in local_rows]))
    path_nonzero = [row for row in path_rows if float(row["t"]) > 0.0]
    path_p95 = _percentile([float(row["step_amplification"]) for row in path_nonzero], 0.95)

    summary["decision_inputs"] = {
        "smallest_k": smallest_k,
        "largest_k": largest_k,
        "terminal_improvement_fraction": terminal_gain,
        "replay_improvement_fraction": replay_gain,
        "largest_k_terminal_rmse": terminal_large,
        "largest_k_median_replay_over_direct": replay_over_direct,
        "small_delta_p95_local_amplification": sensitivity_p95,
        "small_delta_identity_change_fraction": identity_fraction,
        "path_p95_step_amplification": path_p95,
    }
    capacity_ok = (
        terminal_gain >= thresholds["substantial_terminal_improvement"]
        and terminal_large <= thresholds["good_terminal_rmse"]
    )
    replay_ok = (
        replay_gain >= thresholds["substantial_replay_improvement"]
        and replay_over_direct <= 1.0
    )
    smooth = (
        sensitivity_p95 <= thresholds["maximum_p95_local_amplification"]
        and identity_fraction <= thresholds["maximum_small_delta_identity_change_fraction"]
        and path_p95 <= thresholds["maximum_p95_path_step_amplification"]
    )
    if capacity_ok and replay_ok and smooth:
        decision = "continue: more parents improved both fit and replay, and local paths remained smooth"
    elif not capacity_ok:
        decision = "stop: increasing K did not produce a sufficiently accurate terminal representation"
    elif not replay_ok:
        decision = "stop: terminal capacity improved but the replay did not preserve the improvement"
    else:
        decision = "stop: terminal or lambda perturbations produced excessive inverse sensitivity"
    summary["decision"] = decision
    return summary


def _summary_text(summary: dict[str, object]) -> str:
    """Render the most important JSON summary values as plain text."""
    lines = ["H4.5 parent-capacity, sensitivity, and path diagnostics", "", f"decision: {summary.get('decision', 'incomplete')}", ""]
    if "decision_inputs" in summary:
        lines.append("decision inputs:")
        for name, value in dict(summary["decision_inputs"]).items():
            lines.append(f"  {name}: {value}")
    lines.extend(("", "This classification uses declared practical flags, not formal statistical tests."))
    return "\n".join(lines) + "\n"


def self_check() -> dict[str, float]:
    """Check local metric algebra without running an EFS experiment."""
    weights = np.asarray((0.2, 0.3, 0.5))
    assert math.isclose(float(np.sum(weights)), 1.0)
    assert math.isclose(float(1.0 / np.sum(weights * weights)), 1.0 / 0.38)
    assert _lambda_entropy(weights) > 0.0
    source = np.asarray(((0.0, 0.0), (3.0, 4.0)))
    assert np.allclose(_pair_signature(source), (5.0,))
    assert math.isclose(_standardized_rmse(source, source, np.ones(2)), 0.0)
    return {"effective_parent_count": float(1.0 / np.sum(weights * weights))}


def run(args: argparse.Namespace) -> Path:
    """Run selected experiments, save each seed immediately, then plot."""
    efs = _load_efs_parameters(args)
    if args.quick:
        args.seeds = [7]
        args.source_count = 32
        args.targets = 2
        args.particles_per_source = 3
        args.dimension = 2
        args.k_values = [4, 8]
        args.sensitivity_vertices = 2
        args.delta_factors = [1.0e-2, 1.0e-1]
        args.boundary_neighbors = 4
        args.path_pool = 8
        args.path_pairs = 1
        args.path_step = 0.5
        args.workers = 1
        args.lambda_iterations = 100
        args.log_every = 5
        efs["forward_steps"] = 10
        efs["proximal_steps"] = 3
    _validate(args, efs)
    directory = _new_run_directory(args.output_root)
    started = time.perf_counter()
    config: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "experiments": args.experiments,
        "seeds": args.seeds,
        "source_count": args.source_count,
        "target_count": args.targets,
        "particles_per_source": args.particles_per_source,
        "dimension": args.dimension,
        "data_scale": args.data_scale,
        "k_values": args.k_values,
        "delta_factors": args.delta_factors,
        "sensitivity_vertices": args.sensitivity_vertices,
        "boundary_neighbors": args.boundary_neighbors,
        "path_pool": args.path_pool,
        "path_pairs_per_category": args.path_pairs,
        "path_step": args.path_step,
        "lambda_iterations": args.lambda_iterations,
        "workers": args.workers,
        "quick": bool(args.quick),
        "efs": efs,
        "model": "one pooled target-free memory field; passive targets and candidates",
        "neighbor_frame": "d1",
        "lambda_fit_frame": "du",
        "algebra_self_check": {
            "parent_efs": efs_self_check(),
            "parent_data": data_self_check(),
            "local": self_check(),
        },
        "seed_runs": {},
    }
    _write_json(directory / "config.json", config)

    parent_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        context = _prepare_field(seed, args, efs, directory)
        config["seed_runs"][str(seed)] = {
            "forward_seconds": context["forward_seconds"],
            "smoke_passed": context["passed"],
            "smoke_failure_reasons": context["failure_reasons"],
            "smoke": context["smoke"],
        }
        _write_json(directory / "config.json", config)
        if not bool(context["passed"]):
            print(f"seed {seed} skipped: {'; '.join(context['failure_reasons'])}")
            continue

        if "parent-sweep" in args.experiments:
            rows = _run_parent_sweep(context, args, efs, directory)
            parent_rows.extend(rows)
            _append_csv(directory / "parent_sweep.csv", rows)
        if "du-sensitivity" in args.experiments:
            rows = _run_du_sensitivity(context, args, efs, directory)
            sensitivity_rows.extend(rows)
            _append_csv(directory / "du_sensitivity.csv", rows)
        if "lambda-path" in args.experiments:
            rows = _run_lambda_paths(context, args, efs, directory)
            path_rows.extend(rows)
            _append_csv(directory / "lambda_path.csv", rows)

        # The history has already been saved. Release the largest live array
        # before constructing the next seed's field.
        del context

    if sensitivity_rows:
        worst = sorted(
            sensitivity_rows, key=lambda row: float(row["local_amplification"]), reverse=True
        )[:20]
        _append_csv(directory / "du_sensitivity_worst_cases.csv", worst)

    summary = _build_summary(parent_rows, sensitivity_rows, path_rows)
    config["status"] = "complete"
    config["total_seconds"] = time.perf_counter() - started
    config["decision"] = summary.get("decision")
    _write_json(directory / "config.json", config)
    _write_json(directory / "summary.json", summary)
    (directory / "summary.txt").write_text(_summary_text(summary), encoding="utf-8")
    figures = plot_all(directory)
    print(f"saved {directory}")
    print(f"generated {len(figures)} figure(s)")
    return directory


def build_parser() -> argparse.ArgumentParser:
    """Define the Experiment 09 diagnostics CLI."""
    parser = argparse.ArgumentParser(
        description="Run H4.5 parent-capacity, sensitivity, and path diagnostics."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=EXPERIMENT_NAMES,
        default=list(EXPERIMENT_NAMES),
        help="Subset to run; default runs all three with one shared history per seed.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[4652], help="Deterministic field seeds.")
    parser.add_argument("--source-count", type=int, default=256, help="Complete memory sources per seed.")
    parser.add_argument("--targets", type=int, default=8, help="Held-out complete targets for the K sweep.")
    parser.add_argument("--particles-per-source", type=int, default=8, help="Particles in each complete source; maximum 16.")
    parser.add_argument("--dimension", type=int, default=4, help="Coordinates per particle.")
    parser.add_argument("--data-scale", type=float, default=1.0, help="Scale of the chaotic particle law.")
    parser.add_argument("--k-values", nargs="+", type=int, default=list(DEFAULT_K_VALUES), help="Parent counts for experiment 1.")
    parser.add_argument("--lambda-iterations", type=int, default=2000, help="Maximum projected-gradient lambda updates.")
    parser.add_argument("--sensitivity-vertices", type=int, default=12, help="Exact particles perturbed in experiment 2.")
    parser.add_argument("--delta-factors", nargs="+", type=float, default=list(DEFAULT_DELTA_FACTORS), help="Perturbations relative to terminal nearest-neighbor spacing.")
    parser.add_argument("--boundary-neighbors", type=int, default=32, help="Nearby du particles searched for a far-d1 branch direction.")
    parser.add_argument("--path-pool", type=int, default=32, help="Memory sources considered when choosing path pairs.")
    parser.add_argument("--path-pairs", type=int, default=1, help="Pairs selected per path category.")
    parser.add_argument("--path-step", type=float, default=0.01, help="Lambda path spacing from zero to one.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent passive replay batches; maximum 32.")
    parser.add_argument("--log-every", type=int, default=100, help="Print forward state every this many frames.")
    parser.add_argument("--output-root", type=Path, default=Path("results"), help="Parent of the new timestamped run.")
    parser.add_argument("--config", type=Path, default=PARENT_DIRECTORY / "best_config.json", help="Parent search-produced EFS config.")
    parser.add_argument("--ignore-config", action="store_true", help="Ignore the parent EFS config and use built-ins or CLI values.")
    parser.add_argument("--epsilon", type=float, default=None, help="EFS smoothing override.")
    parser.add_argument("--exponent-s", type=float, default=None, help="EFS exponent override.")
    parser.add_argument("--gamma", type=float, default=None, help="Forward Euler step override.")
    parser.add_argument("--forward-steps", type=int, default=None, help="Forward frame count override.")
    parser.add_argument("--beta", type=float, default=None, help="Backward proximal step override.")
    parser.add_argument("--proximal-steps", type=int, default=None, help="Proximal iterations per backward frame override.")
    parser.add_argument("--quick", action="store_true", help="Tiny end-to-end path check, not scientific evidence.")
    return parser


def main() -> None:
    """Parse arguments and run."""
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

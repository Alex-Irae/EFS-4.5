"""Run canonical two-pass H4.5 or the preserved single-pass toys.

Purpose:
    Select neighbors at d1, fit one shared lambda after a target-present
    forward pass, rerun without that target, replay, score, save, and plot.
Dependencies:
    NumPy and Matplotlib.
Outputs:
    One new timestamped directory under ``--output-root``.
Exact command:
    python run.py --output-root experiments\\07_two_pass_reconstruction\\results
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from data import (
    DATA_MODEL,
    METHOD_NAMES,
    UNIFORM_DATA_MODEL,
    build_reconstruction_trials,
    coordinate_scale,
    fit_shared_source_lambda,
    generate_chaotic_sources,
    generate_uniform_sources,
    hold_out_sources,
    interpolate_sources,
    select_source_neighbors,
)
from data import self_check as data_self_check
from efs import backward_replay, forward_history, passive_forward, source_mean_pairwise_distance
from efs import self_check as efs_self_check
from evaluation import (
    marginal_histogram_kl,
    reconstruction_metrics,
    result_summary,
    smoke_gate,
    vertex_replay_metrics,
    write_metrics_csv,
    write_smoke_csv,
)
from evaluation import self_check as evaluation_self_check
from plotting import plot_all

ACTIVE_CONFIG_PATH = Path(__file__).resolve().with_name("best_config.json")
BUILTIN_EFS_DEFAULTS = {
    "epsilon": 0.03,
    "exponent_s": 3.0,
    "gamma": 0.001,
    "forward_steps": 2000,
    "beta": 0.10,
    "proximal_steps": 100,
}


def _trajectory_logs(trajectory: np.ndarray, log_every: int) -> dict[str, np.ndarray]:
    """Rebuild grouped replay logs from ``trajectory [J+1,G,P,D]``."""
    step_count, group_count, _, dimension = (
        trajectory.shape[0] - 1,
        trajectory.shape[1],
        trajectory.shape[2],
        trajectory.shape[3],
    )
    if log_every == 0:
        return {
            "reverse_step": np.empty(0, dtype=np.int64),
            "mean_pair_distance": np.empty((0, group_count), dtype=np.float64),
            "total_position": np.empty((0, group_count, dimension), dtype=np.float64),
            "displacement": np.empty((0, group_count, dimension), dtype=np.float64),
        }

    reverse_step = np.arange(0, step_count + 1, log_every, dtype=np.int64)  # shape: [F]
    if reverse_step[-1] != step_count:
        reverse_step = np.append(reverse_step, step_count)
    frame_index = step_count - reverse_step  # reverse step r is stored at trajectory frame J-r
    frames = trajectory[frame_index]  # advanced indexing [F,G,P,D]
    mean_distance = np.stack([source_mean_pairwise_distance(frame) for frame in frames])  # [F,G]
    total_position = np.sum(frames, axis=2)  # [F,G,P,D] -> [F,G,D]
    displacement = np.zeros_like(total_position)
    for log_index, frame in enumerate(frame_index[1:], start=1):
        previous_total = np.sum(trajectory[frame + 1], axis=1)  # [G,P,D] -> [G,D]
        displacement[log_index] = total_position[log_index] - previous_total  # shape: [G,D]
    return {
        "reverse_step": reverse_step,
        "mean_pair_distance": mean_distance,
        "total_position": total_position,
        "displacement": displacement,
    }


def _parallel_backward_replay(
    terminal_sources: np.ndarray,
    history: np.ndarray,
    gamma: float,
    epsilon: float,
    exponent_s: float,
    beta: float,
    proximal_steps: int,
    method_names: list[str],
    log_every: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Replay independent source batches concurrently while sharing history.

    The EFS equations are unchanged. Threads divide only the passive source
    axis ``G``; every worker reads the same immutable history ``[J+1,N,D]``.
    """
    group_count = terminal_sources.shape[0]
    worker_count = min(workers, group_count)
    if worker_count == 1:
        return backward_replay(
            terminal_sources,
            history,
            gamma,
            epsilon,
            exponent_s,
            beta,
            proximal_steps,
            method_names=method_names,
            log_every=log_every,
        )

    chunks = [indices for indices in np.array_split(np.arange(group_count), worker_count) if indices.size]

    def replay_chunk(item: tuple[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        chunk_number, indices = item
        trajectory, residual, _ = backward_replay(
            terminal_sources[indices],  # advanced indexing [G_chunk,P,D]
            history,
            gamma,
            epsilon,
            exponent_s,
            beta,
            proximal_steps,
            method_names=[method_names[int(index)] for index in indices],
            log_every=log_every if chunk_number == 0 else 0,
        )
        return trajectory, residual

    # NumPy releases the GIL in the large field operations. Threads avoid one
    # full history copy per worker, which matters for long 2D trajectories.
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(replay_chunk, enumerate(chunks)))
    trajectory = np.concatenate([result[0] for result in results], axis=1)  # concatenate G_chunk -> [J+1,G,P,D]
    residual = np.concatenate([result[1] for result in results], axis=1)  # concatenate G_chunk -> [J,G,P]
    return trajectory, residual, _trajectory_logs(trajectory, log_every)


def _apply_parameter_config(args: argparse.Namespace) -> str:
    """Fill unset EFS arguments from a saved search config or built-in values.

    Explicit CLI values remain untouched. The default config path is beside
    this file, so launching ``run.py`` from another directory behaves the same.
    """
    explicit = {name: getattr(args, name, None) is not None for name in BUILTIN_EFS_DEFAULTS}
    loaded: dict[str, object] = {}
    config_path = Path(getattr(args, "config", None) or ACTIVE_CONFIG_PATH).resolve()
    if not bool(getattr(args, "ignore_config", False)) and config_path.exists():
        with config_path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"parameter config must contain one JSON object: {config_path}")
        loaded = value

    for name, fallback in BUILTIN_EFS_DEFAULTS.items():
        if getattr(args, name, None) is None:
            value = loaded.get(name, fallback)
            value = int(value) if isinstance(fallback, int) else float(value)
            setattr(args, name, value)

    args.loaded_parameter_config = loaded
    args.loaded_parameter_config_path = str(config_path) if loaded else ""
    if loaded:
        source = f"saved config {config_path}"
        expected_model = UNIFORM_DATA_MODEL if args.distribution == "uniform" else DATA_MODEL
        if loaded.get("data_model") != expected_model:
            source += "; warning: config data model differs from this run"
        elif int(loaded.get("particles_per_source", args.particles_per_source)) != args.particles_per_source:
            source += "; warning: selected with a different particles-per-source value"
        if any(explicit.values()):
            source += " with explicit CLI overrides"
        return source
    if any(explicit.values()):
        return "built-in defaults with explicit CLI overrides"
    return "built-in defaults"


def _new_run_directory(output_root: Path, seed: int, prefix: str = "one_plane_h45") -> Path:
    """Create one timestamped directory and refuse collisions."""
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = output_root / f"{prefix}_{stamp}_seed{seed}"
    directory.mkdir(exist_ok=False)
    return directory


def _write_json(path: Path, values: dict[str, object]) -> None:
    """Write readable JSON with no custom configuration class."""
    path.write_text(json.dumps(values, indent=2, sort_keys=True), encoding="utf-8")


def _validate(args: argparse.Namespace) -> None:
    """Reject shape or numerical settings that cannot define the experiment."""
    positive_names = (
        "source_count",
        "heldout_sources",
        "particles_per_source",
        "dimension",
        "parents",
        "vertex_sources",
        "forward_steps",
        "proximal_steps",
        "lambda_iterations",
        "log_every",
        "workers",
    )
    for name in positive_names:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.dimension < 2 or args.particles_per_source < 1:
        raise ValueError("dimension must be at least two and particles per source at least one")
    if args.particles_per_source > 16:
        raise ValueError("dependency-free exact matching supports at most 16 particles per source")
    if args.workers > 32:
        raise ValueError("--workers is capped at 32 to avoid accidental thread oversubscription")
    if args.source_count < max(args.parents, args.vertex_sources, 4):
        raise ValueError("the memory library has too few complete sources")
    if args.epsilon <= 0.0 or args.exponent_s <= 0.0 or args.gamma <= 0.0:
        raise ValueError("epsilon, exponent-s, and gamma must be positive")
    if args.beta <= 0.0 or args.data_scale <= 0.0:
        raise ValueError("beta and data-scale must be positive")


def _save_and_plot(
    directory: Path, arrays: dict[str, np.ndarray], config: dict[str, object], summary_lines: list[str]
) -> list[Path]:
    """Write one immutable run payload, then regenerate supported figures."""
    np.savez_compressed(directory / "samples.npz", **arrays)
    _write_json(directory / "config.json", config)
    (directory / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return plot_all(directory)


def _print_trial_creation(
    target_source_ids: np.ndarray,
    parent_source_ids: np.ndarray,
    initial_targets: np.ndarray,
    terminal_targets: np.ndarray,
    matched_initial_parents: np.ndarray,
    matched_terminal_parents: np.ndarray,
    shared_lambdas: np.ndarray,
    direct_candidates: np.ndarray,
    terminal_candidates_array: np.ndarray,
) -> None:
    """Print the selected complete sources and every constructed candidate.

    Targets have ``[R,P,D]``, matched parents have ``[R,K,P,D]``,
    lambdas have ``[R,M=1,K]``, and candidates have ``[R,M=1,P,D]``.
    """
    options = {"precision": 5, "suppress_small": False, "max_line_width": 160}
    for trial in range(target_source_ids.size):
        print("\n" + "=" * 72)
        print(
            f"held-out target {trial}: source_id={int(target_source_ids[trial])} "
            f"parent_source_ids={parent_source_ids[trial].tolist()}"
        )
        print("initial held-out target [P,D] (evaluation witness only):")
        print(np.array2string(initial_targets[trial], **options))
        print("passively forwarded terminal target [P,D] (lambda fit witness):")
        print(np.array2string(terminal_targets[trial], **options))
        print("matched terminal parent sources [K,P,D] (used to fit lambda):")
        print(np.array2string(matched_terminal_parents[trial], **options))
        print(f"terminal-fitted nonnegative lambda [M=1,K], method={METHOD_NAMES.tolist()}:")
        print(np.array2string(shared_lambdas[trial], **options))
        print("terminal candidates sent into backward replay [M,P,D]:")
        print(np.array2string(terminal_candidates_array[trial], **options))
        print("same matched parents in initial space [K,P,D]:")
        print(np.array2string(matched_initial_parents[trial], **options))
        print("initial-space interpolation baseline [M,P,D]:")
        print(np.array2string(direct_candidates[trial], **options))


def _forward_pass(
    sources: np.ndarray, args: argparse.Namespace, label: str
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | bool | str], bool, list[str], dict[str, dict[str, float | bool]], float]:
    """Run one complete pooled EFS forward pass for ``sources [C,P,D]``."""
    initial_plane = sources.reshape(-1, args.dimension).copy()  # [C,P,D] -> [N=C*P,D]
    print(f"{label}: {initial_plane.shape[0]} particles in one pooled plane")
    started = time.perf_counter()
    history, diagnostics = forward_history(
        initial_plane,
        gamma=args.gamma,
        epsilon=args.epsilon,
        exponent_s=args.exponent_s,
        steps=args.forward_steps,
        log_every=args.log_every,
    )  # shape: [J+1,N,D]
    seconds = time.perf_counter() - started
    passed, reasons, smoke = smoke_gate(initial_plane, history[-1], diagnostics)
    return initial_plane, history, diagnostics, passed, reasons, smoke, seconds


def _add_forward_arrays(
    arrays: dict[str, np.ndarray], prefix: str, history: np.ndarray, diagnostics: dict[str, np.ndarray | bool | str]
) -> None:
    """Store one forward pass under a clear pass-specific prefix."""
    arrays[f"{prefix}_history"] = history  # shape: [J+1,N,D]
    for name in (
        "log_step",
        "mean_pair_distance",
        "rms_radius",
        "max_force_norm",
        "max_update_norm",
        "center_drift",
    ):
        arrays[f"{prefix}_forward_{name}"] = np.asarray(diagnostics[name])


def run_single_pass(args: argparse.Namespace) -> Path:
    """Execute the preserved single-pass toy reconstruction dataflow."""
    parameter_source = _apply_parameter_config(args)
    if args.quick:
        args.source_count = 32
        args.heldout_sources = 4
        args.particles_per_source = 4
        args.parents = 3
        args.vertex_sources = 2
        args.forward_steps = 20
        args.proximal_steps = 5
        args.lambda_iterations = 200
        args.log_every = 10
        parameter_source += "; quick mode truncates forward and replay steps"
    _validate(args)
    print(f"EFS parameters: {parameter_source}")

    algebra = {"efs": efs_self_check(), "data": data_self_check()}
    directory = _new_run_directory(Path(args.output_root).resolve(), args.seed)
    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)

    # ``--source-count`` is the memory-library size. Extra complete sources are
    # drawn only as held-out targets, so the EFS plane contains exactly C*P rows.
    generator = generate_uniform_sources if args.distribution == "uniform" else generate_chaotic_sources
    data_model = UNIFORM_DATA_MODEL if args.distribution == "uniform" else DATA_MODEL
    all_sources = generator(
        args.source_count + args.heldout_sources,
        args.particles_per_source,
        args.dimension,
        args.data_scale,
        rng,
    )  # shape: [C+R,P,D]
    split = hold_out_sources(all_sources, args.heldout_sources, np.random.default_rng(args.seed + 1))
    memory_sources = split["memory_sources"]  # shape: [C,P,D]
    target_sources = split["target_sources"]  # shape: [R,P,D]
    initial_plane = memory_sources.reshape(-1, args.dimension).copy()  # [C,P,D] -> [N=C*P,D]
    initial_scale = coordinate_scale(memory_sources)  # shape: [D]
    memory_std = float(np.std(initial_plane))

    print(
        f"one EFS plane: {initial_plane.shape[0]} particles from "
        f"{memory_sources.shape[0]} memory sources; {target_sources.shape[0]} held-out targets"
    )
    forward_started = time.perf_counter()
    history, forward = forward_history(
        initial_plane,
        gamma=args.gamma,
        epsilon=args.epsilon,
        exponent_s=args.exponent_s,
        steps=args.forward_steps,
        log_every=args.log_every,
    )  # history shape: [J+1,N,D]
    forward_seconds = time.perf_counter() - forward_started
    smoke_passed, smoke_reasons, smoke = smoke_gate(initial_plane, history[-1], forward)

    arrays: dict[str, np.ndarray] = {
        "method_names": METHOD_NAMES,
        "all_sources": all_sources,
        "memory_sources": memory_sources,
        "target_sources": target_sources,
        "memory_source_ids": split["memory_source_ids"],
        "target_source_ids": split["target_source_ids"],
        "initial_scale": initial_scale,
        "initial_plane": initial_plane,
        "history": history,
        "forward_log_step": np.asarray(forward["log_step"]),
        "forward_mean_pair_distance": np.asarray(forward["mean_pair_distance"]),
        "forward_rms_radius": np.asarray(forward["rms_radius"]),
        "forward_max_force_norm": np.asarray(forward["max_force_norm"]),
        "forward_max_update_norm": np.asarray(forward["max_update_norm"]),
        "forward_center_drift": np.asarray(forward["center_drift"]),
    }
    config: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "source_count": args.source_count,
        "total_created_source_count": args.source_count + args.heldout_sources,
        "memory_source_count": int(memory_sources.shape[0]),
        "heldout_source_count": args.heldout_sources,
        "particles_per_source": args.particles_per_source,
        "dimension": args.dimension,
        "total_memory_particles": int(initial_plane.shape[0]),
        "parent_count": args.parents,
        "vertex_source_count": args.vertex_sources,
        "data_scale": args.data_scale,
        "epsilon": args.epsilon,
        "exponent_s": args.exponent_s,
        "gamma": args.gamma,
        "forward_steps_requested": args.forward_steps,
        "forward_steps_completed": int(history.shape[0] - 1),
        "beta": args.beta,
        "proximal_steps": args.proximal_steps,
        "lambda_iterations": args.lambda_iterations,
        "log_every": args.log_every,
        "workers": args.workers,
        "quick": bool(args.quick),
        "parameter_source": parameter_source,
        "loaded_parameter_config_path": args.loaded_parameter_config_path,
        "loaded_parameter_config": args.loaded_parameter_config,
        "plane_model": "one pooled [N,D] field; source ownership is metadata only",
        "data_model": data_model,
        "particle_distribution": args.distribution,
        "target_model": "randomly grouped complete sources held out before forward EFS",
        "lambda_model": "one nonnegative whole-source fit on terminal EFS coordinates",
        "lambda_fit_frame": "terminal post-forward frame J",
        "algebra_self_check": algebra,
        "forward_seconds": forward_seconds,
        "smoke_passed": smoke_passed,
        "smoke_failure_reasons": smoke_reasons,
        "smoke": smoke,
    }

    if not smoke_passed:
        config["status"] = "invalid forward field"
        write_smoke_csv(directory / "smoke.csv", smoke, smoke_passed)
        summary = [
            "Single-plane H4.5 held-out reconstruction",
            "",
            "status: invalid forward field",
            f"reasons: {'; '.join(smoke_reasons)}",
            f"memory particles: {initial_plane.shape[0]}",
            f"completed forward frames: {history.shape[0] - 1}",
            f"forward runtime: {forward_seconds:.2f} s",
            "target reconstruction was not run.",
        ]
        _save_and_plot(directory, arrays, config, summary)
        return directory

    terminal_sources = history[-1].reshape(memory_sources.shape)  # [N,D] -> [C,P,D]
    terminal_scale = np.maximum(np.std(history[-1], axis=0), 1.0e-12)  # shape: [D]
    vertex_rng = np.random.default_rng(args.seed + 2)
    vertex_local_ids = vertex_rng.choice(memory_sources.shape[0], size=args.vertex_sources, replace=False)  # shape: [V]
    vertex_started = time.perf_counter()
    vertex_trajectory, vertex_residual, vertex_logs = _parallel_backward_replay(
        terminal_sources[vertex_local_ids],
        history,
        gamma=args.gamma,
        epsilon=args.epsilon,
        exponent_s=args.exponent_s,
        beta=args.beta,
        proximal_steps=args.proximal_steps,
        method_names=["vertex"] * args.vertex_sources,
        log_every=args.log_every,
        workers=args.workers,
    )
    vertex_seconds = time.perf_counter() - vertex_started
    vertex = vertex_replay_metrics(vertex_trajectory[0], memory_sources[vertex_local_ids], memory_std, vertex_residual)
    vertex_global_ids = split["memory_source_ids"][vertex_local_ids]  # local [V] -> original source IDs [V]
    arrays.update({
        "terminal_sources": terminal_sources,
        "terminal_scale": terminal_scale,
        "vertex_source_ids": vertex_global_ids,
        "vertex_terminal": terminal_sources[vertex_local_ids],
        "vertex_trajectory": vertex_trajectory,
        "vertex_residual": vertex_residual,
        "vertex_particle_rmse": np.asarray(vertex["particle_rmse"]),
        "vertex_source_rmse": np.asarray(vertex["source_rmse"]),
        "vertex_particle_relative_rmse": np.asarray(vertex["particle_relative_rmse"]),
        "vertex_source_relative_rmse": np.asarray(vertex["source_relative_rmse"]),
        "vertex_particle_max_residual": np.asarray(vertex["particle_max_residual"]),
        "vertex_backward_log_step": vertex_logs["reverse_step"],
        "vertex_backward_log_mean_distance": vertex_logs["mean_pair_distance"],
        "vertex_backward_log_total_position": vertex_logs["total_position"],
        "vertex_backward_log_displacement": vertex_logs["displacement"],
    })
    config.update({
        "vertex_seconds": vertex_seconds,
        "vertex_passed": bool(vertex["passed"]),
        "vertex_median_rmse": float(np.median(vertex["particle_rmse"])),
        "vertex_p95_rmse": float(np.quantile(vertex["particle_rmse"], 0.95)),
        "vertex_maximum_rmse": float(np.max(vertex["particle_rmse"])),
        "vertex_median_relative_rmse": float(vertex["median"]),
        "vertex_p90_relative_rmse": float(vertex["p90"]),
        "vertex_p95_relative_rmse": float(vertex["p95"]),
        "vertex_maximum_relative_rmse": float(vertex["maximum"]),
    })
    write_smoke_csv(directory / "smoke.csv", smoke, smoke_passed, vertex_global_ids, vertex)

    if not bool(vertex["passed"]):
        config["status"] = "invalid vertex replay"
        summary = [
            "Single-plane H4.5 held-out reconstruction",
            "",
            "status: invalid vertex replay",
            f"vertex median relative RMSE: {float(vertex['median']):.8e}",
            f"vertex p95 relative RMSE: {float(vertex['p95']):.8e}",
            f"vertex maximum relative RMSE: {float(vertex['maximum']):.8e}",
            "hard gate: maximum must be at most 1.0e-1",
            "target reconstruction was not run.",
        ]
        _save_and_plot(directory, arrays, config, summary)
        return directory

    # The target is carried to the terminal field first. Neighbor retrieval,
    # particle matching, and lambda fitting must all happen there. Fitting from
    # the initial distribution would answer a different and biased question.
    oracle_started = time.perf_counter()
    oracle_target_trajectory = passive_forward(
        target_sources, history, gamma=args.gamma, epsilon=args.epsilon, exponent_s=args.exponent_s
    )  # shape: [J+1,R,P,D]
    oracle_seconds = time.perf_counter() - oracle_started

    preparation_started = time.perf_counter()
    terminal_targets = oracle_target_trajectory[-1]  # shape: [R,P,D]
    trials = build_reconstruction_trials(
        terminal_sources,
        terminal_targets,
        parent_count=args.parents,
        scale=terminal_scale,
        lambda_iterations=args.lambda_iterations,
    )
    parent_global_ids = split["memory_source_ids"][trials["parent_ids"]]  # [R,K] local -> original IDs
    matched_terminal = trials["matched_parents"]  # shape: [R,K,P,D]
    terminal_seed = trials["fitted_candidates"]  # shape: [R,M=1,P,D]

    # Reuse terminal particle correspondence and lambda on the same source rows
    # at frame zero. This is only the direct interpolation baseline. It does not
    # influence terminal retrieval, terminal fitting, or EFS replay.
    matched_initial, direct_candidates = interpolate_sources(
        memory_sources, trials["parent_ids"], trials["parent_permutations"], trials["shared_lambdas"]
    )  # shapes: [R,K,P,D], [R,M=1,P,D]
    _print_trial_creation(
        split["target_source_ids"],
        parent_global_ids,
        target_sources,
        terminal_targets,
        matched_initial,
        matched_terminal,
        trials["shared_lambdas"],
        direct_candidates,
        terminal_seed,
    )
    preparation_seconds = time.perf_counter() - preparation_started

    target_count, method_count, particle_count, dimension = terminal_seed.shape
    flat_terminal = terminal_seed.reshape(target_count * method_count, particle_count, dimension)  # [R,M,P,D] -> [R*M,P,D]
    flat_method_names = np.tile(METHOD_NAMES, target_count).tolist()  # method order follows reshape R then M
    target_started = time.perf_counter()
    flat_trajectory, flat_residual, target_logs = _parallel_backward_replay(
        flat_terminal,
        history,
        gamma=args.gamma,
        epsilon=args.epsilon,
        exponent_s=args.exponent_s,
        beta=args.beta,
        proximal_steps=args.proximal_steps,
        method_names=flat_method_names,
        log_every=args.log_every,
        workers=args.workers,
    )
    target_seconds = time.perf_counter() - target_started
    target_trajectory = flat_trajectory.reshape(
        history.shape[0], target_count, method_count, particle_count, dimension
    )  # [J+1,R*M,P,D] -> [J+1,R,M,P,D]
    target_residual = flat_residual.reshape(
        history.shape[0] - 1, target_count, method_count, particle_count
    )  # [J,R*M,P] -> [J,R,M,P]
    generated = target_trajectory[0]  # shape: [R,M,P,D]

    metrics = reconstruction_metrics(
        generated,
        direct_candidates,
        target_sources,
        matched_initial,
        terminal_seed,
        oracle_target_trajectory[-1],
        target_residual,
        memory_sources,
        initial_scale,
        terminal_scale,
    )
    method_summary = result_summary(metrics, METHOD_NAMES)
    write_metrics_csv(
        directory / "metrics.csv",
        split["target_source_ids"],
        parent_global_ids,
        trials["shared_lambdas"],
        trials["lambda_fit_rmse"],
        metrics,
    )

    arrays.update({
        "parent_source_ids": parent_global_ids,
        "parent_memory_ids": trials["parent_ids"],
        "parent_permutations": trials["parent_permutations"],
        "matched_parents": matched_initial,
        "matched_terminal_parents": matched_terminal,
        "aligned_targets": target_sources,
        "terminal_targets": terminal_targets,
        "lambda_fit_rmse": trials["lambda_fit_rmse"],
        "shared_lambdas": trials["shared_lambdas"],
        "direct_candidates": direct_candidates,
        "terminal_candidates": terminal_seed,
        "oracle_target_trajectory": oracle_target_trajectory,
        "target_trajectory": target_trajectory,
        "target_residual": target_residual,
        "generated": generated,
        "target_backward_log_step": target_logs["reverse_step"],
        "target_backward_log_mean_distance": target_logs["mean_pair_distance"].reshape(
            target_logs["mean_pair_distance"].shape[0], target_count, method_count
        ),  # [F,R*M] -> [F,R,M]
        "target_backward_log_total_position": target_logs["total_position"].reshape(
            target_logs["total_position"].shape[0], target_count, method_count, dimension
        ),  # [F,R*M,D] -> [F,R,M,D]
        "target_backward_log_displacement": target_logs["displacement"].reshape(
            target_logs["displacement"].shape[0], target_count, method_count, dimension
        ),  # [F,R*M,D] -> [F,R,M,D]
        **{f"metric_{name}": value for name, value in metrics.items()},
    })

    total_seconds = time.perf_counter() - started
    config.update({
        "status": "complete",
        "preparation_seconds": preparation_seconds,
        "passive_target_forward_seconds": oracle_seconds,
        "target_replay_seconds": target_seconds,
        "total_seconds": total_seconds,
        "method_summary": method_summary,
    })
    summary = [
        "Single-plane H4.5 held-out reconstruction",
        "",
        "status: complete",
        f"parameter source: {parameter_source}",
        f"memory field: {initial_plane.shape[0]} particles in one plane",
        f"memory sources: {memory_sources.shape[0]}",
        f"held-out complete targets: {target_sources.shape[0]}",
        f"forward radial CDF gap: {float(smoke['terminal']['radial_cdf_gap']):.8f}",
        f"forward radius ratio: {float(smoke['terminal']['radius_ratio']):.8f}",
        f"vertex median relative RMSE: {float(vertex['median']):.8e}",
        f"vertex p95 relative RMSE: {float(vertex['p95']):.8e}",
        f"vertex maximum relative RMSE: {float(vertex['maximum']):.8e}",
        f"median terminal shared-lambda fit RMSE: {float(np.median(trials['lambda_fit_rmse'])):.8e}",
    ]
    for name in METHOD_NAMES:
        values = method_summary[str(name)]
        summary.extend((
            "",
            f"method: {name}",
            f"result: {values['status']}",
            f"EFS beats direct interpolation: {values['beats_direct_count']}/{values['targets']}",
            f"EFS beats closest parent: {values['beats_parent_count']}/{values['targets']}",
            f"median direct relative distance: {values['median_direct_relative_distance']:.8f}",
            f"median EFS relative distance: {values['median_generated_relative_distance']:.8f}",
            f"median direct target RMSE: {values['median_direct_set_rmse']:.8f}",
            f"median EFS target RMSE: {values['median_generated_set_rmse']:.8f}",
            f"median EFS/direct ratio: {values['median_efs_direct_ratio']:.8f}",
            f"p90 EFS/direct ratio: {values['p90_efs_direct_ratio']:.8f}",
            f"maximum EFS/direct ratio: {values['maximum_efs_direct_ratio']:.8f}",
        ))
        if args.particles_per_source == 1:
            summary.extend((
                f"median direct point distance: {values['median_direct_point_distance']:.8f}",
                f"median EFS point distance: {values['median_generated_point_distance']:.8f}",
            ))
    summary.extend((
        "",
        "Interpretation:",
        "A ratio below 1 means EFS moved a shared interpolation closer to a real source that was absent from its field.",
        "Vertex replay only checks inversion at exact memory endpoints.",
        f"total runtime: {total_seconds:.2f} s",
    ))
    figures = _save_and_plot(directory, arrays, config, summary)
    print(f"saved {directory}")
    print(f"generated {len(figures)} figure(s)")
    return directory


def run_two_pass(args: argparse.Namespace) -> Path:
    """Run the target-present reference pass and target-removed generation pass."""
    parameter_source = _apply_parameter_config(args)
    if args.quick:
        args.source_count = 32
        args.heldout_sources = 1
        args.particles_per_source = 4
        args.parents = 3
        args.vertex_sources = 2
        args.forward_steps = 20
        args.proximal_steps = 5
        args.lambda_iterations = 200
        args.log_every = 10
        parameter_source += "; quick mode truncates both forward passes and replay"
    _validate(args)
    if args.heldout_sources != 1:
        raise ValueError("the two-pass protocol requires --heldout-sources 1")
    print(f"EFS parameters: {parameter_source}")

    algebra = {
        "efs": efs_self_check(),
        "data": data_self_check(),
        "evaluation": evaluation_self_check(),
    }
    directory = _new_run_directory(Path(args.output_root).resolve(), args.seed, prefix="two_pass_h45")
    started = time.perf_counter()
    generator = generate_uniform_sources if args.distribution == "uniform" else generate_chaotic_sources
    data_model = UNIFORM_DATA_MODEL if args.distribution == "uniform" else DATA_MODEL

    # ``source_count`` is the pass-2 memory size. Pass 1 contains exactly one
    # additional known source, which is removed before the independent rerun.
    all_sources = generator(
        args.source_count + 1,
        args.particles_per_source,
        args.dimension,
        args.data_scale,
        np.random.default_rng(args.seed),
    )  # shape: [C+1,P,D]
    target_id = int(np.random.default_rng(args.seed + 1).integers(all_sources.shape[0]))
    retained_source_ids = np.delete(np.arange(all_sources.shape[0], dtype=np.int64), target_id)  # shape: [C]
    generation_sources = all_sources[retained_source_ids].copy()  # target removed, shape: [C,P,D]
    target_source = all_sources[target_id].copy()  # shape: [P,D]
    initial_scale = coordinate_scale(generation_sources)  # shape: [D]
    memory_std = float(np.std(generation_sources))

    # Source meaning is available only in the original data plane d1. Select
    # K complete neighbors here and preserve both source IDs and the particle
    # correspondence through both deterministic forward passes.
    parent_ids, parent_permutations, matched_initial_reference, parent_distances = select_source_neighbors(
        all_sources, target_id, args.parents, initial_scale
    )  # shapes: [K], [K,P], [K,P,D], [K]

    reference_plane, reference_history, reference_forward, reference_passed, reference_reasons, reference_smoke, reference_seconds = _forward_pass(
        all_sources, args, "pass 1, target present"
    )
    arrays: dict[str, np.ndarray] = {
        "two_pass_protocol": np.asarray(1, dtype=np.int64),
        "method_names": METHOD_NAMES,
        "all_sources": all_sources,
        "memory_sources": generation_sources,
        "target_sources": target_source[None, :, :],  # [P,D] -> [R=1,P,D]
        "target_source_ids": np.asarray((target_id,), dtype=np.int64),
        "retained_source_ids": retained_source_ids,
        "parent_source_ids": parent_ids[None, :],  # [K] -> [R=1,K]
        "parent_permutations": parent_permutations[None, :, :],  # [K,P] -> [R=1,K,P]
        "parent_initial_distances": parent_distances,
        "matched_initial_reference_parents": matched_initial_reference,
        "initial_scale": initial_scale,
        "reference_initial_plane": reference_plane,
    }
    _add_forward_arrays(arrays, "reference", reference_history, reference_forward)
    config: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "two_pass_target_removed",
        "seed": args.seed,
        "reference_source_count": int(all_sources.shape[0]),
        "generation_source_count": int(generation_sources.shape[0]),
        "target_source_id": target_id,
        "parent_source_ids": parent_ids.tolist(),
        "parent_initial_distances": parent_distances.tolist(),
        "particles_per_source": args.particles_per_source,
        "dimension": args.dimension,
        "parent_count": args.parents,
        "vertex_source_count": args.vertex_sources,
        "data_scale": args.data_scale,
        "epsilon": args.epsilon,
        "exponent_s": args.exponent_s,
        "gamma": args.gamma,
        "forward_steps_requested": args.forward_steps,
        "beta": args.beta,
        "proximal_steps": args.proximal_steps,
        "lambda_iterations": args.lambda_iterations,
        "log_every": args.log_every,
        "workers": args.workers,
        "quick": bool(args.quick),
        "parameter_source": parameter_source,
        "loaded_parameter_config_path": args.loaded_parameter_config_path,
        "loaded_parameter_config": args.loaded_parameter_config,
        "plane_model": "one pooled [N,D] field; source ownership is metadata only",
        "data_model": data_model,
        "particle_distribution": args.distribution,
        "neighbor_frame": "initial data plane d1",
        "lambda_fit_frame": "pass-1 terminal plane du with target present",
        "generation_frame": "pass-2 terminal plane du with target absent",
        "algebra_self_check": algebra,
        "reference_forward_seconds": reference_seconds,
        "reference_smoke_passed": reference_passed,
        "reference_smoke_failure_reasons": reference_reasons,
        "reference_smoke": reference_smoke,
    }
    write_smoke_csv(directory / "reference_smoke.csv", reference_smoke, reference_passed)
    if not reference_passed:
        config["status"] = "invalid reference forward field"
        summary = [
            "Two-pass single-plane H4.5",
            "",
            "status: invalid reference forward field",
            f"reasons: {'; '.join(reference_reasons)}",
            "pass 2 and target replay were not run.",
        ]
        _save_and_plot(directory, arrays, config, summary)
        return directory

    reference_terminal_sources = reference_history[-1].reshape(all_sources.shape)  # [N1,D] -> [C+1,P,D]
    target_terminal = reference_terminal_sources[target_id]  # shape: [P,D]
    reference_terminal_scale = np.maximum(np.std(reference_history[-1], axis=0), 1.0e-12)  # shape: [D]
    matched_reference_terminal = np.empty(
        (args.parents, args.particles_per_source, args.dimension), dtype=np.float64
    )  # shape: [K,P,D]
    for parent, source_id in enumerate(parent_ids):
        matched_reference_terminal[parent] = reference_terminal_sources[source_id][
            parent_permutations[parent]
        ]  # persistent d1 matching applied to exact pass-1 terminal rows, shape: [P,D]

    shared_lambda, reference_candidate, lambda_fit_rmse = fit_shared_source_lambda(
        target_terminal,
        matched_reference_terminal,
        reference_terminal_scale,
        max_iterations=args.lambda_iterations,
    )  # shapes: [K], [P,D], scalar
    shared_lambdas = shared_lambda[None, None, :]  # [K] -> [R=1,M=1,K]

    # Persist the complete pass-1 scientific witness before starting the
    # second expensive forward pass. This is not a resume system; it prevents
    # target/parent/lambda provenance from being lost if pass 2 is interrupted.
    np.savez_compressed(
        directory / "pass1_reference.npz",
        all_sources=all_sources,
        target_source_id=np.asarray(target_id, dtype=np.int64),
        target_source=target_source,
        parent_source_ids=parent_ids,
        parent_permutations=parent_permutations,
        reference_terminal_sources=reference_terminal_sources,
        target_terminal=target_terminal,
        matched_reference_terminal_parents=matched_reference_terminal,
        shared_lambda=shared_lambda,
        reference_candidate=reference_candidate,
    )
    config.update({
        "status": "pass 1 complete; pass 2 pending",
        "shared_lambda": shared_lambda.tolist(),
        "lambda_fit_rmse": lambda_fit_rmse,
    })
    _write_json(directory / "config.json", config)

    generation_plane, generation_history, generation_forward, generation_passed, generation_reasons, generation_smoke, generation_seconds = _forward_pass(
        generation_sources, args, "pass 2, target removed"
    )
    arrays["generation_initial_plane"] = generation_plane
    _add_forward_arrays(arrays, "generation", generation_history, generation_forward)
    config.update({
        "generation_forward_seconds": generation_seconds,
        "generation_smoke_passed": generation_passed,
        "generation_smoke_failure_reasons": generation_reasons,
        "generation_smoke": generation_smoke,
        "shared_lambda": shared_lambda.tolist(),
        "lambda_fit_rmse": lambda_fit_rmse,
    })
    if not generation_passed:
        write_smoke_csv(directory / "generation_smoke.csv", generation_smoke, generation_passed)
        config["status"] = "invalid generation forward field"
        summary = [
            "Two-pass single-plane H4.5",
            "",
            "status: invalid generation forward field",
            f"reasons: {'; '.join(generation_reasons)}",
            "target replay was not run.",
        ]
        _save_and_plot(directory, arrays, config, summary)
        return directory

    generation_terminal_sources = generation_history[-1].reshape(generation_sources.shape)  # [N2,D] -> [C,P,D]
    global_to_local = np.full(all_sources.shape[0], -1, dtype=np.int64)  # global source ID -> pass-2 row
    global_to_local[retained_source_ids] = np.arange(retained_source_ids.size)
    parent_local_ids = global_to_local[parent_ids]  # shape: [K]
    if np.any(parent_local_ids < 0):
        raise RuntimeError("a saved parent was removed with the target")

    matched_initial, direct_candidates = interpolate_sources(
        generation_sources,
        parent_local_ids[None, :],
        parent_permutations[None, :, :],
        shared_lambdas,
    )  # shapes: [R=1,K,P,D], [R=1,M=1,P,D]
    matched_generation_terminal, generation_candidates = interpolate_sources(
        generation_terminal_sources,
        parent_local_ids[None, :],
        parent_permutations[None, :, :],
        shared_lambdas,
    )  # same saved d1 parents and lambda, but exact pass-2 terminal coordinates

    # Compare only retained particle identities. Row r in the first array and
    # row r in the second array came from the same initial particle; the sole
    # difference is whether target T participated in the forward field.
    retained_reference_terminal = reference_terminal_sources[retained_source_ids].reshape(
        generation_plane.shape
    )  # [C,P,D] -> [N2,D]
    field_shift = generation_history[-1] - retained_reference_terminal  # shape: [N2,D]
    field_shift_norm = np.linalg.norm(field_shift, axis=1)  # one raw displacement per retained particle, shape: [N2]
    field_kl = marginal_histogram_kl(retained_reference_terminal, generation_history[-1])
    generation_terminal_scale = np.maximum(np.std(generation_history[-1], axis=0), 1.0e-12)  # shape: [D]
    terminal_start_error = float(
        np.sqrt(
            np.mean(
                ((generation_candidates[0, 0] - target_terminal) / reference_terminal_scale[None, :]) ** 2
            )
        )
    )
    config.update({
        "mean_pass1_to_pass2_kl": float(field_kl["mean_reference_to_candidate"]),
        "mean_pass2_to_pass1_kl": float(field_kl["mean_candidate_to_reference"]),
        "mean_retained_terminal_displacement": float(np.mean(field_shift_norm)),
        "p95_retained_terminal_displacement": float(np.quantile(field_shift_norm, 0.95)),
        "terminal_start_error": terminal_start_error,
    })

    vertex_rng = np.random.default_rng(args.seed + 2)
    vertex_local_ids = vertex_rng.choice(
        generation_sources.shape[0], size=args.vertex_sources, replace=False
    )  # shape: [V]
    vertex_started = time.perf_counter()
    vertex_trajectory, vertex_residual, vertex_logs = _parallel_backward_replay(
        generation_terminal_sources[vertex_local_ids],
        generation_history,
        gamma=args.gamma,
        epsilon=args.epsilon,
        exponent_s=args.exponent_s,
        beta=args.beta,
        proximal_steps=args.proximal_steps,
        method_names=["vertex"] * args.vertex_sources,
        log_every=args.log_every,
        workers=args.workers,
    )
    vertex_seconds = time.perf_counter() - vertex_started
    vertex = vertex_replay_metrics(
        vertex_trajectory[0], generation_sources[vertex_local_ids], memory_std, vertex_residual
    )
    vertex_global_ids = retained_source_ids[vertex_local_ids]
    write_smoke_csv(
        directory / "generation_smoke.csv",
        generation_smoke,
        generation_passed,
        vertex_global_ids,
        vertex,
    )
    config.update({
        "vertex_seconds": vertex_seconds,
        "vertex_passed": bool(vertex["passed"]),
        "vertex_median_relative_rmse": float(vertex["median"]),
        "vertex_p95_relative_rmse": float(vertex["p95"]),
        "vertex_maximum_relative_rmse": float(vertex["maximum"]),
    })

    reference_target_rows = slice(
        target_id * args.particles_per_source, (target_id + 1) * args.particles_per_source
    )
    reference_target_trajectory = reference_history[:, reference_target_rows]  # shape: [J+1,P,D]
    arrays.update({
        "reference_terminal_sources": reference_terminal_sources,
        "generation_terminal_sources": generation_terminal_sources,
        "retained_reference_terminal": retained_reference_terminal,
        "reference_terminal_scale": reference_terminal_scale,
        "generation_terminal_scale": generation_terminal_scale,
        "target_terminal": target_terminal,
        "reference_target_trajectory": reference_target_trajectory,
        "matched_reference_terminal_parents": matched_reference_terminal,
        "matched_generation_terminal_parents": matched_generation_terminal[0],
        "shared_lambdas": shared_lambdas,
        "lambda_fit_rmse": np.asarray((lambda_fit_rmse,)),
        "reference_candidate": reference_candidate,
        "generation_candidates": generation_candidates,
        "direct_candidates": direct_candidates,
        "matched_parents": matched_initial,
        "field_shift": field_shift,
        "field_shift_norm": field_shift_norm,
        "field_kl_reference_to_generation": np.asarray(field_kl["reference_to_candidate"]),
        "field_kl_generation_to_reference": np.asarray(field_kl["candidate_to_reference"]),
        "field_kl_bin_edges": np.asarray(field_kl["bin_edges"]),
        "field_kl_reference_probability": np.asarray(field_kl["reference_probability"]),
        "field_kl_generation_probability": np.asarray(field_kl["candidate_probability"]),
        "vertex_source_ids": vertex_global_ids,
        "vertex_trajectory": vertex_trajectory,
        "vertex_residual": vertex_residual,
        "vertex_particle_relative_rmse": np.asarray(vertex["particle_relative_rmse"]),
        "vertex_source_relative_rmse": np.asarray(vertex["source_relative_rmse"]),
        "vertex_backward_log_step": vertex_logs["reverse_step"],
        "vertex_backward_log_mean_distance": vertex_logs["mean_pair_distance"],
        "vertex_backward_log_total_position": vertex_logs["total_position"],
        "vertex_backward_log_displacement": vertex_logs["displacement"],
    })
    if not bool(vertex["passed"]):
        config["status"] = "invalid generation vertex replay"
        summary = [
            "Two-pass single-plane H4.5",
            "",
            "status: invalid generation vertex replay",
            f"vertex maximum relative RMSE: {float(vertex['maximum']):.8e}",
            "target replay was not run.",
        ]
        _save_and_plot(directory, arrays, config, summary)
        return directory

    target_started = time.perf_counter()
    flat_trajectory, flat_residual, target_logs = _parallel_backward_replay(
        generation_candidates.reshape(1, args.particles_per_source, args.dimension),  # [1,1,P,D] -> [G=1,P,D]
        generation_history,
        gamma=args.gamma,
        epsilon=args.epsilon,
        exponent_s=args.exponent_s,
        beta=args.beta,
        proximal_steps=args.proximal_steps,
        method_names=[str(METHOD_NAMES[0])],
        log_every=args.log_every,
        workers=args.workers,
    )
    target_seconds = time.perf_counter() - target_started
    target_trajectory = flat_trajectory[:, :, None, :, :]  # [J+1,R=1,P,D] -> [J+1,R=1,M=1,P,D]
    target_residual = flat_residual[:, :, None, :]  # [J,R=1,P] -> [J,R=1,M=1,P]
    generated = target_trajectory[0]  # shape: [R=1,M=1,P,D]
    metrics = reconstruction_metrics(
        generated,
        direct_candidates,
        target_source[None, :, :],
        matched_initial,
        generation_candidates,
        target_terminal[None, :, :],
        target_residual,
        generation_sources,
        initial_scale,
        reference_terminal_scale,
    )
    method_summary = result_summary(metrics, METHOD_NAMES)
    write_metrics_csv(
        directory / "metrics.csv",
        np.asarray((target_id,), dtype=np.int64),
        parent_ids[None, :],
        shared_lambdas,
        np.asarray((lambda_fit_rmse,)),
        metrics,
    )
    arrival_error = float(metrics["generated_set_rmse"][0, 0])
    amplification = arrival_error / max(terminal_start_error, 1.0e-12)
    arrays.update({
        "target_trajectory": target_trajectory,
        "target_residual": target_residual,
        "generated": generated,
        "target_backward_log_step": target_logs["reverse_step"],
        "target_backward_log_mean_distance": target_logs["mean_pair_distance"],
        "target_backward_log_total_position": target_logs["total_position"],
        "target_backward_log_displacement": target_logs["displacement"],
        **{f"metric_{name}": value for name, value in metrics.items()},
    })
    total_seconds = time.perf_counter() - started
    config.update({
        "status": "complete",
        "target_replay_seconds": target_seconds,
        "arrival_error": arrival_error,
        "terminal_to_arrival_amplification": amplification,
        "method_summary": method_summary,
        "total_seconds": total_seconds,
    })
    summary = [
        "Two-pass single-plane H4.5",
        "",
        "status: complete",
        f"target source ID: {target_id}",
        f"saved d1 parent IDs: {parent_ids.tolist()}",
        f"shared lambda fitted in pass 1 du: {np.array2string(shared_lambda, precision=6)}",
        f"pass-1 joint lambda fit RMSE: {lambda_fit_rmse:.8e}",
        f"mean KL(pass 1 || pass 2): {float(field_kl['mean_reference_to_candidate']):.8e}",
        f"mean KL(pass 2 || pass 1): {float(field_kl['mean_candidate_to_reference']):.8e}",
        f"mean retained-particle terminal displacement: {float(np.mean(field_shift_norm)):.8e}",
        f"pass-2 terminal start error: {terminal_start_error:.8e}",
        f"arrival error at d1: {arrival_error:.8e}",
        f"terminal-to-arrival amplification: {amplification:.8e}",
        f"vertex maximum relative RMSE: {float(vertex['maximum']):.8e}",
        f"target replay residual: {float(metrics['max_replay_residual'][0, 0]):.8e}",
        "",
        "Interpretation:",
        "KL describes how much removing one source changed terminal coordinate marginals.",
        "The amplification ratio measures whether the pass-2 terminal mismatch grew during backward replay.",
        f"total runtime: {total_seconds:.2f} s",
    ]
    figures = _save_and_plot(directory, arrays, config, summary)
    print(f"saved {directory}")
    print(f"generated {len(figures)} figure(s)")
    return directory


def run(args: argparse.Namespace) -> Path:
    """Dispatch to the canonical two-pass run or a preserved single-pass toy."""
    if args.protocol == "single-pass-toy":
        return run_single_pass(args)
    return run_two_pass(args)


def build_parser() -> argparse.ArgumentParser:
    """Define the small research CLI."""
    parser = argparse.ArgumentParser(description="Run two-pass H4.5 or a preserved single-pass toy.")
    parser.add_argument(
        "--protocol",
        choices=("two-pass", "single-pass-toy"),
        default="two-pass",
        help="Canonical target-removal run or preserved earlier toy behavior.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic NumPy seed.")
    parser.add_argument(
        "--source-count", type=int, default=512, help="Pass-2 memory sources; pass 1 adds one target source."
    )
    parser.add_argument(
        "--heldout-sources",
        type=int,
        default=1,
        help="Two-pass requires one; single-pass toys may use several targets.",
    )
    parser.add_argument(
        "--distribution",
        choices=("chaotic", "uniform"),
        default="chaotic",
        help="Particle law; uniform is the simple two-dimensional point control.",
    )
    parser.add_argument(
        "--particles-per-source", type=int, default=10, help="Randomly grouped particles per source; maximum 16."
    )
    parser.add_argument("--dimension", type=int, default=4, help="Particle dimension for the synthetic experiment.")
    parser.add_argument("--parents", type=int, default=4, help="Nearest complete sources selected in the d1 plane.")
    parser.add_argument("--vertex-sources", type=int, default=4, help="Exact pass-2 memory sources used for inverse checking.")
    parser.add_argument("--data-scale", type=float, default=1.0, help="Per-coordinate scale of the chaotic particle pool.")
    parser.add_argument("--epsilon", type=float, default=None, help="EFS smoothing; CLI overrides best_config.json.")
    parser.add_argument("--exponent-s", type=float, default=None, help="EFS exponent; CLI overrides best_config.json.")
    parser.add_argument("--gamma", type=float, default=None, help="Euler step; CLI overrides best_config.json.")
    parser.add_argument("--forward-steps", type=int, default=None, help="Forward frames; CLI overrides best_config.json.")
    parser.add_argument("--beta", type=float, default=None, help="Replay step; CLI overrides best_config.json.")
    parser.add_argument("--proximal-steps", type=int, default=None, help="Replay iterations; CLI overrides best_config.json.")
    parser.add_argument(
        "--lambda-iterations", type=int, default=1000, help="Projected-gradient steps for evaluation-only lambda fitting."
    )
    parser.add_argument("--log-every", type=int, default=10, help="Print forward/backward state every this many full frames.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent passive replay batches; reduce if CPU oversubscribes.")
    parser.add_argument(
        "--output-root",
        default="experiments/07_two_pass_reconstruction/results",
        help="Parent directory for a new timestamped result.",
    )
    parser.add_argument("--config", type=Path, default=ACTIVE_CONFIG_PATH, help="Search-produced JSON parameter config.")
    parser.add_argument("--ignore-config", action="store_true", help="Use built-in or explicit CLI values only.")
    parser.add_argument("--quick", action="store_true", help="Tiny path check, not a scientific experiment.")
    return parser


def main() -> None:
    """Parse arguments and run."""
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

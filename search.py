"""Search single-plane EFS geometry and replay hyperparameters.

Purpose:
    Minimize pooled-plane radial and angular mismatch, require numerical gates,
    select a stable vertex-replay setting, and write ``best_config.json``.
Dependencies:
    NumPy, Matplotlib, and Python's standard library.
Outputs:
    A timestamped search directory and, after a full search, the active
    ``best_config.json`` beside ``run.py``.
Exact command:
    python search.py --workers 4 --output-root experiments\\03_slot_single_calibration\\results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from data import DATA_MODEL, generate_chaotic_sources
from efs import backward_replay, forward_history
from evaluation import smoke_gate, vertex_replay_metrics


EPSILONS = (0.01, 0.03, 0.1, 0.3)
EXPONENTS = (1.0, 1.5, 2.0, 2.5, 3.0)
GAMMAS = (0.00025, 0.0005, 0.001, 0.002)
TRANSPORT_TIMES = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
BETAS = (0.025, 0.05, 0.1)
PROXIMAL_STEPS = (25, 50, 100, 200)

FORWARD_FIELDS = (
    "stage",
    "task_id",
    "source_count",
    "memory_particles",
    "seed",
    "epsilon",
    "exponent_s",
    "gamma",
    "transport_time",
    "steps",
    "finite",
    "hard_pass",
    "radius_ratio",
    "radial_cdf_gap",
    "radius_cv",
    "expected_radius_cv",
    "radial_zone_rmse",
    "radial_overflow",
    "angular_resultant",
    "angular_moment_error",
    "covariance_ratio",
    "uniformity_score",
    "runtime_seconds",
    "error",
)

REPLAY_FIELDS = (
    "forward_id",
    "source_count",
    "seed",
    "epsilon",
    "exponent_s",
    "gamma",
    "transport_time",
    "steps",
    "beta",
    "proximal_steps",
    "vertex_sources",
    "particle_median_relative_rmse",
    "particle_p95_relative_rmse",
    "particle_max_relative_rmse",
    "max_replay_residual",
    "gate_pass",
    "forward_seconds",
    "replay_seconds",
    "error",
)


def _settings(quick: bool) -> dict[str, object]:
    """Return the full staged grid or a tiny path-check grid."""
    if not quick:
        return {
            "quick": False,
            "epsilons": list(EPSILONS),
            "exponents": list(EXPONENTS),
            "gammas": list(GAMMAS),
            "transport_times": list(TRANSPORT_TIMES),
            "betas": list(BETAS),
            "proximal_steps": list(PROXIMAL_STEPS),
            "stage_one_gamma": 0.0005,
            "screen_sources": 64,
            "validation_sources": 256,
            "replay_sources": 128,
            "screen_seeds": [42, 50],
            "validation_seeds": [61, 73],
            "replay_seed": 89,
            "keep_potentials": 4,
            "keep_forward": 4,
            "replay_forward_candidates": 2,
            "vertex_sources": 1,
        }
    return {
        "quick": True,
        "epsilons": [0.03, 0.1],
        "exponents": [2.0, 3.0],
        "gammas": [0.001, 0.002],
        "transport_times": [0.02, 0.04],
        "betas": [0.05, 0.1],
        "proximal_steps": [5, 10],
        "stage_one_gamma": 0.001,
        "screen_sources": 16,
        "validation_sources": 32,
        "replay_sources": 32,
        "screen_seeds": [42],
        "validation_seeds": [61],
        "replay_seed": 89,
        "keep_potentials": 2,
        "keep_forward": 2,
        "replay_forward_candidates": 2,
        "vertex_sources": 1,
    }


def _new_directory(output_root: Path) -> Path:
    """Create one append-only timestamped search directory."""
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = output_root / f"one_plane_search_{stamp}"
    directory.mkdir(exist_ok=False)
    return directory


def _write_json(path: Path, values: dict[str, object]) -> None:
    """Write one readable JSON object."""
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_header(path: Path, fields: tuple[str, ...]) -> None:
    """Create a CSV with its declared header."""
    with path.open("x", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Read one small search CSV."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _append_rows(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    """Append completed worker rows and flush for interruption-safe resume."""
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        for row in rows:
            writer.writerow(row)
            handle.flush()


def _initial_memory(
    source_count: int, seed: int, particles_per_source: int = 10, dimension: int = 4, data_scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return memory sources ``[C,P,D]`` and one plane ``[N=C*P,D]``.

    Search needs no target holdout. It calibrates only forward geometry and
    exact memory replay, so every independently grouped source enters EFS.
    """
    memory_sources = generate_chaotic_sources(
        source_count, particles_per_source, dimension, data_scale, np.random.default_rng(seed)
    )  # shape: [C,P,D]
    plane = memory_sources.reshape(-1, dimension).copy()  # [C,P,D] -> [N=C*P,D]
    return memory_sources, plane


def _task_id(task: dict[str, object]) -> str:
    """Return a stable ID for one evolution shared by several checkpoints."""
    return (
        f"{task['stage']}_C{task['source_count']}_seed{task['seed']}_"
        f"eps{float(task['epsilon']):g}_s{float(task['exponent_s']):g}_"
        f"g{float(task['gamma']):g}"
    )


def _forward_key(row: dict[str, object]) -> tuple[str, str, str]:
    """Return the resume key for one checkpoint row."""
    return (str(row["stage"]), str(row["task_id"]), f"{float(row['transport_time']):.12g}")


def _failed_forward_rows(task: dict[str, object], runtime: float, error: str) -> list[dict[str, object]]:
    """Represent every checkpoint after a failed forward worker."""
    return [
        {
            "stage": task["stage"],
            "task_id": _task_id(task),
            "source_count": task["source_count"],
            "memory_particles": "",
            "seed": task["seed"],
            "epsilon": task["epsilon"],
            "exponent_s": task["exponent_s"],
            "gamma": task["gamma"],
            "transport_time": transport_time,
            "steps": math.ceil(float(transport_time) / float(task["gamma"])),
            "finite": 0,
            "hard_pass": 0,
            "radius_ratio": math.nan,
            "radial_cdf_gap": math.inf,
            "radius_cv": math.nan,
            "expected_radius_cv": math.nan,
            "radial_zone_rmse": math.inf,
            "radial_overflow": math.nan,
            "angular_resultant": math.inf,
            "angular_moment_error": math.inf,
            "covariance_ratio": 0.0,
            "uniformity_score": math.inf,
            "runtime_seconds": runtime,
            "error": error,
        }
        for transport_time in task["transport_times"]
    ]


def _forward_worker(task: dict[str, object]) -> list[dict[str, object]]:
    """Run one pooled evolution and score all requested transport checkpoints."""
    started = time.perf_counter()
    try:
        _, initial = _initial_memory(int(task["source_count"]), int(task["seed"]))
        gamma = float(task["gamma"])
        checkpoint_steps = [math.ceil(float(value) / gamma) for value in task["transport_times"]]
        history, diagnostics = forward_history(
            initial,
            gamma=gamma,
            epsilon=float(task["epsilon"]),
            exponent_s=float(task["exponent_s"]),
            steps=max(checkpoint_steps),
            log_every=0,
            save_steps=checkpoint_steps,
        )  # sparse history: [checkpoints+1,N,D]
        saved_step = np.asarray(diagnostics["saved_step"], dtype=np.int64)  # [saved_frames]
        saved = {int(step): history[index] for index, step in enumerate(saved_step)}
        runtime = time.perf_counter() - started
        rows: list[dict[str, object]] = []

        for transport_time, step in zip(task["transport_times"], checkpoint_steps):
            if step not in saved:
                rows.extend(
                    _failed_forward_rows({**task, "transport_times": [transport_time]}, runtime, "checkpoint not reached")
                )
                continue

            hard_pass, reasons, smoke = smoke_gate(initial, saved[step], {"finite": True, "failure_reason": ""})
            terminal = smoke["terminal"]
            zones = np.asarray([
                float(terminal[f"radial_zone_{zone}"]) for zone in range(10)
            ])  # shape: [10], ideal value is 0.10 in every zone
            zone_rmse = float(np.sqrt(np.mean((zones - 0.10) ** 2)))
            radial_gap = float(terminal["radial_cdf_gap"])
            angular_resultant = float(terminal["angular_resultant"])
            angular_moment = float(terminal["angular_moment_error"])

            # Lower is better. Radial CDF, directional bias, and angular
            # second-moment mismatch are the three non-overlapping main terms.
            uniformity_score = radial_gap + angular_resultant + angular_moment
            rows.append({
                "stage": task["stage"],
                "task_id": _task_id(task),
                "source_count": task["source_count"],
                "memory_particles": initial.shape[0],
                "seed": task["seed"],
                "epsilon": task["epsilon"],
                "exponent_s": task["exponent_s"],
                "gamma": gamma,
                "transport_time": transport_time,
                "steps": step,
                "finite": 1,
                "hard_pass": int(hard_pass),
                "radius_ratio": float(terminal["radius_ratio"]),
                "radial_cdf_gap": radial_gap,
                "radius_cv": float(terminal["radius_cv"]),
                "expected_radius_cv": float(terminal["expected_radius_cv"]),
                "radial_zone_rmse": zone_rmse,
                "radial_overflow": float(terminal["radial_overflow"]),
                "angular_resultant": angular_resultant,
                "angular_moment_error": angular_moment,
                "covariance_ratio": float(terminal["covariance_ratio"]),
                "uniformity_score": uniformity_score,
                "runtime_seconds": runtime,
                "error": "; ".join(reasons),
            })
        return rows
    except Exception as error:
        return _failed_forward_rows(task, time.perf_counter() - started, f"{type(error).__name__}: {error}")


def _run_forward_tasks(tasks: list[dict[str, object]], path: Path, workers: int) -> None:
    """Run incomplete pooled forward tasks in independent processes."""
    existing = {_forward_key(row) for row in _read_rows(path)}
    pending: list[dict[str, object]] = []
    for task in tasks:
        expected = {(str(task["stage"]), _task_id(task), f"{float(value):.12g}") for value in task["transport_times"]}
        if not expected.issubset(existing):
            pending.append(task)
    if not pending:
        return

    print(f"running {len(pending)} forward task(s) with {workers} worker(s)")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_forward_worker, task) for task in pending]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows = [row for row in future.result() if _forward_key(row) not in existing]
            _append_rows(path, FORWARD_FIELDS, rows)
            existing.update(_forward_key(row) for row in rows)
            print(f"  forward tasks completed: {completed}/{len(futures)}")


def _aggregate_forward(rows: list[dict[str, str]], stage: str) -> list[dict[str, object]]:
    """Combine seed rows using worst-seed geometry values."""
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        if row["stage"] != stage:
            continue
        key = (row["source_count"], row["epsilon"], row["exponent_s"], row["gamma"], row["transport_time"])
        grouped.setdefault(key, []).append(row)

    aggregates: list[dict[str, object]] = []
    for values in grouped.values():
        first = values[0]
        worst_radial = max(float(value["radial_cdf_gap"]) for value in values)
        worst_resultant = max(float(value["angular_resultant"]) for value in values)
        worst_moment = max(float(value["angular_moment_error"]) for value in values)
        aggregates.append({
            "source_count": int(first["source_count"]),
            "memory_particles": min(int(value["memory_particles"] or 0) for value in values),
            "epsilon": float(first["epsilon"]),
            "exponent_s": float(first["exponent_s"]),
            "gamma": float(first["gamma"]),
            "transport_time": float(first["transport_time"]),
            "steps": int(first["steps"]),
            "hard_pass": all(int(value["hard_pass"]) == 1 for value in values),
            "worst_radial_cdf_gap": worst_radial,
            "worst_radial_zone_rmse": max(float(value["radial_zone_rmse"]) for value in values),
            "worst_angular_resultant": worst_resultant,
            "worst_angular_moment_error": worst_moment,
            "minimum_covariance_ratio": min(float(value["covariance_ratio"]) for value in values),
            "radius_ratio_min": min(float(value["radius_ratio"]) for value in values),
            "radius_ratio_max": max(float(value["radius_ratio"]) for value in values),
            "uniformity_score": worst_radial + worst_resultant + worst_moment,
            "runtime_seconds": sum(float(value["runtime_seconds"]) for value in values),
        })
    return aggregates


def _forward_score(config: dict[str, object]) -> tuple[float, ...]:
    """Rank hard-valid fields by aggregate uniformity, then simpler transport."""
    return (
        0.0 if bool(config["hard_pass"]) else 1.0,
        float(config["uniformity_score"]),
        float(config["worst_radial_cdf_gap"]),
        float(config["worst_angular_moment_error"]),
        float(config["transport_time"]),
        float(config["runtime_seconds"]),
    )


def _best_potential_pairs(aggregates: list[dict[str, object]], keep: int) -> list[dict[str, object]]:
    """Keep each potential pair at its best transport checkpoint."""
    grouped: dict[tuple[float, float], list[dict[str, object]]] = {}
    for config in aggregates:
        key = (float(config["epsilon"]), float(config["exponent_s"]))
        grouped.setdefault(key, []).append(config)
    best = [min(values, key=_forward_score) for values in grouped.values()]
    return sorted(best, key=_forward_score)[:keep]


def _best_forward(aggregates: list[dict[str, object]], keep: int) -> list[dict[str, object]]:
    """Return the best complete forward configurations."""
    return sorted(aggregates, key=_forward_score)[:keep]


def _forward_id(config: dict[str, object]) -> str:
    """Return one readable ID shared by replay rows."""
    return (
        f"eps{float(config['epsilon']):g}_s{float(config['exponent_s']):g}_"
        f"g{float(config['gamma']):g}_tau{float(config['transport_time']):g}"
    )


def _replay_key(row: dict[str, object]) -> tuple[str, str, int, int]:
    """Return the resume key for one replay parameter pair."""
    return (str(row["forward_id"]), f"{float(row['beta']):.12g}", int(row["proximal_steps"]), int(row["seed"]))


def _replay_worker(task: dict[str, object]) -> list[dict[str, object]]:
    """Build one full history, then test all beta and proximal-step pairs."""
    forward_started = time.perf_counter()
    pairs = [(float(beta), int(steps)) for beta, steps in task["pairs"]]
    try:
        memory_sources, initial = _initial_memory(int(task["source_count"]), int(task["seed"]))  # shapes: [C,P,D], [N=C*P,D]
        history, diagnostics = forward_history(
            initial,
            gamma=float(task["gamma"]),
            epsilon=float(task["epsilon"]),
            exponent_s=float(task["exponent_s"]),
            steps=int(task["steps"]),
            log_every=0,
        )  # full replay history: [J+1,N,D]
        forward_seconds = time.perf_counter() - forward_started
        hard_pass, reasons, _ = smoke_gate(initial, history[-1], diagnostics)
        terminal_sources = history[-1].reshape(memory_sources.shape)  # [N,D] -> [C,P,D]
        vertex_ids = np.random.default_rng(int(task["seed"]) + 2).choice(
            memory_sources.shape[0], size=int(task["vertex_sources"]), replace=False
        )  # shape: [V]
        memory_std = float(np.std(initial))
        rows: list[dict[str, object]] = []

        for beta, proximal_steps in pairs:
            replay_started = time.perf_counter()
            error_text = "; ".join(reasons)
            try:
                trajectory, residual, _ = backward_replay(
                    terminal_sources[vertex_ids],
                    history,
                    gamma=float(task["gamma"]),
                    epsilon=float(task["epsilon"]),
                    exponent_s=float(task["exponent_s"]),
                    beta=beta,
                    proximal_steps=proximal_steps,
                    method_names=["vertex"] * int(task["vertex_sources"]),
                    log_every=0,
                )
                metrics = vertex_replay_metrics(trajectory[0], memory_sources[vertex_ids], memory_std, residual)
                particle = np.asarray(metrics["particle_relative_rmse"]).reshape(-1)  # [V,P] -> [V*P]
                median = float(np.median(particle))
                p95 = float(np.quantile(particle, 0.95))
                maximum = float(np.max(particle))
                max_residual = float(np.max(metrics["particle_max_residual"]))
                gate_pass = bool(hard_pass and bool(metrics["passed"]))
            except Exception as error:
                median = p95 = maximum = max_residual = math.inf
                gate_pass = False
                error_text = f"{type(error).__name__}: {error}"

            rows.append({
                "forward_id": task["forward_id"],
                "source_count": task["source_count"],
                "seed": task["seed"],
                "epsilon": task["epsilon"],
                "exponent_s": task["exponent_s"],
                "gamma": task["gamma"],
                "transport_time": task["transport_time"],
                "steps": task["steps"],
                "beta": beta,
                "proximal_steps": proximal_steps,
                "vertex_sources": task["vertex_sources"],
                "particle_median_relative_rmse": median,
                "particle_p95_relative_rmse": p95,
                "particle_max_relative_rmse": maximum,
                "max_replay_residual": max_residual,
                "gate_pass": int(gate_pass),
                "forward_seconds": forward_seconds,
                "replay_seconds": time.perf_counter() - replay_started,
                "error": error_text,
            })
        return rows
    except Exception as error:
        forward_seconds = time.perf_counter() - forward_started
        return [
            {
                "forward_id": task["forward_id"],
                "source_count": task["source_count"],
                "seed": task["seed"],
                "epsilon": task["epsilon"],
                "exponent_s": task["exponent_s"],
                "gamma": task["gamma"],
                "transport_time": task["transport_time"],
                "steps": task["steps"],
                "beta": beta,
                "proximal_steps": proximal_steps,
                "vertex_sources": task["vertex_sources"],
                "particle_median_relative_rmse": math.inf,
                "particle_p95_relative_rmse": math.inf,
                "particle_max_relative_rmse": math.inf,
                "max_replay_residual": math.inf,
                "gate_pass": 0,
                "forward_seconds": forward_seconds,
                "replay_seconds": 0.0,
                "error": f"{type(error).__name__}: {error}",
            }
            for beta, proximal_steps in pairs
        ]


def _run_replay_tasks(tasks: list[dict[str, object]], path: Path, workers: int) -> None:
    """Run incomplete replay screens in independent processes."""
    existing = {_replay_key(row) for row in _read_rows(path)}
    pending: list[dict[str, object]] = []
    for task in tasks:
        expected = {
            (str(task["forward_id"]), f"{float(beta):.12g}", int(steps), int(task["seed"])) for beta, steps in task["pairs"]
        }
        if not expected.issubset(existing):
            pending.append(task)
    if not pending:
        return

    print(f"running {len(pending)} replay task(s) with {workers} worker(s)")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_replay_worker, task) for task in pending]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows = [row for row in future.result() if _replay_key(row) not in existing]
            _append_rows(path, REPLAY_FIELDS, rows)
            existing.update(_replay_key(row) for row in rows)
            print(f"  replay tasks completed: {completed}/{len(futures)}")


def _best_replay(rows: list[dict[str, str]], forward_id: str) -> dict[str, object] | None:
    """Choose the smallest near-best fixed-point iteration count."""
    valid = [row for row in rows if row["forward_id"] == forward_id and int(row["gate_pass"]) == 1]
    if not valid:
        return None
    best_p95 = min(float(row["particle_p95_relative_rmse"]) for row in valid)
    tolerance = max(0.10 * best_p95, 1.0e-12)
    near = [row for row in valid if float(row["particle_p95_relative_rmse"]) <= best_p95 + tolerance]
    chosen = min(
        near, key=lambda row: (int(row["proximal_steps"]), float(row["max_replay_residual"]), float(row["replay_seconds"]))
    )
    return {
        "beta": float(chosen["beta"]),
        "proximal_steps": int(chosen["proximal_steps"]),
        "particle_median_relative_rmse": float(chosen["particle_median_relative_rmse"]),
        "particle_p95_relative_rmse": float(chosen["particle_p95_relative_rmse"]),
        "particle_max_relative_rmse": float(chosen["particle_max_relative_rmse"]),
        "max_replay_residual": float(chosen["max_replay_residual"]),
    }


def _plot_selection(
    directory: Path, forward_rows: list[dict[str, str]], replay_rows: list[dict[str, str]], selected: dict[str, object]
) -> None:
    """Plot the three hyperparameter planes and selected geometry quality."""
    import matplotlib.pyplot as plt

    potential = _aggregate_forward(forward_rows, "potential")
    euler = _aggregate_forward(forward_rows, "euler")

    # Show each epsilon/s pair at its best checkpoint across the potential screen.
    potential_by_pair: dict[tuple[float, float], dict[str, object]] = {}
    for value in potential:
        key = (float(value["epsilon"]), float(value["exponent_s"]))
        if key not in potential_by_pair or _forward_score(value) < _forward_score(potential_by_pair[key]):
            potential_by_pair[key] = value

    # Show each gamma/time point at its best retained potential pair.
    euler_by_pair: dict[tuple[float, float], dict[str, object]] = {}
    for value in euler:
        key = (float(value["gamma"]), float(value["transport_time"]))
        if key not in euler_by_pair or _forward_score(value) < _forward_score(euler_by_pair[key]):
            euler_by_pair[key] = value

    selected_forward_id = str(selected["forward_id"])
    replay_by_pair: dict[tuple[float, int], dict[str, str]] = {}
    for row in replay_rows:
        if row["forward_id"] != selected_forward_id:
            continue
        key = (float(row["beta"]), int(row["proximal_steps"]))
        if key not in replay_by_pair or float(row["particle_p95_relative_rmse"]) < float(
            replay_by_pair[key]["particle_p95_relative_rmse"]
        ):
            replay_by_pair[key] = row

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 10.0), constrained_layout=True)

    values = [value for value in potential_by_pair.values() if np.isfinite(float(value["uniformity_score"]))]
    scatter = axes[0, 0].scatter(
        [value["epsilon"] for value in values],
        [value["exponent_s"] for value in values],
        c=[value["uniformity_score"] for value in values],
        cmap="viridis_r",
        s=80,
    )
    axes[0, 0].scatter(
        selected["epsilon"],
        selected["exponent_s"],
        marker="*",
        s=260,
        facecolors="none",
        edgecolors="red",
        linewidths=2.0,
        label="selected",
    )
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("epsilon")
    axes[0, 0].set_ylabel("exponent s")
    axes[0, 0].set_title("Potential selection, lower color is better")
    axes[0, 0].legend()
    figure.colorbar(scatter, ax=axes[0, 0], label="worst-seed uniformity score")

    values = [value for value in euler_by_pair.values() if np.isfinite(float(value["uniformity_score"]))]
    scatter = axes[0, 1].scatter(
        [value["gamma"] for value in values],
        [value["transport_time"] for value in values],
        c=[value["uniformity_score"] for value in values],
        cmap="viridis_r",
        s=80,
    )
    axes[0, 1].scatter(
        selected["gamma"],
        selected["transport_time"],
        marker="*",
        s=260,
        facecolors="none",
        edgecolors="red",
        linewidths=2.0,
        label="selected",
    )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("gamma")
    axes[0, 1].set_ylabel("transport time tau = gamma * steps")
    axes[0, 1].set_title("Euler and duration selection")
    axes[0, 1].legend()
    figure.colorbar(scatter, ax=axes[0, 1], label="worst-seed uniformity score")

    finite_replay = [row for row in replay_by_pair.values() if np.isfinite(float(row["particle_p95_relative_rmse"]))]
    scatter = axes[1, 0].scatter(
        [float(row["beta"]) for row in finite_replay],
        [int(row["proximal_steps"]) for row in finite_replay],
        c=[float(row["particle_p95_relative_rmse"]) for row in finite_replay],
        cmap="plasma_r",
        s=80,
    )
    axes[1, 0].scatter(
        selected["beta"],
        selected["proximal_steps"],
        marker="*",
        s=260,
        facecolors="none",
        edgecolors="red",
        linewidths=2.0,
        label="selected",
    )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("beta")
    axes[1, 0].set_ylabel("proximal iterations T")
    axes[1, 0].set_title("Vertex replay selection")
    axes[1, 0].legend()
    figure.colorbar(scatter, ax=axes[1, 0], label="particle p95 relative RMSE")

    metric_names = ("radial CDF", "radial zones", "mean direction", "angular moment")
    metric_values = np.asarray((
        selected["worst_radial_cdf_gap"] / 0.05,
        selected["worst_radial_zone_rmse"] / 0.05,
        selected["worst_angular_resultant"] / 0.10,
        selected["worst_angular_moment_error"] / 0.15,
    ))
    axes[1, 1].bar(np.arange(metric_values.size), metric_values, color="#377EB8")
    axes[1, 1].axhline(1.0, color="black", linestyle="--", label="interpretation guide")
    axes[1, 1].set_xticks(np.arange(metric_values.size), metric_names, rotation=18)
    axes[1, 1].set_ylabel("selected mismatch / descriptive guide")
    axes[1, 1].set_title("Selected terminal geometry, lower is better")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.savefig(directory / "hyperparameter_selection.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _install_config(config: dict[str, object], quick: bool) -> Path | None:
    """Atomically install a completed full-search configuration."""
    if quick:
        return None

    active = Path(__file__).resolve().with_name("best_config.json")
    temporary = active.with_name("best_config.json.tmp")
    _write_json(temporary, config)
    temporary.replace(active)
    return active


def self_check() -> None:
    """Check that worst-seed aggregation and hard-gate ranking stay intact."""
    base = {
        "stage": "check",
        "task_id": "a",
        "source_count": "16",
        "memory_particles": "120",
        "epsilon": "0.1",
        "exponent_s": "2.0",
        "gamma": "0.001",
        "transport_time": "0.02",
        "steps": "20",
        "finite": "1",
        "hard_pass": "1",
        "radius_ratio": "1.0",
        "radius_cv": "0.2",
        "expected_radius_cv": "0.2",
        "radial_zone_rmse": "0.02",
        "radial_overflow": "0.0",
        "covariance_ratio": "0.5",
        "runtime_seconds": "1.0",
        "error": "",
    }
    rows = [
        {
            **base,
            "seed": "1",
            "radial_cdf_gap": "0.03",
            "angular_resultant": "0.02",
            "angular_moment_error": "0.04",
            "uniformity_score": "0.09",
        },
        {
            **base,
            "seed": "2",
            "radial_cdf_gap": "0.07",
            "angular_resultant": "0.05",
            "angular_moment_error": "0.06",
            "uniformity_score": "0.18",
        },
    ]
    aggregate = _aggregate_forward(rows, "check")[0]
    assert math.isclose(float(aggregate["worst_radial_cdf_gap"]), 0.07)
    assert math.isclose(float(aggregate["uniformity_score"]), 0.18)
    assert _forward_score(aggregate)[0] == 0.0


def run_search(args: argparse.Namespace) -> Path:
    """Run or resume the one-plane staged calibration."""
    if not 1 <= args.workers <= 4:
        raise ValueError("--workers must be between one and four")
    self_check()

    if args.resume is None:
        directory = _new_directory(Path(args.output_root).resolve())
        settings = _settings(args.quick)
        search_config = {
            **settings,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "workers_requested": args.workers,
            "plane_model": "one pooled [N,D] field; no slot planes or slot centroids",
            "data_model": DATA_MODEL,
            "particles_per_source": 10,
            "dimension": 4,
            "data_scale": 1.0,
            "ranking": "hard gates, then worst-seed radial+direction+angular-moment score",
            "targets_used_for_calibration": False,
        }
        _write_json(directory / "search_config.json", search_config)
        _write_header(directory / "forward_grid.csv", FORWARD_FIELDS)
        _write_header(directory / "replay_grid.csv", REPLAY_FIELDS)
    else:
        directory = args.resume.resolve()
        with (directory / "search_config.json").open(encoding="utf-8") as handle:
            search_config = json.load(handle)
        if search_config.get("data_model") != DATA_MODEL or int(search_config.get("particles_per_source", -1)) != 10:
            raise ValueError("cannot resume a search created for the previous particle distribution")
        settings = {key: value for key, value in search_config.items() if key in _settings(bool(search_config["quick"]))}

    forward_path = directory / "forward_grid.csv"
    replay_path = directory / "replay_grid.csv"

    potential_tasks = [
        {
            "stage": "potential",
            "source_count": settings["screen_sources"],
            "seed": seed,
            "epsilon": epsilon,
            "exponent_s": exponent_s,
            "gamma": settings["stage_one_gamma"],
            "transport_times": settings["transport_times"],
        }
        for epsilon in settings["epsilons"]
        for exponent_s in settings["exponents"]
        for seed in settings["screen_seeds"]
    ]
    _run_forward_tasks(potential_tasks, forward_path, args.workers)
    potential = _aggregate_forward(_read_rows(forward_path), "potential")
    retained_potentials = _best_potential_pairs(potential, int(settings["keep_potentials"]))

    euler_tasks = [
        {
            "stage": "euler",
            "source_count": settings["screen_sources"],
            "seed": seed,
            "epsilon": potential_config["epsilon"],
            "exponent_s": potential_config["exponent_s"],
            "gamma": gamma,
            "transport_times": settings["transport_times"],
        }
        for potential_config in retained_potentials
        for gamma in settings["gammas"]
        for seed in settings["screen_seeds"]
    ]
    _run_forward_tasks(euler_tasks, forward_path, args.workers)
    euler = _aggregate_forward(_read_rows(forward_path), "euler")
    retained_forward = _best_forward(euler, int(settings["keep_forward"]))

    validation_tasks = [
        {
            "stage": "validation",
            "source_count": settings["validation_sources"],
            "seed": seed,
            "epsilon": config["epsilon"],
            "exponent_s": config["exponent_s"],
            "gamma": config["gamma"],
            "transport_times": [config["transport_time"]],
        }
        for config in retained_forward
        for seed in settings["validation_seeds"]
    ]
    _run_forward_tasks(validation_tasks, forward_path, args.workers)
    validation = _aggregate_forward(_read_rows(forward_path), "validation")
    forward_candidates = _best_forward(validation, int(settings["replay_forward_candidates"]))
    if not any(bool(config["hard_pass"]) for config in forward_candidates):
        raise RuntimeError("no forward configuration passed finite, scale, and covariance gates")

    replay_pairs = [(beta, steps) for beta in settings["betas"] for steps in settings["proximal_steps"]]
    replay_tasks = [
        {
            "forward_id": _forward_id(config),
            "source_count": settings["replay_sources"],
            "seed": settings["replay_seed"],
            "epsilon": config["epsilon"],
            "exponent_s": config["exponent_s"],
            "gamma": config["gamma"],
            "transport_time": config["transport_time"],
            "steps": config["steps"],
            "vertex_sources": settings["vertex_sources"],
            "pairs": replay_pairs,
        }
        for config in forward_candidates
        if bool(config["hard_pass"])
    ]
    _run_replay_tasks(replay_tasks, replay_path, args.workers)
    replay_rows = _read_rows(replay_path)

    chosen_forward: dict[str, object] | None = None
    chosen_replay: dict[str, object] | None = None
    for config in forward_candidates:
        candidate = _best_replay(replay_rows, _forward_id(config))
        if bool(config["hard_pass"]) and candidate is not None:
            chosen_forward = config
            chosen_replay = candidate
            break
    if chosen_forward is None or chosen_replay is None:
        raise RuntimeError("no forward finalist passed the 10% vertex replay gate")

    geometry_quality = (
        "close to uniform"
        if (
            float(chosen_forward["worst_radial_cdf_gap"]) <= 0.05
            and float(chosen_forward["worst_angular_resultant"]) <= 0.10
            and float(chosen_forward["worst_angular_moment_error"]) <= 0.15
        )
        else "best available but visibly imperfect"
    )
    selected = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_by": "single-plane search.py",
        "data_model": DATA_MODEL,
        "particles_per_source": 10,
        "search_directory": str(directory.resolve()),
        "forward_id": _forward_id(chosen_forward),
        "quick": bool(settings["quick"]),
        "epsilon": float(chosen_forward["epsilon"]),
        "exponent_s": float(chosen_forward["exponent_s"]),
        "gamma": float(chosen_forward["gamma"]),
        "transport_time": float(chosen_forward["transport_time"]),
        "forward_steps": int(chosen_forward["steps"]),
        "beta": float(chosen_replay["beta"]),
        "proximal_steps": int(chosen_replay["proximal_steps"]),
        "geometry_quality": geometry_quality,
        "uniformity_score": float(chosen_forward["uniformity_score"]),
        "worst_radial_cdf_gap": float(chosen_forward["worst_radial_cdf_gap"]),
        "worst_radial_zone_rmse": float(chosen_forward["worst_radial_zone_rmse"]),
        "worst_angular_resultant": float(chosen_forward["worst_angular_resultant"]),
        "worst_angular_moment_error": float(chosen_forward["worst_angular_moment_error"]),
        "minimum_covariance_ratio": float(chosen_forward["minimum_covariance_ratio"]),
        "radius_ratio_min": float(chosen_forward["radius_ratio_min"]),
        "radius_ratio_max": float(chosen_forward["radius_ratio_max"]),
        "vertex_particle_p95_relative_rmse": float(chosen_replay["particle_p95_relative_rmse"]),
        "vertex_particle_max_relative_rmse": float(chosen_replay["particle_max_relative_rmse"]),
        "max_replay_residual": float(chosen_replay["max_replay_residual"]),
        "selection_rule": "forward uniformity first; replay must pass maximum relative RMSE <= 0.10",
    }

    _write_json(directory / "best_config.json", selected)
    _plot_selection(directory, _read_rows(forward_path), replay_rows, selected)
    installed = _install_config(selected, bool(settings["quick"]))
    summary = [
        "Single-plane EFS hyperparameter search",
        "",
        f"geometry quality: {geometry_quality}",
        f"epsilon: {selected['epsilon']}",
        f"exponent s: {selected['exponent_s']}",
        f"gamma: {selected['gamma']}",
        f"transport time: {selected['transport_time']}",
        f"forward steps: {selected['forward_steps']}",
        f"beta: {selected['beta']}",
        f"proximal steps: {selected['proximal_steps']}",
        f"worst radial CDF gap: {selected['worst_radial_cdf_gap']:.9e}",
        f"worst angular resultant: {selected['worst_angular_resultant']:.9e}",
        f"worst angular moment error: {selected['worst_angular_moment_error']:.9e}",
        f"vertex p95 relative RMSE: {selected['vertex_particle_p95_relative_rmse']:.9e}",
        f"installed active config: {installed if installed is not None else 'no, quick searches never install'}",
        "",
        "Radial and angular values select the best grid point but remain descriptive, not hard proof of uniformity.",
    ]
    (directory / "search_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return directory


def build_parser() -> argparse.ArgumentParser:
    """Define search, resume, and quick-path arguments."""
    parser = argparse.ArgumentParser(description="Search one-plane EFS hyperparameters.")
    parser.add_argument("--workers", type=int, default=4, help="Independent process workers; maximum four.")
    parser.add_argument(
        "--output-root",
        default="experiments/03_slot_single_calibration/results",
        help="Parent for a new timestamped search directory.",
    )
    parser.add_argument(
        "--resume", type=Path, default=None, help="Existing search directory whose finished CSV rows are skipped."
    )
    parser.add_argument("--quick", action="store_true", help="Tiny path check that does not install best_config.json.")
    return parser


def main() -> None:
    """Run the search and print its output directory."""
    directory = run_search(build_parser().parse_args())
    print(f"search results: {directory.resolve()}")


if __name__ == "__main__":
    main()

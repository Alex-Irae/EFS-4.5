"""Run the staged four-worker EFS calibration and final H4.5 evaluation.

Purpose:
    Calibrate forward geometry and literal vertex replay without target-error
    feedback, freeze one configuration per layout, then run the controlled
    H4.5 target evaluation.

Dependencies:
    NumPy and Matplotlib from ../../requirements.txt; multiprocessing is Python stdlib.

Outputs:
    One timestamped search directory containing CSV grids, selection JSON,
    plots, commands, joint-vector smoke output, and final per-seed runs.

Exact command:
    python search.py --workers 4 --output-root results
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

from data import arrange_memory, generate_grouped_sources, restore_sources
from efs import backward_replay, forward_history, rms_radius
from evaluation import RADIAL_ZONE_FIELDS, smoke_gate, vertex_replay_errors


EPSILONS = (0.01, 0.03, 0.1, 0.3)
EXPONENTS = (1.0, 1.5, 2.0, 2.5, 3.0)
GAMMAS = (0.00025, 0.0005, 0.001, 0.002)
TAUS = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
BETAS = (0.025, 0.05, 0.1)
PROXIMAL_STEPS = (25, 50, 100, 200)

FORWARD_FIELDS = (
    "stage",
    "task_id",
    "layout",
    "source_count",
    "seed",
    "epsilon",
    "exponent_s",
    "gamma",
    "tau",
    "steps",
    "finite",
    "hard_pass",
    "radius_ratio_min",
    "radius_ratio_max",
    "worst_radial_cdf_gap",
    "plane_radial_cdf_gaps",
    "mean_radius_cv",
    "expected_radius_cv",
    "worst_angular_resultant",
    "worst_angular_moment_error",
    "minimum_covariance_ratio",
    "mean_radial_zone_fractions",
    "maximum_radial_overflow",
    "runtime_seconds",
    "error",
)

REPLAY_FIELDS = (
    "stage",
    "forward_id",
    "layout",
    "source_count",
    "seed",
    "epsilon",
    "exponent_s",
    "gamma",
    "tau",
    "steps",
    "beta",
    "proximal_steps",
    "vertex_sources",
    "particle_count",
    "particle_median_relative_rmse",
    "particle_p90_relative_rmse",
    "particle_p95_relative_rmse",
    "particle_max_relative_rmse",
    "source_relative_rmse",
    "source_max_relative_rmse",
    "max_replay_residual",
    "forward_seconds",
    "replay_seconds",
    "gate_pass",
    "error",
)

FINAL_RUN_FIELDS = ("layout", "role", "seed", "status", "run_directory")


def _settings(quick: bool) -> dict[str, object]:
    """Return the declared full protocol or a small path-verification protocol."""
    if not quick:
        return {
            "quick": False,
            "epsilons": list(EPSILONS),
            "exponents": list(EXPONENTS),
            "gammas": list(GAMMAS),
            "taus": list(TAUS),
            "betas": list(BETAS),
            "proximal_steps": list(PROXIMAL_STEPS),
            "screen_seeds": [42, 50],
            "potential_sources": 64,
            "refinement_sources": 128,
            "validation_sources": 256,
            "validation_seeds": [61, 73, 89],
            "confirmation_seeds": [73, 89],
            "final_sources": 256,
            "final_seeds": [101, 103, 107],
            "final_trials": 16,
            "final_vertex_sources": 4,
        }
    return {
        "quick": True,
        "epsilons": [0.03, 0.1],
        "exponents": [1.5, 2.0],
        "gammas": [0.001, 0.002],
        "taus": [0.02, 0.04],
        "betas": [0.05, 0.1],
        "proximal_steps": [5, 10],
        "screen_seeds": [42, 50],
        "potential_sources": 16,
        "refinement_sources": 24,
        "validation_sources": 32,
        "validation_seeds": [61],
        "confirmation_seeds": [73],
        "final_sources": 32,
        "final_seeds": [101],
        "final_trials": 2,
        "final_vertex_sources": 2,
    }


def _new_search_directory(output_root: str) -> Path:
    """Create one append-only timestamped search directory."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = root / f"h45_search_{timestamp}"
    directory.mkdir(exist_ok=False)
    return directory


def _write_header(path: Path, fields: tuple[str, ...]) -> None:
    """Create an append-only CSV with only its header."""
    with path.open("x", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Read a small grid CSV into dictionaries."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _append_rows(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    """Append completed rows and flush each one for interruption-safe resume."""
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        for row in rows:
            writer.writerow(row)
            handle.flush()


def _forward_key(row: dict[str, object]) -> tuple[str, str, str]:
    return str(row["stage"]), str(row["task_id"]), f"{float(row['tau']):.12g}"


def _replay_key(row: dict[str, object]) -> tuple[str, str, int, str, int, int]:
    return (
        str(row["stage"]),
        str(row["forward_id"]),
        int(row["seed"]),
        f"{float(row['beta']):.12g}",
        int(row["proximal_steps"]),
        int(row["vertex_sources"]),
    )


def _config_id(config: dict[str, object]) -> str:
    """Return a stable readable identifier for one frozen forward configuration."""
    return (
        f"{config['layout']}_eps{float(config['epsilon']):g}_s{float(config['exponent_s']):g}_"
        f"g{float(config['gamma']):g}_tau{float(config['tau']):g}"
    )


def _forward_task_id(task: dict[str, object]) -> str:
    """Identify one evolution shared by all of its tau checkpoints."""
    return (
        f"{task['stage']}_{task['layout']}_C{task['source_count']}_seed{task['seed']}_"
        f"eps{float(task['epsilon']):g}_s{float(task['exponent_s']):g}_g{float(task['gamma']):g}"
    )


def _failed_forward_rows(task: dict[str, object], runtime: float, error: str) -> list[dict[str, object]]:
    """Represent every missing checkpoint explicitly after a worker failure."""
    task_id = _forward_task_id(task)
    return [{
        "stage": task["stage"],
        "task_id": task_id,
        "layout": task["layout"],
        "source_count": task["source_count"],
        "seed": task["seed"],
        "epsilon": task["epsilon"],
        "exponent_s": task["exponent_s"],
        "gamma": task["gamma"],
        "tau": tau,
        "steps": math.ceil(float(tau) / float(task["gamma"])),
        "finite": 0,
        "hard_pass": 0,
        "radius_ratio_min": math.nan,
        "radius_ratio_max": math.nan,
        "worst_radial_cdf_gap": math.inf,
        "plane_radial_cdf_gaps": "",
        "mean_radius_cv": math.nan,
        "expected_radius_cv": math.nan,
        "worst_angular_resultant": math.inf,
        "worst_angular_moment_error": math.inf,
        "minimum_covariance_ratio": 0.0,
        "mean_radial_zone_fractions": "",
        "maximum_radial_overflow": math.nan,
        "runtime_seconds": runtime,
        "error": error,
    } for tau in task["taus"]]


def _forward_worker(task: dict[str, object]) -> list[dict[str, object]]:
    """Run one fixed potential/gamma/seed evolution and score its checkpoints."""
    started = time.perf_counter()
    try:
        rng = np.random.default_rng(int(task["seed"]))
        sources, _, _ = generate_grouped_sources(
            int(task["source_count"]), 8, 4, 2.0, 1.0, 0.05, rng
        )  # shape: [C,P=8,D=4]
        initial = arrange_memory(sources, str(task["layout"]))  # [P,C,D] or [C*P,D]
        gamma = float(task["gamma"])
        checkpoint_steps = [math.ceil(float(tau) / gamma) for tau in task["taus"]]
        history, diagnostics = forward_history(
            initial,
            gamma=gamma,
            epsilon=float(task["epsilon"]),
            exponent_s=float(task["exponent_s"]),
            steps=max(checkpoint_steps),
            log_every=0,
            save_steps=checkpoint_steps,
        )
        saved = {
            int(step): history[index] for index, step in enumerate(np.asarray(diagnostics["saved_step"]))
        }
        runtime = time.perf_counter() - started
        rows: list[dict[str, object]] = []
        task_id = _forward_task_id(task)
        for tau, step in zip(task["taus"], checkpoint_steps):
            if step not in saved:
                rows.extend(_failed_forward_rows({**task, "taus": [tau]}, runtime, "checkpoint not reached"))
                continue
            hard_pass, reasons, metrics = smoke_gate(initial, saved[step], {"finite": True, "failure_reason": ""})
            terminal = metrics["terminal"]
            radial_gaps = [float(values["radial_cdf_gap"]) for values in terminal]
            radius_ratios = [float(values["radius_ratio"]) for values in terminal]
            zone_mean = np.mean(
                [[float(values[field]) for field in RADIAL_ZONE_FIELDS] for values in terminal], axis=0
            )  # [H,10] -> [10]
            rows.append({
                "stage": task["stage"],
                "task_id": task_id,
                "layout": task["layout"],
                "source_count": task["source_count"],
                "seed": task["seed"],
                "epsilon": task["epsilon"],
                "exponent_s": task["exponent_s"],
                "gamma": gamma,
                "tau": tau,
                "steps": step,
                "finite": 1,
                "hard_pass": int(hard_pass),
                "radius_ratio_min": min(radius_ratios),
                "radius_ratio_max": max(radius_ratios),
                "worst_radial_cdf_gap": max(radial_gaps),
                "plane_radial_cdf_gaps": ";".join(f"{value:.9e}" for value in radial_gaps),
                "mean_radius_cv": float(np.mean([values["radius_cv"] for values in terminal])),
                "expected_radius_cv": float(terminal[0]["expected_radius_cv"]),
                "worst_angular_resultant": max(float(values["angular_resultant"]) for values in terminal),
                "worst_angular_moment_error": max(float(values["angular_moment_error"]) for values in terminal),
                "minimum_covariance_ratio": min(float(values["covariance_ratio"]) for values in terminal),
                "mean_radial_zone_fractions": ";".join(f"{value:.9e}" for value in zone_mean),
                "maximum_radial_overflow": max(float(values["radial_overflow"]) for values in terminal),
                "runtime_seconds": runtime,
                "error": "; ".join(reasons),
            })
        return rows
    except Exception as error:
        return _failed_forward_rows(task, time.perf_counter() - started, f"{type(error).__name__}: {error}")


def _run_forward_tasks(tasks: list[dict[str, object]], path: Path, workers: int) -> None:
    """Run missing forward tasks in four independent worker processes."""
    existing = {_forward_key(row) for row in _read_rows(path)}
    pending = []
    for task in tasks:
        task_id = _forward_task_id(task)
        expected = {(str(task["stage"]), task_id, f"{float(tau):.12g}") for tau in task["taus"]}
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
    """Aggregate seed rows into worst-seed scores for each checkpoint config."""
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        if row["stage"] != stage:
            continue
        key = (
            row["layout"], row["epsilon"], row["exponent_s"], row["gamma"], row["tau"], row["source_count"]
        )
        grouped.setdefault(key, []).append(row)

    aggregates: list[dict[str, object]] = []
    for key, values in grouped.items():
        first = values[0]
        hard_pass = all(int(value["hard_pass"]) == 1 for value in values)
        aggregates.append({
            "layout": first["layout"],
            "source_count": int(first["source_count"]),
            "epsilon": float(first["epsilon"]),
            "exponent_s": float(first["exponent_s"]),
            "gamma": float(first["gamma"]),
            "tau": float(first["tau"]),
            "steps": int(first["steps"]),
            "hard_pass": hard_pass,
            "worst_radial_cdf_gap": max(float(value["worst_radial_cdf_gap"]) for value in values),
            "worst_angular_moment_error": max(float(value["worst_angular_moment_error"]) for value in values),
            "runtime_seconds": sum(float(value["runtime_seconds"]) for value in values),
        })
    return aggregates


def _forward_score(config: dict[str, object]) -> tuple[float, float, float, float]:
    """Sort hard-valid geometry first, then radial, angular, and earlier tau."""
    return (
        0.0 if bool(config["hard_pass"]) else 1.0,
        float(config["worst_radial_cdf_gap"]),
        float(config["worst_angular_moment_error"]),
        float(config["tau"]),
    )


def _potential_pairs(aggregates: list[dict[str, object]], keep: int) -> list[dict[str, object]]:
    """Choose each (epsilon,s) pair at its best tau, then keep the best pairs."""
    by_pair: dict[tuple[float, float], list[dict[str, object]]] = {}
    for config in aggregates:
        by_pair.setdefault((float(config["epsilon"]), float(config["exponent_s"])), []).append(config)
    best = [min(values, key=_forward_score) for values in by_pair.values()]
    return sorted(best, key=_forward_score)[: min(keep, len(best))]


def _complete_configs(aggregates: list[dict[str, object]], keep: int) -> list[dict[str, object]]:
    """Keep the best complete forward configurations by declared score."""
    return sorted(aggregates, key=_forward_score)[: min(keep, len(aggregates))]


def _replay_rows(task: dict[str, object], pairs: list[tuple[float, int]]) -> list[dict[str, object]]:
    """Build one history once, then replay one or more beta/T configurations."""
    forward_started = time.perf_counter()
    rng = np.random.default_rng(int(task["seed"]))
    sources, _, _ = generate_grouped_sources(
        int(task["source_count"]), 8, 4, 2.0, 1.0, 0.05, rng
    )  # shape: [C,P,D]
    initial = arrange_memory(sources, str(task["layout"]))
    steps = math.ceil(float(task["tau"]) / float(task["gamma"]))
    history, diagnostics = forward_history(
        initial,
        gamma=float(task["gamma"]),
        epsilon=float(task["epsilon"]),
        exponent_s=float(task["exponent_s"]),
        steps=steps,
        log_every=0,
    )
    forward_seconds = time.perf_counter() - forward_started
    hard_pass, reasons, _ = smoke_gate(initial, history[-1], diagnostics)
    terminal_sources = restore_sources(history[-1], str(task["layout"]), 8)  # [C,P,D]
    vertex_rng = np.random.default_rng(int(task["seed"]) + 1)
    source_ids = vertex_rng.choice(
        int(task["source_count"]), size=int(task["vertex_sources"]), replace=False
    )  # shape: [V]
    memory_std = float(np.std(sources))
    rows: list[dict[str, object]] = []

    for beta, proximal_steps in pairs:
        replay_started = time.perf_counter()
        error_text = "; ".join(reasons)
        try:
            trajectory, residual, _ = backward_replay(
                terminal_sources[source_ids],
                history,
                gamma=float(task["gamma"]),
                epsilon=float(task["epsilon"]),
                exponent_s=float(task["exponent_s"]),
                beta=beta,
                proximal_steps=proximal_steps,
                method_names=["vertex"] * int(task["vertex_sources"]),
                log_every=0,
            )
            metrics = vertex_replay_errors(trajectory[0], sources[source_ids], memory_std, residual)
            particle_relative = metrics["particle_relative_rmse"].reshape(-1)  # [V,P] -> [V*P]
            quantiles = np.quantile(particle_relative, (0.50, 0.90, 0.95))
            source_relative = metrics["source_relative_rmse"]
            maximum = float(np.max(particle_relative))
            gate_pass = bool(hard_pass and np.all(np.isfinite(particle_relative)) and maximum <= 0.10)
            max_residual = float(np.max(metrics["particle_max_residual"]))
        except Exception as error:
            quantiles = np.full(3, math.inf)
            maximum = math.inf
            source_relative = np.full(int(task["vertex_sources"]), math.inf)
            gate_pass = False
            max_residual = math.inf
            error_text = f"{type(error).__name__}: {error}"
        replay_seconds = time.perf_counter() - replay_started
        rows.append({
            "stage": task["stage"],
            "forward_id": task["forward_id"],
            "layout": task["layout"],
            "source_count": task["source_count"],
            "seed": task["seed"],
            "epsilon": task["epsilon"],
            "exponent_s": task["exponent_s"],
            "gamma": task["gamma"],
            "tau": task["tau"],
            "steps": steps,
            "beta": beta,
            "proximal_steps": proximal_steps,
            "vertex_sources": task["vertex_sources"],
            "particle_count": int(task["vertex_sources"]) * 8,
            "particle_median_relative_rmse": float(quantiles[0]),
            "particle_p90_relative_rmse": float(quantiles[1]),
            "particle_p95_relative_rmse": float(quantiles[2]),
            "particle_max_relative_rmse": maximum,
            "source_relative_rmse": ";".join(f"{value:.9e}" for value in source_relative),
            "source_max_relative_rmse": float(np.max(source_relative)),
            "max_replay_residual": max_residual,
            "forward_seconds": forward_seconds,
            "replay_seconds": replay_seconds,
            "gate_pass": int(gate_pass),
            "error": error_text,
        })
    return rows


def _replay_screen_worker(task: dict[str, object]) -> list[dict[str, object]]:
    try:
        return _replay_rows(task, [(float(beta), int(steps)) for beta, steps in task["pairs"]])
    except Exception as error:
        rows = []
        for beta, steps in task["pairs"]:
            failed = {field: "" for field in REPLAY_FIELDS}
            failed.update({
                "stage": task["stage"], "forward_id": task["forward_id"], "layout": task["layout"],
                "source_count": task["source_count"], "seed": task["seed"], "epsilon": task["epsilon"],
                "exponent_s": task["exponent_s"], "gamma": task["gamma"], "tau": task["tau"],
                "steps": math.ceil(float(task["tau"]) / float(task["gamma"])), "beta": beta,
                "proximal_steps": steps, "vertex_sources": task["vertex_sources"],
                "particle_count": int(task["vertex_sources"]) * 8, "gate_pass": 0,
                "error": f"{type(error).__name__}: {error}",
            })
            rows.append(failed)
        return rows


def _replay_confirmation_worker(task: dict[str, object]) -> list[dict[str, object]]:
    return _replay_screen_worker({**task, "pairs": [(task["beta"], task["proximal_steps"])]})


def _run_replay_tasks(
    tasks: list[dict[str, object]], path: Path, workers: int, worker_function
) -> None:
    """Run missing replay tasks while the parent process owns all CSV writes."""
    existing = {_replay_key(row) for row in _read_rows(path)}
    pending = []
    for task in tasks:
        pairs = task["pairs"] if "pairs" in task else [(task["beta"], task["proximal_steps"])]
        expected = {
            (
                str(task["stage"]), str(task["forward_id"]), int(task["seed"]),
                f"{float(beta):.12g}", int(steps), int(task["vertex_sources"]),
            ) for beta, steps in pairs
        }
        if not expected.issubset(existing):
            pending.append(task)
    if not pending:
        return

    print(f"running {len(pending)} replay task(s) with {workers} worker(s)")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker_function, task) for task in pending]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows = [row for row in future.result() if _replay_key(row) not in existing]
            _append_rows(path, REPLAY_FIELDS, rows)
            existing.update(_replay_key(row) for row in rows)
            print(f"  replay tasks completed: {completed}/{len(futures)}")


def _rank_replay_candidates(rows: list[dict[str, str]], forward_id: str) -> list[dict[str, object]]:
    """Rank passing beta/T values with the declared smallest-T near-best rule."""
    values = [
        row for row in rows
        if row["stage"] == "screen" and row["forward_id"] == forward_id and int(row["gate_pass"]) == 1
    ]
    if not values:
        return []
    best_p95 = min(float(row["particle_p95_relative_rmse"]) for row in values)
    near = [row for row in values if float(row["particle_p95_relative_rmse"]) <= 1.10 * best_p95]
    near.sort(key=lambda row: (
        int(row["proximal_steps"]), float(row["max_replay_residual"]), float(row["replay_seconds"])
    ))
    remaining = [row for row in values if row not in near]
    remaining.sort(key=lambda row: (
        float(row["particle_p95_relative_rmse"]), int(row["proximal_steps"]),
        float(row["max_replay_residual"]), float(row["replay_seconds"]),
    ))
    return [{
        "beta": float(row["beta"]),
        "proximal_steps": int(row["proximal_steps"]),
        "screen_p95": float(row["particle_p95_relative_rmse"]),
        "screen_runtime": float(row["forward_seconds"]) + float(row["replay_seconds"]),
    } for row in near + remaining]


def _joint_vector_smoke(path: Path) -> None:
    """Run the declared informational [64,32] joint-vector smoke once."""
    if path.exists():
        return
    started = time.perf_counter()
    rng = np.random.default_rng(45032)
    sources, _, _ = generate_grouped_sources(64, 8, 4, 2.0, 1.0, 0.05, rng)
    joint = sources.reshape(64, 32)  # [C=64,P=8,D=4] -> [C=64,P*D=32]
    history, diagnostics = forward_history(
        joint, gamma=1.0e-4, epsilon=1.0, exponent_s=30.0, steps=500, log_every=0
    )  # shape: [501,64,32]
    trajectory, residual, _ = backward_replay(
        history[-1, :1, :][:, None, :],  # [1,32] -> one complete vertex [G=1,P=1,D=32]
        history,
        gamma=1.0e-4,
        epsilon=1.0,
        exponent_s=30.0,
        beta=0.05,
        proximal_steps=25,
        method_names=["joint_vertex"],
        log_every=0,
    )
    expected = joint[:1, None, :]  # shape: [G=1,P=1,D=32]
    metrics = vertex_replay_errors(trajectory[0], expected, float(np.std(joint)), residual)
    scale_ratio = rms_radius(history[-1]) / max(rms_radius(history[0]), 1.0e-15)
    lines = [
        "Informational joint-vector smoke; excluded from calibration ranking",
        f"initial shape: {joint.shape}",
        f"history shape: {history.shape}",
        f"finite: {bool(diagnostics['finite']) and bool(np.all(np.isfinite(trajectory)))}",
        f"RMS-radius ratio: {scale_ratio:.9e}",
        f"vertex relative RMSE: {float(metrics['source_relative_rmse'][0]):.9e}",
        f"maximum replay residual: {float(np.max(metrics['particle_max_residual'])):.9e}",
        f"runtime seconds: {time.perf_counter() - started:.3f}",
    ]
    with path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _final_worker(task: dict[str, object]) -> dict[str, object]:
    """Run one frozen per-layout target evaluation in a worker process."""
    from argparse import Namespace
    from run import run

    try:
        args = Namespace(
            output_root=task["output_root"],
            layout=task["layout"],
            seed=task["seed"],
            plane_particles=int(task["source_count"]) * 8,
            particles_per_source=8,
            selected_sources=4,
            trials=task["trials"],
            vertex_sources=task["vertex_sources"],
            dimension=4,
            center_std=2.0,
            particle_scale=1.0,
            noise_std=0.05,
            gamma=task["gamma"],
            epsilon=task["epsilon"],
            exponent_s=task["exponent_s"],
            forward_steps=math.ceil(float(task["tau"]) / float(task["gamma"])),
            beta=task["beta"],
            proximal_steps=task["proximal_steps"],
            log_every=0,
            shared_lambda=None,
        )
        directory, status = run(args)
        return {
            "layout": task["layout"], "role": task["role"], "seed": task["seed"],
            "status": status, "run_directory": str(directory.resolve()),
        }
    except Exception as error:
        return {
            "layout": task["layout"], "role": task["role"], "seed": task["seed"],
            "status": "error", "run_directory": f"{type(error).__name__}: {error}",
        }


def _run_final_tasks(tasks: list[dict[str, object]], path: Path, workers: int) -> None:
    """Run missing frozen target evaluations in parallel and append their paths."""
    existing_rows = _read_rows(path)
    existing = {(row["layout"], int(row["seed"])) for row in existing_rows}
    pending = [task for task in tasks if (str(task["layout"]), int(task["seed"])) not in existing]
    if not pending:
        return
    print(f"running {len(pending)} final H4.5 task(s) with {workers} worker(s)")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_final_worker, task) for task in pending]
        for completed, future in enumerate(as_completed(futures), start=1):
            _append_rows(path, FINAL_RUN_FIELDS, [future.result()])
            print(f"  final tasks completed: {completed}/{len(futures)}")


def _pool_final_results(search_directory: Path) -> dict[str, object]:
    """Combine per-seed raw rows and write a 48-trial descriptive summary."""
    final_rows = _read_rows(search_directory / "final_runs.csv")
    pooled: list[dict[str, str]] = []
    for run_row in final_rows:
        directory = Path(run_row["run_directory"])
        metrics_path = directory / "metrics.csv"
        if run_row["status"] == "error" or not metrics_path.exists():
            continue
        for metric in _read_rows(metrics_path):
            pooled.append({"layout": run_row["layout"], "seed": run_row["seed"], **metric})

    pooled_path = search_directory / "final_metrics.csv"
    if pooled:
        fields = tuple(pooled[0].keys())
        with pooled_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(pooled)
    else:
        pooled_path.write_text("layout,seed\n", encoding="utf-8")

    summary: dict[str, object] = {}
    lines = ["Frozen H4.5 pooled descriptive summary", ""]
    for layout in ("slot", "single"):
        rows = [
            row for row in pooled
            if row["layout"] == layout and row["trial_kind"] == "random"
            and int(row["independent_control_nondegenerate"]) == 1
        ]
        by_trial: dict[tuple[str, str], dict[str, float]] = {}
        for row in rows:
            by_trial.setdefault((row["seed"], row["trial"]), {})[row["method"]] = float(
                row["clean_target_relative_rmse"]
            )
        complete = [value for value in by_trial.values() if {"shared", "independent"}.issubset(value)]
        shared = np.asarray([value["shared"] for value in complete])
        independent = np.asarray([value["independent"] for value in complete])
        wins = int(np.count_nonzero(shared < independent)) if complete else 0
        values = {
            "trials": len(complete),
            "shared_wins": wins,
            "shared_median_clean_relative_rmse": float(np.median(shared)) if complete else math.nan,
            "independent_median_clean_relative_rmse": float(np.median(independent)) if complete else math.nan,
        }
        summary[layout] = values
        lines.extend([
            f"{layout} layout:",
            f"  eligible trials: {values['trials']}",
            f"  shared wins: {values['shared_wins']}",
            f"  shared median clean relative RMSE: {values['shared_median_clean_relative_rmse']:.9e}",
            f"  independent median clean relative RMSE: {values['independent_median_clean_relative_rmse']:.9e}",
            "",
        ])
    lines.append("These are descriptive counts and medians, not formal significance claims.")
    (search_directory / "final_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _best_commands(path: Path, selected: dict[str, dict[str, object]], settings: dict[str, object]) -> None:
    """Write exact reproducible run.py commands for every frozen seed/layout."""
    lines = []
    for layout in ("slot", "single"):
        config = selected[layout]
        role = "primary" if config["role"] == "primary" else "control"
        lines.append(f"# {layout} layout ({role})")
        for seed in settings["final_seeds"]:
            lines.append(
                "python run.py "
                f"--layout {layout} --seed {seed} --plane-particles {int(settings['final_sources']) * 8} "
                f"--epsilon {config['epsilon']} --exponent-s {config['exponent_s']} "
                f"--gamma {config['gamma']} --forward-steps {math.ceil(float(config['tau']) / float(config['gamma']))} "
                f"--beta {config['beta']} --proximal-steps {config['proximal_steps']} "
                f"--trials {settings['final_trials']} --vertex-sources {settings['final_vertex_sources']} "
                "--output-root results"
            )
        lines.append("")
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def run_search(args: argparse.Namespace) -> Path:
    """Execute or resume all calibration, replay, and final-evaluation stages."""
    if not 1 <= args.workers <= 4:
        raise ValueError("--workers must be between 1 and 4")

    if args.resume is None:
        directory = _new_search_directory(args.output_root)
        settings = _settings(args.quick)
        search_config = {
            **settings,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "workers_requested": args.workers,
            "dimension": 4,
            "particles_per_source": 8,
            "interpolation_parents": 4,
            "dtype": "float64",
            "attraction_coefficient": 1.0,
            "forward_denominator": "N-1 within each empirical field",
            "passive_replay_denominator": "N within each empirical field",
            "passive_forward_frame": "source frame",
            "backward_replay_frame": "post-forward frame",
        }
        with (directory / "search_config.json").open("x", encoding="utf-8") as handle:
            json.dump(search_config, handle, indent=2, sort_keys=True)
        _write_header(directory / "forward_grid.csv", FORWARD_FIELDS)
        _write_header(directory / "replay_grid.csv", REPLAY_FIELDS)
        _write_header(directory / "final_runs.csv", FINAL_RUN_FIELDS)
    else:
        directory = args.resume.resolve()
        with (directory / "search_config.json").open(encoding="utf-8") as handle:
            search_config = json.load(handle)
        settings = {key: value for key, value in search_config.items() if key in _settings(False)}

    forward_path = directory / "forward_grid.csv"
    replay_path = directory / "replay_grid.csv"
    epsilons = settings["epsilons"]
    exponents = settings["exponents"]
    gammas = settings["gammas"]
    taus = settings["taus"]

    # Stage 1: potential screen. The full protocol fixes gamma=0.0005.
    stage_one_gamma = 0.0005
    potential_tasks = [{
        "stage": "potential", "layout": "slot", "source_count": settings["potential_sources"],
        "seed": seed, "epsilon": epsilon, "exponent_s": exponent_s,
        "gamma": stage_one_gamma, "taus": taus,
    } for epsilon in epsilons for exponent_s in exponents for seed in settings["screen_seeds"]]
    _run_forward_tasks(potential_tasks, forward_path, args.workers)
    aggregates = _aggregate_forward(_read_rows(forward_path), "potential")
    retained_pairs = _potential_pairs(aggregates, 4)

    # Stage 2: Euler-step screen over all gamma and tau values.
    euler_tasks = [{
        "stage": "euler", "layout": "slot", "source_count": settings["potential_sources"],
        "seed": seed, "epsilon": pair["epsilon"], "exponent_s": pair["exponent_s"],
        "gamma": gamma, "taus": taus,
    } for pair in retained_pairs for gamma in gammas for seed in settings["screen_seeds"]]
    _run_forward_tasks(euler_tasks, forward_path, args.workers)
    euler_configs = _complete_configs(_aggregate_forward(_read_rows(forward_path), "euler"), 4)

    # Stage 3: both layouts at C=128, then C=256 validation seeds.
    refinement_tasks = [{
        "stage": "density_refinement", "layout": layout,
        "source_count": settings["refinement_sources"], "seed": seed,
        "epsilon": config["epsilon"], "exponent_s": config["exponent_s"],
        "gamma": config["gamma"], "taus": [config["tau"]],
    } for config in euler_configs for layout in ("slot", "single") for seed in settings["screen_seeds"]]
    _run_forward_tasks(refinement_tasks, forward_path, args.workers)
    refinement = _aggregate_forward(_read_rows(forward_path), "density_refinement")
    finalists: list[dict[str, object]] = []
    for layout in ("slot", "single"):
        finalists.extend(_complete_configs([value for value in refinement if value["layout"] == layout], 2))
    for config in finalists:
        config["forward_id"] = _config_id(config)

    validation_tasks = [{
        "stage": "density_validation", "layout": config["layout"],
        "source_count": settings["validation_sources"], "seed": seed,
        "epsilon": config["epsilon"], "exponent_s": config["exponent_s"],
        "gamma": config["gamma"], "taus": [config["tau"]],
    } for config in finalists for seed in settings["validation_seeds"]]
    _run_forward_tasks(validation_tasks, forward_path, args.workers)

    # Stage 4: one-source replay screen, then four-source confirmation with fallback.
    replay_pairs = [(beta, steps) for beta in settings["betas"] for steps in settings["proximal_steps"]]
    screen_tasks = [{
        "stage": "screen", "forward_id": config["forward_id"], "layout": config["layout"],
        "source_count": settings["validation_sources"], "seed": 61,
        "epsilon": config["epsilon"], "exponent_s": config["exponent_s"],
        "gamma": config["gamma"], "tau": config["tau"], "vertex_sources": 1,
        "pairs": replay_pairs,
    } for config in finalists]
    _run_replay_tasks(screen_tasks, replay_path, args.workers, _replay_screen_worker)

    replay_rows = _read_rows(replay_path)
    ranked = {config["forward_id"]: _rank_replay_candidates(replay_rows, config["forward_id"]) for config in finalists}
    candidate_index = {config["forward_id"]: 0 for config in finalists}
    confirmed: dict[str, dict[str, object]] = {}
    while True:
        active = [
            config for config in finalists
            if config["forward_id"] not in confirmed
            and candidate_index[config["forward_id"]] < len(ranked[config["forward_id"]])
        ]
        if not active:
            break
        confirmation_tasks = []
        for config in active:
            candidate = ranked[config["forward_id"]][candidate_index[config["forward_id"]]]
            for seed in settings["confirmation_seeds"]:
                confirmation_tasks.append({
                    "stage": "confirmation", "forward_id": config["forward_id"],
                    "layout": config["layout"], "source_count": settings["validation_sources"],
                    "seed": seed, "epsilon": config["epsilon"], "exponent_s": config["exponent_s"],
                    "gamma": config["gamma"], "tau": config["tau"], "vertex_sources": 4,
                    "beta": candidate["beta"], "proximal_steps": candidate["proximal_steps"],
                })
        _run_replay_tasks(confirmation_tasks, replay_path, args.workers, _replay_confirmation_worker)
        replay_rows = _read_rows(replay_path)
        for config in active:
            forward_id = config["forward_id"]
            candidate = ranked[forward_id][candidate_index[forward_id]]
            rows = [
                row for row in replay_rows
                if row["stage"] == "confirmation" and row["forward_id"] == forward_id
                and float(row["beta"]) == float(candidate["beta"])
                and int(row["proximal_steps"]) == int(candidate["proximal_steps"])
                and int(row["seed"]) in settings["confirmation_seeds"]
            ]
            if len(rows) == len(settings["confirmation_seeds"]) and all(int(row["gate_pass"]) == 1 for row in rows):
                confirmed[forward_id] = {
                    **candidate,
                    "worst_confirmation_p95": max(float(row["particle_p95_relative_rmse"]) for row in rows),
                    "worst_validation_p95": max(
                        float(candidate["screen_p95"]),
                        max(float(row["particle_p95_relative_rmse"]) for row in rows),
                    ),
                    "validation_runtime": float(candidate["screen_runtime"]) + sum(
                        float(row["forward_seconds"]) + float(row["replay_seconds"]) for row in rows
                    ),
                }
            else:
                candidate_index[forward_id] += 1

    selected: dict[str, dict[str, object]] = {}
    for layout in ("slot", "single"):
        options = [config for config in finalists if config["layout"] == layout and config["forward_id"] in confirmed]
        if not options:
            raise RuntimeError(f"no {layout} finalist passed replay confirmation")
        chosen = min(options, key=lambda config: (
            confirmed[config["forward_id"]]["worst_validation_p95"],
            confirmed[config["forward_id"]]["validation_runtime"],
        ))
        selected[layout] = {**chosen, **confirmed[chosen["forward_id"]]}

    slot_p95 = float(selected["slot"]["worst_validation_p95"])
    single_p95 = float(selected["single"]["worst_validation_p95"])
    close = max(slot_p95, single_p95) <= 1.10 * max(min(slot_p95, single_p95), 1.0e-15)
    if close:
        primary = min(("slot", "single"), key=lambda layout: float(selected[layout]["validation_runtime"]))
    else:
        primary = min(("slot", "single"), key=lambda layout: float(selected[layout]["worst_validation_p95"]))
    for layout in ("slot", "single"):
        selected[layout]["role"] = "primary" if layout == primary else "control"

    _best_commands(directory / "best_commands.txt", selected, settings)
    _joint_vector_smoke(directory / "joint_smoke.txt")

    final_tasks = [{
        **selected[layout], "layout": layout, "seed": seed,
        "source_count": settings["final_sources"], "trials": settings["final_trials"],
        "vertex_sources": settings["final_vertex_sources"],
        "output_root": str((directory / "final_runs").resolve()),
    } for layout in ("slot", "single") for seed in settings["final_seeds"]]
    _run_final_tasks(final_tasks, directory / "final_runs.csv", args.workers)
    pooled_summary = _pool_final_results(directory)

    selection = {
        "potential_pairs": [{"epsilon": value["epsilon"], "exponent_s": value["exponent_s"]} for value in retained_pairs],
        "euler_configs": euler_configs,
        "forward_finalists": finalists,
        "selected_layouts": selected,
        "primary_layout": primary,
        "control_layout": "single" if primary == "slot" else "slot",
        "pooled_final_summary": pooled_summary,
        "targets_used_for_calibration": False,
    }
    with (directory / "selection.json").open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2, sort_keys=True)

    from plotting import plot_grid_summary
    plot_grid_summary(directory)
    return directory


def build_parser() -> argparse.ArgumentParser:
    """Define search, resume, and quick-path arguments."""
    parser = argparse.ArgumentParser(description="Run or resume staged EFS calibration and H4.5 evaluation.")
    parser.add_argument("--workers", type=int, default=4, help="Independent process workers; maximum four.")
    parser.add_argument("--output-root", default="results", help="Parent for a new timestamped search directory.")
    parser.add_argument("--resume", type=Path, default=None, help="Existing search directory whose completed CSV rows are skipped.")
    parser.add_argument("--quick", action="store_true", help="Run a tiny path check, not a scientific calibration.")
    return parser


def main() -> None:
    """Run the requested search and print its absolute output directory."""
    directory = run_search(build_parser().parse_args())
    print(f"search results: {directory.resolve()}")


if __name__ == "__main__":
    main()

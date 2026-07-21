"""Run and aggregate the Experiment 10 particle-count sweep.

Purpose:
    Execute every requested seed and particles-per-source condition through
    the independent Experiment 10 runner, stream prefixed progress, resume
    safely, validate controls, and aggregate readable metrics and figures.
Dependencies:
    Python standard library, NumPy, and Matplotlib.
Outputs:
    One immutable numbered sweep directory under ``--output-root``.
Exact command:
    conda run --no-capture-output -n anewomni python -u sweep.py --particles 1 2 3 4 5 6 7 8 9 10 --seeds 45 46 47 --source-count 512 --condition-workers 4 --workers 4 --log-every 25 --output-root results
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_NAMES = ("shared_d1", "shared_du", "per_particle_d1", "per_particle_du")
SUMMARY_FIELDS = (
    "lambda_fit_rmse",
    "initial_candidate_rmse",
    "reference_terminal_rmse",
    "cross_frame_rmse",
    "generation_terminal_rmse",
    "direct_set_rmse",
    "generated_set_rmse",
    "efs_direct_ratio",
    "terminal_gap",
    "max_replay_residual",
)
REGRESSION_EXPECTED = {
    "lambda_fit_rmse": 0.5464209478324924,
    "direct_set_rmse": 0.6631798704974623,
    "generated_set_rmse": 0.676687965298181,
    "efs_direct_ratio": 1.0190975407145688,
}
PRINT_LOCK = Lock()


def _console(message: str) -> None:
    """Print one complete line without interleaving concurrent conditions."""
    with PRINT_LOCK:
        print(message, flush=True)


def _progress_bar(finished: int, total: int, width: int = 24) -> str:
    """Return a dependency-free terminal progress bar."""
    filled = width if total == 0 else int(width * finished / total)
    return f"[{'#' * filled}{'.' * (width - filled)}] {finished}/{total}"


def _stream_condition(
    command: list[str],
    experiment_directory: Path,
    log_path: Path,
    condition_key: str,
) -> int:
    """Run one condition and tee every unbuffered child line to log and terminal."""
    with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
        process = subprocess.Popen(
            command,
            cwd=experiment_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if process.stdout is None:
            process.terminate()
            raise RuntimeError(f"condition {condition_key} has no output stream")
        for line in process.stdout:
            log_handle.write(line)
            _console(f"[{condition_key}] {line.rstrip()}")
        return process.wait()


def _write_json(path: Path, values: dict[str, object]) -> None:
    """Atomically replace one small progress or summary JSON file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(values, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _new_sweep_directory(output_root: Path, seeds: list[int]) -> Path:
    """Create one numbered sweep directory without reusing an old result."""
    output_root.mkdir(parents=True, exist_ok=True)
    indices = [
        int(path.name[:3])
        for path in output_root.iterdir()
        if path.is_dir() and path.name[:3].isdigit() and path.name[3:4] == "_"
    ]
    index = max(indices, default=0) + 1
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    seed_label = "-".join(str(seed) for seed in seeds)
    directory = output_root / f"{index:03d}_{stamp}_seed{seed_label}"
    directory.mkdir(exist_ok=False)
    return directory


def _condition_key(seed: int, particles: int) -> str:
    return f"seed{seed}_p{particles:02d}"


def _condition_order(seeds: list[int], particles: list[int]) -> list[tuple[int, int]]:
    """Run the Experiment 07 regression condition first, then the grid."""
    conditions = [(seed, particle) for seed in seeds for particle in particles]
    regression = (42, 10)
    if regression in conditions:
        conditions.remove(regression)
        conditions.insert(0, regression)
    return conditions


def _read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _numeric(row: dict[str, str], field: str) -> float:
    return float(row[field])


def _validate_p1(rows: list[dict[str, str]]) -> float:
    """Verify that P=1 shared and per-particle methods are identical."""
    by_method = {row["method"]: row for row in rows}
    maximum = 0.0
    for shared, control in (("shared_d1", "per_particle_d1"), ("shared_du", "per_particle_du")):
        for field in SUMMARY_FIELDS:
            difference = abs(_numeric(by_method[shared], field) - _numeric(by_method[control], field))
            maximum = max(maximum, difference)
    if maximum > 1.0e-10:
        raise RuntimeError(f"P=1 shared/per-particle metric mismatch is {maximum:.3e}")
    return maximum


def _validate_regression(rows: list[dict[str, str]]) -> dict[str, object]:
    """Check the paired shared-du method against the saved Experiment 07 run."""
    shared_du = next(row for row in rows if row["method"] == "shared_du")
    differences = {
        field: abs(_numeric(shared_du, field) - expected)
        for field, expected in REGRESSION_EXPECTED.items()
    }
    passed = all(
        np.isclose(_numeric(shared_du, field), expected, rtol=1.0e-6, atol=1.0e-8)
        for field, expected in REGRESSION_EXPECTED.items()
    )
    if not passed:
        raise RuntimeError(f"Experiment 07 regression mismatch: {differences}")
    return {"passed": True, "expected": REGRESSION_EXPECTED, "absolute_difference": differences}


def _collect_rows(sweep_directory: Path, progress: dict[str, object]) -> list[dict[str, str]]:
    """Collect the latest complete metrics for every finished condition."""
    collected: list[dict[str, str]] = []
    conditions = progress.get("conditions", {})
    if not isinstance(conditions, dict):
        return collected
    for key, record in conditions.items():
        if not isinstance(record, dict) or record.get("status") != "complete":
            continue
        run_directory = Path(str(record["run_directory"]))
        for row in _read_metrics(run_directory / "metrics.csv"):
            enriched = {
                "seed": str(record["seed"]),
                "particles_per_source": str(record["particles_per_source"]),
                "source_count": str(record["source_count"]),
                "condition": key,
                "run_directory": str(run_directory),
                **row,
            }
            collected.append(enriched)
    return collected


def _write_aggregate_csv(path: Path, rows: list[dict[str, str]]) -> None:
    prefix = ("seed", "particles_per_source", "source_count", "condition", "run_directory")
    remaining = [field for field in rows[0] if field not in prefix] if rows else []
    fields = [*prefix, *remaining]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _group_summary(rows: list[dict[str, str]]) -> dict[str, object]:
    grouped: dict[str, object] = {}
    for particles in sorted({int(row["particles_per_source"]) for row in rows}):
        particle_values: dict[str, object] = {}
        for method in METHOD_NAMES:
            selected = [
                row
                for row in rows
                if int(row["particles_per_source"]) == particles and row["method"] == method
            ]
            if not selected:
                continue
            particle_values[method] = {
                "seeds": sorted(int(row["seed"]) for row in selected),
                **{
                    f"median_{field}": float(np.median([_numeric(row, field) for row in selected]))
                    for field in SUMMARY_FIELDS
                },
            }
        grouped[str(particles)] = particle_values
    return grouped


def _paired_arrival_wins(rows: list[dict[str, str]]) -> dict[str, int]:
    """Count paired d1 versus du arrival wins for each lambda capacity."""
    by_condition: dict[tuple[int, int], dict[str, float]] = {}
    for row in rows:
        key = (int(row["seed"]), int(row["particles_per_source"]))
        by_condition.setdefault(key, {})[row["method"]] = _numeric(row, "generated_set_rmse")
    result = {"shared_d1": 0, "shared_du": 0, "per_particle_d1": 0, "per_particle_du": 0, "ties": 0}
    for values in by_condition.values():
        for d1_name, du_name in (("shared_d1", "shared_du"), ("per_particle_d1", "per_particle_du")):
            if d1_name not in values or du_name not in values:
                continue
            if values[d1_name] < values[du_name]:
                result[d1_name] += 1
            elif values[du_name] < values[d1_name]:
                result[du_name] += 1
            else:
                result["ties"] += 1
    return result


def _plot_summary(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    panels = (
        ("lambda_fit_rmse", "In-frame lambda fit RMSE", "lower is better"),
        ("cross_frame_rmse", "Cross-frame RMSE", "lower means better frame transfer"),
        ("generated_set_rmse", "Arrival target RMSE", "lower is better"),
        ("efs_direct_ratio", "EFS / direct error", "below 1 favors replay"),
    )
    colors = dict(zip(METHOD_NAMES, plt.get_cmap("tab10")(np.linspace(0.0, 0.75, 4))))
    markers = dict(zip(METHOD_NAMES, ("o", "s", "^", "D")))
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), constrained_layout=True)
    for axis, (field, title, ylabel) in zip(axes.flat, panels):
        for method in METHOD_NAMES:
            method_rows = [row for row in rows if row["method"] == method]
            particles = sorted({int(row["particles_per_source"]) for row in method_rows})
            medians = []
            for particle in particles:
                values = [
                    _numeric(row, field)
                    for row in method_rows
                    if int(row["particles_per_source"]) == particle
                ]
                medians.append(float(np.median(values)))
                axis.scatter(
                    [particle] * len(values),
                    values,
                    s=18,
                    color=colors[method],
                    alpha=0.3,
                )
            axis.plot(
                particles,
                medians,
                marker=markers[method],
                color=colors[method],
                linewidth=1.6,
                label=method,
            )
        if field == "efs_direct_ratio":
            axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_xlabel("particles per source P")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Experiment 10 particle-count sweep")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _aggregate(
    sweep_directory: Path,
    progress: dict[str, object],
    total_conditions: int,
) -> dict[str, object]:
    rows = _collect_rows(sweep_directory, progress)
    _write_aggregate_csv(sweep_directory / "particle_sweep.csv", rows)
    complete_conditions = len({row["condition"] for row in rows})
    condition_records = progress.get("conditions", {})
    invalid_conditions = []
    if isinstance(condition_records, dict):
        invalid_conditions = [
            key
            for key, value in condition_records.items()
            if isinstance(value, dict) and str(value.get("status", "")).startswith("invalid")
        ]
    regression = progress.get("experiment07_regression", {"status": "not_requested"})
    summary: dict[str, object] = {
        "status": "complete" if complete_conditions + len(invalid_conditions) == total_conditions else "running",
        "completed_conditions": complete_conditions,
        "invalid_conditions": invalid_conditions,
        "total_conditions": total_conditions,
        "metric_rows": len(rows),
        "paired_arrival_wins": _paired_arrival_wins(rows),
        "by_particles": _group_summary(rows),
        "experiment07_regression": regression,
        "maximum_p1_shared_control_difference": progress.get("maximum_p1_shared_control_difference", 0.0),
    }
    _write_json(sweep_directory / "summary.json", summary)
    regression_text = (
        str(bool(regression.get("passed", False)))
        if isinstance(regression, dict) and "passed" in regression
        else "not requested"
    )
    text = [
        "Experiment 10 lambda-frame and particle-count sweep",
        "",
        f"status: {summary['status']}",
        f"completed conditions: {complete_conditions}/{total_conditions}",
        f"invalid conditions: {invalid_conditions}",
        f"metric rows: {len(rows)}",
        f"Experiment 07 regression passed: {regression_text}",
        f"maximum P=1 shared/control difference: {float(summary['maximum_p1_shared_control_difference']):.3e}",
        f"paired arrival wins: {summary['paired_arrival_wins']}",
        "",
        "du methods optimize terminal fit; d1 methods test barycentric coordinate transport.",
        "Per-particle methods are capacity controls, not formal H4.5 generators.",
    ]
    (sweep_directory / "summary.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    _plot_summary(sweep_directory / "particle_sweep.png", rows)
    return summary


def _is_terminal(record: object) -> bool:
    """Return whether one condition no longer needs execution."""
    if not isinstance(record, dict):
        return False
    status = str(record.get("status", ""))
    return status == "complete" or status.startswith("invalid")


def _condition_batches(
    pending: list[tuple[int, int]], condition_workers: int
) -> list[list[tuple[int, int]]]:
    """Batch conditions while preserving the Experiment 07 regression gate."""
    remaining = list(pending)
    batches: list[list[tuple[int, int]]] = []
    if remaining and remaining[0] == (42, 10):
        batches.append([remaining.pop(0)])
    batches.extend(
        remaining[start : start + condition_workers]
        for start in range(0, len(remaining), condition_workers)
    )
    return batches


def self_check() -> dict[str, object]:
    """Check batching and visual progress without launching a condition."""
    conditions = [(42, 10), (42, 1), (42, 2), (42, 3), (42, 4)]
    batches = _condition_batches(conditions, condition_workers=2)
    assert batches == [[(42, 10)], [(42, 1), (42, 2)], [(42, 3), (42, 4)]]
    assert _progress_bar(1, 2, width=4) == "[##..] 1/2"
    return {"batches": batches, "progress": _progress_bar(1, 2, width=4)}


def run_sweep(args: argparse.Namespace) -> Path:
    """Execute or resume the grid with bounded condition-level concurrency."""
    experiment_directory = Path(__file__).resolve().parent
    if args.resume is not None:
        sweep_directory = args.resume.resolve()
        config = json.loads((sweep_directory / "config.json").read_text(encoding="utf-8"))
        seeds = [int(value) for value in config["seeds"]]
        particles = [int(value) for value in config["particles"]]
        source_count = int(config["source_count"])
        workers = int(args.workers if args.workers is not None else config.get("workers", 4))
        condition_workers = int(
            args.condition_workers
            if args.condition_workers is not None
            else config.get("condition_workers", 1)
        )
        log_every = int(args.log_every if args.log_every is not None else config.get("log_every", 100))
    else:
        seeds = list(dict.fromkeys(args.seeds))
        particles = list(dict.fromkeys(args.particles))
        source_count = args.source_count
        workers = int(args.workers if args.workers is not None else 4)
        condition_workers = int(args.condition_workers if args.condition_workers is not None else 1)
        log_every = int(args.log_every if args.log_every is not None else 100)
        if not seeds or not particles:
            raise ValueError("--seeds and --particles cannot be empty")
        if any(value < 1 or value > 16 for value in particles):
            raise ValueError("particle counts must be between 1 and 16")
        sweep_directory = _new_sweep_directory(args.output_root.resolve(), seeds)
        config = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "seeds": seeds,
            "particles": particles,
            "source_count": source_count,
            "dimension": 4,
            "parent_count": 4,
            "vertex_source_count": 4,
            "workers": workers,
            "condition_workers": condition_workers,
            "log_every": log_every,
            "method_names": list(METHOD_NAMES),
            "runner": str(experiment_directory / "run.py"),
            "python": sys.executable,
            "terminal_logging": "unbuffered prefixed child output plus per-attempt log",
            "condition_order": [
                _condition_key(seed, particle) for seed, particle in _condition_order(seeds, particles)
            ],
        }
        _write_json(sweep_directory / "config.json", config)

    if source_count < 4:
        raise ValueError("--source-count must be at least four")
    if workers < 1 or workers > 32:
        raise ValueError("--workers must be between 1 and 32")
    if condition_workers < 1 or condition_workers > 32:
        raise ValueError("--condition-workers must be between 1 and 32")
    if log_every < 1:
        raise ValueError("--log-every must be positive")

    # Resume may intentionally use different resource limits from the original attempt.
    config["workers"] = workers
    config["condition_workers"] = condition_workers
    config["log_every"] = log_every
    _write_json(sweep_directory / "config.json", config)

    progress_path = sweep_directory / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    else:
        progress = {"conditions": {}}
    condition_records = progress.setdefault("conditions", {})
    if not isinstance(condition_records, dict):
        raise ValueError("progress.json conditions must be an object")
    conditions = _condition_order(seeds, particles)
    pending = [
        (seed, particle)
        for seed, particle in conditions
        if not _is_terminal(condition_records.get(_condition_key(seed, particle)))
    ]
    finished = len(conditions) - len(pending)
    _console(
        f"{_progress_bar(finished, len(conditions))} SWEEP {sweep_directory} | "
        f"condition_workers={condition_workers} replay_workers={workers} log_every={log_every}"
    )

    for batch in _condition_batches(pending, condition_workers):
        specifications: list[dict[str, object]] = []
        for seed, particle in batch:
            key = _condition_key(seed, particle)
            existing = condition_records.get(key)
            condition_directory = sweep_directory / "conditions" / key
            run_root = condition_directory / "runs"
            condition_directory.mkdir(parents=True, exist_ok=True)
            run_root.mkdir(parents=True, exist_ok=True)
            attempts = int(existing.get("attempts", 0)) if isinstance(existing, dict) else 0
            attempt = attempts + 1
            log_path = condition_directory / f"attempt_{attempt:03d}.log"
            command = [
                sys.executable,
                "-u",
                str(experiment_directory / "run.py"),
                "--protocol",
                "two-pass",
                "--seed",
                str(seed),
                "--source-count",
                str(source_count),
                "--heldout-sources",
                "1",
                "--particles-per-source",
                str(particle),
                "--dimension",
                "4",
                "--parents",
                "4",
                "--vertex-sources",
                "4",
                "--workers",
                str(workers),
                "--log-every",
                str(log_every),
                "--output-root",
                str(run_root),
            ]
            specification: dict[str, object] = {
                "key": key,
                "seed": seed,
                "particle": particle,
                "run_root": run_root,
                "before": {path.resolve() for path in run_root.iterdir() if path.is_dir()},
                "log_path": log_path,
                "command": command,
            }
            specifications.append(specification)
            condition_records[key] = {
                "seed": seed,
                "particles_per_source": particle,
                "source_count": source_count,
                "status": "running",
                "attempts": attempt,
                "log": str(log_path),
                "command": command,
            }
            _console(
                f"{_progress_bar(finished, len(conditions))} START {key} "
                f"attempt={attempt} log={log_path}"
            )
        _write_json(progress_path, progress)

        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=len(specifications)) as executor:
            futures = {
                executor.submit(
                    _stream_condition,
                    specification["command"],
                    experiment_directory,
                    specification["log_path"],
                    specification["key"],
                ): specification
                for specification in specifications
            }
            for future in as_completed(futures):
                specification = futures[future]
                key = str(specification["key"])
                seed = int(specification["seed"])
                particle = int(specification["particle"])
                run_root = Path(specification["run_root"])
                log_path = Path(specification["log_path"])
                try:
                    return_code = int(future.result())
                except Exception as error:
                    return_code = -1
                    condition_records[key]["runner_error"] = repr(error)

                after = {path.resolve() for path in run_root.iterdir() if path.is_dir()}
                before = set(specification["before"])
                created = sorted(after - before, key=lambda path: path.name)
                if return_code != 0 or len(created) != 1:
                    condition_records[key].update({
                        "status": "failed",
                        "return_code": return_code,
                        "created_directories": [str(path) for path in created],
                    })
                    failures.append(key)
                    _console(f"{_progress_bar(finished, len(conditions))} FAIL {key}; inspect {log_path}")
                    _write_json(progress_path, progress)
                    _aggregate(sweep_directory, progress, len(conditions))
                    continue

                run_directory = created[0]
                try:
                    run_config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
                    status = str(run_config.get("status", "unknown"))
                    if status != "complete" and not status.startswith("invalid"):
                        raise RuntimeError(f"unexpected run status {status!r}")
                    condition_records[key].update({
                        "status": status,
                        "return_code": return_code,
                        "run_directory": str(run_directory),
                    })
                    if status == "complete":
                        rows = _read_metrics(run_directory / "metrics.csv")
                        if particle == 1:
                            p1_difference = _validate_p1(rows)
                            progress["maximum_p1_shared_control_difference"] = max(
                                float(progress.get("maximum_p1_shared_control_difference", 0.0)),
                                p1_difference,
                            )
                        if seed == 42 and particle == 10 and source_count == 512:
                            progress["experiment07_regression"] = _validate_regression(rows)
                except Exception as error:
                    condition_records[key].update({
                        "status": "failed_validation",
                        "return_code": return_code,
                        "run_directory": str(run_directory),
                        "validation_error": repr(error),
                    })
                    failures.append(key)
                    _console(
                        f"{_progress_bar(finished, len(conditions))} FAIL {key} validation; "
                        f"inspect {log_path}"
                    )
                    _write_json(progress_path, progress)
                    _aggregate(sweep_directory, progress, len(conditions))
                    continue

                finished += 1
                _write_json(progress_path, progress)
                _aggregate(sweep_directory, progress, len(conditions))
                _console(
                    f"{_progress_bar(finished, len(conditions))} DONE {key} "
                    f"status={condition_records[key]['status']} result={run_directory}"
                )

        if failures:
            raise RuntimeError(f"condition batch failed: {', '.join(failures)}")

    summary = _aggregate(sweep_directory, progress, len(conditions))
    config["status"] = summary["status"]
    config["completed_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(sweep_directory / "config.json", config)
    _console(f"{_progress_bar(len(conditions), len(conditions))} SWEEP COMPLETE")
    _console(f"SWEEP_DIRECTORY={sweep_directory}")
    return sweep_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or resume the Experiment 10 particle sweep.")
    parser.add_argument("--particles", nargs="+", type=int, default=list(range(1, 11)), help="Particle counts P.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44], help="Independent run seeds.")
    parser.add_argument("--source-count", type=int, default=512, help="Pass-2 memory sources per condition.")
    parser.add_argument(
        "--condition-workers",
        type=int,
        default=None,
        help="Simultaneous full conditions; primary CPU and RAM control (fresh-run default: 1).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Passive replay threads inside each condition (fresh-run default: 4).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=None,
        help="Forward/backward frames between live terminal updates (fresh-run default: 100).",
    )
    parser.add_argument("--output-root", type=Path, default=Path("results"), help="Parent for a new sweep directory.")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Existing sweep directory to resume; resource flags may be changed.",
    )
    return parser


def main() -> None:
    run_sweep(build_parser().parse_args())


if __name__ == "__main__":
    main()

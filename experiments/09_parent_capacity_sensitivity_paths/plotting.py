"""Regenerate Experiment 09 parent-capacity, sensitivity, and path figures.

Purpose:
    Plot the parent-count, local-sensitivity, and lambda-path measurements
    without rerunning EFS.
Dependencies:
    NumPy and Matplotlib.
Outputs:
    PNG figures inside the supplied result directory.
Exact command:
    python plotting.py results\\<run-directory>
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read one result table as plain string dictionaries."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _group(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    """Group rows by one CSV field while retaining file order."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return dict(grouped)


def _save(fig: plt.Figure, path: Path) -> Path:
    """Save and close one figure."""
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_parent_errors(directory: Path, rows: list[dict[str, str]]) -> Path:
    """Plot median and per-target errors against available parent count K."""
    k_values = sorted({int(row["k"]) for row in rows})
    metrics = (
        ("terminal_fit_rmse", "terminal fit"),
        ("replay_target_rmse", "backward arrival"),
        ("direct_same_lambda_rmse", "direct same-lambda"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for field, label in metrics:
        medians = [
            np.median([float(row[field]) for row in rows if int(row["k"]) == parent_count])
            for parent_count in k_values
        ]
        axes[0].plot(k_values, medians, marker="o", linewidth=2, label=label)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(k_values, labels=k_values)
    axes[0].set_xlabel("available parents K")
    axes[0].set_ylabel("standardized complete-source RMSE")
    axes[0].set_title("Median representation and replay error")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    # Each faint curve is one fixed seed and target. This prevents a median
    # from hiding targets that become worse as more parents are available.
    targets = _group(rows, "seed")
    target_groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for seed, seed_rows in targets.items():
        for target, target_rows in _group(seed_rows, "target_index").items():
            target_groups[(seed, target)] = target_rows
    for target_rows in target_groups.values():
        ordered = sorted(target_rows, key=lambda row: int(row["k"]))
        axes[1].plot(
            [int(row["k"]) for row in ordered],
            [float(row["replay_target_rmse"]) for row in ordered],
            color="tab:blue",
            alpha=0.18,
            linewidth=0.9,
        )
    median_replay = [
        np.median([float(row["replay_target_rmse"]) for row in rows if int(row["k"]) == value])
        for value in k_values
    ]
    axes[1].plot(k_values, median_replay, color="black", marker="o", linewidth=2.5, label="median")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(k_values, labels=k_values)
    axes[1].set_xlabel("available parents K")
    axes[1].set_ylabel("backward arrival RMSE")
    axes[1].set_title("Every target, not only the median")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    return _save(fig, directory / "error_vs_k.png")


def plot_parent_amplification(directory: Path, rows: list[dict[str, str]]) -> Path:
    """Plot replay error relative to terminal fit and direct interpolation."""
    k_values = sorted({int(row["k"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, field, title in (
        (axes[0], "replay_over_terminal", "Backward error / terminal fit error"),
        (axes[1], "replay_over_direct", "Backward error / direct same-lambda error"),
    ):
        medians = []
        p90 = []
        for parent_count in k_values:
            values = np.asarray([float(row[field]) for row in rows if int(row["k"]) == parent_count])
            medians.append(float(np.median(values)))
            p90.append(float(np.quantile(values, 0.90)))
            axis.scatter(
                np.full(values.size, parent_count), values, s=12, color="tab:blue", alpha=0.20
            )
        axis.plot(k_values, medians, marker="o", color="black", linewidth=2, label="median")
        axis.plot(k_values, p90, marker="s", color="tab:red", linewidth=1.5, label="p90")
        axis.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        axis.set_xscale("log", base=2)
        axis.set_xticks(k_values, labels=k_values)
        axis.set_yscale("log")
        axis.set_xlabel("available parents K")
        axis.set_ylabel("error ratio")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend()
    return _save(fig, directory / "amplification_vs_k.png")


def plot_effective_parents(directory: Path, rows: list[dict[str, str]]) -> Path:
    """Show how many parents the fitted simplex actually uses."""
    k_values = sorted({int(row["k"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    effective = []
    maximum_weight = []
    for parent_count in k_values:
        selected = [row for row in rows if int(row["k"]) == parent_count]
        effective.append(float(np.median([float(row["effective_parent_count"]) for row in selected])))
        maximum_weight.append(float(np.median([float(row["lambda_max"]) for row in selected])))
    axes[0].plot(k_values, effective, marker="o", linewidth=2)
    axes[0].plot(k_values, k_values, color="gray", linestyle="--", linewidth=1, label="all parents equal")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log", base=2)
    axes[0].set_xticks(k_values, labels=k_values)
    axes[0].set_xlabel("available parents K")
    axes[0].set_ylabel("median effective parent count")
    axes[0].set_title("Active barycentric capacity")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(k_values, maximum_weight, marker="o", color="tab:orange", linewidth=2)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(k_values, labels=k_values)
    axes[1].set_xlabel("available parents K")
    axes[1].set_ylabel("median largest lambda")
    axes[1].set_title("Sparse explanation or diffuse mixture")
    axes[1].grid(alpha=0.25)
    return _save(fig, directory / "effective_k.png")


def plot_sensitivity(directory: Path, rows: list[dict[str, str]]) -> Path:
    """Plot local inverse amplification and identity changes by perturbation."""
    factors = sorted({float(row["delta_factor"]) for row in rows})
    directions = sorted({row["direction"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for direction in directions:
        medians = []
        p95 = []
        for factor in factors:
            values = np.asarray([
                float(row["local_amplification"])
                for row in rows
                if row["direction"] == direction and float(row["delta_factor"]) == factor
            ])
            medians.append(float(np.median(values)))
            p95.append(float(np.quantile(values, 0.95)))
        axes[0].plot(factors, medians, marker="o", linewidth=1.5, label=f"{direction} median")
        axes[0].plot(factors, p95, linestyle=":", linewidth=1.0, alpha=0.8)
    axes[0].axhline(10.0, color="black", linestyle="--", linewidth=1, label="practical flag 10x")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("terminal displacement / nearest-neighbor spacing")
    axes[0].set_ylabel("local output amplification")
    axes[0].set_title("Solid: median, dotted: p95")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)

    for direction in directions:
        fractions = [
            np.mean([
                int(row["identity_changed"])
                for row in rows
                if row["direction"] == direction and float(row["delta_factor"]) == factor
            ])
            for factor in factors
        ]
        axes[1].plot(factors, fractions, marker="o", linewidth=1.5, label=direction)
    axes[1].axhline(0.05, color="black", linestyle="--", linewidth=1, label="practical flag 5%")
    axes[1].set_xscale("log")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xlabel("terminal displacement / nearest-neighbor spacing")
    axes[1].set_ylabel("nearest d1 identity-change fraction")
    axes[1].set_title("Branch or identity changes")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)
    return _save(fig, directory / "amplification_by_scale.png")


def _path_groups(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    """Group lambda rows by seed and path index."""
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["seed"], row["path_index"])].append(row)
    return dict(grouped)


def plot_source_paths(directory: Path, rows: list[dict[str, str]]) -> Path:
    """Plot whole-source displacement and step amplification along lambda."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for (seed, path_index), path_rows in _path_groups(rows).items():
        ordered = sorted(path_rows, key=lambda row: float(row["t"]))
        label = f"{ordered[0]['category']} s{seed} p{path_index}"
        t = [float(row["t"]) for row in ordered]
        axes[0].plot(t, [float(row["output_displacement_from_start"]) for row in ordered], label=label)
        axes[0].plot(
            t,
            [float(row["terminal_displacement_from_start"]) for row in ordered],
            linestyle=":",
            alpha=0.65,
        )
        axes[1].plot(t[1:], [float(row["step_amplification"]) for row in ordered[1:]], label=label)
    axes[0].set_xlabel("t in lambda(t)=[1-t,t]")
    axes[0].set_ylabel("whole-source displacement from t=0")
    axes[0].set_title("Solid output path, dotted terminal path")
    axes[0].grid(alpha=0.25)
    axes[1].axhline(10.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("output step / terminal step")
    axes[1].set_yscale("log")
    axes[1].set_title("Step amplification")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)
    return _save(fig, directory / "source_displacement.png")


def plot_particle_paths(directory: Path, rows: list[dict[str, str]]) -> Path:
    """Plot the worst particle and every individual particle jump along paths."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for (seed, path_index), path_rows in _path_groups(rows).items():
        ordered = sorted(path_rows, key=lambda row: float(row["t"]))
        label = f"{ordered[0]['category']} s{seed} p{path_index}"
        t = np.asarray([float(row["t"]) for row in ordered])  # shape: [T]
        maximum = np.asarray([float(row["maximum_particle_jump"]) for row in ordered])  # [T]
        axes[0].plot(t[1:], maximum[1:], label=label)

        # The semicolon field preserves all P particle jumps without creating
        # a variable number of CSV columns. Stack [T,P], then draw each P line.
        particle = np.asarray([
            [float(value) for value in row["particle_jumps"].split(";")]
            for row in ordered
        ])  # shape: [T,P]
        for particle_index in range(particle.shape[1]):
            axes[1].plot(t[1:], particle[1:, particle_index], alpha=0.35, linewidth=0.8)
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("maximum one-particle jump")
    axes[0].set_title("Largest part-level discontinuity")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("individual particle jump")
    axes[1].set_title("Every particle in every path")
    axes[1].grid(alpha=0.25)
    return _save(fig, directory / "particle_displacement.png")


def plot_all(directory: Path) -> list[Path]:
    """Regenerate every figure supported by the saved CSV files."""
    directory = directory.resolve()
    figures: list[Path] = []
    parent_rows = _read_csv(directory / "parent_sweep.csv")
    sensitivity_rows = _read_csv(directory / "du_sensitivity.csv")
    path_rows = _read_csv(directory / "lambda_path.csv")
    if parent_rows:
        figures.extend((
            plot_parent_errors(directory, parent_rows),
            plot_parent_amplification(directory, parent_rows),
            plot_effective_parents(directory, parent_rows),
        ))
    if sensitivity_rows:
        figures.append(plot_sensitivity(directory, sensitivity_rows))
    if path_rows:
        figures.extend((plot_source_paths(directory, path_rows), plot_particle_paths(directory, path_rows)))
    return figures


def main() -> None:
    """Parse one result directory and regenerate its figures."""
    parser = argparse.ArgumentParser(
        description="Regenerate Experiment 09 parent-capacity, sensitivity, and path figures."
    )
    parser.add_argument("run_directory", type=Path, help="Timestamped result directory created by run.py.")
    args = parser.parse_args()
    figures = plot_all(args.run_directory)
    print(f"generated {len(figures)} figure(s)")


if __name__ == "__main__":
    main()

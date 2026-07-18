"""Run the controlled local EFS replay stability experiment.

Purpose:
    Create known terminal targets by moving paired memory vertices toward one
    another, replay them through the cached EFS field, and measure when their
    original identities are recovered, swapped, amplified, or collapsed.
Dependencies:
    NumPy and Matplotlib.
Outputs:
    One timestamped directory containing CSV, JSON, NPZ, TXT, and PNG files.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Reuse the canonical one-plane EFS engine at the repository root.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(1, str(REPOSITORY_ROOT))

from efs import backward_replay, forward_history, mean_pairwise_distance, rms_radius, self_check as efs_self_check


METRIC_FIELDS = (
    "dimension",
    "beta",
    "pair",
    "alpha",
    "terminal_pair_distance",
    "seed_offset_distance",
    "seed_pair_separation_ratio",
    "target_rmse",
    "relative_plane_error",
    "relative_pair_error",
    "arrival_separation_ratio",
    "local_amplification",
    "identity_swapped",
    "collapsed",
    "max_replay_residual",
    "finite",
    "replay_seconds",
)


def generate_spiral_tube(
    particle_count: int,
    dimension: int,
    tube_width: float,
    data_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return an approximately constant-density thick spiral ``[N,D]``.

    The centerline is sampled uniformly by arc length. Each centerline point
    receives a uniform ``(D-1)``-ball cross-section, so the support is
    full-dimensional rather than a zero-thickness curve.
    """
    if particle_count < 32 or dimension < 2:
        raise ValueError("particle_count must be at least 32 and dimension at least 2")
    if tube_width <= 0.0 or data_scale <= 0.0:
        raise ValueError("tube_width and data_scale must be positive")

    # Build a dense deterministic centerline, then invert its cumulative arc
    # length. Uniform samples on cumulative length give similar density on the
    # inner and outer spiral turns, unlike uniform samples of the angle.
    angle_grid = np.linspace(0.35 * np.pi, 4.35 * np.pi, 20_000)  # shape: [H]
    radius_grid = 0.55 + 0.16 * angle_grid  # shape: [H]
    center_grid = np.column_stack(
        (radius_grid * np.cos(angle_grid), radius_grid * np.sin(angle_grid))
    )  # shape: [H,2]
    segment = np.linalg.norm(np.diff(center_grid, axis=0), axis=1)  # [H,2] -> [H-1]
    cumulative = np.concatenate(([0.0], np.cumsum(segment)))  # shape: [H]
    sampled_length = rng.uniform(0.0, cumulative[-1], size=particle_count)  # shape: [N]
    angle = np.interp(sampled_length, cumulative, angle_grid)  # shape: [N]
    radius = 0.55 + 0.16 * angle  # shape: [N]
    center = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))  # [N,2]

    # The derivative of r(t)[cos(t),sin(t)] gives the tangent. Rotating its
    # normalized first two coordinates by 90 degrees gives the planar normal.
    dx = 0.16 * np.cos(angle) - radius * np.sin(angle)  # shape: [N]
    dy = 0.16 * np.sin(angle) + radius * np.cos(angle)  # shape: [N]
    tangent_norm = np.sqrt(dx * dx + dy * dy)  # shape: [N]
    planar_normal = np.column_stack((-dy / tangent_norm, dx / tangent_norm))  # [N,2]

    # A uniform m-ball uses a random unit direction and radius U^(1/m). The
    # first cross-section coordinate follows the planar normal; remaining
    # coordinates occupy axes 3..D. This thickens the curve into D dimensions.
    cross_dimension = dimension - 1
    direction = rng.normal(size=(particle_count, cross_dimension))  # shape: [N,D-1]
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1.0e-15)  # [N,D-1]
    cross_radius = tube_width * rng.random((particle_count, 1)) ** (1.0 / cross_dimension)  # [N,1]
    cross_section = direction * cross_radius  # [N,D-1] * [N,1] -> [N,D-1]

    particles = np.zeros((particle_count, dimension), dtype=np.float64)  # shape: [N,D]
    particles[:, :2] = center + cross_section[:, :1] * planar_normal  # [N,2] + [N,1]*[N,2]
    if dimension > 2:
        particles[:, 2:] = cross_section[:, 1:]  # [N,D-2]

    # One global scale preserves relative geometry and constant-density intent.
    # It also places every tested dimension on the same RMS-radius scale.
    particles -= np.mean(particles, axis=0, keepdims=True)  # [N,D] - [1,D]
    particles *= data_scale / max(rms_radius(particles), 1.0e-15)  # shape: [N,D]
    if not np.all(np.isfinite(particles)):
        raise FloatingPointError("spiral-tube generator produced non-finite particles")
    return particles


def select_terminal_pairs(terminal: np.ndarray, pair_count: int) -> np.ndarray:
    """Select nearest-neighbor vertex pairs across spacing quantiles as ``[K,2]``."""
    particle_count = terminal.shape[0]
    if not 1 <= pair_count <= particle_count // 4:
        raise ValueError("pair_count must be positive and small relative to the plane")

    # ||x_i-x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i.x_j. The scalar [N,N]
    # matrix avoids allocating a larger [N,N,D] difference tensor.
    squared_norm = np.sum(terminal * terminal, axis=1)  # shape: [N]
    squared_distance = squared_norm[:, None] + squared_norm[None, :] - 2.0 * (terminal @ terminal.T)  # [N,N]
    np.maximum(squared_distance, 0.0, out=squared_distance)
    np.fill_diagonal(squared_distance, np.inf)
    nearest = np.argmin(squared_distance, axis=1)  # shape: [N]
    spacing = np.sqrt(squared_distance[np.arange(particle_count), nearest])  # shape: [N]

    # Even positions in sorted local spacing cover dense through sparse areas.
    # Vertices are not reused, keeping every controlled pair independent.
    order = np.argsort(spacing)  # shape: [N]
    preferred = order[np.linspace(0, particle_count - 1, pair_count * 6, dtype=np.int64)]
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for first in np.concatenate((preferred, order)):
        second = int(nearest[first])
        if int(first) in used or second in used:
            continue
        pairs.append((int(first), second))
        used.update((int(first), second))
        if len(pairs) == pair_count:
            break
    if len(pairs) != pair_count:
        raise RuntimeError("could not select the requested number of disjoint terminal pairs")
    return np.asarray(pairs, dtype=np.int64)  # shape: [K,2]


def create_terminal_targets(
    terminal: np.ndarray, pair_indices: np.ndarray, alphas: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Move each terminal pair toward its midpoint.

    Returns seeds ``[A,K,2,D]`` and terminal pair distances ``[K]``. At alpha
    zero the seeds are exact memory vertices. At alpha 0.49 the pair remains
    distinct but has only 2% of its original terminal separation.
    """
    terminal_pairs = terminal[pair_indices]  # [N,D] indexed by [K,2] -> [K,2,D]
    first = terminal_pairs[None, :, 0, :]  # shape: [1,K,D]
    second = terminal_pairs[None, :, 1, :]  # shape: [1,K,D]
    weight = alphas[:, None, None]  # [A] -> [A,1,1]
    seed_first = (1.0 - weight) * first + weight * second  # broadcasts to [A,K,D]
    seed_second = weight * first + (1.0 - weight) * second  # broadcasts to [A,K,D]
    seeds = np.stack((seed_first, seed_second), axis=2)  # two identities -> [A,K,2,D]
    distance = np.linalg.norm(terminal_pairs[:, 0] - terminal_pairs[:, 1], axis=1)  # shape: [K]
    return seeds, distance


def _replay_worker(payload: tuple[float, np.ndarray, np.ndarray, float, float, float, int]) -> dict[str, object]:
    """Replay one beta configuration in a worker process."""
    beta, flat_seeds, history, gamma, epsilon, exponent_s, proximal_steps = payload
    started = time.perf_counter()
    try:
        trajectory, residual, _ = backward_replay(
            flat_seeds[:, None, :],  # [G,D] -> passive one-particle sources [G,1,D]
            history,
            gamma,
            epsilon,
            exponent_s,
            beta,
            proximal_steps,
            method_names=["created_target"] * flat_seeds.shape[0],
            log_every=0,
        )
        return {
            "beta": beta,
            "trajectory": trajectory,
            "residual": residual,
            "seconds": time.perf_counter() - started,
            "error": "",
        }
    except (FloatingPointError, ValueError) as error:
        return {
            "beta": beta,
            "trajectory": None,
            "residual": None,
            "seconds": time.perf_counter() - started,
            "error": str(error),
        }


def evaluate_replay(
    dimension: int,
    beta: float,
    trajectory: np.ndarray,
    residual: np.ndarray,
    initial: np.ndarray,
    pair_indices: np.ndarray,
    seeds: np.ndarray,
    terminal_pair_distance: np.ndarray,
    alphas: np.ndarray,
    seconds: float,
) -> list[dict[str, object]]:
    """Return one metric row per controlled pair and displacement."""
    alpha_count, pair_count, _, dimension_check = seeds.shape
    if dimension_check != dimension:
        raise ValueError("seed dimension does not match the evaluated dimension")

    arrivals = trajectory[0, :, 0, :].reshape(alpha_count, pair_count, 2, dimension)  # [G,D] -> [A,K,2,D]
    targets = initial[pair_indices]  # [N,D] indexed by [K,2] -> [K,2,D]
    initial_pair_distance = np.linalg.norm(targets[:, 0] - targets[:, 1], axis=1)  # shape: [K]
    plane_scale = rms_radius(initial)

    # Maximum proximal residual for each passive seed. Reshape restores alpha,
    # pair, and side axes that were flattened only for batched replay.
    seed_residual = np.max(residual, axis=(0, 2)).reshape(alpha_count, pair_count, 2)  # [J,G,1] -> [A,K,2]
    baseline = arrivals[0]  # exact-vertex replay for this beta, shape: [K,2,D]
    rows: list[dict[str, object]] = []

    for alpha_index, alpha in enumerate(alphas):
        generated = arrivals[alpha_index]  # shape: [K,2,D]
        difference = generated - targets  # [K,2,D] - [K,2,D]
        target_rmse = np.sqrt(np.mean(np.sum(difference * difference, axis=2), axis=1))  # [K]
        arrival_distance = np.linalg.norm(generated[:, 0] - generated[:, 1], axis=1)  # [K]

        # Swapped cost answers whether the two created identities crossed. It
        # compares A->target B and B->target A against the declared assignment.
        swapped_difference = generated - targets[:, ::-1, :]  # shape: [K,2,D]
        swapped_rmse = np.sqrt(
            np.mean(np.sum(swapped_difference * swapped_difference, axis=2), axis=1)
        )  # shape: [K]

        # Local amplification compares how much the backward output moved from
        # exact-vertex replay against the controlled terminal displacement.
        output_shift = np.sqrt(
            np.mean(np.sum((generated - baseline) ** 2, axis=2), axis=1)
        )  # shape: [K]
        input_shift = float(alpha) * terminal_pair_distance  # shape: [K]
        amplification = np.zeros(pair_count, dtype=np.float64)  # shape: [K]
        if alpha > 0.0:
            amplification = output_shift / np.maximum(input_shift, 1.0e-15)

        for pair in range(pair_count):
            separation_ratio = arrival_distance[pair] / max(initial_pair_distance[pair], 1.0e-15)
            rows.append(
                {
                    "dimension": dimension,
                    "beta": beta,
                    "pair": pair,
                    "alpha": float(alpha),
                    "terminal_pair_distance": float(terminal_pair_distance[pair]),
                    "seed_offset_distance": float(input_shift[pair]),
                    "seed_pair_separation_ratio": float(abs(1.0 - 2.0 * alpha)),
                    "target_rmse": float(target_rmse[pair]),
                    "relative_plane_error": float(target_rmse[pair] / max(plane_scale, 1.0e-15)),
                    "relative_pair_error": float(target_rmse[pair] / max(initial_pair_distance[pair], 1.0e-15)),
                    "arrival_separation_ratio": float(separation_ratio),
                    "local_amplification": float(amplification[pair]),
                    "identity_swapped": bool(swapped_rmse[pair] < target_rmse[pair]),
                    "collapsed": bool(separation_ratio < 0.25),
                    "max_replay_residual": float(np.max(seed_residual[alpha_index, pair])),
                    "finite": bool(np.all(np.isfinite(generated[pair]))),
                    "replay_seconds": float(seconds),
                }
            )
    return rows


def plane_statistics(particles: np.ndarray) -> dict[str, float]:
    """Return small descriptive geometry diagnostics for one ``[N,D]`` plane."""
    centered = particles - np.mean(particles, axis=0, keepdims=True)  # [N,D] - [1,D]
    radius = np.linalg.norm(centered, axis=1)  # shape: [N]
    dimension = particles.shape[1]
    fitted_ball_radius = math.sqrt((dimension + 2.0) * np.mean(radius * radius) / dimension)
    sorted_radius = np.sort(radius)  # shape: [N]
    empirical = np.arange(1, radius.size + 1, dtype=np.float64) / radius.size  # shape: [N]
    reference = np.clip((sorted_radius / max(fitted_ball_radius, 1.0e-15)) ** dimension, 0.0, 1.0)
    covariance = centered.T @ centered / centered.shape[0]  # [D,N] @ [N,D] -> [D,D]
    eigenvalues = np.linalg.eigvalsh(covariance)  # shape: [D]
    return {
        "rms_radius": rms_radius(particles),
        "mean_pair_distance": mean_pairwise_distance(particles),
        "radial_cdf_gap": float(np.max(np.abs(empirical - reference))),
        "covariance_ratio": float(max(eigenvalues[0], 0.0) / max(eigenvalues[-1], 1.0e-15)),
    }


def beta_summary(rows: list[dict[str, object]], dimension: int, beta: float) -> dict[str, object]:
    """Summarize one dimension and beta without formal statistical claims."""
    selected = [row for row in rows if row["dimension"] == dimension and row["beta"] == beta]
    displaced = [row for row in selected if float(row["alpha"]) > 0.0]
    vertex = [row for row in selected if float(row["alpha"]) == 0.0]
    safe_alpha = 0.0
    for alpha in sorted({float(row["alpha"]) for row in selected}):
        at_alpha = [row for row in selected if float(row["alpha"]) == alpha]
        median_pair_error = float(np.median([float(row["relative_pair_error"]) for row in at_alpha]))
        if (
            median_pair_error <= 0.10
            and not any(bool(row["identity_swapped"]) for row in at_alpha)
            and not any(bool(row["collapsed"]) for row in at_alpha)
        ):
            safe_alpha = alpha
    return {
        "dimension": dimension,
        "beta": beta,
        "vertex_median_relative_plane_error": float(
            np.median([float(row["relative_plane_error"]) for row in vertex])
        ),
        "displaced_median_relative_plane_error": float(
            np.median([float(row["relative_plane_error"]) for row in displaced])
        ),
        "displaced_median_relative_pair_error": float(
            np.median([float(row["relative_pair_error"]) for row in displaced])
        ),
        "maximum_residual": max(float(row["max_replay_residual"]) for row in selected),
        "identity_swap_count": sum(bool(row["identity_swapped"]) for row in displaced),
        "collapse_count": sum(bool(row["collapsed"]) for row in displaced),
        "safe_alpha": safe_alpha,
        "replay_seconds": float(selected[0]["replay_seconds"]),
    }


def choose_beta(summaries: list[dict[str, object]], dimension: int) -> float:
    """Choose the smallest converged beta within 1% of the best error."""
    selected = [row for row in summaries if row["dimension"] == dimension]
    converged = [
        row
        for row in selected
        if float(row["maximum_residual"]) <= 1.0e-5
        and float(row["vertex_median_relative_plane_error"]) <= 0.01
    ]
    candidates = converged or selected
    best_error = min(float(row["displaced_median_relative_plane_error"]) for row in candidates)
    near_best = [
        row
        for row in candidates
        if float(row["displaced_median_relative_plane_error"]) <= 1.01 * best_error + 1.0e-15
    ]
    return min(float(row["beta"]) for row in near_best)


def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    """Write metric rows to CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def plot_stability(path: Path, rows: list[dict[str, object]], dimensions: list[int], betas: list[float]) -> None:
    """Plot error, identity separation, and amplification against displacement."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(dimensions), 3, figsize=(14.0, 4.2 * len(dimensions)), squeeze=False)
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(betas)))
    for row_index, dimension in enumerate(dimensions):
        for beta, color in zip(betas, colors):
            selected = [
                row for row in rows if row["dimension"] == dimension and row["beta"] == beta
            ]
            alphas = sorted({float(row["alpha"]) for row in selected})
            error = [
                np.median([float(row["relative_pair_error"]) for row in selected if row["alpha"] == alpha])
                for alpha in alphas
            ]
            separation = [
                np.median([float(row["arrival_separation_ratio"]) for row in selected if row["alpha"] == alpha])
                for alpha in alphas
            ]
            amplification = [
                np.median([float(row["local_amplification"]) for row in selected if row["alpha"] == alpha])
                for alpha in alphas
            ]
            label = rf"$\beta={beta:g}$"
            axes[row_index, 0].plot(alphas, error, marker="o", color=color, label=label)
            axes[row_index, 1].plot(alphas, separation, marker="o", color=color, label=label)
            axes[row_index, 2].plot(alphas, amplification, marker="o", color=color, label=label)

        axes[row_index, 0].axhline(0.10, color="black", linestyle="--", linewidth=1.0, label="10% pair error")
        axes[row_index, 1].axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="target separation")
        axes[row_index, 1].axhline(0.25, color="red", linestyle=":", linewidth=1.0, label="collapse threshold")
        axes[row_index, 2].axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="unit amplification")
        axes[row_index, 0].set_yscale("log")
        axes[row_index, 0].set_ylabel(f"D={dimension}\nrelative error")
        axes[row_index, 1].set_ylabel("arrival separation / target separation")
        axes[row_index, 2].set_ylabel("local backward amplification")
        for axis in axes[row_index]:
            axis.set_xlabel(r"terminal displacement $\alpha$ of pair spacing")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=7)
    figure.suptitle("Created-target local EFS replay stability")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_beta_selection(path: Path, summaries: list[dict[str, object]], dimensions: list[int], betas: list[float]) -> None:
    """Plot which beta minimizes controlled-target replay error."""
    import matplotlib.pyplot as plt

    error = np.full((len(dimensions), len(betas)), np.nan, dtype=np.float64)  # shape: [D_count,B]
    for row in summaries:
        i = dimensions.index(int(row["dimension"]))
        j = betas.index(float(row["beta"]))
        error[i, j] = float(row["displaced_median_relative_plane_error"])
    figure, axis = plt.subplots(figsize=(8.0, 2.3 + 0.8 * len(dimensions)))
    image = axis.imshow(error, aspect="auto", cmap="magma_r")
    axis.set_xticks(np.arange(len(betas)), [f"{beta:g}" for beta in betas])
    axis.set_yticks(np.arange(len(dimensions)), [f"D={dimension}" for dimension in dimensions])
    axis.set_xlabel(r"backward optimizer step $\beta$")
    axis.set_title("Median created-target error; star = smallest converged beta within 1% of best")
    for i in range(len(dimensions)):
        selected_beta = choose_beta(summaries, dimensions[i])
        best = betas.index(selected_beta)
        for j in range(len(betas)):
            marker = " *" if j == best else ""
            axis.text(j, i, f"{error[i, j]:.3g}{marker}", ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis, label="error / initial plane RMS radius")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_created_flow(
    path: Path,
    initial: np.ndarray,
    history: np.ndarray,
    pair_indices: np.ndarray,
    seeds: np.ndarray,
    alphas: np.ndarray,
    best_trajectory: np.ndarray,
) -> None:
    """Show the raw 2D created targets from terminal displacement to arrival."""
    import matplotlib.pyplot as plt

    pair = pair_indices.shape[0] // 2
    alpha_indices = np.unique(np.linspace(0, len(alphas) - 1, min(4, len(alphas)), dtype=np.int64))
    group_count = seeds.shape[0] * seeds.shape[1] * 2
    trajectory = best_trajectory[:, :, 0, :].reshape(
        history.shape[0], seeds.shape[0], seeds.shape[1], 2, 2
    )  # [J+1,G,2] -> [J+1,A,K,2,D=2]
    if best_trajectory.shape[1] != group_count:
        raise ValueError("trajectory group count does not match created seeds")

    figure, axes = plt.subplots(1, 4, figsize=(18.0, 4.4), constrained_layout=True)
    target = initial[pair_indices[pair]]  # shape: [2,2]
    terminal_pair = history[-1, pair_indices[pair]]  # shape: [2,2]
    colors = plt.get_cmap("plasma")(np.linspace(0.08, 0.92, len(alphas)))

    axes[0].scatter(initial[:, 0], initial[:, 1], s=5, color="#999999", alpha=0.25)
    axes[0].scatter(target[:, 0], target[:, 1], marker="X", s=90, color=("#377EB8", "#E41A1C"), edgecolors="black")
    axes[0].set_title("1. Known original destinations")

    axes[1].scatter(history[-1, :, 0], history[-1, :, 1], s=5, color="#999999", alpha=0.25)
    axes[1].plot(terminal_pair[:, 0], terminal_pair[:, 1], color="black", linewidth=1.0)
    for alpha_index in alpha_indices:
        created = seeds[alpha_index, pair]  # shape: [2,2]
        axes[1].scatter(created[:, 0], created[:, 1], s=50, facecolors="none", edgecolors=colors[alpha_index])
    axes[1].scatter(terminal_pair[:, 0], terminal_pair[:, 1], marker="X", s=70, color="black")
    axes[1].set_title("2. Created terminal seeds")

    stride = max(1, (history.shape[0] - 1) // 300)
    for alpha_index in alpha_indices:
        for side in range(2):
            path_values = trajectory[::-1, alpha_index, pair, side][::stride]  # terminal -> initial, [F,2]
            axes[2].plot(path_values[:, 0], path_values[:, 1], color=colors[alpha_index], linewidth=1.1)
    axes[2].set_title("3. Backward paths")

    axes[3].scatter(initial[:, 0], initial[:, 1], s=5, color="#BBBBBB", alpha=0.18)
    axes[3].scatter(target[:, 0], target[:, 1], marker="X", s=90, color=("#377EB8", "#E41A1C"), edgecolors="black", label="targets")
    for alpha_index in alpha_indices:
        arrival = trajectory[0, alpha_index, pair]  # shape: [2,2]
        axes[3].scatter(arrival[:, 0], arrival[:, 1], marker="o", s=45, facecolors="none", edgecolors=colors[alpha_index], label=rf"$\alpha={alphas[alpha_index]:g}$")
    local_points = np.concatenate(
        (target, trajectory[0, alpha_indices, pair].reshape(-1, 2)), axis=0
    )  # target [2,2] plus selected arrivals [2*A,2]
    local_span = np.maximum(np.ptp(local_points, axis=0), 0.02)  # shape: [2]
    local_center = np.mean(local_points, axis=0)  # shape: [2]
    axes[3].set_xlim(local_center[0] - 0.75 * local_span[0], local_center[0] + 0.75 * local_span[0])
    axes[3].set_ylim(local_center[1] - 0.75 * local_span[1], local_center[1] + 0.75 * local_span[1])
    axes[3].set_title("4. Arrivals versus targets")
    axes[3].legend(fontsize=7)

    for axis in axes:
        axis.set_xlabel("coordinate 1")
        axis.set_ylabel("coordinate 2")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
    figure.suptitle("Controlled pair approach in the raw 2D plane")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def self_check() -> dict[str, float]:
    """Run one small algebra check for data and target construction."""
    rng = np.random.default_rng(7)
    initial = generate_spiral_tube(64, 4, 0.12, 1.0, rng)
    assert initial.shape == (64, 4) and np.all(np.isfinite(initial))
    pairs = np.asarray(((0, 1), (2, 3)), dtype=np.int64)
    alpha = np.asarray((0.0, 0.49), dtype=np.float64)
    seeds, pair_distance = create_terminal_targets(initial, pairs, alpha)
    assert np.allclose(seeds[0], initial[pairs])
    compressed = np.linalg.norm(seeds[1, :, 0] - seeds[1, :, 1], axis=1)
    assert np.allclose(compressed / pair_distance, 0.02)
    return {"created_pair_max_error": float(np.max(np.abs(compressed / pair_distance - 0.02)))}


def new_run_directory(output_root: Path, seed: int) -> Path:
    """Create one timestamped output directory without overwriting."""
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = output_root / f"local_replay_{stamp}_seed{seed}"
    directory.mkdir()
    return directory


def run(args: argparse.Namespace) -> Path:
    """Run all configured dimensions and save the controlled experiment."""
    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    output = new_run_directory(args.output_root, args.seed)
    algebra = {"efs": efs_self_check(), "created_targets": self_check()}
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    forward_config: dict[str, object] = {}

    for dimension in args.dimensions:
        exponent_s = max(1.0, float(dimension - 2))
        print(f"\n=== D={dimension}: create homogeneous spiral tube ===")
        initial = generate_spiral_tube(
            args.particles,
            dimension,
            args.tube_width,
            args.data_scale,
            np.random.default_rng(rng.integers(0, 2**63 - 1)),
        )  # shape: [N,D]
        initial_stats = plane_statistics(initial)
        forward_started = time.perf_counter()
        history, forward_logs = forward_history(
            initial,
            args.gamma,
            args.epsilon,
            exponent_s,
            args.forward_steps,
            log_every=args.log_every,
        )  # history shape: [J+1,N,D]
        forward_seconds = time.perf_counter() - forward_started
        if history.shape[0] != args.forward_steps + 1 or not bool(forward_logs["finite"]):
            raise FloatingPointError(f"D={dimension} forward EFS did not complete finitely")
        terminal_stats = plane_statistics(history[-1])

        pair_indices = select_terminal_pairs(history[-1], args.pairs)  # shape: [K,2]
        seeds, pair_distance = create_terminal_targets(history[-1], pair_indices, args.alphas)  # [A,K,2,D], [K]
        flat_seeds = seeds.reshape(-1, dimension)  # [A,K,2,D] -> [G=A*K*2,D]
        payloads = [
            (beta, flat_seeds, history, args.gamma, args.epsilon, exponent_s, args.proximal_steps)
            for beta in args.betas
        ]

        print(f"=== D={dimension}: replay {flat_seeds.shape[0]} created seeds for {len(args.betas)} beta values ===")
        dimension_results: dict[float, dict[str, object]] = {}
        if args.workers == 1:
            for payload in payloads:
                result = _replay_worker(payload)
                dimension_results[float(result["beta"])] = result
        else:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(payloads))) as executor:
                futures = {executor.submit(_replay_worker, payload): payload[0] for payload in payloads}
                for future in as_completed(futures):
                    result = future.result()
                    dimension_results[float(result["beta"])] = result
                    print(
                        f"D={dimension} beta={float(result['beta']):g} finished in "
                        f"{float(result['seconds']):.2f}s error={result['error'] or 'none'}"
                    )

        for beta in args.betas:
            result = dimension_results[beta]
            if result["error"]:
                raise FloatingPointError(f"D={dimension} beta={beta:g}: {result['error']}")
            metric_rows = evaluate_replay(
                dimension,
                beta,
                np.asarray(result["trajectory"]),
                np.asarray(result["residual"]),
                initial,
                pair_indices,
                seeds,
                pair_distance,
                args.alphas,
                float(result["seconds"]),
            )
            rows.extend(metric_rows)
            summaries.append(beta_summary(metric_rows, dimension, beta))

        best_beta = choose_beta(summaries, dimension)
        best_result = dimension_results[best_beta]
        alpha_count, pair_count = len(args.alphas), pair_indices.shape[0]
        arrivals = np.stack(
            [
                np.asarray(dimension_results[beta]["trajectory"])[0, :, 0, :].reshape(
                    alpha_count, pair_count, 2, dimension
                )
                for beta in args.betas
            ],
            axis=0,
        )  # shape: [B,A,K,2,D]
        maximum_residual = np.stack(
            [
                np.max(np.asarray(dimension_results[beta]["residual"]), axis=(0, 2)).reshape(
                    alpha_count, pair_count, 2
                )
                for beta in args.betas
            ],
            axis=0,
        )  # shape: [B,A,K,2]

        prefix = f"d{dimension}_"
        arrays.update(
            {
                prefix + "initial": initial,
                prefix + "history": history,
                prefix + "forward_log_step": np.asarray(forward_logs["log_step"]),
                prefix + "forward_log_mean_pair_distance": np.asarray(forward_logs["mean_pair_distance"]),
                prefix + "forward_log_rms_radius": np.asarray(forward_logs["rms_radius"]),
                prefix + "pair_indices": pair_indices,
                prefix + "alphas": args.alphas,
                prefix + "seeds": seeds,
                prefix + "terminal_pair_distance": pair_distance,
                prefix + "betas": np.asarray(args.betas),
                prefix + "arrivals": arrivals,
                prefix + "maximum_residual": maximum_residual,
                prefix + "best_beta": np.asarray(best_beta),
                prefix + "best_trajectory": np.asarray(best_result["trajectory"]),
                prefix + "best_residual": np.asarray(best_result["residual"]),
            }
        )
        forward_config[str(dimension)] = {
            "exponent_s": exponent_s,
            "forward_seconds": forward_seconds,
            "initial": initial_stats,
            "terminal": terminal_stats,
            "radius_ratio": terminal_stats["rms_radius"] / initial_stats["rms_radius"],
            "best_beta": best_beta,
        }

    write_metrics(output / "metrics.csv", rows)
    np.savez_compressed(output / "samples.npz", **arrays)
    plot_stability(output / "stability_summary.png", rows, args.dimensions, args.betas)
    plot_beta_selection(output / "beta_selection.png", summaries, args.dimensions, args.betas)
    if 2 in args.dimensions:
        prefix = "d2_"
        plot_created_flow(
            output / "created_target_flow.png",
            arrays[prefix + "initial"],
            arrays[prefix + "history"],
            arrays[prefix + "pair_indices"],
            arrays[prefix + "seeds"],
            arrays[prefix + "alphas"],
            arrays[prefix + "best_trajectory"],
        )

    total_seconds = time.perf_counter() - started
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "protocol": "created_terminal_pair_displacement",
        "distribution": "constant-density thick spiral tube",
        "particles": args.particles,
        "dimensions": args.dimensions,
        "pairs": args.pairs,
        "alphas": args.alphas.tolist(),
        "betas": args.betas,
        "gamma": args.gamma,
        "epsilon": args.epsilon,
        "forward_steps": args.forward_steps,
        "proximal_steps": args.proximal_steps,
        "tube_width": args.tube_width,
        "data_scale": args.data_scale,
        "workers": args.workers,
        "algebra_self_check": algebra,
        "forward": forward_config,
        "total_seconds": total_seconds,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    lines = [
        "Created-target local EFS replay stability",
        "",
        "Targets are exact memory vertices with known original positions.",
        "Terminal seeds are created by moving nearest-neighbor pairs toward each other.",
        "",
    ]
    for dimension in args.dimensions:
        lines.append(f"D={dimension}")
        for summary in [row for row in summaries if row["dimension"] == dimension]:
            marker = " selected" if float(summary["beta"]) == float(forward_config[str(dimension)]["best_beta"]) else ""
            lines.append(
                f"  beta={float(summary['beta']):g}{marker}: "
                f"vertex_error={float(summary['vertex_median_relative_plane_error']):.6e}, "
                f"displaced_error={float(summary['displaced_median_relative_plane_error']):.6e}, "
                f"safe_alpha={float(summary['safe_alpha']):.2f}, "
                f"swaps={int(summary['identity_swap_count'])}, "
                f"collapses={int(summary['collapse_count'])}, "
                f"max_residual={float(summary['maximum_residual']):.3e}"
            )
        lines.append("")
    lines.extend(
        (
            "Interpretation:",
            "alpha is the fraction of one terminal neighbor spacing moved toward the other vertex.",
            "safe_alpha is the largest tested alpha with median pair-relative error <= 10%, no swaps, and no collapses.",
            "beta changes the proximal solve. Similar converged results across beta mean the field, not beta, determines arrival.",
            f"total runtime: {total_seconds:.2f} s",
        )
    )
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {output}")
    return output


def build_parser() -> argparse.ArgumentParser:
    """Define the controlled experiment CLI."""
    parser = argparse.ArgumentParser(description="Measure local EFS replay stability with created terminal targets.")
    parser.add_argument("--output-root", type=Path, default=Path("results"), help="Parent directory for timestamped results.")
    parser.add_argument("--seed", type=int, default=45101, help="Deterministic NumPy seed.")
    parser.add_argument("--particles", type=int, default=768, help="Particles in each independent dimension run.")
    parser.add_argument("--dimensions", type=int, nargs="+", default=[2, 4], help="Dimensions evaluated independently.")
    parser.add_argument("--pairs", type=int, default=6, help="Disjoint terminal nearest-neighbor pairs per dimension.")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.10, 0.25, 0.40, 0.49], help="Fractions of pair spacing moved toward the pair midpoint.")
    parser.add_argument("--betas", type=float, nargs="+", default=[0.025, 0.05, 0.10, 0.20, 0.40], help="Backward proximal learning rates.")
    parser.add_argument("--gamma", type=float, default=0.002, help="Forward EFS Euler step.")
    parser.add_argument("--epsilon", type=float, default=0.03, help="Potential regularization epsilon.")
    parser.add_argument("--forward-steps", type=int, default=2000, help="Cached forward EFS frames.")
    parser.add_argument("--proximal-steps", type=int, default=50, help="Inner fixed-point updates per backward frame.")
    parser.add_argument("--tube-width", type=float, default=0.12, help="Spiral cross-section radius before global scaling.")
    parser.add_argument("--data-scale", type=float, default=1.0, help="Initial RMS radius after scaling.")
    parser.add_argument("--workers", type=int, default=5, help="Parallel beta replays; one forward run remains single-process.")
    parser.add_argument("--log-every", type=int, default=100, help="Print forward diagnostics every this many frames.")
    return parser


def main() -> None:
    """Parse, validate, and run."""
    args = build_parser().parse_args()
    if args.particles < 32 or args.pairs < 1 or args.forward_steps < 1 or args.proximal_steps < 1:
        raise ValueError("particles, pairs, forward-steps, and proximal-steps must be positive")
    if any(dimension < 2 for dimension in args.dimensions):
        raise ValueError("all dimensions must be at least two")
    if any(not 0.0 <= alpha < 0.5 for alpha in args.alphas) or 0.0 not in args.alphas:
        raise ValueError("alphas must include zero and remain in [0,0.5)")
    if any(beta <= 0.0 for beta in args.betas) or args.workers < 1:
        raise ValueError("betas and workers must be positive")
    args.dimensions = list(dict.fromkeys(args.dimensions))
    args.betas = list(dict.fromkeys(args.betas))
    args.alphas = np.asarray(sorted(set(args.alphas)), dtype=np.float64)
    run(args)


if __name__ == "__main__":
    main()

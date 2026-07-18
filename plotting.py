"""Regenerate all one-plane H4.5 figures from a saved run.

Purpose:
    Plot forward geometry, vertex replay, held-out reconstruction, shared
    lambda values, raw-2D or PCA flow, and backward movement without rerunning EFS.
Dependencies:
    NumPy and Matplotlib.
Outputs:
    PNG files inside the supplied run directory.
Exact command:
    python plotting.py experiments\\07_two_pass_reconstruction\\results\\<run-directory>

The selected local run must still contain its generated ``samples.npz``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_COLORS = ("#377EB8", "#E41A1C")


def _fit_pca(*arrays: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one two-axis NumPy-SVD basis to row arrays ``[A,D]``."""
    values = np.concatenate(arrays, axis=0)  # multiple [A,D] -> [sum(A),D]
    mean = np.mean(values, axis=0, keepdims=True)  # shape: [1,D]
    centered = values - mean  # shape: [sum(A),D]
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    components = right[:2]  # shape: [2,D]
    ratio = singular[:2] ** 2 / max(float(np.sum(singular**2)), 1.0e-15)
    return mean, components, ratio


def _radial_curve(particles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized sorted radii and empirical/reference CDF arrays."""
    centered = particles - np.mean(particles, axis=0, keepdims=True)  # shape: [N,D]
    radius = np.linalg.norm(centered, axis=1)  # shape: [N]
    dimension = particles.shape[1]
    fitted = np.sqrt((dimension + 2.0) * np.mean(radius * radius) / dimension)
    normalized = np.sort(radius / max(float(fitted), 1.0e-15))  # shape: [N]
    empirical = np.arange(1, radius.size + 1) / radius.size  # shape: [N]
    reference = np.clip(normalized**dimension, 0.0, 1.0)  # shape: [N]
    return normalized, empirical, reference


def _radial_zones(particles: np.ndarray) -> np.ndarray:
    """Return ten equal-probability D-ball zone fractions ``[10]``."""
    centered = particles - np.mean(particles, axis=0, keepdims=True)  # [N,D]
    radius = np.linalg.norm(centered, axis=1)  # [N]
    dimension = particles.shape[1]
    fitted = np.sqrt((dimension + 2.0) * np.mean(radius * radius) / dimension)
    zone = np.floor(10.0 * (radius / max(float(fitted), 1.0e-15)) ** dimension).astype(int)  # [N]
    return np.asarray([np.mean(zone == index) for index in range(10)])  # shape: [10]


def _sample_angular_dots(particles: np.ndarray, seed: int = 0, count: int = 50000) -> np.ndarray:
    """Sample pairwise direction dot products without building ``[N,N]``."""
    centered = particles - np.mean(particles, axis=0, keepdims=True)  # [N,D]
    radius = np.linalg.norm(centered, axis=1, keepdims=True)  # [N,1]
    unit = centered / np.maximum(radius, 1.0e-15)  # [N,D] / [N,1]
    rng = np.random.default_rng(seed)
    first = rng.integers(0, unit.shape[0], size=count)  # shape: [S]
    second = rng.integers(0, unit.shape[0], size=count)  # shape: [S]
    valid = first != second
    return np.sum(unit[first[valid]] * unit[second[valid]], axis=1)  # [S,D] -> [S]


def plot_forward_smoke(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Plot forward scale, ball conformity, directions, and plane geometry."""
    history = arrays["history"]  # shape: [J+1,N,D]
    initial = history[0]  # shape: [N,D]
    terminal = history[-1]  # shape: [N,D]
    steps = arrays["forward_log_step"]  # shape: [F]
    distance = arrays["forward_mean_pair_distance"]  # shape: [F]
    radius = arrays["forward_rms_radius"]  # shape: [F]
    force = arrays["forward_max_force_norm"]  # shape: [F]
    update = arrays["forward_max_update_norm"]  # shape: [F]

    figure, axes = plt.subplots(2, 3, figsize=(16.0, 9.0), constrained_layout=True)
    axes[0, 0].plot(steps, distance / max(float(distance[0]), 1.0e-15), label="mean pair distance")
    axes[0, 0].plot(steps, radius / max(float(radius[0]), 1.0e-15), label="RMS radius")
    axes[0, 0].set_title("Forward scale change")
    axes[0, 0].set_xlabel("forward step")
    axes[0, 0].set_ylabel("value / initial value")
    axes[0, 0].legend()

    for values, label, color in ((initial, "initial", "#777777"), (terminal, "terminal", "#377EB8")):
        normalized, empirical, reference = _radial_curve(values)
        axes[0, 1].plot(normalized, empirical, color=color, label=f"{label} empirical")
        if label == "terminal":
            axes[0, 1].plot(normalized, reference, color="black", linestyle="--", label="uniform D-ball")
    axes[0, 1].set_title("Full radial CDF")
    axes[0, 1].set_xlabel("radius / fitted ball radius")
    axes[0, 1].set_ylabel("cumulative fraction")
    axes[0, 1].legend(fontsize=8)

    zone = np.arange(10)
    axes[0, 2].bar(zone - 0.18, _radial_zones(initial), width=0.36, color="#AAAAAA", label="initial")
    axes[0, 2].bar(zone + 0.18, _radial_zones(terminal), width=0.36, color="#377EB8", label="terminal")
    axes[0, 2].axhline(0.10, color="black", linestyle="--", label="ideal 0.10")
    axes[0, 2].set_title("Equal-probability radial zones")
    axes[0, 2].set_xlabel("zone")
    axes[0, 2].set_ylabel("particle fraction")
    axes[0, 2].legend(fontsize=8)

    terminal_dots = _sample_angular_dots(terminal)
    rng = np.random.default_rng(1)
    reference_direction = rng.normal(size=terminal.shape)  # shape: [N,D]
    reference_dots = _sample_angular_dots(reference_direction, seed=2)
    bins = np.linspace(-1.0, 1.0, 60)
    axes[1, 0].hist(reference_dots, bins=bins, density=True, histtype="step", color="black", label="uniform directions")
    axes[1, 0].hist(terminal_dots, bins=bins, density=True, alpha=0.55, color="#377EB8", label="terminal")
    axes[1, 0].set_title("Angular dot products")
    axes[1, 0].set_xlabel("direction dot product")
    axes[1, 0].set_ylabel("density")
    axes[1, 0].legend(fontsize=8)

    if initial.shape[1] == 2:
        # In D=2 the EFS plane already is the requested picture. Projecting it
        # again with PCA would rotate the axes and make trajectories less direct.
        initial_view = initial  # shape: [N,2]
        terminal_view = terminal  # shape: [N,2]
        x_label, y_label = "coordinate 1", "coordinate 2"
        axes[1, 1].set_title("One pooled plane, raw coordinates")
        axes[1, 1].set_aspect("equal", adjustable="box")
    else:
        # For D>2, use one shared projection so initial and terminal positions
        # remain comparable in the same two-dimensional coordinate system.
        mean, components, ratio = _fit_pca(initial, terminal)
        initial_view = (initial - mean) @ components.T  # [N,D] @ [D,2] -> [N,2]
        terminal_view = (terminal - mean) @ components.T  # [N,D] @ [D,2] -> [N,2]
        x_label = f"PC1 ({100 * ratio[0]:.1f}%)"
        y_label = f"PC2 ({100 * ratio[1]:.1f}%)"
        axes[1, 1].set_title("One pooled plane, PCA view")

    axes[1, 1].scatter(initial_view[:, 0], initial_view[:, 1], s=5, alpha=0.20, color="#777777", label="initial")
    axes[1, 1].scatter(terminal_view[:, 0], terminal_view[:, 1], s=5, alpha=0.35, color="#377EB8", label="terminal")
    axes[1, 1].set_xlabel(x_label)
    axes[1, 1].set_ylabel(y_label)
    axes[1, 1].legend(fontsize=8)

    axes[1, 2].plot(steps, np.maximum(force, 1.0e-18), label="max force")
    axes[1, 2].plot(steps, np.maximum(update, 1.0e-18), label="max update")
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_title("Euler numerical size")
    axes[1, 2].set_xlabel("forward step")
    axes[1, 2].legend()

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_vertex_replay(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Plot exact-memory replay errors on a logarithmic scale."""
    particle = arrays["vertex_particle_relative_rmse"]  # shape: [V,P]
    source = arrays["vertex_source_relative_rmse"]  # shape: [V]
    source_ids = arrays["vertex_source_ids"]  # shape: [V]
    flat = particle.reshape(-1)  # [V,P] -> [V*P]
    x = np.arange(flat.size)
    centers = np.arange(source.size) * particle.shape[1] + (particle.shape[1] - 1) / 2.0

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), constrained_layout=True)
    axes[0].scatter(x, np.maximum(flat, 1.0e-15), s=26, color="#377EB8", label="particle error")
    axes[0].scatter(centers, np.maximum(source, 1.0e-15), marker="D", s=42, color="black", label="complete source")
    axes[0].axhline(0.10, color="#E41A1C", linestyle="--", label="10% hard gate")
    axes[0].set_yscale("log")
    axes[0].set_xticks(centers, [str(int(value)) for value in source_ids])
    axes[0].set_xlabel("memory source ID")
    axes[0].set_ylabel("RMSE / memory standard deviation")
    axes[0].set_title("Exact terminal vertices replayed")
    axes[0].legend(fontsize=8)

    sorted_error = np.sort(np.maximum(flat, 1.0e-15))
    empirical = np.arange(1, sorted_error.size + 1) / sorted_error.size
    axes[1].step(sorted_error, empirical, where="post", color="#377EB8")
    axes[1].axvline(0.10, color="#E41A1C", linestyle="--", label="10% hard gate")
    for quantile in (0.50, 0.90, 0.95):
        value = float(np.quantile(flat, quantile))
        axes[1].axvline(value, linestyle=":", label=f"p{int(100 * quantile)}={value:.2e}")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("particle relative RMSE")
    axes[1].set_ylabel("cumulative fraction")
    axes[1].set_title("What near-zero actually means")
    axes[1].legend(fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_reconstruction(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Plot held-out target errors and whether EFS improved direct interpolation."""
    point_mode = arrays["target_sources"].shape[1] == 1
    distance_name = "point_distance" if point_mode else "relative_distance"
    direct = arrays[f"metric_direct_{distance_name}"]  # shape: [R,M=1]
    generated = arrays[f"metric_generated_{distance_name}"]  # shape: [R,M=1]
    ratio = arrays["metric_efs_direct_ratio"]  # shape: [R,M]
    direct_relation = arrays["metric_direct_relation_rmse"]  # shape: [R,M]
    generated_relation = arrays["metric_generated_relation_rmse"]  # shape: [R,M]
    fit_error = arrays["lambda_fit_rmse"]  # shape: [R]
    names = arrays["method_names"]  # shape: [M]

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    for method, name in enumerate(names):
        x = np.full(direct.shape[0], method, dtype=float)
        for trial in range(direct.shape[0]):
            axes[0, 0].plot(
                (method - 0.12, method + 0.12), (direct[trial, method], generated[trial, method]), color="#BBBBBB", linewidth=0.7
            )
        axes[0, 0].scatter(
            x - 0.12, direct[:, method], marker="o", facecolors="none", edgecolors=METHOD_COLORS[method], label=f"{name} direct"
        )
        axes[0, 0].scatter(x + 0.12, generated[:, method], marker="x", color=METHOD_COLORS[method], label=f"{name} EFS")
    axes[0, 0].set_xticks(np.arange(names.size), names)
    axes[0, 0].set_ylabel("raw Euclidean point distance" if point_mode else "relative source distance")
    axes[0, 0].set_title("Direct interpolation versus EFS")
    axes[0, 0].legend(fontsize=7)

    for method, name in enumerate(names):
        axes[0, 1].scatter(np.arange(ratio.shape[0]), ratio[:, method], s=30, color=METHOD_COLORS[method], label=str(name))
    axes[0, 1].axhline(1.0, color="black", linestyle="--", label="EFS adds nothing")
    axes[0, 1].set_xlabel("held-out target")
    axes[0, 1].set_ylabel("EFS error / direct error")
    axes[0, 1].set_title("Below 1 means EFS polished the target")
    axes[0, 1].legend(fontsize=8)

    for method, name in enumerate(names):
        axes[1, 0].scatter(fit_error, ratio[:, method], s=34, color=METHOD_COLORS[method], label=str(name))
    axes[1, 0].axhline(1.0, color="black", linestyle="--")
    fitted_dimension = arrays["terminal_candidates"].shape[2] * arrays["terminal_candidates"].shape[3]
    axes[1, 0].set_xlabel("shared-lambda terminal fit RMSE")
    axes[1, 0].set_ylabel("EFS error / direct error")
    axes[1, 0].set_title(f"Does the {fitted_dimension}D terminal fit predict EFS damage?")
    axes[1, 0].legend(fontsize=8)

    if point_mode:
        comparison = arrays["metric_nearest_parent_point_distance"]  # shape: [R]
        for method, name in enumerate(names):
            axes[1, 1].scatter(
                comparison, generated[:, method], s=34, color=METHOD_COLORS[method], label=str(name)
            )
        limit = max(float(np.max(comparison)), float(np.max(generated)), 1.0e-12)
        axes[1, 1].set_xlabel("nearest-parent point error")
        axes[1, 1].set_ylabel("EFS point error")
        axes[1, 1].set_title("Below diagonal beats nearest retrieval")
    else:
        for method, name in enumerate(names):
            axes[1, 1].scatter(
                direct_relation[:, method], generated_relation[:, method], s=34, color=METHOD_COLORS[method], label=str(name)
            )
        limit = max(float(np.max(direct_relation)), float(np.max(generated_relation)), 1.0e-12)
        axes[1, 1].set_xlabel("direct internal-relation error")
        axes[1, 1].set_ylabel("EFS internal-relation error")
        axes[1, 1].set_title("Complete-source geometry")
    axes[1, 1].plot((0, limit), (0, limit), color="black", linestyle="--", label="unchanged")
    axes[1, 1].legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_lambda_diagnostics(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Show the one fitted shared lambda for every held-out target."""
    weights = arrays["shared_lambdas"][:, 0, :]  # [R,M=1,K] -> [R,K]
    fit_error = arrays["lambda_fit_rmse"]  # shape: [R]
    trial = int(np.argsort(fit_error)[fit_error.size // 2])

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    image = axes[0].imshow(weights, vmin=0.0, vmax=1.0, aspect="auto", cmap="viridis")
    axes[0].set_xlabel("parent source within local set")
    axes[0].set_ylabel("held-out target")
    axes[0].set_title("One shared nonnegative lambda per target")
    figure.colorbar(image, ax=axes[0], label="weight")

    x = np.arange(weights.shape[1])
    axes[1].bar(x, weights[trial], width=0.65, color=METHOD_COLORS[0])
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xlabel("parent source within local set")
    axes[1].set_ylabel("shared weight")
    axes[1].set_title(f"Median-difficulty target {trial}, fit RMSE={fit_error[trial]:.3g}")
    axes[1].grid(alpha=0.2)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_flow(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Plot one representative source in raw 2D or a PCA view for D > 2."""
    generated_error = arrays["metric_generated_relative_distance"][:, 0]  # shape: [R]
    trial = int(np.argsort(generated_error)[generated_error.size // 2])
    history = arrays["history"]  # [J+1,N,D]
    parents_initial = arrays["matched_parents"][trial]  # [K,P,D]
    parents_terminal = arrays["matched_terminal_parents"][trial]  # [K,P,D]
    target = arrays["aligned_targets"][trial]  # [P,D]
    oracle_trajectory = arrays["oracle_target_trajectory"][:, trial]  # [J+1,P,D]
    terminal_candidate = arrays["terminal_candidates"][trial]  # [M,P,D]
    direct = arrays["direct_candidates"][trial]  # [M,P,D]
    generated = arrays["generated"][trial]  # [M,P,D]
    replay = arrays["target_trajectory"][:, trial]  # [J+1,M,P,D]
    dimension = history.shape[2]

    raw_2d = dimension == 2
    if raw_2d:
        mean = np.zeros((1, 2), dtype=np.float64)  # raw coordinates, no fitted projection
        components = np.eye(2, dtype=np.float64)
        ratio = np.asarray((np.nan, np.nan))
    else:
        mean, components, ratio = _fit_pca(
            history[0],
            history[-1],
            parents_initial.reshape(-1, dimension),
            parents_terminal.reshape(-1, dimension),
            target,
            oracle_trajectory[-1],
            terminal_candidate.reshape(-1, dimension),
            generated.reshape(-1, dimension),
        )

    def project(values: np.ndarray) -> np.ndarray:
        return (values - mean) @ components.T  # [...,D] @ [D,2] -> [...,2]

    initial_pca = project(history[0])
    terminal_pca = project(history[-1])
    parent_initial_pca = project(parents_initial)
    parent_terminal_pca = project(parents_terminal)
    target_pca = project(target)
    oracle_terminal_pca = project(oracle_trajectory[-1])
    terminal_candidate_pca = project(terminal_candidate)
    direct_pca = project(direct)
    generated_pca = project(generated)

    # Follow a fixed subset of memory rows. Row identity is preserved by EFS,
    # so each line is one actual particle moving through the one pooled field.
    forward_stride = max(1, (history.shape[0] - 1) // 250)
    forward_particle = np.linspace(
        0, history.shape[1] - 1, min(128, history.shape[1]), dtype=np.int64
    )  # shape: [S], deterministic visual subset of the N memory rows
    forward_pca = project(history[::forward_stride, forward_particle])  # [F,S,D] @ [D,2] -> [F,S,2]
    oracle_pca = project(oracle_trajectory[::forward_stride])  # passive target flow, shape: [F,P,2]

    stride = max(1, (replay.shape[0] - 1) // 250)
    replay_pca = project(replay[::-1][::stride])  # terminal-to-output, [F,M,P,2]
    figure, axes = plt.subplots(1, 4, figsize=(17.5, 4.6), sharex=True, sharey=True, constrained_layout=True)

    axes[0].scatter(initial_pca[:, 0], initial_pca[:, 1], s=5, color="#BBBBBB", alpha=0.20)
    for parent in range(parent_initial_pca.shape[0]):
        axes[0].scatter(parent_initial_pca[parent, :, 0], parent_initial_pca[parent, :, 1], s=24, label=f"parent {parent + 1}")
    axes[0].scatter(target_pca[:, 0], target_pca[:, 1], marker="X", s=55, color="black", label="held-out target")
    axes[0].set_title("1. Initial plane and target")
    axes[0].legend(fontsize=6)

    for particle in range(forward_pca.shape[1]):
        axes[1].plot(forward_pca[:, particle, 0], forward_pca[:, particle, 1], color="#999999", alpha=0.10, linewidth=0.45)
    for particle in range(oracle_pca.shape[1]):
        axes[1].plot(
            oracle_pca[:, particle, 0], oracle_pca[:, particle, 1], color="black", alpha=0.60, linewidth=0.8, linestyle=":"
        )
    axes[1].scatter(terminal_pca[:, 0], terminal_pca[:, 1], s=5, color="#BBBBBB", alpha=0.25)
    axes[1].scatter(
        oracle_terminal_pca[:, 0], oracle_terminal_pca[:, 1], marker="X", s=55, color="black", label="passive target endpoint"
    )
    for method, name in enumerate(arrays["method_names"]):
        axes[1].scatter(
            terminal_candidate_pca[method, :, 0],
            terminal_candidate_pca[method, :, 1],
            s=38,
            color=METHOD_COLORS[method],
            label=str(name),
        )
    axes[1].set_title("2. Forward flow and terminal interpolation")
    axes[1].legend(fontsize=6)

    for method, name in enumerate(arrays["method_names"]):
        for particle in range(replay_pca.shape[2]):
            axes[2].plot(
                replay_pca[:, method, particle, 0],
                replay_pca[:, method, particle, 1],
                color=METHOD_COLORS[method],
                alpha=0.65,
                linewidth=0.8,
            )
        axes[2].plot([], [], color=METHOD_COLORS[method], label=str(name))
    axes[2].scatter(
        terminal_candidate_pca[:, :, 0].reshape(-1),
        terminal_candidate_pca[:, :, 1].reshape(-1),
        marker="o",
        s=30,
        facecolors="none",
        edgecolors=METHOD_COLORS[0],
        label="replay start",
    )
    axes[2].scatter(
        generated_pca[:, :, 0].reshape(-1),
        generated_pca[:, :, 1].reshape(-1),
        marker="x",
        s=34,
        color=METHOD_COLORS[0],
        label="replay output",
    )
    axes[2].set_title("3. Backward replay")
    axes[2].legend(fontsize=7)

    axes[3].scatter(initial_pca[:, 0], initial_pca[:, 1], s=5, color="#BBBBBB", alpha=0.18)
    axes[3].scatter(target_pca[:, 0], target_pca[:, 1], marker="X", s=55, color="black", label="target")
    for method, name in enumerate(arrays["method_names"]):
        axes[3].scatter(
            direct_pca[method, :, 0],
            direct_pca[method, :, 1],
            s=34,
            facecolors="none",
            edgecolors=METHOD_COLORS[method],
            label=f"{name} direct",
        )
        axes[3].scatter(
            generated_pca[method, :, 0],
            generated_pca[method, :, 1],
            s=35,
            color=METHOD_COLORS[method],
            marker="x",
            label=f"{name} EFS",
        )
    axes[3].set_title("4. Output versus held-out source")
    axes[3].legend(fontsize=5)

    x_label = "coordinate 1" if raw_2d else f"PC1 ({100 * ratio[0]:.1f}%)"
    y_label = "coordinate 2" if raw_2d else f"PC2 ({100 * ratio[1]:.1f}%)"
    for axis in axes:
        axis.set_xlabel(x_label)
        axis.grid(alpha=0.2)
        if raw_2d:
            axis.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel(y_label)
    view_name = "raw two-dimensional coordinates" if raw_2d else "one fitted PCA basis"
    figure.suptitle(f"Representative held-out target {trial}, {view_name}")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_point_trials_2d(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Plot every ``P=1,D=2`` target and its full raw-coordinate EFS route."""
    history = arrays["history"]  # shape: [J+1,N,2]
    targets = arrays["target_sources"][:, 0]  # [R,1,2] -> [R,2]
    terminal_targets = arrays["oracle_target_trajectory"][-1, :, 0]  # shape: [R,2]
    terminal_candidates = arrays["terminal_candidates"][:, 0, 0]  # [R,1,1,2] -> [R,2]
    direct = arrays["direct_candidates"][:, 0, 0]  # shape: [R,2]
    generated = arrays["generated"][:, 0, 0]  # shape: [R,2]
    replay = arrays["target_trajectory"][::-1, :, 0, 0]  # terminal-to-output, shape: [J+1,R,2]
    ratio = arrays["metric_efs_direct_ratio"][:, 0]  # shape: [R]
    colors = np.where(ratio < 1.0, "#2CA02C", "#D62728")  # green improves, red worsens

    stride = max(1, (replay.shape[0] - 1) // 500)
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.1), constrained_layout=True)

    axes[0].scatter(history[-1, :, 0], history[-1, :, 1], s=5, color="#BBBBBB", alpha=0.25)
    for trial in range(targets.shape[0]):
        axes[0].plot(
            (terminal_targets[trial, 0], terminal_candidates[trial, 0]),
            (terminal_targets[trial, 1], terminal_candidates[trial, 1]),
            color=colors[trial],
            alpha=0.45,
            linewidth=0.8,
        )
    axes[0].scatter(terminal_targets[:, 0], terminal_targets[:, 1], marker="X", s=28, color="black", label="passive targets")
    axes[0].scatter(
        terminal_candidates[:, 0], terminal_candidates[:, 1], marker="o", s=28, facecolors="none", edgecolors="#377EB8", label="lambda candidates"
    )
    axes[0].set_title("Terminal fit gap")
    axes[0].legend(fontsize=7)

    for trial in range(targets.shape[0]):
        route = replay[::stride, trial]  # shape: [F,2]
        axes[1].plot(route[:, 0], route[:, 1], color=colors[trial], alpha=0.65, linewidth=0.8)
    axes[1].scatter(terminal_candidates[:, 0], terminal_candidates[:, 1], s=20, facecolors="none", edgecolors="#377EB8")
    axes[1].scatter(generated[:, 0], generated[:, 1], marker="x", s=24, color=colors)
    axes[1].plot([], [], color="#2CA02C", label="EFS improves")
    axes[1].plot([], [], color="#D62728", label="EFS worsens")
    axes[1].set_title("All backward trajectories")
    axes[1].legend(fontsize=7)

    axes[2].scatter(history[0, :, 0], history[0, :, 1], s=5, color="#BBBBBB", alpha=0.22)
    for trial in range(targets.shape[0]):
        axes[2].plot(
            (targets[trial, 0], generated[trial, 0]),
            (targets[trial, 1], generated[trial, 1]),
            color=colors[trial],
            alpha=0.45,
            linewidth=0.8,
        )
    axes[2].scatter(targets[:, 0], targets[:, 1], marker="X", s=30, color="black", label="held-out targets")
    axes[2].scatter(direct[:, 0], direct[:, 1], s=25, facecolors="none", edgecolors="#777777", label="direct baseline")
    axes[2].scatter(generated[:, 0], generated[:, 1], marker="x", s=26, color=colors, label="EFS output")
    axes[2].set_title("Output error in the initial plane")
    axes[2].legend(fontsize=7)

    for axis in axes:
        axis.set_xlabel("coordinate 1")
        axis.set_ylabel("coordinate 2")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_replay_movement(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Plot complete-source spread and per-frame total-position movement."""
    replay = arrays["target_trajectory"][::-1]  # terminal-to-output, [J+1,R,M,P,D]
    method_names = arrays["method_names"]  # shape: [M]
    particle_count = replay.shape[3]
    if particle_count == 1:
        target = arrays["target_sources"][:, 0]  # [R,1,D] -> [R,D]
        difference = replay[:, :, :, 0] - target[None, :, None, :]  # [J+1,R,M,D] - [1,R,1,D]
        mean_distance = np.linalg.norm(difference, axis=3)  # shape: [J+1,R,M]
        left_label = "distance to held-out point"
        left_title = "Point error during replay"
    else:
        mean_distance = np.zeros(replay.shape[:3], dtype=np.float64)  # [J+1,R,M]
        pair_count = 0
        for first in range(particle_count):
            for second in range(first + 1, particle_count):
                difference = replay[:, :, :, first] - replay[:, :, :, second]  # [J+1,R,M,D]
                mean_distance += np.linalg.norm(difference, axis=3)  # [J+1,R,M]
                pair_count += 1
        mean_distance /= pair_count
        left_label = "mean internal particle distance"
        left_title = "Complete-source spread"
    total = np.sum(replay, axis=3)  # [J+1,R,M,P,D] -> [J+1,R,M,D]
    displacement = np.linalg.norm(np.diff(total, axis=0), axis=3)  # [J,R,M]

    x = np.arange(replay.shape[0])
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.5), constrained_layout=True)
    for method, name in enumerate(method_names):
        median = np.median(mean_distance[:, :, method], axis=1)
        low = np.quantile(mean_distance[:, :, method], 0.25, axis=1)
        high = np.quantile(mean_distance[:, :, method], 0.75, axis=1)
        axes[0].plot(x, median, color=METHOD_COLORS[method], label=str(name))
        axes[0].fill_between(x, low, high, color=METHOD_COLORS[method], alpha=0.18)
        axes[1].plot(x[1:], np.median(displacement[:, :, method], axis=1), color=METHOD_COLORS[method], label=str(name))
    axes[0].set_xlabel("completed backward frames")
    axes[0].set_ylabel(left_label)
    axes[0].set_title(left_title)
    axes[1].set_xlabel("completed backward frames")
    axes[1].set_ylabel("total-position displacement")
    axes[1].set_yscale("log")
    axes[1].set_title("Movement per full backward frame")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _two_pass_view(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, str, str, bool]:
    """Return one shared raw-2D or PCA projection for both forward passes."""
    dimension = arrays["reference_history"].shape[2]
    if dimension == 2:
        return np.zeros((1, 2)), np.eye(2), "coordinate 1", "coordinate 2", True
    mean, components, ratio = _fit_pca(
        arrays["reference_initial_plane"],
        arrays["reference_history"][-1],
        arrays["generation_history"][-1],
    )
    return mean, components, f"PC1 ({100 * ratio[0]:.1f}%)", f"PC2 ({100 * ratio[1]:.1f}%)", False


def plot_two_pass_target(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Show the known target, saved parents, two terminal seeds, and replay."""
    mean, components, x_label, y_label, raw_2d = _two_pass_view(arrays)

    def project(values: np.ndarray) -> np.ndarray:
        return (values - mean) @ components.T  # [...,D] @ [D,2] -> [...,2]

    target_initial = project(arrays["target_sources"][0])  # [P,D] -> [P,2]
    parent_initial = project(arrays["matched_initial_reference_parents"])  # [K,P,D] -> [K,P,2]
    reference_terminal = project(arrays["reference_history"][-1])  # [N1,D] -> [N1,2]
    target_terminal = project(arrays["target_terminal"])  # [P,D] -> [P,2]
    parent_reference = project(arrays["matched_reference_terminal_parents"])  # [K,P,D] -> [K,P,2]
    candidate_reference = project(arrays["reference_candidate"])  # [P,D] -> [P,2]
    generation_terminal = project(arrays["generation_history"][-1])  # [N2,D] -> [N2,2]
    parent_generation = project(arrays["matched_generation_terminal_parents"])  # [K,P,D] -> [K,P,2]
    candidate_generation = project(arrays["generation_candidates"][0, 0])  # [P,D] -> [P,2]
    replay = project(arrays["target_trajectory"][:, 0, 0])  # [J+1,P,D] -> [J+1,P,2]
    output = project(arrays["generated"][0, 0])  # [P,D] -> [P,2]

    particle_count = target_initial.shape[0]
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, particle_count))
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 11.0), constrained_layout=True)

    axes[0, 0].scatter(
        project(arrays["reference_initial_plane"])[:, 0],
        project(arrays["reference_initial_plane"])[:, 1],
        s=4,
        color="#BBBBBB",
        alpha=0.18,
    )
    axes[0, 0].scatter(parent_initial[:, :, 0], parent_initial[:, :, 1], s=18, color="#377EB8", alpha=0.55, label="saved d1 parents")
    axes[0, 0].scatter(target_initial[:, 0], target_initial[:, 1], marker="X", s=58, color=colors, edgecolors="black", label="target")
    axes[0, 0].set_title("1. d1 target and saved neighbors")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].scatter(reference_terminal[:, 0], reference_terminal[:, 1], s=4, color="#BBBBBB", alpha=0.18)
    axes[0, 1].scatter(parent_reference[:, :, 0], parent_reference[:, :, 1], s=18, color="#377EB8", alpha=0.55, label="same parents")
    for particle in range(particle_count):
        axes[0, 1].plot(
            (target_terminal[particle, 0], candidate_reference[particle, 0]),
            (target_terminal[particle, 1], candidate_reference[particle, 1]),
            color=colors[particle],
            linewidth=1.0,
        )
    axes[0, 1].scatter(target_terminal[:, 0], target_terminal[:, 1], marker="X", s=58, color=colors, edgecolors="black", label="target endpoint")
    axes[0, 1].scatter(candidate_reference[:, 0], candidate_reference[:, 1], marker="o", s=38, facecolors="none", edgecolors=colors, label="pass-1 lambda seed")
    axes[0, 1].set_title("2. Pass 1 lambda fitted at du")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].scatter(generation_terminal[:, 0], generation_terminal[:, 1], s=4, color="#BBBBBB", alpha=0.18)
    axes[1, 0].scatter(parent_generation[:, :, 0], parent_generation[:, :, 1], s=18, color="#377EB8", alpha=0.55, label="shifted parents")
    for particle in range(particle_count):
        axes[1, 0].plot(
            (target_terminal[particle, 0], candidate_generation[particle, 0]),
            (target_terminal[particle, 1], candidate_generation[particle, 1]),
            color=colors[particle],
            linewidth=1.0,
        )
    axes[1, 0].scatter(target_terminal[:, 0], target_terminal[:, 1], marker="X", s=58, color=colors, edgecolors="black", label="stored target endpoint")
    axes[1, 0].scatter(candidate_generation[:, 0], candidate_generation[:, 1], marker="o", s=38, facecolors="none", edgecolors=colors, label="pass-2 seed")
    axes[1, 0].set_title("3. Same lambda after removing target")
    axes[1, 0].legend(fontsize=8)

    stride = max(1, (replay.shape[0] - 1) // 500)
    replay_terminal_to_initial = replay[::-1][::stride]  # reverse saved order, shape: [F,P,2]
    for particle in range(particle_count):
        axes[1, 1].plot(
            replay_terminal_to_initial[:, particle, 0],
            replay_terminal_to_initial[:, particle, 1],
            color=colors[particle],
            linewidth=1.0,
        )
    axes[1, 1].scatter(target_initial[:, 0], target_initial[:, 1], marker="X", s=58, color=colors, edgecolors="black", label="real target d1")
    axes[1, 1].scatter(output[:, 0], output[:, 1], marker="x", s=50, color=colors, label="generated arrival")
    axes[1, 1].set_title("4. Backward arrival versus target")
    axes[1, 1].legend(fontsize=8)

    for axis in axes.flat:
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.2)
        if raw_2d:
            axis.set_aspect("equal", adjustable="box")
    figure.suptitle("Two-pass target-removed H4.5")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_two_pass_field_shift(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Plot retained endpoint movement and descriptive marginal KL."""
    mean, components, x_label, y_label, raw_2d = _two_pass_view(arrays)

    def project(values: np.ndarray) -> np.ndarray:
        return (values - mean) @ components.T  # [...,D] @ [D,2] -> [...,2]

    reference = arrays["retained_reference_terminal"]  # shape: [N,D]
    generation = arrays["generation_history"][-1]  # shape: [N,D]
    reference_view = project(reference)  # shape: [N,2]
    generation_view = project(generation)  # shape: [N,2]
    kl_forward = arrays["field_kl_reference_to_generation"]  # shape: [D]
    kl_reverse = arrays["field_kl_generation_to_reference"]  # shape: [D]
    worst_coordinate = int(np.argmax(0.5 * (kl_forward + kl_reverse)))
    edges = arrays["field_kl_bin_edges"][worst_coordinate]  # shape: [B+1]
    centers = 0.5 * (edges[:-1] + edges[1:])  # shape: [B]

    figure, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), constrained_layout=True)
    witness = np.linspace(0, reference.shape[0] - 1, min(256, reference.shape[0]), dtype=np.int64)
    for particle in witness:
        axes[0].plot(
            (reference_view[particle, 0], generation_view[particle, 0]),
            (reference_view[particle, 1], generation_view[particle, 1]),
            color="#999999",
            alpha=0.18,
            linewidth=0.5,
        )
    axes[0].scatter(reference_view[:, 0], reference_view[:, 1], s=4, color="#377EB8", alpha=0.18, label="pass 1 retained")
    axes[0].scatter(generation_view[:, 0], generation_view[:, 1], s=4, color="#E41A1C", alpha=0.18, label="pass 2")
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)
    axes[0].set_title("Terminal field shift after removing T")
    axes[0].legend(fontsize=8)
    if raw_2d:
        axes[0].set_aspect("equal", adjustable="box")

    coordinate = np.arange(kl_forward.size)
    axes[1].bar(coordinate - 0.18, kl_forward, width=0.36, color="#377EB8", label="KL(pass 1 || pass 2)")
    axes[1].bar(coordinate + 0.18, kl_reverse, width=0.36, color="#E41A1C", label="KL(pass 2 || pass 1)")
    axes[1].set_xticks(coordinate)
    axes[1].set_xlabel("coordinate")
    axes[1].set_ylabel("marginal histogram KL")
    axes[1].set_title(f"Mean directional KL: {np.mean(kl_forward):.3g} / {np.mean(kl_reverse):.3g}")
    axes[1].legend(fontsize=7)

    axes[2].step(
        centers,
        arrays["field_kl_reference_probability"][worst_coordinate],
        where="mid",
        color="#377EB8",
        label="pass 1 retained",
    )
    axes[2].step(
        centers,
        arrays["field_kl_generation_probability"][worst_coordinate],
        where="mid",
        color="#E41A1C",
        label="pass 2",
    )
    axes[2].set_xlabel(f"coordinate {worst_coordinate}")
    axes[2].set_ylabel("smoothed bin probability")
    axes[2].set_title("Coordinate with largest symmetric shift")
    axes[2].legend(fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def load_arrays(run_directory: Path) -> dict[str, np.ndarray]:
    """Load ``samples.npz`` into a plain dictionary."""
    with np.load(run_directory / "samples.npz", allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def plot_all(run_directory: Path) -> list[Path]:
    """Regenerate every figure supported by the arrays saved in one run."""
    arrays = load_arrays(run_directory)
    written: list[Path] = []
    if "two_pass_protocol" in arrays:
        jobs: list[tuple[str, object]] = []
        if "field_kl_reference_to_generation" in arrays:
            jobs.append(("two_pass_field_shift.png", plot_two_pass_field_shift))
        if "generated" in arrays:
            jobs.append(("two_pass_target.png", plot_two_pass_target))
        for name, function in jobs:
            path = run_directory / name
            function(path, arrays)
            written.append(path)
        return written

    jobs = [("forward_smoke.png", plot_forward_smoke)]
    if "vertex_particle_relative_rmse" in arrays:
        jobs.append(("vertex_replay.png", plot_vertex_replay))
    if "metric_generated_relative_distance" in arrays:
        flow_name = "target_flow_2d.png" if arrays["history"].shape[2] == 2 else "target_flow_pca.png"
        jobs.extend((
            ("target_reconstruction.png", plot_reconstruction),
            ("lambda_diagnostics.png", plot_lambda_diagnostics),
            (flow_name, plot_flow),
            ("replay_movement.png", plot_replay_movement),
        ))
        if arrays["target_sources"].shape[1:] == (1, 2):
            jobs.append(("point_trials_2d.png", plot_point_trials_2d))
    for name, function in jobs:
        path = run_directory / name
        function(path, arrays)
        written.append(path)
    return written


def main() -> None:
    """Parse one saved run directory and regenerate its figures."""
    parser = argparse.ArgumentParser(description="Regenerate one-plane H4.5 figures from saved arrays.")
    parser.add_argument("run_directory", type=Path, help="Timestamped run directory containing samples.npz.")
    args = parser.parse_args()
    paths = plot_all(args.run_directory.resolve())
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

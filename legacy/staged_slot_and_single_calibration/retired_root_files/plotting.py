"""Regenerate every H4.5 figure from one saved result directory.

Purpose:
    Plot forward smoke diagnostics, vertex replay, target recovery, one fixed
    particle-level PCA flow, and backward movement without rerunning EFS.

Dependencies:
    NumPy and Matplotlib from req.txt.

Outputs:
    PNG figures written inside the supplied result directory.

Exact command:
    python plotting.py results\\<run-directory>
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
from matplotlib.ticker import FuncFormatter

tab10 = colormaps["tab10"]
Set1 = colormaps["Set1"]


METHOD_COLORS = ("#E41A1C", "#A65628")
METHOD_MARKERS = ("o", "s")


def _memory_planes(memory: np.ndarray) -> np.ndarray:
    """Return pooled or slot memory through a common ``[H,N,D]`` view."""
    memory = np.asarray(memory, dtype=np.float64)
    if memory.ndim == 2:
        return memory[None, :, :]  # [N,D] -> [H=1,N,D]
    if memory.ndim == 3:
        return memory  # shape: [H=P,C,D]
    raise ValueError("memory must have shape [N,D] or [P,C,D]")


def _flat_memory(memory: np.ndarray) -> np.ndarray:
    """Flatten empirical field axes while preserving coordinate axis D."""
    memory = np.asarray(memory, dtype=np.float64)
    return memory.reshape(-1, memory.shape[-1])  # [N,D] or [P,C,D] -> [H*N,D]


def _radial_zone_fraction(particles: np.ndarray) -> tuple[np.ndarray, float]:
    """Return ten fitted-ball equal-probability zones and overflow."""
    radius, fitted_radius = _centered_radius(particles)
    dimension = particles.shape[1]
    uniform_coordinate = (radius / max(fitted_radius, 1.0e-15)) ** dimension  # shape: [N]
    zone = np.floor(10.0 * uniform_coordinate).astype(np.int64)  # shape: [N]
    fraction = np.asarray([np.mean(zone == index) for index in range(10)])  # shape: [10]
    return fraction, float(np.mean(uniform_coordinate >= 1.0))


def _centered_radius(particles: np.ndarray) -> tuple[np.ndarray, float]:
    """Return centered radii ``[N]`` and the fitted uniform-ball radius."""
    centered = particles - np.mean(particles, axis=0, keepdims=True)  # [N,D]
    radius = np.linalg.norm(centered, axis=1)  # shape: [N]
    dimension = particles.shape[1]
    fitted_radius = math.sqrt((dimension + 2.0) / dimension * float(np.mean(radius * radius)))
    return radius, fitted_radius


def _angular_dots(particles: np.ndarray) -> np.ndarray:
    """Return all distinct centered direction dot products as ``[N*(N-1)]``."""
    centered = particles - np.mean(particles, axis=0, keepdims=True)  # [N,D]
    radius = np.linalg.norm(centered, axis=1)  # shape: [N]
    unit = centered / np.maximum(radius[:, None], 1.0e-15)  # [N,D] / [N,1]
    dot = unit @ unit.T  # [N,D] @ [D,N] -> [N,N]
    return dot[~np.eye(dot.shape[0], dtype=bool)]


def plot_forward_smoke(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Plot scale flow, radial CDF/zones, and angular dot distributions."""
    initial = arrays.get("initial_memory", arrays.get("initial_plane"))
    if initial is None:
        raise KeyError("saved arrays contain no initial memory")
    terminal = arrays["history"][-1]
    initial_planes = _memory_planes(initial)  # shape: [H,N,D]
    terminal_planes = _memory_planes(terminal)  # shape: [H,N,D]
    steps = arrays["forward_log_step"]  # shape: [F]
    pair_distance = np.asarray(arrays["forward_mean_pair_distance"], dtype=np.float64)
    radius_log = np.asarray(arrays["forward_rms_radius"], dtype=np.float64)
    if pair_distance.ndim == 1:
        pair_distance = pair_distance[:, None]  # legacy [F] -> [F,H=1]
    if radius_log.ndim == 1:
        radius_log = radius_log[:, None]  # legacy [F] -> [F,H=1]

    figure, axes = plt.subplots(1, 4, figsize=(19.5, 4.4), constrained_layout=True)

    pair_relative = pair_distance / np.maximum(pair_distance[:1], 1.0e-15)  # [F,H] / [1,H]
    radius_relative = radius_log / np.maximum(radius_log[:1], 1.0e-15)  # [F,H] / [1,H]
    for values, label, color in (
        (pair_relative, "mean pair distance", "#377EB8"),
        (radius_relative, "RMS radius", "#4DAF4A"),
    ):
        mean = np.mean(values, axis=1)  # shape: [F]
        axes[0].plot(steps, mean, label=label, color=color)
        if values.shape[1] > 1:
            axes[0].fill_between(
                steps, np.min(values, axis=1), np.max(values, axis=1), color=color, alpha=0.14
            )
    axes[0].set_title("Forward scale flow")
    axes[0].set_xlabel("fixed forward frame")
    axes[0].set_ylabel("fraction of each field's initial scale")
    axes[0].legend(fontsize=8)

    for planes, label, color in (
        (initial_planes, "initial fields", "#777777"),
        (terminal_planes, "terminal fields", "#377EB8"),
    ):
        for plane_index, values in enumerate(planes):
            radius, fitted_radius = _centered_radius(values)
            normalized = np.sort(radius / max(fitted_radius, 1.0e-15))  # shape: [N]
            empirical = np.arange(1, normalized.size + 1) / normalized.size
            axes[1].plot(
                normalized,
                empirical,
                color=color,
                alpha=0.35 if planes.shape[0] > 1 else 1.0,
                label=label if plane_index == 0 else None,
            )
    radial_axis = np.linspace(0.0, 1.0, 300)
    dimension = initial.shape[-1]
    axes[1].plot(
        radial_axis,
        radial_axis**dimension,
        color="black",
        linestyle="--",
        label=f"uniform {dimension}D ball",
    )
    axes[1].set_title("Centered radial CDF")
    axes[1].set_xlabel("radius / fitted ball radius")
    axes[1].set_ylabel("empirical CDF")
    axes[1].legend(fontsize=8)

    initial_zones = np.stack([_radial_zone_fraction(plane)[0] for plane in initial_planes])  # [H,10]
    terminal_zone_values = [_radial_zone_fraction(plane) for plane in terminal_planes]
    terminal_zones = np.stack([value[0] for value in terminal_zone_values])  # shape: [H,10]
    overflow = np.asarray([value[1] for value in terminal_zone_values])  # shape: [H]
    zone_x = np.arange(10)
    axes[2].plot(zone_x, np.mean(initial_zones, axis=0), color="#777777", marker="o", label="initial mean")
    axes[2].bar(zone_x, np.mean(terminal_zones, axis=0), color="#377EB8", alpha=0.75, label="terminal mean")
    if terminal_zones.shape[0] > 1:
        axes[2].vlines(
            zone_x,
            np.min(terminal_zones, axis=0),
            np.max(terminal_zones, axis=0),
            color="#1F4E79",
            linewidth=1.2,
            label="terminal slot range",
        )
    axes[2].axhline(0.10, color="black", linestyle="--", label="uniform expectation")
    axes[2].text(
        0.02,
        0.97,
        f"terminal overflow: mean {np.mean(overflow):.3f}, max {np.max(overflow):.3f}",
        transform=axes[2].transAxes,
        va="top",
        fontsize=8,
    )
    axes[2].set_title("Equal-probability radial zones")
    axes[2].set_xlabel("zone of u=(r/R)^D")
    axes[2].set_ylabel("particle fraction")
    axes[2].legend(fontsize=7)

    bins = np.linspace(-1.0, 1.0, 81)
    for planes, label, color, width in (
        (initial_planes, "initial", "#777777", 1.0),
        (terminal_planes, "terminal", "#377EB8", 1.6),
    ):
        dots = np.concatenate([_angular_dots(plane) for plane in planes])  # all within-field ordered pairs
        axes[3].hist(
            dots, bins=bins, density=True, histtype="step", linewidth=width, color=color, label=label
        )
    dot_axis = np.linspace(-1.0 + 1.0e-6, 1.0 - 1.0e-6, 300)
    normalizer = math.gamma(dimension / 2.0) / (
        math.sqrt(math.pi) * math.gamma((dimension - 1.0) / 2.0)
    )
    theoretical = normalizer * np.maximum(1.0 - dot_axis * dot_axis, 0.0) ** ((dimension - 3.0) / 2.0)
    axes[3].plot(dot_axis, theoretical, color="black", linestyle="--", label="uniform directions")
    axes[3].set_title("Within-field angular dots")
    axes[3].set_xlabel("direction dot product")
    axes[3].set_ylabel("density")
    axes[3].legend(fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)

def plot_vertex_replay(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Plot all particle-level vertex errors, source RMSE, and empirical CDF."""
    source_ids = arrays["vertex_source_ids"]  # shape: [V]
    if "vertex_particle_relative_rmse" in arrays:
        particle_error = arrays["vertex_particle_relative_rmse"]  # shape: [V,P]
        source_error = arrays["vertex_source_relative_rmse"]  # shape: [V]
    else:
        particle_error = arrays["vertex_relative_rmse"][:, None]  # legacy [V] -> [V,P=1]
        source_error = arrays["vertex_relative_rmse"]  # shape: [V]

    vertex_count, particles_per_source = particle_error.shape
    flat_error = particle_error.reshape(-1)  # [V,P] -> [V*P]
    x = np.arange(flat_error.size)
    colors = np.repeat(tab10(np.linspace(0.0, 0.8, vertex_count)), particles_per_source, axis=0)

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.3), constrained_layout=True)
    axes[0].scatter(x, flat_error, color=colors, s=27, label="particle RMSE")
    source_centers = np.arange(vertex_count) * particles_per_source + (particles_per_source - 1) / 2.0
    axes[0].scatter(
        source_centers, source_error, color="black", marker="D", s=38, label="complete-source RMSE"
    )
    for boundary in range(1, vertex_count):
        axes[0].axvline(boundary * particles_per_source - 0.5, color="#BBBBBB", linewidth=0.7)
    axes[0].axhline(0.10, color="#E41A1C", linestyle="--", label="10% hard gate")
    axes[0].set_xticks(source_centers, [str(value) for value in source_ids])
    axes[0].set_xlabel("complete source ID; points are its particles")
    axes[0].set_ylabel("RMSE / memory standard deviation")
    axes[0].set_title(f"All {flat_error.size} vertex particles")
    axes[0].legend(fontsize=8)

    sorted_error = np.sort(flat_error)
    empirical = np.arange(1, sorted_error.size + 1) / sorted_error.size
    axes[1].step(sorted_error, empirical, where="post", color="#377EB8")
    axes[1].axvline(0.10, color="#E41A1C", linestyle="--", label="10% hard gate")
    for quantile, style in ((0.50, ":"), (0.90, "-."), (0.95, "--")):
        value = float(np.quantile(flat_error, quantile))
        axes[1].axvline(value, color="#555555", linestyle=style, label=f"p{int(100*quantile)}={value:.3f}")
    axes[1].set_xlabel("particle relative RMSE")
    axes[1].set_ylabel("empirical cumulative fraction")
    axes[1].set_title("Particle error distribution")
    axes[1].legend(fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)

def plot_target_recovery(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Plot paired target errors and the terminal commutation diagnostic."""
    random_mask = arrays["trial_kind"] == "random"  # shape: [R]
    clean = arrays["metric_clean_rmse"][random_mask]  # shape: [E,M=2]
    noisy = arrays["metric_noisy_rmse"][random_mask]  # shape: [E,M=2]
    jitter_floor = arrays["metric_jitter_floor"][random_mask]  # shape: [E]
    commutation = arrays["metric_commutation_gap"][random_mask]  # shape: [E]

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), constrained_layout=True)
    for trial in range(clean.shape[0]):
        axes[0].plot((0, 1), clean[trial], color="#AAAAAA", linewidth=0.8, alpha=0.7)
        axes[1].plot((0, 1), noisy[trial], color="#AAAAAA", linewidth=0.8, alpha=0.7)
    for method in range(2):
        axes[0].scatter(
            np.full(clean.shape[0], method),
            clean[:, method],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            label=("shared" if method == 0 else "independent"),
            zorder=3,
        )
        axes[1].scatter(
            np.full(noisy.shape[0], method),
            noisy[:, method],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            zorder=3,
        )

    axes[0].set_xticks((0, 1), ("shared", "independent"))
    axes[0].set_title("Exact clean-target recovery")
    axes[0].set_ylabel("RMSE")
    axes[1].scatter(
        np.full(jitter_floor.size, 2),
        jitter_floor,
        color="#4DAF4A",
        marker="_",
        s=70,
        label="jitter floor",
    )
    axes[1].set_xticks((0, 1, 2), ("shared", "independent", "jitter floor"))
    axes[1].set_title("Held-out noisy-target recovery")
    axes[1].set_ylabel("RMSE")

    axes[2].scatter(commutation, clean[:, 0], color=METHOD_COLORS[0], marker="o")
    for trial, (x_value, y_value) in enumerate(zip(commutation, clean[:, 0])):
        axes[2].annotate(str(trial), (x_value, y_value), xytext=(3, 3), textcoords="offset points", fontsize=7)
    axes[2].set_title("Terminal gap versus final error")
    axes[2].set_xlabel("terminal commutation RMSE")
    axes[2].set_ylabel("shared clean-target RMSE")

    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _fit_pca(*arrays: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one centered particle PCA and return mean, two axes, variance ratios."""
    values = np.concatenate(arrays, axis=0)  # shape: [A,D]
    mean = np.mean(values, axis=0, keepdims=True)  # shape: [1,D]
    centered = values - mean  # shape: [A,D]
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    components = right_vectors[:2]  # shape: [2,D]
    ratio = singular_values[:2] ** 2 / max(float(np.sum(singular_values**2)), 1.0e-15)
    return mean, components, ratio


def plot_target_flow_pca(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Plot one representative target from initial plane through replay."""
    sources = arrays["sources"]  # shape: [C,P,D]
    history = arrays["history"]  # shape: [J+1,N,D]
    source_ids = arrays["target_source_ids"]  # shape: [R,S]
    clean_targets = arrays["clean_targets"]  # shape: [R,P,D]
    noisy_targets = arrays["noisy_targets"]  # shape: [R,P,D]
    terminal_candidates = arrays["terminal_candidates"]  # shape: [R,M,P,D]
    trajectory = arrays["target_trajectory"]  # shape: [J+1,R,M,P,D]
    generated = arrays["generated"]  # shape: [R,M,P,D]
    trial_kind = arrays["trial_kind"]  # shape: [R]
    shared_error = arrays["metric_clean_rmse"][:, 0]  # shape: [R]

    random_indices = np.flatnonzero(trial_kind == "random")
    median_error = np.median(shared_error[random_indices])
    trial = int(random_indices[np.argmin(np.abs(shared_error[random_indices] - median_error))])
    parents_initial = sources[source_ids[trial]]  # shape: [S,P,D]
    if "terminal_sources" in arrays:
        terminal_sources = arrays["terminal_sources"]  # shape: [C,P,D]
    elif history[-1].ndim == 3:
        terminal_sources = np.swapaxes(history[-1], 0, 1)  # legacy slot [P,C,D] -> [C,P,D]
    else:
        terminal_sources = history[-1].reshape(sources.shape)  # pooled [C*P,D] -> [C,P,D]
    parents_terminal = terminal_sources[source_ids[trial]]  # shape: [S,P,D]
    particle_count = sources.shape[1]
    dimension = sources.shape[2]

    mean, components, ratio = _fit_pca(
        _flat_memory(history[0]),
        _flat_memory(history[-1]),
        parents_initial.reshape(-1, dimension),
        parents_terminal.reshape(-1, dimension),
        clean_targets[trial],
        noisy_targets[trial],
        terminal_candidates[trial].reshape(-1, dimension),
        generated[trial].reshape(-1, dimension),
    )

    def project(values: np.ndarray) -> np.ndarray:
        """Project rows ``[A,D]`` into the fixed particle PCA basis."""
        return (values - mean) @ components.T  # [A,D] @ [D,2] -> [A,2]

    initial_pca = project(_flat_memory(history[0]))  # shape: [C*P,2]
    terminal_pca = project(_flat_memory(history[-1]))  # shape: [C*P,2]
    parent_initial_pca = project(parents_initial.reshape(-1, dimension)).reshape(
        parents_initial.shape[0], particle_count, 2
    )  # [S*P,2] -> [S,P,2]
    parent_terminal_pca = project(parents_terminal.reshape(-1, dimension)).reshape(
        parents_terminal.shape[0], particle_count, 2
    )  # [S*P,2] -> [S,P,2]
    clean_pca = project(clean_targets[trial])  # shape: [P,2]
    noisy_pca = project(noisy_targets[trial])  # shape: [P,2]
    terminal_candidate_pca = project(terminal_candidates[trial].reshape(-1, dimension)).reshape(2, particle_count, 2)
    generated_pca = project(generated[trial].reshape(-1, dimension)).reshape(2, particle_count, 2)

    stride = max(1, (trajectory.shape[0] - 1) // 250)
    replay = trajectory[::-1][::stride, trial]  # terminal-to-output, [F,M,P,D]
    if (trajectory.shape[0] - 1) % stride != 0:
        replay = np.concatenate((replay, trajectory[:1, trial]), axis=0)  # include exact output frame
    replay_pca = project(replay.reshape(-1, dimension)).reshape(
        replay.shape[0], replay.shape[1], replay.shape[2], 2
    )  # [F*M*P,2] -> [F,M,P,2]

    source_colors = tab10(np.linspace(0.0, 0.6, parents_initial.shape[0]))
    figure, axes = plt.subplots(1, 4, figsize=(17.0, 4.5), sharex=True, sharey=True, constrained_layout=True)

    axes[0].scatter(initial_pca[:, 0], initial_pca[:, 1], s=5, color="#BBBBBB", alpha=0.25)
    for source, color in enumerate(source_colors):
        points = parent_initial_pca[source]
        axes[0].scatter(points[:, 0], points[:, 1], s=24, color=color, label=f"source {source + 1}")
    axes[0].scatter(clean_pca[:, 0], clean_pca[:, 1], s=48, color="#000000", marker="X", label="clean target")
    axes[0].scatter(noisy_pca[:, 0], noisy_pca[:, 1], s=34, facecolors="none", edgecolors="#000000", label="noisy target")
    axes[0].set_title("1. Initial plane and known target")
    axes[0].legend(fontsize=7)

    axes[1].scatter(terminal_pca[:, 0], terminal_pca[:, 1], s=5, color="#BBBBBB", alpha=0.25)
    for source, color in enumerate(source_colors):
        points = parent_terminal_pca[source]
        axes[1].scatter(points[:, 0], points[:, 1], s=22, color=color)
    for method in range(2):
        points = terminal_candidate_pca[method]
        axes[1].scatter(
            points[:, 0], points[:, 1], s=45, color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method], label=("shared" if method == 0 else "independent")
        )
    axes[1].set_title("2. Terminal interpolation")
    axes[1].legend(fontsize=7)

    for method in range(2):
        for particle in range(particle_count):
            points = replay_pca[:, method, particle]
            axes[2].plot(points[:, 0], points[:, 1], color=METHOD_COLORS[method], alpha=0.65, linewidth=0.8)
        axes[2].plot([], [], color=METHOD_COLORS[method], label=("shared" if method == 0 else "independent"))
    axes[2].set_title("3. Vanilla backward replay")
    axes[2].legend(fontsize=7)

    axes[3].scatter(initial_pca[:, 0], initial_pca[:, 1], s=5, color="#BBBBBB", alpha=0.25)
    axes[3].scatter(clean_pca[:, 0], clean_pca[:, 1], s=48, color="#000000", marker="X", label="clean target")
    axes[3].scatter(noisy_pca[:, 0], noisy_pca[:, 1], s=34, facecolors="none", edgecolors="#000000", label="noisy target")
    for method in range(2):
        points = generated_pca[method]
        axes[3].scatter(
            points[:, 0], points[:, 1], s=42, color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method], label=("shared output" if method == 0 else "independent output")
        )
    axes[3].set_title("4. Output versus target")
    axes[3].legend(fontsize=7)

    x_label = f"PC1 ({100.0 * ratio[0]:.1f}% variance)"
    y_label = f"PC2 ({100.0 * ratio[1]:.1f}% variance)"
    for axis in axes:
        axis.set_xlabel(x_label)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel(y_label)
    figure.suptitle(f"Representative target trial {trial}")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_replay_movement(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Plot median particle spread and total-position movement by method."""
    trajectory = arrays["target_trajectory"]  # shape: [J+1,R,M,P,D]
    random_mask = arrays["trial_kind"] == "random"  # shape: [R]
    replay = trajectory[::-1]  # terminal-to-output, [J+1,R,M,P,D]
    particle_count = replay.shape[3]
    mean_distance = np.zeros(replay.shape[:3], dtype=np.float64)  # [J+1,R,M]
    pair_count = 0
    for first in range(particle_count):
        for second in range(first + 1, particle_count):
            difference = replay[:, :, :, first, :] - replay[:, :, :, second, :]  # [J+1,R,M,D]
            mean_distance += np.linalg.norm(difference, axis=3)  # shape: [J+1,R,M]
            pair_count += 1
    mean_distance /= pair_count

    total_position = np.sum(replay, axis=3)  # shape: [J+1,R,M,D]
    displacement = total_position[1:] - total_position[:-1]  # shape: [J,R,M,D]
    displacement_norm = np.linalg.norm(displacement, axis=3)  # shape: [J,R,M]
    steps = np.arange(replay.shape[0])

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    for method in range(2):
        values = mean_distance[:, random_mask, method]  # shape: [J+1,E]
        median = np.median(values, axis=1)
        low, high = np.quantile(values, (0.25, 0.75), axis=1)
        label = "shared" if method == 0 else "independent"
        axes[0].plot(steps, median, color=METHOD_COLORS[method], label=label)
        axes[0].fill_between(steps, low, high, color=METHOD_COLORS[method], alpha=0.15)

        movement = displacement_norm[:, random_mask, method]  # shape: [J,E]
        move_median = np.median(movement, axis=1)
        move_low, move_high = np.quantile(movement, (0.25, 0.75), axis=1)
        axes[1].plot(steps[1:], np.maximum(move_median, 1.0e-16), color=METHOD_COLORS[method], label=label)
        axes[1].fill_between(
            steps[1:], np.maximum(move_low, 1.0e-16), np.maximum(move_high, 1.0e-16),
            color=METHOD_COLORS[method], alpha=0.15
        )

    axes[0].set_title("Generated-source particle spread")
    axes[0].set_xlabel("completed backward frames")
    axes[0].set_ylabel("median mean pair distance")
    axes[1].set_title("Complete-source movement per frame")
    axes[1].set_xlabel("completed backward frames")
    axes[1].set_ylabel("median total-position displacement norm")
    axes[1].set_yscale("log")
    axes[1].yaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{value:.0e}")
    )  # plain scientific labels avoid a math-font dependency
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_all(run_directory: Path) -> list[Path]:
    """Load one saved run and regenerate every applicable figure."""
    run_directory = Path(run_directory)
    with np.load(run_directory / "samples.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}

    written: list[Path] = []
    forward_path = run_directory / "forward_smoke.png"
    plot_forward_smoke(forward_path, arrays)
    written.append(forward_path)

    if "vertex_particle_relative_rmse" in arrays or "vertex_relative_rmse" in arrays:
        vertex_path = run_directory / "vertex_replay.png"
        plot_vertex_replay(vertex_path, arrays)
        written.append(vertex_path)

    if "metric_clean_rmse" in arrays:
        target_path = run_directory / "target_recovery.png"
        pca_path = run_directory / "target_flow_pca.png"
        movement_path = run_directory / "replay_movement.png"
        plot_target_recovery(target_path, arrays)
        plot_target_flow_pca(pca_path, arrays)
        plot_replay_movement(movement_path, arrays)
        written.extend((target_path, pca_path, movement_path))
    return written


def plot_grid_summary(search_directory: Path) -> Path:
    """Regenerate the calibration overview from the two append-only CSV files."""
    search_directory = Path(search_directory)
    with (search_directory / "forward_grid.csv").open(newline="", encoding="utf-8") as handle:
        forward_rows = list(csv.DictReader(handle))
    with (search_directory / "replay_grid.csv").open(newline="", encoding="utf-8") as handle:
        replay_rows = list(csv.DictReader(handle))

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)

    potential = [
        row for row in forward_rows
        if row.get("stage") == "potential" and math.isfinite(float(row["worst_radial_cdf_gap"]))
    ]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in potential:
        grouped.setdefault((row["epsilon"], row["exponent_s"]), []).append(row)
    for (epsilon, exponent_s), rows in grouped.items():
        rows.sort(key=lambda row: float(row["tau"]))
        axes[0, 0].plot(
            [float(row["tau"]) for row in rows],
            [float(row["worst_radial_cdf_gap"]) for row in rows],
            alpha=0.35,
            linewidth=0.9,
            label=f"eps={epsilon}, s={exponent_s}",
        )
    axes[0, 0].set_title("Potential screen: descriptive radial gap")
    axes[0, 0].set_xlabel("transport time tau")
    axes[0, 0].set_ylabel("worst field CDF gap")
    if len(grouped) <= 8:
        axes[0, 0].legend(fontsize=6)

    validation = [
        row for row in forward_rows
        if row.get("stage") == "density_validation"
        and math.isfinite(float(row["worst_radial_cdf_gap"]))
    ]
    for layout, color in (("slot", "#377EB8"), ("single", "#E41A1C")):
        rows = [row for row in validation if row.get("layout") == layout]
        axes[0, 1].scatter(
            [float(row["runtime_seconds"]) for row in rows],
            [float(row["worst_radial_cdf_gap"]) for row in rows],
            color=color,
            label=layout,
            alpha=0.75,
        )
    axes[0, 1].set_title("Density validation")
    axes[0, 1].set_xlabel("forward runtime (s)")
    axes[0, 1].set_ylabel("worst field CDF gap")
    axes[0, 1].legend(fontsize=8)

    screen = [
        row for row in replay_rows
        if row.get("stage") == "screen" and row.get("particle_p95_relative_rmse")
        and math.isfinite(float(row["particle_p95_relative_rmse"]))
    ]
    for beta in sorted({row.get("beta", "") for row in screen}):
        rows = [row for row in screen if row.get("beta") == beta]
        axes[1, 0].scatter(
            [float(row["proximal_steps"]) for row in rows],
            [float(row["particle_p95_relative_rmse"]) for row in rows],
            label=f"beta={beta}",
            alpha=0.65,
        )
    axes[1, 0].axhline(0.10, color="black", linestyle="--", label="vertex gate")
    axes[1, 0].set_title("Replay screen")
    axes[1, 0].set_xlabel("proximal iterations T")
    axes[1, 0].set_ylabel("particle p95 relative RMSE")
    axes[1, 0].legend(fontsize=7)

    confirmation = [
        row for row in replay_rows
        if row.get("stage") == "confirmation" and row.get("particle_p95_relative_rmse")
        and math.isfinite(float(row["particle_p95_relative_rmse"]))
    ]
    labels = [f"{row['layout']} s{row['seed']}" for row in confirmation]
    values = [float(row["particle_p95_relative_rmse"]) for row in confirmation]
    if values:
        axes[1, 1].bar(np.arange(len(values)), values, color="#4DAF4A")
        axes[1, 1].set_xticks(np.arange(len(values)), labels, rotation=45, ha="right", fontsize=7)
    else:
        axes[1, 1].text(0.5, 0.5, "no confirmation rows", ha="center", va="center")
    axes[1, 1].axhline(0.10, color="black", linestyle="--")
    axes[1, 1].set_title("Replay confirmation")
    axes[1, 1].set_ylabel("particle p95 relative RMSE")

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    output = search_directory / "grid_summary.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    """Regenerate either one H4.5 run or one calibration-search figure."""
    parser = argparse.ArgumentParser(description="Regenerate H4.5 figures from saved arrays or grid CSV files.")
    parser.add_argument("run_directory", type=Path, help="Timestamped run or search directory.")
    directory = parser.parse_args().run_directory
    if (directory / "samples.npz").exists():
        for path in plot_all(directory):
            print(path.resolve())
    elif (directory / "forward_grid.csv").exists() and (directory / "replay_grid.csv").exists():
        print(plot_grid_summary(directory).resolve())
    else:
        raise FileNotFoundError("directory contains neither samples.npz nor grid CSV files")


if __name__ == "__main__":
    main()

"""Run the single-plane shared-lambda Hypothesis 4.5 experiment.

Purpose:
    Group one exact EFS plane into complete sources, select four sources, create
    an eight-particle chimera with one shared simplex vector, and replay shared
    and independently weighted controls together through the same history.

Dependencies:
    NumPy and Matplotlib from req.txt.

Outputs:
    A new timestamped directory containing arrays, metrics, logs, and figures.

Exact command:
    python run_single_plane_h45.py --output-root results
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps

from efs_single_plane import backward_replay, forward_history

tab10 = colormaps["tab10"]
Set1 = colormaps["Set1"]


def generate_grouped_sources(
    source_count: int,
    particles_per_source: int,
    dimension: int,
    center_std: float,
    particle_scale: float,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create complete sources shaped ``[C,P,D]`` in one common coordinate plane.

    The synthetic equation is

        S_(c,p) = center_c + template_p + noise_(c,p).

    A source-wide center couples its eight particles. The fixed particle template
    supplies correspondence between source particles without rotations, families,
    retrieval, or separate EFS planes.

    Returns:
        ``(sources, template)`` shaped ``[C,P,D]`` and ``[P,D]``.
    """
    if source_count < 4 or particles_per_source < 2 or dimension < 2:
        raise ValueError("need at least four sources, two particles per source, and D >= 2")
    if center_std <= 0.0 or particle_scale <= 0.0 or noise_std < 0.0:
        raise ValueError("scales must be positive and noise_std must be non-negative")

    rng = np.random.default_rng(seed)
    centers = rng.normal(
        scale=center_std, size=(source_count, dimension)
    )  # shape: [C,D], one latent state shared by a complete source
    template = rng.normal(
        scale=particle_scale, size=(particles_per_source, dimension)
    )  # shape: [P,D], fixed particle identity/correspondence
    template -= np.mean(template, axis=0, keepdims=True)  # [P,D] - [1,D] -> [P,D]
    noise = rng.normal(scale=noise_std, size=(source_count, particles_per_source, dimension))  # shape: [C,P,D]
    sources = centers[:, None, :] + template[None, :, :] + noise  # [C,1,D] + [1,P,D] + [C,P,D] -> [C,P,D]
    return sources, template


def parse_shared_lambda(text: str | None, selected_source_count: int) -> np.ndarray | None:
    """Parse an optional operator simplex vector shaped ``[S]``."""
    if text is None:
        return None

    values = np.asarray([float(value.strip()) for value in text.split(",")], dtype=np.float64)
    if values.shape != (selected_source_count,):
        raise ValueError(f"--shared-lambda requires exactly {selected_source_count} comma-separated values")
    if np.any(values < 0.0) or not np.isclose(np.sum(values), 1.0, atol=1.0e-8):
        raise ValueError("--shared-lambda values must be non-negative and sum to one")
    return values / np.sum(values)


def build_terminal_ensembles(
    selected_terminal: np.ndarray,
    random_shared_lambda: np.ndarray,
    independent_lambdas: np.ndarray,
    operator_lambda: np.ndarray | None,
) -> tuple[list[str], np.ndarray]:
    """Construct every complete candidate and batch them as ``[G,P,D]``.

    Shared interpolation implements

        Y_(p,d) = sum_s lambda_s source_(s,p,d).

    The independent control replaces ``lambda_s`` with ``lambda_(p,s)``. It is
    only a falsification control, not a proposed operator input.
    """
    labels: list[str] = []
    candidates: list[np.ndarray] = []

    if operator_lambda is not None:
        operator_candidate = np.einsum(
            "s,spd->pd", operator_lambda, selected_terminal
        )  # [S] with [S,P,D], sum over selected-source axis S -> [P,D]
        labels.append("shared_operator")
        candidates.append(operator_candidate)

    random_shared = np.einsum(
        "s,spd->pd", random_shared_lambda, selected_terminal
    )  # one [S] vector is reused for every corresponding particle p -> [P,D]
    labels.append("shared_random_reference")
    candidates.append(random_shared)

    independent = np.einsum(
        "ps,spd->pd", independent_lambdas, selected_terminal
    )  # particle p receives its own [S] control vector; source axis S is reduced
    labels.append("independent_random_control")
    candidates.append(independent)

    return labels, np.stack(candidates, axis=0)  # shape: [G,P,D]


def template_coherence_error(ensembles: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Measure source-wide state disagreement after removing particle identity.

    For each candidate, ``ensemble-template`` should leave one common source
    center. The returned RMS deviation has shape ``[G]``; lower is better.
    """
    adjusted = ensembles - template[None, :, :]  # [G,P,D] - [1,P,D] -> [G,P,D]
    source_center = np.mean(adjusted, axis=1, keepdims=True)  # shape: [G,1,D]
    squared_error = (adjusted - source_center) ** 2  # shape: [G,P,D]
    return np.sqrt(np.mean(squared_error, axis=(1, 2)))  # shape: [G]


def pair_shape_error(ensembles: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Compare each source's internal particle distances with the template."""
    source_difference = ensembles[:, :, None, :] - ensembles[:, None, :, :]  # [G,P,1,D] - [G,1,P,D] -> [G,P,P,D]
    source_distance = np.linalg.norm(source_difference, axis=3)  # shape: [G,P,P]
    template_difference = template[:, None, :] - template[None, :, :]  # [P,P,D]
    template_distance = np.linalg.norm(template_difference, axis=2)  # shape: [P,P]
    return np.mean(np.abs(source_distance - template_distance[None, :, :]), axis=(1, 2))  # shape: [G]


def nearest_source_distance(ensembles: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Return flattened complete-source distance to the nearest memory source."""
    generated_flat = ensembles.reshape(ensembles.shape[0], -1)  # [G,P,D] -> [G,P*D]
    source_flat = sources.reshape(sources.shape[0], -1)  # [C,P,D] -> [C,P*D]
    difference = generated_flat[:, None, :] - source_flat[None, :, :]  # [G,1,P*D] - [1,C,P*D] -> [G,C,P*D]
    squared_distance = np.sum(difference * difference, axis=2)  # shape: [G,C]
    return np.sqrt(np.min(squared_distance, axis=1))  # shape: [G]


def fit_pca(*arrays: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit centered particle-level PCA and return mean, two axes, variance ratios."""
    fit_data = np.concatenate(arrays, axis=0)  # shape: [R,D], particles stay separate
    mean = np.mean(fit_data, axis=0, keepdims=True)  # shape: [1,D]
    centered = fit_data - mean  # shape: [R,D]
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    components = right_vectors[:2]  # shape: [2,D]
    ratio = singular_values[:2] ** 2 / np.sum(singular_values**2)  # shape: [2]
    return mean, components, ratio


def plot_particle_pca(
    path: Path,
    sources: np.ndarray,
    history: np.ndarray,
    source_ids: np.ndarray,
    selected_terminal: np.ndarray,
    terminal_ensembles: np.ndarray,
    trajectory: np.ndarray,
    labels: list[str],
) -> None:
    """Plot the plane, four source ensembles, interpolation, and replay in one PCA."""
    initial_plane = history[0]  # shape: [N,D]
    terminal_plane = history[-1]  # shape: [N,D]
    output_ensembles = trajectory[0]  # shape: [G,P,D]
    source_count, particles_per_source, dimension = sources.shape

    mean, components, ratio = fit_pca(
        initial_plane,
        terminal_plane,
        selected_terminal.reshape(-1, dimension),
        terminal_ensembles.reshape(-1, dimension),
        output_ensembles.reshape(-1, dimension),
    )

    def project(values: np.ndarray) -> np.ndarray:
        """Project rows ``[R,D]`` through the one fixed particle PCA basis."""
        return (values - mean) @ components.T  # [R,D] @ [D,2] -> [R,2]

    initial_pca = project(initial_plane)  # shape: [N,2]
    terminal_pca = project(terminal_plane)  # shape: [N,2]
    selected_initial = sources[source_ids]  # shape: [S,P,D]
    selected_initial_pca = project(selected_initial.reshape(-1, dimension)).reshape(
        selected_initial.shape[0], particles_per_source, 2
    )  # [S*P,2] -> [S,P,2]
    selected_terminal_pca = project(selected_terminal.reshape(-1, dimension)).reshape(
        selected_terminal.shape[0], particles_per_source, 2
    )  # [S*P,2] -> [S,P,2]
    candidate_terminal_pca = project(terminal_ensembles.reshape(-1, dimension)).reshape(
        terminal_ensembles.shape[0], particles_per_source, 2
    )  # [G*P,2] -> [G,P,2]

    selected_flat_ids = (source_ids[:, None] * particles_per_source + np.arange(particles_per_source)[None, :]).reshape(
        -1
    )  # [S,1] + [1,P] -> [S,P] -> [S*P]
    forward_stride = max(1, (history.shape[0] - 1) // 250)
    forward_frames = history[::forward_stride, selected_flat_ids, :]  # [F,S*P,D]
    forward_pca = project(forward_frames.reshape(-1, dimension)).reshape(
        forward_frames.shape[0], selected_flat_ids.size, 2
    )  # [F*S*P,2] -> [F,S*P,2]

    replay = trajectory[::-1]  # [J+1,G,P,D], terminal to generated output
    replay_pca = project(replay.reshape(-1, dimension)).reshape(
        replay.shape[0], replay.shape[1], replay.shape[2], 2
    )  # [(J+1)*G*P,2] -> [J+1,G,P,2]

    source_colors = tab10(np.linspace(0.0, 0.6, selected_terminal.shape[0]))
    method_colors = Set1(np.linspace(0.0, 0.7, terminal_ensembles.shape[0]))
    figure, axes = plt.subplots(1, 4, figsize=(17.0, 4.5), sharex=True, sharey=True, constrained_layout=True)

    axes[0].scatter(initial_pca[:, 0], initial_pca[:, 1], s=5, color="#AAAAAA", alpha=0.35)
    for source, color in enumerate(source_colors):
        points = selected_initial_pca[source]  # shape: [P,2]
        axes[0].scatter(points[:, 0], points[:, 1], s=30, color=color, label=f"source {source + 1}")
    axes[0].set_title("1. One base EFS plane")
    axes[0].legend(fontsize=7)

    axes[1].scatter(terminal_pca[:, 0], terminal_pca[:, 1], s=5, color="#AAAAAA", alpha=0.35)
    for path_index in range(selected_flat_ids.size):
        path_points = forward_pca[:, path_index, :]  # shape: [F,2]
        source = path_index // particles_per_source
        axes[1].plot(path_points[:, 0], path_points[:, 1], color=source_colors[source], alpha=0.55, linewidth=0.7)
    axes[1].set_title("2. Selected-particle forward paths")

    for source, color in enumerate(source_colors):
        points = selected_terminal_pca[source]  # shape: [P,2]
        axes[2].scatter(points[:, 0], points[:, 1], s=25, color=color, marker="o")
    for group, color in enumerate(method_colors):
        points = candidate_terminal_pca[group]  # shape: [P,2]
        axes[2].scatter(points[:, 0], points[:, 1], s=48, color=color, marker="X", label=labels[group])
    # Draw each corresponding source particle toward the first shared candidate.
    for particle in range(particles_per_source):
        target = candidate_terminal_pca[0, particle]  # shape: [2]
        for source in range(selected_terminal.shape[0]):
            origin = selected_terminal_pca[source, particle]  # shape: [2]
            axes[2].plot([origin[0], target[0]], [origin[1], target[1]], color="#777777", alpha=0.25, linewidth=0.6)
    axes[2].set_title("3. Four-source particle interpolation")
    axes[2].legend(fontsize=7)

    axes[3].scatter(initial_pca[:, 0], initial_pca[:, 1], s=5, color="#BBBBBB", alpha=0.25)
    for group, color in enumerate(method_colors):
        for particle in range(particles_per_source):
            points = replay_pca[:, group, particle, :]  # shape: [J+1,2]
            axes[3].plot(points[:, 0], points[:, 1], color=color, alpha=0.7, linewidth=0.8)
            axes[3].scatter(points[0, 0], points[0, 1], marker="x", s=22, color=color)
            axes[3].scatter(points[-1, 0], points[-1, 1], marker="o", s=22, color=color)
        axes[3].plot([], [], color=color, label=labels[group])
    axes[3].set_title("4. Eight-particle backward replay")
    axes[3].legend(fontsize=7)

    x_label = f"PC1 ({100.0 * ratio[0]:.1f}% variance)"
    y_label = f"PC2 ({100.0 * ratio[1]:.1f}% variance)"
    for axis in axes:
        axis.set_xlabel(x_label)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel(y_label)
    figure.suptitle("Single-plane H4.5 particle flow")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def trajectory_diagnostics(trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return full-frame internal distance and total displacement norm.

    Input trajectory is ``[J+1,G,P,D]`` stored output-to-terminal. Results are
    returned in the actual replay direction, terminal-to-output.
    """
    replay = trajectory[::-1]  # shape: [J+1,G,P,D]
    particle_count = replay.shape[2]
    difference = replay[:, :, :, None, :] - replay[:, :, None, :, :]  # [J+1,G,P,1,D] - [J+1,G,1,P,D] -> [J+1,G,P,P,D]
    distance = np.linalg.norm(difference, axis=4)  # shape: [J+1,G,P,P]
    mean_distance = np.sum(distance, axis=(2, 3)) / float(particle_count * (particle_count - 1))  # shape: [J+1,G]

    total_position = np.sum(replay, axis=2)  # shape: [J+1,G,D]
    displacement = total_position[1:] - total_position[:-1]  # shape: [J,G,D]
    displacement_norm = np.linalg.norm(displacement, axis=2)  # shape: [J,G]
    return mean_distance, displacement_norm


def plot_replay_diagnostics(path: Path, trajectory: np.ndarray, labels: list[str]) -> None:
    """Plot within-source spread and one-frame total displacement for each method."""
    mean_distance, displacement_norm = trajectory_diagnostics(trajectory)
    steps = np.arange(mean_distance.shape[0])  # shape: [J+1]
    colors = Set1(np.linspace(0.0, 0.7, len(labels)))
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)

    for group, (label, color) in enumerate(zip(labels, colors)):
        axes[0].plot(steps, mean_distance[:, group], color=color, label=label)
        axes[1].plot(steps[1:], np.maximum(displacement_norm[:, group], 1.0e-16), color=color, label=label)
    axes[0].set_title("Generated-source particle spread")
    axes[0].set_xlabel("completed backward frames")
    axes[0].set_ylabel("mean pairwise distance")
    axes[1].set_title("Complete-source movement per frame")
    axes[1].set_xlabel("completed backward frames")
    axes[1].set_ylabel("norm of total-position displacement")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_method_comparison(
    path: Path,
    labels: list[str],
    coherence: np.ndarray,
    shape_error: np.ndarray,
    nearest: np.ndarray,
    coherence_threshold: float,
    shape_threshold: float,
) -> None:
    """Plot the three complete-source diagnostics used in the summary."""
    colors = Set1(np.linspace(0.0, 0.7, len(labels)))
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), constrained_layout=True)
    values = (coherence, shape_error, nearest)
    titles = ("Template coherence", "Internal shape error", "Nearest memory source")
    thresholds = (coherence_threshold, shape_threshold, None)

    for axis, metric, title, threshold in zip(axes, values, titles, thresholds):
        axis.bar(labels, metric, color=colors)
        if threshold is not None:
            axis.axhline(threshold, color="black", linestyle="--", label="training 95%")
            axis.legend(fontsize=8)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_metrics_csv(
    path: Path,
    labels: list[str],
    coherence: np.ndarray,
    shape_error: np.ndarray,
    nearest: np.ndarray,
    residual: np.ndarray,
    valid: np.ndarray,
) -> None:
    """Write one row per generated complete source."""
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "method",
                "template_coherence_error",
                "pair_shape_error",
                "nearest_source_distance",
                "max_replay_residual",
                "raw_valid",
            ),
        )
        writer.writeheader()
        for index, label in enumerate(labels):
            writer.writerow({
                "method": label,
                "template_coherence_error": f"{coherence[index]:.9e}",
                "pair_shape_error": f"{shape_error[index]:.9e}",
                "nearest_source_distance": f"{nearest[index]:.9e}",
                "max_replay_residual": f"{residual[index]:.9e}",
                "raw_valid": int(valid[index]),
            })


def run(args: argparse.Namespace) -> Path:
    """Run one exact shared plane and batch all H4.5 candidates through replay."""
    if args.plane_particles % args.particles_per_source != 0:
        raise ValueError("--plane-particles must be divisible by --particles-per-source")
    source_count = args.plane_particles // args.particles_per_source
    if source_count < args.selected_sources:
        raise ValueError("the plane does not contain enough complete sources")

    exponent_s = float(args.dimension - 2 if args.exponent_s is None else args.exponent_s)
    history_gib = (args.forward_steps + 1) * args.plane_particles * args.dimension * np.dtype(np.float64).itemsize / 1024**3
    scalar_pair_mib = args.plane_particles**2 * np.dtype(np.float64).itemsize / 1024**2
    print(f"single EFS plane: N={args.plane_particles}, D={args.dimension}, ordered pairs/frame={args.plane_particles**2:,}")
    print(
        f"complete sources: C={source_count}, particles/source={args.particles_per_source}; "
        f"history≈{history_gib:.2f} GiB, each scalar pair matrix≈{scalar_pair_mib:.1f} MiB"
    )

    sources, template = generate_grouped_sources(
        source_count=source_count,
        particles_per_source=args.particles_per_source,
        dimension=args.dimension,
        center_std=args.center_std,
        particle_scale=args.particle_scale,
        noise_std=args.noise_std,
        seed=args.seed,
    )  # sources: [C,P,D], template: [P,D]
    initial_plane = sources.reshape(
        args.plane_particles, args.dimension
    )  # [C,P,D] -> [C*P,D], group and particle identities retain row order

    start = time.perf_counter()
    history, forward_log_step, forward_log_distance = forward_history(
        initial_plane,
        gamma=args.gamma,
        epsilon=args.epsilon,
        exponent_s=exponent_s,
        steps=args.forward_steps,
        log_every=args.log_every,
    )  # history: [J+1,N,D]
    forward_seconds = time.perf_counter() - start

    source_rng = np.random.default_rng(args.seed + 1)
    source_ids = source_rng.choice(
        source_count, size=args.selected_sources, replace=False
    )  # shape: [S], complete source IDs selected without locality filtering
    terminal_grouped = history[-1].reshape(
        source_count, args.particles_per_source, args.dimension
    )  # [N,D] -> [C,P,D], restoring complete-source membership
    selected_terminal = terminal_grouped[source_ids]  # shape: [S,P,D]

    weight_rng = np.random.default_rng(args.seed + 2)
    random_shared_lambda = weight_rng.dirichlet(
        np.ones(args.selected_sources, dtype=np.float64)
    )  # shape: [S], one witness vector shared by all P particles
    independent_lambdas = weight_rng.dirichlet(
        np.ones(args.selected_sources, dtype=np.float64), size=args.particles_per_source
    )  # shape: [P,S], experimental falsification control only
    operator_lambda = parse_shared_lambda(args.shared_lambda, args.selected_sources)
    labels, terminal_ensembles = build_terminal_ensembles(
        selected_terminal, random_shared_lambda, independent_lambdas, operator_lambda
    )  # terminal_ensembles: [G,P,D]

    print("\ncreation: selected terminal sources")
    print(f"source IDs: {source_ids.tolist()}")
    for source, source_id in enumerate(source_ids):
        print(
            f"source {source + 1}, ID={source_id}:\n"
            f"{np.array2string(selected_terminal[source], precision=6, suppress_small=True)}"
        )
    if operator_lambda is not None:
        print(f"operator shared lambda: {np.array2string(operator_lambda, precision=8)}")
    print(f"random shared reference lambda: {np.array2string(random_shared_lambda, precision=8)}")
    print(f"independent control lambdas [particle, source]:\n{np.array2string(independent_lambdas, precision=8)}")
    for group, label in enumerate(labels):
        print(
            f"interpolated terminal source, {label}:\n"
            f"{np.array2string(terminal_ensembles[group], precision=6, suppress_small=True)}"
        )

    start = time.perf_counter()
    trajectory, replay_residual, backward_logs = backward_replay(
        terminal_ensembles,
        history,
        gamma=args.gamma,
        epsilon=args.epsilon,
        exponent_s=exponent_s,
        beta=args.beta,
        proximal_steps=args.proximal_steps,
        labels=labels,
        log_every=args.log_every,
    )  # [J+1,G,P,D], [J,G,P], periodic complete-frame diagnostics
    backward_seconds = time.perf_counter() - start
    generated = trajectory[0]  # shape: [G,P,D], replayed complete sources

    training_coherence = template_coherence_error(sources, template)  # shape: [C]
    training_shape = pair_shape_error(sources, template)  # shape: [C]
    coherence_threshold = float(np.quantile(training_coherence, 0.95))
    shape_threshold = float(np.quantile(training_shape, 0.95))
    coherence = template_coherence_error(generated, template)  # shape: [G]
    shape_error = pair_shape_error(generated, template)  # shape: [G]
    nearest = nearest_source_distance(generated, sources)  # shape: [G]
    max_residual = np.max(replay_residual, axis=(0, 2))  # [J,G,P] -> [G]
    valid = (coherence <= coherence_threshold) & (shape_error <= shape_threshold)  # [G]

    arrays = (sources, history, terminal_ensembles, trajectory, replay_residual)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise FloatingPointError("experiment produced non-finite values")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_directory = output_root / f"single_plane_{timestamp}_seed{args.seed}"
    run_directory.mkdir(exist_ok=False)

    config = vars(args).copy()
    config.update({
        "source_count": source_count,
        "source_ids": source_ids.tolist(),
        "random_shared_lambda": random_shared_lambda.tolist(),
        "operator_lambda_values": (operator_lambda.tolist() if operator_lambda is not None else None),
        "independent_control_lambdas": independent_lambdas.tolist(),
        "exponent_s_actual": exponent_s,
        "method_labels": labels,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "coherence_threshold_training_95": coherence_threshold,
        "shape_threshold_training_95": shape_threshold,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    with (run_directory / "config.json").open("x", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)

    # Uncompressed NPZ avoids spending additional CPU time compressing a roughly
    # 0.31 GiB default history. The timestamped directory prevents overwrite.
    np.savez(
        run_directory / "samples.npz",
        sources=sources,
        template=template,
        initial_plane=initial_plane,
        history=history,
        source_ids=source_ids,
        selected_terminal=selected_terminal,
        random_shared_lambda=random_shared_lambda,
        operator_lambda=(operator_lambda if operator_lambda is not None else np.empty(0, dtype=np.float64)),
        independent_lambdas=independent_lambdas,
        method_labels=np.asarray(labels, dtype="U40"),
        terminal_ensembles=terminal_ensembles,
        trajectory=trajectory,
        replay_residual=replay_residual,
        forward_log_step=forward_log_step,
        forward_log_distance=forward_log_distance,
        backward_log_step=backward_logs["reverse_step"],
        backward_log_mean_distance=backward_logs["mean_pair_distance"],
        backward_log_total_position=backward_logs["total_position"],
        backward_log_displacement=backward_logs["displacement"],
        generated=generated,
    )
    write_metrics_csv(run_directory / "metrics.csv", labels, coherence, shape_error, nearest, max_residual, valid)

    summary = [
        "Single-plane Hypothesis 4.5 witness run",
        "",
        f"plane particles: {args.plane_particles}",
        f"complete memory sources: {source_count}",
        f"selected source IDs: {source_ids.tolist()}",
        f"random shared reference lambda: {random_shared_lambda.tolist()}",
        f"operator shared lambda: {operator_lambda.tolist() if operator_lambda is not None else 'not supplied'}",
        f"independent control lambdas: {independent_lambdas.tolist()}",
        f"forward runtime: {forward_seconds:.2f} s",
        f"backward runtime: {backward_seconds:.2f} s",
        "",
    ]
    for index, label in enumerate(labels):
        summary.extend((
            label,
            f"  template coherence error: {coherence[index]:.8e}",
            f"  pair shape error: {shape_error[index]:.8e}",
            f"  nearest memory source: {nearest[index]:.8e}",
            f"  max replay residual: {max_residual[index]:.8e}",
            f"  raw valid: {bool(valid[index])}",
        ))
    candidate_description = (
        "This run contains one operator input, one seeded shared witness, and one independently weighted falsification control."
        if operator_lambda is not None
        else "This run contains one seeded shared witness and one independently weighted falsification control."
    )
    summary.extend(("", candidate_description, "It is descriptive, not a replicated statistical verification of H4.5."))
    with (run_directory / "summary.txt").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(summary) + "\n")

    plot_particle_pca(
        run_directory / "pca_particle_flow.png",
        sources,
        history,
        source_ids,
        selected_terminal,
        terminal_ensembles,
        trajectory,
        labels,
    )
    plot_replay_diagnostics(run_directory / "replay_movement.png", trajectory, labels)
    plot_method_comparison(
        run_directory / "method_comparison.png", labels, coherence, shape_error, nearest, coherence_threshold, shape_threshold
    )
    return run_directory


def build_parser() -> argparse.ArgumentParser:
    """Define the direct single-plane research parameters."""
    parser = argparse.ArgumentParser(description="Run four-source shared-lambda interpolation on one exact EFS plane.")
    parser.add_argument("--output-root", default="results", help="Parent directory for a new timestamped run.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for data, source selection, and lambdas.")
    parser.add_argument("--plane-particles", type=int, default=2048, help="Total particles N in the one shared EFS plane.")
    parser.add_argument("--particles-per-source", type=int, default=8, help="Corresponding particles P in every complete source.")
    parser.add_argument("--selected-sources", type=int, default=4, help="Complete sources S used in the interpolation.")
    parser.add_argument("--dimension", type=int, default=4, help="Dimension D of every plane particle.")
    parser.add_argument("--center-std", type=float, default=2.0, help="Spread of complete-source centers.")
    parser.add_argument("--particle-scale", type=float, default=1.0, help="Spread of the fixed particle-identity template.")
    parser.add_argument("--noise-std", type=float, default=0.05, help="Independent source-particle noise standard deviation.")
    parser.add_argument("--gamma", type=float, default=0.0005, help="Forward Euler step and backward field scale.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Positive inverse-power regularizer.")
    parser.add_argument("--exponent-s", type=float, default=None, help="EFS exponent s; default is D-2.")
    parser.add_argument("--forward-steps", type=int, default=5000, help="Number J of exact forward frames.")
    parser.add_argument("--beta", type=float, default=0.05, help="Backward fixed-point update size.")
    parser.add_argument("--proximal-steps", type=int, default=100, help="Inner replay iterations T per frame.")
    parser.add_argument(
        "--log-every", type=int, default=10, help="Print complete forward/backward diagnostics every this many frames."
    )
    parser.add_argument(
        "--shared-lambda", default=None, help="Optional operator weights, for example 0.4,0.3,0.2,0.1; must sum to one."
    )
    return parser


def main() -> None:
    """Parse arguments, run both candidates together, and print the result path."""
    run_directory = run(build_parser().parse_args())
    print(f"results: {run_directory.resolve()}")


if __name__ == "__main__":
    main()

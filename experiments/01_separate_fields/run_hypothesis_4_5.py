"""Run the minimal shared-lambda Hypothesis 4.5 experiment.

Exact command:
    python run_hypothesis_4_5.py --output-root results

This script deliberately tests only one variable: whether the same two-source
barycentric coefficient is shared across all ordered slots or chosen
independently per slot. It uses one synthetic population, one topology, one
coordinate gauge, identical source complexes, and identical EFS histories.
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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from efs import backward_replay, forward_history


def generate_complete_chains(
    particle_count: int,
    slot_count: int,
    slot_spacing: float,
    curve_amplitude: float,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one population of intact, ordered, mutually coherent chains.

    Every complete complex has one scalar state ``t_i`` shared by all slots:

        x_(i,l) = t_i + noise
        y_(i,l) = l * spacing + amplitude * sin(t_i) + noise.

    The common ``t_i`` is the simplest possible whole-complex dependency.
    Shared lambda should keep all slots at one interpolated state, while
    independent per-slot lambdas create a chain assembled from different states.

    Args:
        particle_count: Number of complete source complexes ``N``.
        slot_count: Ordered slots ``L`` in every complex.
        slot_spacing: Expected vertical distance between adjacent slots.
        curve_amplitude: Common nonlinear vertical displacement controlled by t.
        noise_std: Independent Gaussian noise added to every coordinate.
        seed: NumPy random seed.

    Returns:
        ``(complexes, states)`` with shapes ``[N,L,D=2]`` and ``[N]``.
    """
    if particle_count < 3:
        raise ValueError("particle_count must be at least three")
    if slot_count < 2:
        raise ValueError("slot_count must be at least two")
    if slot_spacing <= 0.0 or noise_std < 0.0:
        raise ValueError("slot_spacing must be positive and noise_std non-negative")

    rng = np.random.default_rng(seed)
    states = np.sort(rng.uniform(-2.0, 2.0, size=particle_count))  # shape: [N]
    slot_index = np.arange(slot_count, dtype=np.float64)[None, :]  # shape: [1,L]

    shared_x = states[:, None]  # shape: [N,1], broadcasts over ordered slot axis L
    shared_curve = curve_amplitude * np.sin(states)[:, None]  # shape: [N,1]
    x_coordinate = shared_x + noise_std * rng.normal(size=(particle_count, slot_count))  # shape: [N,L]
    y_coordinate = (
        slot_spacing * slot_index + shared_curve + noise_std * rng.normal(size=(particle_count, slot_count))
    )  # [1,L] + [N,1] + [N,L] broadcasts to [N,L]

    complexes = np.stack(
        (x_coordinate, y_coordinate),
        axis=2,
    )  # shape: [N,L,D=2], final axis stores x and y
    return complexes, states


def select_source_ids(
    states: np.ndarray,
    source_low: float,
    source_high: float,
) -> tuple[int, int]:
    """Select two intact complexes nearest the requested scalar states."""
    source_a = int(np.argmin(np.abs(states - source_low)))
    source_b = int(np.argmin(np.abs(states - source_high)))
    if source_a == source_b:
        raise ValueError("source-low and source-high selected the same complex")
    return source_a, source_b


def make_lambda_controls(
    candidate_count: int,
    slot_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create matched shared and independent two-source coefficient grids.

    The shared matrix repeats one ``u`` along every slot. The independent matrix
    circularly shifts the same grid for each slot. Thus every slot sees exactly
    the same marginal coefficient values, and only cross-slot synchronization
    changes.

    Args:
        candidate_count: Number of interior two-source coefficients ``B``.
        slot_count: Number of ordered slots ``L``.

    Returns:
        ``(u, shared, independent)`` with shapes ``[B]``, ``[B,L]``, ``[B,L]``.
    """
    if candidate_count < 3 or candidate_count % 2 == 0:
        raise ValueError("candidate_count must be odd and at least three")

    u = np.linspace(0.1, 0.9, candidate_count, dtype=np.float64)  # shape: [B]
    shared = np.broadcast_to(
        u[:, None],
        (candidate_count, slot_count),
    ).copy()  # [B,1] broadcasts to [B,L]

    independent = np.empty((candidate_count, slot_count), dtype=np.float64)
    shift = max(1, candidate_count // slot_count)
    for slot in range(slot_count):
        independent[:, slot] = np.roll(u, slot * shift)
    return u, shared, independent


def barycentric_terminal_seeds(
    terminal_a: np.ndarray,
    terminal_b: np.ndarray,
    lambdas: np.ndarray,
) -> np.ndarray:
    """Apply two-source coefficients synchronously or independently by slot.

    The equation is

        y_(b,l)^K = lambda_(b,l) * x_(A,l)^K
                    + (1-lambda_(b,l)) * x_(B,l)^K.

    Args:
        terminal_a: Complete source A at terminal frame, shape ``[L,D]``.
        terminal_b: Complete source B at terminal frame, shape ``[L,D]``.
        lambdas: Candidate coefficients with shape ``[B,L]``.

    Returns:
        Terminal candidate complexes with shape ``[B,L,D]``.
    """
    coefficient = lambdas[:, :, None]  # shape: [B,L,1], broadcasts over D
    source_a = terminal_a[None, :, :]  # shape: [1,L,D], broadcasts over B
    source_b = terminal_b[None, :, :]  # shape: [1,L,D], broadcasts over B
    return coefficient * source_a + (1.0 - coefficient) * source_b


def coherence_error(samples: np.ndarray) -> np.ndarray:
    """Measure whether all slots still encode one shared scalar state.

    In the toy distribution the x coordinate is the shared state. The error is
    the RMS deviation of slot x coordinates from their within-complex mean.

    Args:
        samples: Complete complexes with shape ``[B,L,D=2]``.

    Returns:
        Coherence errors with shape ``[B]``. Lower is better.
    """
    slot_state = samples[:, :, 0]  # shape: [B,L], x coordinate only
    complex_state = np.mean(
        slot_state,
        axis=1,
        keepdims=True,
    )  # shape: [B,1], broadcasts over L
    squared_deviation = (slot_state - complex_state) ** 2  # shape: [B,L]
    return np.sqrt(np.mean(squared_deviation, axis=1))  # shape: [B]


def bond_error(samples: np.ndarray, slot_spacing: float) -> np.ndarray:
    """Measure adjacent-slot distance error relative to the declared topology.

    Args:
        samples: Complete complexes with shape ``[B,L,D=2]``.
        slot_spacing: Expected distance between adjacent ordered slots.

    Returns:
        Mean absolute bond-length error with shape ``[B]``.
    """
    bond_vectors = samples[:, 1:, :] - samples[:, :-1, :]  # shape: [B,L-1,D]
    bond_lengths = np.linalg.norm(bond_vectors, axis=2)  # shape: [B,L-1]
    return np.mean(np.abs(bond_lengths - slot_spacing), axis=1)  # shape: [B]


def nearest_training_distance(samples: np.ndarray, training: np.ndarray) -> np.ndarray:
    """Return whole-complex distance to the nearest intact training complex."""
    sample_flat = samples.reshape(samples.shape[0], -1)  # [B,L,D] -> [B,L*D]
    training_flat = training.reshape(training.shape[0], -1)  # [N,L,D] -> [N,L*D]
    difference = sample_flat[:, None, :] - training_flat[None, :, :]  # [B,1,L*D] - [1,N,L*D] -> [B,N,L*D]
    squared_distance = np.sum(difference * difference, axis=2)  # shape: [B,N]
    return np.sqrt(np.min(squared_distance, axis=1))  # shape: [B]


def evaluate_samples(
    samples: np.ndarray,
    training: np.ndarray,
    slot_spacing: float,
    coherence_threshold: float,
    bond_threshold: float,
) -> dict[str, np.ndarray]:
    """Evaluate raw toy joint validity without rejection or likelihood models."""
    coherence = coherence_error(samples)  # shape: [B]
    bonds = bond_error(samples, slot_spacing)  # shape: [B]
    nearest = nearest_training_distance(samples, training)  # shape: [B]
    valid = (coherence <= coherence_threshold) & (bonds <= bond_threshold)  # [B]
    return {
        "coherence": coherence,
        "bond_error": bonds,
        "nearest_training": nearest,
        "valid": valid,
    }


def plot_chains(
    output_path: Path,
    training: np.ndarray,
    source_a: np.ndarray,
    source_b: np.ndarray,
    shared: np.ndarray,
    independent: np.ndarray,
    u: np.ndarray,
    independent_lambdas: np.ndarray,
) -> None:
    """Plot source chains and representative shared and independent proposals."""
    midpoint = int(np.argmin(np.abs(u - 0.5)))
    independent_example = int(np.argmax(np.std(independent_lambdas, axis=1)))
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharex=True, sharey=True)

    for chain in training[:: max(1, training.shape[0] // 16)]:
        axes[0].plot(chain[:, 0], chain[:, 1], color="#BBBBBB", alpha=0.45)
    axes[0].plot(source_a[:, 0], source_a[:, 1], "o-", label="source A")
    axes[0].plot(source_b[:, 0], source_b[:, 1], "o-", label="source B")
    axes[0].set_title("Intact training sources")
    axes[0].legend(fontsize=8)

    axes[1].plot(shared[midpoint, :, 0], shared[midpoint, :, 1], "o-", color="#228833")
    axes[1].set_title(f"Shared lambda, u={u[midpoint]:.2f}")

    axes[2].plot(
        independent[independent_example, :, 0],
        independent[independent_example, :, 1],
        "o-",
        color="#EE7733",
    )
    values = ", ".join(f"{value:.2f}" for value in independent_lambdas[independent_example])
    axes[2].set_title(f"Independent lambdas\n[{values}]")

    for axis in axes:
        axis.set_xlabel("shared-state coordinate x")
        axis.grid(alpha=0.2)
        axis.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("ordered-slot coordinate y")
    figure.suptitle("Hypothesis 4.5: complete-chain interpolation")
    figure.tight_layout()
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_metric_comparison(
    output_path: Path,
    shared_metrics: dict[str, np.ndarray],
    independent_metrics: dict[str, np.ndarray],
    coherence_threshold: float,
    bond_threshold: float,
) -> None:
    """Plot the two direct coherence metrics and raw joint-valid percentage."""
    figure, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))
    labels = ["shared lambda", "independent lambda"]
    colors = ["#228833", "#EE7733"]

    coherence_boxes = axes[0].boxplot(
        [shared_metrics["coherence"], independent_metrics["coherence"]],
        tick_labels=labels,
        patch_artist=True,
    )
    bond_boxes = axes[1].boxplot(
        [shared_metrics["bond_error"], independent_metrics["bond_error"]],
        tick_labels=labels,
        patch_artist=True,
    )
    for boxes in (coherence_boxes, bond_boxes):
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

    axes[0].axhline(coherence_threshold, color="black", linestyle="--", label="training 95%")
    axes[0].set_title("Shared-state coherence error")
    axes[0].set_ylabel("RMS slot disagreement, lower is better")
    axes[0].legend(fontsize=8)
    axes[1].axhline(bond_threshold, color="black", linestyle="--", label="training 95%")
    axes[1].set_title("Adjacent-slot bond error")
    axes[1].set_ylabel("mean absolute error, lower is better")
    axes[1].legend(fontsize=8)

    valid_percent = [
        100.0 * float(np.mean(shared_metrics["valid"])),
        100.0 * float(np.mean(independent_metrics["valid"])),
    ]
    axes[2].bar(labels, valid_percent, color=colors)
    axes[2].set_ylim(0.0, 100.0)
    axes[2].set_title("Raw joint validity")
    axes[2].set_ylabel("valid candidates (%)")
    for index, value in enumerate(valid_percent):
        axes[2].text(index, min(98.0, value + 2.0), f"{value:.1f}%", ha="center")

    for axis in axes:
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_lambda_and_replay(
    output_path: Path,
    u: np.ndarray,
    shared_lambdas: np.ndarray,
    independent_lambdas: np.ndarray,
    shared_metrics: dict[str, np.ndarray],
    independent_metrics: dict[str, np.ndarray],
    vertex_relative_rmse: np.ndarray,
) -> None:
    """Show lambda synchronization, coherence, and mandatory vertex replay."""
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    axes[0].plot(u, shared_metrics["coherence"], "o-", color="#228833")
    axes[0].set_xlabel("shared u")
    axes[0].set_ylabel("coherence error")
    axes[0].set_title("Shared-lambda sweep")

    shared_spread = np.std(shared_lambdas, axis=1)  # shape: [B], always zero
    independent_spread = np.std(independent_lambdas, axis=1)  # shape: [B]
    axes[1].scatter(
        shared_spread,
        shared_metrics["coherence"],
        color="#228833",
        label="shared",
    )
    axes[1].scatter(
        independent_spread,
        independent_metrics["coherence"],
        color="#EE7733",
        label="independent",
    )
    axes[1].set_xlabel("standard deviation of slot lambdas")
    axes[1].set_ylabel("coherence error")
    axes[1].set_title("Coefficient disagreement")
    axes[1].legend(fontsize=8)

    axes[2].bar(["source A", "source B"], vertex_relative_rmse, color=["#4477AA", "#66CCEE"])
    axes[2].set_title("Vertex replay")
    axes[2].set_ylabel("RMSE / training RMS scale")

    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_pca_flow(
    output_path: Path,
    training: np.ndarray,
    history: np.ndarray,
    shared_terminal: np.ndarray,
    shared_trajectory: np.ndarray,
    u: np.ndarray,
    source_ids: tuple[int, int],
) -> None:
    """Project the complete-complex EFS flow onto one shared two-dimensional PCA plane.

    Each complete complex ``[L,D]`` is flattened to one vector ``[L*D]``. PCA is
    fitted once to the initial particles, terminal particles, shared terminal
    interpolations, and shared replay outputs:

        centered = flattened - mean
        centered = U @ diag(S) @ Vt
        PCA_2(flattened) = centered @ Vt[:2].T.

    One basis is reused scientifically so apparent movement has the same meaning
    at the initial, forward, interpolation, and backward stages. Coordinates are
    centered but not standardized because every flattened value has the same toy
    coordinate unit; scaling them separately would change the displayed geometry.

    Args:
        output_path: PNG path to create.
        training: Initial complete complexes with shape ``[N,L,D]``.
        history: Slot-wise EFS history with shape ``[K+1,L,N,D]``. Every
            empirical forward path is projected and drawn.
        shared_terminal: Shared-lambda terminal seeds with shape ``[B,L,D]``.
        shared_trajectory: Shared-lambda replay with shape ``[K+1,B,L,D]``.
        u: Shared two-source coefficients with shape ``[B]``.
        source_ids: Complete-complex particle IDs for sources A and B.
    """
    particle_count, slot_count, dimension = training.shape
    candidate_count = shared_terminal.shape[0]

    terminal_complete = np.transpose(
        history[-1],
        (1, 0, 2),
    )  # [L,N,D] -> [N,L,D], restoring complete complexes at terminal frame K
    shared_output = shared_trajectory[0]  # shape: [B,L,D], replay frame zero

    initial_flat = training.reshape(particle_count, slot_count * dimension)  # [N,L*D]
    terminal_flat = terminal_complete.reshape(
        particle_count,
        slot_count * dimension,
    )  # shape: [N,L*D]
    interpolation_flat = shared_terminal.reshape(
        candidate_count,
        slot_count * dimension,
    )  # shape: [B,L*D]
    output_flat = shared_output.reshape(
        candidate_count,
        slot_count * dimension,
    )  # shape: [B,L*D]

    pca_fit = np.concatenate(
        (initial_flat, terminal_flat, interpolation_flat, output_flat),
        axis=0,
    )  # shape: [2N+2B,L*D], endpoints define the common projection
    pca_mean = np.mean(pca_fit, axis=0, keepdims=True)  # shape: [1,L*D]
    centered_fit = pca_fit - pca_mean  # [2N+2B,L*D], mean broadcasts over rows
    _, singular_values, right_vectors = np.linalg.svd(
        centered_fit,
        full_matrices=False,
    )
    components = right_vectors[:2]  # shape: [2,L*D], first two principal axes
    explained_ratio = singular_values[:2] ** 2 / np.sum(singular_values**2)  # [2]

    def project(flattened: np.ndarray) -> np.ndarray:
        """Apply the fitted PCA basis to rows with shape ``[Q,L*D]``."""
        return (flattened - pca_mean) @ components.T  # [Q,L*D] @ [L*D,2] -> [Q,2]

    initial_pca = project(initial_flat)  # shape: [N,2]
    terminal_pca = project(terminal_flat)  # shape: [N,2]
    interpolation_pca = project(interpolation_flat)  # shape: [B,2]

    forward_complete = np.transpose(
        history,
        (0, 2, 1, 3),
    )  # [K+1,L,N,D] -> [K+1,N,L,D], restoring complete-complex paths
    forward_flat = forward_complete.reshape(
        history.shape[0] * particle_count,
        slot_count * dimension,
    )  # [K+1,N,L,D] -> [(K+1)*N,L*D]
    forward_pca = project(forward_flat).reshape(
        history.shape[0],
        particle_count,
        2,
    )  # [(K+1)*N,2] -> [K+1,N,2], preserving frame and particle axes

    trajectory_flat = shared_trajectory.reshape(
        shared_trajectory.shape[0] * candidate_count,
        slot_count * dimension,
    )  # [K+1,B,L,D] -> [(K+1)*B,L*D]
    trajectory_pca = project(trajectory_flat).reshape(
        shared_trajectory.shape[0],
        candidate_count,
        2,
    )  # [(K+1)*B,2] -> [K+1,B,2], preserving frame and candidate axes

    color_scale = Normalize(vmin=float(u[0]), vmax=float(u[-1]))
    colors = matplotlib.colormaps["viridis"](color_scale(u))  # shape: [B,4], RGBA per u
    figure, axes = plt.subplots(
        1,
        4,
        figsize=(17.0, 4.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    axes[0].scatter(initial_pca[:, 0], initial_pca[:, 1], s=18, color="#777777", alpha=0.7)
    for source_id, label, marker in zip(source_ids, ("A", "B"), ("s", "D")):
        axes[0].scatter(
            initial_pca[source_id, 0],
            initial_pca[source_id, 1],
            s=65,
            marker=marker,
            edgecolor="black",
            label=f"source {label}",
        )
    axes[0].set_title("1. Base distribution")
    axes[0].legend(fontsize=8)

    for particle in range(particle_count):
        path = forward_pca[:, particle, :]  # shape: [K+1,2]
        axes[1].plot(path[:, 0], path[:, 1], color="#4477AA", alpha=0.13, linewidth=0.55)
    axes[1].scatter(terminal_pca[:, 0], terminal_pca[:, 1], s=18, color="#4477AA", alpha=0.7)
    for source_id, marker in zip(source_ids, ("s", "D")):
        source_path = forward_pca[:, source_id, :]  # shape: [K+1,2]
        axes[1].plot(
            source_path[:, 0],
            source_path[:, 1],
            color="#CC3311",
            linewidth=1.2,
        )
        axes[1].scatter(
            terminal_pca[source_id, 0],
            terminal_pca[source_id, 1],
            s=65,
            marker=marker,
            edgecolor="black",
        )
    axes[1].set_title("2. Every forward path and final frame")

    axes[2].scatter(terminal_pca[:, 0], terminal_pca[:, 1], s=12, color="#BBBBBB", alpha=0.35)
    axes[2].plot(
        interpolation_pca[:, 0],
        interpolation_pca[:, 1],
        color="#666666",
        linewidth=1.0,
    )
    for candidate, color in enumerate(colors):
        axes[2].scatter(
            interpolation_pca[candidate, 0],
            interpolation_pca[candidate, 1],
            s=38,
            color=color,
            edgecolor="black",
            linewidth=0.35,
        )
    axes[2].set_title("3. Every shared-lambda seed")

    axes[3].scatter(initial_pca[:, 0], initial_pca[:, 1], s=12, color="#BBBBBB", alpha=0.35)
    for candidate, color in enumerate(colors):
        # The array is stored output-to-terminal. Reverse the frame axis to draw
        # the actual replay direction from the terminal seed to the final output.
        path = trajectory_pca[::-1, candidate, :]  # shape: [K+1,2]
        axes[3].plot(path[:, 0], path[:, 1], color=color, alpha=0.8, linewidth=1.0)
        axes[3].scatter(path[0, 0], path[0, 1], marker="x", s=28, color=color)
        axes[3].scatter(
            path[-1, 0],
            path[-1, 1],
            marker="o",
            s=28,
            color=color,
            edgecolor="black",
            linewidth=0.35,
        )
    axes[3].scatter([], [], marker="x", color="black", label="terminal seed")
    axes[3].scatter(
        [],
        [],
        marker="o",
        facecolor="none",
        edgecolor="black",
        label="replayed output",
    )
    axes[3].set_title("4. Shared-lambda backward replay")
    axes[3].legend(fontsize=8)

    x_label = f"PC1 ({100.0 * explained_ratio[0]:.1f}% variance)"
    y_label = f"PC2 ({100.0 * explained_ratio[1]:.1f}% variance)"
    for axis in axes:
        axis.set_xlabel(x_label)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel(y_label)
    colorbar = figure.colorbar(
        ScalarMappable(norm=color_scale, cmap="viridis"),
        ax=axes,
        fraction=0.025,
        pad=0.02,
    )
    colorbar.set_label("shared coefficient u")
    figure.suptitle("Whole-complex EFS flow in one fixed PCA projection")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_metrics_csv(
    path: Path,
    u: np.ndarray,
    shared_lambdas: np.ndarray,
    independent_lambdas: np.ndarray,
    shared_metrics: dict[str, np.ndarray],
    independent_metrics: dict[str, np.ndarray],
    shared_residual: np.ndarray,
    independent_residual: np.ndarray,
) -> None:
    """Write one transparent row per raw proposal without rejection."""
    fieldnames = [
        "method",
        "candidate",
        "u",
        "slot_lambdas",
        "lambda_spread",
        "coherence_error",
        "bond_error",
        "nearest_training_distance",
        "max_proximal_residual",
        "raw_joint_valid",
    ]
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, lambdas, metrics, residual in (
            ("shared", shared_lambdas, shared_metrics, shared_residual),
            ("independent", independent_lambdas, independent_metrics, independent_residual),
        ):
            for candidate in range(u.size):
                writer.writerow({
                    "method": method,
                    "candidate": candidate,
                    "u": f"{u[candidate]:.6f}" if method == "shared" else "",
                    "slot_lambdas": ";".join(f"{value:.6f}" for value in lambdas[candidate]),
                    "lambda_spread": f"{np.std(lambdas[candidate]):.8f}",
                    "coherence_error": f"{metrics['coherence'][candidate]:.8f}",
                    "bond_error": f"{metrics['bond_error'][candidate]:.8f}",
                    "nearest_training_distance": (f"{metrics['nearest_training'][candidate]:.8f}"),
                    "max_proximal_residual": f"{residual[candidate]:.8e}",
                    "raw_joint_valid": int(metrics["valid"][candidate]),
                })


def run(args: argparse.Namespace) -> Path:
    """Execute the complete minimal H4.5 comparison and save every raw proposal."""
    training, states = generate_complete_chains(
        particle_count=args.particles,
        slot_count=args.slots,
        slot_spacing=args.slot_spacing,
        curve_amplitude=args.curve_amplitude,
        noise_std=args.noise_std,
        seed=args.seed,
    )  # training: [N,L,D=2], states: [N]
    source_a_id, source_b_id = select_source_ids(
        states,
        args.source_low,
        args.source_high,
    )

    # EFS expects [L,N,D]. Transpose swaps complete-complex and slot axes while
    # preserving particle index N across every ordered slot.
    initial_particles = np.transpose(
        training,
        (1, 0, 2),
    )  # [N,L,D] -> [L,N,D]

    start = time.perf_counter()
    history = forward_history(
        initial_particles,
        gamma=args.gamma,
        epsilon=args.epsilon,
        exponent_s=args.exponent_s,
        steps=args.forward_steps,
    )  # shape: [K+1,L,N,D]
    forward_seconds = time.perf_counter() - start

    terminal_frame = history[-1]  # shape: [L,N,D]
    terminal_a = terminal_frame[:, source_a_id, :]  # shape: [L,D]
    terminal_b = terminal_frame[:, source_b_id, :]  # shape: [L,D]

    u, shared_lambdas, independent_lambdas = make_lambda_controls(
        args.candidates,
        args.slots,
    )
    shared_terminal = barycentric_terminal_seeds(
        terminal_a,
        terminal_b,
        shared_lambdas,
    )  # shape: [B,L,D]
    independent_terminal = barycentric_terminal_seeds(
        terminal_a,
        terminal_b,
        independent_lambdas,
    )  # shape: [B,L,D]

    vertex_terminal = np.stack(
        (terminal_a, terminal_b),
        axis=0,
    )  # shape: [2,L,D]
    all_terminal = np.concatenate(
        (vertex_terminal, shared_terminal, independent_terminal),
        axis=0,
    )  # shape: [2+2B,L,D], all candidates share one batched replay call

    start = time.perf_counter()
    replay_trajectory, replay_residual = backward_replay(
        all_terminal,
        history,
        gamma=args.gamma,
        epsilon=args.epsilon,
        exponent_s=args.exponent_s,
        beta=args.beta,
        proximal_steps=args.proximal_steps,
    )  # [K+1,2+2B,L,D], [K,2+2B,L]
    backward_seconds = time.perf_counter() - start

    final = replay_trajectory[0]  # shape: [2+2B,L,D]
    vertex_replay = final[:2]  # shape: [2,L,D]
    shared = final[2 : 2 + args.candidates]  # shape: [B,L,D]
    independent = final[2 + args.candidates :]  # shape: [B,L,D]

    expected_vertices = training[[source_a_id, source_b_id]]  # shape: [2,L,D]
    vertex_rmse = np.sqrt(np.mean((vertex_replay - expected_vertices) ** 2, axis=(1, 2)))  # shape: [2]
    training_center = np.mean(training, axis=0, keepdims=True)  # shape: [1,L,D]
    training_rms = float(np.sqrt(np.mean((training - training_center) ** 2)))
    vertex_relative_rmse = vertex_rmse / max(training_rms, 1.0e-12)  # shape: [2]

    training_coherence = coherence_error(training)  # shape: [N]
    training_bonds = bond_error(training, args.slot_spacing)  # shape: [N]
    coherence_threshold = float(np.quantile(training_coherence, 0.95))
    bond_threshold = float(np.quantile(training_bonds, 0.95))

    shared_metrics = evaluate_samples(
        shared,
        training,
        args.slot_spacing,
        coherence_threshold,
        bond_threshold,
    )
    independent_metrics = evaluate_samples(
        independent,
        training,
        args.slot_spacing,
        coherence_threshold,
        bond_threshold,
    )
    shared_residual = np.max(
        replay_residual[:, 2 : 2 + args.candidates, :],
        axis=(0, 2),
    )  # [K,B,L] -> [B]
    independent_residual = np.max(
        replay_residual[:, 2 + args.candidates :, :],
        axis=(0, 2),
    )  # [K,B,L] -> [B]

    arrays = (
        training,
        history,
        shared_terminal,
        independent_terminal,
        replay_trajectory,
        replay_residual,
    )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise FloatingPointError("experiment produced non-finite values")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_directory = output_root / f"run_{timestamp}_seed{args.seed}"
    run_directory.mkdir(exist_ok=False)

    config = vars(args).copy()
    config.update({
        "source_a_id": source_a_id,
        "source_b_id": source_b_id,
        "source_a_state": float(states[source_a_id]),
        "source_b_state": float(states[source_b_id]),
        "coherence_threshold_training_95": coherence_threshold,
        "bond_threshold_training_95": bond_threshold,
        "training_rms_scale": training_rms,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    with (run_directory / "config.json").open("x", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)

    np.savez_compressed(
        run_directory / "samples.npz",
        training=training,
        states=states,
        slot_ids=np.arange(args.slots, dtype=np.int64),
        source_ids=np.array([source_a_id, source_b_id], dtype=np.int64),
        history=history,
        u=u,
        shared_lambdas=shared_lambdas,
        independent_lambdas=independent_lambdas,
        shared_terminal=shared_terminal,
        independent_terminal=independent_terminal,
        replay_trajectory=replay_trajectory,
        replay_residual=replay_residual,
        vertex_replay=vertex_replay,
        shared=shared,
        independent=independent,
    )
    write_metrics_csv(
        run_directory / "metrics.csv",
        u,
        shared_lambdas,
        independent_lambdas,
        shared_metrics,
        independent_metrics,
        shared_residual,
        independent_residual,
    )

    shared_valid = 100.0 * float(np.mean(shared_metrics["valid"]))
    independent_valid = 100.0 * float(np.mean(independent_metrics["valid"]))
    shared_coherence = float(np.median(shared_metrics["coherence"]))
    independent_coherence = float(np.median(independent_metrics["coherence"]))
    max_vertex_error = float(np.max(vertex_relative_rmse))

    if max_vertex_error > 0.1:
        conclusion = "Do not interpret H4.5: vertex replay exceeds 10% of training RMS scale."
    elif shared_valid > independent_valid and shared_coherence < independent_coherence:
        conclusion = "The minimal toy supports shared lambda over independent slot lambdas."
    else:
        conclusion = "The minimal toy does not support shared lambda over the control."

    summary = [
        "Minimal Hypothesis 4.5 result",
        "",
        f"source IDs: {source_a_id}, {source_b_id}",
        f"source states: {states[source_a_id]:.6f}, {states[source_b_id]:.6f}",
        f"vertex relative RMSE: {vertex_relative_rmse[0]:.6e}, {vertex_relative_rmse[1]:.6e}",
        f"shared raw joint validity: {shared_valid:.2f}%",
        f"independent raw joint validity: {independent_valid:.2f}%",
        f"shared median coherence error: {shared_coherence:.6e}",
        f"independent median coherence error: {independent_coherence:.6e}",
        f"forward runtime: {forward_seconds:.2f} s",
        f"backward runtime: {backward_seconds:.2f} s",
        "",
        conclusion,
    ]
    with (run_directory / "summary.txt").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(summary) + "\n")

    plot_chains(
        run_directory / "chains.png",
        training,
        training[source_a_id],
        training[source_b_id],
        shared,
        independent,
        u,
        independent_lambdas,
    )
    plot_metric_comparison(
        run_directory / "coherence_comparison.png",
        shared_metrics,
        independent_metrics,
        coherence_threshold,
        bond_threshold,
    )
    plot_lambda_and_replay(
        run_directory / "lambda_and_vertex_replay.png",
        u,
        shared_lambdas,
        independent_lambdas,
        shared_metrics,
        independent_metrics,
        vertex_relative_rmse,
    )
    plot_pca_flow(
        run_directory / "pca_flow.png",
        training,
        history,
        shared_terminal,
        replay_trajectory[:, 2 : 2 + args.candidates, :, :],
        u,
        (source_a_id, source_b_id),
    )
    return run_directory


def build_parser() -> argparse.ArgumentParser:
    """Define the small set of parameters needed by the H4.5 experiment."""
    parser = argparse.ArgumentParser(description="Test shared versus independent two-source lambdas with slot-wise EFS.")
    parser.add_argument("--output-root", default="results", help="Parent directory for a new run folder.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the one-population chain toy.")
    parser.add_argument("--particles", type=int, default=64, help="Number N of intact training complexes.")
    parser.add_argument("--slots", type=int, default=4, help="Ordered slots L in every complete complex.")
    parser.add_argument("--candidates", type=int, default=9, help="Odd number B of interior two-source coefficients.")
    parser.add_argument("--slot-spacing", type=float, default=1.0, help="Expected adjacent-slot distance.")
    parser.add_argument("--curve-amplitude", type=float, default=0.35, help="Common nonlinear displacement controlled by chain state.")
    parser.add_argument("--noise-std", type=float, default=0.03, help="Coordinate noise standard deviation.")
    parser.add_argument("--source-low", type=float, default=-0.75, help="Requested scalar state for source A.")
    parser.add_argument("--source-high", type=float, default=0.75, help="Requested scalar state for source B.")
    parser.add_argument("--gamma", type=float, default=0.0005, help="Forward EFS Euler step size.")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Positive inverse-power smoothing constant.")
    parser.add_argument("--exponent-s", type=float, default=0.0, help="EFS exponent s, with d-2=0 for two dimensions.")
    parser.add_argument("--forward-steps", type=int, default=5000, help="Cached forward updates K.")
    parser.add_argument("--beta", type=float, default=0.05, help="Backward proximal optimizer step size.")
    parser.add_argument("--proximal-steps", type=int, default=100, help="Inner backward iterations T per frame.")
    return parser


def main() -> None:
    """Parse arguments, run the experiment, and print the immutable result path."""
    args = build_parser().parse_args()
    run_directory = run(args)
    print(f"results: {run_directory.resolve()}")


if __name__ == "__main__":
    main()

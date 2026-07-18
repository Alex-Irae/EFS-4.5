"""Exact single-plane Estimation-Free Sampling for Hypothesis 4.5.

Purpose:
    Evolve one empirical particle plane forward, then replay complete generated
    sources through that same cached plane. Generated particles are batched but
    do not exert forces on one another during vanilla EFS replay.

Dependencies:
    NumPy only.

Exact self-check command:
    python efs_single_plane.py
"""

from __future__ import annotations

import numpy as np


def potential_gradient(displacement: np.ndarray, epsilon: float, exponent_s: float) -> np.ndarray:
    """Evaluate ``grad W(z)`` for displacement vectors shaped ``[...,D]``.

    The implemented equation is

        grad W(z) = z - z / (||z||^2 + epsilon)^(s/2 + 1).

    The quadratic term attracts globally. The inverse-power term prevents
    short-range collapse.
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    displacement = np.asarray(displacement, dtype=np.float64)
    squared_norm = np.sum(displacement * displacement, axis=-1, keepdims=True)  # shape: [...,1], coordinate axis D is reduced
    inverse_power = np.power(squared_norm + epsilon, -(exponent_s / 2.0 + 1.0))  # shape: [...,1], broadcasts over D
    gradient = displacement * (1.0 - inverse_power)  # shape: [...,D]

    if not np.all(np.isfinite(gradient)):
        raise FloatingPointError("EFS potential gradient became non-finite")
    return gradient


def mean_pairwise_distance(particles: np.ndarray) -> float:
    """Return the exact mean distance between distinct rows of ``[N,D]``.

    Pairwise distances use

        ||x_i-x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i dot x_j.

    This avoids allocating the much larger displacement tensor ``[N,N,D]``.
    """
    particles = np.asarray(particles, dtype=np.float64)
    if particles.ndim != 2 or particles.shape[0] < 2:
        raise ValueError("particles must have shape [N,D] with N >= 2")

    particle_count = particles.shape[0]
    squared_norm = np.sum(particles * particles, axis=1)  # shape: [N]
    squared_distance = (
        squared_norm[:, None] + squared_norm[None, :] - 2.0 * (particles @ particles.T)
    )  # [N,1] + [1,N] - [N,D]@[D,N] -> [N,N]
    np.maximum(squared_distance, 0.0, out=squared_distance)
    np.sqrt(squared_distance, out=squared_distance)  # reuse [N,N] as distances
    return float(np.sum(squared_distance) / (particle_count * (particle_count - 1)))


def forward_field(particles: np.ndarray, epsilon: float, exponent_s: float) -> np.ndarray:
    """Compute the exact EFS forward field on one shared plane ``[N,D]``.

    For particle ``i``:

        field_i = 1/(N-1) sum_(a != i) grad W(x_i-x_a).

    All particles from every complete source occupy this same field. The code
    stores scalar pair matrices ``[N,N]`` but never ``[N,N,D]``.
    """
    particles = np.asarray(particles, dtype=np.float64)
    if particles.ndim != 2 or particles.shape[0] < 2:
        raise ValueError("particles must have shape [N,D] with N >= 2")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if not np.all(np.isfinite(particles)):
        raise ValueError("particles contain non-finite values")

    particle_count = particles.shape[0]
    squared_norm = np.sum(particles * particles, axis=1)  # shape: [N]
    pair_dot = particles @ particles.T  # [N,D] @ [D,N] -> [N,N]
    squared_distance = (
        squared_norm[:, None] + squared_norm[None, :] - 2.0 * pair_dot
    )  # shape: [N,N], singleton axes broadcast over both particle axes
    np.maximum(squared_distance, 0.0, out=squared_distance)

    weights = np.power(
        squared_distance + epsilon, -(exponent_s / 2.0 + 1.0)
    )  # shape: [N,N], inverse-power coefficient for every ordered pair
    np.fill_diagonal(weights, 0.0)  # a forward particle does not act on itself

    particle_sum = np.sum(particles, axis=0, keepdims=True)  # shape: [1,D]
    attractive_sum = particle_count * particles - particle_sum  # shape: [N,D]

    weighted_particle_sum = weights @ particles  # [N,N] @ [N,D] -> [N,D]
    weight_sum = np.sum(weights, axis=1, keepdims=True)  # shape: [N,1]
    repulsive_sum = particles * weight_sum - weighted_particle_sum  # shape: [N,D]

    field = (attractive_sum - repulsive_sum) / float(particle_count - 1)
    if not np.all(np.isfinite(field)):
        raise FloatingPointError("EFS forward field became non-finite")
    return field


def forward_history(
    initial_particles: np.ndarray, gamma: float, epsilon: float, exponent_s: float, steps: int, log_every: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evolve one plane and cache ``[J+1,N,D]`` with periodic distance logs.

    The Euler update is

        x^(j+1) = x^j - gamma * forward_field(x^j).

    Returns:
        ``(history, log_steps, log_distances)`` with shapes ``[J+1,N,D]``,
        ``[R]``, and ``[R]``.
    """
    initial_particles = np.asarray(initial_particles, dtype=np.float64)
    if initial_particles.ndim != 2:
        raise ValueError("initial_particles must have shape [N,D]")
    if gamma <= 0.0 or steps < 1 or log_every < 0:
        raise ValueError("gamma and steps must be positive; log_every must be non-negative")

    particle_count, dimension = initial_particles.shape
    history = np.empty((steps + 1, particle_count, dimension), dtype=np.float64)  # shape: [J+1,N,D]
    history[0] = initial_particles
    logged_steps: list[int] = []
    logged_distances: list[float] = []

    # ponytail: exact EFS is O(N^2) per frame. The scalar pair matrices are the
    # simplest exact implementation; approximate neighbors belong in a later study.
    for frame in range(steps):
        if log_every and frame % log_every == 0:
            distance = mean_pairwise_distance(history[frame])
            print(f"forward step {frame:5d}/{steps}: mean_pair_distance={distance:.8f}")
            logged_steps.append(frame)
            logged_distances.append(distance)

        field = forward_field(history[frame], epsilon=epsilon, exponent_s=exponent_s)  # shape: [N,D]
        history[frame + 1] = history[frame] - gamma * field

    final_distance = mean_pairwise_distance(history[-1])
    print(f"forward step {steps:5d}/{steps}: mean_pair_distance={final_distance:.8f}")
    logged_steps.append(steps)
    logged_distances.append(final_distance)

    return (history, np.asarray(logged_steps, dtype=np.int64), np.asarray(logged_distances, dtype=np.float64))


def backward_field(candidates: np.ndarray, particles: np.ndarray, gamma: float, epsilon: float, exponent_s: float) -> np.ndarray:
    """Evaluate the vanilla EFS replay field for queries ``[Q,D]``.

    At cached frame ``j``:

        F_j(v_q) = gamma/N sum_i grad W(v_q-x_i^j).

    Queries share the same empirical plane but do not occur in one another's
    force sums. This is the literal candidate-batched EFS rule.
    """
    candidates = np.asarray(candidates, dtype=np.float64)
    particles = np.asarray(particles, dtype=np.float64)
    if candidates.ndim != 2 or particles.ndim != 2:
        raise ValueError("candidates and particles must have shapes [Q,D] and [N,D]")
    if candidates.shape[1] != particles.shape[1]:
        raise ValueError("candidate and plane dimensions differ")
    if gamma <= 0.0 or epsilon <= 0.0:
        raise ValueError("gamma and epsilon must be positive")

    particle_count = particles.shape[0]
    candidate_norm = np.sum(candidates * candidates, axis=1, keepdims=True)  # [Q,1]
    particle_norm = np.sum(particles * particles, axis=1)[None, :]  # [1,N]
    pair_dot = candidates @ particles.T  # [Q,D] @ [D,N] -> [Q,N]
    squared_distance = candidate_norm + particle_norm - 2.0 * pair_dot  # [Q,N]
    np.maximum(squared_distance, 0.0, out=squared_distance)

    weights = np.power(squared_distance + epsilon, -(exponent_s / 2.0 + 1.0))  # shape: [Q,N]
    particle_sum = np.sum(particles, axis=0, keepdims=True)  # shape: [1,D]
    attractive_sum = particle_count * candidates - particle_sum  # shape: [Q,D]
    weighted_particle_sum = weights @ particles  # [Q,N] @ [N,D] -> [Q,D]
    weight_sum = np.sum(weights, axis=1, keepdims=True)  # shape: [Q,1]
    repulsive_sum = candidates * weight_sum - weighted_particle_sum  # shape: [Q,D]

    field = gamma * (attractive_sum - repulsive_sum) / float(particle_count)
    if not np.all(np.isfinite(field)):
        raise FloatingPointError("EFS backward field became non-finite")
    return field


def ensemble_mean_distance(ensembles: np.ndarray) -> np.ndarray:
    """Return within-ensemble mean distances for arrays shaped ``[G,P,D]``."""
    ensembles = np.asarray(ensembles, dtype=np.float64)
    if ensembles.ndim != 3 or ensembles.shape[1] < 2:
        raise ValueError("ensembles must have shape [G,P,D] with P >= 2")

    particle_count = ensembles.shape[1]
    difference = ensembles[:, :, None, :] - ensembles[:, None, :, :]  # [G,P,1,D] - [G,1,P,D] -> [G,P,P,D]
    distance = np.linalg.norm(difference, axis=3)  # shape: [G,P,P]
    return np.sum(distance, axis=(1, 2)) / float(particle_count * (particle_count - 1))


def backward_replay(
    terminal_ensembles: np.ndarray,
    history: np.ndarray,
    gamma: float,
    epsilon: float,
    exponent_s: float,
    beta: float,
    proximal_steps: int,
    labels: list[str],
    log_every: int = 10,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Replay complete candidate sources together through one cached plane.

    At frame ``j``, for every generated particle:

        delta_t = v_t - y^j - F_j(v_t)
        v_(t+1) = v_t - beta * delta_t
        y^(j-1) = v_T.

    Args:
        terminal_ensembles: Candidate sources shaped ``[G,P,D]``.
        history: One shared forward history shaped ``[J+1,N,D]``.
        labels: ``G`` names used only in ten-frame logs.

    Returns:
        ``(trajectory, residuals, diagnostics)``. Shapes are ``[J+1,G,P,D]``
        and ``[J,G,P]``. Diagnostics contain only printed complete-frame states.
    """
    terminal_ensembles = np.asarray(terminal_ensembles, dtype=np.float64)
    history = np.asarray(history, dtype=np.float64)
    if terminal_ensembles.ndim != 3 or history.ndim != 3:
        raise ValueError("expected terminal [G,P,D] and history [J+1,N,D]")
    if terminal_ensembles.shape[2] != history.shape[2]:
        raise ValueError("terminal and history dimensions differ")
    if len(labels) != terminal_ensembles.shape[0]:
        raise ValueError("labels must contain one name per candidate ensemble")
    if beta <= 0.0 or proximal_steps < 1 or log_every < 0:
        raise ValueError("beta and proximal_steps must be positive; log_every must be non-negative")

    forward_steps = history.shape[0] - 1
    group_count, particles_per_group, dimension = terminal_ensembles.shape
    trajectory = np.empty(
        (forward_steps + 1, group_count, particles_per_group, dimension), dtype=np.float64
    )  # shape: [J+1,G,P,D]
    residuals = np.empty((forward_steps, group_count, particles_per_group), dtype=np.float64)  # shape: [J,G,P]

    current = terminal_ensembles.copy()  # shape: [G,P,D]
    trajectory[forward_steps] = current
    log_steps: list[int] = [0]
    log_distance: list[np.ndarray] = [ensemble_mean_distance(current)]
    log_total: list[np.ndarray] = [np.sum(current, axis=1)]  # each entry: [G,D]
    log_displacement: list[np.ndarray] = [np.zeros((group_count, dimension))]

    print("backward step     0/{:d}: terminal candidate state".format(forward_steps))
    for group, label in enumerate(labels):
        print(
            f"  {label}: mean_pair_distance={log_distance[-1][group]:.8f} "
            f"total_position={np.array2string(log_total[-1][group], precision=6)} "
            f"displacement={np.zeros(dimension)} displacement_norm=0.00000000"
        )

    for frame in range(forward_steps, 0, -1):
        anchor = current.copy()  # y^j, shape: [G,P,D]
        value = anchor.copy()  # v_0 = y^j, shape: [G,P,D]
        particles = history[frame]  # post-forward frame j, shape: [N,D]

        for _ in range(proximal_steps):
            flat_value = value.reshape(
                group_count * particles_per_group, dimension
            )  # [G,P,D] -> [G*P,D], batches all generated particles
            force = backward_field(flat_value, particles, gamma=gamma, epsilon=epsilon, exponent_s=exponent_s).reshape(
                group_count, particles_per_group, dimension
            )  # [G*P,D] -> [G,P,D]
            delta = value - anchor - force  # shape: [G,P,D]
            value = value - beta * delta

        flat_value = value.reshape(group_count * particles_per_group, dimension)
        force = backward_field(flat_value, particles, gamma=gamma, epsilon=epsilon, exponent_s=exponent_s).reshape(
            group_count, particles_per_group, dimension
        )
        delta = value - anchor - force  # shape: [G,P,D]
        residuals[frame - 1] = np.linalg.norm(delta, axis=2) / (
            1.0 + np.linalg.norm(anchor, axis=2)
        )  # shape: [G,P], coordinate axis D is reduced

        trajectory[frame - 1] = value
        current = value
        reverse_step = forward_steps - frame + 1

        if log_every and (reverse_step % log_every == 0 or reverse_step == forward_steps):
            distance = ensemble_mean_distance(value)  # shape: [G]
            total = np.sum(value, axis=1)  # shape: [G,D]
            previous_total = np.sum(anchor, axis=1)  # shape: [G,D]
            displacement = total - previous_total  # one complete reverse-frame move, [G,D]
            print(f"backward step {reverse_step:5d}/{forward_steps} (frame {frame}->{frame - 1})")
            for group, label in enumerate(labels):
                print(
                    f"  {label}: mean_pair_distance={distance[group]:.8f} "
                    f"total_position={np.array2string(total[group], precision=6)} "
                    f"displacement={np.array2string(displacement[group], precision=6)} "
                    f"displacement_norm={np.linalg.norm(displacement[group]):.8f}"
                )
            log_steps.append(reverse_step)
            log_distance.append(distance)
            log_total.append(total)
            log_displacement.append(displacement)

    if not np.all(np.isfinite(trajectory)):
        raise FloatingPointError("EFS backward replay became non-finite")

    diagnostics = {
        "reverse_step": np.asarray(log_steps, dtype=np.int64),  # shape: [R]
        "mean_pair_distance": np.asarray(log_distance),  # shape: [R,G]
        "total_position": np.asarray(log_total),  # shape: [R,G,D]
        "displacement": np.asarray(log_displacement),  # shape: [R,G,D]
    }
    return trajectory, residuals, diagnostics


def self_check() -> dict[str, float]:
    """Compare the single-plane algebra with the literal pairwise EFS sum."""
    rng = np.random.default_rng(7)
    particles = rng.normal(size=(6, 4))  # shape: [N=6,D=4]
    vectorized = forward_field(particles, epsilon=0.1, exponent_s=2.0)
    explicit = np.zeros_like(vectorized)  # shape: [N,D]

    for query in range(particles.shape[0]):
        for other in range(particles.shape[0]):
            if query == other:
                continue
            explicit[query] += potential_gradient(particles[query] - particles[other], epsilon=0.1, exponent_s=2.0)
        explicit[query] /= float(particles.shape[0] - 1)

    field_error = float(np.max(np.abs(vectorized - explicit)))
    center_drift = float(np.max(np.abs(np.sum(vectorized, axis=0))))
    assert field_error < 1.0e-12, field_error
    assert center_drift < 1.0e-12, center_drift
    return {"field_max_abs_error": field_error, "center_drift": center_drift}


if __name__ == "__main__":
    print(self_check())

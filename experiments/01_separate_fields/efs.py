"""Minimal batched Estimation-Free Sampling core for Hypothesis 4.5.

Run the built-in numerical check with:
    python efs.py

The leading axes are kept explicit throughout:
    L: ordered slots in one complete complex
    N: empirical particles, one particle per complete-complex slot
    D: dimensions in one slot
    B: generated barycentric candidates
    K: cached forward frames
"""

from __future__ import annotations

import numpy as np


def potential_gradient(displacement: np.ndarray, epsilon: float, exponent_s: float) -> np.ndarray:
    """Evaluate the regularized EFS interaction gradient.

    The implemented equation is

        grad W(z) = z - z / (||z||^2 + epsilon)^(s/2 + 1).

    Scientifically, the first term attracts particles globally and the second
    term prevents collapse at short distances.

    Args:
        displacement: Pair displacement vectors with shape ``[..., D]``.
        epsilon: Positive smoothing constant for the inverse-power term.
        exponent_s: Potential exponent ``s``. The two-dimensional default is 0.

    Returns:
        Gradient vectors with the same shape as ``displacement``.
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    displacement = np.asarray(displacement, dtype=np.float64)
    squared_norm = np.sum(displacement * displacement, axis=-1, keepdims=True)  # shape: [..., 1], summed over coordinate axis D
    inverse_power = np.power(
        squared_norm + epsilon, -(exponent_s / 2.0 + 1.0)
    )  # shape: [..., 1], broadcasts across coordinate axis D
    gradient = displacement * (1.0 - inverse_power)  # shape: [..., D]

    if not np.all(np.isfinite(gradient)):
        raise FloatingPointError("EFS potential gradient became non-finite")
    return gradient


def forward_field(particles: np.ndarray, epsilon: float, exponent_s: float) -> np.ndarray:
    """Compute one EFS forward field for every slot and empirical particle.

    The equation for slot ``l`` and particle ``i`` is

        field[l,i] = 1/(N-1) * sum_(a != i) grad W(x[l,i] - x[l,a]).

    The code avoids a displacement tensor ``[L,N,N,D]``. Pairwise squared
    distances use

        ||q-x||^2 = ||q||^2 + ||x||^2 - 2 q dot x.

    The attractive and repulsive vector sums are then reduced algebraically.
    This keeps all slots batched while preserving their independent EFS fields.

    Args:
        particles: Current slot clouds with shape ``[L,N,D]``.
        epsilon: Positive smoothing constant.
        exponent_s: Potential exponent ``s``.

    Returns:
        Normalized forward field with shape ``[L,N,D]``.
    """
    particles = np.asarray(particles, dtype=np.float64)
    if particles.ndim != 3:
        raise ValueError("particles must have shape [L,N,D]")
    if particles.shape[1] < 2:
        raise ValueError("forward EFS requires at least two particles")
    if not np.all(np.isfinite(particles)):
        raise ValueError("particles contain non-finite values")

    _, particle_count, _ = particles.shape
    squared_norm = np.sum(particles * particles, axis=2)  # shape: [L,N], coordinate axis D is reduced
    pair_dot = np.matmul(particles, np.swapaxes(particles, 1, 2))  # [L,N,D] @ [L,D,N] -> [L,N,N]
    squared_distance = (
        squared_norm[:, :, None] + squared_norm[:, None, :] - 2.0 * pair_dot
    )  # shape: [L,N,N], singleton axes broadcast over both particle axes
    np.maximum(squared_distance, 0.0, out=squared_distance)

    weights = np.power(squared_distance + epsilon, -(exponent_s / 2.0 + 1.0))  # shape: [L,N,N]
    particle_index = np.arange(particle_count)
    weights[:, particle_index, particle_index] = 0.0  # remove self-interaction

    particle_sum = np.sum(particles, axis=1, keepdims=True)  # shape: [L,1,D], broadcasts over query particle i
    attractive_sum = particle_count * particles - particle_sum  # shape: [L,N,D]

    weighted_particle_sum = np.matmul(weights, particles)  # [L,N,N] @ [L,N,D] -> [L,N,D]
    weight_sum = np.sum(weights, axis=2, keepdims=True)  # shape: [L,N,1], broadcasts over coordinate axis D
    repulsive_sum = particles * weight_sum - weighted_particle_sum  # shape: [L,N,D]

    field = (attractive_sum - repulsive_sum) / float(particle_count - 1)
    if not np.all(np.isfinite(field)):
        raise FloatingPointError("EFS forward field became non-finite")
    return field


def forward_history(initial_particles: np.ndarray, gamma: float, epsilon: float, exponent_s: float, steps: int) -> np.ndarray:
    """Run slot-wise EFS forward while retaining particle and slot identities.

    The update is

        x^(j+1) = x^j - gamma * forward_field(x^j).

    Args:
        initial_particles: Initial clouds with shape ``[L,N,D]``. Particle index
            ``i`` refers to the same complete source complex in every slot.
        gamma: Positive forward Euler step size.
        epsilon: Positive smoothing constant.
        exponent_s: Potential exponent ``s``.
        steps: Number of forward updates ``K``.

    Returns:
        Cached history with shape ``[K+1,L,N,D]``.
    """
    initial_particles = np.asarray(initial_particles, dtype=np.float64)
    if initial_particles.ndim != 3:
        raise ValueError("initial_particles must have shape [L,N,D]")
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    if steps < 1:
        raise ValueError("steps must be at least one")

    slot_count, particle_count, dimension = initial_particles.shape
    history = np.empty((steps + 1, slot_count, particle_count, dimension), dtype=np.float64)  # shape: [K+1,L,N,D]
    history[0] = initial_particles

    # ponytail: full [L,N,N] batching is simplest and fast for local memories;
    # chunk the particle axes only if N grows enough to pressure available RAM.
    for frame in range(steps):
        field = forward_field(history[frame], epsilon, exponent_s)  # shape: [L,N,D]
        history[frame + 1] = history[frame] - gamma * field

    return history


def backward_field(candidates: np.ndarray, particles: np.ndarray, gamma: float, epsilon: float, exponent_s: float) -> np.ndarray:
    """Compute the literal paper replay force for all candidates and slots.

    The equation is

        F_j(v)[b,l] = gamma/N * sum_i grad W(v[b,l] - x[l,i]^j).

    Args:
        candidates: Candidate slot vectors with shape ``[B,L,D]``.
        particles: One cached history frame with shape ``[L,N,D]``.
        gamma: Forward step size appearing in the replay equation.
        epsilon: Positive smoothing constant.
        exponent_s: Potential exponent ``s``.

    Returns:
        Replay force with shape ``[B,L,D]``.
    """
    candidates = np.asarray(candidates, dtype=np.float64)
    particles = np.asarray(particles, dtype=np.float64)
    if candidates.ndim != 3 or particles.ndim != 3:
        raise ValueError("expected candidates [B,L,D] and particles [L,N,D]")
    if candidates.shape[1:] != (particles.shape[0], particles.shape[2]):
        raise ValueError("candidate slots or dimensions do not match history")

    particle_count = particles.shape[1]
    candidate_norm = np.sum(candidates * candidates, axis=2, keepdims=True)  # shape: [B,L,1]
    particle_norm = np.sum(particles * particles, axis=2)  # shape: [L,N]
    pair_dot = np.einsum(
        "bld,lnd->bln", candidates, particles
    )  # for each b,l: dot candidates[b,l,:] with particles[l,n,:] -> [B,L,N]
    squared_distance = candidate_norm + particle_norm[None, :, :] - 2.0 * pair_dot  # [B,L,1] and [1,L,N] broadcast to [B,L,N]
    np.maximum(squared_distance, 0.0, out=squared_distance)

    weights = np.power(squared_distance + epsilon, -(exponent_s / 2.0 + 1.0))  # shape: [B,L,N]
    particle_sum = np.sum(particles, axis=1)[None, :, :]  # shape: [1,L,D], broadcasts over candidate axis B
    attractive_sum = particle_count * candidates - particle_sum  # shape: [B,L,D]

    weighted_particle_sum = np.einsum(
        "bln,lnd->bld", weights, particles
    )  # sum weights[b,l,n] * particles[l,n,d] over empirical particle n
    weight_sum = np.sum(weights, axis=2, keepdims=True)  # shape: [B,L,1]
    repulsive_sum = candidates * weight_sum - weighted_particle_sum  # [B,L,D]

    field = gamma * (attractive_sum - repulsive_sum) / float(particle_count)
    if not np.all(np.isfinite(field)):
        raise FloatingPointError("EFS backward field became non-finite")
    return field


def backward_replay(
    terminal_candidates: np.ndarray,
    history: np.ndarray,
    gamma: float,
    epsilon: float,
    exponent_s: float,
    beta: float,
    proximal_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay shared and independent candidates through the same EFS histories.

    At cached frame ``j`` the fixed-point iteration is

        delta_t = v_t - y^j - F_j(v_t)
        v_(t+1) = v_t - beta * delta_t
        y^(j-1) = v_T.

    Args:
        terminal_candidates: All candidates with shape ``[B,L,D]``.
        history: Cached forward frames with shape ``[K+1,L,N,D]``.
        gamma: Forward step size used by the replay force.
        epsilon: Positive smoothing constant.
        exponent_s: Potential exponent ``s``.
        beta: Positive proximal optimizer step size.
        proximal_steps: Inner fixed-point iterations ``T`` per history frame.

    Returns:
        ``(trajectory, residuals)`` where trajectory has shape ``[K+1,B,L,D]``
        and residuals has shape ``[K,B,L]``.
    """
    terminal_candidates = np.asarray(terminal_candidates, dtype=np.float64)
    history = np.asarray(history, dtype=np.float64)
    if terminal_candidates.ndim != 3 or history.ndim != 4:
        raise ValueError("expected terminal [B,L,D] and history [K+1,L,N,D]")
    if terminal_candidates.shape[1] != history.shape[1]:
        raise ValueError("terminal and history slot counts differ")
    if terminal_candidates.shape[2] != history.shape[3]:
        raise ValueError("terminal and history dimensions differ")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if proximal_steps < 1:
        raise ValueError("proximal_steps must be at least one")

    forward_steps = history.shape[0] - 1
    candidate_count, slot_count, dimension = terminal_candidates.shape
    trajectory = np.empty((forward_steps + 1, candidate_count, slot_count, dimension), dtype=np.float64)  # shape: [K+1,B,L,D]
    residuals = np.empty((forward_steps, candidate_count, slot_count), dtype=np.float64)  # shape: [K,B,L]

    current = terminal_candidates.copy()  # shape: [B,L,D]
    trajectory[forward_steps] = current

    for frame in range(forward_steps, 0, -1):
        anchor = current.copy()  # y^j, shape: [B,L,D]
        value = anchor.copy()  # v_0 = y^j, shape: [B,L,D]
        particles = history[frame]  # literal post-forward frame, shape: [L,N,D]

        for _ in range(proximal_steps):
            force = backward_field(value, particles, gamma, epsilon, exponent_s)  # shape: [B,L,D]
            delta = value - anchor - force  # shape: [B,L,D]
            value = value - beta * delta

        force = backward_field(value, particles, gamma, epsilon, exponent_s)  # shape: [B,L,D]
        delta = value - anchor - force  # shape: [B,L,D]
        residuals[frame - 1] = np.linalg.norm(delta, axis=2) / (
            1.0 + np.linalg.norm(anchor, axis=2)
        )  # shape: [B,L], coordinate axis D is reduced
        trajectory[frame - 1] = value
        current = value

    if not np.all(np.isfinite(trajectory)):
        raise FloatingPointError("EFS backward replay became non-finite")
    return trajectory, residuals


def self_check() -> dict[str, float]:
    """Compare the batched forward field with the explicit EFS particle sum."""
    rng = np.random.default_rng(7)
    particles = rng.normal(size=(2, 5, 2))  # shape: [L=2,N=5,D=2]
    vectorized = forward_field(particles, epsilon=0.1, exponent_s=0.0)
    explicit = np.zeros_like(vectorized)  # shape: [L,N,D]

    for slot in range(particles.shape[0]):
        for query in range(particles.shape[1]):
            for other in range(particles.shape[1]):
                if query == other:
                    continue
                displacement = particles[slot, query] - particles[slot, other]  # [D]
                explicit[slot, query] += potential_gradient(displacement, epsilon=0.1, exponent_s=0.0)
            explicit[slot, query] /= float(particles.shape[1] - 1)

    field_error = float(np.max(np.abs(vectorized - explicit)))
    center_drift = float(np.max(np.abs(np.sum(vectorized, axis=1))))
    assert field_error < 1.0e-12, field_error
    assert center_drift < 1.0e-12, center_drift
    return {"field_max_abs_error": field_error, "center_drift": center_drift}


if __name__ == "__main__":
    print(self_check())

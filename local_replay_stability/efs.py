"""Single-plane Estimation-Free Sampling (EFS) mathematics.

Purpose:
    Evolve one pooled particle field, carry passive held-out sources forward,
    and replay generated sources backward through the saved field.
Dependencies:
    NumPy only.
Outputs:
    Arrays returned to ``run.py``. Running this file prints one algebra check.
Exact command:
    python efs.py
"""

from __future__ import annotations

import numpy as np


def potential_gradient(difference: np.ndarray, epsilon: float, exponent_s: float) -> np.ndarray:
    """Return ``grad W`` for differences shaped ``[...,D]``.

    Implements

        grad W(z) = z - z / (||z||^2 + epsilon)^(s/2 + 1).
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if exponent_s <= 0.0:
        raise ValueError("exponent_s must be positive")

    difference = np.asarray(difference, dtype=np.float64)
    squared_norm = np.sum(difference * difference, axis=-1, keepdims=True)  # [...,D] -> [...,1]
    inverse_power = np.power(squared_norm + epsilon, -(exponent_s / 2.0 + 1.0))  # shape: [...,1], broadcast across coordinate D
    return difference * (1.0 - inverse_power)  # shape: [...,D]


def rms_radius(particles: np.ndarray) -> float:
    """Return RMS distance from the center of one plane ``[N,D]``."""
    center = np.mean(particles, axis=0, keepdims=True)  # shape: [1,D]
    centered = particles - center  # [N,D] - [1,D] -> [N,D]
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


def mean_pairwise_distance(particles: np.ndarray) -> float:
    """Return the exact mean distance over ordered distinct pairs in ``[N,D]``."""
    particles = np.asarray(particles, dtype=np.float64)
    particle_count = particles.shape[0]
    squared_norm = np.sum(particles * particles, axis=1)  # shape: [N]
    squared_distance = (
        squared_norm[:, None] + squared_norm[None, :] - 2.0 * (particles @ particles.T)
    )  # [N,1] + [1,N] - [N,N] -> [N,N]
    np.maximum(squared_distance, 0.0, out=squared_distance)
    np.fill_diagonal(squared_distance, 0.0)
    return float(np.sum(np.sqrt(squared_distance)) / (particle_count * (particle_count - 1)))


def source_mean_pairwise_distance(sources: np.ndarray) -> np.ndarray:
    """Return mean internal particle distance for sources shaped ``[G,P,D]`` as ``[G]``."""
    sources = np.asarray(sources, dtype=np.float64)
    group_count, particle_count, _ = sources.shape
    if particle_count == 1:
        return np.zeros(group_count, dtype=np.float64)  # one point has no internal pair
    total = np.zeros(group_count, dtype=np.float64)  # shape: [G]
    pair_count = 0
    for first in range(particle_count):
        for second in range(first + 1, particle_count):
            difference = sources[:, first] - sources[:, second]  # [G,D] - [G,D] -> [G,D]
            total += np.linalg.norm(difference, axis=1)  # shape: [G]
            pair_count += 1
    return total / float(pair_count)


def forward_field(particles: np.ndarray, epsilon: float, exponent_s: float) -> np.ndarray:
    """Return the exact EFS field for one pooled plane ``[N,D]``.

    Implements

        field_i = 1/(N-1) sum_(a != i) grad W(x_i-x_a).

    Scalar ``[N,N]`` distance matrices avoid allocating ``[N,N,D]``.
    """
    particles = np.asarray(particles, dtype=np.float64)
    if particles.ndim != 2 or particles.shape[0] < 2:
        raise ValueError("particles must have shape [N,D] with N >= 2")
    if epsilon <= 0.0 or exponent_s <= 0.0:
        raise ValueError("epsilon and exponent_s must be positive")
    if not np.all(np.isfinite(particles)):
        raise FloatingPointError("forward particles contain non-finite values")

    particle_count = particles.shape[0]
    squared_norm = np.sum(particles * particles, axis=1)  # shape: [N]
    pair_dot = particles @ particles.T  # [N,D] @ [D,N] -> [N,N]
    squared_distance = squared_norm[:, None] + squared_norm[None, :] - 2.0 * pair_dot  # [N,1] + [1,N] - [N,N] -> [N,N]
    np.maximum(squared_distance, 0.0, out=squared_distance)

    weights = np.power(
        squared_distance + epsilon, -(exponent_s / 2.0 + 1.0)
    )  # shape: [N,N], scalar inverse-power weight for every ordered pair
    np.fill_diagonal(weights, 0.0)  # particle i never acts on itself

    particle_sum = np.sum(particles, axis=0, keepdims=True)  # shape: [1,D]
    attractive_sum = particle_count * particles - particle_sum  # shape: [N,D]
    weighted_particle_sum = weights @ particles  # [N,N] @ [N,D] -> [N,D]
    weight_sum = np.sum(weights, axis=1, keepdims=True)  # shape: [N,1]
    repulsive_sum = particles * weight_sum - weighted_particle_sum  # shape: [N,D]
    field = (attractive_sum - repulsive_sum) / float(particle_count - 1)  # shape: [N,D]

    if not np.all(np.isfinite(field)):
        raise FloatingPointError("EFS forward field became non-finite")
    return field


def forward_history(
    initial_particles: np.ndarray,
    gamma: float,
    epsilon: float,
    exponent_s: float,
    steps: int,
    log_every: int = 100,
    save_steps: list[int] | tuple[int, ...] | np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray | bool | str]]:
    """Evolve one pooled plane and return saved frames plus logs.

    Euler equation:

        x^(j+1) = x^j - gamma * field(x^j).

    With ``save_steps=None``, every frame is returned as ``[J+1,N,D]`` for
    replay. A calibration search may instead request selected frame numbers;
    their actual indices are returned in ``diagnostics["saved_step"]``.
    """
    initial_particles = np.asarray(initial_particles, dtype=np.float64)
    if initial_particles.ndim != 2:
        raise ValueError("initial_particles must have shape [N,D]")
    if gamma <= 0.0 or steps < 1 or log_every < 0:
        raise ValueError("gamma and steps must be positive; log_every must be non-negative")

    keep_all = save_steps is None
    if keep_all:
        history_buffer = np.empty((steps + 1, *initial_particles.shape), dtype=np.float64)  # shape: [J+1,N,D]
        history_buffer[0] = initial_particles
        requested_steps: set[int] = set()
        sparse_frames: list[np.ndarray] = []
        sparse_step_values: list[int] = []
    else:
        requested_steps = {int(value) for value in np.asarray(save_steps).reshape(-1)}
        if any(value < 0 or value > steps for value in requested_steps):
            raise ValueError("save_steps must lie between zero and steps")
        requested_steps.update((0, steps))
        sparse_frames = [initial_particles.copy()]  # first saved frame, shape: [N,D]
        sparse_step_values = [0]

    current = initial_particles.copy()  # shape: [N,D]
    initial_center = np.mean(current, axis=0)  # shape: [D]

    logged_step: list[int] = []
    logged_distance: list[float] = []
    logged_radius: list[float] = []
    logged_force: list[float] = []
    logged_update: list[float] = []
    failure_reason = ""
    completed_steps = 0

    def record(step: int, force_norm: float, update_norm: float) -> None:
        distance = mean_pairwise_distance(current)
        radius = rms_radius(current)
        logged_step.append(step)
        logged_distance.append(distance)
        logged_radius.append(radius)
        logged_force.append(force_norm)
        logged_update.append(update_norm)
        print(
            f"forward step {step:5d}/{steps}: mean_particle_distance={distance:.8f} "
            f"rms_radius={radius:.8f} max_force={force_norm:.6e} max_update={update_norm:.6e}"
        )

    if log_every:
        record(0, 0.0, 0.0)

    for step in range(1, steps + 1):
        try:
            field = forward_field(current, epsilon, exponent_s)  # shape: [N,D]
            update = gamma * field  # shape: [N,D]
            current = current - update  # shape: [N,D]
            if not np.all(np.isfinite(update)) or not np.all(np.isfinite(current)):
                raise FloatingPointError("forward update became non-finite")
        except FloatingPointError as error:
            failure_reason = str(error)
            break

        if keep_all:
            history_buffer[step] = current
        elif step in requested_steps:
            sparse_frames.append(current.copy())  # one requested [N,D] checkpoint
            sparse_step_values.append(step)
        completed_steps = step
        if log_every and (step % log_every == 0 or step == steps):
            record(step, float(np.max(np.linalg.norm(field, axis=1))), float(np.max(np.linalg.norm(update, axis=1))))

    if keep_all:
        history = history_buffer[: completed_steps + 1]  # shape: [actual_J+1,N,D]
        saved_step = np.arange(completed_steps + 1, dtype=np.int64)  # shape: [actual_J+1]
    else:
        if sparse_step_values[-1] != completed_steps:
            sparse_frames.append(current.copy())
            sparse_step_values.append(completed_steps)
        history = np.stack(sparse_frames, axis=0)  # shape: [saved_frames,N,D]
        saved_step = np.asarray(sparse_step_values, dtype=np.int64)  # shape: [saved_frames]

    center_drift = float(np.linalg.norm(np.mean(current, axis=0) - initial_center))
    diagnostics: dict[str, np.ndarray | bool | str] = {
        "finite": not bool(failure_reason),
        "failure_reason": failure_reason,
        "completed_steps": np.asarray(completed_steps, dtype=np.int64),
        "saved_step": saved_step,
        "log_step": np.asarray(logged_step, dtype=np.int64),  # shape: [F]
        "mean_pair_distance": np.asarray(logged_distance, dtype=np.float64),  # shape: [F]
        "rms_radius": np.asarray(logged_radius, dtype=np.float64),  # shape: [F]
        "max_force_norm": np.asarray(logged_force, dtype=np.float64),  # shape: [F]
        "max_update_norm": np.asarray(logged_update, dtype=np.float64),  # shape: [F]
        "center_drift": np.asarray(center_drift, dtype=np.float64),
    }
    return history, diagnostics


def replay_field(candidates: np.ndarray, particles: np.ndarray, gamma: float, epsilon: float, exponent_s: float) -> np.ndarray:
    """Evaluate the paper replay field for queries ``[Q,D]`` in memory ``[N,D]``.

    Implements

        F_j(v_q) = gamma/N sum_i grad W(v_q-x_i^j).

    Queries are passive: they feel the memory but never affect it or one another.
    """
    candidates = np.asarray(candidates, dtype=np.float64)
    particles = np.asarray(particles, dtype=np.float64)
    if gamma <= 0.0 or epsilon <= 0.0 or exponent_s <= 0.0:
        raise ValueError("gamma, epsilon, and exponent_s must be positive")
    if candidates.ndim != 2 or particles.ndim != 2:
        raise ValueError("candidates and particles must have shapes [Q,D] and [N,D]")
    if candidates.shape[1] != particles.shape[1]:
        raise ValueError("candidate and memory dimensions differ")

    particle_count = particles.shape[0]
    candidate_norm = np.sum(candidates * candidates, axis=1, keepdims=True)  # [Q,1]
    particle_norm = np.sum(particles * particles, axis=1)[None, :]  # [1,N]
    pair_dot = candidates @ particles.T  # [Q,D] @ [D,N] -> [Q,N]
    squared_distance = candidate_norm + particle_norm - 2.0 * pair_dot  # shape: [Q,N]
    np.maximum(squared_distance, 0.0, out=squared_distance)

    weights = np.power(squared_distance + epsilon, -(exponent_s / 2.0 + 1.0))  # shape: [Q,N]
    particle_sum = np.sum(particles, axis=0, keepdims=True)  # shape: [1,D]
    attractive_sum = particle_count * candidates - particle_sum  # shape: [Q,D]
    weighted_particle_sum = weights @ particles  # [Q,N] @ [N,D] -> [Q,D]
    weight_sum = np.sum(weights, axis=1, keepdims=True)  # shape: [Q,1]
    repulsive_sum = candidates * weight_sum - weighted_particle_sum  # shape: [Q,D]
    field = gamma * (attractive_sum - repulsive_sum) / float(particle_count)  # [Q,D]

    if not np.all(np.isfinite(field)):
        raise FloatingPointError("EFS replay field became non-finite")
    return field


def passive_forward(
    initial_sources: np.ndarray, history: np.ndarray, gamma: float, epsilon: float, exponent_s: float
) -> np.ndarray:
    """Carry passive sources ``[G,P,D]`` through history ``[J+1,N,D]``.

    Returns a trajectory shaped ``[J+1,G,P,D]``. Held-out sources never enter
    the memory force calculation.
    """
    initial_sources = np.asarray(initial_sources, dtype=np.float64)
    history = np.asarray(history, dtype=np.float64)
    if initial_sources.ndim != 3 or history.ndim != 3:
        raise ValueError("expected initial_sources [G,P,D] and history [J+1,N,D]")
    if initial_sources.shape[2] != history.shape[2]:
        raise ValueError("source and history dimensions differ")

    steps = history.shape[0] - 1
    group_count, particle_count, dimension = initial_sources.shape
    trajectory = np.empty((steps + 1, group_count, particle_count, dimension), dtype=np.float64)  # [J+1,G,P,D]
    trajectory[0] = initial_sources
    current = initial_sources.reshape(group_count * particle_count, dimension).copy()  # [G,P,D] -> [G*P,D]

    for frame in range(steps):
        field = replay_field(current, history[frame], gamma, epsilon, exponent_s)  # [G*P,D]
        current = current - field  # passive paper forward step, shape: [G*P,D]
        trajectory[frame + 1] = current.reshape(group_count, particle_count, dimension)  # [G*P,D] -> [G,P,D]

    if not np.all(np.isfinite(trajectory)):
        raise FloatingPointError("passive forward trajectory became non-finite")
    return trajectory


def backward_replay(
    terminal_sources: np.ndarray,
    history: np.ndarray,
    gamma: float,
    epsilon: float,
    exponent_s: float,
    beta: float,
    proximal_steps: int,
    method_names: list[str] | None = None,
    log_every: int = 100,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Replay passive sources through one cached field.

    At cached post-forward frame ``j``:

        delta_t = v_t - y^j - F_j(v_t)
        v_(t+1) = v_t - beta * delta_t.

    Returns trajectory ``[J+1,G,P,D]``, residual ``[J,G,P]``, and logs.
    """
    terminal_sources = np.asarray(terminal_sources, dtype=np.float64)
    history = np.asarray(history, dtype=np.float64)
    if terminal_sources.ndim != 3 or history.ndim != 3:
        raise ValueError("expected terminal_sources [G,P,D] and history [J+1,N,D]")
    if terminal_sources.shape[2] != history.shape[2]:
        raise ValueError("source and history dimensions differ")
    if beta <= 0.0 or proximal_steps < 1 or log_every < 0:
        raise ValueError("beta and proximal_steps must be positive; log_every must be non-negative")

    steps = history.shape[0] - 1
    group_count, particle_count, dimension = terminal_sources.shape
    if method_names is None:
        method_names = ["candidate"] * group_count
    if len(method_names) != group_count:
        raise ValueError("method_names must contain one name per source")

    trajectory = np.empty((steps + 1, group_count, particle_count, dimension), dtype=np.float64)  # [J+1,G,P,D]
    residual = np.empty((steps, group_count, particle_count), dtype=np.float64)  # [J,G,P]
    current = terminal_sources.copy()  # shape: [G,P,D]
    trajectory[steps] = current

    logged_step: list[int] = []
    logged_distance: list[np.ndarray] = []
    logged_total: list[np.ndarray] = []
    logged_displacement: list[np.ndarray] = []
    method_order = list(dict.fromkeys(method_names))

    def record(reverse_step: int, previous: np.ndarray | None) -> None:
        distance = source_mean_pairwise_distance(current)  # shape: [G]
        total = np.sum(current, axis=1)  # shape: [G,D]
        displacement = np.zeros_like(total) if previous is None else total - np.sum(previous, axis=1)  # [G,D]
        logged_step.append(reverse_step)
        logged_distance.append(distance)
        logged_total.append(total)
        logged_displacement.append(displacement)
        print(f"backward step {reverse_step:5d}/{steps}")
        for name in method_order:
            selected = np.asarray([value == name for value in method_names])  # shape: [G]
            witness = int(np.flatnonzero(selected)[0])
            movement = np.linalg.norm(displacement[selected], axis=1)
            print(
                f"  {name}: median_particle_distance={np.median(distance[selected]):.8f} "
                f"mean_total_displacement={np.mean(movement):.6e} "
                f"witness_total={np.array2string(total[witness], precision=5)}"
            )

    if log_every:
        record(0, None)

    for frame in range(steps, 0, -1):
        anchor = current.copy()  # y^j, shape: [G,P,D]
        value = anchor.reshape(group_count * particle_count, dimension).copy()  # [G,P,D] -> [G*P,D]
        flat_anchor = value.copy()  # shape: [G*P,D]
        memory = history[frame]  # literal post-forward frame j, shape: [N,D]

        for _ in range(proximal_steps):
            force = replay_field(value, memory, gamma, epsilon, exponent_s)  # shape: [G*P,D]
            delta = value - flat_anchor - force  # fixed-point equation residual, shape: [G*P,D]
            value = value - beta * delta  # proximal update, shape: [G*P,D]

        force = replay_field(value, memory, gamma, epsilon, exponent_s)
        delta = value - flat_anchor - force  # shape: [G*P,D]
        value_grouped = value.reshape(group_count, particle_count, dimension)  # [G*P,D] -> [G,P,D]
        residual[frame - 1] = np.linalg.norm(delta.reshape(group_count, particle_count, dimension), axis=2) / (
            1.0 + np.linalg.norm(anchor, axis=2)
        )  # shape: [G,P]

        trajectory[frame - 1] = value_grouped
        current = value_grouped
        reverse_step = steps - frame + 1
        if log_every and (reverse_step % log_every == 0 or reverse_step == steps):
            record(reverse_step, anchor)

    if not np.all(np.isfinite(trajectory)) or not np.all(np.isfinite(residual)):
        raise FloatingPointError("EFS backward replay became non-finite")

    diagnostics = {
        "reverse_step": np.asarray(logged_step, dtype=np.int64),  # shape: [F]
        "mean_pair_distance": np.asarray(logged_distance, dtype=np.float64),  # shape: [F,G]
        "total_position": np.asarray(logged_total, dtype=np.float64),  # shape: [F,G,D]
        "displacement": np.asarray(logged_displacement, dtype=np.float64),  # shape: [F,G,D]
    }
    return trajectory, residual, diagnostics


def self_check() -> dict[str, float]:
    """Run one tiny finite-difference and literal-sum algebra check."""
    rng = np.random.default_rng(7)
    particles = rng.normal(size=(6, 4))  # shape: [N=6,D=4]
    epsilon = 0.2
    exponent_s = 2.0

    vectorized = forward_field(particles, epsilon, exponent_s)  # shape: [N,D]
    literal = np.empty_like(particles)
    for particle in range(particles.shape[0]):
        difference = particles[particle] - particles  # [D] - [N,D] -> [N,D]
        difference[particle] = 0.0
        gradient = potential_gradient(difference, epsilon, exponent_s)  # shape: [N,D]
        literal[particle] = np.sum(gradient, axis=0) / (particles.shape[0] - 1)

    point = rng.normal(size=4)  # shape: [D]
    direction = rng.normal(size=4)  # shape: [D]
    direction /= np.linalg.norm(direction)
    step = 1.0e-6

    def potential(value: np.ndarray) -> float:
        squared = float(value @ value)
        return 0.5 * squared + (squared + epsilon) ** (-exponent_s / 2.0) / exponent_s

    finite_difference = (potential(point + step * direction) - potential(point - step * direction)) / (2.0 * step)
    analytic = float(potential_gradient(point, epsilon, exponent_s) @ direction)
    field_error = float(np.max(np.abs(vectorized - literal)))
    gradient_error = abs(finite_difference - analytic)
    center_force = float(np.linalg.norm(np.sum(vectorized, axis=0)))

    full_history, _ = forward_history(particles, gamma=1.0e-4, epsilon=epsilon, exponent_s=exponent_s, steps=3, log_every=0)
    sparse_history, sparse_diagnostics = forward_history(
        particles, gamma=1.0e-4, epsilon=epsilon, exponent_s=exponent_s, steps=3, log_every=0, save_steps=[2, 3]
    )
    sparse_step = np.asarray(sparse_diagnostics["saved_step"], dtype=np.int64)
    sparse_error = float(np.max(np.abs(sparse_history - full_history[sparse_step])))
    assert field_error < 1.0e-12, field_error
    assert gradient_error < 1.0e-8, gradient_error
    assert center_force < 1.0e-12, center_force
    assert sparse_error < 1.0e-15, sparse_error
    return {
        "field_max_abs_error": field_error,
        "gradient_directional_error": gradient_error,
        "center_force_norm": center_force,
        "sparse_history_max_abs_error": sparse_error,
    }


if __name__ == "__main__":
    print(self_check())

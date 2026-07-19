"""Exact Estimation-Free Sampling fields for Hypothesis 4.5.

Purpose:
    Evolve either corresponding-slot planes or one pooled particle plane,
    transport passive target queries, and replay complete generated sources.

Dependencies:
    NumPy only.

Outputs:
    Numerical arrays returned to run.py and search.py. Running this file prints
    one algebra self-check and does not create files.

Exact command:
    python efs.py
"""

from __future__ import annotations

import numpy as np


def potential_gradient(displacement: np.ndarray, epsilon: float, exponent_s: float) -> np.ndarray:
    """Evaluate ``grad W(z)`` for displacement vectors shaped ``[...,D]``.

    Equation:
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

    This allocates one scalar ``[N,N]`` matrix, never ``[N,N,D]``.
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


def rms_radius(particles: np.ndarray) -> float:
    """Return RMS distance from the center for a plane shaped ``[N,D]``."""
    particles = np.asarray(particles, dtype=np.float64)
    if particles.ndim != 2:
        raise ValueError("particles must have shape [N,D]")
    centered = particles - np.mean(particles, axis=0, keepdims=True)  # [N,D] - [1,D]
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


def forward_field(particles: np.ndarray, epsilon: float, exponent_s: float) -> np.ndarray:
    """Compute exact EFS fields for ``[N,D]`` or slot planes ``[P,C,D]``.

    Equation for particle ``i``:

        field_i = 1/(N-1) sum_(a != i) grad W(x_i-x_a).

    A two-dimensional input is one pooled plane. A three-dimensional input is
    ``P`` independent literal fields, one for each corresponding slot. Scalar
    distance matrices replace the much larger displacement tensor.
    """
    particles = np.asarray(particles, dtype=np.float64)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if not np.all(np.isfinite(particles)):
        raise FloatingPointError("forward particles contain non-finite values")

    if particles.ndim == 3:
        if particles.shape[1] < 2:
            raise ValueError("slot particles must have shape [P,C,D] with C >= 2")

        source_count = particles.shape[1]
        squared_norm = np.sum(particles * particles, axis=2)  # shape: [P,C]
        pair_dot = particles @ np.swapaxes(particles, 1, 2)  # [P,C,D] @ [P,D,C] -> [P,C,C]
        squared_distance = (
            squared_norm[:, :, None] + squared_norm[:, None, :] - 2.0 * pair_dot
        )  # [P,C,1] + [P,1,C] - [P,C,C] -> [P,C,C]
        np.maximum(squared_distance, 0.0, out=squared_distance)

        weights = np.power(
            squared_distance + epsilon, -(exponent_s / 2.0 + 1.0)
        )  # shape: [P,C,C], one scalar coefficient per slot and ordered pair
        diagonal = np.arange(source_count)
        weights[:, diagonal, diagonal] = 0.0  # source c does not act on itself

        particle_sum = np.sum(particles, axis=1, keepdims=True)  # shape: [P,1,D]
        attractive_sum = source_count * particles - particle_sum  # shape: [P,C,D]
        weighted_particle_sum = weights @ particles  # [P,C,C] @ [P,C,D] -> [P,C,D]
        weight_sum = np.sum(weights, axis=2, keepdims=True)  # shape: [P,C,1]
        repulsive_sum = particles * weight_sum - weighted_particle_sum  # shape: [P,C,D]
        field = (attractive_sum - repulsive_sum) / float(source_count - 1)
        if not np.all(np.isfinite(field)):
            raise FloatingPointError("slot EFS forward field became non-finite")
        return field

    if particles.ndim != 2 or particles.shape[0] < 2:
        raise ValueError("particles must have shape [N,D] or [P,C,D]")

    particle_count = particles.shape[0]
    squared_norm = np.sum(particles * particles, axis=1)  # shape: [N]
    pair_dot = particles @ particles.T  # [N,D] @ [D,N] -> [N,N]
    squared_distance = (
        squared_norm[:, None] + squared_norm[None, :] - 2.0 * pair_dot
    )  # singleton axes broadcast over both particle axes -> [N,N]
    np.maximum(squared_distance, 0.0, out=squared_distance)

    weights = np.power(
        squared_distance + epsilon, -(exponent_s / 2.0 + 1.0)
    )  # shape: [N,N], inverse-power coefficient for every ordered pair
    np.fill_diagonal(weights, 0.0)  # particle i does not act on itself

    particle_sum = np.sum(particles, axis=0, keepdims=True)  # shape: [1,D]
    attractive_sum = particle_count * particles - particle_sum  # shape: [N,D]

    weighted_particle_sum = weights @ particles  # [N,N] @ [N,D] -> [N,D]
    weight_sum = np.sum(weights, axis=1, keepdims=True)  # shape: [N,1]
    repulsive_sum = particles * weight_sum - weighted_particle_sum  # shape: [N,D]

    field = (attractive_sum - repulsive_sum) / float(particle_count - 1)
    if not np.all(np.isfinite(field)):
        raise FloatingPointError("EFS forward field became non-finite")
    return field


def _plane_scale_values(particles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return pair distance and RMS radius per empirical field as [H] arrays."""
    planes = particles[None, :, :] if particles.ndim == 2 else particles  # [N,D] -> [1,N,D], or [P,C,D]
    pair_distance = np.asarray([mean_pairwise_distance(plane) for plane in planes])  # shape: [H]
    radius = np.asarray([rms_radius(plane) for plane in planes])  # shape: [H]
    return pair_distance, radius


def forward_history(
    initial_particles: np.ndarray,
    gamma: float,
    epsilon: float,
    exponent_s: float,
    steps: int,
    log_every: int = 10,
    save_steps: list[int] | tuple[int, ...] | np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray | int | bool | str]]:
    """Evolve fixed-duration fields and retain all or selected frames.

    Euler update:

        x^(j+1) = x^j - gamma * forward_field(x^j).

    Args:
        initial_particles: One pooled plane [N,D] or corresponding-slot
            planes [P,C,D].
        save_steps: Optional frame indices used by the calibration grid. When
            omitted, every frame is saved for literal backward replay.

    Returns:
        (history, diagnostics). Full history is [J+1,N,D] or [J+1,P,C,D].
        Sparse histories record actual frame numbers in diagnostics["saved_step"]
        and must not be used for replay.
    """
    initial_particles = np.asarray(initial_particles, dtype=np.float64)
    if initial_particles.ndim not in (2, 3):
        raise ValueError("initial_particles must have shape [N,D] or [P,C,D]")
    if gamma <= 0.0 or steps < 1 or log_every < 0:
        raise ValueError("gamma and steps must be positive; log_every must be non-negative")

    keep_all = save_steps is None
    requested_steps: set[int] = set()
    if keep_all:
        history_buffer = np.empty((steps + 1, *initial_particles.shape), dtype=np.float64)
        history_buffer[0] = initial_particles  # shape: [J+1,N,D] or [J+1,P,C,D]
        sparse_frames: list[np.ndarray] = []
        sparse_step_values: list[int] = []
    else:
        requested_steps = {int(value) for value in np.asarray(save_steps).reshape(-1)}
        if any(value < 0 or value > steps for value in requested_steps):
            raise ValueError("save_steps must lie between zero and steps")
        requested_steps.update((0, steps))
        sparse_frames = [initial_particles.copy()]
        sparse_step_values = [0]

    initial_pair, initial_radius = _plane_scale_values(initial_particles)
    log_steps: list[int] = [0]
    log_pair_distance: list[np.ndarray] = [initial_pair]
    log_rms_radius: list[np.ndarray] = [initial_radius]
    log_max_force: list[float] = [0.0]
    log_max_update: list[float] = [0.0]
    if log_every:
        print(
            f"forward step {0:5d}/{steps}: mean_pair_distance={np.mean(initial_pair):.8f} "
            f"rms_radius={np.mean(initial_radius):.8f}"
        )

    failure_reason = ""
    actual_steps = 0
    current = initial_particles.copy()

    # exact all-pair EFS is O(H*N^2) per frame. Approximate
    # neighborhoods belong in a later study only if this reference is unusable.
    for step in range(1, steps + 1):
        try:
            field = forward_field(current, epsilon, exponent_s)  # [N,D] or [P,C,D]
        except FloatingPointError as error:
            failure_reason = f"frame {step - 1}: {error}"
            break

        update = gamma * field  # same shape as current
        next_particles = current - update
        if not np.all(np.isfinite(next_particles)):
            failure_reason = f"frame {step}: forward update became non-finite"
            break

        current = next_particles
        actual_steps = step
        if keep_all:
            history_buffer[step] = current
        elif step in requested_steps:
            sparse_frames.append(current.copy())
            sparse_step_values.append(step)

        max_force = float(np.max(np.linalg.norm(field, axis=-1)))
        max_update = float(np.max(np.linalg.norm(update, axis=-1)))
        is_log = bool(log_every and (step % log_every == 0 or step == steps))
        if is_log:
            pair_distance, radius = _plane_scale_values(current)
            log_steps.append(step)
            log_pair_distance.append(pair_distance)
            log_rms_radius.append(radius)
            log_max_force.append(max_force)
            log_max_update.append(max_update)
            print(
                f"forward step {step:5d}/{steps}: "
                f"mean_pair_distance={np.mean(pair_distance):.8f} "
                f"rms_radius={np.mean(radius):.8f} max_update={max_update:.8e}"
            )

    if log_steps[-1] != actual_steps:
        pair_distance, radius = _plane_scale_values(current)
        log_steps.append(actual_steps)
        log_pair_distance.append(pair_distance)
        log_rms_radius.append(radius)
        log_max_force.append(np.nan)
        log_max_update.append(np.nan)

    if keep_all:
        history = history_buffer[: actual_steps + 1]
        saved_step = np.arange(actual_steps + 1, dtype=np.int64)
    else:
        if sparse_step_values[-1] != actual_steps:
            sparse_frames.append(current.copy())
            sparse_step_values.append(actual_steps)
        history = np.stack(sparse_frames, axis=0)
        saved_step = np.asarray(sparse_step_values, dtype=np.int64)

    diagnostics: dict[str, np.ndarray | int | bool | str] = {
        "saved_step": saved_step,
        "log_step": np.asarray(log_steps, dtype=np.int64),
        "mean_pair_distance": np.stack(log_pair_distance),  # shape: [F,H]
        "rms_radius": np.stack(log_rms_radius),  # shape: [F,H]
        "max_force_norm": np.asarray(log_max_force, dtype=np.float64),
        "max_update_norm": np.asarray(log_max_update, dtype=np.float64),
        "actual_steps": actual_steps,
        "finite": not bool(failure_reason),
        "failure_reason": failure_reason,
    }
    return history, diagnostics

def replay_field(candidates: np.ndarray, particles: np.ndarray, gamma: float, epsilon: float, exponent_s: float) -> np.ndarray:
    """Evaluate the vanilla EFS replay field for queries shaped ``[Q,D]``.

    Equation at cached frame ``j``:

        F_j(v_q) = gamma/N sum_i grad W(v_q-x_i^j).

    Queries share the empirical plane but do not occur in one another's force
    sums. Candidate batching therefore does not change the mathematics.
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
        raise FloatingPointError("EFS replay field became non-finite")
    return field


def layout_replay_field(
    candidates: np.ndarray,
    particles: np.ndarray,
    gamma: float,
    epsilon: float,
    exponent_s: float,
) -> np.ndarray:
    """Evaluate passive candidates ``[G,P,D]`` in either memory layout.

    For a pooled plane ``[N,D]``, all ``G*P`` queries use the same field. For
    slot planes ``[P,C,D]``, query ``(g,p)`` uses only the ``C`` training
    particles in slot ``p``:

        F_(g,p) = gamma/C sum_c grad W(v_(g,p) - x_(p,c)).

    The slot implementation builds scalar ``[P,G,C]`` weights. It never builds
    a displacement array ``[P,G,C,D]`` and candidates never interact.
    """
    candidates = np.asarray(candidates, dtype=np.float64)
    particles = np.asarray(particles, dtype=np.float64)
    if candidates.ndim != 3:
        raise ValueError("candidates must have shape [G,P,D]")

    group_count, slot_count, dimension = candidates.shape
    if particles.ndim == 2:
        flat = candidates.reshape(group_count * slot_count, dimension)  # [G,P,D] -> [G*P,D]
        return replay_field(flat, particles, gamma, epsilon, exponent_s).reshape(
            group_count, slot_count, dimension
        )  # [G*P,D] -> [G,P,D]

    if particles.ndim != 3 or particles.shape[0] != slot_count or particles.shape[2] != dimension:
        raise ValueError("slot memory must have shape [P,C,D] matching candidates")
    if gamma <= 0.0 or epsilon <= 0.0:
        raise ValueError("gamma and epsilon must be positive")

    source_count = particles.shape[1]
    query = np.swapaxes(candidates, 0, 1)  # [G,P,D] -> [P,G,D]
    query_norm = np.sum(query * query, axis=2, keepdims=True)  # shape: [P,G,1]
    particle_norm = np.sum(particles * particles, axis=2)[:, None, :]  # shape: [P,1,C]
    pair_dot = query @ np.swapaxes(particles, 1, 2)  # [P,G,D] @ [P,D,C] -> [P,G,C]
    squared_distance = query_norm + particle_norm - 2.0 * pair_dot  # shape: [P,G,C]
    np.maximum(squared_distance, 0.0, out=squared_distance)

    weights = np.power(
        squared_distance + epsilon, -(exponent_s / 2.0 + 1.0)
    )  # shape: [P,G,C]
    particle_sum = np.sum(particles, axis=1)[:, None, :]  # shape: [P,1,D]
    attractive_sum = source_count * query - particle_sum  # shape: [P,G,D]
    weighted_particle_sum = weights @ particles  # [P,G,C] @ [P,C,D] -> [P,G,D]
    weight_sum = np.sum(weights, axis=2, keepdims=True)  # shape: [P,G,1]
    repulsive_sum = query * weight_sum - weighted_particle_sum  # shape: [P,G,D]
    field = gamma * (attractive_sum - repulsive_sum) / float(source_count)
    if not np.all(np.isfinite(field)):
        raise FloatingPointError("slot EFS replay field became non-finite")
    return np.swapaxes(field, 0, 1)  # [P,G,D] -> [G,P,D]


def passive_forward(
    initial_queries: np.ndarray,
    history: np.ndarray,
    gamma: float,
    epsilon: float,
    exponent_s: float,
) -> np.ndarray:
    """Transport passive queries through a contiguous cached history.

    Queries may be complete sources [Q,P,D] or ordinary rows [Q,D]. A pooled
    history has shape [J+1,N,D]; a slot history has shape [J+1,P,C,D].
    Forward interval j -> j+1 uses the source frame history[j]:

        q^(j+1) = q^j - F_j(q^j).

    The queries never modify either empirical field.
    """
    values = np.asarray(initial_queries, dtype=np.float64)
    history = np.asarray(history, dtype=np.float64)
    was_rows = values.ndim == 2
    if was_rows:
        values = values[:, None, :]  # [Q,D] -> [Q,P=1,D]
    if values.ndim != 3 or history.ndim not in (3, 4):
        raise ValueError("expected queries [Q,P,D] and history [J+1,N,D] or [J+1,P,C,D]")
    if values.shape[2] != history.shape[-1]:
        raise ValueError("query and history dimensions differ")
    if history.ndim == 4 and values.shape[1] != history.shape[1]:
        raise ValueError("query slot count and slot history differ")

    current = values.copy()  # shape: [Q,P,D]
    for source_frame in range(history.shape[0] - 1):
        force = layout_replay_field(
            current, history[source_frame], gamma, epsilon, exponent_s
        )  # shape: [Q,P,D]
        current = current - force

    if not np.all(np.isfinite(current)):
        raise FloatingPointError("passive target transport became non-finite")
    return current[:, 0] if was_rows else current

def ensemble_mean_distance(ensembles: np.ndarray) -> np.ndarray:
    """Return within-ensemble mean distances for arrays shaped ``[G,P,D]``."""
    ensembles = np.asarray(ensembles, dtype=np.float64)
    if ensembles.ndim != 3 or ensembles.shape[1] < 1:
        raise ValueError("ensembles must have shape [G,P,D] with P >= 1")
    if ensembles.shape[1] == 1:
        return np.zeros(ensembles.shape[0], dtype=np.float64)  # one joint vector has no internal pair

    particle_count = ensembles.shape[1]
    difference = ensembles[:, :, None, :] - ensembles[:, None, :, :]  # [G,P,P,D]
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
    method_names: list[str] | None = None,
    log_every: int = 10,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Replay complete passive sources through pooled or corresponding-slot fields.

    At frame ``j``, for every generated particle:

        delta_t = v_t - y^j - F_j(v_t)
        v_(t+1) = v_t - beta * delta_t
        y^(j-1) = v_T.

    Args:
        terminal_ensembles: Candidate sources shaped ``[G,P,D]``.
        history: Pooled history ``[J+1,N,D]`` or slot history
            ``[J+1,P,C,D]``.
        method_names: Optional ``G`` labels. Console logs aggregate repeated
            labels while saved diagnostics retain all candidates.

    Returns:
        ``(trajectory, residuals, diagnostics)`` with primary shapes
        ``[J+1,G,P,D]`` and ``[J,G,P]``.
    """
    terminal_ensembles = np.asarray(terminal_ensembles, dtype=np.float64)
    history = np.asarray(history, dtype=np.float64)
    if terminal_ensembles.ndim != 3 or history.ndim not in (3, 4):
        raise ValueError("expected terminal [G,P,D] and pooled or slot history")
    if terminal_ensembles.shape[2] != history.shape[-1]:
        raise ValueError("terminal and history dimensions differ")
    if history.ndim == 4 and terminal_ensembles.shape[1] != history.shape[1]:
        raise ValueError("terminal source and history slot counts differ")
    if beta <= 0.0 or proximal_steps < 1 or log_every < 0:
        raise ValueError("beta and proximal_steps must be positive; log_every must be non-negative")

    forward_steps = history.shape[0] - 1
    group_count, particles_per_group, dimension = terminal_ensembles.shape
    if method_names is None:
        method_names = ["candidate"] * group_count
    if len(method_names) != group_count:
        raise ValueError("method_names must contain one label per candidate ensemble")

    trajectory = np.empty(
        (forward_steps + 1, group_count, particles_per_group, dimension), dtype=np.float64
    )  # shape: [J+1,G,P,D]
    residuals = np.empty((forward_steps, group_count, particles_per_group), dtype=np.float64)  # shape: [J,G,P]

    current = terminal_ensembles.copy()  # shape: [G,P,D]
    trajectory[forward_steps] = current
    log_steps: list[int] = [0]
    log_distance: list[np.ndarray] = [ensemble_mean_distance(current)]
    log_total: list[np.ndarray] = [np.sum(current, axis=1)]  # each: [G,D]
    log_displacement: list[np.ndarray] = [np.zeros((group_count, dimension))]

    method_order = list(dict.fromkeys(method_names))

    def print_state(reverse_step: int, distance: np.ndarray, total: np.ndarray, displacement: np.ndarray) -> None:
        """Print one witness vector and aggregate movement for each method."""
        print(f"backward step {reverse_step:5d}/{forward_steps}")
        for method in method_order:
            indices = np.asarray([name == method for name in method_names])  # shape: [G]
            witness = int(np.flatnonzero(indices)[0])
            displacement_norm = np.linalg.norm(displacement[indices], axis=1)
            print(
                f"  {method}: median_pair_distance={np.median(distance[indices]):.8f} "
                f"mean_displacement_norm={np.mean(displacement_norm):.8e} "
                f"witness_total={np.array2string(total[witness], precision=6)}"
            )

    if log_every:
        print_state(0, log_distance[0], log_total[0], log_displacement[0])

    for frame in range(forward_steps, 0, -1):
        anchor = current.copy()  # y^j, shape: [G,P,D]
        value = anchor.copy()  # v_0 = y^j, shape: [G,P,D]
        particles = history[frame]  # literal post-forward frame j, [N,D] or [P,C,D]

        for _ in range(proximal_steps):
            force = layout_replay_field(
                value, particles, gamma, epsilon, exponent_s
            )  # passive queries [G,P,D]; no candidate-candidate interactions
            delta = value - anchor - force  # shape: [G,P,D]
            value = value - beta * delta

        force = layout_replay_field(value, particles, gamma, epsilon, exponent_s)
        delta = value - anchor - force  # shape: [G,P,D]
        residuals[frame - 1] = np.linalg.norm(delta, axis=2) / (1.0 + np.linalg.norm(anchor, axis=2))  # shape: [G,P]

        trajectory[frame - 1] = value
        current = value
        reverse_step = forward_steps - frame + 1

        if log_every and (reverse_step % log_every == 0 or reverse_step == forward_steps):
            distance = ensemble_mean_distance(value)  # shape: [G]
            total = np.sum(value, axis=1)  # shape: [G,D]
            previous_total = np.sum(anchor, axis=1)  # shape: [G,D]
            displacement = total - previous_total  # complete reverse-frame movement, [G,D]
            print_state(reverse_step, distance, total, displacement)
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
    """Check pooled and eight-slot fields against literal pair sums."""
    rng = np.random.default_rng(7)
    pooled = rng.normal(size=(6, 4))  # shape: [N=6,D=4]
    slot_particles = rng.normal(size=(8, 6, 4))  # shape: [P=8,C=6,D=4]

    def literal(values: np.ndarray) -> np.ndarray:
        """Evaluate one [N,D] field with the equation's nested sums."""
        explicit = np.zeros_like(values)
        for query in range(values.shape[0]):
            for other in range(values.shape[0]):
                if query != other:
                    explicit[query] += potential_gradient(
                        values[query] - values[other], epsilon=0.1, exponent_s=2.0
                    )
            explicit[query] /= float(values.shape[0] - 1)
        return explicit

    pooled_vectorized = forward_field(pooled, epsilon=0.1, exponent_s=2.0)
    pooled_explicit = literal(pooled)
    slot_vectorized = forward_field(slot_particles, epsilon=0.1, exponent_s=2.0)
    slot_explicit = np.stack([literal(slot_particles[slot]) for slot in range(8)])  # eight [C,D] fields -> [P,C,D]
    queries = rng.normal(size=(3, 8, 4))  # shape: [G=3,P=8,D=4]
    slot_query = layout_replay_field(queries, slot_particles, 0.01, 0.1, 2.0)
    slot_query_explicit = np.stack([
        replay_field(queries[:, slot], slot_particles[slot], 0.01, 0.1, 2.0)
        for slot in range(8)
    ], axis=1)  # eight [G,D] fields -> [G,P,D]

    pooled_error = float(np.max(np.abs(pooled_vectorized - pooled_explicit)))
    slot_error = float(np.max(np.abs(slot_vectorized - slot_explicit)))
    slot_query_error = float(np.max(np.abs(slot_query - slot_query_explicit)))
    pooled_center_drift = float(np.max(np.abs(np.sum(pooled_vectorized, axis=0))))
    slot_center_drift = float(np.max(np.abs(np.sum(slot_vectorized, axis=1))))
    assert pooled_error < 1.0e-12, pooled_error
    assert slot_error < 1.0e-12, slot_error
    assert slot_query_error < 1.0e-12, slot_query_error
    assert pooled_center_drift < 1.0e-12, pooled_center_drift
    assert slot_center_drift < 1.0e-12, slot_center_drift
    return {
        "pooled_field_max_abs_error": pooled_error,
        "slot_field_max_abs_error": slot_error,
        "slot_query_max_abs_error": slot_query_error,
        "pooled_center_drift": pooled_center_drift,
        "slot_center_drift": slot_center_drift,
    }

if __name__ == "__main__":
    print(self_check())

"""Synthetic complete-source and controlled-target construction for H4.5.

This module creates arrays only. It performs no EFS evolution, evaluation,
plotting, or file writing.
"""

from __future__ import annotations

import numpy as np


METHOD_NAMES = np.asarray(("shared", "independent"), dtype="U16")


def arrange_memory(sources: np.ndarray, layout: str) -> np.ndarray:
    """Arrange complete sources ``[C,P,D]`` as one EFS memory layout.

    ``slot`` returns ``[P,C,D]`` so corresponding particle index ``p`` has its
    own literal EFS field. ``single`` returns the pooled control ``[C*P,D]``.
    Only storage and field membership change; source values are unchanged.
    """
    sources = np.asarray(sources, dtype=np.float64)
    if sources.ndim != 3:
        raise ValueError("sources must have shape [C,P,D]")
    if layout == "slot":
        return np.swapaxes(sources, 0, 1).copy()  # [C,P,D] -> [P,C,D]
    if layout == "single":
        return sources.reshape(-1, sources.shape[2]).copy()  # [C,P,D] -> [C*P,D]
    raise ValueError("layout must be 'slot' or 'single'")


def restore_sources(memory: np.ndarray, layout: str, particles_per_source: int) -> np.ndarray:
    """Restore one endpoint memory to complete sources shaped ``[C,P,D]``."""
    memory = np.asarray(memory, dtype=np.float64)
    if layout == "slot":
        if memory.ndim != 3 or memory.shape[0] != particles_per_source:
            raise ValueError("slot memory must have shape [P,C,D]")
        return np.swapaxes(memory, 0, 1).copy()  # [P,C,D] -> [C,P,D]
    if layout == "single":
        if memory.ndim != 2 or memory.shape[0] % particles_per_source:
            raise ValueError("single memory must have shape [C*P,D]")
        source_count = memory.shape[0] // particles_per_source
        return memory.reshape(source_count, particles_per_source, memory.shape[1]).copy()
    raise ValueError("layout must be 'slot' or 'single'")


def generate_grouped_sources(
    source_count: int,
    particles_per_source: int,
    dimension: int,
    center_std: float,
    particle_scale: float,
    noise_std: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create complete corresponding-particle sources before EFS layout.

    Equation:

        X_(c,p) = h_c + q_p + eta_(c,p).

    Here ``h_c`` is one complete-source state, ``q_p`` is a fixed particle
    identity, and ``eta`` is independent jitter. Particle index ``p`` therefore
    has the same meaning in every source.

    Returns:
        ``(sources, centers, template)`` shaped ``[C,P,D]``, ``[C,D]``, and
        ``[P,D]``.
    """
    if source_count < 4 or particles_per_source < 2 or dimension < 2:
        raise ValueError("need at least four sources, two particles per source, and D >= 2")
    if center_std <= 0.0 or particle_scale <= 0.0 or noise_std < 0.0:
        raise ValueError("center_std and particle_scale must be positive; noise_std must be non-negative")

    centers = rng.normal(scale=center_std, size=(source_count, dimension))  # shape: [C,D], shared state of each complete source
    template = rng.normal(
        scale=particle_scale, size=(particles_per_source, dimension)
    )  # shape: [P,D], declared correspondence between source particles
    template -= np.mean(template, axis=0, keepdims=True)  # [P,D] - [1,D]
    noise = rng.normal(scale=noise_std, size=(source_count, particles_per_source, dimension))  # shape: [C,P,D]
    sources = centers[:, None, :] + template[None, :, :] + noise  # [C,1,D] + [1,P,D] + [C,P,D] -> [C,P,D]
    return sources, centers, template


def parse_shared_lambda(text: str | None, selected_source_count: int) -> np.ndarray | None:
    """Parse an optional non-negative operator vector shaped ``[S]``."""
    if text is None:
        return None

    values = np.asarray(
        [float(value.strip()) for value in text.split(",")], dtype=np.float64
    )  # shape: [S] if the operator supplied the expected count
    if values.shape != (selected_source_count,):
        raise ValueError(f"--shared-lambda requires exactly {selected_source_count} comma-separated values")
    if np.any(values < 0.0) or not np.isclose(np.sum(values), 1.0, atol=1.0e-8):
        raise ValueError("--shared-lambda values must be non-negative and sum to one")
    return values / np.sum(values)


def build_target_trials(
    sources: np.ndarray,
    trial_count: int,
    selected_source_count: int,
    noise_std: float,
    rng: np.random.Generator,
    operator_lambda: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Create known clean and noise-matched held-out targets.

    For trial ``r``:

        clean_(r,p) = sum_s lambda_(r,s) X_(source[r,s],p)

        noisy_(r,p) = clean_(r,p) + xi_(r,p)

        std(xi_r) = noise_std * sqrt(1 - sum_s lambda_(r,s)^2).

    The added variance compensates for the variance reduction caused by
    averaging independent source noise. Neither target is inserted into the
    empirical EFS plane.

    One operator trial is appended when ``operator_lambda`` is supplied.
    Returned arrays use ``R=trial_count`` or ``trial_count+1``.
    """
    sources = np.asarray(sources, dtype=np.float64)
    if sources.ndim != 3:
        raise ValueError("sources must have shape [C,P,D]")
    if trial_count < 1 or selected_source_count < 2:
        raise ValueError("trial_count must be positive and selected_source_count at least two")
    if selected_source_count > sources.shape[0]:
        raise ValueError("not enough complete sources for one target trial")
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative")

    total_trials = trial_count + int(operator_lambda is not None)
    source_ids = np.empty((total_trials, selected_source_count), dtype=np.int64)  # [R,S]
    for trial in range(total_trials):
        source_ids[trial] = rng.choice(
            sources.shape[0], size=selected_source_count, replace=False
        )  # select S intact sources independently for each trial

    lambdas = rng.dirichlet(np.ones(selected_source_count, dtype=np.float64), size=trial_count)  # shape: [random_R,S]
    trial_kind = np.full(total_trials, "random", dtype="U16")  # shape: [R]
    if operator_lambda is not None:
        lambdas = np.concatenate((lambdas, operator_lambda[None, :]), axis=0)  # [random_R,S] + [1,S] -> [R,S]
        trial_kind[-1] = "operator"

    selected_sources = sources[source_ids]  # advanced indexing: [R,S,P,D]
    clean_targets = np.einsum("rs,rspd->rpd", lambdas, selected_sources)  # sum selected-source axis S -> [R,P,D]

    lambda_square_sum = np.sum(lambdas * lambdas, axis=1)  # shape: [R]
    jitter_std = noise_std * np.sqrt(np.maximum(1.0 - lambda_square_sum, 0.0))  # shape: [R]
    jitter = rng.normal(size=clean_targets.shape) * jitter_std[:, None, None]  # [R,P,D] * [R,1,1] -> [R,P,D]
    noisy_targets = clean_targets + jitter  # shape: [R,P,D]

    particles_per_source = sources.shape[1]
    independent_lambdas = np.empty(
        (total_trials, particles_per_source, selected_source_count), dtype=np.float64
    )  # shape: [R,P,S]
    for trial in range(total_trials):
        for particle in range(particles_per_source):
            independent_lambdas[trial, particle] = np.roll(
                lambdas[trial], particle % selected_source_count
            )  # cyclic source weights preserve the exact lambda values and entropy

    nondegenerate = np.ptp(lambdas, axis=1) > 1.0e-12  # shape: [R]
    return {
        "trial_kind": trial_kind,
        "source_ids": source_ids,
        "lambdas": lambdas,
        "independent_lambdas": independent_lambdas,
        "nondegenerate": nondegenerate,
        "clean_targets": clean_targets,
        "jitter_std": jitter_std,
        "jitter": jitter,
        "noisy_targets": noisy_targets,
    }


def build_terminal_candidates(
    terminal_sources: np.ndarray, source_ids: np.ndarray, lambdas: np.ndarray, independent_lambdas: np.ndarray
) -> np.ndarray:
    """Build shared and independent terminal candidates ``[R,2,P,D]``.

    Method axis 0 is shared and axis 1 is the falsification control. The shared
    contraction reuses one ``[S]`` vector for all ``P`` corresponding particles.
    """
    terminal_sources = np.asarray(terminal_sources, dtype=np.float64)
    selected_terminal = terminal_sources[source_ids]  # [C,P,D] indexed by [R,S] -> [R,S,P,D]
    shared = np.einsum("rs,rspd->rpd", lambdas, selected_terminal)  # [R,S] with [R,S,P,D], sum S -> [R,P,D]
    independent = np.einsum(
        "rps,rspd->rpd", independent_lambdas, selected_terminal
    )  # [R,P,S] gives every particle its cyclically shifted control -> [R,P,D]
    return np.stack((shared, independent), axis=1)  # shape: [R,M=2,P,D]

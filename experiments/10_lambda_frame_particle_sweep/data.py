"""Independent particle data, unordered source matching, and lambda fitting.

Purpose:
    Draw one chaotic particle population, group particles into sources only
    after sampling, retrieve unordered neighboring sources, and fit one shared
    nonnegative barycentric vector.
Dependencies:
    NumPy only.
Outputs:
    Arrays consumed by ``run.py`` and ``search.py``.
Exact command:
    python data.py
"""

from __future__ import annotations

import numpy as np


DATA_MODEL = "independent_chaotic_particles_v1"
UNIFORM_DATA_MODEL = "independent_uniform_box_v1"
METHOD_NAMES = np.asarray(
    ("shared_d1", "shared_du", "per_particle_d1", "per_particle_du"), dtype="U24"
)


def generate_chaotic_sources(
    source_count: int,
    particles_per_source: int,
    dimension: int,
    data_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return randomly grouped independent particles as sources ``[C,P,D]``.

    The final particle law is a nonlinear, multimodal mixture rather than one
    named Gaussian, uniform, shell, or exponential shape. Crucially, source
    identity is assigned only after every particle has been drawn, so it does
    not alter particle position.
    """
    if source_count < 4 or particles_per_source < 1 or dimension < 2:
        raise ValueError("need at least four sources, one particle per source, and D >= 2")
    if data_scale <= 0.0:
        raise ValueError("data_scale must be positive")

    particle_count = source_count * particles_per_source
    component_count = max(5, dimension + 3)

    # Draw the global distribution first. Source count then changes only how
    # many independent particles are sampled, not the law they come from.
    centers = rng.normal(size=(component_count, dimension))  # shape: [Q,D]
    linear_maps = rng.normal(size=(component_count, dimension, dimension))  # shape: [Q,D,D]
    warp_map = rng.normal(size=(dimension, dimension)) / np.sqrt(dimension)  # shape: [D,D]
    phases = rng.uniform(-np.pi, np.pi, size=(component_count, dimension))  # shape: [Q,D]

    # Every row below is drawn independently. None of these arrays contains a
    # source axis or a source-specific parameter.
    latent = rng.normal(size=(particle_count, dimension))  # shape: [N,D]
    component = rng.integers(0, component_count, size=particle_count)  # shape: [N]
    amplitude = np.exp(rng.uniform(np.log(0.20), np.log(2.50), size=(particle_count, 1)))  # [N,1]

    # Normalize the distribution parameters, not the sampled particles. This
    # keeps coordinates comparable without coupling one draw to another or
    # letting held-out targets alter memory-particle positions.
    centers -= np.mean(centers, axis=0, keepdims=True)  # [Q,D] - [1,D]
    centers /= np.maximum(np.std(centers, axis=0, keepdims=True), 1.0e-12)  # [Q,D] / [1,D]
    centers *= 1.5
    map_norm = np.sqrt(np.sum(linear_maps * linear_maps, axis=1, keepdims=True))  # [Q,D,D] -> [Q,1,D]
    linear_maps /= np.maximum(map_norm, 1.0e-12)  # broadcasts [Q,1,D] over input coordinate D

    # For particle n, select one global component matrix A_q and evaluate
    #
    #   x_n = c_q + a_n [ z_n A_q + 0.45 sin(z_n B + phi_q) + 0.08 z_n^3 ].
    #
    # ``linear_maps[component]`` has shape [N,D,D]. Einsum contracts the input
    # coordinate d, leaving one transformed [N,D] particle array. The sine and
    # cubic terms warp and thicken the modes so the result has no simple named
    # global shape.
    linear = np.einsum("nd,ndk->nk", latent, linear_maps[component])  # [N,D] with [N,D,D] -> [N,D]
    warped = np.sin(latent @ warp_map + phases[component])  # [N,D] @ [D,D] + [N,D] -> [N,D]
    particles = 0.5 * data_scale * (
        centers[component] + amplitude * (linear + 0.45 * warped + 0.08 * latent**3)
    )  # [N,D]

    # Source ownership is declared only here. The permutation makes it explicit
    # that consecutive draws do not acquire a shared center, family, or shape.
    particles = particles[rng.permutation(particle_count)]  # shape: [N,D]
    sources = particles.reshape(source_count, particles_per_source, dimension)  # [N,D] -> [C,P,D]
    if not np.all(np.isfinite(sources)):
        raise FloatingPointError("chaotic particle generator produced non-finite values")
    return sources


def generate_uniform_sources(
    source_count: int,
    particles_per_source: int,
    dimension: int,
    data_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return independent uniform-box particles grouped as ``[C,P,D]``.

    Each coordinate is sampled from ``[-data_scale, data_scale]``. This small
    control distribution exists for the two-dimensional ``P=1`` experiment.
    """
    if source_count < 4 or particles_per_source < 1 or dimension < 2:
        raise ValueError("need at least four sources, one particle per source, and D >= 2")
    if data_scale <= 0.0:
        raise ValueError("data_scale must be positive")

    particle_count = source_count * particles_per_source
    particles = rng.uniform(-data_scale, data_scale, size=(particle_count, dimension))  # shape: [N,D]
    particles = particles[rng.permutation(particle_count)]  # source grouping is independent of draw order
    return particles.reshape(source_count, particles_per_source, dimension)  # [N,D] -> [C,P,D]


def hold_out_sources(sources: np.ndarray, heldout_count: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Remove complete target sources before EFS and return both groups."""
    source_count = sources.shape[0]
    if not 1 <= heldout_count < source_count - 3:
        raise ValueError("heldout_count must leave at least four memory sources")
    target_ids = np.sort(rng.choice(source_count, size=heldout_count, replace=False))  # shape: [R]
    memory_mask = np.ones(source_count, dtype=bool)  # shape: [C+R]
    memory_mask[target_ids] = False
    memory_ids = np.flatnonzero(memory_mask)  # shape: [C]
    return {
        "memory_sources": sources[memory_ids].copy(),  # shape: [C,P,D]
        "target_sources": sources[target_ids].copy(),  # shape: [R,P,D]
        "memory_source_ids": memory_ids,
        "target_source_ids": target_ids,
    }


def coordinate_scale(sources: np.ndarray) -> np.ndarray:
    """Return safe per-dimension standard deviations ``[D]`` for matching."""
    flat = np.asarray(sources, dtype=np.float64).reshape(-1, sources.shape[-1])  # [C,P,D] -> [C*P,D]
    scale = np.std(flat, axis=0)  # shape: [D]
    return np.maximum(scale, 1.0e-12)


def best_permutation(reference: np.ndarray, candidate: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Return the exact minimum-cost one-to-one candidate order ``[P]``.

    The cost is squared distance after dividing every coordinate by ``scale``.
    A small bit-mask dynamic program is exact for the intended small ``P``.

    # O(P*2^P) exact matching; use scipy's assignment solver if P > 16.
    """
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("reference and candidate must share shape [P,D]")
    particle_count = reference.shape[0]
    if particle_count > 16:
        raise ValueError("exact dependency-free matching is limited to P <= 16")

    difference = (reference[:, None, :] - candidate[None, :, :]) / scale[None, None, :]  # [P,1,D] - [1,P,D]
    cost = np.mean(difference * difference, axis=2)  # shape: [P,P]
    state_count = 1 << particle_count
    best = np.full(state_count, np.inf, dtype=np.float64)  # one value per used-candidate mask
    previous = np.full(state_count, -1, dtype=np.int64)
    chosen = np.full(state_count, -1, dtype=np.int64)
    best[0] = 0.0

    for mask in range(state_count):
        row = mask.bit_count()
        if row >= particle_count or not np.isfinite(best[mask]):
            continue
        for column in range(particle_count):
            bit = 1 << column
            if mask & bit:
                continue
            next_mask = mask | bit
            score = best[mask] + cost[row, column]
            if score < best[next_mask]:
                best[next_mask] = score
                previous[next_mask] = mask
                chosen[next_mask] = column

    permutation = np.empty(particle_count, dtype=np.int64)  # reference row p -> candidate row permutation[p]
    mask = state_count - 1
    for row in range(particle_count - 1, -1, -1):
        permutation[row] = chosen[mask]
        mask = previous[mask]
    return permutation


def align_source(reference: np.ndarray, candidate: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Align unordered ``candidate [P,D]`` to ``reference [P,D]``."""
    permutation = best_permutation(reference, candidate, scale)  # shape: [P]
    aligned = candidate[permutation]  # advanced indexing [P,D]
    standardized = (aligned - reference) / scale[None, :]  # [P,D] / [1,D]
    distance = float(np.sqrt(np.mean(standardized * standardized)))
    return aligned, permutation, distance


def source_distance(first: np.ndarray, second: np.ndarray, scale: np.ndarray) -> float:
    """Return permutation-invariant standardized RMSE between two ``[P,D]`` sources."""
    return align_source(first, second, scale)[2]


def select_source_neighbors(
    sources: np.ndarray, target_id: int, parent_count: int, scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select and align the K closest sources to one target at frame zero.

    Args:
        sources: Complete initial sources shaped ``[C+1,P,D]``.
        target_id: Source row used as the known evaluation target.
        parent_count: Number ``K`` of neighboring complete sources.
        scale: Coordinate standard deviations shaped ``[D]``.

    Returns:
        Global parent IDs ``[K]``, saved particle permutations ``[K,P]``,
        aligned initial parents ``[K,P,D]``, and their distances ``[K]``.
    """
    sources = np.asarray(sources, dtype=np.float64)
    if sources.ndim != 3 or not 0 <= target_id < sources.shape[0]:
        raise ValueError("sources must have shape [C+1,P,D] and target_id must be valid")
    if not 2 <= parent_count < sources.shape[0]:
        raise ValueError("parent_count must leave the target out of the neighbor set")

    target = sources[target_id]  # shape: [P,D]
    candidate_ids = np.delete(np.arange(sources.shape[0], dtype=np.int64), target_id)  # shape: [C]
    distance = np.asarray(
        [source_distance(target, sources[source_id], scale) for source_id in candidate_ids]
    )  # one complete-source distance per non-target source, shape: [C]
    order = np.argsort(distance)[:parent_count]  # indices of K nearest candidates, shape: [K]
    parent_ids = candidate_ids[order]  # map candidate rows back to global source IDs, shape: [K]

    particle_count, dimension = target.shape
    permutations = np.empty((parent_count, particle_count), dtype=np.int64)  # shape: [K,P]
    matched = np.empty((parent_count, particle_count, dimension), dtype=np.float64)  # shape: [K,P,D]
    for parent, source_id in enumerate(parent_ids):
        aligned, permutation, _ = align_source(target, sources[source_id], scale)
        matched[parent] = aligned  # target row p corresponds to aligned parent row p, shape: [P,D]
        permutations[parent] = permutation  # persistent parent-particle identity, shape: [P]
    return parent_ids, permutations, matched, distance[order]


def project_simplex(values: np.ndarray) -> np.ndarray:
    """Project one vector onto nonnegative values that sum to one."""
    values = np.asarray(values, dtype=np.float64)
    sorted_values = np.sort(values)[::-1]  # shape: [K]
    cumulative = np.cumsum(sorted_values) - 1.0  # shape: [K]
    index = np.arange(1, values.size + 1, dtype=np.float64)  # shape: [K]
    valid = sorted_values - cumulative / index > 0.0  # shape: [K]
    rho = int(np.flatnonzero(valid)[-1])
    threshold = cumulative[rho] / float(rho + 1)
    projected = np.maximum(values - threshold, 0.0)  # shape: [K]
    return projected / np.sum(projected)


def fit_simplex_weights(
    parent_values: np.ndarray,
    target_value: np.ndarray,
    max_iterations: int = 1000,
    tolerance: float = 1.0e-10,
) -> tuple[np.ndarray, float, int]:
    """Fit one nonnegative source-weight vector whose entries sum to one.

    ``parent_values`` has shape ``[K,F]`` and ``target_value`` has ``[F]``.
    Projected gradient descent minimizes

        || target - sum_k lambda_k parent_k ||^2

    on the probability simplex.
    """
    parent_values = np.asarray(parent_values, dtype=np.float64)
    target_value = np.asarray(target_value, dtype=np.float64).reshape(-1)
    if parent_values.ndim != 2 or parent_values.shape[1] != target_value.size:
        raise ValueError("parent_values [K,F] and target_value [F] are incompatible")
    if max_iterations < 1 or tolerance <= 0.0:
        raise ValueError("max_iterations and tolerance must be positive")

    design = parent_values.T  # [K,F] -> [F,K]
    largest_singular = float(np.linalg.norm(design, ord=2))
    step = 1.0 / max(largest_singular * largest_singular, 1.0e-12)
    weights = np.full(parent_values.shape[0], 1.0 / parent_values.shape[0], dtype=np.float64)  # [K]

    used_iterations = max_iterations
    for iteration in range(1, max_iterations + 1):
        residual = design @ weights - target_value  # [F,K] @ [K] - [F] -> [F]
        gradient = design.T @ residual  # [K,F] @ [F] -> [K]
        updated = project_simplex(weights - step * gradient)  # shape: [K]
        if np.linalg.norm(updated - weights) <= tolerance:
            weights = updated
            used_iterations = iteration
            break
        weights = updated

    final_residual = design @ weights - target_value  # shape: [F]
    rmse = float(np.sqrt(np.mean(final_residual * final_residual)))
    return weights, rmse, used_iterations


def fit_shared_source_lambda(
    target: np.ndarray,
    matched_parents: np.ndarray,
    scale: np.ndarray,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit one lambda jointly over every particle and coordinate.

    Args:
        target: Known terminal target shaped ``[P,D]``.
        matched_parents: The saved K parents shaped ``[K,P,D]``.
        scale: Terminal coordinate scales shaped ``[D]``.
        max_iterations: Maximum projected-gradient updates.

    Returns:
        Shared weights ``[K]``, their terminal candidate ``[P,D]``, and
        standardized joint fit RMSE. No per-particle lambda is fitted.
    """
    target = np.asarray(target, dtype=np.float64)
    matched_parents = np.asarray(matched_parents, dtype=np.float64)
    if matched_parents.ndim != 3 or target.shape != matched_parents.shape[1:]:
        raise ValueError("target [P,D] and matched_parents [K,P,D] are incompatible")

    parent_count, particle_count, dimension = matched_parents.shape
    # Scientific equation:
    #
    #   lambda* = argmin_{lambda >= 0, sum(lambda)=1}
    #             (1/(P*D)) sum_(p,d)
    #             [T_(p,d) - sum_k lambda_k S_(k,p,d)]^2.
    #
    # Flattening P and D makes one optimizer see the complete source. It does
    # not create separate weights for individual particles.
    normalized_parents = matched_parents / scale[None, None, :]  # [K,P,D] / [1,1,D]
    normalized_target = target / scale[None, :]  # [P,D] / [1,D]
    weights, error, _ = fit_simplex_weights(
        normalized_parents.reshape(parent_count, particle_count * dimension),  # [K,P,D] -> [K,P*D]
        normalized_target.reshape(particle_count * dimension),  # [P,D] -> [P*D]
        max_iterations=max_iterations,
    )
    candidate = np.einsum("k,kpd->pd", weights, matched_parents)  # [K] with [K,P,D], sum K -> [P,D]
    return weights, candidate, error


def fit_per_particle_lambdas(
    target: np.ndarray,
    matched_parents: np.ndarray,
    scale: np.ndarray,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Fit one simplex vector per corresponding particle.

    Args:
        target: Known source shaped ``[P,D]``.
        matched_parents: The same saved complete parents shaped ``[K,P,D]``.
        scale: Coordinate scales shaped ``[D]``.
        max_iterations: Maximum projected-gradient updates per particle.

    Returns:
        Per-particle weights ``[P,K]``, candidate ``[P,D]``, joint
        standardized RMSE, and individual particle RMSE values ``[P]``.

    This is a capacity control. It keeps parent identities and correspondence
    fixed, but removes H4.5's requirement that all particles share one lambda.
    """
    target = np.asarray(target, dtype=np.float64)
    matched_parents = np.asarray(matched_parents, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if matched_parents.ndim != 3 or target.shape != matched_parents.shape[1:]:
        raise ValueError("target [P,D] and matched_parents [K,P,D] are incompatible")
    if scale.shape != (target.shape[1],):
        raise ValueError("scale must have shape [D]")

    parent_count, particle_count, dimension = matched_parents.shape
    weights = np.empty((particle_count, parent_count), dtype=np.float64)  # shape: [P,K]
    candidate = np.empty((particle_count, dimension), dtype=np.float64)  # shape: [P,D]
    particle_rmse = np.empty(particle_count, dtype=np.float64)  # shape: [P]
    for particle in range(particle_count):
        # Fit each corresponding target particle inside the convex hull of the
        # same K parent particles, after coordinate standardization.
        normalized_parents = matched_parents[:, particle, :] / scale[None, :]  # [K,D] / [1,D]
        normalized_target = target[particle] / scale  # [D] / [D]
        fitted, error, _ = fit_simplex_weights(
            normalized_parents, normalized_target, max_iterations=max_iterations
        )
        weights[particle] = fitted  # shape: [K]
        particle_rmse[particle] = error
        candidate[particle] = np.einsum(
            "k,kd->d", fitted, matched_parents[:, particle, :]
        )  # [K] with [K,D], sum K -> [D]

    # Aggregate every standardized coordinate so the control remains directly
    # comparable with the shared source-level fit RMSE.
    standardized_residual = (candidate - target) / scale[None, :]  # [P,D] / [1,D]
    joint_rmse = float(np.sqrt(np.mean(standardized_residual * standardized_residual)))
    return weights, candidate, joint_rmse, particle_rmse


def build_reconstruction_trials(
    memory_sources: np.ndarray,
    target_sources: np.ndarray,
    parent_count: int,
    scale: np.ndarray,
    lambda_iterations: int,
) -> dict[str, np.ndarray]:
    """Retrieve K unordered neighbors and fit one shared lambda per target.

    Each parent is matched directly to the held-out target, so no global row or
    slot identity is assumed. The matched ``[K,P,D]`` parents are flattened to
    ``[K,P*D]`` and fitted together against the target ``[P*D]`` vector.
    """
    target_count, particle_count, dimension = target_sources.shape
    memory_count = memory_sources.shape[0]
    if not 2 <= parent_count <= memory_count:
        raise ValueError("parent_count must be between two and the memory source count")

    parent_ids = np.empty((target_count, parent_count), dtype=np.int64)  # shape: [R,K]
    parent_permutations = np.empty((target_count, parent_count, particle_count), dtype=np.int64)  # [R,K,P]
    matched_parents = np.empty((target_count, parent_count, particle_count, dimension), dtype=np.float64)  # [R,K,P,D]
    nearest_parent_distance = np.empty(target_count, dtype=np.float64)  # shape: [R]
    shared_lambda = np.empty((target_count, parent_count), dtype=np.float64)  # shape: [R,K]
    lambda_fit_rmse = np.empty(target_count, dtype=np.float64)  # shape: [R]

    for trial, target in enumerate(target_sources):
        # Source proximity is the minimum D*P Euclidean assignment cost, not a
        # fixed row-by-row comparison. Select the four closest complete sets.
        distances = np.asarray([
            source_distance(target, memory_source, scale) for memory_source in memory_sources
        ])  # shape: [C]
        selected = np.argsort(distances)[:parent_count]  # shape: [K]
        parent_ids[trial] = selected
        nearest_parent_distance[trial] = distances[selected[0]]

        for parent, source_id in enumerate(selected):
            aligned, permutation, _ = align_source(target, memory_sources[source_id], scale)
            matched_parents[trial, parent] = aligned  # shape: [P,D]
            parent_permutations[trial, parent] = permutation  # shape: [P]

        weights, _, error = fit_shared_source_lambda(
            target, matched_parents[trial], scale, max_iterations=lambda_iterations
        )
        shared_lambda[trial] = weights
        lambda_fit_rmse[trial] = error

    # Keep a singleton candidate axis M=1 so the existing batched replay code
    # can process every target in one call without a special-case path.
    shared_lambdas = shared_lambda[:, None, :]  # [R,K] -> [R,M=1,K]
    fitted_candidates = np.einsum(
        "rmk,rkpd->rmpd", shared_lambdas, matched_parents
    )  # [R,1,K] with [R,K,P,D], sum K -> [R,1,P,D]
    return {
        "parent_ids": parent_ids,
        "parent_permutations": parent_permutations,
        "matched_parents": matched_parents,
        "aligned_targets": target_sources.copy(),
        "nearest_parent_distance": nearest_parent_distance,
        "shared_lambdas": shared_lambdas,
        "lambda_fit_rmse": lambda_fit_rmse,
        "fitted_candidates": fitted_candidates,
    }


def interpolate_sources(
    sources: np.ndarray,
    parent_ids: np.ndarray,
    parent_permutations: np.ndarray,
    shared_lambdas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply saved parent matching and shared lambdas in one source space."""
    target_count, parent_count, particle_count = parent_permutations.shape
    dimension = sources.shape[2]
    matched = np.empty((target_count, parent_count, particle_count, dimension), dtype=np.float64)  # [R,K,P,D]
    for trial in range(target_count):
        for parent in range(parent_count):
            source = sources[parent_ids[trial, parent]]  # shape: [P,D]
            matched[trial, parent] = source[parent_permutations[trial, parent]]  # shape: [P,D]
    candidates = np.einsum(
        "rmk,rkpd->rmpd", shared_lambdas, matched
    )  # [R,1,K] with [R,K,P,D], sum K -> [R,1,P,D]
    return matched, candidates


def interpolate_source_methods(
    sources: np.ndarray,
    parent_ids: np.ndarray,
    parent_permutations: np.ndarray,
    particle_lambdas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply shared or per-particle lambdas to saved complete parents.

    ``particle_lambdas`` has shape ``[R,M,P,K]``. A shared source-level
    lambda is represented by repeating the same K-vector over the P axis.
    """
    target_count, parent_count, particle_count = parent_permutations.shape
    if particle_lambdas.ndim != 4 or particle_lambdas.shape[:3] != (
        target_count,
        particle_lambdas.shape[1],
        particle_count,
    ):
        raise ValueError("particle_lambdas must have shape [R,M,P,K]")
    if particle_lambdas.shape[3] != parent_count:
        raise ValueError("particle_lambdas K axis must match parent_ids")

    dimension = sources.shape[2]
    matched = np.empty(
        (target_count, parent_count, particle_count, dimension), dtype=np.float64
    )  # shape: [R,K,P,D]
    for trial in range(target_count):
        for parent in range(parent_count):
            source = sources[parent_ids[trial, parent]]  # shape: [P,D]
            matched[trial, parent] = source[parent_permutations[trial, parent]]  # shape: [P,D]

    # Each output particle p combines only the corresponding p rows from the
    # saved complete parents. Shared methods merely repeat weights over P.
    candidates = np.einsum(
        "rmpk,rkpd->rmpd", particle_lambdas, matched
    )  # [R,M,P,K] with [R,K,P,D], sum K -> [R,M,P,D]
    return matched, candidates


def self_check() -> dict[str, float]:
    """Check generation, exact matching, and nonnegative simplex fitting."""
    sources = generate_chaotic_sources(6, 4, 3, 1.0, np.random.default_rng(11))
    assert sources.shape == (6, 4, 3)
    assert np.all(np.isfinite(sources))
    points = generate_uniform_sources(5, 1, 2, 1.0, np.random.default_rng(12))
    assert points.shape == (5, 1, 2)
    assert np.all(np.abs(points) <= 1.0)

    reference = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    candidate = reference[[2, 0, 1]]  # same unordered set
    aligned, permutation, distance = align_source(reference, candidate, np.ones(2))
    assert np.allclose(aligned, reference)
    assert np.array_equal(permutation, np.asarray((1, 2, 0)))
    assert distance < 1.0e-15

    parents = np.eye(3, dtype=np.float64)  # shape: [K=3,F=3]
    expected = np.asarray((0.2, 0.3, 0.5))
    fitted, error, _ = fit_simplex_weights(parents, expected)
    weight_error = float(np.max(np.abs(fitted - expected)))
    assert weight_error < 1.0e-8, weight_error
    assert error < 1.0e-8, error
    assert np.all(fitted >= 0.0) and np.isclose(np.sum(fitted), 1.0)

    simple_sources = np.asarray(
        [[[-2.0, 0.0]], [[-1.0, 0.0]], [[0.0, 0.0]], [[1.0, 0.0]], [[2.0, 0.0]]]
    )  # shape: [C+1=5,P=1,D=2]
    neighbor_ids, saved_permutations, _, _ = select_source_neighbors(
        simple_sources, target_id=2, parent_count=2, scale=np.ones(2)
    )
    assert 2 not in neighbor_ids and set(neighbor_ids.tolist()) == {1, 3}
    matched = simple_sources[neighbor_ids][:, saved_permutations[0]]  # shape: [K=2,P=1,D=2]
    shared, reconstructed, joint_error = fit_shared_source_lambda(
        simple_sources[2], matched, np.ones(2), max_iterations=1000
    )
    assert np.allclose(shared, (0.5, 0.5), atol=1.0e-8)
    assert np.allclose(reconstructed, simple_sources[2], atol=1.0e-8)
    assert joint_error < 1.0e-8
    per_particle, per_candidate, per_error, _ = fit_per_particle_lambdas(
        simple_sources[2], matched, np.ones(2), max_iterations=1000
    )
    p1_difference = float(np.max(np.abs(per_particle[0] - shared)))
    assert p1_difference < 1.0e-10, p1_difference
    assert np.allclose(per_candidate, reconstructed, atol=1.0e-10)
    assert per_error < 1.0e-8
    return {
        "matching_rmse": distance,
        "lambda_max_abs_error": weight_error,
        "p1_shared_control_max_abs_error": p1_difference,
    }


if __name__ == "__main__":
    print(self_check())

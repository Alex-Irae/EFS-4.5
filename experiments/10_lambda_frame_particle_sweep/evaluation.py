"""Numerical gates and held-out complete-source reconstruction metrics."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from data import METHOD_NAMES, align_source, source_distance
from efs import mean_pairwise_distance


METRIC_FIELDS = (
    "target",
    "target_source_id",
    "method",
    "lambda_fit_frame",
    "parent_source_ids",
    "lambda",
    "lambda_fit_rmse",
    "initial_candidate_rmse",
    "reference_terminal_rmse",
    "cross_frame_rmse",
    "generation_terminal_rmse",
    "direct_relative_distance",
    "generated_relative_distance",
    "direct_point_distance",
    "generated_point_distance",
    "direct_set_rmse",
    "generated_set_rmse",
    "efs_direct_ratio",
    "nearest_parent_relative_distance",
    "nearest_parent_point_distance",
    "nearest_parent_rmse",
    "efs_parent_ratio",
    "direct_aligned_rmse",
    "generated_aligned_rmse",
    "direct_relation_rmse",
    "generated_relation_rmse",
    "terminal_gap",
    "max_replay_residual",
    "generated_nearest_memory_rmse",
    "memory_distance_ratio",
)


def radial_statistics(particles: np.ndarray) -> dict[str, float]:
    """Describe radial agreement of one centered plane ``[N,D]`` with a D-ball."""
    particles = np.asarray(particles, dtype=np.float64)
    centered = particles - np.mean(particles, axis=0, keepdims=True)  # [N,D] - [1,D]
    radius = np.linalg.norm(centered, axis=1)  # shape: [N]
    dimension = particles.shape[1]
    mean_square = float(np.mean(radius * radius))
    fitted_radius = math.sqrt((dimension + 2.0) * mean_square / dimension)
    normalized = radius / max(fitted_radius, 1.0e-15)  # shape: [N]

    sorted_radius = np.sort(radius)  # shape: [N]
    empirical_cdf = np.arange(1, radius.size + 1, dtype=np.float64) / radius.size  # [N]
    reference_cdf = np.clip((sorted_radius / max(fitted_radius, 1.0e-15)) ** dimension, 0.0, 1.0)  # [N]
    cdf_gap = float(np.max(np.abs(empirical_cdf - reference_cdf)))

    uniform_coordinate = normalized**dimension  # uniform [0,1] for an ideal fitted D-ball
    zone_index = np.floor(10.0 * uniform_coordinate).astype(np.int64)  # shape: [N]
    zone_fraction = np.asarray([np.mean(zone_index == zone) for zone in range(10)])  # shape: [10]
    quantile = np.quantile(normalized, (0.05, 0.25, 0.50, 0.75, 0.95))  # shape: [5]
    mean_radius = float(np.mean(radius))
    return {
        "rms_radius": math.sqrt(mean_square),
        "mean_radius": mean_radius,
        "radius_cv": float(np.std(radius) / max(mean_radius, 1.0e-15)),
        "expected_radius_cv": 1.0 / math.sqrt(dimension * (dimension + 2.0)),
        "fitted_ball_radius": fitted_radius,
        "radial_cdf_gap": cdf_gap,
        "radius_q05": float(quantile[0]),
        "radius_q25": float(quantile[1]),
        "radius_q50": float(quantile[2]),
        "radius_q75": float(quantile[3]),
        "radius_q95": float(quantile[4]),
        "radial_overflow": float(np.mean(uniform_coordinate >= 1.0)),
        **{f"radial_zone_{zone}": float(value) for zone, value in enumerate(zone_fraction)},
    }


def angular_statistics(particles: np.ndarray) -> dict[str, float]:
    """Describe direction balance and dimensional support of one plane ``[N,D]``."""
    particles = np.asarray(particles, dtype=np.float64)
    centered = particles - np.mean(particles, axis=0, keepdims=True)  # shape: [N,D]
    radius = np.linalg.norm(centered, axis=1)  # shape: [N]
    valid = radius > 1.0e-15  # shape: [N]
    if np.count_nonzero(valid) < particles.shape[1] + 1:
        return {"angular_resultant": math.inf, "angular_moment_error": math.inf, "covariance_ratio": 0.0}

    unit = centered[valid] / radius[valid, None]  # [V,D] / [V,1] -> [V,D]
    resultant = float(np.linalg.norm(np.mean(unit, axis=0)))
    second_moment = unit.T @ unit / unit.shape[0]  # [D,V] @ [V,D] -> [D,D]
    expected_moment = np.eye(particles.shape[1]) / particles.shape[1]  # shape: [D,D]

    covariance = centered.T @ centered / centered.shape[0]  # [D,N] @ [N,D] -> [D,D]
    eigenvalues = np.linalg.eigvalsh(covariance)  # shape: [D]
    largest = float(np.max(eigenvalues))
    return {
        "angular_resultant": resultant,
        "angular_moment_error": float(np.linalg.norm(second_moment - expected_moment)),
        "covariance_ratio": float(max(float(np.min(eigenvalues)), 0.0) / max(largest, 1.0e-15)),
    }


def plane_statistics(particles: np.ndarray) -> dict[str, float | bool]:
    """Return finite, scale, radial, and angular values for one plane ``[N,D]``."""
    particles = np.asarray(particles, dtype=np.float64)
    finite = bool(np.all(np.isfinite(particles)))
    if not finite:
        return {"finite": False}
    values: dict[str, float | bool] = {"finite": True, "mean_pair_distance": mean_pairwise_distance(particles)}
    values.update(radial_statistics(particles))
    values.update(angular_statistics(particles))
    return values


def smoke_gate(
    initial_plane: np.ndarray, terminal_plane: np.ndarray, forward_diagnostics: dict[str, np.ndarray | bool | str]
) -> tuple[bool, list[str], dict[str, dict[str, float | bool]]]:
    """Apply only finite-value, scale, and dimensional-collapse hard gates."""
    initial = plane_statistics(initial_plane)
    terminal = plane_statistics(terminal_plane)
    reasons: list[str] = []

    if not bool(forward_diagnostics.get("finite", True)):
        reasons.append(str(forward_diagnostics.get("failure_reason", "forward evolution failed")))
    if not bool(initial.get("finite", False)) or not bool(terminal.get("finite", False)):
        reasons.append("initial or terminal plane contains non-finite values")
    else:
        radius_ratio = float(terminal["rms_radius"]) / max(float(initial["rms_radius"]), 1.0e-15)
        initial["radius_ratio"] = 1.0
        terminal["radius_ratio"] = radius_ratio
        if not 0.05 <= radius_ratio <= 5.0:
            reasons.append(f"terminal RMS-radius ratio {radius_ratio:.6f} is outside [0.05, 5.0]")
        if float(terminal["covariance_ratio"]) < 1.0e-3:
            reasons.append(f"terminal covariance ratio {float(terminal['covariance_ratio']):.6e} is below 1e-3")
    return not reasons, reasons, {"initial": initial, "terminal": terminal}


def marginal_histogram_kl(
    reference: np.ndarray, candidate: np.ndarray, bins: int = 48, pseudocount: float = 0.5
) -> dict[str, np.ndarray | float]:
    """Describe pass-1/pass-2 endpoint shift with fixed marginal histograms.

    Args:
        reference: Retained pass-1 particles shaped ``[N,D]``.
        candidate: Corresponding pass-2 particles shaped ``[N,D]``.
        bins: Shared histogram bins per coordinate.
        pseudocount: Positive count added to avoid undefined log ratios.

    Returns:
        Per-coordinate directional KL values ``[D]``, their means, shared bin
        edges ``[D,bins+1]``, and normalized histograms ``[D,bins]``.

    This is descriptive marginal KL, not a multivariate density estimate or a
    formal statistical test. Zero means identical coordinate histograms.
    """
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.ndim != 2 or candidate.ndim != 2 or reference.shape[1] != candidate.shape[1]:
        raise ValueError("reference and candidate must have shapes [N1,D] and [N2,D]")
    if bins < 2 or pseudocount <= 0.0:
        raise ValueError("bins must be at least two and pseudocount must be positive")

    dimension = reference.shape[1]
    edges = np.empty((dimension, bins + 1), dtype=np.float64)  # shape: [D,B+1]
    reference_probability = np.empty((dimension, bins), dtype=np.float64)  # shape: [D,B]
    candidate_probability = np.empty_like(reference_probability)
    reference_to_candidate = np.empty(dimension, dtype=np.float64)  # shape: [D]
    candidate_to_reference = np.empty(dimension, dtype=np.float64)  # shape: [D]

    for coordinate in range(dimension):
        combined = np.concatenate((reference[:, coordinate], candidate[:, coordinate]))  # shape: [N1+N2]
        lower = float(np.min(combined))
        upper = float(np.max(combined))
        if upper <= lower:
            upper = lower + 1.0
        padding = 1.0e-12 * max(1.0, abs(lower), abs(upper))
        coordinate_edges = np.linspace(lower - padding, upper + padding, bins + 1)  # shape: [B+1]
        reference_count, _ = np.histogram(reference[:, coordinate], bins=coordinate_edges)  # shape: [B]
        candidate_count, _ = np.histogram(candidate[:, coordinate], bins=coordinate_edges)  # shape: [B]

        # For histogram probabilities p and q, directional KL is
        #
        #   KL(p || q) = sum_b p_b log(p_b / q_b).
        #
        # The same small count is added to every bin in both passes so empty
        # bins remain finite without introducing a new dependency.
        p = (reference_count + pseudocount) / (reference_count.sum() + bins * pseudocount)  # [B]
        q = (candidate_count + pseudocount) / (candidate_count.sum() + bins * pseudocount)  # [B]
        edges[coordinate] = coordinate_edges
        reference_probability[coordinate] = p
        candidate_probability[coordinate] = q
        reference_to_candidate[coordinate] = np.sum(p * np.log(p / q))
        candidate_to_reference[coordinate] = np.sum(q * np.log(q / p))

    return {
        "reference_to_candidate": reference_to_candidate,
        "candidate_to_reference": candidate_to_reference,
        "mean_reference_to_candidate": float(np.mean(reference_to_candidate)),
        "mean_candidate_to_reference": float(np.mean(candidate_to_reference)),
        "bin_edges": edges,
        "reference_probability": reference_probability,
        "candidate_probability": candidate_probability,
    }


def vertex_replay_metrics(
    replayed: np.ndarray, expected: np.ndarray, memory_std: float, replay_residual: np.ndarray
) -> dict[str, np.ndarray | float | bool]:
    """Measure exact-memory inverse accuracy for ``[V,P,D]`` sources.

    Vertex replay checks only whether exact terminal memory particles return to
    their saved initial values. It does not test interpolation or source validity.
    """
    error = replayed - expected  # shape: [V,P,D]
    particle_rmse = np.sqrt(np.mean(error * error, axis=2))  # [V,P,D] -> [V,P]
    source_rmse = np.sqrt(np.mean(error * error, axis=(1, 2)))  # [V,P,D] -> [V]
    scale = max(memory_std, 1.0e-15)
    particle_relative = particle_rmse / scale  # shape: [V,P]
    source_relative = source_rmse / scale  # shape: [V]
    particle_residual = np.max(replay_residual, axis=0)  # [J,V,P] -> [V,P]
    return {
        "particle_rmse": particle_rmse,
        "source_rmse": source_rmse,
        "particle_relative_rmse": particle_relative,
        "source_relative_rmse": source_relative,
        "particle_max_residual": particle_residual,
        "median": float(np.median(particle_relative)),
        "p90": float(np.quantile(particle_relative, 0.90)),
        "p95": float(np.quantile(particle_relative, 0.95)),
        "maximum": float(np.max(particle_relative)),
        "passed": bool(np.max(particle_relative) <= 0.10),
    }


def relation_signature(source: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Return sorted standardized within-source particle distances ``[P(P-1)/2]``."""
    source = np.asarray(source, dtype=np.float64) / scale[None, :]  # [P,D] / [1,D]
    values: list[float] = []
    for first in range(source.shape[0]):
        for second in range(first + 1, source.shape[0]):
            values.append(float(np.linalg.norm(source[first] - source[second])))
    return np.sort(np.asarray(values, dtype=np.float64))


def relation_rmse(first: np.ndarray, second: np.ndarray, scale: np.ndarray) -> float:
    """Return RMSE between two permutation-invariant internal-distance signatures."""
    difference = relation_signature(first, scale) - relation_signature(second, scale)  # shape: [P(P-1)/2]
    if difference.size == 0:
        return 0.0  # P=1 has no internal geometry to compare
    return float(np.sqrt(np.mean(difference * difference)))


def nearest_memory_distance(source: np.ndarray, memory_sources: np.ndarray, scale: np.ndarray) -> float:
    """Return minimum permutation-invariant distance from one source to memory."""
    return float(min(source_distance(source, memory, scale) for memory in memory_sources))


def relative_source_distance(
    target: np.ndarray, candidate: np.ndarray, point_scale: np.ndarray | None = None
) -> float:
    """Return unordered Euclidean error divided by target spatial extent.

    For target ``T [P,D]`` and candidate ``G [P,D]`` this implements

        d_rel(G,T) = min_pi ||G_pi - T||_F / ||T - mean(T)||_F.

    The assignment makes particle order irrelevant. A value of one means the
    total reconstruction displacement equals the target source's own extent.
    For ``P=1`` that extent is zero, so ``point_scale [D]`` supplies the
    memory-plane RMS coordinate radius instead.
    """
    unit_scale = np.ones(target.shape[1], dtype=np.float64)  # shape: [D]
    aligned, _, _ = align_source(target, candidate, unit_scale)  # shape: [P,D]
    displacement = np.linalg.norm(aligned - target)  # Frobenius norm over [P,D]
    centered_target = target - np.mean(target, axis=0, keepdims=True)  # [P,D] - [1,D]
    target_extent = np.linalg.norm(centered_target)
    if target_extent <= 1.0e-15:
        if point_scale is None:
            raise ValueError("point_scale is required when a source contains one particle")
        target_extent = np.linalg.norm(np.asarray(point_scale, dtype=np.float64))
    return float(displacement / max(float(target_extent), 1.0e-15))


def point_distance(target: np.ndarray, candidate: np.ndarray) -> float:
    """Return raw Euclidean target error for two single-particle sources."""
    if target.shape[0] != 1 or candidate.shape != target.shape:
        raise ValueError("point_distance expects target and candidate shaped [1,D]")
    return float(np.linalg.norm(candidate[0] - target[0]))


def reconstruction_metrics(
    generated: np.ndarray,
    direct: np.ndarray,
    targets: np.ndarray,
    matched_parents: np.ndarray,
    terminal_candidates: np.ndarray,
    oracle_terminal: np.ndarray,
    replay_residual: np.ndarray,
    memory_sources: np.ndarray,
    initial_scale: np.ndarray,
    terminal_scale: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate held-out reconstruction arrays.

    Shapes:
        generated/direct/terminal_candidates: ``[R,M,P,D]``;
        targets/oracle_terminal: ``[R,P,D]``;
        matched_parents: ``[R,K,P,D]``;
        replay_residual: ``[J,R,M,P]``.
    """
    target_count, method_count = generated.shape[:2]
    generated_set = np.empty((target_count, method_count), dtype=np.float64)  # [R,M]
    direct_set = np.empty_like(generated_set)
    generated_relative = np.empty_like(generated_set)
    direct_relative = np.empty_like(generated_set)
    generated_point = np.full_like(generated_set, np.nan)
    direct_point = np.full_like(generated_set, np.nan)
    generated_relation = np.empty_like(generated_set)
    direct_relation = np.empty_like(generated_set)
    nearest_memory = np.empty_like(generated_set)
    nearest_parent = np.empty(target_count, dtype=np.float64)  # [R]
    nearest_parent_relative = np.empty(target_count, dtype=np.float64)  # [R]
    nearest_parent_point = np.full(target_count, np.nan, dtype=np.float64)  # [R]

    for trial in range(target_count):
        target = targets[trial]  # shape: [P,D]
        nearest_parent[trial] = min(source_distance(target, parent, initial_scale) for parent in matched_parents[trial])
        nearest_parent_relative[trial] = min(
            relative_source_distance(target, parent, initial_scale) for parent in matched_parents[trial]
        )
        if target.shape[0] == 1:
            nearest_parent_point[trial] = min(point_distance(target, parent) for parent in matched_parents[trial])
        for method in range(method_count):
            generated_set[trial, method] = source_distance(target, generated[trial, method], initial_scale)
            direct_set[trial, method] = source_distance(target, direct[trial, method], initial_scale)
            generated_relative[trial, method] = relative_source_distance(target, generated[trial, method], initial_scale)
            direct_relative[trial, method] = relative_source_distance(target, direct[trial, method], initial_scale)
            if target.shape[0] == 1:
                generated_point[trial, method] = point_distance(target, generated[trial, method])
                direct_point[trial, method] = point_distance(target, direct[trial, method])
            generated_relation[trial, method] = relation_rmse(target, generated[trial, method], initial_scale)
            direct_relation[trial, method] = relation_rmse(target, direct[trial, method], initial_scale)
            nearest_memory[trial, method] = nearest_memory_distance(generated[trial, method], memory_sources, initial_scale)

    target_expanded = targets[:, None, :, :]  # [R,P,D] -> [R,1,P,D]
    generated_scaled_error = (generated - target_expanded) / initial_scale[None, None, None, :]  # [R,M,P,D]
    direct_scaled_error = (direct - target_expanded) / initial_scale[None, None, None, :]  # [R,M,P,D]
    terminal_scaled_error = (terminal_candidates - oracle_terminal[:, None, :, :]) / terminal_scale[
        None, None, None, :
    ]  # [R,M,P,D]
    generated_aligned = np.sqrt(np.mean(generated_scaled_error**2, axis=(2, 3)))  # [R,M]
    direct_aligned = np.sqrt(np.mean(direct_scaled_error**2, axis=(2, 3)))  # [R,M]
    terminal_gap = np.sqrt(np.mean(terminal_scaled_error**2, axis=(2, 3)))  # [R,M]
    max_residual = np.max(replay_residual, axis=(0, 3))  # [J,R,M,P] -> [R,M]

    return {
        "generated_relative_distance": generated_relative,
        "direct_relative_distance": direct_relative,
        "generated_point_distance": generated_point,
        "direct_point_distance": direct_point,
        "generated_set_rmse": generated_set,
        "direct_set_rmse": direct_set,
        "efs_direct_ratio": generated_relative / np.maximum(direct_relative, 1.0e-15),
        "nearest_parent_relative_distance": nearest_parent_relative,
        "nearest_parent_point_distance": nearest_parent_point,
        "nearest_parent_rmse": nearest_parent,
        "efs_parent_ratio": generated_relative / np.maximum(nearest_parent_relative[:, None], 1.0e-15),
        "generated_aligned_rmse": generated_aligned,
        "direct_aligned_rmse": direct_aligned,
        "generated_relation_rmse": generated_relation,
        "direct_relation_rmse": direct_relation,
        "terminal_gap": terminal_gap,
        "max_replay_residual": max_residual,
        "generated_nearest_memory_rmse": nearest_memory,
        "memory_distance_ratio": nearest_memory / np.maximum(nearest_parent[:, None], 1.0e-15),
    }


def result_summary(metrics: dict[str, np.ndarray], method_names: np.ndarray) -> dict[str, dict[str, float | int | str]]:
    """Return simple per-method counts and medians without formal claims."""
    values: dict[str, dict[str, float | int | str]] = {}
    trial_count = metrics["generated_set_rmse"].shape[0]
    for method, name in enumerate(method_names):
        ratio = metrics["efs_direct_ratio"][:, method]  # shape: [R]
        parent_ratio = metrics["efs_parent_ratio"][:, method]  # shape: [R]
        direct_wins = int(np.count_nonzero(ratio < 1.0))
        parent_wins = int(np.count_nonzero(parent_ratio < 1.0))
        if direct_wins >= math.ceil(0.75 * trial_count) and float(np.max(ratio)) < 2.0:
            status = "strong target improvement"
        elif direct_wins > trial_count / 2.0:
            status = "mixed target improvement"
        else:
            status = "no consistent target improvement"
        values[str(name)] = {
            "status": status,
            "targets": trial_count,
            "beats_direct_count": direct_wins,
            "beats_parent_count": parent_wins,
            "median_generated_relative_distance": float(np.median(metrics["generated_relative_distance"][:, method])),
            "median_direct_relative_distance": float(np.median(metrics["direct_relative_distance"][:, method])),
            "median_generated_set_rmse": float(np.median(metrics["generated_set_rmse"][:, method])),
            "median_direct_set_rmse": float(np.median(metrics["direct_set_rmse"][:, method])),
            "median_efs_direct_ratio": float(np.median(ratio)),
            "p90_efs_direct_ratio": float(np.quantile(ratio, 0.90)),
            "maximum_efs_direct_ratio": float(np.max(ratio)),
            "median_relation_rmse": float(np.median(metrics["generated_relation_rmse"][:, method])),
        }
        if np.all(np.isfinite(metrics["generated_point_distance"][:, method])):
            values[str(name)]["median_generated_point_distance"] = float(
                np.median(metrics["generated_point_distance"][:, method])
            )
            values[str(name)]["median_direct_point_distance"] = float(
                np.median(metrics["direct_point_distance"][:, method])
            )
    return values


def write_smoke_csv(
    path: Path,
    smoke: dict[str, dict[str, float | bool]],
    smoke_passed: bool,
    vertex_source_ids: np.ndarray | None = None,
    vertex: dict[str, np.ndarray | float | bool] | None = None,
) -> None:
    """Write initial/terminal plane values and optional particle vertex rows."""
    fields = (
        "stage",
        "source_id",
        "particle",
        "finite",
        "mean_pair_distance",
        "rms_radius",
        "radius_ratio",
        "radius_cv",
        "expected_radius_cv",
        "radial_cdf_gap",
        "radial_zone_0",
        "radial_zone_1",
        "radial_zone_2",
        "radial_zone_3",
        "radial_zone_4",
        "radial_zone_5",
        "radial_zone_6",
        "radial_zone_7",
        "radial_zone_8",
        "radial_zone_9",
        "radial_overflow",
        "angular_resultant",
        "angular_moment_error",
        "covariance_ratio",
        "vertex_particle_rmse",
        "vertex_source_rmse",
        "vertex_particle_relative_rmse",
        "vertex_source_relative_rmse",
        "vertex_replay_residual",
        "gate_pass",
    )
    rows: list[dict[str, object]] = []
    for stage in ("initial", "terminal"):
        row = {field: "" for field in fields}
        row.update(smoke[stage])
        stage_passed = bool(smoke[stage].get("finite", False))
        if stage == "terminal":
            stage_passed = stage_passed and smoke_passed
        row.update({"stage": stage, "gate_pass": int(stage_passed)})
        rows.append(row)
    if vertex_source_ids is not None and vertex is not None:
        particle_error = np.asarray(vertex["particle_relative_rmse"])  # shape: [V,P]
        source_error = np.asarray(vertex["source_relative_rmse"])  # shape: [V]
        particle_absolute = np.asarray(vertex["particle_rmse"])  # shape: [V,P]
        source_absolute = np.asarray(vertex["source_rmse"])  # shape: [V]
        residual = np.asarray(vertex["particle_max_residual"])  # shape: [V,P]
        for source in range(particle_error.shape[0]):
            for particle in range(particle_error.shape[1]):
                rows.append({
                    "stage": "vertex",
                    "source_id": int(vertex_source_ids[source]),
                    "particle": particle,
                    "vertex_particle_rmse": f"{particle_absolute[source, particle]:.9e}",
                    "vertex_source_rmse": f"{source_absolute[source]:.9e}",
                    "vertex_particle_relative_rmse": f"{particle_error[source, particle]:.9e}",
                    "vertex_source_relative_rmse": f"{source_error[source]:.9e}",
                    "vertex_replay_residual": f"{residual[source, particle]:.9e}",
                    "gate_pass": int(bool(vertex["passed"])),
                })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_csv(
    path: Path,
    target_source_ids: np.ndarray,
    parent_source_ids: np.ndarray,
    particle_lambdas: np.ndarray,
    lambda_fit_rmse: np.ndarray,
    metrics: dict[str, np.ndarray],
) -> None:
    """Write one reconstruction row per target and lambda method."""
    particle_lambdas = np.asarray(particle_lambdas, dtype=np.float64)
    lambda_fit_rmse = np.asarray(lambda_fit_rmse, dtype=np.float64)
    if particle_lambdas.ndim != 4:
        raise ValueError("particle_lambdas must have shape [R,M,P,K]")
    if lambda_fit_rmse.shape != particle_lambdas.shape[:2]:
        raise ValueError("lambda_fit_rmse must have shape [R,M]")

    rows: list[dict[str, object]] = []
    for target in range(target_source_ids.size):
        for method, name in enumerate(METHOD_NAMES):
            weights = particle_lambdas[target, method]  # shape: [P,K]
            is_shared = bool(np.allclose(weights, weights[:1], atol=1.0e-12, rtol=0.0))
            if is_shared:
                serialized_lambda = ";".join(f"{value:.9f}" for value in weights[0])
            else:
                serialized_lambda = "|".join(
                    ";".join(f"{value:.9f}" for value in particle_weights)
                    for particle_weights in weights
                )
            rows.append({
                "target": target,
                "target_source_id": int(target_source_ids[target]),
                "method": str(name),
                "lambda_fit_frame": "d1" if str(name).endswith("d1") else "du",
                "parent_source_ids": ";".join(str(int(value)) for value in parent_source_ids[target]),
                "lambda": serialized_lambda,
                "lambda_fit_rmse": f"{lambda_fit_rmse[target, method]:.9e}",
                **{
                    field: f"{metrics[field][target, method]:.9e}"
                    for field in (
                        "initial_candidate_rmse",
                        "reference_terminal_rmse",
                        "cross_frame_rmse",
                        "generation_terminal_rmse",
                        "direct_relative_distance",
                        "generated_relative_distance",
                        "direct_point_distance",
                        "generated_point_distance",
                        "direct_set_rmse",
                        "generated_set_rmse",
                        "efs_direct_ratio",
                        "efs_parent_ratio",
                        "direct_aligned_rmse",
                        "generated_aligned_rmse",
                        "direct_relation_rmse",
                        "generated_relation_rmse",
                        "terminal_gap",
                        "max_replay_residual",
                        "generated_nearest_memory_rmse",
                        "memory_distance_ratio",
                    )
                },
                "nearest_parent_relative_distance": f"{metrics['nearest_parent_relative_distance'][target]:.9e}",
                "nearest_parent_point_distance": f"{metrics['nearest_parent_point_distance'][target]:.9e}",
                "nearest_parent_rmse": f"{metrics['nearest_parent_rmse'][target]:.9e}",
            })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def self_check() -> dict[str, float]:
    """Check that identical empirical planes have zero descriptive KL."""
    values = np.random.default_rng(7).normal(size=(128, 3))  # shape: [N=128,D=3]
    result = marginal_histogram_kl(values, values.copy(), bins=16)
    maximum = float(np.max(np.asarray(result["reference_to_candidate"])))
    assert maximum < 1.0e-15, maximum
    return {"identical_plane_kl": maximum}


if __name__ == "__main__":
    print(self_check())

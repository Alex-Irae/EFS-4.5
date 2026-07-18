"""Numerical smoke gates and target-recovery evaluation for H4.5."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from efs import mean_pairwise_distance


RADIAL_ZONE_FIELDS = tuple(f"radial_zone_{index}" for index in range(10))

SMOKE_FIELDS = (
    "stage",
    "plane",
    "finite",
    "mean_pair_distance",
    "rms_radius",
    "radius_ratio",
    "radius_mean",
    "radius_cv",
    "expected_radius_cv",
    "radius_q05_normalized",
    "radius_q25_normalized",
    "radius_q50_normalized",
    "radius_q75_normalized",
    "radius_q95_normalized",
    "uniform_ball_radius",
    "radial_cdf_gap",
    *RADIAL_ZONE_FIELDS,
    "radial_overflow",
    "angular_resultant",
    "angular_moment_error",
    "covariance_ratio",
    "vertex_source_id",
    "vertex_particle",
    "vertex_particle_rmse",
    "vertex_particle_relative_rmse",
    "vertex_source_relative_rmse",
    "vertex_replay_residual",
    "gate_pass",
)

METRIC_FIELDS = (
    "trial",
    "trial_kind",
    "method",
    "source_ids",
    "lambda",
    "independent_control_nondegenerate",
    "clean_target_rmse",
    "clean_target_relative_rmse",
    "noisy_target_rmse",
    "jitter_error_floor",
    "centroid_rmse",
    "centered_shape_rmse",
    "per_particle_rmse",
    "mean_particle_rmse",
    "max_particle_rmse",
    "terminal_commutation_gap",
    "terminal_commutation_relative",
    "max_replay_residual",
    "template_coherence_error",
    "nearest_memory_source",
)


def radial_statistics(particles: np.ndarray) -> dict[str, float]:
    """Describe the full centered radial distribution of one plane [N,D].

    For a uniform D-ball, E[r^2] = D*R^2/(D+2), so R is fitted from the
    observed second moment. The transformed radius u=(r/R)^D is uniform on
    [0,1] under that reference model. Ten equal-width u zones therefore each
    contain 10% of an ideal sample. Values with r>R are reported as overflow.

    The CDF gap is only a descriptive maximum difference. It has no p-value and
    is not a hard validity gate.
    """
    particles = np.asarray(particles, dtype=np.float64)
    if particles.ndim != 2 or particles.shape[0] < 2:
        raise ValueError("particles must have shape [N,D] with N >= 2")

    centered = particles - np.mean(particles, axis=0, keepdims=True)  # [N,D]
    radius = np.linalg.norm(centered, axis=1)  # shape: [N]
    dimension = particles.shape[1]
    mean_square = float(np.mean(radius * radius))
    fitted_radius = math.sqrt((dimension + 2.0) / dimension * mean_square)
    sorted_radius = np.sort(radius)  # shape: [N]
    empirical_cdf = np.arange(1, radius.size + 1, dtype=np.float64) / radius.size
    if fitted_radius > 0.0:
        normalized = radius / fitted_radius  # shape: [N]
        theoretical_cdf = np.clip(
            (sorted_radius / fitted_radius) ** dimension, 0.0, 1.0
        )  # shape: [N]
        radial_cdf_gap = float(np.max(np.abs(empirical_cdf - theoretical_cdf)))
        uniform_coordinate = normalized**dimension  # shape: [N], uniform on [0,1] for an ideal D-ball
        zone_index = np.floor(10.0 * uniform_coordinate).astype(np.int64)  # shape: [N]
        zone_fraction = np.asarray([
            np.mean(zone_index == index) for index in range(10)
        ])  # shape: [10]
        overflow = float(np.mean(uniform_coordinate >= 1.0))
    else:
        normalized = np.full_like(radius, math.inf)
        radial_cdf_gap = math.inf
        zone_fraction = np.zeros(10, dtype=np.float64)
        overflow = 1.0

    quantile = np.quantile(normalized, (0.05, 0.25, 0.50, 0.75, 0.95))
    radius_mean = float(np.mean(radius))
    result = {
        "rms_radius": math.sqrt(mean_square),
        "radius_mean": radius_mean,
        "radius_cv": float(np.std(radius) / max(radius_mean, 1.0e-15)),
        "expected_radius_cv": 1.0 / math.sqrt(dimension * (dimension + 2.0)),
        "radius_q05_normalized": float(quantile[0]),
        "radius_q25_normalized": float(quantile[1]),
        "radius_q50_normalized": float(quantile[2]),
        "radius_q75_normalized": float(quantile[3]),
        "radius_q95_normalized": float(quantile[4]),
        "uniform_ball_radius": fitted_radius,
        "radial_cdf_gap": radial_cdf_gap,
        "radial_overflow": overflow,
    }
    result.update({field: float(value) for field, value in zip(RADIAL_ZONE_FIELDS, zone_fraction)})
    return result


def angular_statistics(particles: np.ndarray) -> dict[str, float]:
    """Describe centered directions for one empirical plane [N,D].

    Uniform directions have zero mean and second moment I/D. The covariance
    eigenvalue ratio detects collapse onto a lower-dimensional subspace.
    """
    particles = np.asarray(particles, dtype=np.float64)
    centered = particles - np.mean(particles, axis=0, keepdims=True)  # [N,D]
    radius = np.linalg.norm(centered, axis=1)  # shape: [N]
    valid = radius > 1.0e-15  # shape: [N]
    if np.count_nonzero(valid) < particles.shape[1] + 1:
        return {
            "angular_resultant": math.inf,
            "angular_moment_error": math.inf,
            "covariance_ratio": 0.0,
        }

    unit = centered[valid] / radius[valid, None]  # [V,D] / [V,1] -> [V,D]
    resultant = float(np.linalg.norm(np.mean(unit, axis=0)))
    second_moment = unit.T @ unit / unit.shape[0]  # [D,V] @ [V,D] -> [D,D]
    expected = np.eye(particles.shape[1]) / particles.shape[1]  # shape: [D,D]
    moment_error = float(np.linalg.norm(second_moment - expected))

    covariance = centered.T @ centered / centered.shape[0]  # [D,N] @ [N,D] -> [D,D]
    eigenvalues = np.linalg.eigvalsh(covariance)  # shape: [D]
    largest = float(np.max(eigenvalues))
    covariance_ratio = float(max(np.min(eigenvalues), 0.0) / max(largest, 1.0e-15))
    return {
        "angular_resultant": resultant,
        "angular_moment_error": moment_error,
        "covariance_ratio": covariance_ratio,
    }


def cloud_statistics(particles: np.ndarray) -> dict[str, float | bool]:
    """Return radial, angular, scale, and finiteness values for one [N,D] plane."""
    particles = np.asarray(particles, dtype=np.float64)
    finite = bool(np.all(np.isfinite(particles)))
    if not finite:
        return {"finite": False}

    statistics: dict[str, float | bool] = {
        "finite": True,
        "mean_pair_distance": mean_pairwise_distance(particles),
    }
    statistics.update(radial_statistics(particles))
    statistics.update(angular_statistics(particles))
    return statistics


def layout_cloud_statistics(particles: np.ndarray) -> list[dict[str, float | bool]]:
    """Return one statistics dictionary per field in pooled or slot memory."""
    particles = np.asarray(particles, dtype=np.float64)
    if particles.ndim == 2:
        planes = particles[None, :, :]  # [N,D] -> [H=1,N,D]
    elif particles.ndim == 3:
        planes = particles  # shape: [H=P,C,D]
    else:
        raise ValueError("memory must have shape [N,D] or [P,C,D]")

    values: list[dict[str, float | bool]] = []
    for plane_index, plane in enumerate(planes):
        statistics = cloud_statistics(plane)
        statistics["plane"] = plane_index
        values.append(statistics)
    return values


def smoke_gate(
    initial_memory: np.ndarray,
    terminal_memory: np.ndarray,
    forward_diagnostics: dict[str, np.ndarray | int | bool | str],
) -> tuple[bool, list[str], dict[str, list[dict[str, float | bool]]]]:
    """Apply only numerical, scale, and covariance hard gates.

    Radial CDF agreement, radial-zone occupancy, angular resultant, angular
    second moment, and convergence are calibration diagnostics, not blockers.
    """
    initial = layout_cloud_statistics(initial_memory)
    terminal = layout_cloud_statistics(terminal_memory)
    reasons: list[str] = []

    if not bool(forward_diagnostics.get("finite", True)):
        reasons.append(str(forward_diagnostics.get("failure_reason", "forward evolution failed")))
    if len(initial) != len(terminal):
        reasons.append("initial and terminal field counts differ")
        return False, reasons, {"initial": initial, "terminal": terminal}

    for plane_index, (initial_plane, terminal_plane) in enumerate(zip(initial, terminal)):
        finite = bool(initial_plane.get("finite", False)) and bool(terminal_plane.get("finite", False))
        if not finite:
            reasons.append(f"plane {plane_index} contains non-finite values")
            continue

        radius_ratio = float(terminal_plane["rms_radius"]) / max(float(initial_plane["rms_radius"]), 1.0e-15)
        initial_plane["radius_ratio"] = 1.0
        terminal_plane["radius_ratio"] = radius_ratio
        if not 0.05 <= radius_ratio <= 5.0:
            reasons.append(f"plane {plane_index} RMS-radius ratio {radius_ratio:.6f} is outside [0.05, 5.0]")
        if float(terminal_plane["covariance_ratio"]) < 1.0e-3:
            reasons.append(
                f"plane {plane_index} covariance ratio {float(terminal_plane['covariance_ratio']):.6e} is below 1e-3"
            )

    return len(reasons) == 0, reasons, {"initial": initial, "terminal": terminal}


def vertex_replay_errors(
    replayed_sources: np.ndarray,
    expected_sources: np.ndarray,
    memory_std: float,
    replay_residual: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return source and particle vertex errors for arrays [V,P,D].

    Particle RMSE reduces only coordinate axis D, producing [V,P]. Source RMSE
    reduces both P and D, producing [V]. Relative values use the initial memory
    standard deviation, which is the declared 10% gate scale.
    """
    replayed_sources = np.asarray(replayed_sources, dtype=np.float64)
    expected_sources = np.asarray(expected_sources, dtype=np.float64)
    if replayed_sources.shape != expected_sources.shape or replayed_sources.ndim != 3:
        raise ValueError("vertex arrays must share shape [V,P,D]")

    error = replayed_sources - expected_sources  # shape: [V,P,D]
    particle_rmse = np.sqrt(np.mean(error * error, axis=2))  # shape: [V,P]
    source_rmse = np.sqrt(np.mean(error * error, axis=(1, 2)))  # shape: [V]
    scale = max(memory_std, 1.0e-15)
    if replay_residual is None:
        particle_residual = np.full_like(particle_rmse, np.nan)
    else:
        replay_residual = np.asarray(replay_residual, dtype=np.float64)
        if replay_residual.ndim != 3 or replay_residual.shape[1:] != particle_rmse.shape:
            raise ValueError("replay_residual must have shape [J,V,P]")
        particle_residual = np.max(replay_residual, axis=0)  # [J,V,P] -> [V,P]

    return {
        "particle_rmse": particle_rmse,
        "particle_relative_rmse": particle_rmse / scale,
        "source_rmse": source_rmse,
        "source_relative_rmse": source_rmse / scale,
        "particle_max_residual": particle_residual,
    }

def template_coherence_error(ensembles: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Return source-wide template disagreement for ``[R,M,P,D]`` as ``[R,M]``."""
    adjusted = ensembles - template[None, None, :, :]  # [R,M,P,D] - [1,1,P,D]
    center = np.mean(adjusted, axis=2, keepdims=True)  # shape: [R,M,1,D]
    return np.sqrt(np.mean((adjusted - center) ** 2, axis=(2, 3)))  # shape: [R,M]


def nearest_source_distance(ensembles: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Return flattened nearest-memory distance for ``[R,M,P,D]`` as ``[R,M]``."""
    trial_count, method_count = ensembles.shape[:2]
    generated_flat = ensembles.reshape(trial_count * method_count, -1)  # [R,M,P,D] -> [R*M,P*D]
    source_flat = sources.reshape(sources.shape[0], -1)  # [C,P,D] -> [C,P*D]
    difference = generated_flat[:, None, :] - source_flat[None, :, :]  # [R*M,C,P*D]
    squared_distance = np.sum(difference * difference, axis=2)  # shape: [R*M,C]
    return np.sqrt(np.min(squared_distance, axis=1)).reshape(trial_count, method_count)


def target_recovery_metrics(
    generated: np.ndarray,
    clean_targets: np.ndarray,
    noisy_targets: np.ndarray,
    jitter: np.ndarray,
    template: np.ndarray,
    sources: np.ndarray,
    replay_residual: np.ndarray,
    commutation_gap: np.ndarray,
    memory_std: float,
    terminal_std: float,
) -> dict[str, np.ndarray]:
    """Evaluate generated methods against known targets.

    Shapes:
        generated: ``[R,M,P,D]``;
        targets and jitter: ``[R,P,D]``;
        replay_residual: ``[J,R,M,P]``;
        commutation_gap: ``[R]``.
    """
    generated = np.asarray(generated, dtype=np.float64)
    clean_error = generated - clean_targets[:, None, :, :]  # [R,M,P,D]
    noisy_error = generated - noisy_targets[:, None, :, :]  # [R,M,P,D]
    clean_rmse = np.sqrt(np.mean(clean_error * clean_error, axis=(2, 3)))  # [R,M]
    noisy_rmse = np.sqrt(np.mean(noisy_error * noisy_error, axis=(2, 3)))  # [R,M]

    generated_center = np.mean(generated, axis=2)  # shape: [R,M,D]
    target_center = np.mean(clean_targets, axis=1)[:, None, :]  # shape: [R,1,D]
    centroid_rmse = np.sqrt(np.mean((generated_center - target_center) ** 2, axis=2))  # [R,M]

    generated_centered = generated - generated_center[:, :, None, :]  # [R,M,P,D]
    target_centered = clean_targets - np.mean(clean_targets, axis=1, keepdims=True)  # [R,P,D]
    shape_rmse = np.sqrt(
        np.mean((generated_centered - target_centered[:, None, :, :]) ** 2, axis=(2, 3))
    )  # shape: [R,M]
    particle_rmse = np.sqrt(np.mean(clean_error * clean_error, axis=3))  # shape: [R,M,P]

    return {
        "clean_rmse": clean_rmse,
        "clean_relative_rmse": clean_rmse / max(memory_std, 1.0e-15),
        "noisy_rmse": noisy_rmse,
        "jitter_floor": np.sqrt(np.mean(jitter * jitter, axis=(1, 2))),  # [R]
        "centroid_rmse": centroid_rmse,
        "shape_rmse": shape_rmse,
        "particle_rmse": particle_rmse,
        "mean_particle_rmse": np.mean(particle_rmse, axis=2),
        "max_particle_rmse": np.max(particle_rmse, axis=2),
        "commutation_gap": commutation_gap,
        "commutation_relative": commutation_gap / max(terminal_std, 1.0e-15),
        "max_residual": np.max(replay_residual, axis=(0, 3)),  # [J,R,M,P] -> [R,M]
        "template_coherence": template_coherence_error(generated, template),
        "nearest_source": nearest_source_distance(generated, sources),
    }


def classify_result(
    smoke_passed: bool,
    vertex_passed: bool,
    trial_kind: np.ndarray,
    nondegenerate: np.ndarray,
    metrics: dict[str, np.ndarray] | None,
) -> dict[str, float | int | str]:
    """Classify a run as invalid, negative, mixed, or promising."""
    if not smoke_passed or not vertex_passed or metrics is None:
        return {
            "status": "invalid",
            "eligible_trials": 0,
            "shared_wins": 0,
            "shared_median_clean_relative_rmse": math.nan,
            "independent_median_clean_relative_rmse": math.nan,
        }

    eligible = (trial_kind == "random") & nondegenerate  # shape: [R]
    clean = metrics["clean_rmse"][eligible]  # shape: [E,M=2]
    clean_relative = metrics["clean_relative_rmse"][eligible]  # shape: [E,M=2]
    eligible_count = clean.shape[0]
    if eligible_count == 0:
        raise ValueError("classification needs at least one nondegenerate random trial")
    shared_wins = int(np.count_nonzero(clean[:, 0] < clean[:, 1]))
    shared_median = float(np.median(clean[:, 0]))
    independent_median = float(np.median(clean[:, 1]))
    shared_relative_median = float(np.median(clean_relative[:, 0]))
    independent_relative_median = float(np.median(clean_relative[:, 1]))

    promising_wins = math.ceil(0.75 * eligible_count)
    if shared_wins >= promising_wins and shared_relative_median <= 0.10 and shared_median < independent_median:
        status = "promising"
    elif shared_wins <= eligible_count / 2.0 or shared_median >= independent_median:
        status = "negative"
    else:
        status = "mixed"

    return {
        "status": status,
        "eligible_trials": eligible_count,
        "shared_wins": shared_wins,
        "shared_median_clean_rmse": shared_median,
        "independent_median_clean_rmse": independent_median,
        "shared_median_clean_relative_rmse": shared_relative_median,
        "independent_median_clean_relative_rmse": independent_relative_median,
    }


def build_metric_rows(
    trial_kind: np.ndarray,
    source_ids: np.ndarray,
    lambdas: np.ndarray,
    nondegenerate: np.ndarray,
    method_names: np.ndarray,
    metrics: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    """Flatten trial and method metrics into CSV-ready dictionaries."""
    rows: list[dict[str, object]] = []
    for trial in range(trial_kind.size):
        for method, method_name in enumerate(method_names):
            rows.append({
                "trial": trial,
                "trial_kind": str(trial_kind[trial]),
                "method": str(method_name),
                "source_ids": ";".join(str(value) for value in source_ids[trial]),
                "lambda": ";".join(f"{value:.9f}" for value in lambdas[trial]),
                "independent_control_nondegenerate": int(nondegenerate[trial]),
                "clean_target_rmse": f"{metrics['clean_rmse'][trial, method]:.9e}",
                "clean_target_relative_rmse": f"{metrics['clean_relative_rmse'][trial, method]:.9e}",
                "noisy_target_rmse": f"{metrics['noisy_rmse'][trial, method]:.9e}",
                "jitter_error_floor": f"{metrics['jitter_floor'][trial]:.9e}",
                "centroid_rmse": f"{metrics['centroid_rmse'][trial, method]:.9e}",
                "centered_shape_rmse": f"{metrics['shape_rmse'][trial, method]:.9e}",
                "per_particle_rmse": ";".join(
                    f"{value:.9e}" for value in metrics["particle_rmse"][trial, method]
                ),
                "mean_particle_rmse": f"{metrics['mean_particle_rmse'][trial, method]:.9e}",
                "max_particle_rmse": f"{metrics['max_particle_rmse'][trial, method]:.9e}",
                "terminal_commutation_gap": (
                    f"{metrics['commutation_gap'][trial]:.9e}" if method == 0 else ""
                ),
                "terminal_commutation_relative": (
                    f"{metrics['commutation_relative'][trial]:.9e}" if method == 0 else ""
                ),
                "max_replay_residual": f"{metrics['max_residual'][trial, method]:.9e}",
                "template_coherence_error": f"{metrics['template_coherence'][trial, method]:.9e}",
                "nearest_memory_source": f"{metrics['nearest_source'][trial, method]:.9e}",
            })
    return rows


def write_metrics_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write target metrics, or only the header when smoke gates failed."""
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_smoke_csv(
    path: Path,
    cloud_metrics: dict[str, list[dict[str, float | bool]]],
    smoke_passed: bool,
    vertex_source_ids: np.ndarray | None = None,
    vertex_metrics: dict[str, np.ndarray] | None = None,
) -> None:
    """Write per-field endpoint diagnostics and optional particle vertices."""
    rows: list[dict[str, object]] = []
    for stage in ("initial", "terminal"):
        for statistics in cloud_metrics[stage]:
            row: dict[str, object] = {field: "" for field in SMOKE_FIELDS}
            row.update(statistics)
            row["stage"] = stage
            row["gate_pass"] = int(smoke_passed)
            rows.append(row)

    if vertex_source_ids is not None and vertex_metrics is not None:
        particle_rmse = vertex_metrics["particle_rmse"]  # shape: [V,P]
        particle_relative = vertex_metrics["particle_relative_rmse"]  # shape: [V,P]
        source_relative = vertex_metrics["source_relative_rmse"]  # shape: [V]
        particle_residual = vertex_metrics["particle_max_residual"]  # shape: [V,P]
        for vertex, source_id in enumerate(vertex_source_ids):
            for particle in range(particle_rmse.shape[1]):
                relative_error = float(particle_relative[vertex, particle])
                row = {field: "" for field in SMOKE_FIELDS}
                row.update({
                    "stage": "vertex",
                    "vertex_source_id": int(source_id),
                    "vertex_particle": particle,
                    "vertex_particle_rmse": f"{particle_rmse[vertex, particle]:.9e}",
                    "vertex_particle_relative_rmse": f"{relative_error:.9e}",
                    "vertex_source_relative_rmse": f"{source_relative[vertex]:.9e}",
                    "vertex_replay_residual": f"{particle_residual[vertex, particle]:.9e}",
                    "gate_pass": int(relative_error <= 0.10),
                })
                rows.append(row)

    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SMOKE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

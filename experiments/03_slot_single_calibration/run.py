"""Run one fixed, calibrated H4.5 target-recovery configuration.

Purpose:
    Build slot or pooled EFS memory, apply hard numerical and vertex gates,
    recover controlled shared-lambda targets, and save arrays and figures.

Dependencies:
    NumPy and Matplotlib from ../../requirements.txt.

Outputs:
    A new timestamped directory below --output-root.

Exact command:
    python run.py --layout slot --forward-steps 5000 --output-root results
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from data import (
    METHOD_NAMES,
    arrange_memory,
    build_target_trials,
    build_terminal_candidates,
    generate_grouped_sources,
    parse_shared_lambda,
    restore_sources,
)
from efs import backward_replay, forward_history, passive_forward
from evaluation import (
    build_metric_rows,
    classify_result,
    smoke_gate,
    target_recovery_metrics,
    vertex_replay_errors,
    write_metrics_csv,
    write_smoke_csv,
)
from plotting import plot_all


def _validate_arguments(args: argparse.Namespace) -> tuple[int, float, np.ndarray | None]:
    """Validate CLI relationships and return C, actual s, and operator lambda."""
    positive_integer_names = (
        "plane_particles",
        "particles_per_source",
        "selected_sources",
        "trials",
        "vertex_sources",
        "dimension",
        "forward_steps",
        "proximal_steps",
    )
    for name in positive_integer_names:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.layout not in ("slot", "single"):
        raise ValueError("--layout must be slot or single")
    if args.plane_particles % args.particles_per_source != 0:
        raise ValueError("--plane-particles must be divisible by --particles-per-source")
    if args.dimension < 2 or args.particles_per_source < 2 or args.selected_sources < 2:
        raise ValueError("dimension, particles per source, and selected sources must be at least two")
    if args.gamma <= 0.0 or args.epsilon <= 0.0 or args.beta <= 0.0:
        raise ValueError("gamma, epsilon, and beta must be positive")
    if args.center_std <= 0.0 or args.particle_scale <= 0.0 or args.noise_std < 0.0:
        raise ValueError("invalid synthetic-data scale")
    if args.log_every < 0:
        raise ValueError("--log-every must be non-negative")

    source_count = args.plane_particles // args.particles_per_source
    if source_count < max(args.selected_sources, args.vertex_sources, 4):
        raise ValueError("the memory does not contain enough complete sources")
    exponent_s = float(args.dimension - 2 if args.exponent_s is None else args.exponent_s)
    if exponent_s <= 0.0:
        raise ValueError("--exponent-s must be positive")
    operator_lambda = parse_shared_lambda(args.shared_lambda, args.selected_sources)
    return source_count, exponent_s, operator_lambda


def _new_run_directory(output_root: str, seed: int, layout: str) -> Path:
    """Create one append-only timestamped result directory."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_directory = root / f"h45_target_{timestamp}_{layout}_seed{seed}"
    run_directory.mkdir(exist_ok=False)
    return run_directory


def _save_run(
    run_directory: Path,
    arrays: dict[str, np.ndarray],
    config: dict[str, object],
    summary_lines: list[str],
    cloud_metrics: dict[str, list[dict[str, float | bool]]],
    smoke_passed: bool,
    metric_rows: list[dict[str, object]],
    vertex_source_ids: np.ndarray | None = None,
    vertex_metrics: dict[str, np.ndarray] | None = None,
) -> list[Path]:
    """Write one run exactly once, then generate every applicable figure."""
    with (run_directory / "config.json").open("x", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    write_smoke_csv(
        run_directory / "smoke.csv",
        cloud_metrics,
        smoke_passed,
        vertex_source_ids,
        vertex_metrics,
    )
    write_metrics_csv(run_directory / "metrics.csv", metric_rows)
    with (run_directory / "summary.txt").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines) + "\n")

    # Histories are intentionally uncompressed because replay and plotting are
    # CPU-bound already, and the user's 32 GiB RAM budget can hold these arrays.
    np.savez(run_directory / "samples.npz", **arrays)
    return plot_all(run_directory)


def _forward_arrays(
    layout: str,
    sources: np.ndarray,
    centers: np.ndarray,
    template: np.ndarray,
    initial_memory: np.ndarray,
    history: np.ndarray,
    diagnostics: dict[str, np.ndarray | int | bool | str],
) -> dict[str, np.ndarray]:
    """Collect arrays common to successful and failed forward runs."""
    return {
        "layout": np.asarray(layout),
        "sources": sources,
        "centers": centers,
        "template": template,
        "initial_memory": initial_memory,
        "history": history,
        "forward_saved_step": np.asarray(diagnostics["saved_step"]),
        "forward_log_step": np.asarray(diagnostics["log_step"]),
        "forward_mean_pair_distance": np.asarray(diagnostics["mean_pair_distance"]),
        "forward_rms_radius": np.asarray(diagnostics["rms_radius"]),
        "forward_max_force_norm": np.asarray(diagnostics["max_force_norm"]),
        "forward_max_update_norm": np.asarray(diagnostics["max_update_norm"]),
    }


def run(args: argparse.Namespace) -> tuple[Path, str]:
    """Execute one fixed H4.5 configuration and return its directory and status."""
    source_count, exponent_s, operator_lambda = _validate_arguments(args)
    run_directory = _new_run_directory(args.output_root, args.seed, args.layout)
    created_utc = datetime.now(timezone.utc).isoformat()

    if args.layout == "slot":
        interactions = args.particles_per_source * source_count**2
        field_description = f"{args.particles_per_source} slot fields of C={source_count}"
    else:
        interactions = args.plane_particles**2
        field_description = f"one pooled field of N={args.plane_particles}"
    history_gib = (
        (args.forward_steps + 1)
        * args.plane_particles
        * args.dimension
        * np.dtype(np.float64).itemsize
        / 1024**3
    )
    print(
        f"{field_description}, D={args.dimension}, scalar pairs/frame={interactions:,}, "
        f"history={history_gib:.2f} GiB"
    )

    data_rng = np.random.default_rng(args.seed)
    sources, centers, template = generate_grouped_sources(
        source_count,
        args.particles_per_source,
        args.dimension,
        args.center_std,
        args.particle_scale,
        args.noise_std,
        data_rng,
    )  # [C,P,D], [C,D], [P,D]
    initial_memory = arrange_memory(sources, args.layout)  # slot [P,C,D] or single [C*P,D]

    forward_start = time.perf_counter()
    history, forward_diagnostics = forward_history(
        initial_memory,
        gamma=args.gamma,
        epsilon=args.epsilon,
        exponent_s=exponent_s,
        steps=args.forward_steps,
        log_every=args.log_every,
    )
    forward_seconds = time.perf_counter() - forward_start

    smoke_passed, smoke_reasons, cloud_metrics = smoke_gate(
        initial_memory, history[-1], forward_diagnostics
    )
    arrays = _forward_arrays(
        args.layout,
        sources,
        centers,
        template,
        initial_memory,
        history,
        forward_diagnostics,
    )
    config: dict[str, object] = vars(args).copy()
    config.update({
        "created_utc": created_utc,
        "source_count": source_count,
        "exponent_s_actual": exponent_s,
        "operator_lambda_values": operator_lambda.tolist() if operator_lambda is not None else None,
        "actual_forward_steps": int(forward_diagnostics["actual_steps"]),
        "forward_finite": bool(forward_diagnostics["finite"]),
        "forward_failure_reason": str(forward_diagnostics["failure_reason"]),
        "forward_seconds": forward_seconds,
        "forward_smoke_passed": smoke_passed,
        "forward_smoke_reasons": smoke_reasons,
        "smoke_metrics": cloud_metrics,
    })

    if not smoke_passed:
        config.update({"status": "invalid", "vertex_passed": False, "backward_seconds": 0.0})
        summary = [
            f"H4.5 target-recovery run ({args.layout} layout)",
            "",
            "status: invalid",
            f"forward frames: {history.shape[0] - 1}",
            f"forward runtime: {forward_seconds:.2f} s",
            "target interpolation was skipped because a hard forward gate failed.",
            "",
            "failure reasons:",
            *(f"- {reason}" for reason in smoke_reasons),
        ]
        figures = _save_run(run_directory, arrays, config, summary, cloud_metrics, False, [])
        print(f"saved invalid smoke run and {len(figures)} figure(s)")
        return run_directory, "invalid"

    terminal_sources = restore_sources(
        history[-1], args.layout, args.particles_per_source
    )  # [P,C,D] or [C*P,D] -> [C,P,D]
    arrays["terminal_sources"] = terminal_sources
    memory_std = float(np.std(sources))
    terminal_std = float(np.std(history[-1]))

    vertex_rng = np.random.default_rng(args.seed + 1)
    vertex_source_ids = vertex_rng.choice(
        source_count, size=args.vertex_sources, replace=False
    )  # shape: [V]
    vertex_terminal = terminal_sources[vertex_source_ids]  # shape: [V,P,D]
    vertex_start = time.perf_counter()
    try:
        vertex_trajectory, vertex_residual, vertex_logs = backward_replay(
            vertex_terminal,
            history,
            gamma=args.gamma,
            epsilon=args.epsilon,
            exponent_s=exponent_s,
            beta=args.beta,
            proximal_steps=args.proximal_steps,
            method_names=["vertex"] * args.vertex_sources,
            log_every=args.log_every,
        )  # [J+1,V,P,D], [J,V,P]
        vertex_metrics = vertex_replay_errors(
            vertex_trajectory[0],
            sources[vertex_source_ids],
            memory_std,
            vertex_residual,
        )
        particle_relative = vertex_metrics["particle_relative_rmse"]  # shape: [V,P]
        vertex_passed = bool(
            np.all(np.isfinite(particle_relative)) and np.max(particle_relative) <= 0.10
        )
        vertex_failure = "" if vertex_passed else "maximum particle vertex RMSE exceeds 10% of memory scale"
        arrays.update({
            "vertex_source_ids": vertex_source_ids,
            "vertex_terminal": vertex_terminal,
            "vertex_trajectory": vertex_trajectory,
            "vertex_residual": vertex_residual,
            "vertex_particle_rmse": vertex_metrics["particle_rmse"],
            "vertex_particle_relative_rmse": particle_relative,
            "vertex_source_rmse": vertex_metrics["source_rmse"],
            "vertex_source_relative_rmse": vertex_metrics["source_relative_rmse"],
            "vertex_particle_max_residual": vertex_metrics["particle_max_residual"],
            "vertex_backward_log_step": vertex_logs["reverse_step"],
            "vertex_backward_log_mean_distance": vertex_logs["mean_pair_distance"],
            "vertex_backward_log_total_position": vertex_logs["total_position"],
            "vertex_backward_log_displacement": vertex_logs["displacement"],
        })
    except FloatingPointError as error:
        vertex_passed = False
        vertex_failure = str(error)
        shape = (args.vertex_sources, args.particles_per_source)
        vertex_metrics = {
            "particle_rmse": np.full(shape, np.nan),
            "particle_relative_rmse": np.full(shape, np.inf),
            "source_rmse": np.full(args.vertex_sources, np.nan),
            "source_relative_rmse": np.full(args.vertex_sources, np.inf),
            "particle_max_residual": np.full(shape, np.inf),
        }
        arrays.update({
            "vertex_source_ids": vertex_source_ids,
            "vertex_terminal": vertex_terminal,
            "vertex_particle_rmse": vertex_metrics["particle_rmse"],
            "vertex_particle_relative_rmse": vertex_metrics["particle_relative_rmse"],
            "vertex_source_rmse": vertex_metrics["source_rmse"],
            "vertex_source_relative_rmse": vertex_metrics["source_relative_rmse"],
            "vertex_particle_max_residual": vertex_metrics["particle_max_residual"],
        })
    vertex_seconds = time.perf_counter() - vertex_start

    particle_relative = vertex_metrics["particle_relative_rmse"]
    finite_particle_error = particle_relative[np.isfinite(particle_relative)]
    percentiles = (
        np.quantile(finite_particle_error, (0.50, 0.90, 0.95))
        if finite_particle_error.size
        else np.full(3, np.nan)
    )
    config.update({
        "vertex_source_ids": vertex_source_ids.tolist(),
        "vertex_passed": vertex_passed,
        "vertex_failure_reason": vertex_failure,
        "vertex_particle_median_relative_rmse": float(percentiles[0]),
        "vertex_particle_p90_relative_rmse": float(percentiles[1]),
        "vertex_particle_p95_relative_rmse": float(percentiles[2]),
        "vertex_particle_max_relative_rmse": float(np.max(particle_relative)),
        "vertex_source_relative_rmse": vertex_metrics["source_relative_rmse"].tolist(),
        "vertex_seconds": vertex_seconds,
    })

    if not vertex_passed:
        config.update({"status": "invalid", "backward_seconds": vertex_seconds})
        summary = [
            f"H4.5 target-recovery run ({args.layout} layout)",
            "",
            "status: invalid",
            f"forward frames: {history.shape[0] - 1}",
            f"forward runtime: {forward_seconds:.2f} s",
            f"vertex runtime: {vertex_seconds:.2f} s",
            f"vertex particle median relative RMSE: {percentiles[0]:.8e}",
            f"vertex particle p95 relative RMSE: {percentiles[2]:.8e}",
            f"vertex particle maximum relative RMSE: {np.max(particle_relative):.8e}",
            "target interpolation was skipped because vertex replay failed.",
            "",
            f"failure reason: {vertex_failure}",
        ]
        figures = _save_run(
            run_directory,
            arrays,
            config,
            summary,
            cloud_metrics,
            True,
            [],
            vertex_source_ids,
            vertex_metrics,
        )
        print(f"saved invalid vertex run and {len(figures)} figure(s)")
        return run_directory, "invalid"

    target_rng = np.random.default_rng(args.seed + 2)
    target_data = build_target_trials(
        sources,
        args.trials,
        args.selected_sources,
        args.noise_std,
        target_rng,
        operator_lambda,
    )
    trial_count = target_data["trial_kind"].size
    terminal_candidates = build_terminal_candidates(
        terminal_sources,
        target_data["source_ids"],
        target_data["lambdas"],
        target_data["independent_lambdas"],
    )  # shape: [R,M=2,P,D]
    arrays.update({
        "method_names": METHOD_NAMES,
        "trial_kind": target_data["trial_kind"],
        "target_source_ids": target_data["source_ids"],
        "target_lambdas": target_data["lambdas"],
        "independent_lambdas": target_data["independent_lambdas"],
        "independent_control_nondegenerate": target_data["nondegenerate"],
        "clean_targets": target_data["clean_targets"],
        "target_jitter_std": target_data["jitter_std"],
        "target_jitter": target_data["jitter"],
        "noisy_targets": target_data["noisy_targets"],
        "terminal_candidates": terminal_candidates,
    })

    target_start = time.perf_counter()
    try:
        oracle_terminal = passive_forward(
            target_data["clean_targets"],
            history,
            gamma=args.gamma,
            epsilon=args.epsilon,
            exponent_s=exponent_s,
        )  # shape: [R,P,D]
        commutation_gap = np.sqrt(
            np.mean((terminal_candidates[:, 0] - oracle_terminal) ** 2, axis=(1, 2))
        )  # shape: [R]

        flat_terminal = terminal_candidates.reshape(
            trial_count * METHOD_NAMES.size,
            args.particles_per_source,
            args.dimension,
        )  # [R,M,P,D] -> [R*M,P,D]
        flat_method_names = np.tile(METHOD_NAMES, trial_count).tolist()
        flat_trajectory, flat_residual, target_logs = backward_replay(
            flat_terminal,
            history,
            gamma=args.gamma,
            epsilon=args.epsilon,
            exponent_s=exponent_s,
            beta=args.beta,
            proximal_steps=args.proximal_steps,
            method_names=flat_method_names,
            log_every=args.log_every,
        )
        target_trajectory = flat_trajectory.reshape(
            history.shape[0],
            trial_count,
            METHOD_NAMES.size,
            args.particles_per_source,
            args.dimension,
        )  # [J+1,R*M,P,D] -> [J+1,R,M,P,D]
        target_residual = flat_residual.reshape(
            history.shape[0] - 1,
            trial_count,
            METHOD_NAMES.size,
            args.particles_per_source,
        )  # [J,R*M,P] -> [J,R,M,P]
        generated = target_trajectory[0]  # shape: [R,M,P,D]

        metrics = target_recovery_metrics(
            generated,
            target_data["clean_targets"],
            target_data["noisy_targets"],
            target_data["jitter"],
            template,
            sources,
            target_residual,
            commutation_gap,
            memory_std,
            terminal_std,
        )
        classification = classify_result(
            True,
            True,
            target_data["trial_kind"],
            target_data["nondegenerate"],
            metrics,
        )
        metric_rows = build_metric_rows(
            target_data["trial_kind"],
            target_data["source_ids"],
            target_data["lambdas"],
            target_data["nondegenerate"],
            METHOD_NAMES,
            metrics,
        )
        arrays.update({
            "oracle_terminal": oracle_terminal,
            "terminal_commutation_gap": commutation_gap,
            "target_trajectory": target_trajectory,
            "target_residual": target_residual,
            "generated": generated,
            "target_backward_log_step": target_logs["reverse_step"],
            "target_backward_log_mean_distance": target_logs["mean_pair_distance"].reshape(
                -1, trial_count, METHOD_NAMES.size
            ),
            "target_backward_log_total_position": target_logs["total_position"].reshape(
                -1, trial_count, METHOD_NAMES.size, args.dimension
            ),
            "target_backward_log_displacement": target_logs["displacement"].reshape(
                -1, trial_count, METHOD_NAMES.size, args.dimension
            ),
            **{f"metric_{name}": value for name, value in metrics.items()},
        })
        target_failure = ""
    except FloatingPointError as error:
        metrics = None
        metric_rows = []
        classification = {"status": "invalid"}
        target_failure = str(error)
    target_seconds = time.perf_counter() - target_start
    total_backward_seconds = vertex_seconds + target_seconds

    config.update({
        "status": str(classification["status"]),
        "trial_count_actual": trial_count,
        "target_failure_reason": target_failure,
        "target_seconds": target_seconds,
        "backward_seconds": total_backward_seconds,
        "classification": classification,
    })

    terminal_gaps = [float(values["radial_cdf_gap"]) for values in cloud_metrics["terminal"]]
    if metrics is None:
        summary = [
            f"H4.5 target-recovery run ({args.layout} layout)",
            "",
            "status: invalid",
            f"forward frames: {history.shape[0] - 1}",
            f"vertex particle p95 relative RMSE: {percentiles[2]:.8e}",
            f"target runtime before failure: {target_seconds:.2f} s",
            f"failure reason: {target_failure}",
        ]
    else:
        summary = [
            f"H4.5 target-recovery run ({args.layout} layout)",
            "",
            f"status: {classification['status']}",
            f"forward frames: {history.shape[0] - 1}",
            f"worst descriptive radial CDF gap: {max(terminal_gaps):.8f}",
            f"vertex particle median relative RMSE: {percentiles[0]:.8e}",
            f"vertex particle p90 relative RMSE: {percentiles[1]:.8e}",
            f"vertex particle p95 relative RMSE: {percentiles[2]:.8e}",
            f"vertex particle maximum relative RMSE: {np.max(particle_relative):.8e}",
            f"eligible random trials: {classification['eligible_trials']}",
            f"shared clean-target wins: {classification['shared_wins']}",
            f"shared median clean relative RMSE: {classification['shared_median_clean_relative_rmse']:.8e}",
            f"independent median clean relative RMSE: {classification['independent_median_clean_relative_rmse']:.8e}",
            f"median terminal commutation relative gap: {np.median(metrics['commutation_relative']):.8e}",
            f"forward runtime: {forward_seconds:.2f} s",
            f"vertex runtime: {vertex_seconds:.2f} s",
            f"target runtime: {target_seconds:.2f} s",
            "",
            "Radial and angular diagnostics are descriptive. Classification makes no formal significance claim.",
        ]

    figures = _save_run(
        run_directory,
        arrays,
        config,
        summary,
        cloud_metrics,
        True,
        metric_rows,
        vertex_source_ids,
        vertex_metrics,
    )
    print(f"saved {len(figures)} figure(s)")
    return run_directory, str(classification["status"])


def build_parser() -> argparse.ArgumentParser:
    """Define one frozen-layout H4.5 experiment."""
    parser = argparse.ArgumentParser(description="Run fixed-parameter H4.5 target recovery.")
    parser.add_argument("--output-root", default="results", help="Parent directory for a new timestamped run.")
    parser.add_argument("--layout", choices=("slot", "single"), default="slot", help="Corresponding-slot model or pooled control.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for data, vertices, sources, lambdas, and jitter.")
    parser.add_argument("--plane-particles", type=int, default=2048, help="Total memory particles C*P.")
    parser.add_argument("--particles-per-source", type=int, default=8, help="Corresponding particles P per complete source.")
    parser.add_argument("--selected-sources", type=int, default=4, help="Complete parents S in every interpolation.")
    parser.add_argument("--trials", type=int, default=16, help="Random controlled targets sharing one history.")
    parser.add_argument("--vertex-sources", type=int, default=4, help="Complete source vertices replayed before targets.")
    parser.add_argument("--dimension", type=int, default=4, help="Dimension D of every particle.")
    parser.add_argument("--center-std", type=float, default=2.0, help="Standard deviation of complete-source centers.")
    parser.add_argument("--particle-scale", type=float, default=1.0, help="Scale of the corresponding-particle template.")
    parser.add_argument("--noise-std", type=float, default=0.05, help="Per-source particle jitter standard deviation.")
    parser.add_argument("--gamma", type=float, default=0.0005, help="Forward Euler and passive replay field scale.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Positive inverse-power regularizer.")
    parser.add_argument("--exponent-s", type=float, default=None, help="EFS exponent s; default is D-2.")
    parser.add_argument("--forward-steps", type=int, default=5000, help="Fixed forward frames selected before target evaluation.")
    parser.add_argument("--beta", type=float, default=0.05, help="Backward fixed-point optimizer step.")
    parser.add_argument("--proximal-steps", type=int, default=100, help="Inner replay iterations T per reverse frame.")
    parser.add_argument("--log-every", type=int, default=10, help="Print forward and reverse diagnostics every N frames; zero disables.")
    parser.add_argument("--shared-lambda", default=None, help="Optional operator vector such as 0.4,0.3,0.2,0.1; adds one trial.")
    return parser


def main() -> None:
    """Run the parsed configuration and print final status and directory."""
    run_directory, status = run(build_parser().parse_args())
    print(f"status: {status}")
    print(f"results: {run_directory.resolve()}")


if __name__ == "__main__":
    main()

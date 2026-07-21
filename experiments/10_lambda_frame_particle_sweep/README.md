# Experiment 10: Lambda Frame and Particle-Count Sweep

This experiment is an independent, copy-based extension of Experiment 07. It asks two controlled questions:

1. Does fitting barycentric coefficients at the original data plane $d_1$ produce better final reconstruction than fitting them at the terminal EFS plane $d_u$?
2. Is failure caused by forcing every particle in a complete source to share one coefficient vector?

The $d_u$ fit remains the frame-consistent H4.5 seed construction. The $d_1$ fit tests the separate commutation hypothesis

$$
F\!\left(\sum_k\lambda_k x_k\right)
\stackrel{?}{\approx}
\sum_k\lambda_k F(x_k),
$$

which a nonlinear EFS map does not guarantee.

Per-particle lambdas are capacity controls. They deliberately remove the shared source-level coupling that defines formal H4.5.

## One-line pipeline

`d1 target and four complete parents -> fit shared/per-particle lambda at d1 and du -> remove target -> rebuild EFS -> create four du seeds -> replay -> paired reconstruction metrics`

## Hypothesis-to-code mapping

| Requirement | Code | Scientific reason |
| --- | --- | --- |
| Independent Experiment 07 implementation | copied `run.py`, `data.py`, `efs.py`, `evaluation.py`, and `plotting.py` | Later root changes cannot alter this experiment. |
| Parent meaning fixed at $d_1$ | `select_source_neighbors()` | EFS is not allowed to redefine which complete sources are related. |
| One formal H4.5 lambda | `fit_shared_source_lambda()` | One K-vector is reused for every corresponding particle. |
| Per-particle capacity control | `fit_per_particle_lambdas()` | Tests whether the shared K-vector is the representational bottleneck. |
| Same fields for all methods | `run_two_pass()` | Removes field randomness from the method comparison. |
| Target-free operator path | `run_operator()` | Separates deployable parent/lambda input from hidden-target evaluation. |
| Resource-controlled particle sweep | `sweep.py` | Runs bounded concurrent conditions, streams prefixed logs, and preserves interrupted attempts. |

## Mathematical and algorithmic formulation

### Saved parents and correspondence

For target $T\in\mathbb{R}^{P\times D}$, the four closest complete sources are selected at $d_1$ using exact permutation-invariant standardized RMSE. Their identities and target-relative particle permutations are frozen through both forward fields.

### Shared lambda

At frame $f\in\{d_1,d_u\}$, one source-level vector is fitted:

$$
\lambda_f^*
=
\underset{\lambda_k\geq0,\ \sum_k\lambda_k=1}{\operatorname{argmin}}
\frac{1}{PD}
\sum_{p,d}
\left[
T_{p,d}^{(f)}-\sum_k\lambda_kS_{k,p,d}^{(f)}
\right]^2.
$$

### Per-particle control

Each corresponding particle receives its own K-vector:

$$
\lambda_{p,f}^*
=
\underset{\lambda_{p,k}\geq0,\ \sum_k\lambda_{p,k}=1}{\operatorname{argmin}}
\frac{1}{D}
\sum_d
\left[
T_{p,d}^{(f)}-\sum_k\lambda_{p,k}S_{k,p,d}^{(f)}
\right]^2.
$$

The complete parent identities and correspondence remain fixed. Only the sharing constraint changes.

### Four paired methods

The method order is:

1. `shared_d1`;
2. `shared_du`;
3. `per_particle_d1`;
4. `per_particle_du`.

Every fitted lambda is applied to the same pass-2 terminal parents, after the target has been removed and EFS has been recomputed from the original retained particles. Every terminal candidate is then replayed through that same pass-2 history.

### Target-free operator inference

Operator mode requires four parent source IDs. The first parent defines the temporary $d_1$ particle order, and every other parent is exactly aligned to it. A supplied shared lambda is validated as a simplex vector. If omitted, one deterministic $\operatorname{Dirichlet}(1)$ vector is sampled.

The same lambda constructs:

$$
Y^{(1)}=\sum_k\lambda_kS_k^{(1)},
\qquad
Y^{(u)}=\sum_k\lambda_kS_k^{(u)}.
$$

$Y^{(u)}$ is replayed, and its output is compared with $Y^{(1)}$. No target-accuracy claim is possible because operator mode has no hidden target.

## Result interpretation

### Core metrics

| Metric | Inputs and output | Meaning | Desired direction and limitations |
| --- | --- | --- | --- |
| `lambda_fit_rmse` | Target and parents in the declared fit frame | Best achieved convex representation in that frame | Lower is better; it does not measure replay. |
| `cross_frame_rmse` | Fitted lambda applied in the other frame | Barycentric frame-transfer error | Lower is better; large values reject approximate commutation. |
| `generation_terminal_rmse` | Pass-2 seed and saved pass-1 target endpoint | Target-removal plus interpolation error | Lower is better; it includes the small field change. |
| `generated_set_rmse` | Replayed output and original target | Main reconstruction error | Lower is better; it remains a synthetic source metric. |
| `direct_set_rmse` | Same lambda and parents mixed directly at $d_1$ | No-EFS baseline | EFS adds value only if replay is lower. |
| `efs_direct_ratio` | Replay error divided by direct error | Relative EFS contribution | Below one favors EFS; unstable when direct error is nearly zero. |
| `terminal_gap` | Pass-2 seed and pass-1 terminal target | Terminal starting mismatch | Lower is better; it is not a replay residual. |
| `max_replay_residual` | Backward fixed-point equation | Numerical solver convergence | Near zero is desired; it does not establish validity. |

At $P=1$, shared and per-particle fits are algebraically identical and must agree within `1e-10`. This does not guarantee exact reconstruction. With $K=4$, the convex simplex has only three free degrees of freedom, while the default particle dimension is $D=4$.

### Figures

`two_pass_target.png` shows the $d_1$ candidates, pass-1 terminal candidates, pass-2 seeds, and all four replay outputs. Close terminal candidates followed by distant arrivals indicate replay distortion. Poor candidates before replay indicate representation failure.

`two_pass_field_shift.png` shows retained-particle terminal displacement and marginal histogram KL between the target-present and target-removed fields.

`particle_sweep.png` shows individual seeds and median trends against particle count for in-frame fit, cross-frame transfer, final arrival, and EFS/direct ratio. A useful H4.5 result would require the shared method to remain accurate as $P$ grows and to outperform its direct baseline.

`operator_inference.png` shows operator parents and their direct $d_1$ mixture, the same lambda at $d_u$, and replayed output versus the direct mixture.

## Files

- `run.py`: paired two-pass reconstruction and target-free operator mode.
- `data.py`: copied source generation, matching, simplex fitting, and the new per-particle control.
- `efs.py`: unchanged copied EFS equations.
- `evaluation.py`: method-aware reconstruction CSV and metrics.
- `plotting.py`: compact-result figures.
- `sweep.py`: bounded concurrent execution, live prefixed logs, resume state, validation, aggregation, and sweep figure.
- `best_config.json`: copied Experiment 07 calibration.
- `../../requirements.txt`: root NumPy and Matplotlib requirements.

## Environment and commands

Use the existing Python 3.10 Conda environment. No new dependency is required.

Small checks:

```powershell
conda run -n anewomni python run.py --protocol two-pass --quick --particles-per-source 1 --workers 4 --output-root results\smoke_checks\p1
conda run -n anewomni python run.py --protocol two-pass --quick --particles-per-source 4 --workers 4 --output-root results\smoke_checks\p4
```

Fresh three-seed sweep with visible progress:

```powershell
$conditionWorkers = 4
$replayWorkers = 4
conda run --no-capture-output -n anewomni python -u sweep.py --particles 1 2 3 4 5 6 7 8 9 10 --seeds 45 46 47 --source-count 512 --condition-workers $conditionWorkers --workers $replayWorkers --log-every 25 --output-root results
```

`--condition-workers` is the main CPU and RAM knob because each full EFS forward recurrence is otherwise single-condition work. `--workers` only divides passive replay and has no useful value above `4` for the four-method batches here. The command runs at most four full conditions concurrently and at most sixteen replay threads transiently. The interrupted run observed roughly 0.5 to 1.5 GB working memory for one condition, depending on $P$ and protocol phase, so four concurrent conditions can require several GB plus NumPy and operating-system overhead.

The `--no-capture-output`, `python -u`, and `--log-every 25` combination is intentional. The terminal shows a sweep bar, `START`/`DONE` condition counts, and child lines such as `[seed45_p08] forward step 250/2000`. The same child output is written to `conditions/<condition>/attempt_XXX.log`.

Resume an interrupted sweep:

```powershell
conda run --no-capture-output -n anewomni python -u sweep.py --resume results\<sweep-directory> --condition-workers 4 --workers 4 --log-every 25
```

Target-free operator example:

```powershell
conda run -n anewomni python run.py --protocol operator --parent-source-ids 0,1,2,3 --shared-lambda 0.4,0.3,0.2,0.1 --output-root results\operator
```

## Parameters

| Argument | Sweep value | Meaning |
| --- | ---: | --- |
| `--particles` | `1` through `10` | Particles per complete source. |
| `--seeds` | `45 46 47` | Fresh independent synthetic fields and targets. |
| `--source-count` | `512` | Retained pass-2 source library. |
| `--dimension` | `4` | Coordinates per particle. |
| `--parents` | `4` | Complete parent sources. |
| `--condition-workers` | `4` | Simultaneous full conditions and primary CPU/RAM control. |
| `--workers` | `4` | Passive replay threads inside each condition. |
| `--log-every` | `25` | EFS frames between live terminal updates. |
| `--forward-steps` | `2000` | Copied calibrated forward frames. |
| `--proximal-steps` | `50` | Copied backward fixed-point updates per frame. |
| `--lambda-iterations` | `1000` | Projected-gradient updates per simplex fit. |

The exact forward field remains $O((CP)^2)$. Concurrent conditions trade RAM for wall-clock time, so speedup is hardware-dependent and ceases to be useful once memory bandwidth, RAM, or thermal limits dominate. Lower `--condition-workers` first if the machine swaps or becomes unresponsive. Lower `--workers` if replay briefly oversubscribes CPU.

## Assumptions and edge cases

- Source ownership remains metadata and never changes EFS forces.
- All four methods use identical parents, matching, fields, and numerical replay settings.
- Per-particle controls cannot select different complete parents.
- The same Experiment 07 calibration is used for every $P$ so the lambda comparison is not confounded by per-condition parameter tuning.
- Forward and vertex hard gates remain authoritative. Invalid conditions are saved and labeled rather than excluded silently.
- Exact unordered matching is limited to $P\leq16$; this sweep stops at $P=10$.
- Random particle grouping contains no designed source-level coherence. A negative result does not mathematically disprove H4.5 on meaningful ANewOmni latents.
- No experiment here measures decoder-level molecular validity.

## Saved outputs

Every condition creates a new numbered result directory with:

- `config.json`;
- `reference_smoke.csv` and `generation_smoke.csv`;
- `metrics.csv`, containing four method rows;
- `pass1_reference.npz`, a compact interruption witness;
- `samples.npz`, containing endpoints, fitted lambdas, candidates, and replay trajectories but not full EFS histories;
- `summary.txt`;
- two PNG figures.

The sweep directory adds `progress.json`, per-attempt logs, `particle_sweep.csv`, `summary.json`, `summary.txt`, and `particle_sweep.png`. Existing attempts and results are never overwritten.

## Result

The interrupted first sweep at `results/001_20260720T111809_739219Z_seed42-43-44` contains 16 complete conditions and 64 method rows. Seed 42 is complete for $P=1,\ldots,10$; seed 43 is complete for $P=1,\ldots,6$. The interrupted seed-43, $P=7$ attempt is not counted. This is an unbalanced partial grid, so it is not a finished three-seed comparison.

The partial evidence is negative for EFS replay. Replay beat the same-lambda direct $d_1$ interpolation in only 3 of 64 method rows. Median replay/direct ratios were `1.256` for `shared_d1`, `1.165` for `shared_du`, `1.278` for `per_particle_d1`, and `1.241` for `per_particle_du`; values above one mean replay increased error. Median arrival RMSE was `0.481`, `0.463`, `0.340`, and `0.402`, respectively. Per-particle $d_1$ capacity often reduced absolute arrival error, but it is an evaluation control rather than the formal shared-lambda operator, and its median replay/direct ratio was the worst of the four.

Frame choice is inconclusive rather than a clean win. In paired arrival comparisons, shared $d_u$ won 9 conditions and shared $d_1$ won 6, while per-particle $d_1$ won 10 and per-particle $d_u$ won 4; the three ties are the algebraically identical $P=1$ controls. This interaction says representation capacity and frame transfer are entangled. It does not support the claim that fitting at either frame repairs replay distortion.

The acceptance controls that were reached passed: $P=1$ shared/per-particle differences were exactly zero, no completed EFS condition was invalid, and Experiment 07 shared-$d_u$, $P=10$, seed-42 reproduced to absolute differences between approximately $10^{-12}$ and $3\times10^{-10}$. These checks support implementation consistency, not molecular validity or a positive H4.5 result.

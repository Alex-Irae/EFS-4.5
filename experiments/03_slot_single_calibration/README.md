# Hypothesis 4.5 EFS Calibration and Target-Recovery Harness

This folder tests one narrow mechanism:

> Given several complete sources with corresponding particles, does one barycentric coefficient vector shared across every particle recover a known complete target better than changing the source coefficients from particle to particle?

There are no conformational families, rotations, learned embeddings, or ANewOmni checkpoints in this experiment. One source is a complete object containing $P=8$ corresponding particles in dimension $D=4$.

The written block H4.5 model is the primary scientific model. Each corresponding particle slot has its own EFS field. The pooled single-plane formulation is retained as an explicit control, not treated as mathematically equivalent.

## Hypothesis-to-code matrix and why

| Hypothesis element | Implementation | Why it exists |
| --- | --- | --- |
| One source is a complete particle ensemble | `generate_grouped_sources()` returns `[C,P,D]` | A complete source is not reduced to one point |
| Corresponding-slot H4.5 | `arrange_memory(..., "slot")` returns `[P,C,D]` | Slot $p$ interacts only with slot $p$ from the $C$ memory sources |
| Pooled-field control | `arrange_memory(..., "single")` returns `[C*P,D]` | Measures what changes when all slot identities share one empirical field |
| Shared H4.5 coefficient vector | `build_terminal_candidates()`, method `shared` | The same $\lambda$ is reused for all $P$ corresponding particles |
| Cyclic falsification control | `build_terminal_candidates()`, method `independent` | Preserves coefficient values and entropy while breaking source consistency between particles |
| Exact controlled target | `build_target_trials()` | Gives a known initial-space target for direct recovery measurement |
| Passive generated particles | `layout_replay_field()` | Generated particles query memory fields but never alter them or interact |
| Forward interpolation diagnostic | `passive_forward()` | Separates forward-map nonlinearity from backward inverse error |
| Inverse validity gate | `vertex_replay_errors()` | Memory vertices must replay accurately before interpolation is interpreted |
| Calibration without target leakage | `search.py` | Parameters are frozen using geometry and vertex replay, never target recovery |

## One-line pipeline

`complete sources [C,P,D] -> slot and pooled calibration -> freeze one configuration per layout -> vertex replay -> known targets -> shared/control interpolation -> passive backward replay -> metrics and plots`

## Mathematical and algorithmic formulations

### 1. Complete synthetic sources

For memory source $c$ and corresponding particle $p$:

$$
X_{c,p}=h_c+q_p+\eta_{c,p}.
$$

Here:

- $h_c\in\mathbb{R}^D$ is the source-wide center;
- $q_p\in\mathbb{R}^D$ is the corresponding-particle template;
- $\eta_{c,p}\in\mathbb{R}^D$ is independent Gaussian jitter.

The complete memory has shape `[C,P,D]`.

This construction tests whether all particles of a generated source retain one coherent parent mixture. It does not model molecular conformations.

### 2. Empirical-field layouts

#### Corresponding-slot layout

The written block H4.5 layout is:

$$
X^{\mathrm{slot}}_{p,c,:}=X_{c,p,:}.
$$

Its history has shape `[J+1,P,C,D]`. For slot $p$, the empirical field contains:

$$
\left\{X_{1,p},X_{2,p},\ldots,X_{C,p}\right\}.
$$

The $P$ slot fields are independent EFS calculations. Particles in different slots do not exert EFS forces on one another.

#### Pooled single-plane control

The pooled layout is:

$$
X^{\mathrm{single}}
=
\operatorname{reshape}\!\left(X,[CP,D]\right).
$$

Its history has shape `[J+1,C*P,D]`. All $CP$ particles interact in one field.

Increasing $C$ reduces sampling noise in either layout. It does not make the two layouts equivalent unless all slot distributions are identical.

### 3. EFS potential gradient

The potential gradient is:

$$
\nabla W(z)
=
z-\frac{z}{\left(\lVert z\rVert^2+\epsilon\right)^{s/2+1}}.
$$

`potential_gradient()` evaluates this equation for arrays whose final axis is dimension $D$.

### 4. Fixed forward evolution

For one empirical field containing $N_f$ particles:

$$
x_i^{j+1}
=
x_i^j
-
\frac{\gamma}{N_f-1}
\sum_{a\neq i}
\nabla W\!\left(x_i^j-x_a^j\right).
$$

For the slot layout, $N_f=C$. For the pooled layout, $N_f=CP$.

`forward_field()` evaluates the exact all-pair sum without allocating `[N_f,N_f,D]`. It uses scalar distance matrices and matrix products.

`forward_history()` uses a fixed duration:

$$
k=\left\lceil\frac{\tau}{\gamma}\right\rceil.
$$

Scale changes and apparent convergence are reported, not used as hard stopping requirements.

### 5. Controlled clean and noisy targets

Every trial selects $S=4$ distinct complete sources and one coefficient vector:

$$
\lambda\in\mathbb{R}^S,
\qquad
\lambda_s\geq0,
\qquad
\sum_{s=1}^{S}\lambda_s=1.
$$

The clean target is:

$$
T_{\mathrm{clean},p}
=
\sum_{s=1}^{S}
\lambda_s X_{c_s,p}.
$$

The noisy held-out target is:

$$
T_{\mathrm{noisy}}
=
T_{\mathrm{clean}}+\xi,
$$

with:

$$
\sigma_\xi
=
\sigma_{\mathrm{noise}}
\sqrt{1-\sum_{s=1}^{S}\lambda_s^2}.
$$

Neither target is inserted into EFS memory.

### 6. Shared terminal interpolation

For terminal parent particles $X_{c_s,p}^{J}$:

$$
Y_{\mathrm{shared},p}^{J}
=
\sum_{s=1}^{S}
\lambda_s X_{c_s,p}^{J}.
$$

The same $\lambda$ is used for every particle $p$. This is Hypothesis 4.5.

### 7. Cyclic independent control

For particle $p$, the control cyclically permutes the same coefficient values:

$$
Y_{\mathrm{control},p}^{J}
=
\sum_{s=1}^{S}
\lambda_{\pi_p(s)}
X_{c_s,p}^{J}.
$$

The control preserves the parents, coefficient values, non-negativity, sum, and entropy. It changes only coefficient-to-parent assignment between particles.

If $\lambda$ is uniform, shared and cyclic interpolation are identical. These trials remain saved but are excluded from paired win counts.

### 8. Passive forward oracle

The clean target is transported without entering memory:

$$
T^{j+1}
=
T^j
-
\frac{\gamma}{N_f}
\sum_{i=1}^{N_f}
\nabla W\!\left(T^j-x_i^j\right).
$$

`passive_forward()` uses source frame `history[j]` for the interval from $j$ to $j+1$.

### 9. Terminal commutation gap

The shared terminal interpolation is compared with the passively transported clean target:

$$
\Delta_{\mathrm{commute}}
=
\operatorname{RMSE}
\left(
F(T_{\mathrm{clean}}),
\sum_s\lambda_sF(X_{c_s})
\right).
$$

A large gap means the nonlinear forward map does not preserve barycentric interpolation before backward replay is considered.

### 10. Literal backward replay

At post-forward frame $j$:

$$
F_j(v)
=
\frac{\gamma}{N_f}
\sum_{i=1}^{N_f}
\nabla W\!\left(v-x_i^j\right).
$$

Each proximal iteration is:

$$
v
\leftarrow
v
-
\beta\left(v-y^j-F_j(v)\right).
$$

After $T$ iterations, $v$ becomes the candidate at the preceding reverse frame.

`backward_replay()` batches passive candidates for CPU efficiency. Candidates do not exert forces on each other.

### 11. Radial diagnostics

For center $c$ and particle $i$:

$$
r_i=\lVert x_i-c\rVert.
$$

For a uniform $D$-ball of radius $R$:

$$
P(r\leq a)
=
\left(\frac{a}{R}\right)^D,
$$

$$
E[r]
=
\frac{DR}{D+1},
\qquad
E[r^2]
=
\frac{DR^2}{D+2}.
$$

The fitted reference radius is:

$$
\widehat{R}
=
\sqrt{
\frac{D+2}{D}
\operatorname{mean}(r^2)
}.
$$

The expected radial coefficient of variation is:

$$
\mathrm{CV}_{\mathrm{ball}}
=
\frac{1}{\sqrt{D(D+2)}}.
$$

`radial_statistics()` reports:

- `radial_cdf_gap`;
- observed and theoretical radial CV;
- normalized radius quantiles;
- ten radial zones;
- overflow beyond $\widehat{R}$.

For radial zones:

$$
u_i
=
\left(\frac{r_i}{\widehat{R}}\right)^D.
$$

Under the fitted uniform-ball reference, $u_i$ is uniform on $[0,1]$, so each of ten zones should contain approximately $10\%$ of particles.

`radial_cdf_gap` has no p-value and is not a hard gate.

### 12. Angular diagnostics

For non-central particles:

$$
u_i
=
\frac{x_i-c}{\lVert x_i-c\rVert}.
$$

For ideal uniform directions:

$$
E[u_i]=0,
\qquad
E[u_i u_i^\top]
=
\frac{I}{D}.
$$

`angular_statistics()` reports mean-direction resultant, second-moment error, pairwise dot products, and covariance eigenvalue ratio.

Only covariance collapse is a hard gate. Other angular statistics are descriptive.

### 13. Vertex replay

Actual terminal memory particles are replayed:

$$
x_i^J
\stackrel{\mathrm{backward}}{\longrightarrow}
\widehat{x}_i^0.
$$

For particle $p$ in source $v$:

$$
\operatorname{RMSE}_{v,p}
=
\sqrt{
\frac{1}{D}
\sum_{d=1}^{D}
\left(
\widehat{x}_{v,p,d}^{0}
-
x_{v,p,d}^{0}
\right)^2
}.
$$

Four complete vertex sources produce $4P=32$ particle errors.

The report contains particle median, p90, p95, maximum, complete-source RMSE, relative errors, and fixed-point residual.

## Files and function dataflow

### `data.py`

- `generate_grouped_sources()` creates sources `[C,P,D]`, centers `[C,D]`, and template `[P,D]`.
- `arrange_memory()` creates slot `[P,C,D]` or pooled `[C*P,D]` memory.
- `restore_sources()` restores terminal memory to `[C,P,D]`.
- `parse_shared_lambda()` validates an optional operator vector.
- `build_target_trials()` creates parents, lambdas, clean/noisy targets, and cyclic weights.
- `build_terminal_candidates()` returns `[R,2,P,D]` shared/control candidates.

### `efs.py`

- `potential_gradient()` implements $\nabla W(z)$.
- `forward_field()` evaluates pooled or slot fields.
- `forward_history()` runs fixed forward evolution and stores full histories or calibration checkpoints.
- `replay_field()` evaluates passive queries against one field.
- `layout_replay_field()` routes `[G,P,D]` queries through the correct layout.
- `passive_forward()` transports targets without changing memory.
- `backward_replay()` stores trajectories, residuals, total-position vectors, and reverse-frame displacement.
- `mean_pairwise_distance()`, `rms_radius()`, and `ensemble_mean_distance()` provide scale diagnostics.
- `self_check()` compares optimized calculations with literal sums.

### `evaluation.py`

- `radial_statistics()` computes CDF gap, CV, quantiles, zones, and overflow.
- `angular_statistics()` computes resultant, second moment, and covariance ratio.
- `layout_cloud_statistics()` evaluates every empirical field separately.
- `smoke_gate()` applies hard numerical and geometric gates.
- `vertex_replay_errors()` computes particle and source replay errors.
- `target_recovery_metrics()` computes target, centroid, shape, particle, commutation, residual, coherence, and novelty metrics.
- `classify_result()` assigns descriptive result labels.
- `write_smoke_csv()` and `write_metrics_csv()` save tables.

### `plotting.py`

- `plot_forward_smoke()` shows scale flow, radial CDF, radial zones, and angular dots.
- `plot_vertex_replay()` shows particle errors and their empirical CDF.
- `plot_target_recovery()` shows paired recovery and jitter floor.
- `plot_target_flow_pca()` shows memory, interpolation, replay, and outputs in one PCA basis.
- `plot_replay_movement()` shows internal spread and total-position movement.
- `plot_grid_summary()` regenerates calibration plots from CSV.
- `plot_all()` regenerates one run from `samples.npz`.

PCA flattening is used only for visualization. It does not change EFS fields.

### `run.py`

`run()` performs:

1. data generation;
2. layout arrangement;
3. fixed forward evolution;
4. hard forward gates;
5. vertex replay;
6. vertex gate;
7. controlled target creation;
8. terminal interpolation;
9. passive target transport;
10. shared/control replay;
11. metric calculation;
12. saving and plotting.

### `search.py`

`run_search()` performs:

1. potential screen;
2. Euler-step screen;
3. layout and density refinement;
4. replay screen;
5. replay confirmation with fallback;
6. layout selection;
7. informational joint-vector smoke;
8. frozen H4.5 evaluation.

`ProcessPoolExecutor` runs at most four independent configurations concurrently. Workers do not create nested pools.

The parent process writes and flushes CSV rows. `--resume` skips completed keys.

## Environment

Python 3.10 is preferred.

Environment creation and package installation are left to the user.

From this experiment directory:

```powershell
python -m pip install -r ..\..\requirements.txt
```

Dependencies are NumPy and Matplotlib.

The implementation is CPU-only, uses `float64`, and has no Torch, CUDA, ROCm, SciPy, pandas, YAML, or unit-test framework.

## Run commands

Algebra self-check:

```powershell
python efs.py
```

Full calibration and frozen evaluation:

```powershell
python search.py --workers 4 --output-root results
```

Resume:

```powershell
python search.py --resume results\<search-directory> --workers 4
```

Small execution-path check:

```powershell
python search.py --quick --workers 4 --output-root results
```

The quick protocol verifies execution and array flow. It is not scientific evidence.

One explicit configuration:

```powershell
python run.py --layout slot --epsilon 0.1 --exponent-s 2.0 --gamma 0.0005 --forward-steps 5000 --beta 0.05 --proximal-steps 100 --output-root results
```

After calibration, `best_commands.txt` contains the exact frozen commands.

Regenerate one run:

```powershell
python plotting.py results\<run-directory>
```

Regenerate a search summary:

```powershell
python plotting.py results\<search-directory>
```

## Parameters

### Fixed structure

| Parameter | Value |
| --- | ---: |
| Particle dimension $D$ | `4` |
| Particles per source $P$ | `8` |
| Interpolation parents $S$ | `4` |
| Full validation sources $C$ | `256` |
| Final trials | `16` per seed |
| Final seeds | `101`, `103`, `107` |
| Arithmetic | `float64` |
| Attraction coefficient | `1` |

### Calibration grid

| Parameter | Values |
| --- | --- |
| $\epsilon$ | `0.01`, `0.03`, `0.1`, `0.3` |
| $s$ | `1.0`, `1.5`, `2.0`, `2.5`, `3.0` |
| $\gamma$ | `0.00025`, `0.0005`, `0.001`, `0.002` |
| $\tau=\gamma k$ | `1.5`, `2.0`, `2.5`, `3.0`, `3.5`, `4.0` |
| $\beta$ | `0.025`, `0.05`, `0.1` |
| Proximal iterations $T$ | `25`, `50`, `100`, `200` |

`run.py` has readable baseline defaults. They are not declared calibrated. `search.py` writes selected values to `selection.json` and `best_commands.txt`.

## Calibration protocol

### Stage 1: potential screen

- layout `slot`;
- `64` sources;
- seeds `42` and `50`;
- all $20$ pairs $(\epsilon,s)$;
- $\gamma=0.0005$;
- every declared $\tau$ checkpoint;
- rank by worst-seed, worst-slot `radial_cdf_gap`;
- tie by angular second-moment error, then earlier $\tau$;
- retain four $(\epsilon,s)$ pairs.

### Stage 2: Euler-step screen

- every retained potential pair;
- every $\gamma$;
- common transport range through $\tau=4$;
- every $\tau$ checkpoint;
- retain four complete $(\epsilon,s,\gamma,\tau)$ configurations.

### Stage 3: layout and density refinement

- `128` sources;
- layouts `slot` and `single`;
- seeds `42` and `50`;
- retain two configurations per layout;
- validate with `256` sources on seeds `61`, `73`, and `89`.

### Stage 4: replay screen

For each finalist:

- seed `61`;
- one complete source;
- all eight particles;
- all $12$ combinations $(\beta,T)$.

The replay gate is:

$$
\max_{v,p}
\frac{
\operatorname{RMSE}_{v,p}
}{
\sigma_{\mathrm{memory}}
}
\leq0.10.
$$

Choose the smallest $T$ whose p95 is within $10\%$ of the best p95. Tie by residual and runtime.

### Stage 5: replay confirmation

Confirm four complete sources on seeds `73` and `89`. If a candidate fails, try the next ranked replay configuration.

### Stage 6: layout choice

Compare worst validation-seed particle p95.

If layouts differ by at most $10\%$, choose the faster layout as primary. Otherwise choose the lower p95. The other layout remains the control.

### Stage 7: frozen target evaluation

After calibration:

- seeds `101`, `103`, and `107`;
- `256` sources;
- `16` randomized targets per seed;
- identical parents and coefficient values for shared and cyclic methods.

Target recovery never influences calibration.

## Informational joint-vector smoke

`search.py` also runs an excluded check:

- reshape `[64,8,4]` into `[64,32]`;
- $s=30$;
- $\epsilon=1$;
- $\gamma=10^{-4}$;
- `500` forward frames;
- $\beta=0.05$;
- `25` proximal iterations;
- one vertex replay.

It reports shapes, finiteness, scale, and vertex error. It cannot select parameters or layouts.

## Hard gates

A configuration is invalid only if:

1. forward or replay values become nonfinite;
2. terminal-to-initial RMS-radius ratio leaves $[0.05,5.0]$;
3. terminal covariance eigenvalue ratio is below $10^{-3}$;
4. maximum particle vertex relative RMSE exceeds $0.10$.

These are descriptive, not hard gates:

- convergence rate;
- `radial_cdf_gap`;
- radial CV;
- normalized radial quantiles;
- radial-zone occupancy;
- radial overflow;
- angular resultant;
- angular second-moment error.

A shell-like finite endpoint can remain valid when its vertices replay accurately.

## Target-recovery metrics

Primary metrics:

- clean-target RMSE;
- clean-target relative RMSE;
- noisy-target RMSE;
- jitter-only floor;
- centroid RMSE;
- centered internal-shape RMSE;
- per-particle RMSE;
- terminal commutation gap;
- fixed-point residual;
- paired shared/control wins.

Secondary diagnostics:

- template-coherence error;
- nearest-memory distance;
- replay movement;
- scale change.

## Interpretation notes

Read results in this order:

1. Hard forward gates establish finite scale and dimensional support.
2. Vertex replay establishes inverse quality.
3. Commutation gap measures forward interpolation nonlinearity.
4. Clean-target RMSE measures final recovery.
5. Paired trials isolate shared $\lambda$ from the cyclic control.
6. Noisy-target error and jitter floor provide a stochastic scale.
7. Centroid and shape errors distinguish translation from deformation.
8. Novelty and coherence remain secondary.

Classification is descriptive:

- `invalid`: a hard gate fails;
- `negative`: shared wins no more than half the eligible trials, or its median is not lower;
- `promising`: shared wins at least $75\%$, has lower median error, and median clean relative RMSE is at most $0.10$;
- `mixed`: valid but between negative and promising.

No formal significance claim is made.

## Assumptions and edge cases

- Particle index $p$ has the same meaning in every source.
- Without correspondence, particle-wise H4.5 interpolation is undefined.
- Lambda entries are non-negative and sum to one.
- Extrapolation is not tested.
- Targets use distinct parents and never enter memory.
- Slot candidates query only matching slot fields.
- Pooled candidates query one common field.
- Batched candidates never exert forces on one another.
- Uniform $\lambda$ makes shared and cyclic interpolation identical.
- A fitted uniform ball is a reference, not the only valid endpoint.
- Exact all-pair evolution is quadratic within each field.
- The joint-vector check is informational only.
- Synthetic success does not establish biochemical validity.

## Saved outputs

### One `run.py` result

Every timestamped run contains:

- `config.json`;
- `smoke.csv`;
- `metrics.csv`;
- `samples.npz`;
- `summary.txt`;
- `forward_smoke.png`;
- `vertex_replay.png`;
- `target_recovery.png` when targets run;
- `target_flow_pca.png` when targets run;
- `replay_movement.png` when targets run.

Important shapes:

- sources: `[C,P,D]`;
- slot history: `[J+1,P,C,D]`;
- pooled history: `[J+1,C*P,D]`;
- terminal candidates: `[R,2,P,D]`;
- target trajectory: `[J+1,R,2,P,D]`;
- vertex particle errors: `[V,P]`.

### One `search.py` result

Every search contains:

- `search_config.json`;
- `forward_grid.csv`;
- `replay_grid.csv`;
- `selection.json`;
- `best_commands.txt`;
- `grid_summary.png`;
- `joint_smoke.txt`;
- `final_runs.csv`;
- `final_metrics.csv`;
- `final_summary.txt`;
- `final_runs\`.

`--resume` skips completed row keys without duplicating them.

## Existing seed-50 evidence

The historical pooled run has:

- `radial_cdf_gap`: `0.2026196`;
- revised hard forward gate: passed;
- particle median relative RMSE: `0.0159371`;
- particle p90 relative RMSE: `0.0237386`;
- particle p95 relative RMSE: `0.0239004`;
- particle maximum relative RMSE: `0.0244961`;
- maximum replay residual: $2.63\times10^{-6}$.

This is why `radial_cdf_gap` is descriptive rather than a hard rejection threshold.

## Legacy material

The previous prototypes remain unchanged:

- `../01_separate_fields/`;
- `../02_single_plane_witness/`.

The active implementation is:

- `data.py`;
- `efs.py`;
- `evaluation.py`;
- `plotting.py`;
- `run.py`;
- `search.py`;
- `README.md`.

The authoritative project hypothesis is [`../../docs/h45_project.md`](../../docs/h45_project.md).

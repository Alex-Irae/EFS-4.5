# Single-Plane H4.5 Two-Pass Reconstruction

This folder tests one narrow question:

> If one shared terminal-space lambda places a complete candidate near a known source, does target-removed backward EFS return it to that source's original neighborhood?

There is one and only one EFS plane. Source ownership is metadata declared before EFS. It never changes particle forces.

The test is intentionally limited. It can measure particle-distribution behavior and recovery of a held-out synthetic source. It cannot prove that an ANewOmni decoder would produce a coherent molecule.

## One-line pipeline

`d1 target and K neighbors -> pass 1 with target -> fit lambda at du -> reset -> remove target -> pass 2 -> same parents and lambda -> backward replay -> target error`

Calibration pipeline:

`chaotic particle planes -> forward geometry grid -> vertex replay grid -> best_config.json -> later run.py calls`

## Hypothesis-to-code matrix

| Requirement | Code | Why |
| --- | --- | --- |
| One particle population | `data.generate_chaotic_sources()` | Every particle is drawn independently from the same global distribution. |
| Random source ownership | final shuffle and reshape in `generate_chaotic_sources()` | No center, family, or shape is created for a source. |
| Exactly one EFS plane | `efs.forward_history()` | All `C*P` memory particles interact together. |
| Unordered particles | `data.best_permutation()` | A source comparison does not assume permanent particle slots. |
| Four meaningful complete sources | `data.select_source_neighbors()` | Target and neighbors are selected only in the original data plane `d1`. |
| One positive shared lambda | `data.fit_shared_source_lambda()` | Pass-1 terminal coordinates jointly determine one vector for all `P` particles. |
| Counterfactual target removal | `run.run_two_pass()` | Pass 1 contains the target; pass 2 independently reruns from `d1` without it. |
| Literal EFS generation | `data.interpolate_sources()` and `efs.backward_replay()` | The terminal-fitted candidate is replayed without extra source forces. |
| Field-change measurement | `evaluation.marginal_histogram_kl()` | Directional marginal KL describes the terminal change caused by removing one source. |
| Physically readable error | `evaluation.relative_source_distance()` | Total Euclidean reconstruction displacement is divided by target extent. |
| Automatic numerical calibration | `search.py` | Forward geometry and vertex replay choose `best_config.json` without using target recovery. |

## Objects and shapes

- `D`: dimensions per particle, default `4`.
- `P`: particles per source, default `10`.
- `C`: pass-2 memory sources, default `512`.
- one target source per run.
- `K`: nearest parent sources, default `4`.
- `N=C*P`: particles in EFS, default `5120`.

Important arrays:

- pass-1 sources: `[C+1,P,D]`;
- pass-2 sources: `[C,P,D]`;
- pass-1 history: `[J+1,(C+1)P,D]`;
- pass-2 history: `[J+1,CP,D]`;
- saved parents: `[K,P,D]`;
- shared lambda: `[K]`;
- generated candidate: `[1,1,P,D]`.

The singleton dimension `1` is the candidate-method axis. Only `shared_lambda` exists. Keeping that axis lets all targets use the existing batched replay path.

## Mathematical and algorithmic formulation

### 1. Chaotic independent particle distribution

Any reproducible random generator necessarily defines a distribution. The code therefore cannot be literally distribution-free. It avoids imposing one simple final shape such as a Gaussian, uniform ball, shell, Dirichlet law, or exponential law.

For particle $n$, a global mixture component $q$ is sampled and:

$$
x_n
=
\frac{\text{data scale}}{2}
\left[
c_q
+a_n
\left(
z_nA_q
+0.45\sin(z_nB+\phi_q)
+0.08z_n^3
\right)
\right].
$$

The component centers, dense linear maps, nonlinear phases, amplitudes, and latent draws produce a multimodal, warped, variable-scale cloud. They define one global particle law.

No source parameter appears in this equation. After all `(C+1)*P` particles are drawn, they are randomly permuted and reshaped:

$$
[(C+1)P,D]
\longrightarrow
[C+1,P,D].
$$

Consequences:

- particles within one source are independent draws;
- a source may contain particles from unrelated regions;
- source identity has no positional signature;
- changing a source ID does not change EFS.

### 2. Two independent forward passes

The runner creates `C+1` sources. One source is the known target `T`.

Pass 1 contains every source:

$$
N_1=(C+1)P.
$$

Pass 2 resets the original coordinates, removes only `T`, and recomputes EFS from scratch:

$$
N_2=CP.
$$

Pass 1 supplies the known target endpoint and lambda. Pass 2 approximates real generation, where the target is absent from the empirical field.

### 3. EFS forward field

For a difference vector $z$:

$$
\nabla W(z)
=
z-
\frac{z}{(\lVert z\rVert^2+\epsilon)^{s/2+1}}.
$$

For memory particle $i$:

$$
x_i^{(j+1)}
=
x_i^{(j)}
-
\frac{\gamma}{N-1}
\sum_{a\ne i}
\nabla W(x_i^{(j)}-x_a^{(j)}).
$$

`efs.forward_field()` uses scalar `[N,N]` distance and weight matrices. It does not allocate `[N,N,D]`.

### 4. Neighbor selection only at d1

For target $T$ and library source $S$, particle correspondence is:

$$
\pi^*
=
\arg\min_\pi
\sum_{p=1}^{P}
\left\|
\frac{T_p-S_{\pi(p)}}{\sigma}
\right\|^2,
$$

where $\sigma\in\mathbb{R}^D$ is the memory coordinate scale.

The resulting matched source distance is the standardized RMSE over all `P*D` values. The `K=4` smallest complete-source distances are selected at `d1`. Their source IDs and particle permutations are saved. No new neighbor search occurs after either forward pass.

The exact dependency-free assignment uses $O(P2^P)$ work. It is practical for the default `P=10` and capped at `P=16`.

### 5. Pass-1 terminal lambda

Pass 1 transports the target and its saved `d1` parents together. At its terminal plane, the same parent particles are flattened from `[K,P,D]` to `[K,P*D]`. One lambda is fitted entirely in pass-1 terminal coordinates:

$$
\lambda^*
=
\arg\min_{\lambda}
\left\|
\operatorname{vec}(T^{(J)})
-
\sum_{k=1}^{K}
\lambda_k\operatorname{vec}(S_k^{(J)})
\right\|_2^2,
$$

subject to:

$$
\lambda_k\ge0,
\qquad
\sum_{k=1}^{K}\lambda_k=1.
$$

Projected gradient descent and exact simplex projection enforce both constraints. There are no per-particle lambdas and no averaging step.

This is target-assisted evaluation. The target supplies a known valid endpoint and therefore a measurable answer. In real generation, an operator would choose lambda directly instead of fitting it to a hidden target.

### 6. Pass-2 terminal candidate

After removing `T`, pass 2 changes the field slightly. The same saved parent IDs, particle correspondence, and pass-1 lambda are applied to their new pass-2 terminal positions:

$$
Y_{p,2}^{(J)}
=
\sum_{k=1}^{K}\lambda_k S_{k,p,2}^{(J)}.
$$

This candidate is passive. It never enters the memory field or changes another particle.

For comparison only, the same saved parents and lambda are also applied at frame zero:

$$
G_{p,\mathrm{direct}}
=
\sum_{k=1}^{K}\lambda_k S_{k,p}^{(0)}.
$$

### 7. Pass-1/pass-2 field shift

The retained particles have identical initial coordinates in both passes. Their terminal positions differ only because pass 1 included `T` and pass 2 did not.

For each coordinate, shared histograms estimate:

$$
D_{KL}(p\|q)=\sum_b p_b\log\frac{p_b}{q_b}.
$$

Both `KL(pass 1 || pass 2)` and `KL(pass 2 || pass 1)` are reported. This is a descriptive marginal histogram comparison, not a multivariate density estimate or formal test.

### 8. Literal backward replay

At saved post-forward frame $j$:

$$
F_j(v)
=
\frac{\gamma}{N}
\sum_i\nabla W(v-x_i^{(j)}),
$$

$$
v_{q+1}
=
v_q
-
\beta\left(v_q-y^{(j)}-F_j(v_q)\right).
$$

Generated particles are passive queries. They do not push memory particles and do not push one another.

## Function dataflow

### `data.py`

- `generate_chaotic_sources()` draws independent chaotic particles and randomly groups them.
- `generate_uniform_sources()` supplies the uniform-box `P=1,D=2` control.
- `hold_out_sources()` removes complete targets before EFS.
- `coordinate_scale()` measures memory coordinate scales.
- `best_permutation()` solves exact one-to-one particle assignment.
- `align_source()` reorders one unordered source to another temporarily.
- `source_distance()` measures complete-source proximity without fixed particle order.
- `select_source_neighbors()` selects one target's K parents at `d1` and saves their particle correspondence.
- `project_simplex()` enforces nonnegative weights summing to one.
- `fit_simplex_weights()` fits the shared lambda.
- `fit_shared_source_lambda()` fits one lambda jointly over all `P*D` terminal values.
- `build_reconstruction_trials()` performs top-K retrieval, matching, and one full-source fit.
- `interpolate_sources()` applies saved source IDs, matching, and lambda to another frame.

### `efs.py`

- `potential_gradient()` implements $\nabla W$.
- `forward_field()` evaluates the one pooled field.
- `forward_history()` evolves and saves memory particles.
- `replay_field()` evaluates passive candidates against one frame.
- `passive_forward()` transports held-out witnesses.
- `backward_replay()` reconstructs candidates and records movement.
- `self_check()` compares vectorized equations with literal calculations.

### `evaluation.py`

- `radial_statistics()` describes terminal radial conformity.
- `angular_statistics()` describes directional balance.
- `smoke_gate()` blocks numerical explosion or dimensional collapse.
- `marginal_histogram_kl()` measures descriptive pass-1/pass-2 coordinate shifts.
- `vertex_replay_metrics()` checks inversion at actual memory endpoints.
- `relative_source_distance()` computes the main target metric.
- `relation_signature()` compares unordered within-source pair distances.
- `reconstruction_metrics()` compares direct and replayed candidates.
- `result_summary()` reports counts, medians, and ratios.

### `run.py`

Canonical path:

`d1 target/K selection -> pass 1 with target -> du lambda -> reset/remove T -> pass 2 -> same K/lambda -> replay -> compare with T`

`--protocol single-pass-toy` preserves the earlier point and slow-convergence experiments unchanged.

### `search.py`

Search evaluates only forward geometry and exact vertex replay. Target errors never select hyperparameters. A completed full search atomically installs `best_config.json` beside `run.py`; later runs load it automatically unless `--ignore-config` is used.

## Files

- `data.py`: particle generation, random grouping, unordered matching, and lambda fitting.
- `efs.py`: one-plane EFS forward, passive transport, and replay.
- `evaluation.py`: gates and reconstruction metrics.
- `plotting.py`: all saved-run figures.
- `run.py`: canonical two-pass experiment plus preserved single-pass toy path.
- `search.py`: resumable four-worker calibration.
- `best_config.json`: active search result loaded automatically.
- `configs/point_uniform_2d.json`: single-particle uniform control parameters.
- `configs/strong_contraction_2d.json`: long, strongly contracting 2D parameters.
- `req.txt`: NumPy and Matplotlib requirements.
- `results/`: append-only run and search directories.
- `legacy/`: preserved previous implementations and results.

## Environment

Preferred Python version: `3.10`.

Create the environment yourself, then install:

```powershell
pip install -r req.txt
```

Dependencies are only NumPy and Matplotlib. There is no Torch, SciPy, pandas, YAML, CUDA, ROCm, or test framework.

## Commands

Run from `E:\Desktop\AnewOmni\4.5hypothesis`.

Calibrate EFS with four worker processes:

```powershell
python search.py --workers 4 --output-root results
```

Resume an interrupted search:

```powershell
python search.py --resume results\<search-directory> --workers 4
```

Small path check that never replaces the active config:

```powershell
python search.py --quick --workers 4 --output-root results
```

Run the canonical two-pass experiment using `best_config.json`:

```powershell
$workers = [Math]::Min(32, [Environment]::ProcessorCount)
python run.py --protocol two-pass --heldout-sources 1 --workers $workers --output-root results\two_pass
```

Two-dimensional uniform point control, using every reported logical CPU up to the runner's 32-worker cap:

```powershell
$workers = [Math]::Min(32, [Environment]::ProcessorCount)
python run.py --protocol single-pass-toy --config configs\point_uniform_2d.json --distribution uniform --dimension 2 --particles-per-source 1 --source-count 4096 --heldout-sources 64 --parents 4 --vertex-sources 16 --workers $workers --log-every 50 --seed 501 --output-root results\point_uniform_2d
```

Two-dimensional strong-contraction source test:

```powershell
$workers = [Math]::Min(32, [Environment]::ProcessorCount)
python run.py --protocol single-pass-toy --config configs\strong_contraction_2d.json --distribution chaotic --dimension 2 --particles-per-source 8 --source-count 512 --heldout-sources 16 --parents 4 --vertex-sources 16 --workers $workers --log-every 50 --seed 502 --output-root results\strong_contraction_2d
```

`--workers` divides passive vertex and target replay over threads that share the saved history. It does not parallelize the time-dependent forward recurrence. NumPy may separately use its own BLAS threads for the forward field. If the machine becomes unresponsive or nested threading reduces speed, lower `--workers` first.

Small path check:

```powershell
python run.py --quick --output-root results
```

Explicitly override scale while retaining other saved values:

```powershell
python run.py --source-count 768 --particles-per-source 10 --output-root results
```

Ignore the saved calibration:

```powershell
python run.py --ignore-config --output-root results
```

Regenerate figures without rerunning EFS:

```powershell
python plotting.py results\<run-directory>
```

Algebra checks are available but are not required by `run.py` because the runner calls them itself:

```powershell
python efs.py
python data.py
python evaluation.py
```

## Parameters

| CLI argument | Default | Meaning |
| --- | ---: | --- |
| `--protocol` | `two-pass` | canonical counterfactual or preserved `single-pass-toy` |
| `--seed` | `42` | deterministic NumPy seed |
| `--source-count` | `512` | pass-2 memory sources; pass 1 adds one target |
| `--heldout-sources` | `1` | must equal one for the canonical two-pass protocol |
| `--distribution` | `chaotic` | `chaotic` mixture or uniform-box point control |
| `--particles-per-source` | `10` | independent particles randomly grouped per source |
| `--dimension` | `4` | values per particle |
| `--parents` | `4` | nearest complete sources used in the fit |
| `--vertex-sources` | `4` | exact memory sources used for inverse checking |
| `--data-scale` | `1.0` | global coordinate scale |
| `--epsilon` | config, else `0.03` | EFS smoothing |
| `--exponent-s` | config, else `3.0` | inverse-power exponent |
| `--gamma` | config, else `0.001` | Euler step |
| `--forward-steps` | config, else `2000` | saved forward frames |
| `--beta` | config, else `0.10` | replay update step |
| `--proximal-steps` | config, else `100` | fixed-point updates per frame |
| `--lambda-iterations` | `1000` | maximum projected-gradient iterations |
| `--log-every` | `10` | full forward or backward frames between logs |
| `--workers` | `1` | concurrent passive replay batches sharing one history |

The point control uses `epsilon=0.03`, `s=1`, `gamma=0.002`, 3,000 frames, and 100 proximal updates. Its transport time is $\tau=6$.

The strong-contraction test uses `epsilon=0.3`, `s=1`, `gamma=0.002`, 6,000 frames, and 200 proximal updates. Its transport time is $\tau=12$, six times the completed run's $\tau=2$. The larger epsilon weakens short-range repulsion and should contract more strongly. This is deliberately aggressive, so the normal finite, radius, covariance, and vertex gates still run before target interpolation.

More proximal updates improve the numerical fixed-point solve. They do not make EFS optimize target error. More forward frames extend the transformation and therefore add the same number of backward frames, but once the field reaches equilibrium, extra frames cannot force indefinite contraction.

An earlier small prototype used 1,920 EFS particles. The current default uses 5,120, and each new 2D command uses 4,096. Since the exact forward field is $O(N^2)$, more allocated RAM is not itself a fidelity metric. Increase density only when runtime remains useful.

Increasing `P` has two separate effects:

- the fitted source vector grows from `P*D`, making one shared lambda stiffer;
- exact unordered matching grows as $O(P2^P)$.

For that reason `P` is capped at 16.

# RESULT INTERPRETATION

Read outputs in the following order.

## 1. Forward hard gates

| Metric | Input | Output | Preferred | Hard failure |
| --- | --- | --- | ---: | --- |
| finite | all forward arrays | boolean | true | false |
| RMS-radius ratio | initial and terminal planes | terminal radius / initial radius | descriptive | outside `[0.05,5]` |
| covariance ratio | terminal covariance eigenvalues | minimum / maximum | near `1` | below `1e-3` |
| center drift | initial and terminal centers | Euclidean movement | near machine precision | descriptive |

Both passes must satisfy these gates. Passing means neither field exploded, collapsed to a point, or lost almost all dimensional support. It does not prove target quality.

## 2. Forward uniform-ball diagnostics

| Metric | Reference | Meaning |
| --- | ---: | --- |
| radial CDF gap | `0` | maximum gap from a fitted uniform-ball radial CDF |
| radial CV in `D=4` | `0.204124` | expected radial spread |
| ten radial zones | `0.10` each | equal-probability radial occupancy |
| angular resultant | `0` | no preferred direction |
| angular moment error | `0` | directional second moment equals $I/D$ |

Practical descriptive guides:

- radial CDF gap below `0.05`: close;
- `0.05` to `0.10`: visibly imperfect;
- above `0.10`: clear mismatch.

These values rank search configurations and describe both passes, but do not invalidate an otherwise finite and invertible field.

## 3. Pass-1/pass-2 KL

Input: the same retained particle identities at the two terminal endpoints.

Output: one directional histogram KL per coordinate and its mean across coordinates.

| Value | Interpretation |
| ---: | --- |
| `0` | identical smoothed coordinate histograms |
| near `0` | removing one source barely changed the terminal marginals |
| larger positive value | visible pass-dependent field shift |

There is no universal hard threshold because the number depends on sample count, bin count, smoothing, dimension, and coordinate system. The code fixes 48 shared bins and a `0.5` pseudocount so runs are internally comparable. KL is directional, so both orders are reported.

The matched retained-particle displacement is saved beside KL. It catches identity-level movement that marginal histograms can hide.

## 4. Vertex replay

Input: actual terminal memory particles.

Output: their reconstructed initial positions.

For particle $p$:

$$
e_{v,p}
=
\frac{
\sqrt{D^{-1}\sum_d(\hat{x}_{p,d}^{(0)}-x_{p,d}^{(0)})^2}
}{\sigma_{\mathrm{memory}}}.
$$

| Relative error | Interpretation |
| ---: | --- |
| `0` | exact replay |
| below `0.001` | exceptionally accurate |
| `0.001` to `0.01` | good |
| `0.01` to `0.10` | increasingly questionable |
| above `0.10` | hard failure |

Vertex replay tests the inverse only at known EFS particles. It does not test lambda, interpolation, novelty, or decoded validity.

## 5. Shared-lambda fit

`lambda_fit_rmse` measures the standardized pass-1 terminal residual after flattening the known target and its four saved `d1` parents to `P*D`.

- input: target `[P,D]` and parents `[K,P,D]`;
- output: one nonnegative lambda `[K]` and one residual;
- best value: `0`;
- hard threshold: none.

A large value means the saved `d1` parents cannot express the pass-1 target endpoint with one convex shared mixture.

## 6. Main relative source distance

For candidate $G$ and held-out target $T$:

$$
d_{\mathrm{relative}}(G,T)
=
\frac{
\min_\pi\lVert G_\pi-T\rVert_F
}{
\lVert T-\bar{T}\rVert_F
}.
$$

Input: two unordered `[P,D]` sources.

Output: total best-matched Euclidean displacement relative to the target's own extent.

For `P=1`, target extent is zero. The denominator therefore becomes the memory-plane RMS coordinate scale:

$$
d_{\mathrm{point,relative}}
=
\frac{\lVert g-t\rVert_2}{\lVert\sigma_{\mathrm{memory}}\rVert_2}.
$$

The point control also saves the unnormalized Euclidean values `direct_point_distance`, `generated_point_distance`, and `nearest_parent_point_distance`. Zero is optimal. Unlike the relative value, these remain in raw coordinate units.

| Value | Meaning |
| ---: | --- |
| `0` | exact target recovery |
| below `0.10` | error below 10% of target extent |
| `0.10` to `0.25` | close but visible error |
| `0.25` to `0.50` | large error |
| `1` | displacement equals the target's full extent |
| above `1` | very poor recovery |

These bands are interpretation guides, not statistical or biochemical validity thresholds.

Two versions are saved:

- `direct_relative_distance`: error before EFS;
- `generated_relative_distance`: error after terminal interpolation and replay.

The main comparison is:

$$
\rho_{\mathrm{EFS}}
=
\frac{d_{\mathrm{relative}}(G_{\mathrm{EFS}},T)}
{d_{\mathrm{relative}}(G_{\mathrm{direct}},T)}.
$$

- below `1`: EFS improved the reconstruction;
- equal to `1`: EFS added nothing;
- above `1`: EFS made it worse.

The canonical run contains one target and therefore makes no cross-target consistency claim. The preserved single-pass toy retains its older multi-target summary labels.

For the canonical run, the more direct diagnostic is:

$$
A
=
\frac{\text{standardized arrival error at }d_1}
{\max(\text{standardized pass-2 terminal start error},10^{-12})}.
$$

- near or below `1`: the terminal mismatch did not grow;
- moderately above `1`: backward replay amplified the field-change error;
- very large: a small terminal mismatch became a poor arrival;
- a large denominator means the pass-2 parents already moved too far for replay to be isolated.

## 7. Secondary reconstruction metrics

| Metric | Best | Meaning |
| --- | ---: | --- |
| set RMSE | `0` | permutation-invariant coordinate-standardized error |
| nearest-parent relative distance | `0` | retrieval-only baseline |
| EFS/parent ratio | below `1` | replay beats returning the nearest source |
| internal-relation RMSE | `0` | error in sorted within-source pair distances |
| terminal gap | `0` | pass-2 candidate agrees with the known pass-1 target endpoint |
| replay residual | near `0`, usually below `1e-5` | fixed-point equation was solved accurately |
| memory-distance ratio | around `1` | generated locality resembles held-out target locality |

A tiny replay residual cannot rescue poor target recovery. It only says the backward numerical equation was solved.

## Figure interpretation

### `two_pass_target.png`

1. The real target and its saved nearest sources at `d1`.
2. Their pass-1 endpoints and the lambda fit made while the target exists.
3. The same parents and lambda after pass 2 removes the target.
4. Every generated particle's backward path and arrival beside the real target.

Short pass-1 fit lines but long pass-2 lines mean target removal changed the relevant local field. A close pass-2 seed followed by a distant arrival means backward replay amplified a small terminal error.

### `two_pass_field_shift.png`

- left: identical retained particle IDs at the two terminal endpoints;
- middle: directional marginal KL for every coordinate;
- right: the two histograms for the coordinate with the largest average directional KL.

The canonical protocol intentionally creates only these two figures. The earlier single-pass toy protocol retains its existing smoke, vertex, reconstruction, lambda, flow, point, and movement figures. `hyperparameter_selection.png` remains a search output.

## Interpretation notes

1. Both forward gates answer whether each independently recomputed EFS field is numerically usable.
2. KL and retained-particle displacement answer how much removing `T` changed the terminal field.
3. Vertex replay answers whether exact pass-2 endpoints invert.
4. Lambda-fit RMSE answers whether the saved `d1` parents express `T` in pass 1.
5. Terminal start error answers how far the same lambda moves after recomputing the field.
6. Arrival error and amplification answer whether backward replay preserves that neighborhood.
7. The target makes this a controlled reconstruction test, not literal deployment generation.
8. Decoder-level molecular coherence remains untested.

## Assumptions and edge cases

- Every source has the same particle count.
- Particle order has no global meaning.
- Matching is exact only up to 16 particles.
- Source ownership never changes particle forces.
- The target participates normally in pass 1 and is absent from every pass-2 frame.
- Parent IDs and particle matching are selected once at `d1` and never recomputed at `du`.
- Lambda is fitted at pass-1 `du`, then frozen for pass 2.
- Initial-space target coordinates are used for scoring, never for lambda fitting.
- Lambda is nonnegative and sums to one, so this is interpolation rather than extrapolation.
- Independent random grouping means there is no synthetic source-coherence signal to recover.
- Distribution-wise target recovery is the strongest conclusion available before decoding.
- Generated particles remain passive during replay.
- Search calibration depends on the particle generator, density, dimension, and seed set.
- Existing result directories remain valid records of their older code and data model, but their metric columns are not directly comparable with this revision.

## Saved outputs

Every run creates a new timestamped directory and refuses collisions.

- `config.json`: parameters, data model, gates, and runtimes.
- `pass1_reference.npz`: target, saved parents, correspondence, lambda, and pass-1 endpoints written before pass 2 starts.
- `reference_smoke.csv`: pass-1 geometry with target present.
- `generation_smoke.csv`: pass-2 geometry and vertex replay with target absent.
- `metrics.csv`: one shared-lambda row per target.
- `samples.npz`: both histories, source IDs, saved matching, lambda, KL histograms, candidates, replay, and metrics.
- `summary.txt`: target ID, parents, lambda, KL, terminal error, arrival error, amplification, and runtimes.
- `two_pass_target.png`
- `two_pass_field_shift.png`

Single-pass toy runs retain their older `smoke.csv` and figure set.

Search outputs remain append-only and include `search_config.json`, `forward_grid.csv`, `replay_grid.csv`, `best_config.json`, `search_summary.txt`, and `hyperparameter_selection.png`.

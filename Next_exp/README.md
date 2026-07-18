# Final Single-Plane H4.5 Diagnostics

This folder implements the three required experiments from [`H45_NEXT_EXPERIMENTS_README.md`](H45_NEXT_EXPERIMENTS_README.md). It does not change the parent H4.5 code.

The goal is to distinguish:

1. a representation failure, where one shared lambda cannot place a complete source near its target;
2. an inverse-geometry failure, where a good terminal seed is not mapped smoothly or accurately back to the original plane.

There is one pooled EFS field. Generated particles and held-out targets are passive. They feel the saved field but never alter it and never interact with one another.

## One-line pipeline

`create one target-free field per seed -> select d1 correspondences -> forward EFS once -> passively transport targets -> run K sweep, local du perturbations, and lambda paths through the same history -> save metrics -> classify`

## Hypothesis-to-code matrix

| Scientific question | Code | Why |
| --- | --- | --- |
| Does increasing parent count repair the shared-lambda fit? | `_run_parent_sweep()` | Changes representational capacity without changing the EFS field. |
| Are small terminal displacements amplified? | `_run_du_sensitivity()` | Measures the local inverse around exact memory vertices. |
| Does a smooth lambda produce a smooth complete output? | `_run_lambda_paths()` | Detects whole-source or single-particle jumps along a controlled path. |
| Are parents meaningful in the original data? | `_rank_target_neighbors()` | Ranks sources and fixes particle matching in $d_1$, before EFS. |
| Is lambda fitted in the correct coordinates? | `_run_parent_sweep()` | Fits lambda from the fixed parents' positions in $d_u$. |
| Are the EFS equations unchanged? | imports from `../efs.py` | The experiment reuses the canonical one-plane forward, passive-forward, and backward replay. |
| Are source comparisons unordered? | imports from `../data.py` | Exact particle matching is reused instead of assuming arbitrary row order. |

## Data flow

For each seed:

1. Draw `source_count + targets` complete random sources from the existing chaotic particle generator.
2. Remove the held-out targets before EFS.
3. Rank every target's memory neighbors in the initial plane $d_1$.
4. Save the particle permutation from each target to each candidate parent.
5. Run one forward EFS history using memory particles only.
6. Carry held-out targets passively through that same field to obtain evaluation witnesses in $d_u$.
7. Reuse the same history for all three experiments.
8. Save the history, source identities, target identities, parent rankings, permutations, candidates, outputs, and metrics.

The target-free field is intentional. It matches generation time and avoids the pass-1 versus pass-2 field difference from the earlier two-pass test.

## Mathematical and algorithmic formulation

### Shared-lambda fit

For a target $T\in\mathbb{R}^{P\times D}$ and $K$ matched terminal parents $S_k$, fit:

$$
\lambda^*
=
\underset{\lambda_k\geq 0,\ \sum_k\lambda_k=1}{\operatorname{argmin}}
\frac{1}{PD}
\sum_{p=1}^{P}\sum_{d=1}^{D}
\left[
T_{p,d}-\sum_{k=1}^{K}\lambda_k S_{k,p,d}
\right]^2.
$$

The `[P,D]` source is flattened only for this optimization. One vector `[K]` is fitted over the whole source. No particle receives its own lambda.

### Parent-count sweep

The default sweep is:

$$
K\in\{4,8,16,32,64,128\}.
$$

The first $K$ identities are taken from one fixed $d_1$ ranking. Their saved particle permutations are reused at $d_u$. The terminal seed is:

$$
Y^{(T)}_p=\sum_{k=1}^{K}\lambda_k S^{(T)}_{k,p}.
$$

The direct baseline uses the same fitted lambda and the same matched parents in $d_1$:

$$
Y^{(0)}_{\mathrm{direct},p}=\sum_{k=1}^{K}\lambda_k S^{(0)}_{k,p}.
$$

This direct value is a baseline, not the proposed EFS generator.

### Local terminal sensitivity

For an exact memory trajectory $x^{(0)}\rightarrow x^{(T)}$, construct:

$$
y^{(T)}=x^{(T)}+\delta.
$$

The perturbation length is a declared fraction $a$ of that vertex's terminal nearest-neighbor spacing $r_{\mathrm{NN}}$:

$$
\lVert\delta\rVert=a\,r_{\mathrm{NN}},
\qquad
a\in\{10^{-4},3\times10^{-4},10^{-3},3\times10^{-3},10^{-2},3\times10^{-2},10^{-1}\}.
$$

Directions are:

- one random unit direction per vertex;
- toward its nearest terminal particle;
- toward a particle that is close in $d_u$ but far in $d_1$;
- the first singular direction of local terminal offsets;
- radial from the terminal field center;
- tangential to that radial direction.

Two amplification values are saved:

$$
A_{\mathrm{original}}
=
\frac{\lVert B(y^{(T)})-x^{(0)}\rVert}{\lVert\delta\rVert},
$$

and

$$
A_{\mathrm{local}}
=
\frac{\lVert B(y^{(T)})-B(x^{(T)})\rVert}{\lVert\delta\rVert}.
$$

The first is the exact metric requested in the plan. The second subtracts ordinary vertex replay error, which otherwise dominates the ratio for extremely tiny perturbations.

### Shared-lambda path

For each fixed source pair:

$$
\lambda(t)=[1-t,t],
\qquad
t=0,0.01,\ldots,1.
$$

Every corresponding particle uses the same $t$. The experiment measures the terminal step and backward-output step:

$$
A_{\mathrm{step}}(t)
=
\frac{
\lVert B(Y^{(T)}(t))-B(Y^{(T)}(t-\Delta t))\rVert
}{
\lVert Y^{(T)}(t)-Y^{(T)}(t-\Delta t)\rVert
}.
$$

Pairs are selected in four categories:

- globally close within a fixed sampled pool in $d_1$;
- close within that pool in $d_u$;
- close in $d_u$ but far in $d_1$ under the same terminal matching;
- an ordinary $d_1$ nearest neighbor for a sampled anchor.

## Result interpretation

All complete-source RMSE values are divided by the relevant plane's coordinate standard deviations before aggregation. This prevents one coordinate with a large raw scale from controlling the result.

### Parent-count metrics

| Metric | Input and output | Project meaning | Desired result |
| --- | --- | --- | --- |
| `terminal_fit_rmse` | Terminal target and terminal shared-lambda seed | Pure representation capacity before replay | Decreases strongly with $K$; practical target at most `0.10` |
| `replay_target_rmse` | Backward arrival and original held-out target | Full H4.5 recovery error | Decreases with $K$ and remains close to terminal fit error |
| `direct_same_lambda_rmse` | Initial parents mixed with the terminal-fitted lambda | Baseline without EFS replay | EFS should be no worse if EFS adds value |
| `replay_over_terminal` | Replay error divided by terminal fit error | Error amplification across coordinate frames | Near `1`; large values identify inverse loss |
| `replay_over_direct` | Replay error divided by direct error | Value added by EFS | Below `1` is better than direct interpolation |
| `lambda_max` | Largest fitted coefficient | Dominance by one parent | Context dependent; near `1` means almost retrieval |
| `lambda_entropy` | All fitted coefficients | How diffuse the source recipe is | Lower is simpler, but zero means one copied parent |
| `effective_parent_count` | $1/\sum_k\lambda_k^2$ | Number of meaningfully active parents | Much smaller than available $K$ is sparse and interpretable |
| `fit_iterations` | Projected-gradient updates | Lambda solver convergence effort | Below the configured maximum |
| `max_replay_residual` | Backward fixed-point residual | Numerical convergence, not validity | Small and stable across $K$ |

Interpretation patterns:

- Poor terminal fit for every $K$: the shared convex representation is insufficient.
- Good terminal fit but poor arrival: the inverse map is the bottleneck.
- Both improve: the earlier $K=4$ setup was underparameterized.
- Good results only with a very large effective parent count: the method may work numerically but provenance becomes diffuse.

### Local sensitivity metrics

| Metric | Input and output | Project meaning | Desired result |
| --- | --- | --- | --- |
| `terminal_delta` | Perturbed seed minus exact terminal vertex | Actual local input change | Matches the declared fraction of nearest-neighbor spacing |
| `output_delta` | Replayed perturbation minus original vertex | Total recovery error | Small, but includes exact vertex replay error |
| `output_delta_from_exact_replay` | Perturbed replay minus exact replay | Local inverse response | Approximately proportional to terminal delta |
| `local_amplification` | Local output change divided by terminal change | Local expansion of the inverse | Smooth; p95 below `10` is the declared practical flag |
| `identity_changed` | Nearest initial particle before and after | Branch or identity crossing | Rare below the `0.01` perturbation scale; target at most `5%` |
| `output_distance_to_original_source` | Output to particles owned by its source | Whether the result remains near its source | Small relative to initial scale |
| `exact_vertex_error` | Exact terminal vertex replayed to $d_1$ | Numerical inverse baseline | Ideally below `1%` of field scale |

Very large `amplification` at the smallest perturbations may reflect the exact vertex error in its numerator. Use `local_amplification` to judge the inverse geometry itself.

### Lambda-path metrics

| Metric | Input and output | Project meaning | Desired result |
| --- | --- | --- | --- |
| `terminal_step_norm` | Consecutive terminal seeds | Known smooth input motion | Nearly constant for a linear two-parent path |
| `output_step_norm` | Consecutive backward outputs | Actual generated motion | Smooth without isolated spikes |
| `step_amplification` | Output step divided by terminal step | Pathwise inverse expansion | p95 below `10` is the practical flag |
| `nearest_source_identity` | Generated source against all memory sources | Mode or retrieval-region changes | Changes can occur, but not through abrupt particle jumps |
| `pairwise_relation_change` | Consecutive internal particle-distance vectors | Whole-source shape continuity | Smooth and small |
| `maximum_particle_jump` | Largest particle movement in one path step | One-part failure hidden by source averages | No isolated spike much larger than neighboring steps |

## Decision rule

`summary.json` and `summary.txt` apply transparent practical flags. They are not formal statistical tests.

Continuation requires all of the following:

1. median terminal fit improves by at least `25%` from the smallest to largest tested $K$;
2. largest-$K$ median terminal RMSE is at most `0.10`;
3. median replay error improves by at least `20%`;
4. largest-$K$ replay is no worse than direct same-lambda interpolation;
5. p95 local amplification for perturbations up to `0.01` is at most `10`;
6. at most `5%` of those small perturbations change nearest initial identity;
7. pooled p95 lambda-path step amplification is at most `10`.

A hard numerical invalidation still occurs if the imported EFS smoke gate detects nonfinite values, an RMS-radius ratio outside `[0.05,5]`, or covariance collapse below `1e-3`. Radial CDF mismatch remains descriptive.

## Figures

| Figure | Meaning |
| --- | --- |
| `error_vs_k.png` | Median terminal, replay, and direct errors plus every target's replay curve. |
| `amplification_vs_k.png` | Replay error relative to terminal fit and direct interpolation. Values above one are worse. |
| `effective_k.png` | How much of the available parent library lambda actually uses. |
| `amplification_by_scale.png` | Median and p95 local amplification plus identity-change fraction by perturbation size and direction. |
| `source_displacement.png` | Whole-source motion and step amplification along each lambda path. |
| `particle_displacement.png` | Maximum and individual particle jumps, exposing failures hidden by source averages. |

## Files

- `run.py`: data preparation, all three experiments, CSV/NPZ saving, and CLI.
- `plotting.py`: plot regeneration from saved CSV files.
- `H45_NEXT_EXPERIMENTS_README.md`: original experiment specification, unchanged.
- `README.md`: implemented protocol and interpretation guide.
- `req.txt`: the same two dependencies as the parent harness.

The EFS potential, forward field, passive transport, backward replay, chaotic data generator, exact particle matching, and simplex optimizer are imported from the parent `4.5hypothesis` folder.

## Environment

- Python `3.10` preferred.
- CPU-only.
- NumPy and Matplotlib only.
- A saved history uses approximately:

$$
8(J+1)N D\ \text{bytes}
$$

for `float64`, where $N=C\times P$.

From this folder:

```powershell
python -m pip install -r req.txt
```

Environment creation and package installation remain under user control.

## Run commands

Small path check:

```powershell
python run.py --quick --output-root results
```

Default moderate run, one seed and one shared history:

```powershell
python run.py --output-root results
```

Larger final run matching the earlier source scale more closely:

```powershell
python run.py --seeds 4652 61 73 --source-count 512 --targets 16 --particles-per-source 10 --workers 12 --output-root results
```

Run only one experiment:

```powershell
python run.py --experiments parent-sweep --output-root results
python run.py --experiments du-sensitivity --output-root results
python run.py --experiments lambda-path --output-root results
```

Running all three together is more efficient because they reuse each seed's forward history in memory.

Regenerate plots without EFS:

```powershell
python plotting.py results\<run-directory>
```

## Main parameters

| Argument | Default | Meaning |
| --- | ---: | --- |
| `--seeds` | `4652` | Independent data and field seeds. |
| `--source-count` | `256` | Memory sources. |
| `--targets` | `8` | Held-out targets used by the parent sweep. |
| `--particles-per-source` | `8` | Particles in each complete source. |
| `--dimension` | `4` | Coordinates per particle. |
| `--k-values` | `4 8 16 32 64 128` | Parent-count sweep. |
| `--sensitivity-vertices` | `12` | Exact memory particles used for local perturbations. |
| `--delta-factors` | seven values from `1e-4` to `1e-1` | Perturbation length relative to local terminal spacing. |
| `--path-pool` | `32` | Fixed source subset used to choose path categories. |
| `--path-pairs` | `1` | Pairs per path category. |
| `--path-step` | `0.01` | Lambda path resolution. |
| `--workers` | `8` | Concurrent passive replay batches. |
| `--config` | `../best_config.json` | Existing forward and replay parameters. |

Explicit `--epsilon`, `--exponent-s`, `--gamma`, `--forward-steps`, `--beta`, and `--proximal-steps` values override the saved parent configuration.

## Assumptions and edge cases

- Source ownership is metadata and never changes particle forces.
- Held-out targets are absent from the memory and transported passively only as evaluation witnesses.
- Parent identities and target-to-parent particle permutations are fixed in $d_1$.
- Lambda is fitted in $d_u$ using those exact saved identities and permutations.
- Exact dependency-free matching supports at most `16` particles per source.
- $K$ cannot exceed the memory source count.
- A large available $K$ may still produce a sparse lambda. `effective_parent_count` measures this.
- Exact nearest-source evaluation in the path experiment is intentionally expensive but avoids a heuristic identity metric.
- Threads share one immutable history. Too many workers can reduce speed through memory-bandwidth contention, so `--workers` remains adjustable.
- The synthetic random source grouping does not establish decoded biomolecular validity.
- The optional joint-vector control from the specification is not implemented. It requires a separate high-dimensional EFS model and is outside the passive one-plane decision rule.

## Saved outputs

Every timestamped run contains:

- `config.json`: complete parameters, smoke diagnostics, seeds, and status;
- `summary.json`: grouped descriptive metrics and decision inputs;
- `summary.txt`: short decision summary;
- `history_seed<seed>.npy`: the single forward history reused by all experiments;
- `field_seed<seed>.npz`: sources, target IDs, parent IDs, permutations, passive target trajectory, scales, and forward logs;
- `parent_sweep.csv` and `parent_sweep_seed<seed>.npz`;
- `du_sensitivity.csv`, `du_sensitivity_worst_cases.csv`, and `du_sensitivity_seed<seed>.npz`;
- `lambda_path.csv` and `lambda_path_seed<seed>.npz`;
- six regenerable PNG figures listed above.

Runs are timestamped and never overwrite an existing result directory.

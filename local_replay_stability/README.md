# Created-Target Local EFS Replay Stability

This experiment removes source retrieval and shared-lambda fitting entirely.
It asks a smaller question first:

> If two known terminal EFS particles are moved close together, can backward
> replay still return them to their different original positions?

The target is therefore not selected from a source library. It is constructed
from exact memory-particle identities whose original and terminal coordinates
are both known.

## One-line pipeline

Constant-density thick spiral $d_1$ -> EFS forward history -> select terminal neighbor pairs -> move each pair toward its midpoint -> replay every created seed -> compare arrivals with the known original vertices.

## Hypothesis-to-code matrix

| Question | Code | Why |
|---|---|---|
| Does density concentration cause apparent merging? | `generate_spiral_tube()` | The initial support has roughly constant local density and no central density spike. |
| Do close terminal seeds retain different identities? | `create_terminal_targets()` | Two exact vertices are moved symmetrically toward one another. |
| Is failure caused by the backward optimizer step? | `_replay_worker()` sweeps $eta$ | All beta values replay the same seeds through the same history. |
| Does stability depend on terminal distance? | $alpha=0,0.10,0.25,0.40,0.49$ | $alpha$ directly controls how close the two seeds become. |
| Does stability depend on dimension? | independent $D=2$ and $D=4$ runs | The same experiment is repeated in separate fields. |
| Are particles attracting one another? | passive `backward_replay()` | Created particles feel memory only and never interact with each other. |

Why this comes before another shared-lambda experiment: the previous run mixed
three problems together: terminal lambda-fit error, non-homogeneous density,
and backward replay. Here there is no lambda fit. Any error is caused by local
terminal displacement, the cached inverse field, or an under-converged
proximal solve.

## Mathematical and algorithmic formulation

### 1. Homogeneous nontrivial support

The centerline is an Archimedean spiral:

$$
r(t)=0.55+0.16t,
\qquad
c(t)=r(t)(\cos t,\sin t).
$$

The code samples $c(t)$ uniformly by arc length, not uniformly by angle. Each
point receives an independent sample from a uniform $(D-1)$-ball placed in the
normal cross-section. This creates a thick, full-dimensional tube.

The tube is only approximately constant-density in ambient coordinates because
curvature slightly changes cross-sectional volume. Its small width keeps that
effect minor. The important control is the absence of a large central density
peak.

### 2. EFS forward field

For memory particles $x_i^{(j)}\in\mathbb{R}^D$:

$$
\nabla W(z)
=
z-\frac{z}{(\lVert z\rVert^2+\epsilon)^{s/2+1}},
$$

$$
x_i^{(j+1)}
=
x_i^{(j)}
-
\frac{\gamma}{N-1}
\sum_{a\ne i}\nabla W(x_i^{(j)}-x_a^{(j)}).
$$

Every frame is cached as `history[J+1,N,D]`. The exponent is
$s=\max(1,D-2)$, giving $s=1$ in 2D and $s=2$ in 4D.

### 3. Created terminal targets

Choose a terminal nearest-neighbor pair $(x_i^{(J)},x_j^{(J)})$. For one
declared displacement fraction $0\le\alpha<0.5$, create:

$$
y_i^{(J)}(\alpha)
=
(1-\alpha)x_i^{(J)}+\alpha x_j^{(J)},
$$

$$
y_j^{(J)}(\alpha)
=
\alpha x_i^{(J)}+(1-\alpha)x_j^{(J)}.
$$

Their remaining separation is:

$$
\lVert y_i^{(J)}-y_j^{(J)}\rVert
=
(1-2\alpha)
\lVert x_i^{(J)}-x_j^{(J)}\rVert.
$$

Therefore:

- $alpha=0$: exact terminal vertices;
- $alpha=0.25$: half of the original separation remains;
- $alpha=0.40$: 20% remains;
- $alpha=0.49$: 2% remains, but the seeds are not identical.

The memory plane is not modified. The created targets are passive copies. If
the real memory particles were replaced, the field itself would change and the
experiment would no longer isolate local replay.

### 4. Backward replay

At cached post-forward frame $j$:

$$
F_j(v)=\frac{\gamma}{N}\sum_i\nabla W(v-x_i^{(j)}),
$$

$$
v_{t+1}
=
v_t-\beta\left(v_t-y^{(j)}-F_j(v_t)\right).
$$

$\beta$ is the proximal optimizer step. It should change convergence speed and
stability, not the destination after the residual has converged.

### 5. Local backward amplification

For exact terminal vertex $x^{(J)}$ and displaced seed $y^{(J)}$:

$$
A=
\frac{\lVert B(y^{(J)})-B(x^{(J)})\rVert}
{\lVert y^{(J)}-x^{(J)}\rVert}.
$$

$A$ measures how strongly the backward map enlarges or suppresses a small
terminal displacement. It is not a formal hypothesis test.

# RESULT INTERPRETATION

## Metrics

### `relative_plane_error`

- Input: generated arrivals and the known original memory vertices.
- Output: pair RMS arrival error divided by the initial plane RMS radius.
- Project meaning: physical reconstruction error on the scale of the complete field.
- Target: below `0.01` is excellent; above `0.10` means local replay is poor for this diagnostic.
- Invalidation: $alpha=0$ above `0.10` means vertex replay itself failed, so displaced results should not be trusted.

### `relative_pair_error`

- Input: the same arrival error and the original distance between the paired targets.
- Output: error relative to the local target-pair scale.
- Project meaning: whether two nearby identities are recovered accurately enough to remain distinguishable.
- Target: at most `0.10` for the declared `safe_alpha` summary.
- Caution: a very small original pair distance can make this ratio large even when absolute error is small.

### `arrival_separation_ratio`

- Input: generated pair separation and real original pair separation.
- Output: generated separation divided by target separation.
- Project meaning: whether replay restores the two distinct destinations.
- Target: near `1.0`.
- Collapse label: below `0.25`. This is a declared descriptive threshold, not a theorem.

### `identity_swapped`

- Input: direct assignment cost and the cost obtained by exchanging the two targets.
- Output: `True` when the exchanged assignment is closer.
- Project meaning: the two paths crossed or lost their declared particle identities.
- Target: always `False`.

### `local_amplification`

- Input: terminal displacement and the corresponding change from exact-vertex replay.
- Output: the amplification ratio $A$ defined above.
- Project meaning: sensitivity of the inverse field around a known endpoint.
- Target: there is no universal optimum. Values near `1` are easy to interpret; values much larger than `1` indicate local sensitivity; values near `0` indicate displacement erasure.

### `max_replay_residual`

- Input: the proximal fixed-point equation at every backward frame.
- Output: maximum normalized residual over the replay.
- Project meaning: distinguishes an under-converged optimizer from a converged field result.
- Target: below $10^{-5}$ is strong; above $10^{-3}$ suggests that $T$ or $\beta$ is inadequate.
- Invalidation: nonfinite residuals or positions invalidate that beta configuration.

### `safe_alpha`

The largest tested displacement satisfying all three conditions:

1. median `relative_pair_error <= 0.10`;
2. no identity swaps;
3. no separation collapse.

It is a compact operating-range description for this synthetic field, not a
guarantee for ANewOmni latents.

### Forward radial CDF gap

The configuration records the maximum difference between the empirical radial
CDF and a fitted uniform $D$-ball CDF. It is descriptive only. It does not stop
the replay test.

## Figure interpretation

### `created_target_flow.png`

Raw 2D coordinates for one representative pair:

1. known original destinations in $d_1$;
2. exact terminal vertices and created close seeds in $d_u$;
3. backward trajectories;
4. arrivals compared with the two known targets.

If different colors converge to the same arrival while their target crosses
remain separate, the local inverse erased terminal distinctions.

### `stability_summary.png`

One row per dimension:

- left: reconstruction error against displacement;
- center: restored target separation;
- right: local backward amplification.

Similar curves for all converged beta values mean beta is not controlling the
scientific result. Strongly different curves accompanied by large residuals
mean the proximal solve is under-converged.

### `beta_selection.png`

Rows are dimensions and columns are beta values. Each cell is median displaced
target error relative to initial RMS radius. The starred cell is the smallest
beta whose residual is at most $10^{-5}$, whose vertex error is at most 1%, and
whose displaced error is within 1% of the best converged result.

## Interpretation notes

- This tests local inverse stability, not shared-lambda source validity.
- The targets are deliberately based on training vertices so their destinations are known exactly.
- Created particles never interact with one another.
- Success does not prove decoded molecular validity.
- Failure at large $alpha$ identifies a finite local basin, not necessarily a global EFS failure.
- If changing beta no longer changes output after residual convergence, the cached field determines the result.

## Files

- `efs.py`: unchanged single-plane EFS field, forward history, and backward replay copied from the current research harness.
- `run.py`: spiral data, created target construction, beta sweep, evaluation, saving, and plots.
- `req.txt`: NumPy and Matplotlib versions.
- `README.md`: protocol, equations, commands, and result interpretation.

## Environment

- Python 3.10 preferred.
- CPU only.
- NumPy and Matplotlib only.
- No CUDA, ROCm, Torch, SciPy, pandas, YAML, or test framework.

Install dependencies in an environment you created:

```powershell
python -m pip install -r req.txt
```

## Run commands

From `E:\Desktop\AnewOmni\CG\local_replay_stability`:

Default 2D and 4D run:

```powershell
python run.py --output-root results
```

Small smoke run:

```powershell
python run.py --particles 256 --forward-steps 400 --pairs 3 --betas 0.05 0.1 0.2 --workers 3 --output-root results_smoke
```

Add an 11D field after the smaller runs are interpretable:

```powershell
python run.py --dimensions 2 4 11 --particles 768 --workers 5 --output-root results_11d
```

## Parameters

| Argument | Default | Meaning |
|---|---:|---|
| `--particles` | `768` | Memory particles per independent field. |
| `--dimensions` | `2 4` | Independent dimensions evaluated. |
| `--pairs` | `6` | Disjoint terminal nearest-neighbor pairs. |
| `--alphas` | `0 0.10 0.25 0.40 0.49` | Fractions moved toward the pair midpoint. |
| `--betas` | `0.025 0.05 0.10 0.20 0.40` | Backward proximal learning rates. |
| `--gamma` | `0.002` | Forward Euler step. |
| `--epsilon` | `0.03` | Potential regularization. |
| `--forward-steps` | `2000` | Cached forward frames. |
| `--proximal-steps` | `50` | Inner optimizer steps per backward frame. |
| `--tube-width` | `0.12` | Thick spiral cross-section radius before scaling. |
| `--workers` | `5` | Beta configurations replayed in parallel. |

## Assumptions and edge cases

- $alpha$ must remain below `0.5`; at exactly `0.5` both seeds are identical and identity is mathematically unrecoverable from position alone.
- Pair selection spans terminal nearest-neighbor spacing quantiles and does not reuse vertices.
- The main replay uses the literal paper post-forward frame convention.
- The dimension runs have different $s=\max(1,D-2)$ values, so the comparison includes the declared EFS dimension rule as well as dimensionality.
- Process workers receive read-only copies of the cached history. RAM use therefore increases with worker count.
- The runner refuses to overwrite an existing timestamped directory.

## Saved outputs

Each run creates:

- `config.json`: complete parameters, geometry diagnostics, selected beta, and runtime;
- `metrics.csv`: one row per dimension, beta, pair, and alpha;
- `samples.npz`: initial planes, histories, created seeds, arrivals, residuals, and best-beta trajectories;
- `summary.txt`: compact numerical interpretation;
- `created_target_flow.png`: raw 2D construction and replay;
- `stability_summary.png`: error, separation, and amplification;
- `beta_selection.png`: beta-by-dimension error map.

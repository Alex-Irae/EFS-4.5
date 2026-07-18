# Minimal Hypothesis 4.5 Experiment

This folder tests only the fixed shared-barycentric-coordinate hypothesis:

> Given the same two intact source complexes and the same slot-wise EFS histories, does one coefficient shared across every ordered slot preserve whole-complex coherence better than independent per-slot coefficients?

There are no conformational families, rotations, retrieval model, joint-vector EFS, ANewOmni checkpoint, decoder, likelihood model, or rejection stage.

## One-line pipeline

`coherent training chains -> slot-wise EFS forward history -> intact sources A/B -> shared or independent terminal lambdas -> identical batched EFS replay -> raw coherence comparison -> plots`

## Hypothesis-to-code matrix and why

| Hypothesis requirement                          | Code                                             | Why it is present                                                                                  |
| ----------------------------------------------- | ------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| Ordered complete complexes                      | `generate_complete_chains()`                     | One row is an intact chain with fixed slot correspondence.                                         |
| Same source identities across slots             | `select_source_ids()`                            | Source A and B are selected as complete complexes before any block is mixed.                       |
| One shared (\lambda)                            | `make_lambda_controls()` shared matrix           | Every slot of one candidate receives the same (u).                                                 |
| Independent-slot falsification control          | `make_lambda_controls()` shifted matrix          | Each slot uses the same marginal grid, but coefficient alignment is broken.                        |
| Barycentric terminal initialization             | `barycentric_terminal_seeds()`                   | Implements the exact two-source convex combination at terminal frame (K).                          |
| Ordinary slot-wise EFS                          | `forward_history()` and `backward_replay()`      | Shared lambda changes initialization only. Reverse maps remain independent by slot.                |
| Identical EFS budget and histories              | One batched replay call                          | Shared and independent candidates see the same frames, particles, parameters, and iteration count. |
| Failed vertex replay invalidates interpretation | Two terminal source particles are replayed first | Tests whether the finite EFS inverse approximately returns its own empirical vertices.             |
| Raw joint validity                              | `coherence_error()` and `bond_error()`           | The result is measured before filtering or rejection.                                              |
| Whole-flow visualization                        | `plot_pca_flow()`                                | One fixed whole-complex PCA plane makes forward transport, interpolation, and replay comparable.   |

### Why this toy is intentionally simple

Each training complex is one ordered chain controlled by a scalar state (t_i). Every slot in complex (i) shares that same state. Shared lambda should interpolate one state consistently across the whole chain. Independent lambdas deliberately assemble slots from different interpolation states.

This is a mechanism check. If shared lambda cannot win here, the base H4.5 idea or its EFS implementation is broken. Success here is necessary but not sufficient for ANewOmni peptides.

## Mathematical and algorithmic formulation

### Training complexes

For complex (i) and ordered slot (\ell):

$x_{i,\ell}=t_i+\eta^x_{i,\ell}$

$
y_{i,\ell}=\ell\,c+a\sin(t_i)+\eta^y_{i,\ell}$

where (c) is slot spacing, (a) is a common curve amplitude, and (\eta) is small Gaussian noise. The complete training array has shape `[N,L,2]`.

### EFS potential gradient

For displacement (z):

\nabla W_\epsilon^{(s)}(z)
=z-\frac{z}{(\lVert z\rVert^2+\epsilon)^{s/2+1}}

The two-dimensional default uses (s=d-2=0).

### Forward transport

For every slot independently:


$x_{i,\ell}^{(j+1)}
=x_{i,\ell}^{(j)}
-\frac{\gamma}{N-1}
\sum_{a\ne i}
\nabla W_\epsilon^{(s)}
\left(x_{i,\ell}^{(j)}-x_{a,\ell}^{(j)}\right)$


All slots are computed in one NumPy batch, but no force crosses between slots. Particle index (i) continues to identify the same complete source complex in every slot and frame.

### Shared two-source initialization

For sources A and B and (u\in[0,1]):

\[
y_{\ell}^{(K)}
=u\,x_{A,\ell}^{(K)}
+(1-u)x_{B,\ell}^{(K)}
\]

Shared H4.5 uses the same (u) for every (\ell).

The independent control uses (u_\ell):

\[
y_{\ell}^{(K)}
=u_\ell x_{A,\ell}^{(K)}
+(1-u_\ell)x_{B,\ell}^{(K)}
\]

Every slot receives exactly the same set of coefficient values across candidates. The grid is circularly shifted between slots, so only synchronization changes.

The primary proposal grid uses interior values from 0.1 to 0.9. The exact vertices (u=0) and (u=1) are handled separately by the mandatory vertex-replay diagnostic.

### Backward replay

At frame (j):

\[
F_j(v)=\frac{\gamma}{N}
\sum_i \nabla W_\epsilon^{(s)}(v-x_i^{(j)})
\]

\[
\Delta_t=v_t-y^{(j)}-F_j(v_t)
\]

\[
v_{t+1}=v_t-\beta\Delta_t
\]

After (T) iterations, (y^{(j-1)}=v_T). The implementation uses the literal post-forward frame `history[j]` and denominator (N), matching the project document and the original `EFS/efs.py` convention.

### Coherence and raw validity

The x coordinate represents the common chain state. Coherence error is:

\[
E_{\mathrm{coherence}}(Y)
=\sqrt{\frac{1}{L}\sum_\ell
\left(Y_{\ell,x}-\overline{Y_x}\right)^2}
\]

Bond error is the mean absolute deviation of adjacent-slot distance from the declared spacing (c).

A raw proposal is valid when both errors are below their respective 95th-percentile values in the unmodified training population. There is no rejection, repair, or learned score.

## Files

- `efs.py`: batched EFS forward field, cached history, backward replay, and one explicit-loop self-check.
- `run_hypothesis_4_5.py`: toy data, source selection, shared and independent candidates, metrics, saved outputs, and plots.
- `../../requirements.txt`: repository-wide Python dependencies.
- `results/`: created when the experiment is run. Every run receives a new timestamped directory.

## Environment

Preferred Python version: 3.10.

Environment creation and installation are intentionally left to the user:

```powershell
python -m pip install -r ..\..\requirements.txt
```

The implementation is CPU-only and uses NumPy plus Matplotlib. It batches all slots and all generated candidates. With the default `N=64`, its full pairwise arrays are small relative to 32 GB RAM.

## Run commands

Run the small built-in algebra check:

```powershell
python efs.py
```

Run the complete H4.5 experiment:

```powershell
python run_hypothesis_4_5.py --output-root results
```

For a shorter first execution while checking numerical behavior:

```powershell
python run_hypothesis_4_5.py --forward-steps 500 --proximal-steps 10 --output-root results
```

The short command is a speed and shape check only. With `T=10`, the proximal solve can be too incomplete for vertex replay, so its H4.5 comparison may be declared non-interpretable.

## Parameters

| Argument            | Default | Meaning                                                                                          |
| ------------------- | -------:| ------------------------------------------------------------------------------------------------ |
| `--particles`       | 64      | Complete empirical complexes (N).                                                                |
| `--slots`           | 4       | Ordered slots (L) per complex.                                                                   |
| `--candidates`      | 9       | Interior coefficient values (B), including 0.5. Must be odd.                                     |
| `--slot-spacing`    | 1.0     | Expected adjacent-slot distance.                                                                 |
| `--curve-amplitude` | 0.35    | Common nonlinear displacement controlled by state (t).                                           |
| `--noise-std`       | 0.03    | Per-coordinate training noise.                                                                   |
| `--source-low`      | -0.75   | Requested state for intact source A.                                                             |
| `--source-high`     | 0.75    | Requested state for intact source B.                                                             |
| `--gamma`           | 0.0005  | Forward Euler step and replay field scale.                                                       |
| `--epsilon`         | 0.01    | Inverse-power smoothing.                                                                         |
| `--exponent-s`      | 0.0     | EFS exponent (s=d-2) for (d=2).                                                                  |
| `--forward-steps`   | 5000    | Cached forward updates (K).                                                                      |
| `--beta`            | 0.05    | Backward proximal optimizer step.                                                                |
| `--proximal-steps`  | 100     | Inner iterations (T) per frame; vertex replay still decides whether the result is interpretable. |
| `--seed`            | 42      | Reproducible toy data seed.                                                                      |

## Data flow

1. `generate_complete_chains()` creates `training [N,L,2]` and one state per intact complex.
2. The array is transposed to `[L,N,2]` so EFS operates independently over slots while preserving complete-complex particle IDs.
3. `forward_history()` stores `[K+1,L,N,2]`.
4. Two intact source IDs are selected once and reused at every slot.
5. `make_lambda_controls()` creates shared and independent coefficient matrices `[B,L]` with identical per-slot marginal grids.
6. `barycentric_terminal_seeds()` produces shared and independent terminal complexes `[B,L,2]`.
7. Two exact terminal source vertices and both proposal sets are concatenated into `[2+2B,L,2]`.
8. One `backward_replay()` call gives every method identical histories and compute budget.
9. Frame zero is split into vertex, shared, and independent outputs.
10. Coherence, bond error, nearest-training distance, replay residual, and raw validity are saved and plotted.
11. `plot_pca_flow()` flattens each complete `[L,2]` complex to `[2L]`, fits one two-component NumPy-SVD projection, and reuses it for all four flow stages.

## Interpretation notes

Interpret H4.5 only if vertex replay is acceptably small. The script marks results non-interpretable when either source vertex replay RMSE exceeds 10 percent of the training RMS scale. This is a declared finite-numerical diagnostic, not a theorem from the EFS paper.

Evidence for H4.5 requires both:

- higher raw joint-valid percentage for shared lambda;
- lower median coherence error for shared lambda.

The result only establishes behavior in this direct controlled toy. It does not establish peptide validity, ANewOmni compatibility, or performance on nonlinear multimodal complexes.

Nearest-training distance is descriptive. Low distance may mean copying; high distance may mean either novelty or failure. It is not used alone as validity.

The PCA flow plot represents whole complexes, not individual slots. PCA is fitted once to the initial particles, terminal particles, shared terminal seeds, and shared replay outputs. It then projects every stored forward path and every stored replay frame through that fixed basis. Values are centered but not standardized because all toy coordinates use the same unit. The backward panel draws all shared-lambda candidates; an `x` marks a terminal seed and a circle marks its replayed output. PCA is descriptive and can hide motion outside its first two components.

## Assumptions and edge cases

- Every complex has one topology, one length, one slot ordering, and the identity coordinate gauge.
- Source A and B must be distinct complete complexes.
- Shared and independent methods use identical sources, terminal frames, EFS parameters, and reverse histories.
- The toy has one population and no hidden family labels.
- Full pairwise forward memory is (O(LN^2)). With large `N`, chunking would be needed even with 32 GB RAM.
- Runtime remains (O(KLN^2D)) forward and (O(KTBLND)) backward. Extra RAM batches slots and candidates but cannot remove sequential frame and proximal iterations.
- Negative squared distances caused only by floating-point roundoff are clamped to zero before the inverse power. Non-finite forces or trajectories fail explicitly.
- The 95th-percentile validity thresholds are fitted on the same toy training population because this is a mechanism check, not a held-out performance estimate.
- Force provenance, retrieval, alignment, decoder correction, and counterfactual removal are excluded because they do not test the shared-lambda mechanism itself.

## Saved outputs

Each `results/run_<UTC>_seed<seed>/` directory contains:

- `config.json`: all parameters, source IDs and states, thresholds, and runtimes.
- `samples.npz`: training complexes, complete cached history, source IDs, coefficient matrices, terminal seeds, replay trajectory, residuals, and raw generated complexes.
- `metrics.csv`: one row per raw shared or independent proposal.
- `summary.txt`: vertex replay, raw validity, coherence comparison, runtime, and a guarded conclusion.
- `chains.png`: intact sources and representative shared and independent generated chains.
- `coherence_comparison.png`: coherence error, bond error, and raw validity.
- `lambda_and_vertex_replay.png`: shared sweep, coefficient disagreement, and vertex replay.
- `pca_flow.png`: base distribution, every complete-complex forward path and final particle, every shared-lambda terminal seed, and every shared candidate's backward path in one fixed whole-complex PCA plane.

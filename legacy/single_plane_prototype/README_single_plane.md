# Single-Plane Hypothesis 4.5 Experiment

This is the current corrected experiment. It tests complete sources whose particles all inhabit one shared EFS plane.

Each complete source contains eight corresponding 4D particles. Four complete sources are selected, one simplex vector is applied to every corresponding particle, and the resulting eight-particle source is replayed through the same cached EFS field.

The older `efs.py` and `run_hypothesis_4_5.py` files are preserved as the previous four-slot, separate-field toy. They are not used by this experiment.

## One-line pipeline

`256 grouped sources -> flatten 2,048 particles into one EFS plane -> exact forward history -> select four complete sources -> shared and independent-control interpolation -> one batched vanilla replay -> frame logs, metrics, and particle-level PCA`

## Hypothesis-to-code matrix and why

| Requirement                  | Code                          | Why                                                                                      |
| ---------------------------- | ----------------------------- | ---------------------------------------------------------------------------------------- |
| One shared EFS plane         | `sources.reshape(N,D)`        | Every particle from every complete source contributes to the same empirical field.       |
| Complete sources             | `sources [C,P,D]`             | Source identity and its eight-particle membership remain recoverable.                    |
| Four source chimera          | `selected_terminal [S=4,P,D]` | Corresponding particles can be mixed across four complete sources.                       |
| One shared lambda            | `einsum("s,spd->pd", ...)`    | The same operator-style source weights are reused for all eight particles.               |
| Random shared witness        | Seeded Dirichlet vector `[S]` | Provides one reproducible unconstrained shared initialization.                           |
| Independent control          | `einsum("ps,spd->pd", ...)`   | Deliberately breaks weight synchronization for falsification only.                       |
| Same replay conditions       | One `backward_replay()` call  | Every candidate sees identical frames, parameters, and compute budget.                   |
| Vanilla EFS replay           | `backward_field()`            | Generated particles query the same memory field but do not exert forces on one another.  |
| Runtime diagnostics          | Ten-frame logs                | Shows field spread, complete-source movement, and stagnation without proximal-step spam. |
| Particle-level visualization | `plot_particle_pca()`         | A complete source appears as eight points rather than one flattened PCA point.           |

### Why the independent lambda still exists

The independently weighted candidate is not a proposed runtime input. A human operator cannot assign it one meaningful chimera composition because each particle receives a different source mixture. It exists only to answer the H4.5 comparison: does synchronizing one valid source-level input preserve a complete source better than breaking that synchronization?

Both candidates are evaluated together. Starting two operating-system processes would duplicate the large forward history and make the comparison less controlled.

## Mathematical and algorithmic formulation

### Array meanings

```text
C = complete sources in memory, default 256
P = particles in each complete source, default 8
D = dimensions per particle, default 4
N = total particles in the shared plane, C*P = 2048
S = selected complete sources, default 4
J = forward and backward frames, default 5000
G = generated methods, default 2
```

```text
sources                 [C,P,D]
initial plane           [N,D]
forward history         [steps+1,N,D]
selected terminal       [S,P,D]
shared lambda           [S]
independent lambdas     [P,S]
terminal candidates     [G,P,D]
backward trajectory     [steps+1,G,P,D]
```

### Synthetic grouped sources

For memory source `c` and corresponding particle `p`:


$S_{c,p}=h_c+q_p+\eta_{c,p}$


where:

- \(h_c\in\mathbb{R}^4\) is the complete-source center;
- \(q_p\in\mathbb{R}^4\) is a fixed particle-identity template;
- ($\eta_{c,p}$) is small independent noise.

This is one population with no rotations, conformational families, retrieval, or separate planes. The template defines particle correspondence. The randomly selected complete sources are not filtered by distance.

### One exact EFS plane

The grouped array is reshaped without changing row order:

\[
[C,P,D]\longrightarrow[N=C P,D]
\]

Every plane particle interacts with every other plane particle:

\[
\nabla W(z)=z-\frac{z}{(\lVert z\rVert^2+\epsilon)^{s/2+1}}
\]

\[
x_i^{j+1}=x_i^j-
\frac{\gamma}{N-1}\sum_{a\ne i}\nabla W(x_i^j-x_a^j)
\]

The implementation keeps scalar pair arrays `[N,N]` and algebraically reduces the vector sums. It never allocates `[N,N,D]`.

At the default `N=2048`:

```text
ordered interactions per frame = 2048^2 = 4,194,304
one float64 scalar pair matrix = 32 MiB
cached history at 5,000 steps and D=4 = about 0.31 GiB
```

Several scalar pair matrices may coexist during one forward calculation. CPU time, not the 32 GB memory capacity, is the main limitation.

### Four-source shared interpolation

One simplex vector satisfies:

\[
\lambda_k\ge0,
\qquad
\sum_{k=1}^4\lambda_k=1
\]

The shared generated source is:

\[
Y_{p,d}=\sum_{k=1}^{4}\lambda_k S_{k,p,d}
\]

The same four values are applied to `p=1,...,8`.

The optional operator input is explicit:

```powershell
python run_single_plane_h45.py --shared-lambda 0.4,0.3,0.2,0.1 --output-root results
```

When no operator vector is supplied, the seeded random shared witness is the only shared candidate. When an operator vector is supplied, the runner batches:

1. the operator candidate;
2. the random shared reference;
3. the independent random control.

### Independent falsification control

For the control only:

\[
Y^{\mathrm{ind}}_{p,d}
=\sum_{k=1}^{4}\lambda_{p,k} S_{k,p,d}
\]

Each particle receives a separately sampled simplex vector. These weights are reproducible from the seed but have no operator interpretation.

### Vanilla backward replay

For every generated particle query `v` at frame `j`:

\[
F_j(v)=\frac{\gamma}{N}\sum_i\nabla W(v-x_i^j)
\]

\[
\Delta_t=v_t-y^j-F_j(v_t)
\]

\[
v_{t+1}=v_t-\beta\Delta_t
\]

All generated particles query the same `history[j]`. Candidate batching changes execution speed only. No generated particle appears in another generated particle's force sum.

## Runtime logging

### Forward

Every ten complete forward frames:

```text
forward step  100/5000: mean_pair_distance=...
```

The value is the exact mean Euclidean distance over all distinct pairs in the 2,048-particle plane.

### Creation

After the forward pass, the runner prints:

- selected complete source IDs;
- every selected terminal source `[8,4]`;
- the shared reference lambda `[4]`;
- optional operator lambda `[4]`;
- independent control lambdas `[8,4]`;
- every interpolated terminal source `[8,4]`.

### Backward

Every ten complete reverse frames, for every generated method:

```text
mean_pair_distance
total_position
displacement
displacement_norm
```

The total position is:

\[
R_j=\sum_{p=1}^{8}Y_p^j\in\mathbb{R}^4
\]

The displayed displacement is one complete reverse-frame movement:

\[
R_{j-1}-R_j
\]

It is not a proximal optimizer iteration and not the accumulated movement between printed frames.

## Files

- `efs_single_plane.py`: one-plane forward transport, vanilla replay, logging, and one algebra self-check.
- `run_single_plane_h45.py`: grouped data, source selection, lambdas, metrics, saved arrays, and figures.
- `README_single_plane.md`: this document.
- `req.txt`: NumPy and Matplotlib dependencies.
- `efs.py`, `run_hypothesis_4_5.py`, `README.md`: preserved previous two-dimensional experiment.

## Environment

Preferred Python version: 3.10.

Environment creation remains user-managed:

```powershell
python -m pip install -r req.txt
```

The code uses one CPU process. NumPy's matrix multiplication may use the BLAS threads supplied by the installed NumPy build.

## Run commands

From `4.5hypothesis`:

```powershell
python efs_single_plane.py
python run_single_plane_h45.py --output-root results
```

Small execution for checking shapes and logs only:

```powershell
python run_single_plane_h45.py --plane-particles 128 --forward-steps 20 --proximal-steps 10 --output-root results
```

The small command is not an equivalent EFS experiment.

## Parameters

| Argument                 | Default | Meaning                                                         |
| ------------------------ | -------:| --------------------------------------------------------------- |
| `--plane-particles`      | 2048    | Total particles `N` in the one plane. Must be divisible by `P`. |
| `--particles-per-source` | 8       | Corresponding particles `P` in every source.                    |
| `--selected-sources`     | 4       | Complete sources `S` used for interpolation.                    |
| `--dimension`            | 4       | Particle dimension `D`.                                         |
| `--center-std`           | 2.0     | Spread of complete-source centers.                              |
| `--particle-scale`       | 1.0     | Spread of fixed particle identities.                            |
| `--noise-std`            | 0.05    | Source-particle noise.                                          |
| `--gamma`                | 0.0005  | Forward step and replay-field scale.                            |
| `--epsilon`              | 0.1     | Near-collision regularizer.                                     |
| `--exponent-s`           | `D-2`   | Interaction exponent, therefore 2 at `D=4`.                     |
| `--forward-steps`        | 5000    | Exact forward frames.                                           |
| `--beta`                 | 0.05    | Proximal fixed-point step.                                      |
| `--proximal-steps`       | 100     | Inner replay iterations per frame.                              |
| `--log-every`            | 10      | Complete-frame logging interval.                                |
| `--shared-lambda`        | unset   | Optional comma-separated operator vector.                       |
| `--seed`                 | 42      | Reproducible data, sources, and random controls.                |

## Interpretation notes

- The random shared candidate is one witness, not a statistical sample of performance.
- The independent candidate is a negative control, not a meaningful chimera input.
- One plane supplies a shared marginal EFS field. Shared lambda supplies initialization-level complete-source coupling.
- Vanilla replay remains particle-wise after initialization. The generated particles do not dynamically couple one another.
- A straight terminal interpolation in PCA is expected because both interpolation and PCA projection are linear.
- Lower complete-source displacement can indicate convergence or stagnation. Replay residual and location relative to the memory plane must be considered with it.

## Assumptions and edge cases

- Every complete source has exactly eight particles and one declared particle-index correspondence.
- Equal particle count alone is not enough if indices do not identify corresponding particles.
- All particles use one coordinate gauge and one four-dimensional metric.
- Selected sources are random and may be distant. This is broader than the locality assumption in the formal H4.5 statement.
- Exact forward work is `O(steps*N^2*D)`; 32 GB RAM does not remove that CPU cost.
- Exact backward work is `O(steps*T*G*P*N*D)`.
- Increasing `P` adds particles to every complete source. Increasing `N` increases empirical field density. They are different controls.
- The full history is intentionally retained for exact replay and visualization.
- Non-finite fields or trajectories stop the run.
- Existing result directories are never overwritten or deleted.

## Saved outputs

Each `results/single_plane_<UTC>_seed<seed>/` directory contains:

- `config.json`: numerical parameters, source IDs, thresholds, labels, and runtimes.
- `samples.npz`: grouped sources, complete forward history, lambda provenance, terminal candidates, replay trajectory, residuals, and printed diagnostic arrays.
- `metrics.csv`: one row per generated method.
- `summary.txt`: compact numerical report without a formal statistical claim.
- `pca_particle_flow.png`: one particle-level PCA for the base plane, selected forward paths, four-source interpolation, and eight-particle replay.
- `replay_movement.png`: internal particle spread and total-position displacement norm for every complete reverse frame.
- `method_comparison.png`: template coherence, internal shape error, and nearest-source distance.

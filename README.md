# H4.5 Shared-Lambda EFS Research

This repository contains the complete CPU research trail for Hypothesis 4.5: use one source-level barycentric vector for every particle of a generated object, place those particles in a shared EFS terminal field, then replay them backward.

The experiments are organized chronologically under `experiments/`. Reusable one-plane EFS code remains at the repository root. Scientific documents and the consolidated result ledger live under `docs/`.

## Current conclusion

The passive shared-lambda route is **not validated and is stopped in its current form**.

The EFS field itself was numerically usable: accepted runs stayed finite and exact memory vertices usually replayed accurately. The generation mechanism failed at the harder off-vertex step:

- shared convex seeds fitted arbitrary complete sources poorly, even with $K=128$ parents;
- replay did not preserve the improvement obtained from additional parents;
- close terminal seeds were not reliably separated into their original identities;
- smooth source-level lambda paths produced p95 step amplification of $55.59$ in the final run;
- no experiment could measure decoded molecular validity on the available hardware.

See [`docs/experiment_ledger.md`](docs/experiment_ledger.md) for the evidence and failure justification of every tested variant.

## One-line pipeline

`synthetic complete sources -> one pooled EFS field -> terminal shared-lambda seed -> passive backward replay -> direct and vertex controls -> source-level error and continuity diagnostics`

## Hypothesis-to-code matrix

| Requirement                      | Implementation                    | Why                                                                                  |
| -------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------ |
| One pooled particle field        | `efs.py`                          | Every memory particle participates in the same vanilla EFS interaction.              |
| Source ownership as metadata     | `data.py`                         | Group membership never changes particle forces.                                      |
| Unordered source matching        | `data.best_permutation()`         | Synthetic source rows have no meaningful fixed order.                                |
| One complete-source lambda       | `data.fit_shared_source_lambda()` | One coefficient vector is fitted over all $P\times D$ values.                        |
| Forward and backward diagnostics | `evaluation.py`                   | Separates field failure, exact-vertex inversion, and off-vertex generation.          |
| Canonical reconstruction runner  | `run.py`                          | Runs the one-plane single-pass toys or two-pass target-removal protocol.             |
| Numerical calibration            | `search.py`                       | Selects EFS parameters without using target-recovery outcomes.                       |
| Historical model variants        | `experiments/01` through `03`     | Preserves equations that differ from the canonical pooled model.                     |
| Final mechanism tests            | `experiments/08` and `09`         | Tests identity compression, parent capacity, local sensitivity, and path continuity. |

## Mathematical formulation

The canonical potential gradient is:

$$
\nabla W(z)
=
z-
\frac{z}{(\lVert z\rVert^2+\epsilon)^{s/2+1}}.
$$

One shared barycentric vector satisfies:

$$
\lambda_k\geq0,
\qquad
\sum_{k=1}^{K}\lambda_k=1.
$$

For corresponding particle $p$ from complete parent sources $S_k$, the terminal seed is:

$$
Y_p^{(T)}
=
\sum_{k=1}^{K}\lambda_k S_{k,p}^{(T)}.
$$

The same lambda is used for every $p$. It correlates initialization only. Generated particles remain passive during replay and do not exert forces on one another.

## Repository layout

```text
4.5hypothesis/
  README.md
  requirements.txt
  best_config.json
  data.py
  efs.py
  evaluation.py
  plotting.py
  run.py
  search.py
  docs/
    h45_project.md
    broad_hypotheses.md
    adjacent_hypotheses.md
    experiment_ledger.md
  experiments/
    01_separate_fields/
    02_single_plane_witness/
    03_slot_single_calibration/
    04_heldout_reconstruction/
    05_point_interpolation/
    06_strong_contraction/
    07_two_pass_reconstruction/
    08_local_replay_stability/
    09_final_diagnostics/
```

## Experiment index

| Folder                                                                  | Question                                                                                       | Result                                                                                                 |
| ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| [`01_separate_fields`](experiments/01_separate_fields/)                 | Does one field per declared slot preserve coherence?                                           | Numerically positive after repair, but rejected because synthetic particles had no real slot identity. |
| [`02_single_plane_witness`](experiments/02_single_plane_witness/)       | Is shared lambda cleaner than independent per-particle weights in one field?                   | Shared was cleaner, but both outputs failed validity and only one witness was tested.                  |
| [`03_slot_single_calibration`](experiments/03_slot_single_calibration/) | Can geometry and replay be calibrated, and does shared beat a constructed independent control? | Vertex replay worked; constructed shared targets favored the tested algebra and were not decisive.     |
| [`04_heldout_reconstruction`](experiments/04_heldout_reconstruction/)   | Can four nearby sources reconstruct an absent random complete source?                          | Negative: EFS beat direct interpolation in only $3/16$ trials.                                         |
| [`05_point_interpolation`](experiments/05_point_interpolation/)         | Does the mechanism work when one source is one 2D point?                                       | Mixed: $33/64$ wins, but no multi-particle coherence was tested.                                       |
| [`06_strong_contraction`](experiments/06_strong_contraction/)           | Do more forward steps and stronger contraction repair recovery?                                | Negative: $2/16$ wins and a shell-like endpoint.                                                       |
| [`07_two_pass_reconstruction`](experiments/07_two_pass_reconstruction/) | Is target removal the main failure cause?                                                      | No: the field shift was tiny and the shared terminal fit was already poor.                             |
| [`08_local_replay_stability`](experiments/08_local_replay_stability/)   | Does a clean homogeneous field restore identities after terminal compression?                  | No in the clean 2D control; compressed pairs remained compressed.                                      |
| [`09_final_diagnostics`](experiments/09_final_diagnostics/)             | Do more parents, local stability, and smooth lambda paths rescue H4.5?                         | Stop: $K=128$ remained inaccurate, replay worsened, and path amplification was high.                   |

Each experiment owns its code when that code differs from the canonical root mechanism, plus its CSV, JSON, summaries, and figures.

## Result interpretation

| Metric                  | What it measures                                                           | Desired value                                                       | What it does not prove                            |
| ----------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------- |
| Vertex replay error     | Exact terminal memory particles returning to their known initial positions | Below $10\%$ of field scale; accepted runs were usually below $1\%$ | Validity of a new off-vertex source               |
| Terminal fit RMSE       | Distance from a shared convex seed to its terminal target                  | Near zero; final practical target was at most `0.10`                | Successful replay                                 |
| Replay target RMSE      | Distance from backward output to the known original target                 | Lower than direct interpolation                                     | Decoded molecular validity                        |
| Replay/direct ratio     | EFS error divided by direct same-lambda error                              | Below `1`                                                           | Novelty or usefulness by itself                   |
| Local amplification     | Output change divided by a small terminal perturbation                     | Smooth; p95 practical flag below `10`                               | Whole-source continuity                           |
| Path step amplification | Consecutive output step divided by consecutive terminal step               | Smooth without spikes; p95 practical flag below `10`                | Chemical validity                                 |
| Radial CDF gap          | Descriptive mismatch from a fitted uniform ball                            | Smaller is geometrically cleaner                                    | A formal goodness-of-fit test or inverse validity |

The decisive distinction is exact-vertex versus off-vertex behavior. A low vertex error proves that the numerical inverse works where the field already contains a known particle. H4.5 requires stable behavior between those known particles.

## Environment

- Python `3.10` preferred.
- CPU-only NumPy implementation.
- NumPy and Matplotlib only.
- No CUDA, ROCm, Torch, SciPy, pandas, YAML, or test framework.

```powershell
python -m pip install -r requirements.txt
```

Environment creation and package installation remain user-controlled.

## Main commands

Run the canonical two-pass experiment from the repository root:

```powershell
$workers = [Math]::Min(32, [Environment]::ProcessorCount)
python run.py --protocol two-pass --heldout-sources 1 --workers $workers --output-root experiments\07_two_pass_reconstruction\results
```

Re-run the canonical one-plane calibration without writing a root-level result folder:

```powershell
python search.py --workers 4 --output-root experiments\03_slot_single_calibration\results
```

Run the final diagnostics:

```powershell
cd experiments\09_final_diagnostics
python run.py --output-root results
python plotting.py results\<run-directory>
```

Run only the tiny final-diagnostics path check:

```powershell
cd experiments\09_final_diagnostics
python run.py --quick --output-root results
```

Detailed commands and parameter tables remain in each experiment README.

## Canonical parameters

`best_config.json` is the retained one-plane calibration reference:

- $\epsilon=0.01$;
- $s=3.0$;
- $\gamma=0.001$;
- `forward_steps=2000`;
- $\beta=0.1$;
- `proximal_steps=50`.

Experiment-specific controls remain beside their result folders as `config.json`.

## Assumptions and edge cases

- All canonical experiments use one pooled EFS plane.
- Source ownership does not affect the EFS field.
- Meaningful particle correspondence is unavailable in the random-source controls, so exact permutation matching is used.
- A simplex with $K$ parents has only $K-1$ free coefficients, regardless of the complete-source dimension $P\times D$.
- Shared lambda does not create a persistent force or constraint between generated particles.
- Synthetic source recovery cannot establish decoded ANewOmni validity.
- Exact dependency-free matching is capped at `P <= 16`.
- Historical experiment code is kept only where its equations or particle model differ materially from the canonical root code.

## Saved-output policy

Git tracks the evidence needed for inspection:

- `config.json` and retained experiment configurations;
- CSV metric tables;
- `summary.txt` and `summary.json`;
- PNG result figures;
- one small two-pass `pass1_reference.npz` provenance example.

Large `.npy`, generated `.npz`, and `__pycache__` files are ignored. They are expensive runtime state rather than readable evidence, and the final saved history exceeded GitHub's normal single-file limit. Re-run the owning experiment to recreate them locally.

No result directory is overwritten by the runners.

### Conclusion on EFS x AnewOMNI via H4.5

Exact vanilla EFS is computationally incompatible with the full ANewOmni corpus, while complete-peptide joint EFS introduces an impractical dimensional regime. 

H4.5 tested whether shared complete-source initialization could preserve source-level coherence while retaining tractable 11-dimensional slot-wise EFS. 

Across controlled synthetic diagnostics, shared initialization did not supply persistent coupling, did not reliably reconstruct held-out complete sources, and did not outperform direct interpolation consistently. 

H4.5 is therefore not supported as a practical mechanism for integrating vanilla EFS with ANewOmni. Future work would require either a scalable approximate interaction model, an explicitly coupled structured EFS formulation, or a substantially different representation.

### Main Sources

"DATA GENERATION WITHOUT FUNCTION ESTIMATION, Hadi Daneshmand1 and Ashkan Soleymani2, arXiv:2507.08239v1 [cs.LG] 11 Jul 2025"

"Programming Biomolecular Interactions with All-Atom Generative Model, KONG & al., doi: https://doi.org/10.64898/2026.03.12.711044"

# H4.5 Shared-Lambda EFS Experiment Ledger

## Purpose

This document records every major shared-lambda H4.5 experiment preserved in this folder, what it actually tested, what happened, and why it did or did not provide useful evidence.

`Shared barycentric coefficients provide synchronized initialization but no persistent coupling. In the tested pooled passive EFS systems, they neither supplied enough degrees of freedom to represent arbitrary complete sources nor caused backward replay to restore source identity lost at the terminal plane. The method therefore remains unsupported as a mechanism for coherent multi-component generation.`

## Original claim

For $K$ complete source objects, choose one coefficient vector:

$$
\lambda_a \geq 0,
\qquad
\sum_{a=1}^{K}\lambda_a=1.
$$

Use the same vector for every corresponding particle or residue slot:

$$
y_p^{(T)}=\sum_{a=1}^{K}\lambda_a x_{a,p}^{(T)}.
$$

The hope is that every generated part uses the same source recipe. If the sources are four faces, for example, the eyes, nose, mouth, and ears all use the same proportions of those four faces.

The important limitation was present from the start: shared lambda couples only the terminal initialization. It does not couple the particles during backward replay. Each generated particle follows its own passive EFS path.

The formal H4.5 claim also assumes that the source objects are complete, locally similar, topology matched, slot aligned, and in a compatible coordinate system. Most synthetic tests below did not satisfy all of those conditions.

## Meaning of the result labels

| Label                       | Meaning                                                                                                             |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Numerically invalid         | The forward or backward EFS calculation was not trustworthy.                                                        |
| Rejected model              | The code may have worked, but its particle model did not match the one-plane interpretation adopted later.          |
| Biased evidence             | The target was created using the same rule being evaluated, so success could not establish real generation ability. |
| Negative mechanism evidence | A controlled test contradicted a necessary behavior of the proposed mechanism.                                      |
| Mixed                       | Some cases improved and others worsened, with no reliable advantage.                                                |
| Calibration success         | The test established that the EFS field or inverse worked, but did not test source validity.                        |

## Condensed result table

| Test                                      | Main result                                                                                          | Status for H4.5                                  | Main reason                                                                                              |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| Structured families and rotations         | Shared lambda could preserve deliberately constructed family correlations                            | Biased and over-structured                       | The data already contained the type of joint structure the method was supposed to discover               |
| Separate field per slot                   | Shared was cleaner than independent after replay was repaired                                        | Rejected model for the current one-plane study   | Eight independent fields imposed slot identity and prevented ordinary cross-particle interactions        |
| First pooled single-plane witness         | Shared beat the independent control on coherence, but both outputs failed validity                   | Inconclusive                                     | One witness, no known target, and no decoded validity                                                    |
| Controlled barycentric targets            | Shared won 47 of 48 slot trials and 48 of 48 single-plane trials                                     | Algebraically positive but scientifically biased | The targets were generated with the exact shared-lambda rule                                             |
| Forward and vertex calibration            | Near-uniform terminal field and very accurate vertex replay were achieved                            | Calibration success                              | Exact vertex inversion does not test off-vertex generation                                               |
| Lambda fitted at the initial plane        | Initial interpolation did not commute reliably with nonlinear EFS transport                          | Rejected protocol                                | Coefficients describing the original plane need not describe the terminal plane                          |
| Neighbors selected at the terminal plane  | Terminal fitting became internally consistent                                                        | Rejected protocol                                | EFS was allowed to redefine which sources were semantically similar                                      |
| Held-out random complete sources          | EFS beat direct interpolation in 3 of 16 trials                                                      | Negative for this synthetic source model         | Four weights could not fit an arbitrary 40-dimensional random source, and EFS worsened the median result |
| One-particle 2D toy                       | EFS beat direct interpolation in 33 of 64 trials                                                     | Mixed local result                               | It removed the multi-particle coherence problem that H4.5 is meant to solve                              |
| Strong-contraction 2D toy                 | EFS beat direct interpolation in 2 of 16 trials                                                      | Negative                                         | More contraction produced a shell-like endpoint and did not repair the terminal seed                     |
| Two-pass target removal                   | Arrival was slightly worse than direct interpolation; pass-1 and pass-2 fields were almost identical | Negative for the tested reconstruction           | The shared terminal fit was already poor before replay; removing one source was not the main cause       |
| Homogeneous created-target replay         | Close terminal seeds stayed close in 2D even with accurate inversion                                 | Negative mechanism evidence                      | Passive backward replay did not restore identity after terminal compression                              |
| Final parent, sensitivity, and path tests | $K=128$ still fitted poorly, replay worsened, and lambda paths amplified steps strongly              | Stop result                                      | More parent capacity did not survive replay and the path map remained difficult to control               |

## 1. Structured synthetic families and rotations

### Tested idea

The first broad harness created several conformational families, deterministic orthogonal maps, retrieval descriptors, and cross-slot correlations. It compared shared and independent coefficients.

### Why it was not accepted

This was an extension before the base mechanism had been isolated. The synthetic generator deliberately created family-level cross-slot structure. A shared source coefficient was therefore aligned with the answer built into the data.

This design could show that the implementation recognized a known barycentric pattern. It could not show that shared lambda would preserve unknown structure in real ANewOmni latents. It also made the basic data flow difficult to inspect.

### Conclusion

Rejected as a base H4.5 experiment. It was not evidence that the EFS inverse failed.

This decision was preserved from the development log and is consolidated here.

## 2. Separate EFS field for each particle slot

### Tested idea

Each of the eight particle positions had its own EFS field. The same interpolation coefficient was used in all eight fields. The independent control used different coefficients per slot.

### Results

The first run was numerically invalid because its two tested vertex errors were $37.2\%$ and $43.5\%$ of the training scale.

After replay was improved, vertex errors fell to $0.768\%$ and $0.653\%$. Shared lambda then produced:

- raw joint validity of $11.11\%$, versus $0\%$ for the independent control;
- median coherence error of $0.0684$, versus $0.3753$ for the independent control.

Saved evidence:

- [invalid first run](../experiments/01_separate_fields/results/run_20260716T150646_249685Z_seed42/summary.txt)
- [repaired run](../experiments/01_separate_fields/results/run_20260716T152338_693394Z_seed42/summary.txt)

### Why the model was rejected

The later one-plane interpretation says that all particles belong to one ordinary EFS population. Source ownership is metadata, not a force rule. The synthetic particles also had no meaningful first, second, or eighth slot.

Using eight separate fields injected a correspondence rule that the synthetic data did not contain. It also prevented a particle in one slot from interacting with particles assigned to other slots. The favorable shared-versus-independent result therefore belonged to a different model.

### Conclusion

The repaired run was numerically positive, but it did not validate the adopted one-plane H4.5 mechanism. It also did not disprove the formal ordered-slot version on real topology-aligned data.

## 3. First pooled single-plane witness

### Tested idea

All $2048$ particles interacted in one four-dimensional EFS plane. There were $256$ complete sources with eight particles each. Four sources were mixed using either one shared coefficient vector or separate random vectors for each particle.

### Results

Shared lambda was cleaner than the independent control:

- template coherence error: $0.239$ shared, $0.594$ independent;
- pair-shape error: $0.562$ shared, $0.810$ independent;
- nearest-memory distance: $2.38$ shared, $3.79$ independent.

Backward residuals were small, around $2.5\times10^{-6}$. However, both complete outputs failed the run's raw-validity rule.

Saved evidence: [single-plane witness summary](../experiments/02_single_plane_witness/results/single_plane_20260716T170019_157940Z_seed42/summary.txt).

### Why it was inconclusive

This was one random witness, not a repeated recovery experiment. There was no known valid target and no decoder. Better coherence than an intentionally broken independent control does not establish that the shared output is valid.

### Conclusion

Useful proof that shared and independent initialization behave differently. Not proof that the shared result is a valid new source.

## 4. Controlled exact barycentric targets

### Tested idea

Clean targets were created directly from known source objects and a known shared lambda. The shared candidate and a cyclic independent control were replayed and compared with that target.

### Results

Across the frozen evaluation:

- separate-slot layout: shared won $47/48$ trials, with median relative RMSE $0.237$ versus $0.470$;
- single-plane layout: shared won $48/48$ trials, with median relative RMSE $0.165$ versus $0.453$.

Vertex replay was very accurate, generally below $0.2\%$ relative error in the single-plane runs.

Saved evidence:

- [pooled evaluation summary](../experiments/03_slot_single_calibration/results/h45_hpsearch_20260716T225729_891589Z/final_summary.txt)
- [slot seed 101 example](../experiments/03_slot_single_calibration/results/h45_hpsearch_20260716T225729_891589Z/final_runs/h45_target_20260717T003023_718442Z_slot_seed101/summary.txt)
- [single-plane seed 101 example](../experiments/03_slot_single_calibration/results/h45_hpsearch_20260716T225729_891589Z/final_runs/h45_target_20260717T003024_087685Z_single_seed101/summary.txt)

### Why this did not validate H4.5

The target was made with the same shared-lambda algebra used by the tested method. The independent control was deliberately prevented from following that algebra. Shared lambda should win this comparison if the implementation is correct.

This result verifies the interpolation code and shows that the reverse field does not completely erase an easy, constructed shared signal. It does not tell us whether real complete sources lie near a shared barycentric surface.

The slot-layout branch also inherited the rejected separate-field assumption.

### Conclusion

Positive algebra check, biased scientific test.

## 5. Radial gate correction and hyperparameter calibration

### Tested idea

The search selected forward parameters using terminal radial shape, angular shape, covariance, scale, and vertex replay. An early rule treated a radial CDF gap above $0.15$ as a hard failure.

### Results

The early seed-42 and seed-50 runs were stopped because their shell-like endpoints had radial gaps near $0.19$ and $0.20$. That rule was later corrected because a finite sample can differ from an ideal uniform ball while its inverse remains usable.

The later one-plane search selected:

- $\epsilon=0.01$;
- $s=3$;
- $\gamma=0.001$;
- transport time $\tau=2$;
- $2000$ forward steps;
- $\beta=0.1$;
- $50$ proximal steps.

Its worst radial CDF gap was $0.0381$ and vertex p95 relative RMSE was $0.154\%$.

Saved evidence:

- [seed-50 early gate](../experiments/03_slot_single_calibration/results/early_target_recovery/h45_target_20260716T202628_054257Z_seed50/summary.txt)
- [one-plane search summary](../experiments/03_slot_single_calibration/results/one_plane_search_20260717T183527_965392Z/search_summary.txt)

### Correct interpretation

The radial CDF gap is descriptive. It measures the largest vertical distance between the observed radial CDF and the CDF of a fitted uniform ball. It is not a KL divergence and no p-value is used here.

Vertex replay asks a narrower question: if an exact memory particle is placed at its own terminal position, does backward replay return it to its original position? A low vertex error proves local inversion at known endpoints. It says nothing about whether a new interpolated point is valid.

### Conclusion

Calibration succeeded. This established that later negative target results were not caused by a generally broken inverse. It did not establish H4.5 generation.

## Coordinate-frame protocol variants

Two intermediate protocols exposed a logical issue that is separate from EFS numerical accuracy.

### Lambda fitted at the initial data plane

The target and its neighbors were selected at the initial plane, and lambda was fitted there. The same coefficients were later applied to the parents' terminal positions.

This tested whether barycentric interpolation commutes with nonlinear EFS transport:

$$
F\!\left(\sum_a\lambda_a x_a\right)
\stackrel{?}{=}
\sum_a\lambda_a F(x_a).
$$

There is no general reason for that equality to hold. This protocol therefore placed the backward seeds using coefficients that described the wrong coordinate frame. Its failure was a protocol failure, not evidence that terminal shared-lambda seeding itself was impossible.

### Neighbors selected at the terminal plane

A later variant transported the held-out target passively, selected its neighbors at the terminal plane, and fitted lambda there. This made the terminal fit internally consistent.

It failed as a generation model because source meaning lives in the original data. Allowing EFS to choose new neighbors after transport can replace semantically related sources with unrelated terminal neighbors. The fit may be geometrically good while answering the wrong biological question.

### Adopted correction

The two-pass protocol kept the parent identities selected at the initial plane but fitted lambda from those same parents at the terminal plane. That corrected both frame errors. Its negative result is therefore more informative than either intermediate protocol.

The former development log's coordinate-frame comparison is consolidated in this section.

## 6. Held-out randomly grouped complete sources

### Tested idea

Particles were drawn independently from a chaotic global distribution and then randomly grouped into sources. One held-out source contained $P=10$ particles in $D=4$, so the complete source occupied $40$ numerical dimensions. Four nearby sources supplied one nonnegative shared lambda.

### Results

The forward field and inverse were numerically good:

- radial CDF gap: $0.0448$;
- vertex median relative RMSE: $0.174\%$;
- vertex maximum relative RMSE: $0.630\%$.

The source reconstruction was not good:

- median terminal shared-lambda fit RMSE: $0.419$;
- median direct target RMSE: $0.419$;
- median EFS target RMSE: $0.442$;
- median EFS/direct ratio: $1.103$;
- EFS beat direct interpolation in only $3/16$ trials.

Saved evidence: [held-out reconstruction summary](../experiments/04_heldout_reconstruction/results/one_plane_h45_20260717T200730_842646Z_seed4652/summary.txt).

### Why it failed

Four convex weights have only three free degrees of freedom. Geometrically, four parents define a tetrahedron, even when that tetrahedron sits in a 40-dimensional space. A random 40-dimensional source is very unlikely to lie close to it.

Random grouping created no real source-level relationship for shared lambda to preserve. The experiment deliberately removed the topology, slot alignment, and local source coherence required by formal H4.5.

The direct interpolation was already far from the target, and EFS made the median result worse. Accurate vertex replay could not repair a bad off-vertex terminal seed.

### Conclusion

Negative for arbitrary randomly grouped sources. Not a complete falsification of formal H4.5 on meaningful, aligned real sources.

## 7. One-particle, two-dimensional interpolation toy

### Tested idea

Set one source equal to one particle. Fit four neighbors to held-out points in two dimensions, then compare direct interpolation with EFS replay.

### Results

- median terminal fit RMSE: $0.00631$;
- EFS beat direct interpolation in $33/64$ cases;
- median direct RMSE: $0.00880$;
- median EFS RMSE: $0.00814$;
- median EFS/direct ratio: $0.997$.

The maximum ratio was $4.78$, but ratios become unstable when the direct error in the denominator is already extremely small.

Saved evidence: [one-particle summary](../experiments/05_point_interpolation/results/one_plane_h45_20260717T230444_943530Z_seed501/summary.txt).

### Why it did not settle H4.5

Four parents can geometrically surround many points in two dimensions, so the terminal fit was much easier than the 40-dimensional source fit. More importantly, one particle has no internal source coherence. Shared lambda and per-particle lambda are the same concept when $P=1$.

### Conclusion

Local EFS interpolation can work when the terminal seed is already close. The result was mixed and did not test the multi-particle hypothesis.

## 8. Stronger contraction and more forward steps

### Tested idea

Use a two-dimensional field with $4096$ particles, $6000$ forward steps, and stronger contraction to determine whether longer transport would make source recovery easier.

### Results

- terminal radial CDF gap: $0.366$;
- terminal radius ratio: $0.388$;
- vertex maximum relative RMSE: $0.826\%$;
- median terminal fit RMSE: $0.217$;
- median direct target RMSE: $0.356$;
- median EFS target RMSE: $0.422$;
- median EFS/direct ratio: $1.203$;
- EFS beat direct interpolation in $2/16$ trials.

Saved evidence: [strong-contraction summary](../experiments/06_strong_contraction/results/one_plane_h45_20260717T230222_213327Z_seed502/summary.txt).

### Why it failed

The inverse still replayed vertices accurately, so this was not a general backward explosion. The terminal field became strongly shell-like, and the shared terminal fit remained poor. Longer contraction did not create missing source-level geometry and did not make the EFS result better than direct interpolation.

### Conclusion

Negative. Increasing transport length alone was not a solution.

## 9. Two-pass target removal

### Tested idea

Pass 1 contained the target and selected its four neighbors in the original data plane. Lambda was fitted to those same parents at the terminal plane. Pass 2 restarted from the same initial data after removing the target. The terminal interpolation was then replayed in the target-free field.

This separated two possible failures:

1. removing the target changed the field too much;
2. the shared terminal seed was already a poor representation of the target.

### Results

- pass-1 lambda fit RMSE: $0.5464$;
- mean marginal $\mathrm{KL}(\text{pass 1}\Vert\text{pass 2})$: $1.76\times10^{-4}$;
- mean retained-particle terminal displacement: $0.00255$;
- pass-2 terminal start error: $0.5463$;
- arrival RMSE: $0.6767$;
- direct interpolation RMSE: $0.6632$;
- EFS/direct ratio: $1.019$;
- vertex maximum relative RMSE: $0.371\%$.

Saved evidence: [two-pass summary](../experiments/07_two_pass_reconstruction/results/two_pass_h45_20260718T001147_903684Z_seed42/summary.txt).

### Why it failed

Removing one source barely changed the large field. The pass-2 terminal error was almost identical to the pass-1 lambda-fit error. The main failure therefore existed before backward replay.

Again, four shared weights were asked to approximate an arbitrary multi-particle random source. The replay did not explode, but it also did not improve on direct interpolation.

The marginal KL values only compare coordinate histograms between the two fields. They do not measure source validity.

### Conclusion

Negative for the tested reconstruction. The target-removal field shift was not the main explanation.

## 10. Homogeneous created-target local replay

### Tested idea

This diagnostic removed source fitting and lambda fitting entirely. It used a constant-density thick spiral tube rather than a cloud with a large density peak.

Pairs of exact memory vertices supplied known original destinations. At the terminal plane, each pair was moved toward itself by a fraction $\alpha$. Backward replay then tested whether nearby terminal seeds recovered distinct original identities.

The corrected full run is the authoritative result. The earlier full run contains valid raw beta rows, but selected the wrong best beta because its selection rule favored under-converged runs with fewer recorded collapse labels.

### Results

For $D=2$ with $\beta=0.2$:

- vertex error: $0.126\%$ of initial plane RMS;
- displaced-seed error: $0.250\%$ of initial plane RMS;
- terminal radial CDF gap: $0.0676$;
- terminal covariance ratio: $0.998$;
- safe tested $\alpha$: $0$ under the strict pair-relative rule;
- all six pairs collapsed at both $\alpha=0.40$ and $\alpha=0.49$;
- maximum replay residual: $9.45\times10^{-9}$.

The expected terminal separation ratios were $0.8$, $0.5$, $0.2$, and $0.02$. The arrival ratios were approximately $0.8005$, $0.4989$, $0.1993$, and $0.0199$. The backward map preserved the compressed separation instead of restoring the vertices' original separation.

For $D=4$ with $\beta=0.2$:

- vertex error: $0.221\%$;
- displaced-seed error: $1.94\%$;
- safe tested $\alpha$: $0.25$;
- all six pairs collapsed at $\alpha=0.49$.

The $D=4$ terminal radial CDF gap was $0.307$, so it is a less clean geometry control than the two-dimensional result.

Results for $\beta=0.2$ and $\beta=0.4$ were almost identical after convergence. A larger backward learning rate solved the same fixed point faster but did not change where the field sent the seed.

Saved evidence:

- [diagnostic README](../experiments/08_local_replay_stability/README.md)
- [corrected full-run summary](../experiments/08_local_replay_stability/results/local_replay_20260718T101011_120468Z_seed45101/summary.txt)
- [corrected metrics](../experiments/08_local_replay_stability/results/local_replay_20260718T101011_120468Z_seed45101/metrics.csv)

### Why this is important

The clean two-dimensional result rules out two simple explanations:

- the earlier collision was not caused only by a huge density peak;
- it was not caused by an unconverged proximal solver.

Passive generated particles do not repel one another and do not know their source identity. If shared interpolation places two of them almost on top of each other, the backward field has no extra rule telling them which identities to recover.

### Conclusion

This is the clearest negative mechanism evidence. It does not prove that every shared-lambda seed fails, but it shows that vanilla passive backward replay cannot be expected to undo terminal identity compression by itself.

## 11. Final parent-capacity, local-sensitivity, and path-continuity tests

### Tested idea

The final suite reused one target-free field to ask whether the earlier failure came from too few parents or from an uncontrollable inverse map. It swept $K\in\{4,8,16,32,64,128\}$, perturbed exact terminal vertices, and replayed smooth two-parent lambda paths.

### Results

- median terminal fit improved by $37.25\%$ from $K=4$ to $K=128$;
- the $K=128$ terminal RMSE remained high at $0.4407$;
- replay error worsened by $15.49\%$ rather than improving;
- median replay error at $K=128$ was $1.151$ times the direct same-lambda baseline;
- the small local perturbations produced no recorded identity changes in this run;
- smooth lambda paths had p95 step amplification of $55.59$.

Saved evidence:

- [final summary](../experiments/09_final_diagnostics/results/h45_final_diagnostics_20260718T120738_714563Z/summary.txt)
- [parent sweep](../experiments/09_final_diagnostics/results/h45_final_diagnostics_20260718T120738_714563Z/parent_sweep.csv)
- [local sensitivity](../experiments/09_final_diagnostics/results/h45_final_diagnostics_20260718T120738_714563Z/du_sensitivity.csv)
- [lambda paths](../experiments/09_final_diagnostics/results/h45_final_diagnostics_20260718T120738_714563Z/lambda_path.csv)

### Conclusion

More parents increased convex capacity but did not make a random complete source representable enough, and the improvement did not survive backward replay. The local vertex test was calm at the tested scales, but whole-source lambda paths still exhibited large step amplification. This satisfies the declared stop rule for the passive shared-lambda route.

## Combined scientific conclusion

### What worked

- The NumPy EFS field remained finite in the accepted runs.
- Forward EFS could produce a reasonably round, well-scaled endpoint.
- Exact memory vertices usually replayed with errors below $1\%$ of field scale.
- Shared lambda consistently beat deliberately independent coefficients when the target itself was constructed from a shared lambda.
- Removing one source from a field containing thousands of particles changed that field only slightly in the measured two-pass run.

### What failed

- Shared lambda did not consistently beat direct interpolation on held-out randomly grouped sources.
- A small shared coefficient vector could not fit arbitrary high-dimensional complete sources.
- Stronger contraction and more forward steps did not repair a poor terminal seed.
- The passive backward map preserved severe terminal pair compression in the clean two-dimensional local test.
- Increasing to $128$ parents still left a large terminal fit error and worsened replay error.
- Smooth source-level lambda paths produced strongly amplified output steps.
- No experiment measured decoded molecular validity.

### Current verdict

H4.5 is **not validated** by these experiments.

The current one-plane, passive shared-lambda mechanism should be considered **unsupported as a standalone generator**. The strongest evidence says that shared lambda is an initialization constraint, not a persistent source-level coupling. Once two generated particles become indistinguishable at the terminal plane, vanilla passive replay has no source-aware mechanism that forces them apart.

This is not a complete mathematical disproof of the formal H4.5 claim. The formal claim requires real complete sources with meaningful correspondence, topology, locality, and a decoder-based validity check. Those conditions were not available in the random-source CPU experiments.

The accurate vertex results also matter: the project did not fail because EFS was simply broken. It failed at the harder step between exact endpoint inversion and coherent off-vertex source generation.

## Practical stop decision

Further tuning of $\beta$, forward-step count, or particle density is not justified by the current evidence alone. Those parameters can improve numerical convergence or endpoint geometry, but they do not add the missing source-level information.

A future H4.5 test would need all of the following before the path should be promoted again:

1. meaningful complete sources rather than random particle grouping;
2. a declared correspondence or permutation-invariant matching rule that reflects the actual representation;
3. terminal parent sets that fit the complete source closely enough before replay;
4. direct-interpolation and retrieval baselines;
5. decoded whole-source validity, not only particle-level distance;
6. a check that close terminal particles retain distinct, correct identities.

Until those conditions are available, the defensible result is a paused or demoted hypothesis, not a claim of successful ANewOmni generation.

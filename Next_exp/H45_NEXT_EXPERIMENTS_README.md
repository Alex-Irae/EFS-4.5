# H4.5: Next Experiments

## Goal

The remaining experiments should separate two possible failure causes:

1. **Representation bottleneck:** too few parents make the shared-lambda terminal seed unable to approximate a complete target.
2. **Inverse-map sensitivity:** even a good terminal seed is mapped far away because small displacements in `du` can correspond to very different regions in `d1`.

Do not spend more time on generic EFS hyperparameter sweeps unless one of these tests indicates that the mechanism is still viable.

---

## Experiment 1: Parent-count sweep

### Question

Does H4.5 fail mainly because `K=4` gives only three free barycentric degrees of freedom for a complete source with `P*D` dimensions?

### Protocol

Reuse the same saved forward histories whenever possible. Increasing `K` does **not** increase the EFS field size because all parents already belong to the existing memory.

Test:

```text
K = 4, 8, 16, 32, 64, 128
```

Cap `K` below the available source count. For each target:

1. Select and rank complete-source neighbors in `d1` once.
2. For each `K`, retain the first `K` parent identities and their saved particle matching.
3. Fit one nonnegative shared lambda in `du`:

   ```text
   lambda >= 0
   sum(lambda) = 1
   ```

4. Construct the terminal candidate from the same lambda for every particle.
5. Replay it through the unchanged EFS backward history.
6. Repeat over the same targets and seeds for every `K`.

### Metrics

Save at least:

```text
terminal_fit_rmse
replay_target_rmse
direct_same_lambda_rmse
replay_over_terminal = replay_target_rmse / terminal_fit_rmse
replay_over_direct = replay_target_rmse / direct_same_lambda_rmse
lambda_max
lambda_entropy
effective_parent_count = 1 / sum(lambda**2)
fit_iterations
fit_runtime
replay_runtime
```

The direct baseline uses the terminal-fitted lambda on the original parent coordinates. It is only a baseline, not the intended H4.5 protocol.

### Plots

Create:

- median terminal error versus `K`;
- median replay error versus `K`;
- replay/terminal amplification versus `K`;
- effective parent count versus available `K`;
- per-target terminal and replay errors, so improvements are not hidden by medians.

### Interpretation

| Observation | Meaning |
|---|---|
| Terminal fit remains poor as `K` increases | A shared convex simplex cannot represent the target well enough. |
| Terminal fit improves but replay stays poor | The EFS inverse is the primary bottleneck. |
| Terminal and replay errors both improve | `K=4` was too restrictive and H4.5 deserves further testing. |
| Good results require many active parents | Generation may work, but the lambda explanation becomes diffuse. |
| Large `K` is available but lambda remains sparse | Best case: more representational capacity without losing simple provenance. |

---

## Experiment 2: Local `du` sensitivity around real particles

### Question

Can a very small displacement around a valid terminal particle send the replayed sample to a distant region of `d1`?

This directly tests the Swiss-roll concern.

### Protocol

For several exact memory particles with known endpoints:

```text
x0 -> xT
```

construct perturbed terminal seeds:

```text
yT = xT + delta
```

Use perturbation magnitudes relative to the terminal nearest-neighbor distance:

```text
1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1
```

Test several directions:

1. random unit directions;
2. toward the nearest terminal particle;
3. toward a terminal particle that is close in `du` but far in `d1`;
4. directions lying in a local parent interpolation span;
5. radial and tangential directions when the terminal geometry permits it.

Replay every perturbed seed with the unchanged field.

### Metrics

```text
terminal_delta = ||yT - xT||
output_delta = ||Backward(yT) - x0||
amplification = output_delta / terminal_delta
nearest_d1_identity_before
nearest_d1_identity_after
identity_changed
output_distance_to_original_source
```

For each perturbation magnitude, report median, p90, p95, and maximum amplification. A few extreme cases matter because one unstable particle can invalidate a complete generated source.

### Important control

Also perturb exact vertices in directions that remain inside a visibly local terminal neighborhood. This separates ordinary smooth amplification from crossing a terminal branch boundary.

### Interpretation

- Smooth, moderate amplification supports further H4.5 work.
- Large amplification only near identifiable boundaries may allow conservative lambda restrictions.
- Large or discontinuous amplification throughout the terminal field is strong negative evidence for controllable shared-lambda generation.

---

## Experiment 3: Shared-lambda path continuity

### Question

Does a smooth change in one source-level lambda produce a smooth generated complete source?

### Protocol

Choose two or more fixed parent sets. For the two-parent case:

```text
lambda(t) = [1-t, t]
t = 0.00, 0.01, ..., 1.00
```

For every `t`:

1. build all particle seeds with the same lambda;
2. replay the complete candidate;
3. compare consecutive outputs.

Repeat with parent pairs that are:

- close in `d1`;
- close in `du`;
- close in `du` but far in `d1`;
- ordinary nearest-neighbor source pairs.

### Metrics

```text
terminal_step_norm(t)
output_step_norm(t)
step_amplification(t)
nearest_source_identity(t)
pairwise_relation_change(t)
maximum_particle_jump(t)
```

Plot output displacement against `t`, both for the whole source and for each particle. The main failure signature is one particle jumping to a different region while the others remain smooth.

---

## Optional Experiment 4: Small joint-vector control

This is not an ANewOmni-scale test. It only isolates whether the main failure comes from splitting one object into independently replayed particles.

Using a small synthetic complete-source dataset:

1. run the current particle-wise shared-lambda replay;
2. flatten each complete source into one vector and run one joint EFS system;
3. use identical parent identities and comparable terminal interpolation;
4. compare complete-source reconstruction and lambda-path continuity.

If joint-vector replay is stable while particle-wise replay is not, the main issue is the factorized particle representation rather than shared lambda itself.

Treat this as lower priority because high-dimensional EFS may introduce separate numerical problems.

---

## Recommended execution order

```text
1. Parent-count sweep using existing histories
2. Local du sensitivity around real vertices
3. Shared-lambda path continuity
4. Optional small joint-vector control
```

The first experiment checks whether the previous setup was underparameterized. The next two test whether the inverse geometry is intrinsically unsuitable for precise source-level control.

---

## Suggested result layout

```text
results/
  parent_sweep_<timestamp>/
    config.json
    trials.csv
    summary.json
    error_vs_k.png
    amplification_vs_k.png
    effective_k.png

  du_sensitivity_<timestamp>/
    config.json
    trials.csv
    summary.json
    amplification_by_scale.png
    worst_cases/

  lambda_path_<timestamp>/
    config.json
    paths.csv
    summary.json
    source_displacement.png
    particle_displacement.png
```

Keep target IDs, parent IDs, particle permutations, lambda vectors, seeds, EFS configuration, and random seeds in every saved run.

---

## Decision rule

Continue H4.5 only if increasing `K` produces substantially better terminal fits **and** those improvements survive backward replay with reasonably smooth lambda paths.

Stop the passive shared-lambda route if either result is consistent across targets:

1. terminal fits become good but replay errors remain large;
2. tiny terminal or lambda perturbations frequently cause large particle-level branch changes.

That outcome would mean the shared-lambda interface is reasonable, but the EFS inverse geometry is not controllable enough to support it.

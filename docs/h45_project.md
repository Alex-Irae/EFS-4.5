# EFS x ANewOmni: Complete Hypothesis 4.5 Project

Status: authoritative project document before ANewOmni integration.

Companion document: [broad_hypotheses.md](broad_hypotheses.md), preserved as the original hypothesis notebook.

Primary sources:

- [Data Generation without Function Estimation](https://arxiv.org/abs/2507.08239)
- [ANewOmni project and paper resources](https://github.com/bytedance/AnewOmni)

---

## 1. Executive summary

### 1.1 Goal

The project tests whether Estimation-Free Sampling (EFS) can replace ANewOmni's learned latent diffusion sampler while retaining useful peptide generation and exposing an auditable relationship between each generated latent and the empirical training complexes used during sampling.

The intended pipeline is:

    target and topology definition
    -> ANewOmni encoder and latent dataset
    -> bounded local empirical memory
    -> EFS latent generation
    -> ANewOmni all-atom decoder
    -> raw validity, novelty, decoder-correction, and provenance analysis

The first contribution is not biological causal attribution. The first defensible interpretation claim is pathwise force provenance, supported by explicit initialization records. Stronger algorithmic-influence language requires complete counterfactual refits after data-removal interventions.

### 1.2 Central scientific problem

ANewOmni represents a peptide as an ordered set of residue-block latents. Each residue has:

- an invariant feature latent ($Z_h\in\mathbb{R}^8$\);
- an equivariant coordinate latent \($Z_x\in\mathbb{R}^3$\).

A peptide of length \($L$\) therefore has a latent tensor with shape ($L\times11$).

Vanilla block-level EFS can make each residue latent individually plausible while failing to preserve the relationships that make the complete set a peptide. Hypothesis 4.5 attempts to preserve those relationships through complete source complexes and a shared barycentric initialization.

### 1.3 Co-primary unchanged-embedding models

Two models remain co-primary until controlled evidence separates them:

1. **Hypothesis 4.5 block EFS:** one EFS system per ordered residue slot, initialized from the same complete source complexes and the same barycentric vector \($\lambda$\).
2. **Joint-vector EFS:** flatten the complete \($L\times11$\) binder tensor into one \(11L\)-dimensional EFS particle and initialize it from complete-complex barycenters.

Flattening does not retrain or rewrite ANewOmni's embedding. It only changes what counts as one EFS particle.

### 1.4 Current status

- A paper-faithful NumPy reference core exists.
- Eleven unit tests pass when launched from the EFS directory.
- A two-dimensional Gaussian-mixture smoke test behaves plausibly.
- A constructed set-coherence toy shows that independent initialization fails while local complete-source barycentric initialization succeeds.
- The current toy uses an oracle family label for local retrieval and must be replaced with observable-distance retrieval before it counts as decisive evidence.
- No ANewOmni checkpoint or processed latent dataset is present locally, so real latent extraction is blocked.
- No further integration code is approved until the incremental code audit and exact scientific contract are complete.

---

## 2. Project claim, novelty, and non-claims

### 2.1 Research claim under test

EFS exposes explicit interaction terms between a generated latent and empirical training particles. Coupling EFS to ANewOmni may therefore produce an auditable generative pipeline whose training-data interactions can be inspected and tested by intervention.

This is a hypothesis, not an established property of the decoded peptide.

### 2.2 Possible contribution

ANewOmni supplies:

- a per-block all-atom VAE;
- a joint equivariant decoder;
- programmable topology and coordinate prompts;
- broad biomolecular and non-canonical representation;
- a released code path for latent generation and decoding.

EFS supplies:

- an empirical particle system rather than a learned score field;
- cached identities and trajectories for training particles;
- explicit per-particle terms in the backward field;
- paper-supported terminal interpolation.

The contribution would be to determine whether these explicit terms remain stable and useful after:

- finite high-dimensional EFS transport;
- local source retrieval;
- barycentric initialization;
- metric choices;
- ANewOmni decoding;
- counterfactual removal of source information.

### 2.3 Claims not made

The project does not initially claim:

- that force mass is causal attribution;
- that barycentric weights are percentages of a final peptide;
- that EFS forces are molecular or physical forces;
- that pinned pocket latents implement \(p(\text{binder}\mid\text{pocket})\);
- that the ANewOmni decoder projects arbitrary latents onto the valid-peptide manifold;
- that retrieval-conditioned EFS samples an exact conditional distribution;
- that the EFS asymptotic theorem certifies one finite run;
- that an output passing rejection filters represents a high-yield generator;
- that five million complexes can be used in one exact vanilla EFS system.

---

## 3. What ANewOmni currently does

The following points come from the released implementation rather than only the paper.

### 3.1 Encoder, diffusion, and decoder roles

ANewOmni has three relevant stages:

1. The VAE encoder maps a biomolecular complex into per-block feature and coordinate latents.
2. The latent diffusion model jointly generates all requested block latents.
3. The VAE decoder converts the complete latent set into block identities and all-atom coordinates.

EFS is intended to replace stage 2 only.

### 3.2 Peptide length is selected outside the denoiser

The generation API samples a peptide length before model inference. A linear-peptide template creates that many placeholder blocks and marks them for generation.

Removing diffusion therefore does not remove the basic length mechanism for fixed-topology linear peptides. Variable topology remains a separate discrete problem.

### 3.3 Diffusion supplies joint coherence

ANewOmni's epsilon network receives every block in a complex and builds graph edges across the complete sample. It updates the latent set jointly.

Independent block EFS instead models a product of marginal slot distributions:

$p(x_1,\ldots,x_L)
\quad\longrightarrow\quad
\prod_{\ell=1}^{L}p_\ell(x_\ell)$

These are not equivalent. A seed-initialization rule can preserve some correlation, but it does not turn independent reverse maps into one joint generative model.

Relevant implementation:

- `AnewOmni/models/LDM/diffusion/dpm_full.py` in the upstream ANewOmni repository
- `AnewOmni/models/LDM/ldm_clean.py` in the upstream ANewOmni repository

### 3.4 Features and coordinates are treated separately by diffusion

ANewOmni diffusion does not apply one Euclidean interaction to concatenated \(Z_h\) and \(Z_x\). It uses:

- separate feature and coordinate noise transitions;
- separate \(H\) and \(X\) noise predictions;
- an equivariant coordinate network;
- separate feature and coordinate losses;
- an explicit feature-loss weight chosen to balance eight feature channels against three coordinate channels.

Consequently, the fact that raw \(Z_h\) and \(Z_x\) work in diffusion does not prove that one unweighted Euclidean EFS potential will use them appropriately.

Raw ANewOmni scaling remains the first EFS baseline. Alternative metrics are controlled ablations, not automatic corrections.

### 3.5 Graph prompts are part of the denoiser

Topology and coordinate prompts are converted into adapter conditions and injected into the epsilon network at each diffusion step. Classifier-free guidance combines conditional and unconditional predictions.

This pathway disappears when the epsilon network is removed. EFS must not be described as inheriting graph-prompt conditioning automatically.

### 3.6 Pocket and binder roles are learned

The encoder uses the generation mask to distinguish context from generated blocks and restricts information flow accordingly. Pocket and binder blocks may share tensor shapes, but they are not exchangeable empirical particles.

### 3.7 The decoder is joint and corrective, but not a projector

The decoder predicts block identity and all-atom coordinates using a graph over the whole complex. It includes clash avoidance and covalent-bond approach behavior.

It may repair moderate EFS inconsistencies. That makes decoder-delegated correction a necessary baseline. It does not establish that arbitrary EFS latents will be projected onto a valid peptide.

Decoder correction must be measured as a transformation separate from initialization and EFS transport.

### 3.8 Diffusion likelihood ranking is not reusable

ANewOmni's released exact-likelihood calculation differentiates the diffusion velocity field. Once diffusion is removed, that score is unavailable.

Remaining validation options include:

- VAE reconstruction behavior;
- the released confidence model where applicable;
- deterministic topology and geometry checks;
- external physics or structure filters;
- an explicitly new EFS-specific ranking method, if later justified.

---

## 4. EFS method used by this project

### 4.1 Training particles

Let the empirical training data be:

$x_1^{(0)},\ldots,x_n^{(0)}\in\mathbb{R}^{d}$

One row is one EFS particle. The scientific meaning of one row is an integration decision:

- one residue-slot latent for block EFS;
- one complete flattened peptide for joint-vector EFS.

### 4.2 Attractive-repulsive potential

For a displacement \(z=x_i-x_j\), the smoothed EFS potential is:

$W_{\epsilon}^{(s)}(z)\frac{\|z\|^2}{2}
+
\frac{1}
{s(\|z\|^2+\epsilon)^{s/2}}$

For \(s=0\), the repulsive component is interpreted through its logarithmic limit.

The implemented gradient is:

$\nabla W_{\epsilon}^{(s)}(z)z\frac{z}
{(\|z\|^2+\epsilon)^{s/2+1}}$

The quadratic term attracts particles globally. The inverse-power term prevents collapse.

### 4.3 Forward transport

At forward frame \(j\):

$x_i^{(j+1)}x_i^{(j)}\frac{\gamma}{n-1}
\sum_{a\ne i}
\nabla W_{\epsilon}^{(s)}
\left(x_i^{(j)}-x_a^{(j)}\right)$

Every frame is cached with particle identity retained. The cached frames are the empirical vector field used during backward generation.

The forward cost is:

$O(k n^2 d)$

and the cached-history storage is:

$O(k n d)$

### 4.4 Terminal distribution and seed construction

For the common theoretical choice \(s=d-2\), the asymptotic minimizer is a uniform ball. Other parameter regimes can give a sphere.

The paper's Algorithm 2 fits a center and mean radius to the terminal particles and draws a seed on the fitted sphere. It also explicitly permits interpolation between terminal training particles.

The project tests:

- fixed-radius uniform sphere seeds;
- two-source barycentric seeds;
- multi-source simplex seeds;
- complete-complex local barycentric seeds.

### 4.5 Backward proximal replay

Given a seed \($y^{(k)}$), backward replay traverses cached frames in reverse.

At frame \(j\), set an anchor \($y^{(j)}$), initialize \($v_0=y^{(j)}$\), and repeat:

$F_j(v_t)\frac{\gamma}{n}
\sum_{i=1}^{n}
\nabla W_{\epsilon}^{(s)}
\left(v_t-x_i^{(j)}\right)$

$\Delta_t=v_t-y^{(j)}-F_j(v_t)$

$v_{t+1}=v_t-\beta\Delta_t$

After \(T\) inner iterations:

$y^{(j-1)}=v_T$

The process continues to \($y^{(0)}$), the generated sample.

### 4.6 Meaning of the theorem

The EFS generation result is asymptotic. Its ideal analysis involves:

- \($n\to\infty$);
- a continuous empirical limit;
- small forward steps or continuous time;
- sufficiently long forward transport;
- the theoretically selected \(s\);
- accurate backward inversion.

The actual project uses finite \($n,\gamma,k,\epsilon,\beta,T$\) and finite precision. Every finite result therefore requires diagnostics and sensitivity analysis.

---

## 5. Representation geometry

### 5.1 Biological meaning is not required

EFS can operate on any reversible vector representation. It does not need to know that one coordinate represents chemistry and another represents space.

However, EFS uses the Euclidean geometry created by that representation. Relative distance is not independent of coordinate convention.

### 5.2 What transformations EFS ignores

A common translation \(t\) and common rotation \(R\) preserve pairwise distances:

$\left\|(R x_i+t)-(R x_j+t)\right\|=\left\|R(x_i-x_j)\right\|=\left\|x_i-x_j\right\|$

Therefore:

- absolute origin is irrelevant;
- one common rotation of the complete dataset is irrelevant;
- one common orthogonal change of basis preserves the EFS field.

### 5.3 Independently oriented complexes

For independently chosen rotations:

$\left\|R_i x_i-R_j x_j\right\|$

generally differs from \($|x_i-x_j\|$\) when \($R_i\ne R_j$).

Ideal EFS can still model the resulting distribution. The problem is practical:

- identical structures can look far apart;
- local retrieval wastes capacity on orientation;
- barycentric interpolation can average camera orientations instead of conformations;
- finite sample complexity increases;
- provenance rankings can reflect representation nuisance rather than structural similarity.

For a fixed target or aligned target family, the first solution is:

1. choose matched pocket anchors;
2. calculate a proper Kabsch rotation with determinant \(+1\);
3. align the pocket and binder together;
4. store rotation, translation, anchors, transform ID, and residual;
5. flag poor or ambiguous alignments;
6. apply and verify the inverse transformation before final output.

Alignment is variance reduction for finite local generation. It is not a claim that EFS requires biological coordinates.

### 5.4 Feature and coordinate metric

A complete peptide particle can be written:

$X=
\left[
Z_h^{(1)},Z_x^{(1)},\ldots,Z_h^{(L)},Z_x^{(L)}
\right]
\in\mathbb{R}^{11L}$

Raw Euclidean distance assumes that one unit in \(Z_h\) is commensurate with one unit in normalized \(Z_x\).

For:

$A=(h=0,x=0),\quad B=(1,0),\quad C=(0,1)$

the original distances satisfy:

$d(A,B)=d(A,C)=1$

Multiplying the feature coordinate by 10 changes:

$d(A,B)=10,\qquad d(A,C)=1$

This changes:

- nearest neighbors;
- source retrieval;
- attraction and repulsion;
- terminal geometry;
- backward trajectories;
- force-provenance rankings.

The first experiments compare:

1. raw ANewOmni scaling;
2. one global feature scale and one isotropic coordinate scale;
3. per-feature \(Z_h\) standardization with one isotropic \(Z_x\) scale.

Statistics are fitted on training data only, stored with the experiment, and inverted before decoding.

Metric scaling does not necessarily invalidate ideal asymptotic EFS. It is an empirical and interpretive issue for finite runs.

### 5.5 Translation and scale used by ANewOmni

ANewOmni normalizes coordinate latents approximately as:

$Z_x^{\mathrm{norm}}\frac{Z_x-\text{pocket center}}{10}$

This removes translation and sets a numerical coordinate scale. It does not remove independent rotation or calibrate \(Z_h\) against \(Z_x\).

### 5.6 Ball, sphere, and effective dimension

For \(s=d-2\), the theorem describes a uniform \(d\)-ball. Algorithm 2 draws from a fitted sphere and invokes high-dimensional concentration near the ball boundary.

This approximation can fail when effective dimension is low, even if the ambient flattened dimension \(11L\) is large. The latent export stage must therefore estimate effective dimension and terminal radial behavior.

### 5.7 High-dimensional numerical sharpness

With \(s=d-2\):

$\frac{s}{2}+1=\frac{d}{2}$

For a ten-residue joint particle:

d=110,

so the inverse-power exponent is 55:

$(\|z\|^2+\epsilon)^{-55}$
Values slightly below or above one can produce enormous or negligible forces. This connects representation scaling directly to numerical stability.

---

## 6. Hypothesis 4.5

### 6.1 Set-coherence failure

A peptide is not a bag of independently valid residue blocks. Validity depends on relationships:

- ordered graph correspondence;
- covalent connectivity;
- bond lengths and angles;
- consistent conformation;
- excluded volume;
- coordinated feature and spatial states;
- terminal chemistry.

Independent marginal generation can create individually plausible blocks whose complete assembly is invalid.

### 6.2 Formal hypothesis

Let a complete peptide have \(L\) ordered slots. Select \(K\) complete source peptides with the same topology.

At terminal EFS frame \(T\), source peptide \(a\) has:

$$
X_a^{(T)}
=
\left(
x_{a,1}^{(T)},\ldots,x_{a,L}^{(T)}
\right).
$$

Draw one barycentric vector:

$$
\lambda\in\Delta^{K-1}
\qquad
\lambda_a\ge0,
\qquad
\sum_{a=1}^{K}\lambda_a=1

$$

Initialize every target slot synchronously:

$y_\ell^{(T)}\sum_{a=1}^{K}
\lambda_a x_{a,\ell}^{(T)},
\qquad
\ell=1,\ldots,L$

Each slot then follows its ordinary, unmodified EFS backward replay.

**Hypothesis 4.5:** when source peptides are complete, topology matched, slot aligned, locally similar, and represented in a compatible coordinate gauge, shared-\($\lambda$) initialization produces higher raw joint validity than independent seeds or independent per-slot barycentric weights.

### 6.3 Why it may work

Shared sources and shared coefficients preserve:

- common source-complex identity;
- ordered slot correspondence;
- one interpolation location along all selected complexes;
- some correlation between feature and coordinate states;
- a path that remains local when sources are compatible.

### 6.4 What it does not guarantee

Shared \($\lambda$) only correlates initialization. It does not couple the later reverse maps.

Independent nonlinear EFS maps can:

- amplify small slot differences;
- send corresponding slots toward different modes;
- erase source-family correlation;
- produce individually plausible but jointly invalid blocks.

This is why the backward map's local amplification must be measured.

### 6.5 Relation to joint-vector EFS

In joint-vector EFS, the same complete-complex barycentric initialization is:

$Y^{(T)}\sum_{a=1}^{K}
\lambda_a X_a^{(T)}
\in\mathbb{R}^{11L}$

One backward map then moves the complete vector jointly.

The joint model is structurally closer to ANewOmni's joint denoiser. The block model is closer to the original per-block EFS idea and may be numerically easier because each EFS system remains 11-dimensional.

Neither is selected by intuition alone.

### 6.6 Scope and non-goals

| Fixed in the first project              | Excluded from the first project                |
| --------------------------------------- | ---------------------------------------------- |
| Existing ANewOmni encoder and decoder   | Encoder or decoder redesign                    |
| Fixed peptide length and graph topology | One variable-topology sampler                  |
| Complete source complexes               | Reconstructing a peptide from unrelated blocks |
| Shared \(\lambda\) per candidate        | Learned in-flight selector                     |
| Vanilla EFS reverse equations           | Springs or projections hidden inside EFS       |
| Bounded local empirical memory          | Exact EFS over five million complexes          |
| Raw pre-rejection reporting             | Success rates hidden by rejection              |
| Pathwise force provenance               | Biological causal attribution                  |

### 6.7 Non-negotiable implementation rules

1. Use \(K\) for source complexes and \(L\) for residue slots.
2. Select complete source complexes before selecting or interpolating blocks.
3. Mix only topology- and length-matched complexes.
4. Preserve complex ID, slot ID, topology ID, and EFS frame for every cached particle.
5. Define slot correspondence before interpolation.
6. Store every coordinate transformation and its inverse.
7. Use only information available to the sampler for decisive local retrieval.
8. Treat hidden family labels as evaluation information, never as the decisive retrieval rule.
9. Store \(\lambda\) as initialization provenance.
10. Store accumulated forces as pathwise force provenance.
11. Keep initialization, EFS transport, and decoder correction separate.
12. Refit the affected forward history for every data-removal counterfactual.
13. Report metric, EFS parameters, retrieval memory, and random seed with provenance.
14. Report raw proposal yield before rejection or ranking.

### 6.8 Reference pseudocode

    choose one topology bucket G with L ordered slots
    choose an anchor complex using observable information
    retrieve K-1 compatible complete complexes from G
    align all selected complexes using the declared transform
    sample or enumerate one lambda vector on the K-simplex
    
    for each residue slot l:
        construct y_l[T] from the same source IDs and lambda
        replay vanilla slot EFS backward to obtain y_l[0]
    
    alternatively:
        construct one complete Y[T] in the joint 11L space
        replay joint-vector EFS backward to obtain Y[0]
    
    invert metric and coordinate transforms
    decode the complete proposal with ANewOmni
    measure decoder correction
    validate raw geometry, topology, novelty, and provenance
    record accepted and rejected proposals separately

---

## 7. Interpretation and provenance

### 7.1 Claim ladder

| Term                                 | Exact meaning                                                                             | Required evidence                                                                         |
| ------------------------------------ | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Initialization provenance            | Selected source complex IDs, slot correspondence, and barycentric coefficients            | Stored source metadata and \(\lambda\)                                                    |
| Pathwise force provenance            | Signed force and force mass accumulated along one realized backward trajectory            | Per-frame, per-complex force logs with metric and EFS configuration                       |
| Counterfactual algorithmic influence | Output change after data or parameters are intervened on and the affected system is rerun | Full leave-one-out, group-removal, neighborhood-removal, metric, or parameter experiments |
| Biological causal attribution        | A claim that a training complex caused a biological property or motif                     | A separate causal estimand and experimental design, not currently present                 |

### 7.2 Why force provenance is not causal attribution

At one backward step:

$F(v)=\frac{\gamma}{n}
\sum_{i=1}^{n}
\nabla W(v-x_i)$

The contribution associated with particle \(i\) can be recorded explicitly. That accounting remains insufficient for causality.

#### Path dependence

The force from particle \(i\) at a later step depends on the current generated position. That position already depends on all earlier forces. Removing one particle changes the path on which all later forces are evaluated.

#### Cancellation

Two particles can contribute:

$c_A=(5,0),
\qquad
c_B=(-5,0)$

Their signed sum is zero although both exerted large forces. Force mass records both magnitudes but loses cancellation. Signed force records cancellation but can hide large opposing activity.

Both must be reported.

#### Representation dependence

Changing feature or coordinate scaling changes distances, trajectories, and force rankings. A provenance ranking without its metric is incomplete.

#### Missing influence channels

Backward force accounting does not automatically contain:

- a particle's effect on the forward history;
- its effect through source retrieval;
- its barycentric initialization weight;
- the effect of weight renormalization;
- the decoder's nonlinear correction.

### 7.3 Three transformations must remain separate

Every generated design passes through:

1. barycentric or uniform initialization;
2. EFS backward transport;
3. ANewOmni decoding.

A transparent EFS trajectory does not make the decoded all-atom structure causally attributable. Decoder correction can amplify, suppress, or reorganize a latent perturbation.

### 7.4 Candidate record

Each generated proposal must store:

- candidate ID;
- source complex IDs;
- source slot IDs;
- source topology and target IDs;
- retrieval rule and source distances;
- barycentric coefficients;
- terminal seed;
- metric convention and fitted statistics;
- coordinate alignment transform and residual;
- EFS configuration;
- signed force by complex and frame;
- force mass by complex and frame;
- complete latent trajectory;
- pre-decoder generated latent;
- decoded all-atom structure;
- decoder displacement;
- raw validity metrics;
- novelty metrics;
- acceptance or rejection reason.

### 7.5 Counterfactual protocol

For a selected candidate:

1. identify the highest-force-mass complex;
2. identify its highest-force-mass neighborhood;
3. choose random control groups matched by size and retrieval distance;
4. remove each selected barycentric parent in turn;
5. rebuild the affected local EFS forward history;
6. apply the predefined seed-replacement or weight-renormalization rule;
7. rerun backward EFS;
8. decode again;
9. compare latent and all-atom changes.

Reusing the old forward history after removing a particle is not a valid counterfactual refit.

Repeat interventions across:

- random seeds;
- metric choices;
- local-memory sizes;
- nearby \($\gamma,\beta,\epsilon,s,k,T$\);
- nearby terminal seeds where comparisons remain well defined.

### 7.6 Allowed language

Allowed before intervention:

> Complex \(i\) accounted for a stated fraction of accumulated backward force mass under this recorded metric, retrieval memory, seed, and EFS configuration.

Not allowed:

> Complex \(i\) caused this motif.

Not allowed:

> The peptide is a stated percentage derived from complex \(i\).

After stable intervention, the stronger term is counterfactual algorithmic influence, not biological causation.

---

## 8. Relevant rejected and deferred routes

The original hypothesis notebook remains available separately. Only routes that constrain the current design are retained here.

### 8.1 Learned per-step selector

Rejected for strict EFS. A learned attention or selection model inside the reverse update is a function estimator. It may be useful as a hybrid comparator but cannot inherit the estimation-free sampling claim.

Deterministic pre-sampling topology bucketing or retrieval is different because it does not learn the reverse vector field.

### 8.2 Fixed pocket particles

Rejected as the first conditioning mechanism. EFS forces live in sample space, not molecular space. A pocket latent is not an additional independent binder example, and attraction between pocket and binder latents is not a model of \($p(\text{binder}\mid\text{pocket})$).

The first conditioning approach is retrieval-conditioned EFS:

1. encode the query pocket into a descriptor;
2. retrieve topology-compatible complexes with similar pockets;
3. align their binder coordinates target-relatively;
4. fit local EFS to their complete binder latents;
5. decode generated binders using the actual query pocket as context.

### 8.3 Sequential pinning

Deferred. A newly generated residue has no cached forward trajectory at every reverse frame. Holding its final value fixed at all frames is not paper-faithful replay. Reusing its generated reverse path creates a new order-dependent system.

It can later be studied as a named heuristic with explicit weights, paths, and controls.

### 8.4 Springs, constraints, and hard projection

Deferred as modified samplers.

- A spring on all 11 latent dimensions is physically meaningless because only three dimensions are spatial.
- A decaying spring does not prove arrival on the decoder manifold.
- There is no known closed-form projection onto a valid ANewOmni peptide latent.
- Decode, repair, and re-encode is a learned non-convex hybrid.

These can be baselines after vanilla block and joint EFS exist.

### 8.5 Decoder-delegated coherence

Retained as a baseline, rejected as a guarantee. The decoder may repair moderate inconsistencies, but its correction must be measured. Large corrections weaken the connection between latent provenance and final structure.

### 8.6 Motif or complete-complex re-embedding

Strong fallback, not the first move. If \(11L\) EFS is numerically unstable or both co-primary models fail coherence, options include:

- a frozen-latent complete-complex autoencoder;
- rigid motif tokens;
- a set-valued interaction potential;
- another joint representation.

These change the model class and complicate provenance through the learned representation.

---

## 11. Endpoint and numerical diagnostics

### 11.1 Radial distribution

For terminal particle \(i\):

$r_i=\|x_i^{(k)}-c\|$

For a uniform \(d\)-ball of radius \(R\):

$
P(r\le a)
=
\left(\frac{a}{R}\right)^d,
$

$
E[r]=\frac{dR}{d+1},
\qquad
E[r^2]=\frac{dR^2}{d+2}.
$

In two dimensions the radial coefficient of variation is approximately 0.35355. In high dimension the mass concentrates near the boundary.

Use the full radial empirical distribution, not only one coefficient.

### 11.2 Angular uniformity

For:

$
u_i=\frac{x_i-c}{\|x_i-c\|},
$

test:

- mean direction;
- pairwise dot-product distribution;
- angular sectors in low dimension;
- variance near \(1/d\) for high-dimensional uniform directions;
- a spherical MMD or other declared uniformity statistic where appropriate.

### 11.3 Vertex replay

Use actual terminal training particles as backward seeds:

$
x_i^{(k)}
\stackrel{\text{backward}}{\longrightarrow}
\hat{x}_i^{(0)}.
$

Report:

- RMSE;
- median;
- percentiles;
- worst case;
- relation to training-set scale;
- decoded reconstruction where ANewOmni is available.

A failed vertex replay invalidates interpolation experiments.

### 11.4 Numerical overflow

Record all non-finite values and fail explicitly. Do not silently clip them.

Monitor:

- minimum and maximum squared distances;
- inverse-power factors;
- force norms;
- update norms;
- terminal radius;
- proximal residual;
- sensitivity to numeric precision.

### 11.5 Nearest-neighbor behavior

Compare generated-to-training distance with:

- training-to-training nearest-neighbor distance;
- source-to-source distance;
- source-to-generated distance;
- decoded sequence identity;
- decoded structural RMSD;
- motif similarity.

Extremely small distance may indicate retrieval or memorization. Extremely large distance may indicate an off-manifold latent.

### 11.6 Hyperparameter sensitivity

| Parameter    | Role                                        | Too small                             | Too large                                  |
| ------------ | ------------------------------------------- | ------------------------------------- | ------------------------------------------ |
| \(\gamma\)   | Forward step and inverse-field scale        | Slow transport                        | Instability or skipped dynamics            |
| \(\beta\)    | Backward inner optimizer step               | Inaccurate proximal solve             | Oscillation or divergence                  |
| \(\epsilon\) | Repulsion smoothing                         | Extreme forces and overflow           | Excessively weak repulsion                 |
| \(s\)        | Interaction exponent and equilibrium regime | Different target geometry             | Numerical sharpness                        |
| \(k\)        | Forward frames                              | Terminal distribution remains complex | Cost and accumulated error                 |
| \(T\)        | Inner steps per backward frame              | Poor inversion                        | High cost or divergence with bad \(\beta\) |

Look for a stable performance region, not one isolated successful parameter combination.

---

## 12. Complete project roadmap

### Stage 0: exact scientific contract

Before further code, define:

- particle meaning for both co-primary models;
- fixed peptide length and topology;
- training, validation, and held-out target distributions;
- coordinate transform and failure rule;
- metric convention;
- retrieval rule and local memory;
- seed rule and \(\lambda\) distribution;
- EFS equations and parameters;
- decoder context;
- provenance and intervention records;
- baselines;
- metrics;
- pre-registered gates and stop conditions.

Claim-to-falsification requirements:

| Proposed claim                         | Required falsification experiment                             |
| -------------------------------------- | ------------------------------------------------------------- |
| Shared \(\lambda\) preserves coherence | Shared versus independent \(\lambda\) with identical sources  |
| Locality matters                       | Observable local versus random same-topology sources          |
| Joint particles preserve coherence     | Joint versus block and slot-permuted controls                 |
| Alignment removes nuisance variation   | Random independent rotations followed by declared realignment |
| Metric is adequate                     | Feature and coordinate scale perturbations                    |
| Provenance predicts influence          | Top-provenance versus matched random full-refit removals      |
| Latent provenance survives decoding    | Decoder displacement and decoded counterfactual changes       |
| EFS adds value beyond retrieval        | Retrieval-only and nearest-neighbor baselines                 |

**Gate 0:** every major claim has a declared falsification experiment and a reader can reproduce all definitions without guessing.

### Stage 1: incremental audit of the current code

Review one unit at a time:

| Unit | Subject                                 |
| ----:| --------------------------------------- |
| 1    | Array shapes, validation, and EFSConfig |
| 2    | Attractive-repulsive potential          |
| 3    | Forward transport and cached history    |
| 4    | Sphere and barycentric terminal seeds   |
| 5    | Backward proximal replay                |
| 6    | Signed force and force-mass accounting  |
| 7    | ANewOmni latent packing and scaling     |
| 8    | Unit tests and toy experiments          |

For each unit, write:

1. plain-language purpose;
2. exact input and output shapes;
3. corresponding paper or ANewOmni equation;
4. one hand-worked numerical example;
5. short pseudocode;
6. what the code guarantees;
7. what it does not guarantee;
8. diagnostics and failure modes.

Do not replace the audit with a bulk code dump.

**Gate 1:** every prototype subsystem has an understandable equation-to-code mapping and any required correction is documented before implementation.

### Stage 2: latent export and contract

For one fixed length, export one complete record per complex:

- deterministic \(Z_h\), shape \(L\times8\);
- raw \(Z_x\), shape \(L\times3\);
- normalized \(Z_x\), shape \(L\times3\);
- ordered slots and residue identities;
- graph topology and topology key;
- complex ID and target ID;
- pocket center;
- alignment transform, anchors, transform ID, and residual;
- generation mask;
- decoder context;
- encode-decode reconstruction metrics.

Measure:

- latent means and scales;
- effective dimension;
- distance concentration;
- duplicate and near-duplicate rates;
- neighbor rank changes across metrics;
- rotation and alignment consistency;
- decoder response to controlled feature and coordinate perturbations.

**Gate 2:** deterministic encode-decode reconstruction works, transforms are invertible, poor alignments are explicit, and metric effects are measured.

### Stage 3: controlled coherence and granularity experiments

#### Toy A: correlated motifs

Use three or four ordered blocks with overlapping marginals and multiple complete conformational families.

#### Toy B: flexible chains

Use four to six beads with:

- approximate bond lengths;
- angle ranges;
- excluded volume;
- multiple global conformations;
- feature vectors linked to conformation.

#### Toy C: topology mismatch

Mix lengths or graph structures and verify:

- cross-topology mixing fails;
- topology buckets prevent the failure;
- variable topology is a distinct sampling problem.

#### Required methods

1. retrieval-only complete complexes;
2. independent uniform block EFS;
3. independent \(\lambda_\ell\) per slot;
4. shared \(\lambda\) with random same-topology sources;
5. shared \(\lambda\) with observable local sources;
6. shared \(\lambda\) with permuted slots;
7. joint-vector uniform seeds;
8. joint-vector local barycentric seeds.

All methods use identical:

- source data;
- proposal count;
- EFS budget;
- metric;
- validation;
- random-seed policy.

#### Lambda sequence

Start with \(K=2\):

$
\lambda=(u,1-u),
$

and sweep \(u\) from one vertex to the other, including the midpoint.

Only after this is interpretable:

- test \(K=3\);
- use a small declared Dirichlet grid;
- vary concentration from near-one-source to balanced mixtures.

#### Safe-simplex map

Map joint validity against:

- source-set dispersion;
- barycentric entropy;
- source-family compatibility;
- original-frame locality;
- terminal-frame locality;
- both-frame locality.

Estimate local backward amplification:

$
\frac{\|B_T(y)-B_T(y')\|}
{\|y-y'\|}.
$

**Gate 3:** a method advances only if it improves raw joint validity over independent blocks, works without oracle retrieval, remains more than retrieval copying, and does not use rejection to conceal low raw yield.

### Stage 4: provenance and counterfactual validation

Run:

- top-complex removal;
- top-neighborhood removal;
- matched random removal;
- barycentric-parent removal;
- metric perturbation;
- nearby EFS-parameter perturbation;
- terminal-seed perturbation.

Refit the affected EFS system after every data intervention and decode every rerun.

Report effect distributions across many candidates rather than one illustrative example.

**Gate 4:** use counterfactual algorithmic-influence language only if top-provenance interventions differ reproducibly from matched random interventions across seeds, metrics, and nearby parameters.

### Stage 5: scalability design

Do not fit exact vanilla EFS to all five million complexes.

Use:

- topology and length buckets;
- target or target-family partitions;
- bounded local retrieval;
- alignment;
- local memory \(M\);
- chunked exact interactions;
- retrieval-only baselines.

Sweep \(M\) and report:

- runtime;
- peak memory;
- proposal throughput;
- validity;
- novelty;
- provenance stability;
- intervention stability;
- sensitivity to retrieval boundaries.

Any neighbor truncation, coreset, clustering, hierarchy, or approximate field is named modified EFS and compared with exact local EFS where feasible.

**Gate 5:** quality and provenance stabilize at a tractable \(M\), and EFS adds value over retrieval-only generation.

### Stage 6: first real ANewOmni pilot

Use:

- fixed-length short linear peptides;
- one target or tightly aligned target family;
- deterministic latents;
- one fixed topology;
- no pocket pins;
- no learned guidance;
- no variable non-canonical chemistry in the first pilot;
- identical data, retrieval, metric, and validation across methods.

Compare:

- independent block EFS;
- H4.5 shared-\(\lambda\) block EFS;
- joint-vector local barycentric EFS;
- retrieval-only complete complexes;
- decoder-only perturbation and repair.

Report:

- raw decode success;
- bond and angle validity;
- clash rate;
- graph and sequence validity;
- structural and sequence novelty;
- nearest-training-complex similarity;
- decoder correction magnitude;
- provenance stability;
- counterfactual removal effects;
- accepted and rejected counts separately.

**Gate 6:** the selected method improves pre-rejection validity over independent block EFS, remains more than near-copy retrieval, and uses only the interpretation claims earned at Gate 4.

### Stage 7: later extensions only if warranted

Possible later branches:

- retrieval-conditioned transfer to held-out targets;
- non-canonical chemistry;
- modified EFS approximations;
- constrained dynamics as named baselines;
- motif tokenization;
- a frozen-latent complete-complex compressor;
- a joint set-valued potential;
- a flow-based alternative if EFS fails.

No later method inherits the original EFS theorem or provenance claim by wording alone.

---

## 13. Scalability

### 13.1 Why matrix reshaping is insufficient

With:

$
n=5{,}000{,}000,
$

the full forward step contains:

$
n^2=25\text{ trillion}
$

ordered pair interactions before symmetry or implementation details.

A scalar float32 pair matrix requires approximately 100 terabytes.

For \(d=110\), a full float32 displacement tensor would require approximately 11 petabytes.

Chunking avoids storing the entire tensor at once. It does not remove the total pair count.

### 13.2 Forward history

For:

- \(n=5\) million;
- \(d=110\);
- 101 cached frames;
- float32 storage;

the latent history alone is roughly 222 gigabytes before metadata and temporary arrays.

### 13.3 First scalable approach

The first credible route is bounded local empirical memory:

1. choose a topology and length bucket;
2. retrieve a bounded pocket-compatible neighborhood;
3. align it;
4. fit exact local EFS;
5. compare against returning or interpolating the retrieved examples directly.

Retrieval is therefore both:

- a conditioning hypothesis;
- a scalability requirement.

### 13.4 Possible later acceleration

The attractive component:

$
\sum_j(v-x_j)
$

can be reduced using sums and means.

The inverse-power repulsive component remains the difficult many-body term. Possible approximations include:

- approximate neighbor truncation;
- hierarchical methods;
- fast multipole analogues where dimension permits;
- coresets;
- clustered memories;
- learned or non-learned reduced representations.

Every approximation changes finite EFS behavior and provenance. Exact local EFS remains the reference.

---

## 14. Validation metrics

### 14.1 Latent and numerical metrics

- radial distribution discrepancy;
- angular uniformity;
- vertex replay RMSE;
- proximal residual;
- finite-value and overflow counts;
- update and force norms;
- effective dimension;
- pairwise-distance concentration;
- metric sensitivity;
- nearest-training distance;
- training-to-training nearest-neighbor reference;
- local backward amplification.

### 14.2 Peptide validity

- decoder success;
- correct block count;
- graph and topology validity;
- peptide bond length error;
- angle and torsion plausibility;
- clash rate;
- excluded-volume violations;
- termination validity;
- sequence validity;
- all-atom structural confidence.

### 14.3 Novelty and diversity

- nearest training-complex latent distance;
- sequence identity;
- backbone RMSD;
- motif similarity;
- unique-output rate;
- diversity across source neighborhoods;
- diversity after removing duplicate decodes;
- retrieval-copy rate.

### 14.4 Interpretation metrics

- source and \(\lambda\) records;
- signed force by complete complex;
- force mass by complete complex;
- top-\(q\) provenance stability;
- decoder correction magnitude;
- top-removal versus random-removal latent change;
- top-removal versus random-removal decoded change;
- stability across metric and EFS perturbations.

### 14.5 Reporting rule

Always report:

- number of raw proposals;
- number decoded;
- number valid before rejection;
- number rejected by each rule;
- number retained;
- metrics for both raw and retained sets.

---

## 15. Resources and timeline

### 15.1 Resources

| Resource                              | Role                                                    | Current access       |
| ------------------------------------- | ------------------------------------------------------- | -------------------- |
| ANewOmni source                       | Encoder, decoder, latent interface, prompts, validation | Present              |
| ANewOmni checkpoint                   | Real deterministic latent export                        | Not present locally  |
| EFS paper                             | Sampling equations and theory                           | Present              |
| ANewOmni paper                        | Model design and evaluation                             | Present              |
| SIU, PepBench, ProtFrag, PDBbind-plus | Potential data and benchmarks                           | Not prepared locally |
| PepBDB and non-canonical PDB subsets  | Later non-canonical evaluation                          | Not prepared locally |
| Laptop and bundled Python             | Documentation, NumPy core, toys, tests                  | Available            |
| Cluster                               | Real latent extraction and larger local EFS             | Expected later       |

### 15.2 Phase 0: before cluster access

- complete the incremental code audit;
- write the exact Stage 0 contract;
- correct the toy retrieval oracle;
- add the missing controls;
- map the safe simplex;
- specify latent export and metadata;
- define counterfactual intervention records;
- estimate compute versus local memory.

### 15.3 Phase 1: first real-data year

- acquire or train against a declared checkpoint;
- export deterministic fixed-length latents;
- verify encode-decode round trips;
- compare the co-primary EFS models;
- measure raw validity and decoder correction;
- run provenance stability and counterfactual tests;
- write a result only at the claim level supported by evidence.

### 15.4 Phase 2: extensions

- held-out target transfer;
- retrieval-conditioned generation;
- scalable approximate EFS if justified;
- non-canonical chemistry;
- alternative representations only after first-model failure is measured.

---

## 16. Current blockers and factual checks

### 16.1 Immediate blockers

- ANewOmni/checkpoints contains only a placeholder file.
- No processed deterministic latent dataset is present.
- The local ANewOmni environment has CPU-only PyTorch and lacks required packages for full model execution.
- The workspace is not a Git worktree.

These block real-data latent extraction, not documentation, the NumPy reference core, or toy experiments.

### 16.2 Factual checks before real integration

- obtain the exact released checkpoint and configuration;
- confirm the intended deterministic encoder call;
- confirm the correct preprocessing and vocabulary assets;
- confirm block ordering and topology metadata;
- confirm pocket-center and coordinate-normalization conventions;
- confirm the decoder input contract for externally supplied latents;
- confirm the recommended confidence model and validation path;
- confirm data licensing and split rules;
- define held-out targets before retrieval rules are tuned.

---

## 17. Stop conditions

Demote or stop a path if:

1. shared \(\lambda\) does not improve raw joint validity after topology, locality, and correspondence are controlled;
2. the benefit exists only for nearly identical source complexes;
3. outputs are indistinguishable from retrieval copies;
4. observable retrieval fails where oracle retrieval succeeded;
5. the backward map erases initialization coherence;
6. joint-vector EFS is numerically unstable at practical dimensions;
7. decoder corrections disconnect latent provenance from final structure;
8. force rankings change materially under reasonable metric or parameter perturbations;
9. top-provenance removals behave like matched random removals;
10. local memory remains computationally intractable;
11. no reliable complete-complex correspondence exists;
12. coordinate transforms cannot be made invertible and auditable;
13. raw proposal validity remains poor and success depends on severe rejection;
14. EFS adds no measurable value over retrieval-only or decoder-only baselines.

A stop result remains informative. It identifies whether the failure lies in:

- particle granularity;
- source construction;
- representation geometry;
- finite EFS transport;
- decoder compatibility;
- provenance interpretation;
- scalability.

---

## 18. Next work item

The next approved activity is Stage 1, Unit 1:

**Array shapes, input validation, and EFSConfig.**

It must be explained incrementally with:

- each input and output shape;
- each configuration field;
- one small numerical example;
- the relationship to the paper;
- what the code guarantees;
- what it does not guarantee.

No integration code is added during that explanation.

---

## 19. Compact glossary

| Term                                 | Meaning in this project                                                                        |
| ------------------------------------ | ---------------------------------------------------------------------------------------------- |
| Block                                | One ANewOmni residue or molecular token with feature and coordinate latent                     |
| Slot                                 | One ordered topology position in a fixed-length peptide                                        |
| Complete complex                     | One intact training peptide and its target/context metadata                                    |
| Particle                             | One row operated on by one EFS system                                                          |
| Joint particle                       | One complete flattened peptide latent                                                          |
| Local memory                         | Bounded empirical subset used to fit one EFS system                                            |
| Barycentric seed                     | Convex combination of terminal training particles                                              |
| Shared \(\lambda\)                   | One coefficient vector applied to corresponding slots of complete sources                      |
| Initialization provenance            | Source IDs, slot correspondence, and \(\lambda\)                                               |
| Pathwise force provenance            | Recorded force terms along one realized EFS trajectory                                         |
| Counterfactual algorithmic influence | Output change under a complete rerun after an intervention                                     |
| Decoder correction                   | Difference between the generated latent proposal and the decoder's realized all-atom structure |
| Raw validity                         | Validity before rejection or ranking                                                           |
| Retrieval collapse                   | Generation that produces near-copies without measurable value beyond retrieval                 |

## Primary Problems Faced

1. **Global Computational Scaling:** Exact EFS requires pairwise interactions between all particles during the forward transport. For $N$ empirical particles in $d$ dimensions, the forward cost scales approximately as $O(KN^2d)$, where $K$ is the number of cached EFS frames. This is incompatible with a five-million-complex training bank.

2. **Set-Coherence under Hypothesis 4.5:** Shared source complexes and shared barycentric weights correlate the initial residue seeds, but each residue slot still follows an independent nonlinear backward trajectory. The common initialization may therefore be progressively erased during generation.

3. **Global Data Utilization versus Local Feasibility:** Restricting EFS to a retrieved local neighborhood makes the computation feasible, but the sampler no longer acts on the complete AnewOmni empirical distribution. The complete training bank becomes an indexing and retrieval resource rather than the particle system directly transported by EFS.

4. **Approximation and Interpretability:** Any coreset, sparse interaction graph, or hierarchical decomposition modifies the original EFS dynamics. These methods may preserve practical interpretability, but they cannot automatically inherit the exact theoretical claims of vanilla EFS.

## Additional Hypotheses

**6. Hierarchical Local Hypothesis 4.5**

Instead of running EFS over the complete AnewOmni particle bank, use the complete bank as a deterministic retrieval inventory and run exact Hypothesis 4.5 only inside a bounded local empirical memory.

- **Global Inventory:** Organize all training complexes using observable metadata and descriptors, including molecular modality, peptide length, topology, pocket geometry, binder conformation, interaction patterns, and coarse latent-space descriptors.

- **Target-Conditioned Retrieval:** Given a target pocket $q$, peptide length $L$, and topology $\tau$, retrieve a compatible subset

$$
\mathcal{S}(q,L,\tau)=\{C_1,\ldots,C_M\}, \qquad M \ll N.
$$

- **Complete-Complex Selection:** Select the source complexes before extracting residue blocks. Every residue slot uses the same complex IDs and the same barycentric vector $\lambda$.

- **Local EFS:** For residue slot $\ell$, construct the local particle system

$$
\mathcal{X}_\ell=\{x_{1,\ell},\ldots,x_{M,\ell}\}\subset\mathbb{R}^{11},
$$

and initialize

$$
y_\ell^{(T)}=\sum_{a=1}^{K}\lambda_a x_{a,\ell}^{(T)}.
$$

Each slot then follows ordinary EFS backward replay inside its own local 11-dimensional system.

- **Interpretability:** Separate the record into two layers:
  
  1. **Retrieval provenance:** why each complete complex entered $\mathcal{S}$;
  2. **EFS provenance:** how each selected complex influenced each residue trajectory.

- **Advantages:** Preserves the original AnewOmni bank as a global source of diversity while avoiding global EFS cost. It also keeps the EFS state dimension at 11 rather than $11L$.

- **Trade-offs:** The resulting model approximates a retrieval-conditioned empirical distribution rather than the global AnewOmni training distribution. Poor retrieval can remove useful modes before EFS begins, and independent slot dynamics can still destroy set coherence.

**7. Weighted Medoid or Coreset EFS**

Compress the complete empirical bank into a weighted set of real representative complexes rather than discarding all but a local neighborhood.

- **Partition:** Divide the complete training bank into clusters

$$
\mathcal{C}_1,\ldots,\mathcal{C}_R.
$$

- **Representative Complexes:** For each cluster $\mathcal{C}_r$, select an empirical medoid $m_r$ corresponding to a real training complex rather than a synthetic centroid.

- **Cluster Weights:** Assign

$$
w_r=\frac{|\mathcal{C}_r|}{N}, \qquad \sum_{r=1}^{R}w_r=1.
$$

- **Weighted EFS Field:** Replace the exact empirical force

$$
F(v)=\frac{\gamma}{N}\sum_{i=1}^{N}\nabla W(v-x_i)
$$

with

$$
\widetilde{F}(v)=\gamma\sum_{r=1}^{R}w_r\nabla W(v-m_r).
$$

- **Multiscale Refinement:** Use the weighted medoid system to identify a coarse region, then expand the influential medoid clusters and run exact local Hypothesis 4.5 on their member complexes.

- **Interpretability:** Provenance is initially assigned to representative medoids and cluster weights. A second local stage can resolve the contribution of individual member complexes.

- **Advantages:** The complete dataset contributes through the cluster weights, representatives remain real empirical complexes, and the primary EFS cost depends on $R$ rather than $N$.

- **Trade-offs:** This is approximate EFS. Rare modes can disappear when represented by low-weight or poorly chosen medoids. The clustering metric becomes part of the model and directly affects both generation and provenance.

**8. Sparse-Force EFS**

Exploit the algebraic structure of the EFS potential to compute the global attractive component exactly while approximating only the expensive short-range repulsive component.

For the smoothed EFS interaction,

$$
\nabla W(x_i-x_j)
=
(x_i-x_j)
-
\frac{x_i-x_j}
{\left(\|x_i-x_j\|^2+\epsilon\right)^{s/2+1}}.
$$

The global attractive term satisfies

$$
\sum_{j=1}^{N}(x_i-x_j)=Nx_i-\sum_{j=1}^{N}x_j,
$$

and can therefore be evaluated using the global mean in $O(Nd)$ rather than $O(N^2d)$.

- **Exact Global Attraction:** Compute

$$
F_i^{\mathrm{attr}}=x_i-\bar{x},
\qquad
\bar{x}=\frac{1}{N}\sum_{j=1}^{N}x_j.
$$

- **Sparse Repulsion:** Approximate the repulsive component using a neighborhood $\mathcal{N}_k(i)$:

$$
F_i^{\mathrm{rep}}
\approx
\frac{1}{N}
\sum_{j\in\mathcal{N}_k(i)}
\frac{x_i-x_j}
{\left(\|x_i-x_j\|^2+\epsilon\right)^{s/2+1}}.
$$

- **Possible Approximations:** Use $k$-nearest-neighbor graphs, radius cutoffs, hierarchical trees, clustered far-field expansions, or fast multipole approximations.

- **Application to Hypothesis 4.5:** Construct one sparse interaction graph for each ordered residue slot, while forcing every slot to use neighborhoods derived from the same complete-complex inventory.

- **Advantages:** Potentially permits EFS over a much larger empirical bank while retaining exact global attraction and explicit particle-level repulsive terms.

- **Trade-offs:** Sparse repulsion modifies the EFS equilibrium and invalidates direct use of the original convergence theorem. The neighborhood graph may also change during transport, introducing discontinuities and additional computational cost. Attribution becomes conditional on the sparse graph construction.

**9. Shared Relational Sidecar for Hypothesis 4.5**

Retain the original 11-dimensional residue EFS systems, but accompany them with a deterministic descriptor of the complete peptide geometry. The sidecar does not replace or compress the residue latents; it records inter-residue relationships that Hypothesis 4.5 otherwise fails to preserve explicitly.

- **Relative Coordinate Descriptor:** For adjacent residue slots, define

$$
r_\ell=x_{\ell+1}-x_\ell\in\mathbb{R}^{3},
\qquad \ell=1,\ldots,L-1.
$$

For a peptide of length $L$, the adjacent-coordinate sidecar has dimension $3(L-1)$ rather than $11L$.

- **Alternative Relations:** The sidecar may contain selected pairwise distances, local turning angles, terminal displacement, radius of gyration, principal-axis statistics, or low-frequency coefficients of the coordinate trajectory.

- **Shared Retrieval:** Retrieve complete complexes using both pocket compatibility and relational-sidecar similarity. This forces every residue slot to inherit source complexes with compatible global geometry.

- **Trajectory Diagnostics:** During backward replay, reconstruct the current sidecar

$$
g\left(y_1^{(t)},\ldots,y_L^{(t)}\right)
$$

and compare it with the barycentric source relation

$$
\bar{g}^{(t)}=\sum_{a=1}^{K}\lambda_a g\left(x_{a,1}^{(t)},\ldots,x_{a,L}^{(t)}\right).
$$

- **Weak Constrained Variant:** Introduce a deterministic relation-preservation term

$$
\mathcal{L}_{\mathrm{rel}}^{(t)}
=
\left\|
 g(y_1^{(t)},\ldots,y_L^{(t)})-\bar{g}^{(t)}
\right\|^2.
$$

This term may be used only for rejection and diagnostics, or may be added as an explicit coupling force during backward replay.

- **Advantages:** Attacks the principal weakness of Hypothesis 4.5 without creating a single $11L$-dimensional EFS particle. It preserves residue-level EFS provenance while providing a measurable notion of whole-peptide coherence.

- **Trade-offs:** If used only for retrieval and rejection, the sidecar does not prevent divergence during generation. If used as a coupling force, the system is no longer vanilla EFS and the relation metric becomes an additional hand-designed model component.

## Current Ranking

1. **Hierarchical Local Hypothesis 4.5:** Most practical unchanged-embedding route and the strongest immediate baseline.

2. **Sparse-Force EFS:** Most promising route for exploiting a substantially larger empirical bank without explicitly transporting every pair of particles.

3. **Weighted Medoid or Coreset EFS:** Plausible multiscale approximation when sparse global transport remains too expensive.

4. **Shared Relational Sidecar:** Useful for diagnosing and potentially correcting the set-coherence failure of Hypothesis 4.5 after the unmodified version has been measured.

## Shared Non-Claim

None of these hypotheses constitutes exact EFS over the complete AnewOmni training distribution. Each introduces retrieval, compression, sparse interactions, or deterministic relational structure. Their scientific value lies in testing whether a foundation-scale empirical bank can support an auditable non-neural sampler without requiring the complete quadratic particle system.

---

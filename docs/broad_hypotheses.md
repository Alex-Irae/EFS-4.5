## Primary Problems Faced

1. **Set-Coherence (Inter-block Relationships):** EFS transports particles independently based on the global training field. Sibling seeds do not naturally interact during generation, which risks producing locally valid amino acid blocks that assemble into physically invalid molecules featuring broken backbones or side-chain interpenetration.

2. **The Decodable Manifold:** The EFS backward pass must land points precisely within regions the AnewOmni VAE recognizes. Out-of-distribution coordinates cause the decoder to fail or output twisted, physically corrupted 3D geometries.

3. **Conditioning Mechanisms:** AnewOmni natively manages constraints via programmable graph prompts and classifier-free guidance. EFS lacks a theoretical mechanism for applying constraints to a subset of particles while maintaining convergence guarantees.

## Abandoned Hypotheses and Justifications for Failure

* **Hypothesis: Decoder-Delegated Coupling**
  
  * _Concept:_ Allow EFS to generate an approximately placed set of independent seeds, then rely on the AnewOmni graph prompt and VAE decoder to absorb those points and force them into a valid configuration.
  
  * _Failure Justification:_ Xiangzhe explicitly confirmed that the decoder cannot tolerate large inconsistencies. If the provided 3D latents are out-of-distribution, the decoder will not cleanly project them to a valid backbone; it will instead produce twisted and corrupted 3D geometries. Furthermore, if the decoder heavily overrides the generated coordinates, the exact attribution provided by EFS is destroyed.

* **Hypothesis: Learned Per-Step Selector (Soft Attention)**
  
  * _Concept:_ Introduce an attention mechanism or picker at each backward timestep to intelligently select the correct structural parts and enforce coherence.
  
  * _Failure Justification:_ This reintroduces a learned estimator (a score-like function) directly into the backward pass. Doing so violates the fundamental premise of Estimation-Free Sampling and completely nullifies the attribution property, reducing the architecture to a latent diffusion model with unnecessary steps.

* **Hypothesis: Forward Pass Target Constraints**
  
  * _Concept:_ Pin the pocket constraint blocks (boundary conditions) during the initial forward mapping of the dataset.
  
  * _Failure Justification:_ The forward pass is strictly an unconditional transport map designed to push the target distribution to a uniform sphere. Introducing fixed constraints during this phase mathematically breaks the uniform convergence. Constraints must operate exclusively as boundary conditions during the backward generation step.

* **Hypothesis: Pure Sequential Pinning (Unweighted Autoregressive EFS)**
  
  * _Concept:_ Generate one seed, freeze it as a fixed charge, and drop the next seed to navigate the combined field of the training data and the newly pinned seed.
  
  * _Failure Justification:_ The gravitational influence of a single pinned seed in an external field of $5,000,000$ training points is statistically insignificant. The external macro-gradient completely washes out the local constraint, meaning the sequential pin fails to enforce geometric coherence at scale.

* **Hypothesis: Independent Initialization with Holonomic Constraints**
  
  * _Concept:_ Sample $M$ independent points randomly across the uniform sphere and apply a rigid harmonic spring during the backward pass to force them into a valid backbone geometry.
  
  * _Failure Justification:_ Topologically distant points on the uniform sphere map to entirely different chemical regimes. The internal spring force would violently conflict with the external EFS macro-gradient. This tug-of-war traps particles in out-of-distribution dead zones, guaranteeing a corrupted decode.

## Hypothesis

**1. Correlated Sphere Sampling with Decaying Holonomic Constraints (The Bead-Spring Model)**
Instead of sampling independently, Seed 1 is drawn randomly on the uniform sphere, and subsequent seeds are sampled as a tight random walk within a microscopic neighborhood around Seed 1. This ensures the entire set is pulled down the same macroscopic EFS trajectory toward a unified chemical cluster. 

- **Initialization:** Sample the first root seed uniformly at random on the boundary sphere. Generate the subsequent seeds as a tightly bound random walk on the sphere's surface relative to the root. This ensures the entire set is pulled down the same general macroscopic trajectory by the EFS vector field.

- **Internal Constraint:** Apply a coarse-grained bead-spring harmonic potential ($U(r) = \frac{1}{2}k(||x_i - x_{i-1}|| - r_0)^2$) exclusively between adjacent seeds during the inner $T$-loop of the backward pass. This enforces rigid backbone distances without biasing the chemical cluster destination.

- **Decay Schedule:** Start with a high spring constant $k$ near the uniform boundary to establish the chain structure. Decay $k$ to zero as $t \rightarrow 0$. The final micro-optimizations are driven entirely by the EFS data gradient, ensuring the blocks land strictly on the decodable manifold without spring-induced suspension in dead zones.



**2, Sub-Manifold Hard Projection (Projected EFS)**
Instead of relying on a soft, decaying harmonic spring to manage geometry, implement a hard mathematical projection operator. At the conclusion of every micro-optimization $T$-loop in the backward pass, calculate the nearest mathematically valid backbone configuration within the 11D space. Snap the temporary $v\_t$ coordinates exactly to this valid sub-manifold before stepping to the next historical frame. This guarantees strict geometric validity at every timestep while maintaining the unweighted deterministic pull of the EFS field.



**3.  Tokenization Overhaul (The Rigid-Body Approach)**
EFS excels at transporting single particles but struggles with multi-particle interacting sets. If set-coherence remains mathematically intractable, the tokenization strategy must be elevated. Instead of treating single amino acid blocks as individual latent points, reconstruct the VAE encoder to map larger, rigid structural motifs (such as entire alpha-helices or predefined dipeptides) to single coordinates. By drastically reducing the number of dynamic joints the generative process must manage, the burden of internal coherence is shifted back to the spatial encoding rather than the generation engine.



**4. Barycentric Interpolation Strategy**

Previous discussion, classy but suffers from all previous annoted problems.

- **Mechanism:** Extract a subset of points from the final uniform frame of the history tensor. Generate random $\lambda$ weights that sum to 1 to compute a convex combination (a new initialization vector). Feed this directly into the backward optimization loop.

- **Trade-offs:** This provides clear interpretability through the $\lambda$ weights, showing exactly which boundary particles contribute to the sample. It requires robust validation to ensure the interpolated points still converge to the dense data regions rather than sparse outliers.

**4.5 Shared Barycentric Coordinates (Complex-Level Interpolation) + on the run tuning**

- Instead of applying independent $\lambda$ weights to single blocks, lock a single set of $\lambda$ weights across the entire multi-particle binder, so that we do not interpolate isolated blocks; but we interpolate entire training peptides.
  
  1. **Define the Target Topology:** decide on the necessary sequence of blocks for the binder (e.g., Block 1 = Amine, Block 2 = C-alpha, Block 3 = R-group).
  
  2. **Filter the Inventory on the Sphere:** To generate Block 1, query the metadata inventory for all training points annotated as Amine. locate those specific $K$ particles on the $t=T$ boundary sphere.
  
  3. **Select Complete Complexes:** extract a subset of $M$ intact training peptides from theuniform history sphere at $t=T$(Peptide A and Peptide B).
  
  4. **Generate One Global $\lambda$ Vector:** generate a single random array of weights, e.g., $\lambda_A = 0.6$ and $\lambda_B = 0.4$.
  
  5. **Apply $\lambda$ Synchronously:** apply this exact same weight ratio to every corresponding structural block in the sequence:
     
     1. $\text{Seed}_1 (\text{Amine}) = 0.6 \cdot A_1 + 0.4 \cdot B_1$
     
     2. $\text{Seed}_2 (\text{C-alpha}) = 0.6 \cdot A_2 + 0.4 \cdot B_2$
     
     3. $\text{Seed}_3 (\text{R-group}) = 0.6 \cdot A_3 + 0.4 \cdot B_3$
- Solves the blind aiming issue of simple linear combination to enforce set coherence,
- Main issue : block count as 2 valid config with different block length confict with each other



**5. Boundary Condition Conditioning (Target Pockets)**

- **Mechanism:** Hard pin the encoded pocket blocks as fixed, non-transported particles at their exact coordinates during the backward pass. The generated binder points condense in the field created by these fixed charges.

- **Trade-offs:** This is mechanically consistent with boundary-condition physical systems and requires no guidance network. The theoretical convergence remains unproven for partially-fixed systems, making this a central research question.

---

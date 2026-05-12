# VelOT: Complete Methodology

## Overview

VelOT (Velocity via Optimal Transport) is a kinetic-free framework for estimating RNA velocity from single-cell transcriptomic data. Instead of modeling the molecular kinetics of transcription, splicing, and degradation, VelOT treats velocity inference as a mass transport problem in high-dimensional gene expression space. The method proceeds in four stages: (1) preprocessing and pseudotime inference, (2) spatial-temporal windowing, (3) optimal transport velocity estimation, and (4) neural velocity field smoothing. All velocity computations are performed in PCA space, with projection to UMAP only for visualization.

<p align="center">
  <img src="article/figures/figure 1/panel1.png">
</p>

---

## Stage 1: Preprocessing and Pseudotime Inference

### Gene expression preparation

Starting from a raw gene expression count matrix $\mathbf{X}_{\text{raw}} \in \mathbb{R}^{N \times G}$, where $N$ is the number of cells and $G$ is the number of genes, we apply the following standard preprocessing steps:

1. **Total-count normalization**: each cell's counts are scaled to a common total (10,000), removing library size effects.

2. **Log-transformation**: $x \mapsto \log(1 + x)$, stabilizing variance and compressing the dynamic range.

3. **Highly variable gene selection**: the top $G'$ genes (typically 2,000) with the highest variance-to-mean ratio are retained, removing uninformative genes that would add noise to downstream computations.

4. **Scaling**: each gene is centered to zero mean and scaled to unit variance, ensuring that PCA captures correlation structure rather than being dominated by highly expressed genes.

### Dimensionality reduction

We compute a PCA representation of the preprocessed data:

$$\mathbf{X} \in \mathbb{R}^{N \times D}$$

where $D$ is the number of principal components (typically 30–50). This PCA space is where all velocity computations take place. It preserves the dominant axes of variation in gene expression while reducing noise from the remaining dimensions.

A $k$-nearest neighbor (KNN) graph is constructed on the PCA coordinates, with $k$ typically set to 30. This graph captures the local manifold structure of the data and is used in three downstream steps: pseudotime computation, OT cost matrix construction, and velocity smoothing.

A UMAP embedding is computed from the KNN graph for visualization purposes only. VelOT does not compute velocity in UMAP space, avoiding the information loss and geometric distortions inherent in nonlinear dimensionality reduction.

### Pseudotime inference via Diffusion Pseudotime

A temporal ordering of cells is established using Diffusion Pseudotime (DPT). DPT constructs a diffusion process on the KNN graph and measures the diffusion distance from a user-specified root cell to every other cell. The key properties of DPT are:

- It depends only on the connectivity structure of the expression manifold, not on any velocity or kinetic model.
- It is robust to branching, as diffusion distance naturally handles tree-like topologies.
- It provides a global ordering that respects the intrinsic geometry of the data.

The raw DPT values are normalized to the unit interval:

$$\tau_i = \frac{\text{DPT}_i - \min_j \text{DPT}_j}{\max_j \text{DPT}_j - \min_j \text{DPT}_j} \in [0, 1]$$

This normalization makes $\tau_i = 0$ correspond to the root (earliest) state and $\tau_i = 1$ to the most distant (latest) state.

The use of DPT is a deliberate design choice to break the circularity of kinetic methods, where scVelo's dynamical mode estimates a latent time as part of the velocity model fit, creating a dependency loop between temporal ordering and velocity estimation. In VelOT, the pseudotime is fully independent of the velocity, ensuring that the temporal axis is not biased by the quantity we are trying to estimate.

Alternatively, users can provide any precomputed pseudotime or temporal ordering, allowing integration with other trajectory inference methods.

---

## Stage 2: Spatial-Temporal Windowing

### Motivation

The naïve approach to OT-based velocity would sort all cells by pseudotime and compute transport between consecutive global time slices. This fails in complex biological systems where cells at different developmental stages coexist in the same spatial neighborhood. For example, in the pancreas endocrinogenesis dataset, progenitor cells and terminally differentiated cells overlap extensively in UMAP and even in PCA space. Global pseudotime windows would contain cells scattered across the entire manifold, and OT would compute transports across large distances, producing biologically nonsensical velocities.

### Two-level windowing

VelOT addresses this with a two-level windowing strategy that ensures OT is computed only between spatially proximate cells.

**Level 1: Spatial clustering.** Cells are partitioned into $K$ spatial clusters using K-means clustering on the PCA coordinates $\mathbf{X}$. The number of clusters $K$ is chosen automatically based on dataset size (typically $K = \max(5, \min(30, N/100))$) or can be set by the user. Alternatively, if prior biological annotations are available (such as cell type labels), these can be used directly as spatial partitions. Let $\mathcal{I}_k$ denote the set of cell indices in cluster $k$.

**Level 2: Local temporal windowing.** Within each spatial cluster $k$, cells are sorted by pseudotime $\tau$. A series of overlapping temporal windows is created with window size $W$ and overlap fraction $f_{\text{overlap}}$. The step size between consecutive windows is:

$$S = \lfloor W \cdot (1 - f_{\text{overlap}}) \rfloor$$

Window $j$ within cluster $k$ contains the cells at sorted positions $[jS, jS + W)$. Consecutive windows within the same cluster form source-target pairs:

$$\text{Pairs}_k = \{(w_{k,1}, w_{k,2}), (w_{k,2}, w_{k,3}), \ldots\}$$

The complete set of window pairs across all clusters is:

$$\text{Pairs} = \bigcup_{k=1}^{K} \text{Pairs}_k$$

### Properties of the windowing scheme

This two-level strategy guarantees several desirable properties:

- **Spatial locality**: OT is only computed between cells that are nearby in PCA space, preventing long-range transports across the manifold.
- **Temporal progression**: within each spatial cluster, the window ordering follows pseudotime, ensuring that transport captures forward biological progression.
- **Coverage**: overlapping windows mean that most cells appear as a source in multiple pairs, providing redundancy and enabling confidence estimation.
- **Scalability**: the number of window pairs scales linearly with the number of cells, not quadratically.

---

## Stage 3: Optimal Transport Velocity Estimation

### Transport plan computation

For each source-target window pair $(w_{\text{src}}, w_{\text{tgt}})$, we compute a transport plan that optimally redistributes the mass of source cells to target cells. Let $n_s = |w_{\text{src}}|$ and $n_t = |w_{\text{tgt}}|$.

We assign uniform marginals:

$$\mathbf{a} = \frac{1}{n_s} \mathbf{1}_{n_s}, \qquad \mathbf{b} = \frac{1}{n_t} \mathbf{1}_{n_t}$$

The transport plan $\mathbf{P}^* \in \mathbb{R}^{n_s \times n_t}$ is the solution to the entropy-regularized optimal transport problem (Sinkhorn algorithm):

$$\mathbf{P}^* = \arg\min_{\mathbf{P} \in \Pi(\mathbf{a}, \mathbf{b})} \langle \mathbf{P}, \mathbf{C} \rangle_F - \varepsilon \, H(\mathbf{P})$$

where $\Pi(\mathbf{a}, \mathbf{b})$ is the set of joint distributions with marginals $\mathbf{a}$ and $\mathbf{b}$, $\langle \cdot, \cdot \rangle_F$ is the Frobenius inner product, $H(\mathbf{P}) = -\sum_{ij} P_{ij} \log P_{ij}$ is the entropy of the plan, and $\varepsilon > 0$ is the regularization strength.

### Structured cost matrix

The cost matrix $\mathbf{C} \in \mathbb{R}^{n_s \times n_t}$ encodes three types of information about the plausibility of each cell-to-cell transport:

$$C_{ij} = \underbrace{\frac{\|\mathbf{x}_i - \mathbf{x}_j\|_2^2}{C_{\max}}}_{\text{geometric distance}} + \underbrace{\lambda_t \cdot \mathbb{1}[\tau_i > \tau_j]}_{\text{pseudotime penalty}} + \underbrace{\lambda_k \cdot \mathbb{1}[j \notin \text{KNN}(i)]}_{\text{locality penalty}}$$

where:

- **Geometric distance**: the squared Euclidean distance in PCA space, normalized by the maximum pairwise distance $C_{\max}$ within the pair. This is the primary cost — cells that are far apart in expression space are expensive to transport.

- **Pseudotime penalty** ($\lambda_t$): an additive penalty applied when the source cell has a later pseudotime than the target cell ($\tau_i > \tau_j$). This discourages backward transport, encoding the assumption that the biological process progresses forward in time. The indicator function $\mathbb{1}[\cdot]$ is 1 when the condition is true and 0 otherwise.

- **KNN locality penalty** ($\lambda_k$): an additive penalty applied when the target cell $j$ is not among the $k$-nearest neighbors of the source cell $i$ in the global KNN graph. This reinforces manifold locality, preventing transport to cells that are geometrically nearby in Euclidean distance but separated along the manifold (e.g., cells on opposite sides of a fold).

### Velocity extraction

From the transport plan $\mathbf{P}^*$, the raw OT velocity for each source cell $i$ is computed as the expected displacement under the transport:

$$\mathbf{v}_i^{\text{OT}} = \sum_{j \in w_{\text{tgt}}} \hat{P}^*_{ij} \left(\mathbf{x}_j - \mathbf{x}_i\right)$$

where $\hat{P}^*_{ij}$ is the row-normalized transport plan:

$$\hat{P}^*_{ij} = \frac{P^*_{ij}}{\sum_{j} P^*_{ij}}$$

This normalization ensures that each source cell's velocity is a proper weighted average of displacements to target cells, independent of the total mass transported.

### Aggregation across windows

A cell may appear as a source in multiple window pairs (due to overlap between consecutive windows and membership in a spatial cluster). The final raw velocity for cell $i$ is the average across all contributing pairs:

$$\mathbf{v}_i = \frac{1}{|\mathcal{W}_i|} \sum_{p \in \mathcal{W}_i} \mathbf{v}_i^{(p)}$$

where $\mathcal{W}_i$ is the set of window pairs in which cell $i$ served as a source, and $\mathbf{v}_i^{(p)}$ is the velocity from pair $p$.

### Confidence estimation

A confidence score is computed for each cell based on the number of window pairs that contributed to its velocity:

$$c_i = \frac{|\mathcal{W}_i|}{\max_j |\mathcal{W}_j|}$$

Cells with high confidence ($c_i \approx 1$) have velocity estimates supported by multiple overlapping windows and are therefore more reliable. Cells with zero confidence ($c_i = 0$) were not included as a source in any window pair — their velocity will be entirely interpolated by the smoothing stage.

### Safety mechanisms

Two additional safety mechanisms are applied during OT velocity computation:

- **Dead-end detection**: if a source cell has no KNN neighbors in the target window (the local adjacency submatrix has a zero row), its velocity is set to zero. This prevents cells at the boundary of a trajectory from acquiring spurious velocities pointing to distant, unrelated cells.

- **Sinkhorn fallback**: if the Sinkhorn algorithm fails to converge (which can happen with poorly conditioned cost matrices), the computation falls back to the exact OT solution via the Earth Mover's Distance (EMD). If both fail, the pair contributes zero velocity.

---

## Stage 4: Neural Velocity Field Smoothing

### Motivation

The raw OT velocity field has three limitations that motivate a smoothing step:

1. **Discreteness**: velocity is defined only at cell positions, not as a continuous field. It cannot be evaluated at arbitrary points in expression space.

2. **Noise**: individual transport plans are subject to statistical noise from finite sample sizes within windows. Cells near the edge of windows may have velocity estimates based on few transport connections.

3. **Gaps**: cells that were not included as a source in any window pair (typically those at the end of the pseudotime range or in very small spatial clusters) have no OT velocity estimate at all.

### Network architecture

A small multi-layer perceptron (MLP) $f_\theta$ is trained to learn a smooth, continuous velocity function over the entire expression space. The network takes as input the concatenation of a cell's PCA coordinates and pseudotime, and outputs a predicted velocity vector:

$$\hat{\mathbf{v}}_i = f_\theta(\mathbf{x}_i, \tau_i) : \mathbb{R}^{D+1} \to \mathbb{R}^D$$

The architecture consists of two hidden layers with SiLU (Sigmoid Linear Unit) activation functions:

$$f_\theta(\mathbf{x}, \tau) = \mathbf{W}_3 \, \sigma(\mathbf{W}_2 \, \sigma(\mathbf{W}_1 [\mathbf{x}; \tau] + \mathbf{b}_1) + \mathbf{b}_2) + \mathbf{b}_3$$

where $[\mathbf{x}; \tau]$ denotes concatenation, $\sigma$ is the SiLU activation, and $\{\mathbf{W}_l, \mathbf{b}_l\}$ are learnable weight matrices and bias vectors. The hidden dimension is typically 128.

The inclusion of pseudotime $\tau_i$ as an input feature is critical: it allows the network to predict different velocities for cells that occupy the same spatial position in PCA space but are at different stages of the biological process. This directly addresses the temporal mixing problem that plagues methods operating in reduced-dimensional spaces.

### Loss function

The network is trained by minimizing a composite loss with two terms:

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{reg}} + \lambda_s \cdot \mathcal{L}_{\text{smooth}}$$

**Regression loss** (data fidelity): the network's prediction should match the raw OT velocity where it is available, weighted by the confidence score:

$$\mathcal{L}_{\text{reg}} = \frac{\sum_{i=1}^{N} c_i \cdot \|f_\theta(\mathbf{x}_i, \tau_i) - \mathbf{v}_i^{\text{OT}}\|_2^2}{\sum_{i=1}^{N} c_i}$$

The confidence weighting ensures that cells with reliable OT estimates (covered by many windows) contribute more to the regression target than cells with uncertain estimates (covered by few windows). Cells with $c_i = 0$ contribute nothing to this term.

**Smoothness loss** (spatial coherence): the predicted velocity at each cell should be consistent with the predictions at its $K$ nearest neighbors:

$$\mathcal{L}_{\text{smooth}} = \frac{1}{NK} \sum_{i=1}^{N} \sum_{j \in \text{KNN}(i)} \|f_\theta(\mathbf{x}_i, \tau_i) - f_\theta(\mathbf{x}_j, \tau_j)\|_2^2$$

This term enforces local coherence of the velocity field, penalizing rapid changes between neighboring cells. For zero-confidence cells, this is the only term that influences their predicted velocity, effectively interpolating from the learned field of their neighbors.

The balance between data fidelity and smoothness is controlled by $\lambda_s$ (typically 0.1–1.0). Higher values produce smoother fields at the cost of less fidelity to the raw OT estimates. Lower values preserve the OT structure but may retain noise.

### Training procedure

The network is trained using the Adam optimizer with cosine annealing learning rate schedule for 100–300 epochs. Each epoch samples a minibatch of cells, computes both losses jointly (the batch cells and all their KNN neighbors are passed through the network in a single forward pass for efficiency), and updates the weights via backpropagation.

After training, the smoothed velocity is extracted for all cells by evaluating $f_\theta$ at each cell's coordinates. The raw OT velocity is preserved as a backup for comparison.

### Relationship to flow matching

The smoothing step is conceptually related to flow matching methods, where a neural network learns a vector field that transports between distributions. In classical conditional flow matching, the network is supervised by displacement vectors between OT-coupled sample pairs drawn from source and target distributions. In VelOT's smoothing, the OT displacement vectors serve an analogous supervisory role — they define the target velocity that the network should approximate. The smoothness constraint plays the role of a regularizer that ensures the learned field is physically plausible (locally coherent), similar to the continuity constraints in flow-based generative models.

However, VelOT's smoothing is deliberately simpler than full conditional flow matching. There is no base distribution, no flow time parameter $t \in [0, 1]$, and no ODE integration during training. The network simply learns a spatial mapping from coordinates to velocity vectors. This simplicity is a feature: it avoids the complexity and potential instability of training normalizing flows while still producing a smooth, continuous velocity field.

---

## Stage 5: Velocity Projection to UMAP

The velocity field is computed and stored in $D$-dimensional PCA space, where the geometric structure of the data is most faithfully represented. For visualization in 2D UMAP coordinates, a local linear projection is applied.

For each cell $i$, the Jacobian of the PCA-to-UMAP mapping is estimated from its $K$ nearest neighbors by solving the least-squares problem:

$$\mathbf{A}_i = \arg\min_{\mathbf{A}} \|\Delta\mathbf{U}_i - \Delta\mathbf{X}_i \mathbf{A}\|_F^2$$

where $\Delta\mathbf{X}_i \in \mathbb{R}^{K \times D}$ is the matrix of PCA displacements from cell $i$ to its neighbors, and $\Delta\mathbf{U}_i \in \mathbb{R}^{K \times 2}$ is the corresponding matrix of UMAP displacements. The solution is:

$$\mathbf{A}_i = (\Delta\mathbf{X}_i^\top \Delta\mathbf{X}_i)^{-1} \Delta\mathbf{X}_i^\top \Delta\mathbf{U}_i$$

The UMAP velocity is then:

$$\mathbf{v}_i^{\text{UMAP}} = \mathbf{v}_i^{\text{PCA}} \cdot \mathbf{A}_i$$

This local linear approximation respects the nonlinear geometry of the UMAP embedding by estimating a different Jacobian at each point, rather than applying a single global transformation. It preserves directional information more faithfully than alternatives such as applying UMAP to displaced coordinates, which would introduce distortions from the nonlinear projection.

---

## Summary of Notation

| Symbol | Description |
|--------|-------------|
| $N$ | Number of cells |
| $G$ | Number of genes |
| $D$ | Number of PCA components |
| $\mathbf{X} \in \mathbb{R}^{N \times D}$ | Cell coordinates in PCA space |
| $\tau_i \in [0, 1]$ | Pseudotime of cell $i$ (from DPT) |
| $K$ | Number of spatial clusters |
| $W$ | Window size (cells per window) |
| $f_{\text{overlap}}$ | Overlap fraction between consecutive windows |
| $\mathbf{P}^* \in \mathbb{R}^{n_s \times n_t}$ | Optimal transport plan |
| $\mathbf{C} \in \mathbb{R}^{n_s \times n_t}$ | Cost matrix |
| $\varepsilon$ | Sinkhorn entropy regularization |
| $\lambda_t$ | Pseudotime backward penalty |
| $\lambda_k$ | KNN locality penalty |
| $\mathbf{v}_i^{\text{OT}} \in \mathbb{R}^D$ | Raw OT velocity of cell $i$ |
| $c_i \in [0, 1]$ | Confidence score of cell $i$ |
| $f_\theta$ | Neural network velocity function |
| $\lambda_s$ | Smoothness regularization weight |
| $\mathbf{v}_i^{\text{UMAP}} \in \mathbb{R}^2$ | UMAP-projected velocity for visualization |

---

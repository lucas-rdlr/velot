"""
velot.datasets — Built-in Datasets
===================================

Real datasets (require scVelo)::

    adata = velot.datasets.pancreas()
    adata = velot.datasets.erythroid()
    adata = velot.datasets.dentategyrus()
    adata = velot.datasets.bonemarrow()

Synthetic datasets (no dependencies)::

    adata = velot.datasets.synthetic_linear()
    adata = velot.datasets.synthetic_bifurcation()
    adata = velot.datasets.synthetic_tree(topology)
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import scanpy as sc
import anndata as ad
from anndata import AnnData


# =====================================================================
# Real datasets (require scVelo)
# =====================================================================


def _require_scvelo():
    """Import scVelo or raise a helpful error."""
    try:
        import scvelo as scv
        return scv
    except ImportError:
        raise ImportError(
            "scVelo is required to load this dataset. "
            "Install with: pip install scvelo"
        )


def pancreas() -> AnnData:
    """
    Pancreas endocrinogenesis (day 15.5) — scVelo.

    ~3,700 cells, 4 main lineages branching from endocrine progenitors.
    Temporally mixed — progenitors and differentiated cells coexist
    spatially. Good stress test for VelOT's spatial-temporal windowing.

    Clusters: Ductal, Ngn3 low EP, Ngn3 high EP, Pre-endocrine,
    Alpha, Beta, Delta, Epsilon.

    Reference: Bastidas-Ponce et al. (2019).
    """
    scv = _require_scvelo()
    return scv.datasets.pancreas()


def erythroid() -> AnnData:
    """
    Gastrulation erythroid lineage — scVelo.

    Erythroid cells from mouse gastrulation. Well-ordered linear
    trajectory where simpler velocity methods already perform well.
    Good positive control.

    Reference: Pijuan-Sala et al. (2019).
    """
    scv = _require_scvelo()
    return scv.datasets.gastrulation_erythroid()


def dentategyrus() -> AnnData:
    """
    Dentate gyrus neurogenesis — scVelo.

    ~2,900 cells from the hippocampal dentate gyrus with neurogenic
    trajectory. Contains both cycling progenitors and mature neurons.

    Reference: Hochgerner et al. (2018).
    """
    scv = _require_scvelo()
    return scv.datasets.dentategyrus()


def bonemarrow() -> AnnData:
    """
    Human bone marrow — scVelo.

    ~5,780 cells from human bone marrow with multiple hematopoietic
    lineages. Complex branching structure.

    Reference: Setty et al. (2019).
    """
    scv = _require_scvelo()
    return scv.datasets.bonemarrow()


# =====================================================================
# Synthetic datasets
# =====================================================================


def synthetic_linear(
    densities: Sequence[int] = (100, 100),
    positions: Sequence[float] = (1.0, 2.0),
    noise_level: float = 0.05,
    extra_dimensions: int = 0,
    n_neighbors: int = 50,
    seed: int = 42,
) -> AnnData:
    """
    Synthetic linear trajectory.

    Generates cells along a straight line with optional noise and
    extra dimensions. Each segment gets a unique cell type label.

    Parameters
    ----------
    densities
        Number of cells in each segment.
    positions
        End coordinate of each segment (first starts at 0).
    noise_level
        Standard deviation of Gaussian noise added to coordinates.
        Set to 0 for a perfectly clean line.
    extra_dimensions
        Number of additional noisy dimensions beyond the 2D plane.
    n_neighbors
        Number of neighbors for the KNN graph.
    seed
        Random seed.

    Returns
    -------
    AnnData with:
      - ``adata.obs['celltype']`` : segment labels
      - ``adata.obs['true_pseudotime']`` : ground truth pseudotime in [0,1]
      - ``adata.obsm['X_pca']`` : PCA coordinates
      - ``adata.obsm['X_umap']`` : UMAP coordinates

    Example
    -------
    ::

        adata = velot.datasets.synthetic_linear(
            densities=[200, 200, 200],
            positions=[1, 2, 3],
            noise_level=0.05,
        )
    """
    np.random.seed(seed)

    if len(densities) != len(positions):
        raise ValueError("densities and positions must have the same length.")

    t_list, x_list, y_list, labels_list = [], [], [], []

    start_pos = 0.0
    for i, (n_cells, end_pos) in enumerate(zip(densities, positions)):
        t = np.random.uniform(start_pos, end_pos, n_cells)
        t_list.append(t)
        x_list.append(t.copy())
        y_list.append(np.zeros(n_cells))
        labels_list.append(np.array([f"Segment_{i+1}"] * n_cells))
        start_pos = end_pos

    return _build_synthetic_adata(
        t_list, x_list, y_list, labels_list,
        noise_level=noise_level,
        extra_dimensions=extra_dimensions,
        n_neighbors=n_neighbors,
    )


def synthetic_bifurcation(
    root_density: int = 200,
    root_position: float = 1.0,
    branch_densities: Sequence[int] = (100, 100),
    branch_positions: Sequence[float] = (2.0, 2.0),
    branch_slopes: Sequence[float] = (2.0, -2.0),
    noise_level: float = 0.05,
    extra_dimensions: int = 0,
    n_neighbors: int = 50,
    seed: int = 42,
) -> AnnData:
    """
    Synthetic bifurcating trajectory.

    A root trunk splits into multiple branches diverging in the
    y-direction. This is the classic test case for velocity methods
    at branching points.

    Parameters
    ----------
    root_density
        Number of cells in the root trunk.
    root_position
        End coordinate of the root (starts at 0).
    branch_densities
        Number of cells per branch.
    branch_positions
        End coordinate of each branch.
    branch_slopes
        Y-axis slope of each branch (controls separation).
    noise_level
        Gaussian noise standard deviation.
    extra_dimensions
        Extra noisy dimensions.
    n_neighbors
        KNN neighbors.
    seed
        Random seed.

    Returns
    -------
    AnnData with celltype labels and ground truth pseudotime.

    Example
    -------
    ::

        adata = velot.datasets.synthetic_bifurcation(
            root_density=300,
            branch_densities=[200, 200],
            branch_slopes=[2, -2],
        )
    """
    np.random.seed(seed)

    if not (len(branch_densities) == len(branch_positions) == len(branch_slopes)):
        raise ValueError(
            "branch_densities, branch_positions, and branch_slopes "
            "must have the same length."
        )

    t_list, x_list, y_list, labels_list = [], [], [], []

    # Root trunk
    t_root = np.random.uniform(0, root_position, root_density)
    t_list.append(t_root)
    x_list.append(t_root.copy())
    y_list.append(np.zeros(root_density))
    labels_list.append(np.array(["Root"] * root_density))

    # Branches
    for i, (n_cells, end_pos, slope) in enumerate(
        zip(branch_densities, branch_positions, branch_slopes)
    ):
        t = np.random.uniform(root_position, end_pos, n_cells)
        t_list.append(t)
        x_list.append(t.copy())
        y_list.append((t - root_position) * slope)
        labels_list.append(np.array([f"Branch_{i+1}"] * n_cells))

    return _build_synthetic_adata(
        t_list, x_list, y_list, labels_list,
        noise_level=noise_level,
        extra_dimensions=extra_dimensions,
        n_neighbors=n_neighbors,
    )


def synthetic_tree(
    topology: Sequence[dict],
    noise_level: float = 0.05,
    extra_dimensions: int = 0,
    n_neighbors: int = 50,
    seed: int = 42,
) -> AnnData:
    """
    Synthetic tree-structured trajectory.

    Build arbitrarily complex branching topologies by specifying
    a list of branch definitions. Each branch starts where its
    parent ended.

    Parameters
    ----------
    topology
        List of dicts, each with keys:
          - ``name`` (str): branch label
          - ``parent`` (str or None): parent branch name
          - ``n_cells`` (int): number of cells
          - ``length`` (float): pseudotime duration
          - ``slope`` (float): y-axis slope
    noise_level
        Gaussian noise standard deviation.
    extra_dimensions
        Extra noisy dimensions.
    n_neighbors
        KNN neighbors.
    seed
        Random seed.

    Returns
    -------
    AnnData with celltype labels and ground truth pseudotime.

    Example
    -------
    ::

        topology = [
            {"name": "Root",     "parent": None,   "n_cells": 400,
             "length": 0.5, "slope": 0.0},
            {"name": "Branch_A", "parent": "Root", "n_cells": 200,
             "length": 0.5, "slope": 2.0},
            {"name": "Branch_B", "parent": "Root", "n_cells": 200,
             "length": 0.5, "slope": -2.0},
            {"name": "Branch_C", "parent": "Branch_B", "n_cells": 150,
             "length": 0.3, "slope": 0.0},
        ]
        adata = velot.datasets.synthetic_tree(topology)
    """
    np.random.seed(seed)

    t_list, x_list, y_list, labels_list = [], [], [], []
    end_states = {}

    for branch in topology:
        name = branch["name"]
        parent = branch["parent"]
        n_cells = branch["n_cells"]
        length = branch["length"]
        slope = branch["slope"]

        if parent is None:
            start_t, start_y = 0.0, 0.0
        else:
            if parent not in end_states:
                raise ValueError(
                    f"Parent '{parent}' not found. Parents must be "
                    f"listed before their children in the topology."
                )
            start_t = end_states[parent]["t"]
            start_y = end_states[parent]["y"]

        end_t = start_t + length

        t = np.random.uniform(start_t, end_t, n_cells)
        x = t.copy()
        y = start_y + (t - start_t) * slope

        end_states[name] = {
            "t": end_t,
            "y": start_y + length * slope,
        }

        t_list.append(t)
        x_list.append(x)
        y_list.append(y)
        labels_list.append(np.array([name] * n_cells))

    return _build_synthetic_adata(
        t_list, x_list, y_list, labels_list,
        noise_level=noise_level,
        extra_dimensions=extra_dimensions,
        n_neighbors=n_neighbors,
    )


# =====================================================================
# Shared builder for synthetic datasets
# =====================================================================


def _build_synthetic_adata(
    t_list: list,
    x_list: list,
    y_list: list,
    labels_list: list,
    noise_level: float = 0.05,
    extra_dimensions: int = 0,
    n_neighbors: int = 50,
) -> AnnData:
    """
    Build an AnnData object from lists of coordinates.

    Handles noise, extra dimensions, PCA, KNN, and UMAP.
    """
    t_all = np.concatenate(t_list)
    x_all = np.concatenate(x_list)
    y_all = np.concatenate(y_list)
    labels_all = np.concatenate(labels_list)

    total = len(t_all)

    # Add noise
    if noise_level > 0:
        x_all = x_all + np.random.normal(0, noise_level, total)
        y_all = y_all + np.random.normal(0, noise_level, total)

    # Build coordinate matrix
    X = np.column_stack([x_all, y_all])

    # Add extra noisy dimensions
    for _ in range(extra_dimensions):
        X = np.column_stack([X, np.random.normal(0, noise_level, total)])

    # Create AnnData
    adata = ad.AnnData(X)

    # Store metadata
    adata.obs["celltype"] = labels_all
    adata.obs["celltype"] = adata.obs["celltype"].astype("category")

    # Ground truth pseudotime normalized to [0, 1]
    pt = t_all.copy()
    pt = (pt - pt.min()) / (pt.max() - pt.min() + 1e-12)
    adata.obs["true_pseudotime"] = pt

    # Store the "clean" 2D coordinates for reference
    adata.obsm["X_spatial_true"] = np.column_stack([
        np.concatenate(x_list),  # without noise
        np.concatenate(y_list),
    ])

    # PCA
    if extra_dimensions > 0:
        sc.tl.pca(adata)
    else:
        # In 2D, PCA is just the coordinates themselves
        adata.obsm["X_pca"] = X.copy()

    # KNN graph + UMAP
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    sc.tl.umap(adata)

    return adata


# =====================================================================
# Quick topology templates
# =====================================================================


def symmetric_bifurcation(
    n_root: int = 400,
    n_branch: int = 200,
    noise_level: float = 0.05,
    extra_dimensions: int = 0,
    seed: int = 42,
) -> AnnData:
    """
    Convenience: symmetric Y-shaped bifurcation.

    ::

        Root ──┬── Branch_A (up)
               └── Branch_B (down)
    """
    return synthetic_bifurcation(
        root_density=n_root,
        branch_densities=[n_branch, n_branch],
        branch_slopes=[2.0, -2.0],
        noise_level=noise_level,
        extra_dimensions=extra_dimensions,
        seed=seed,
    )


def multifurcation(
    n_root: int = 400,
    n_per_branch: int = 150,
    n_branches: int = 3,
    noise_level: float = 0.05,
    extra_dimensions: int = 0,
    seed: int = 42,
) -> AnnData:
    """
    Convenience: root splitting into multiple branches.

    Branches are evenly spaced in angle.
    """
    angles = np.linspace(-np.pi / 3, np.pi / 3, n_branches)
    slopes = [float(np.tan(a)) for a in angles]

    return synthetic_bifurcation(
        root_density=n_root,
        branch_densities=[n_per_branch] * n_branches,
        branch_positions=[2.0] * n_branches,
        branch_slopes=slopes,
        noise_level=noise_level,
        extra_dimensions=extra_dimensions,
        seed=seed,
    )


def linear_with_cycle(
    n_linear: int = 300,
    n_cycle: int = 200,
    noise_level: float = 0.05,
    seed: int = 42,
) -> AnnData:
    """
    Convenience: linear trajectory that loops back (challenging case).

    This creates a dataset where pseudotime doesn't correspond to
    a single direction — cells at the end are spatially near the
    start. Tests whether VelOT can handle cyclic structures.
    """
    np.random.seed(seed)

    # Linear part
    t_lin = np.random.uniform(0, 1.0, n_linear)
    x_lin = t_lin * 3.0
    y_lin = np.zeros(n_linear)

    # Cyclic part: arc from end back toward start
    t_cyc = np.random.uniform(1.0, 2.0, n_cycle)
    phase = (t_cyc - 1.0) * np.pi  # 0 to pi
    x_cyc = 3.0 * np.cos(phase)
    y_cyc = 1.5 * np.sin(phase)

    t_all = np.concatenate([t_lin, t_cyc])
    x_all = np.concatenate([x_lin, x_cyc])
    y_all = np.concatenate([y_lin, y_cyc])

    if noise_level > 0:
        x_all += np.random.normal(0, noise_level, len(t_all))
        y_all += np.random.normal(0, noise_level, len(t_all))

    labels = np.array(
        ["Linear"] * n_linear + ["Cycle"] * n_cycle
    )

    X = np.column_stack([x_all, y_all])
    adata = ad.AnnData(X)
    adata.obs["celltype"] = labels
    adata.obs["celltype"] = adata.obs["celltype"].astype("category")

    pt = t_all.copy()
    pt = (pt - pt.min()) / (pt.max() - pt.min() + 1e-12)
    adata.obs["true_pseudotime"] = pt

    adata.obsm["X_pca"] = X.copy()
    sc.pp.neighbors(adata, n_neighbors=50)
    sc.tl.umap(adata)

    return adata


def synthetic_cycle(
    densities: Sequence[int] = (100, 100, 100, 100),
    spans: Optional[Sequence[float]] = None,
    names: Optional[Sequence[str]] = None,
    n_rotations: float = 1.0,
    x_amplitude: float = 1.0,
    y_amplitude: float = 1.0,
    z_drift: float = 0.0,
    noise_level: float = 0.05,
    extra_dimensions: int = 0,
    n_neighbors: int = 50,
    seed: int = 42,
) -> AnnData:
    """
    Synthetic cyclic / helicoidal trajectory.

    Cells follow a circular or helical path parameterized by
    pseudotime. This creates a challenging scenario where cells
    at the end are spatially near the start, and pseudotime
    does not correspond to a single spatial direction.

    The trajectory is divided into segments (cell types), each
    with controllable density and pseudotime span.

    Parameters
    ----------
    densities
        Number of cells per segment. Length determines the
        number of cell types.
    spans
        Pseudotime span of each segment. Must sum to any positive
        value (will be normalized internally). If None, segments
        are equally spaced.
    names
        Labels for each segment. If None, generates
        "Phase_1", "Phase_2", etc.
    n_rotations
        Number of full 2π rotations over the entire pseudotime
        range. 1.0 = one full circle, 2.0 = two loops, 0.5 = half
        circle, etc.
    x_amplitude
        Semi-axis length in x direction (controls elliptical shape).
    y_amplitude
        Semi-axis length in y direction.
    z_drift
        If > 0, adds a z-coordinate that increases linearly with
        pseudotime, turning the circle into a helix. The value
        controls the total height of the helix.
    noise_level
        Gaussian noise standard deviation.
    extra_dimensions
        Additional noisy dimensions.
    n_neighbors
        KNN neighbors.
    seed
        Random seed.

    Returns
    -------
    AnnData with:
      - ``adata.obs['celltype']`` : phase/segment labels
      - ``adata.obs['true_pseudotime']`` : ground truth pseudotime in [0,1]
      - ``adata.obsm['X_pca']`` : PCA coordinates
      - ``adata.obsm['X_umap']`` : UMAP coordinates

    Examples
    --------
    Simple circle with 4 equal phases::

        adata = velot.datasets.synthetic_cycle()

    Elliptical orbit with unequal phases::

        adata = velot.datasets.synthetic_cycle(
            densities=[200, 50, 200, 50],
            spans=[0.4, 0.1, 0.4, 0.1],
            names=["G1", "S", "G2", "M"],
            x_amplitude=2.0,
            y_amplitude=1.0,
        )

    Double helix (two full turns with vertical drift)::

        adata = velot.datasets.synthetic_cycle(
            densities=[150, 150, 150, 150],
            n_rotations=2.0,
            z_drift=3.0,
        )

    Dense start, sparse end::

        adata = velot.datasets.synthetic_cycle(
            densities=[400, 200, 100, 50],
            names=["Early", "Mid", "Late", "Terminal"],
        )
    """
    np.random.seed(seed)

    n_segments = len(densities)

    # Validate and set defaults
    if names is None:
        names = [f"Cluster {i+1}" for i in range(n_segments)]
    if len(names) != n_segments:
        raise ValueError(
            f"Length of names ({len(names)}) must match "
            f"length of densities ({n_segments})."
        )

    if spans is None:
        spans = [1.0 / n_segments] * n_segments
    if len(spans) != n_segments:
        raise ValueError(
            f"Length of spans ({len(spans)}) must match "
            f"length of densities ({n_segments})."
        )

    # Normalize spans to [0, 1]
    spans = np.array(spans, dtype=np.float64)
    spans = spans / spans.sum()

    # Build cumulative boundaries
    boundaries = np.concatenate([[0.0], np.cumsum(spans)])

    # Generate cells per segment
    t_list = []
    x_list = []
    y_list = []
    labels_list = []

    for i in range(n_segments):
        n_cells = densities[i]
        t_start = boundaries[i]
        t_end = boundaries[i + 1]

        # Uniform pseudotime within this segment
        t = np.random.uniform(t_start, t_end, n_cells)

        # Map pseudotime to angle
        angle = t * n_rotations * 2.0 * np.pi

        # Parametric coordinates
        x = x_amplitude * np.cos(angle)
        y = y_amplitude * np.sin(angle)

        t_list.append(t)
        x_list.append(x)
        y_list.append(y)
        labels_list.append(np.array([names[i]] * n_cells))

    # Concatenate
    t_all = np.concatenate(t_list)
    x_all = np.concatenate(x_list)
    y_all = np.concatenate(y_list)
    labels_all = np.concatenate(labels_list)
    total = len(t_all)

    # Add noise
    if noise_level > 0:
        x_all = x_all + np.random.normal(0, noise_level, total)
        y_all = y_all + np.random.normal(0, noise_level, total)

    # Build coordinate matrix
    if z_drift > 0:
        z_all = t_all * z_drift
        if noise_level > 0:
            z_all = z_all + np.random.normal(0, noise_level, total)
        X = np.column_stack([x_all, y_all, z_all])
    else:
        X = np.column_stack([x_all, y_all])

    # Add extra noisy dimensions
    for _ in range(extra_dimensions):
        X = np.column_stack([
            X, np.random.normal(0, noise_level if noise_level > 0 else 0.05, total),
        ])

    # Create AnnData
    adata = ad.AnnData(X)

    adata.obs["celltype"] = labels_all
    adata.obs["celltype"] = adata.obs["celltype"].astype("category")

    # Ground truth pseudotime
    pt = t_all.copy()
    pt = (pt - pt.min()) / (pt.max() - pt.min() + 1e-12)
    adata.obs["true_pseudotime"] = pt

    # Store clean 2D coordinates for reference
    adata.obsm["X_spatial_true"] = np.column_stack([
        np.concatenate([x_amplitude * np.cos(t * n_rotations * 2 * np.pi)
                        for t in t_list]),
        np.concatenate([y_amplitude * np.sin(t * n_rotations * 2 * np.pi)
                        for t in t_list]),
    ])

    # PCA
    if X.shape[1] > 2:
        sc.pp.pca(adata)
    else:
        adata.obsm["X_pca"] = X.copy()

    # KNN + UMAP
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_pca")
    sc.tl.umap(adata)

    return adata
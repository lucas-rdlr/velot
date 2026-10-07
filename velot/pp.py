"""
Prepares an AnnData object for the VelOT pipeline.
Follows the scanpy convention: functions modify adata in place
and return it for optional chaining.

Typical usage::

    import velot

    # All-in-one
    velot.pp.prepare(adata, n_pcs=30, root_cluster="Root")

    # Or step by step
    velot.pp.normalize(adata)
    velot.pp.select_genes(adata, n_hvg=2000)
    velot.pp.pca(adata, n_pcs=30)
    velot.pp.neighbors(adata)
    velot.pp.umap(adata)
    velot.pp.pseudotime(adata, root_cluster="Root")
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import scanpy as sc
from anndata import AnnData


# ======================================================================
# Individual preprocessing steps
# ======================================================================


def normalize(adata: AnnData, target_sum: float = 1e4) -> AnnData:
    """
    Total-count normalize and log-transform.

    Parameters
    ----------
    adata
        Annotated data matrix with raw counts in adata.X.
    target_sum
        Target total counts per cell after normalization.

    Returns
    -------
    adata, modified in place.
    """
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata


def select_genes(
    adata: AnnData,
    n_hvg: int = 2000,
    flavor: str = "seurat",
) -> AnnData:
    """
    Select highly variable genes and subset the data.

    Parameters
    ----------
    adata
        Annotated data matrix (should be log-normalized).
    n_hvg
        Number of highly variable genes to keep.
    flavor
        HVG selection method passed to scanpy.

    Returns
    -------
    adata, subsetted to HVGs in place.
    """
    if adata.n_vars <= n_hvg:
        return adata

    sc.pp.highly_variable_genes(
        adata, n_top_genes=n_hvg, flavor=flavor, subset=False,
    )
    adata._inplace_subset_var(adata.var["highly_variable"].values)
    return adata


def scale(adata: AnnData) -> AnnData:
    """
    Scale to zero mean and unit variance per gene.

    This ensures PCA captures correlation structure rather than
    being dominated by highly-expressed genes.

    Returns
    -------
    adata, modified in place.
    """
    sc.pp.scale(adata)
    return adata


def pca(adata: AnnData, n_pcs: int = 30) -> AnnData:
    """
    Compute PCA embedding.

    The PCA coordinates (adata.obsm['X_pca']) are the space where
    velocity will be computed. This is NOT just for visualization.

    Returns
    -------
    adata with adata.obsm['X_pca'] populated.
    """
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack")
    return adata


def neighbors(
    adata: AnnData,
    n_pcs: int = 30,
    n_neighbors: int = 30,
) -> AnnData:
    """
    Compute the KNN graph.

    The KNN graph is used downstream for:
      - OT cost matrix locality penalties
      - Velocity smoothing (KNN consistency)
      - Pseudotime computation (DPT)

    Returns
    -------
    adata with adata.obsp['connectivities'] and adata.obsp['distances'].
    """
    sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    return adata


def umap(adata: AnnData) -> AnnData:
    """
    Compute UMAP embedding (for visualization only).

    VelOT does NOT compute velocity in UMAP space. UMAP coordinates
    are used only for plotting.

    Returns
    -------
    adata with adata.obsm['X_umap'] populated.
    """
    sc.tl.umap(adata)
    return adata


def pseudotime(
    adata: AnnData,
    *,
    key: Optional[str] = None,
    root_cluster: Optional[str] = None,
    root_cell: Optional[int] = None,
    cluster_key: str = "clusters",
    root_selection: str = "distal",
    basis: str = "X_pca",
    dpt_impl: str = "velot",
) -> AnnData:
    """
    Compute or load pseudotime ordering.

    Three modes:
      1. ``key`` provided: load precomputed pseudotime from adata.obs[key]
      2. ``root_cell`` provided: run DPT from that cell index
      3. ``root_cluster`` provided: run DPT from a cell chosen inside that
         cluster by ``root_selection``

    DPT (Diffusion Pseudotime) computes temporal ordering from the
    expression geometry alone — no velocity or spliced/unspliced
    information is used. This keeps the velocity estimation independent.

    Parameters
    ----------
    adata
        Must already have the KNN graph computed (run ``velot.pp.neighbors``
        first).
    key
        Column name in adata.obs with precomputed pseudotime.
    root_cluster
        Cluster name to use as root for DPT.
    root_cell
        Cell index to use as root for DPT. Overrides root_cluster.
    cluster_key
        Column in adata.obs with cluster labels.
    root_selection
        How the root cell is chosen inside ``root_cluster``.
        ``"distal"`` (default) takes the cell furthest, in diffusion
        space, from everything outside the cluster; ``"medoid"`` takes the
        most central cell of the cluster; ``"first"`` takes the first cell
        in storage order, which is what VelOT did up to v1.0 and depends
        on how the file happens to be sorted. See
        :func:`_select_root_cell`. The chosen cell is recorded in
        ``adata.uns['velot_root']``.

    Returns
    -------
    adata with adata.obs['pseudotime'] in [0, 1].
    """
    if key is not None:
        pt = _load_pseudotime(adata, key)
    else:
        pt = _compute_dpt(
            adata,
            root_cluster=root_cluster,
            root_cell=root_cell,
            cluster_key=cluster_key,
            root_selection=root_selection,
            basis=basis,
            dpt_impl=dpt_impl,
        )

    adata.obs["pseudotime"] = _normalize_01(pt)
    return adata


# ======================================================================
# All-in-one convenience function
# ======================================================================


def prepare(
    adata: AnnData,
    *,
    n_pcs: int = 30,
    n_neighbors: int = 30,
    n_hvg: Optional[int] = 2000,
    pseudotime_key: Optional[str] = None,
    root_cluster: Optional[str] = None,
    root_cell: Optional[int] = None,
    cluster_key: str = "clusters",
    root_selection: str = "distal",
    basis: str = "X_pca",
    dpt_impl: str = "velot",
    do_normalize: bool = True,
    copy: bool = True,
) -> AnnData:
    """
    Full preprocessing in one call.

    Runs: normalize → select_genes → scale → PCA → neighbors →
    UMAP → pseudotime.

    Parameters
    ----------
    adata
        Raw or partially processed AnnData object.
    n_pcs
        Number of principal components.
    n_neighbors
        Number of neighbors for KNN graph.
    n_hvg
        Number of HVGs to select. None to skip.
    pseudotime_key
        Precomputed pseudotime column name. If provided, DPT is skipped.
    root_cluster
        Root cluster for DPT.
    root_cell
        Root cell index for DPT.
    cluster_key
        Column with cluster labels.
    do_normalize
        Whether to normalize + log1p. Set False if already done.
    copy
        Whether to operate on a copy of adata.

    Returns
    -------
    Preprocessed adata.

    Example
    -------
    ::

        import velot
        import scvelo as scv

        adata = scv.datasets.pancreas()
        velot.pp.prepare(adata, root_cluster="Ductal", cluster_key="clusters")
    """
    if copy:
        adata = adata.copy()

    if do_normalize:
        normalize(adata)

    if n_hvg is not None:
        select_genes(adata, n_hvg=n_hvg)

    scale(adata)
    pca(adata, n_pcs=n_pcs)
    neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    umap(adata)

    pseudotime(
        adata,
        key=pseudotime_key,
        root_cluster=root_cluster,
        root_cell=root_cell,
        cluster_key=cluster_key,
        root_selection=root_selection,
        basis=basis,
        dpt_impl=dpt_impl,
    )

    # Store pipeline parameters for reproducibility
    adata.uns["velot_params"] = {
        "n_pcs": n_pcs,
        "n_neighbors": n_neighbors,
        "n_hvg": n_hvg,
        "pseudotime_source": pseudotime_key or "dpt",
    }

    return adata


# ======================================================================
# Internal helpers
# ======================================================================


def _load_pseudotime(adata: AnnData, key: str) -> np.ndarray:
    """Load and validate a precomputed pseudotime column."""
    if key not in adata.obs:
        raise KeyError(
            f"Pseudotime column '{key}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    pt = adata.obs[key].values.astype(np.float64)

    n_nan = np.isnan(pt).sum()
    if n_nan > 0:
        warnings.warn(
            f"Pseudotime column '{key}' contains {n_nan} NaN values. "
            f"Assigning them maximum pseudotime.",
            stacklevel=2,
        )
        pt[np.isnan(pt)] = np.nanmax(pt)

    return pt


def _select_root_cell(
    adata: AnnData,
    mask: np.ndarray,
    method: str = "distal",
    basis: str = "X_pca",
    min_purity: float = 0.5,
    max_reference: int = 5000,
    random_state: int = 0,
) -> int:
    """
    Pick the root cell inside the root cluster.

    ``"distal"`` (default)
        The cell of the root cluster whose mean diffusion distance to all
        cells OUTSIDE the cluster is largest: the end of the cluster that
        faces away from everything else, which is where a trajectory
        starts. Mean distance is a convex function of position, so its
        maximum is always attained at the edge of the cluster - on the
        side pointing away from the bulk of the other cells. For a trunk
        that is exactly the free end. For a round root cluster with
        branches leaving in several directions the "start" is arguably its
        centre instead, and this rule will still return an edge cell, on
        whichever side has the largest gap between branches; use
        ``"medoid"`` there.
    ``"medoid"``
        The most central cell of the root cluster (smallest mean diffusion
        distance to the other cells OF THE CLUSTER). Order-independent
        like ``"distal"``, and the better choice when the root cluster is
        a ball that radiates in several directions.
    ``"first"``
        The first cell of the cluster in storage order — the scanpy
        tutorial convention, and what VelOT used up to v1.0. It depends on
        the row order of the file, so two copies of the same data sorted
        differently give different pseudotime. Kept only to reproduce
        earlier runs.

    Distances are measured in ``basis`` — the representation the rest of
    the pipeline works in — and NOT in diffusion space. Diffusion
    components beyond the first few are local noise modes on which single
    cells take extreme values, and because ``sc.tl.diffmap`` stores
    unweighted eigenvectors whose eigenvalues here are all ~0.9-1.0, those
    modes carry the same weight as the components that encode the
    trajectory. Maximising a mean distance in that space reliably returns
    an outlier rather than the end of the trunk: on the dyngen bifurcation
    the old rule picked a cell that drew 48 % of its distance from a
    single component (DC7, on which it was the global maximum) and 2 %
    from DC1, producing a pseudotime ANTI-correlated with the simulation
    time (Spearman -0.21 against +0.87 for the rule below).

    ``min_purity`` additionally drops candidates whose kNN neighbourhood
    is less than that fraction root-cluster: a cell labelled with the root
    cluster but sitting inside another one is a labelling artefact, not
    the start of the trajectory. Set it to 0 to disable.

    For speed, the cells outside the cluster are subsampled to
    ``max_reference`` (deterministically).
    """
    idx = np.where(mask)[0]
    if method == "first":
        return int(idx[0])
    if method not in ("distal", "medoid"):
        raise ValueError("root_selection must be 'distal', 'medoid' or "
                         f"'first', got {method!r}")
    if len(idx) == 1:
        return int(idx[0])

    if basis in adata.obsm:
        D = np.asarray(adata.obsm[basis])
    elif "X_diffmap" in adata.obsm:
        warnings.warn(f"basis {basis!r} not in adata.obsm; falling back to "
                      "X_diffmap for root selection, which is sensitive to "
                      "outliers in high-order diffusion components.",
                      RuntimeWarning)
        D = np.asarray(adata.obsm["X_diffmap"])[:, 1:]
    else:
        raise ValueError(f"root selection needs {basis!r} (or X_diffmap) "
                         "in adata.obsm.")

    # Drop candidates that are not really in the cluster: a cell carrying
    # the root label whose neighbours are almost all other clusters is a
    # labelling artefact and must not anchor the pseudotime.
    candidates = idx
    if min_purity > 0 and "distances" in adata.obsp:
        conn = adata.obsp["distances"]
        purity = np.array([
            float(np.mean(mask[conn[i].indices])) if conn[i].nnz else 0.0
            for i in idx])
        keep = purity >= min_purity
        if keep.any():
            if not keep.all():
                warnings.warn(
                    f"{int((~keep).sum())} of {len(idx)} root-cluster cells "
                    f"have a neighbourhood less than {min_purity:.0%} "
                    "root-cluster and were excluded from root selection.",
                    RuntimeWarning)
            candidates = idx[keep]

    if method == "medoid":
        others = candidates              # central within its own cluster
    else:
        others = np.where(~mask)[0]
        if len(others) == 0:             # a single cluster: use every cell
            others = np.arange(adata.n_obs)
    if len(others) > max_reference:
        rng = np.random.default_rng(random_state)
        others = rng.choice(others, max_reference, replace=False)

    from scipy.spatial.distance import cdist
    mean_dist = cdist(D[candidates], D[others]).mean(axis=1)
    pick = np.argmin(mean_dist) if method == "medoid" else np.argmax(mean_dist)
    return int(candidates[int(pick)])


# scanpy's sc.tl.dpt down-weights every diffusion component whose
# eigenvalue is >= this to a weight of 1, instead of lambda/(1-lambda).
# The cut-off is hard-coded in scanpy.tools._dpt.DPT._get_dpt_row and is
# described there as a leftover "contribution from the stationary state".
_SCANPY_DPT_CUTOFF = 0.9994


def _dpt_from_diffmap(adata: AnnData, root: int, n_dcs: int = 10) -> np.ndarray:
    """Diffusion pseudotime as the diffusion distance from ``root``.

    ``dpt(i) = || (lambda_k/(1-lambda_k)) (psi_k(root) - psi_k(i)) ||``
    over the non-trivial components, which is the definition in Haghverdi
    et al. 2016.

    This is computed here rather than with ``sc.tl.dpt`` because that
    implementation gives every component with an eigenvalue at or above
    0.9994 a weight of 1 instead of lambda/(1-lambda). On a slow-mixing
    graph - many cells strung along a low-dimensional trajectory - the
    leading components sit above that cut-off, so the very components
    that carry the ordering are suppressed by two to three orders of
    magnitude relative to later, noisier ones, and the pseudotime stops
    tracking the trajectory. On the dyngen synthetics reduced to 5 PCs
    this turned a component correlating 0.96 with the simulation time
    into one contributing less than a noise component correlating 0.19,
    giving a pseudotime that peaked in the middle of the root cluster.
    """
    psi = np.asarray(adata.obsm["X_diffmap"])
    evals = np.asarray(adata.uns["diffmap_evals"])
    n = min(n_dcs, psi.shape[1], evals.size)
    psi = psi[:, :n]

    # component 0 is the trivial one; drop anything numerically at 1
    use = np.zeros(n, dtype=bool)
    weights = np.zeros(n)
    for j in range(1, n):
        if evals[j] < 1.0 - 1e-12:
            weights[j] = evals[j] / (1.0 - evals[j])
            use[j] = True
    if not use.any():
        raise ValueError("no usable diffusion components for DPT")

    delta = (psi[:, use] - psi[root, use]) * weights[use]
    dpt = np.sqrt((delta ** 2).sum(axis=1))

    # cells the root cannot reach have no pseudotime, as in scanpy
    if "connectivities" in adata.obsp:
        from scipy.sparse.csgraph import connected_components
        ncomp, comp = connected_components(adata.obsp["connectivities"],
                                           directed=False)
        if ncomp > 1:
            dpt[comp != comp[root]] = np.inf

    suppressed = int((evals[1:n] >= _SCANPY_DPT_CUTOFF).sum())
    if suppressed:
        warnings.warn(
            f"{suppressed} leading diffusion component(s) have an "
            f"eigenvalue >= {_SCANPY_DPT_CUTOFF}; scanpy's sc.tl.dpt would "
            "have given them unit weight and returned a pseudotime that "
            "does not follow the trajectory. VelOT uses the unmodified "
            "diffusion distance instead.", RuntimeWarning, stacklevel=3)
    return dpt


def _compute_dpt(
    adata: AnnData,
    *,
    root_cluster: Optional[str] = None,
    root_cell: Optional[int] = None,
    cluster_key: str = "clusters",
    root_selection: str = "distal",
    basis: str = "X_pca",
    dpt_impl: str = "velot",
) -> np.ndarray:
    """
    Compute Diffusion Pseudotime.

    DPT builds a diffusion process on the KNN graph and measures
    diffusion distance from a root cell. It uses only the expression
    geometry — no velocity or spliced/unspliced information.

    The diffusion map is computed first, because the root cell is chosen
    from that geometry (see :func:`_select_root_cell`).
    """
    sc.tl.diffmap(adata)

    if root_cell is not None:
        adata.uns["iroot"] = int(root_cell)

    elif root_cluster is not None:
        if cluster_key not in adata.obs:
            raise ValueError(
                f"cluster_key '{cluster_key}' not found in adata.obs. "
                f"Available: {list(adata.obs.columns)}"
            )
        mask = (adata.obs[cluster_key].astype(str) == str(root_cluster)).values
        if not mask.any():
            available = sorted(
                adata.obs[cluster_key].astype(str).unique().tolist()
            )
            raise ValueError(
                f"Root cluster '{root_cluster}' not found. "
                f"Available: {available}"
            )
        adata.uns["iroot"] = _select_root_cell(
            adata, mask, method=root_selection, basis=basis)
        adata.uns["velot_root"] = {
            "cluster": str(root_cluster),
            "cell": int(adata.uns["iroot"]),
            "cell_name": str(adata.obs_names[adata.uns["iroot"]]),
            "selection": root_selection,
        }

    else:
        raise ValueError(
            "For pseudotime, provide one of: "
            "key (precomputed), root_cluster, or root_cell."
        )

    if dpt_impl == "scanpy":
        sc.tl.dpt(adata)
        pt = adata.obs["dpt_pseudotime"].values.astype(np.float64)
    elif dpt_impl == "velot":
        pt = _dpt_from_diffmap(adata, int(adata.uns["iroot"]))
        adata.obs["dpt_pseudotime"] = pt          # scanpy-compatible key
    else:
        raise ValueError("dpt_impl must be 'velot' or 'scanpy', "
                         f"got {dpt_impl!r}")

    # DPT can produce infinities for disconnected components
    inf_mask = ~np.isfinite(pt)
    if inf_mask.any():
        warnings.warn(
            f"DPT produced {inf_mask.sum()} non-finite values "
            f"(disconnected cells). Assigning maximum pseudotime.",
            stacklevel=2,
        )
        pt[inf_mask] = np.nanmax(pt[np.isfinite(pt)])

    return pt


def _normalize_01(pt: np.ndarray) -> np.ndarray:
    """Normalize an array to [0, 1], handling edge cases."""
    pt = pt.astype(np.float64)
    pt_min = np.nanmin(pt)
    pt_max = np.nanmax(pt)

    if pt_max - pt_min < 1e-12:
        warnings.warn(
            "Pseudotime has near-zero range. Returning all zeros.",
            stacklevel=2,
        )
        return np.zeros_like(pt)

    return (pt - pt_min) / (pt_max - pt_min)
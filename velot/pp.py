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
) -> AnnData:
    """
    Compute or load pseudotime ordering.

    Three modes:
      1. ``key`` provided: load precomputed pseudotime from adata.obs[key]
      2. ``root_cell`` provided: run DPT from that cell index
      3. ``root_cluster`` provided: run DPT from the first cell in that cluster

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


def _compute_dpt(
    adata: AnnData,
    *,
    root_cluster: Optional[str] = None,
    root_cell: Optional[int] = None,
    cluster_key: str = "clusters",
) -> np.ndarray:
    """
    Compute Diffusion Pseudotime.

    DPT builds a diffusion process on the KNN graph and measures
    diffusion distance from a root cell. It uses only the expression
    geometry — no velocity or spliced/unspliced information.
    """
    if root_cell is not None:
        adata.uns["iroot"] = int(root_cell)

    elif root_cluster is not None:
        if cluster_key not in adata.obs:
            raise ValueError(
                f"cluster_key '{cluster_key}' not found in adata.obs. "
                f"Available: {list(adata.obs.columns)}"
            )
        mask = adata.obs[cluster_key].astype(str) == str(root_cluster)
        if not mask.any():
            available = sorted(
                adata.obs[cluster_key].astype(str).unique().tolist()
            )
            raise ValueError(
                f"Root cluster '{root_cluster}' not found. "
                f"Available: {available}"
            )
        adata.uns["iroot"] = int(np.where(mask)[0][0])

    else:
        raise ValueError(
            "For pseudotime, provide one of: "
            "key (precomputed), root_cluster, or root_cell."
        )

    sc.tl.diffmap(adata)
    sc.tl.dpt(adata)

    pt = adata.obs["dpt_pseudotime"].values.astype(np.float64)

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
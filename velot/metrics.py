"""
velot.metrics — Evaluation
==========================

Metrics for evaluating velocity field quality.

Adapted from UniTVelo (Gao et al. 2022) and the original VelOT pipeline.

Usage::

    import velot

    # Cross-boundary direction correctness
    edges = [("Root", "Branch_A"), ("Root", "Branch_B")]
    scores = velot.metrics.cross_boundary_correctness(
        adata, edges, cluster_key="clusters",
    )

    # Inner-cluster coherence
    scores = velot.metrics.inner_cluster_coherence(adata, cluster_key="clusters")
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import issparse
from sklearn.metrics.pairwise import cosine_similarity
from anndata import AnnData


# =====================================================================
# Neighbor utilities
# =====================================================================


def _get_neighbor_indices(adata: AnnData) -> list:
    """Extract per-cell neighbor indices from the KNN graph."""
    if "connectivities" not in adata.obsp:
        raise ValueError(
            "KNN graph not found. Run velot.pp.neighbors() or "
            "sc.pp.neighbors() first."
        )

    conn = adata.obsp["connectivities"]
    return [conn.getrow(i).indices for i in range(conn.shape[0])]


def _keep_type(
    adata: AnnData,
    nodes: np.ndarray,
    target: str,
    cluster_key: str,
) -> np.ndarray:
    """Filter node indices to keep only those matching a cluster label."""
    return nodes[adata.obs[cluster_key].values[nodes] == target]


def _safe_dense(X):
    """Convert to dense array if sparse."""
    if issparse(X):
        return np.asarray(X.toarray())
    return np.asarray(X)


# =====================================================================
# Cross-boundary direction correctness
# =====================================================================


def cross_boundary_correctness(
    adata: AnnData,
    cluster_edges: Sequence[Tuple[str, str]],
    cluster_key: str = "clusters",
    velocity_key: str = "velocity_umap",
    embedding_key: str = "X_umap",
    majority_vote: bool = True,
    return_raw: bool = False,
) -> dict:
    """
    Cross-Boundary Direction Correctness Score (CBDir).

    For each directed edge A→B, measures whether cells of type A
    have velocity vectors pointing toward their neighbors of type B.

    Adapted from UniTVelo (Gao et al. 2022).

    Parameters
    ----------
    adata
        Annotated data matrix with velocity computed.
    cluster_edges
        List of (source, target) cluster name pairs defining
        expected transitions. E.g., ``[("Root", "Branch_A")]``.
    cluster_key
        Column in adata.obs with cluster labels.
    velocity_key
        Key in adata.obsm with velocity vectors.
    embedding_key
        Key in adata.obsm with cell coordinates (same space as velocity).
    majority_vote
        If True, score is fraction of neighbors with positive cosine.
        If False, score is mean cosine similarity.
    return_raw
        If True, return per-cell scores for each edge (for box plots).
        If False, return mean score per edge and global mean.

    Returns
    -------
    If ``return_raw=True``:
        dict mapping (source, target) to list of per-cell scores.
    If ``return_raw=False``:
        tuple of (dict mapping edges to mean scores, global mean score).

    Example
    -------
    ::

        edges = [("Root", "Branch_A"), ("Root", "Branch_B")]
        scores, mean_score = velot.metrics.cross_boundary_correctness(
            adata, edges, cluster_key="clusters",
        )
    """
    if velocity_key not in adata.obsm:
        raise ValueError(
            f"Velocity key '{velocity_key}' not found in adata.obsm."
        )

    X_emb = _safe_dense(adata.obsm[embedding_key])
    V_emb = _safe_dense(adata.obsm[velocity_key])

    all_nbs = _get_neighbor_indices(adata)
    scores = {}
    all_scores = {}

    for u, v in cluster_edges:
        # Select source cells
        sel = np.where(adata.obs[cluster_key].astype(str) == str(u))[0]
        if len(sel) == 0:
            all_scores[(u, v)] = []
            scores[(u, v)] = float("nan")
            continue

        type_score = []
        for cell_idx in sel:
            # Find neighbors of target type
            nbs = all_nbs[cell_idx]
            boundary = _keep_type(adata, nbs, str(v), cluster_key)

            if len(boundary) == 0:
                continue

            # Direction from cell to boundary neighbors
            x_pos = X_emb[cell_idx]
            x_vel = V_emb[cell_idx]
            target_coords = X_emb[boundary]

            position_diff = target_coords - x_pos
            dir_scores = cosine_similarity(
                position_diff, x_vel.reshape(1, -1),
            ).flatten()

            if majority_vote:
                type_score.append(np.mean(dir_scores > 0))
            else:
                type_score.append(np.mean(dir_scores))

        all_scores[(u, v)] = type_score
        scores[(u, v)] = float(np.mean(type_score)) if type_score else float("nan")

    if return_raw:
        return all_scores

    global_mean = np.nanmean(list(scores.values())) if scores else float("nan")
    return scores, float(global_mean)


# =====================================================================
# Inner-cluster coherence
# =====================================================================


def inner_cluster_coherence(
    adata: AnnData,
    cluster_key: str = "clusters",
    velocity_key: str = "velocity_umap",
    return_raw: bool = False,
) -> dict:
    """
    Inner-Cluster Coherence Score (ICCoh).

    For each cluster, measures how consistent the velocity vectors
    are among cells of the same type by computing mean pairwise
    cosine similarity within each cell's same-type neighbors.

    Adapted from UniTVelo (Gao et al. 2022).

    Parameters
    ----------
    adata
        Annotated data matrix with velocity computed.
    cluster_key
        Column in adata.obs with cluster labels.
    velocity_key
        Key in adata.obsm with velocity vectors.
    return_raw
        If True, return per-cell scores per cluster.
        If False, return mean score per cluster and global mean.

    Returns
    -------
    If ``return_raw=True``:
        dict mapping cluster name to list of per-cell scores.
    If ``return_raw=False``:
        tuple of (dict of mean scores, global mean score).

    Example
    -------
    ::

        scores, mean = velot.metrics.inner_cluster_coherence(
            adata, cluster_key="clusters",
        )
    """
    if velocity_key not in adata.obsm:
        raise ValueError(
            f"Velocity key '{velocity_key}' not found in adata.obsm."
        )

    V = _safe_dense(adata.obsm[velocity_key])
    all_nbs = _get_neighbor_indices(adata)
    clusters = adata.obs[cluster_key].astype(str).values
    unique_clusters = np.unique(clusters)

    scores = {}
    all_scores = {}

    for cat in unique_clusters:
        sel = np.where(clusters == cat)[0]

        cat_score = []
        for cell_idx in sel:
            nbs = all_nbs[cell_idx]
            same_type = _keep_type(adata, nbs, cat, cluster_key)

            if len(same_type) == 0:
                continue

            sim = cosine_similarity(
                V[cell_idx : cell_idx + 1], V[same_type],
            ).mean()
            cat_score.append(float(sim))

        all_scores[cat] = cat_score
        scores[cat] = float(np.mean(cat_score)) if cat_score else float("nan")

    if return_raw:
        return all_scores

    global_mean = np.nanmean(list(scores.values())) if scores else float("nan")
    return scores, float(global_mean)


# =====================================================================
# Summary display
# =====================================================================

def summary(
    adata: AnnData,
    cluster_edges: Optional[Sequence[Tuple[str, str]]] = None,
    cluster_key: str = "clusters",
    embedding_key: str = "X_umap",
    velocity_key: str = "velot_velocity_umap",
    print_results: bool = True
) -> dict:
    """
    Compute and print a summary of velocity metrics.

    Returns both aggregated means and raw per-cell scores
    so that the output can be passed directly to
    ``velot.pl.metric_summary()`` for box plots.

    Parameters
    ----------
    adata
        Annotated data matrix with velocity computed.
    cluster_edges
        Transition edges for CBDir. If None, CBDir is skipped.
    cluster_key
        Cluster column.
    velocity_key
        Velocity key.

    Returns
    -------
    Dictionary with:
      - ``"iccoh"``: dict of cluster → list of per-cell scores
      - ``"iccoh_mean"``: global mean ICCoh
      - ``"cbdir"``: dict of (src, tgt) → list of per-cell scores
      - ``"cbdir_mean"``: global mean CBDir

    Example
    -------
    ::

        results = velot.metrics.summary(adata, cluster_edges=edges)
        velot.pl.metric_summary(results)
    """
    results = {}

    # ICCoh — get raw scores
    iccoh_raw = inner_cluster_coherence(
        adata, cluster_key=cluster_key,
        velocity_key=velocity_key, return_raw=True,
    )
    iccoh_means = {
        k: float(np.mean(v)) if len(v) > 0 else float("nan")
        for k, v in iccoh_raw.items()
    }
    iccoh_median = {
        k: float(np.median(v)) if len(v) > 0 else float("nan")
        for k, v in iccoh_raw.items()
    }
    iccoh_global = float(np.nanmean(list(iccoh_means.values()))) if iccoh_means else float("nan")
    iccoh_global_median = float(np.nanmedian(list(iccoh_median.values()))) if iccoh_median else float("nan")

    results["iccoh"] = iccoh_raw
    results["iccoh_mean"] = iccoh_global
    results["iccoh_median"] = iccoh_global_median

    if print_results:
        print("=" * 50)
        print("VelOT Velocity Metrics")
        print("=" * 50)
        print(f"\nInner-Cluster Coherence (mean: {iccoh_global:.3f}):")
        for cat in sorted(iccoh_means.keys()):
            print(f"  {str(cat):>25s}: {iccoh_means[cat]:.3f}")
        
        print(f"\nInner-Cluster Coherence (median: {iccoh_global_median:.3f}):")
        for cat in sorted(iccoh_median.keys()):
            print(f"  {str(cat):>25s}: {iccoh_median[cat]:.3f}")

    # CBDir — get raw scores
    if cluster_edges is not None:
        cbdir_raw = cross_boundary_correctness(
            adata, cluster_edges,
            cluster_key=cluster_key,
            embedding_key=embedding_key,
            velocity_key=velocity_key,
            return_raw=True,
        )
        cbdir_means = {
            k: float(np.mean(v)) if len(v) > 0 else float("nan")
            for k, v in cbdir_raw.items()
        }
        cbdir_median = {
            k: float(np.median(v)) if len(v) > 0 else float("nan")
            for k, v in cbdir_raw.items()
        }
        cbdir_global = float(np.nanmean(list(cbdir_means.values()))) if cbdir_means else float("nan")
        cbdir_global_median = float(np.nanmedian(list(cbdir_median.values()))) if cbdir_median else float("nan")

        results["cbdir"] = cbdir_raw
        results["cbdir_mean"] = cbdir_global
        results["cbdir_median"] = cbdir_global_median

        if print_results:
            print(f"\nCross-Boundary Correctness (mean: {cbdir_global:.3f}):")
            for (u, v) in cbdir_raw.keys():
                print(f"  {u:>15s} → {v:<15s}: {cbdir_means[(u, v)]:.3f}")

    if print_results:
        print("=" * 50)
    return results
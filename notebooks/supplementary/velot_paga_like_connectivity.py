
"""
VelOT PAGA-like connectivity analysis for VAMPFlow and QuantumMSM meta-states.

This module adapts ideas from the PAGA connectivity-measure notebook to the
VelOT-MetaFlow / VelOT-QuantumMSM framework, without using CellRank or GPCCA.

Main goal
---------
Given cell-level meta-states from either:
    adata.obs["velot_vampflow_state"]
    adata.obs["velot_quantum_msm_state"]

and a VelOT/MetaFlow vector field:
    adata.obsm["velot_quantum_metaflow_velocity_latent_final"]
or
    adata.obsm["velot_metaflow_velocity_latent_final"]
or
    adata.obsm["velot_velocity_umap"] lifted to latent space,

compute a quantitative PAGA-like abstraction:

1. Undirected PAGA-like connectivity
   observed inter-meta-state kNN edges / degree-corrected null expectation.

2. Permutation/null-normalized connectivity
   empirical p-values, z-scores and BH q-values for inter-state edges.

3. Directed vector-field connectivity
   group-to-group flux induced by the learned VelOT/OT/Quantum/VAMP vector field.

4. Directionality
   antisymmetric flux between pairs:
       D_ab = (F_ab - F_ba) / (F_ab + F_ba)

5. Quantitative summaries
   group size, pseudotime, divergence, speed, curl, source/sink scores,
   incoming/outgoing flux, entropy.

6. Side-by-side comparison of VAMPFlow and QuantumMSM graphs
   ARI/NMI between assignments, edge overlap, transition correlation,
   directionality correlation.

7. Publication-ready figures
   - PAGA-like connectivity heatmap
   - null z-score heatmap
   - directed flux heatmap
   - directionality heatmap
   - abstracted graph on embedding
   - vector-field graph on embedding
   - edge significance scatter
   - group diagnostics
   - VAMP vs Quantum comparison summary

Dependencies
------------
pip install numpy pandas scipy scikit-learn matplotlib seaborn networkx scanpy anndata

Optional:
    velot_quantum_metaflow.py in the same folder if you want to call
    run_full_metaflow_plus_connectivity(...).

Example
-------
from velot_paga_like_connectivity import (
    PagaLikeConnectivityConfig,
    run_velot_paga_like_connectivity_analysis,
)

cfg = PagaLikeConnectivityConfig(
    group_keys=["velot_vampflow_state", "velot_quantum_msm_state"],
    latent_key="X_velot_quantum_metaflow_latent_scaled",
    embedding_key="X_umap",
    velocity_key="velot_quantum_metaflow_velocity_latent_final",
    pseudotime_key="velot_quantum_metaflow_pseudotime",
    output_dir="velot_paga_like_connectivity_outputs",
)

out = run_velot_paga_like_connectivity_analysis(adata, cfg)
"""

from __future__ import annotations

import os
import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

try:
    import scanpy as sc
except Exception:
    sc = None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PagaLikeConnectivityConfig:
    # Existing outputs from velot_quantum_metaflow.py
    group_keys: Sequence[str] = field(
        default_factory=lambda: ["velot_vampflow_state", "velot_quantum_msm_state"]
    )

    # Data representation
    latent_key: Optional[str] = "X_velot_quantum_metaflow_latent_scaled"
    fallback_latent_keys: Sequence[str] = field(
        default_factory=lambda: [
            "X_velot_quantum_metaflow_latent_scaled",
            "X_velot_metaflow_latent_scaled",
            "X_metaflow_latent_scaled",
            "X_scVI",
            "X_pca",
        ]
    )
    embedding_key: str = "X_umap"
    pseudotime_key: Optional[str] = "velot_quantum_metaflow_pseudotime"

    # Velocity fields
    velocity_key: Optional[str] = "velot_quantum_metaflow_velocity_latent_final"
    fallback_velocity_keys: Sequence[str] = field(
        default_factory=lambda: [
            "velot_quantum_metaflow_velocity_latent_final",
            "velot_metaflow_velocity_latent_final",
            "velot_metaflow_velocity_latent_from_velot",
            "metaflow_velocity_latent",
            "velot_velocity_latent",
        ]
    )
    velot_velocity_umap_key: str = "velot_velocity_umap"
    velocity_to_latent_k: int = 30
    velocity_to_latent_ridge: float = 1e-3

    # Graph construction
    use_existing_connectivities: bool = True
    connectivities_key: str = "connectivities"
    n_neighbors: int = 30
    graph_metric: str = "euclidean"
    weighted_knn: bool = True
    symmetrize_graph: bool = True

    # PAGA-like null model
    n_permutations: int = 200
    permutation_random_state: int = 0
    min_group_size: int = 5
    edge_confidence_threshold: float = 0.15
    edge_qvalue_threshold: float = 0.10

    # Directed vector-field transitions
    directed_k: int = 30
    directed_temperature: float = 0.25
    directed_distance_scale: Optional[float] = None
    directed_clip_negative_alignment: bool = True
    directed_include_self: bool = False

    # Plotting
    output_dir: str = "velot_paga_like_connectivity_outputs"
    figure_dpi: int = 300
    max_arrows_plot: int = 1200
    graph_edge_threshold: float = 0.15
    graph_directed_edge_threshold: float = 0.05
    max_edge_labels: int = 60

    # Reproducibility
    random_state: int = 0


# =============================================================================
# Basic utilities
# =============================================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def savefig(path: str, cfg: PagaLikeConnectivityConfig) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=cfg.figure_dpi, bbox_inches="tight")
    plt.close()


def row_normalize(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return M / (M.sum(axis=1, keepdims=True) + eps)


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x)
    if x.std() > 0:
        x = (x - x.mean()) / (x.std() + 1e-8)
    return x.astype(np.float32)


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR for a matrix/array."""
    p = np.asarray(pvals, dtype=float).ravel()
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty_like(ranked)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[i] = prev
    out = np.empty_like(q)
    out[order] = q
    return out.reshape(np.asarray(pvals).shape)


def sanitize_key_for_filename(key: str) -> str:
    return key.replace("/", "_").replace(" ", "_").replace(":", "_")


def get_available_key(adata, preferred: Optional[str], fallbacks: Sequence[str], where: str = "obsm") -> Optional[str]:
    store = getattr(adata, where)
    if preferred is not None and preferred in store:
        return preferred
    for key in fallbacks:
        if key in store:
            return key
    return None


# =============================================================================
# Representation and velocity handling
# =============================================================================

def get_latent(adata, cfg: PagaLikeConnectivityConfig) -> Tuple[np.ndarray, str]:
    key = get_available_key(adata, cfg.latent_key, cfg.fallback_latent_keys, where="obsm")
    if key is not None:
        Z = np.asarray(adata.obsm[key], dtype=np.float32)
        return Z, key

    if sc is not None:
        if "X_pca" not in adata.obsm:
            sc.tl.pca(adata, n_comps=50, svd_solver="arpack")
        Z = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
        return Z, "X_pca_auto"

    raise KeyError(
        "No latent representation found. Provide cfg.latent_key or one of "
        f"{list(cfg.fallback_latent_keys)} in adata.obsm."
    )


def get_embedding(adata, Z: np.ndarray, cfg: PagaLikeConnectivityConfig) -> Tuple[np.ndarray, str]:
    if cfg.embedding_key in adata.obsm:
        E = np.asarray(adata.obsm[cfg.embedding_key], dtype=np.float32)
        if E.ndim == 2 and E.shape[1] >= 2:
            return E[:, :2], cfg.embedding_key
    warnings.warn(f"Embedding key {cfg.embedding_key!r} not found; using first two latent dimensions.")
    return Z[:, :2].astype(np.float32), "latent_first2"


def get_pseudotime(adata, cfg: PagaLikeConnectivityConfig) -> Optional[np.ndarray]:
    candidate_keys = []
    if cfg.pseudotime_key is not None:
        candidate_keys.append(cfg.pseudotime_key)
    candidate_keys += [
        "velot_quantum_metaflow_pseudotime",
        "velot_metaflow_pseudotime",
        "metaflow_pseudotime",
        "velot_pseudotime",
        "dpt_pseudotime",
        "pseudotime",
    ]
    for key in candidate_keys:
        if key in adata.obs:
            pt = pd.to_numeric(adata.obs[key], errors="coerce").to_numpy(dtype=float)
            pt = np.nan_to_num(pt, nan=np.nanmedian(pt[np.isfinite(pt)]))
            pt = (pt - pt.min()) / (pt.max() - pt.min() + 1e-8)
            return pt.astype(np.float32)
    return None


def lift_umap_velocity_to_latent(
    E: np.ndarray,
    Z: np.ndarray,
    V_emb: np.ndarray,
    k: int = 30,
    ridge: float = 1e-3,
) -> np.ndarray:
    n, d = Z.shape
    k = min(k, n)
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(E)
    _, ind = nn.kneighbors(E)
    V_lat = np.zeros((n, d), dtype=np.float32)
    I = np.eye(E.shape[1], dtype=np.float32)

    for i in range(n):
        nb = ind[i]
        X = E[nb] - E[i]
        Y = Z[nb] - Z[i]
        XtX = X.T @ X + ridge * I
        XtY = X.T @ Y
        try:
            B = np.linalg.solve(XtX, XtY)
        except np.linalg.LinAlgError:
            B = np.linalg.lstsq(XtX, XtY, rcond=1e-6)[0]
        V_lat[i] = V_emb[i] @ B

    return np.nan_to_num(V_lat.astype(np.float32))


def get_velocity(
    adata,
    Z: np.ndarray,
    E: np.ndarray,
    cfg: PagaLikeConnectivityConfig,
) -> Tuple[Optional[np.ndarray], str, Optional[np.ndarray]]:
    """
    Returns:
        V_latent, velocity_source, V_embedding_if_available
    """
    key = get_available_key(adata, cfg.velocity_key, cfg.fallback_velocity_keys, where="obsm")
    V_emb = None

    if cfg.velot_velocity_umap_key in adata.obsm:
        V_emb = np.asarray(adata.obsm[cfg.velot_velocity_umap_key], dtype=np.float32)
        if V_emb.ndim == 2 and V_emb.shape[1] >= 2:
            V_emb = V_emb[:, :2]
        else:
            V_emb = None

    if key is not None:
        V = np.asarray(adata.obsm[key], dtype=np.float32)
        if V.shape != Z.shape:
            warnings.warn(
                f"Velocity key {key!r} has shape {V.shape}, expected {Z.shape}; "
                "falling back to lifted UMAP velocity if available."
            )
        else:
            return np.nan_to_num(V.astype(np.float32)), key, V_emb

    if V_emb is not None:
        V_lift = lift_umap_velocity_to_latent(
            E, Z, V_emb,
            k=cfg.velocity_to_latent_k,
            ridge=cfg.velocity_to_latent_ridge,
        )
        return V_lift, f"{cfg.velot_velocity_umap_key}->latent_lift", V_emb

    warnings.warn("No velocity field found. Directed vector-field analysis will be skipped.")
    return None, "none", None


# =============================================================================
# KNN graph and PAGA-like connectivity
# =============================================================================

def build_or_get_knn_graph(
    adata,
    Z: np.ndarray,
    cfg: PagaLikeConnectivityConfig,
) -> Tuple[sp.csr_matrix, str]:
    """
    Returns a symmetric weighted cell-cell graph.
    If adata.obsp["connectivities"] exists and is requested, use it.
    Otherwise build a kNN graph from Z.
    """
    if cfg.use_existing_connectivities and cfg.connectivities_key in adata.obsp:
        A = adata.obsp[cfg.connectivities_key].tocsr().astype(np.float32)
        A.setdiag(0)
        A.eliminate_zeros()
        if cfg.symmetrize_graph:
            A = 0.5 * (A + A.T)
        return A.tocsr(), f"adata.obsp[{cfg.connectivities_key!r}]"

    nn = NearestNeighbors(
        n_neighbors=min(cfg.n_neighbors + 1, Z.shape[0]),
        metric=cfg.graph_metric,
    )
    nn.fit(Z)
    dist, ind = nn.kneighbors(Z)

    rows, cols, vals = [], [], []
    # skip self neighbor at column 0 when present
    for i in range(Z.shape[0]):
        neigh = ind[i]
        d = dist[i]
        mask = neigh != i
        neigh = neigh[mask][:cfg.n_neighbors]
        d = d[mask][:cfg.n_neighbors]
        if len(neigh) == 0:
            continue
        if cfg.weighted_knn:
            sigma = np.median(d) + 1e-8
            w = np.exp(-(d ** 2) / (2 * sigma ** 2))
        else:
            w = np.ones_like(d)
        rows.extend([i] * len(neigh))
        cols.extend(neigh.tolist())
        vals.extend(w.tolist())

    A = sp.csr_matrix((vals, (rows, cols)), shape=(Z.shape[0], Z.shape[0]), dtype=np.float32)
    A.setdiag(0)
    A.eliminate_zeros()
    if cfg.symmetrize_graph:
        A = A.maximum(A.T)
    return A.tocsr(), f"knn_Z_k{cfg.n_neighbors}"


def encode_groups(groups: Sequence[object]) -> Tuple[np.ndarray, List[str]]:
    s = pd.Series(groups).astype(str)
    cats = list(pd.Categorical(s).categories)
    codes = pd.Categorical(s, categories=cats).codes.astype(int)
    return codes, cats


def filter_small_groups(
    codes: np.ndarray,
    names: List[str],
    min_group_size: int,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Removes small groups by assigning them to -1. Returns valid mask.
    """
    counts = pd.Series(codes).value_counts().to_dict()
    valid_groups = [i for i in range(len(names)) if counts.get(i, 0) >= min_group_size]
    valid_set = set(valid_groups)

    new_codes = np.full_like(codes, fill_value=-1)
    new_names = []
    remap = {}
    for new_i, old_i in enumerate(valid_groups):
        remap[old_i] = new_i
        new_names.append(names[old_i])

    for i, c in enumerate(codes):
        if c in valid_set:
            new_codes[i] = remap[c]

    valid_mask = new_codes >= 0
    return new_codes, new_names, valid_mask


def get_undirected_edge_table(A: sp.csr_matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = sp.triu(A, k=1).tocoo()
    return A.row.astype(int), A.col.astype(int), A.data.astype(float)


def group_edge_counts_from_edges(
    rows: np.ndarray,
    cols: np.ndarray,
    weights: np.ndarray,
    codes: np.ndarray,
    K: int,
) -> np.ndarray:
    C = np.zeros((K, K), dtype=np.float64)
    gr = codes[rows]
    gc = codes[cols]
    valid = (gr >= 0) & (gc >= 0)
    gr = gr[valid]
    gc = gc[valid]
    ww = weights[valid]
    for a, b, w in zip(gr, gc, ww):
        C[a, b] += w
        if a != b:
            C[b, a] += w
    return C


def degree_corrected_expected_counts(A: sp.csr_matrix, codes: np.ndarray, K: int) -> np.ndarray:
    deg = np.asarray(A.sum(axis=1)).ravel().astype(float)
    valid = codes >= 0
    total_undirected_weight = deg[valid].sum() / 2.0 + 1e-12
    vols = np.zeros(K, dtype=float)
    for k in range(K):
        vols[k] = deg[codes == k].sum()

    E = np.zeros((K, K), dtype=float)
    for a in range(K):
        for b in range(K):
            if a == b:
                # Expected within-group undirected edge count.
                E[a, b] = (vols[a] * vols[a]) / (4.0 * total_undirected_weight + 1e-12)
            else:
                # Expected inter-group edge count, stored symmetrically.
                E[a, b] = (vols[a] * vols[b]) / (2.0 * total_undirected_weight + 1e-12)
    return E


def permutation_null_counts(
    rows: np.ndarray,
    cols: np.ndarray,
    weights: np.ndarray,
    codes: np.ndarray,
    K: int,
    n_perm: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Permutes group labels across cells, preserving group sizes, and recomputes
    inter-group counts. Returns mean, std, p_upper for observed enrichment.
    """
    rng = np.random.default_rng(seed)
    observed = group_edge_counts_from_edges(rows, cols, weights, codes, K)
    perms = np.zeros((n_perm, K, K), dtype=np.float32)

    valid_idx = np.where(codes >= 0)[0]
    valid_codes = codes[valid_idx].copy()

    for p in range(n_perm):
        shuffled = codes.copy()
        shuffled[valid_idx] = rng.permutation(valid_codes)
        perms[p] = group_edge_counts_from_edges(rows, cols, weights, shuffled, K)

    mean = perms.mean(axis=0)
    std = perms.std(axis=0) + 1e-8
    p_upper = (1.0 + np.sum(perms >= observed[None, :, :], axis=0)) / (n_perm + 1.0)
    return mean, std, p_upper


def compute_paga_like_connectivity(
    A: sp.csr_matrix,
    codes: np.ndarray,
    K: int,
    cfg: PagaLikeConnectivityConfig,
) -> Dict[str, np.ndarray]:
    rows, cols, weights = get_undirected_edge_table(A)
    observed = group_edge_counts_from_edges(rows, cols, weights, codes, K)
    expected_degree = degree_corrected_expected_counts(A, codes, K)

    enrichment = observed / (expected_degree + 1e-8)
    confidence = np.minimum(1.0, enrichment)
    # Put diagonal on a separate interpretation; most PAGA-like plots focus on off-diagonal topology.
    np.fill_diagonal(confidence, 1.0)

    perm_mean, perm_std, p_upper = permutation_null_counts(
        rows, cols, weights, codes, K,
        n_perm=cfg.n_permutations,
        seed=cfg.permutation_random_state,
    )
    z = (observed - perm_mean) / (perm_std + 1e-8)
    q = bh_fdr(p_upper)

    return {
        "observed_counts": observed.astype(np.float32),
        "expected_degree_counts": expected_degree.astype(np.float32),
        "enrichment": enrichment.astype(np.float32),
        "confidence": confidence.astype(np.float32),
        "perm_mean": perm_mean.astype(np.float32),
        "perm_std": perm_std.astype(np.float32),
        "zscore": z.astype(np.float32),
        "pvalue": p_upper.astype(np.float32),
        "qvalue": q.astype(np.float32),
    }


# =============================================================================
# Directed vector-field transition/flux analysis
# =============================================================================

def build_directed_cell_transition_from_velocity(
    Z: np.ndarray,
    V: np.ndarray,
    cfg: PagaLikeConnectivityConfig,
) -> sp.csr_matrix:
    """
    Builds a local directed transition matrix:
        i -> j is high when neighbor displacement Z_j - Z_i aligns with V_i.
    """
    n = Z.shape[0]
    k = min(cfg.directed_k + 1, n)
    nn = NearestNeighbors(n_neighbors=k, metric=cfg.graph_metric)
    nn.fit(Z)
    dist, ind = nn.kneighbors(Z)

    rows, cols, vals = [], [], []

    if cfg.directed_distance_scale is None:
        sigma_global = np.median(dist[:, -1]) + 1e-8
    else:
        sigma_global = cfg.directed_distance_scale

    V_norm = np.linalg.norm(V, axis=1) + 1e-8

    for i in range(n):
        neigh = ind[i]
        d = dist[i]
        if not cfg.directed_include_self:
            mask = neigh != i
            neigh = neigh[mask]
            d = d[mask]
        neigh = neigh[:cfg.directed_k]
        d = d[:cfg.directed_k]
        if len(neigh) == 0:
            continue

        disp = Z[neigh] - Z[i]
        disp_norm = np.linalg.norm(disp, axis=1) + 1e-8
        align = (disp @ V[i]) / (disp_norm * V_norm[i])

        if cfg.directed_clip_negative_alignment:
            align_term = np.maximum(align, 0.0)
        else:
            align_term = align

        direction_weight = np.exp(align_term / max(cfg.directed_temperature, 1e-4))
        distance_weight = np.exp(-(d ** 2) / (2.0 * sigma_global ** 2))
        w = direction_weight * distance_weight

        if w.sum() <= 0:
            w = np.ones_like(w)
        w = w / (w.sum() + 1e-12)

        rows.extend([i] * len(neigh))
        cols.extend(neigh.tolist())
        vals.extend(w.tolist())

    P = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
    return P


def group_directed_flux(P: sp.csr_matrix, codes: np.ndarray, K: int) -> np.ndarray:
    P = P.tocoo()
    F = np.zeros((K, K), dtype=np.float64)
    gr = codes[P.row]
    gc = codes[P.col]
    valid = (gr >= 0) & (gc >= 0)
    for a, b, w in zip(gr[valid], gc[valid], P.data[valid]):
        F[a, b] += w

    # Normalize by number of source cells in each group.
    sizes = np.array([(codes == k).sum() for k in range(K)], dtype=float) + 1e-8
    F = F / sizes[:, None]
    F = row_normalize(F)
    return F.astype(np.float32)


def compute_directionality(F: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    D = (F - F.T) / (F + F.T + eps)
    np.fill_diagonal(D, 0.0)
    return D.astype(np.float32)


# =============================================================================
# Group diagnostics and comparisons
# =============================================================================

def weighted_group_mean(x: Optional[np.ndarray], codes: np.ndarray, K: int) -> np.ndarray:
    if x is None:
        return np.zeros(K, dtype=np.float32)
    out = np.zeros(K, dtype=np.float32)
    for k in range(K):
        mask = codes == k
        if mask.sum() > 0:
            out[k] = float(np.mean(x[mask]))
    return out


def summarize_groups(
    codes: np.ndarray,
    names: List[str],
    F: np.ndarray,
    V: Optional[np.ndarray],
    pt: Optional[np.ndarray],
    divergence: Optional[np.ndarray],
    curl: Optional[np.ndarray],
) -> pd.DataFrame:
    K = len(names)
    sizes = np.array([(codes == k).sum() for k in range(K)], dtype=int)
    speed = np.linalg.norm(V, axis=1) if V is not None else None
    self_flux = np.diag(F)
    outgoing = F.sum(axis=1) - self_flux
    incoming = F.sum(axis=0) - self_flux
    entropy = compute_row_entropy(F)

    df = pd.DataFrame({
        "group": names,
        "n_cells": sizes,
        "fraction": sizes / max(sizes.sum(), 1),
        "mean_pseudotime": weighted_group_mean(pt, codes, K),
        "mean_speed": weighted_group_mean(speed, codes, K),
        "mean_divergence": weighted_group_mean(divergence, codes, K),
        "mean_curl": weighted_group_mean(curl, codes, K),
        "self_flux": self_flux,
        "incoming_flux": incoming,
        "outgoing_flux": outgoing,
        "flux_entropy": entropy,
        "source_score": outgoing - incoming + weighted_group_mean(divergence, codes, K),
        "sink_score": self_flux + incoming - outgoing - weighted_group_mean(divergence, codes, K),
    })
    return df


def compute_row_entropy(T: np.ndarray) -> np.ndarray:
    H = -np.sum(T * np.log(T + 1e-8), axis=1)
    H = H / (np.log(T.shape[1]) + 1e-8)
    return H.astype(np.float32)


def compare_two_group_analyses(
    res_a: Dict[str, object],
    res_b: Dict[str, object],
    threshold: float = 0.15,
) -> Dict[str, object]:
    codes_a = res_a["codes"]
    codes_b = res_b["codes"]
    valid = (codes_a >= 0) & (codes_b >= 0)

    ari = adjusted_rand_score(codes_a[valid], codes_b[valid])
    nmi = normalized_mutual_info_score(codes_a[valid], codes_b[valid])

    Ca = np.asarray(res_a["connectivity"]["confidence"])
    Cb = np.asarray(res_b["connectivity"]["confidence"])
    Fa = np.asarray(res_a["directed_flux"])
    Fb = np.asarray(res_b["directed_flux"])
    Da = np.asarray(res_a["directionality"])
    Db = np.asarray(res_b["directionality"])

    # Matrix sizes can differ if VAMP and quantum choose different K.
    out = {"ari": ari, "nmi": nmi}

    if Ca.shape == Cb.shape:
        iu = np.triu_indices(Ca.shape[0], k=1)
        ea = Ca[iu] >= threshold
        eb = Cb[iu] >= threshold
        union = np.logical_or(ea, eb).sum()
        inter = np.logical_and(ea, eb).sum()
        jaccard = inter / max(union, 1)
        out["edge_jaccard"] = jaccard
        out["confidence_corr"] = np.corrcoef(Ca[iu], Cb[iu])[0, 1] if len(iu[0]) > 1 else np.nan
        out["directed_flux_corr"] = np.corrcoef(Fa.ravel(), Fb.ravel())[0, 1]
        out["directionality_corr"] = np.corrcoef(Da.ravel(), Db.ravel())[0, 1]
    else:
        out["edge_jaccard"] = np.nan
        out["confidence_corr"] = np.nan
        out["directed_flux_corr"] = np.nan
        out["directionality_corr"] = np.nan

    return out


# =============================================================================
# Plotting
# =============================================================================

def plot_heatmap(
    M: np.ndarray,
    labels: List[str],
    title: str,
    path: str,
    cfg: PagaLikeConnectivityConfig,
    cmap: str = "viridis",
    center: Optional[float] = None,
    fmt: str = ".2f",
) -> str:
    fig, ax = plt.subplots(figsize=(0.65 * len(labels) + 3.2, 0.65 * len(labels) + 2.7))
    sns.heatmap(
        M,
        cmap=cmap,
        center=center,
        annot=len(labels) <= 14,
        fmt=fmt,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
    )
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("target group")
    ax.set_ylabel("source group")
    savefig(path, cfg)
    return path


def group_positions_on_embedding(E: np.ndarray, codes: np.ndarray, K: int) -> np.ndarray:
    pos = np.zeros((K, 2), dtype=np.float32)
    for k in range(K):
        mask = codes == k
        if mask.sum() == 0:
            pos[k] = np.array([0, 0])
        else:
            pos[k] = np.median(E[mask], axis=0)
    return pos

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from matplotlib.patches import FancyArrowPatch

def plot_paga_like_graph_on_embedding(
    E: np.ndarray,
    codes: np.ndarray,
    names: List[str],
    confidence: np.ndarray,
    qvalue: np.ndarray,
    directed_flux: np.ndarray,
    cfg: PagaLikeConnectivityConfig, # Assuming this is defined elsewhere
    prefix: str,
) -> str:
    path = os.path.join(cfg.output_dir, f"{prefix}_abstracted_graph_on_embedding.png")
    K = len(names)
    pos = group_positions_on_embedding(E, codes, K) # Assuming this is defined elsewhere

    fig, ax = plt.subplots(figsize=(8.0, 8))
    ax.scatter(E[:, 0], E[:, 1], s=4, alpha=0.18, linewidths=0)

    sizes = np.array([(codes == k).sum() for k in range(K)], dtype=float)
    node_sizes = 280 + 2200 * sizes / max(sizes.max(), 1)

    # Undirected confident edges
    for a in range(K):
        for b in range(a + 1, K):
            if confidence[a, b] >= cfg.graph_edge_threshold and qvalue[a, b] <= cfg.edge_qvalue_threshold:
                lw = 0.5 + 5.0 * confidence[a, b]
                ax.plot([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]],
                        color="black", lw=lw, alpha=0.45, zorder=1)

    # Directed flux arrows for asymmetric transitions
    for a in range(K):
        for b in range(K):
            if a == b:
                continue
            if directed_flux[a, b] >= cfg.graph_directed_edge_threshold:
                
                # Calculate exact node radius in points to stop arrows perfectly at the edge
                radius_a = np.sqrt(node_sizes[a] / np.pi)
                radius_b = np.sqrt(node_sizes[b] / np.pi)
                
                arrow = FancyArrowPatch(
                    pos[a], 
                    pos[b], 
                    connectionstyle="arc3,rad=0.15", 
                    arrowstyle="simple,head_width=5,head_length=6,tail_width=1.5",
                    color="black",          # Changed from "tab:red" to "black"
                    alpha=0.4,
                    lw=0.8 + 2.5 * directed_flux[a, b],
                    shrinkA=radius_a + 2,   # Dynamic shrink based on start node size
                    shrinkB=radius_b + 2,   # Dynamic shrink based on end node size
                    zorder=2
                )
                ax.add_patch(arrow)

    # 1. Changed edgecolor="white" to edgecolor="black"
    ax.scatter(pos[:, 0], pos[:, 1], s=node_sizes, c=np.arange(K), cmap="tab20", 
               edgecolor="black", lw=1.0, zorder=3)
               
    # # 3. Removed bbox (frame) and forced color="black"
    # for k, name in enumerate(names):
    #     ax.text(pos[k, 0], pos[k, 1], str(name), ha="center", va="center", 
    #             fontsize=8, color="black", weight="bold", zorder=4)

    # 3. Dynamic labels placed just above the node radius
    for k, name in enumerate(names):
        # Calculate the exact radius of this specific node in points
        radius_pt = np.sqrt(node_sizes[k] / np.pi)
        
        ax.annotate(
            str(name),
            xy=(pos[k, 0], pos[k, 1]),        # Anchor exactly at the center of the node
            xytext=(0, radius_pt + 3),        # Offset by 0 points horizontally, and (radius + 3) points vertically
            textcoords="offset points",       # Tell matplotlib the offset is in display points, not data units
            ha="center",                      # Center horizontally above the node
            va="bottom",                      # Align the bottom of the text to the offset point
            fontsize=8,
            color="black",
            weight="bold",
            zorder=4
        )

    # ax.set_title("PAGA-like topology + vector-field directionality", fontsize=13, weight="bold")
    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")
    sns.despine(ax=ax)
    umap_axis(ax, linewidth=2)
    savefig(path, cfg) # Assuming this is defined elsewhere
    return path

def umap_axis(ax, pos=(0.02, 0.02), length=0.15, fontsize=9, linewidth=1.2):
    x0, y0 = pos
    ax.axis("off")
    ax.annotate("", xy=(x0 + length, y0), xytext=(x0, y0),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=linewidth, color="black"))
    ax.annotate("", xy=(x0, y0 + length), xytext=(x0, y0),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=linewidth, color="black"))
    ax.text(x0 + length, y0 - 0.02, "UMAP1",
            transform=ax.transAxes, ha="right", va="top", fontsize=fontsize)
    ax.text(x0 - 0.02, y0 + length, "UMAP2",
            transform=ax.transAxes, ha="right", va="top",
            rotation=90, fontsize=fontsize)


# def plot_paga_like_graph_on_embedding(
#     E: np.ndarray,
#     codes: np.ndarray,
#     names: List[str],
#     confidence: np.ndarray,
#     qvalue: np.ndarray,
#     directed_flux: np.ndarray,
#     cfg: PagaLikeConnectivityConfig,
#     prefix: str,
# ) -> str:
#     path = os.path.join(cfg.output_dir, f"{prefix}_abstracted_graph_on_embedding.png")
#     K = len(names)
#     pos = group_positions_on_embedding(E, codes, K)

#     fig, ax = plt.subplots(figsize=(8.0, 6.8))
#     ax.scatter(E[:, 0], E[:, 1], s=4, alpha=0.18, linewidths=0)

#     sizes = np.array([(codes == k).sum() for k in range(K)], dtype=float)
#     node_sizes = 280 + 2200 * sizes / max(sizes.max(), 1)

#     # Undirected confident edges
#     for a in range(K):
#         for b in range(a + 1, K):
#             if confidence[a, b] >= cfg.graph_edge_threshold and qvalue[a, b] <= cfg.edge_qvalue_threshold:
#                 lw = 0.5 + 5.0 * confidence[a, b]
#                 ax.plot([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]],
#                         color="black", lw=lw, alpha=0.45, zorder=1)

#     # Directed flux arrows for asymmetric transitions
#     for a in range(K):
#         for b in range(K):
#             if a == b:
#                 continue
#             if directed_flux[a, b] >= cfg.graph_directed_edge_threshold:
#                 dx, dy = pos[b] - pos[a]
#                 # shrink = 0.15
#                 # ax.arrow(
#                 #     pos[a, 0] + shrink * dx,
#                 #     pos[a, 1] + shrink * dy,
#                 #     (1 - 2 * shrink) * dx,
#                 #     (1 - 2 * shrink) * dy,
#                 #     length_includes_head=True,
#                 #     head_width=0.025 * max(np.ptp(E[:, 0]), np.ptp(E[:, 1])),
#                 #     head_length=0.035 * max(np.ptp(E[:, 0]), np.ptp(E[:, 1])),
#                 #     lw=0.8 + 2.5 * directed_flux[a, b],
#                 #     alpha=0.35,
#                 #     color="tab:red",
#                 #     zorder=2,
#                 # )

#                 from matplotlib.patches import FancyArrowPatch
#                 shrink = 0.15
#                 arrow = FancyArrowPatch(
#                     pos[a], # Start point
#                     pos[b], # End point
#                     connectionstyle="arc3,rad=0.15", # Adds a subtle, elegant curve
#                     arrowstyle="simple,head_width=5,head_length=6,tail_width=1.5",
#                     color="tab:red",
#                     alpha=0.6,
#                     lw=0.8 + 2.5 * directed_flux[a, b],
#                     shrinkA=shrink * 100, # Shrink from start (in points)
#                     shrinkB=shrink * 100, # Shrink from end (in points)
#                     zorder=2
#                 )
#                 ax.add_patch(arrow)

#     ax.scatter(pos[:, 0], pos[:, 1], s=node_sizes, c=np.arange(K), cmap="tab20", edgecolor="white", lw=1.0, zorder=3)
#     for k, name in enumerate(names):
#         ax.text(pos[k, 0], pos[k, 1], str(name), ha="center", va="center", fontsize=8,
#                 bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="black", lw=0.4, alpha=0.85), zorder=4)

#     ax.set_title("PAGA-like topology + vector-field directionality", fontsize=13, weight="bold")
#     ax.set_xlabel("Embedding 1")
#     ax.set_ylabel("Embedding 2")
#     sns.despine(ax=ax)
#     savefig(path, cfg)
#     return path


def plot_edge_significance(
    confidence: np.ndarray,
    zscore_mat: np.ndarray,
    qvalue: np.ndarray,
    names: List[str],
    cfg: PagaLikeConnectivityConfig,
    prefix: str,
) -> str:
    path = os.path.join(cfg.output_dir, f"{prefix}_edge_significance_scatter.png")
    K = len(names)
    rows = []
    for a in range(K):
        for b in range(a + 1, K):
            rows.append({
                "edge": f"{names[a]}--{names[b]}",
                "confidence": confidence[a, b],
                "zscore": zscore_mat[a, b],
                "minus_log10_q": -np.log10(qvalue[a, b] + 1e-12),
                "significant": (confidence[a, b] >= cfg.edge_confidence_threshold) and (qvalue[a, b] <= cfg.edge_qvalue_threshold),
            })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    sns.scatterplot(data=df, x="confidence", y="minus_log10_q", hue="significant", s=50, ax=ax)
    ax.axvline(cfg.edge_confidence_threshold, color="black", lw=0.8, ls="--")
    ax.axhline(-np.log10(cfg.edge_qvalue_threshold), color="black", lw=0.8, ls="--")
    ax.set_title("PAGA-like edge confidence and permutation significance", fontsize=13, weight="bold")
    ax.set_xlabel("degree-corrected confidence min(obs/expected, 1)")
    ax.set_ylabel("-log10(q-value)")
    ax.legend(frameon=False, title="")
    sns.despine(ax=ax)
    savefig(path, cfg)
    return path


def plot_group_diagnostics(
    group_summary: pd.DataFrame,
    cfg: PagaLikeConnectivityConfig,
    prefix: str,
) -> str:
    path = os.path.join(cfg.output_dir, f"{prefix}_group_diagnostics.png")
    metrics = [
        "mean_pseudotime",
        "mean_speed",
        "mean_divergence",
        "mean_curl",
        "incoming_flux",
        "outgoing_flux",
        "source_score",
        "sink_score",
    ]
    use_metrics = [m for m in metrics if m in group_summary.columns]
    df = group_summary[["group"] + use_metrics].melt(id_vars="group", var_name="metric", value_name="value")

    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    sns.barplot(data=df, x="group", y="value", hue="metric", ax=ax)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Group-level vector-field and connectivity diagnostics", fontsize=13, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("score / normalized value")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    sns.despine(ax=ax)
    savefig(path, cfg)
    return path


# def plot_vector_field_on_embedding(
#     E: np.ndarray,
#     Z: np.ndarray,
#     V: Optional[np.ndarray],
#     cfg: PagaLikeConnectivityConfig,
# ) -> Optional[str]:
#     if V is None:
#         return None

#     path = os.path.join(cfg.output_dir, "shared_vector_field_on_embedding.png")
#     Z_future = Z + 0.15 * V
#     nn = NearestNeighbors(n_neighbors=1).fit(Z)
#     _, ind = nn.kneighbors(Z_future)
#     E_future = E[ind[:, 0]]
#     dE = E_future - E

#     rng = np.random.default_rng(cfg.random_state)
#     idx = np.arange(E.shape[0])
#     if E.shape[0] > cfg.max_arrows_plot:
#         idx = rng.choice(idx, size=cfg.max_arrows_plot, replace=False)

#     fig, ax = plt.subplots(figsize=(7.2, 6.2))
#     ax.scatter(E[:, 0], E[:, 1], s=4, alpha=0.20, linewidths=0)
#     ax.quiver(E[idx, 0], E[idx, 1], dE[idx, 0], dE[idx, 1],
#               angles="xy", scale_units="xy", scale=1.0, width=0.0022, alpha=0.58)
#     ax.set_title("Vector field projected to embedding", fontsize=13, weight="bold")
#     ax.set_xlabel("Embedding 1")
#     ax.set_ylabel("Embedding 2")
#     sns.despine(ax=ax)
#     savefig(path, cfg)
#     return path

import scvelo as scv

def plot_vector_field_on_embedding(
    adata,  # Replaced E, Z, V with the AnnData object
    cfg: PagaLikeConnectivityConfig,
) -> Optional[str]:
    
    # 1. Extract necessary arrays from adata
    # Using .get to safely check if the velocity exists
    V = adata.obsm.get("velot_quantum_metaflow_velocity_latent_final")
    if V is None:
        return None

    Z = adata.obsm["X_velot_quantum_metaflow_latent_scaled"]
    
    # Assuming 'umap' basis translates to 'X_umap' in obsm, which is standard
    E = adata.obsm["X_umap"] 

    path = os.path.join(cfg.output_dir, "shared_vector_field_on_embedding.png")
    
    # 2. Calculate the projected future states in the embedding
    Z_future = Z + 0.15 * V
    nn = NearestNeighbors(n_neighbors=1).fit(Z)
    _, ind = nn.kneighbors(Z_future)
    E_future = E[ind[:, 0]]
    dE = E_future - E

    # 3. Subsample arrows to avoid overcrowding
    rng = np.random.default_rng(cfg.random_state)
    idx = np.arange(E.shape[0])
    if E.shape[0] > cfg.max_arrows_plot:
        idx = rng.choice(idx, size=cfg.max_arrows_plot, replace=False)

    fig, ax = plt.subplots(figsize=(8,8))
    scv.pl.scatter(
        adata, 
        basis="umap", 
        color="velot_vampflow_state", 
        ax=ax,               # <--- This anchors scvelo to our matplotlib figure
        show=False,          # <--- Critical: prevents scvelo from closing the figure immediately
        title="",            # We will set a custom title below
        frameon=False, 
        legend_loc="on data",
        palette="tab20"
    )
    umap_axis(ax, linewidth=2)
    fig.savefig("metastates.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8,8))

    # 5. Plot the scvelo scatter directly onto our custom 'ax'
    scv.pl.scatter(
        adata, 
        basis="umap", 
        color="velot_vampflow_state", 
        ax=ax,               # <--- This anchors scvelo to our matplotlib figure
        show=False,          # <--- Critical: prevents scvelo from closing the figure immediately
        title="",            # We will set a custom title below
        frameon=False, 
        legend_loc="on data",
        palette="tab20"
    )

    # 6. Overlay the quiver plot on the same axis
    # Added zorder=3 to ensure arrows render on top of the scvelo scatter points
    ax.quiver(E[idx, 0], E[idx, 1], dE[idx, 0], dE[idx, 1],
              angles="xy", scale_units="xy", scale=10, width=0.0022, alpha=0.4, color="black", zorder=3)
    
    # 7. Final styling
    # ax.set_title("Vector field projected to embedding", fontsize=13, weight="bold")
    
    # Optional: scvelo usually removes axes labels when frameon=False, 
    # but if you want to force them back on, keep these lines:
    umap_axis(ax, linewidth=2)
    
    savefig(path, cfg) # Assuming this is defined elsewhere
    
    # Important: Since show=False was passed to scvelo, you might want to call plt.close(fig) 
    # if you are running this in a loop to prevent memory leaks, depending on your savefig implementation.
    return path


def plot_partition_comparison(
    comparison: Dict[str, object],
    cfg: PagaLikeConnectivityConfig,
) -> str:
    path = os.path.join(cfg.output_dir, "vamp_vs_quantum_connectivity_comparison.png")
    metrics = {
        "ARI": comparison.get("ari", np.nan),
        "NMI": comparison.get("nmi", np.nan),
        "Edge Jaccard": comparison.get("edge_jaccard", np.nan),
        "Confidence corr": comparison.get("confidence_corr", np.nan),
        "Directed flux corr": comparison.get("directed_flux_corr", np.nan),
        "Directionality corr": comparison.get("directionality_corr", np.nan),
    }
    df = pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())})

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    sns.barplot(data=df, x="metric", y="value", ax=ax)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("VAMPFlow vs QuantumMSM PAGA-like topology agreement", fontsize=13, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("agreement / correlation")
    ax.tick_params(axis="x", rotation=30)
    sns.despine(ax=ax)
    savefig(path, cfg)
    return path


# =============================================================================
# Single-key analysis
# =============================================================================

def run_paga_like_analysis_for_group_key(
    adata,
    group_key: str,
    Z: np.ndarray,
    E: np.ndarray,
    V: Optional[np.ndarray],
    V_emb: Optional[np.ndarray],
    A: sp.csr_matrix,
    pt: Optional[np.ndarray],
    cfg: PagaLikeConnectivityConfig,
) -> Dict[str, object]:
    if group_key not in adata.obs:
        raise KeyError(f"Group key {group_key!r} not found in adata.obs.")

    raw_codes, raw_names = encode_groups(adata.obs[group_key].astype(str).to_numpy())
    codes, names, valid_mask = filter_small_groups(raw_codes, raw_names, cfg.min_group_size)
    K = len(names)
    if K < 2:
        raise ValueError(f"After filtering small groups, {group_key!r} has fewer than 2 groups.")

    print(f"[{group_key}] groups={K}; valid cells={valid_mask.sum()}/{len(codes)}")

    conn = compute_paga_like_connectivity(A, codes, K, cfg)

    if V is not None:
        P_dir = build_directed_cell_transition_from_velocity(Z, V, cfg)
        F = group_directed_flux(P_dir, codes, K)
        D = compute_directionality(F)
        divergence = zscore(np.asarray(A.sum(axis=1)).ravel() * 0.0)  # placeholder overwritten below
        divergence = estimate_divergence_knn(Z, V, k=min(30, Z.shape[0]))
        curl = estimate_embedding_curl(E, V_emb, k=min(30, Z.shape[0])) if V_emb is not None else None
    else:
        P_dir = None
        F = np.zeros((K, K), dtype=np.float32)
        D = np.zeros((K, K), dtype=np.float32)
        divergence = None
        curl = None

    group_summary = summarize_groups(
        codes=codes,
        names=names,
        F=F,
        V=V,
        pt=pt,
        divergence=divergence,
        curl=curl,
    )

    prefix = sanitize_key_for_filename(group_key)
    outdir = cfg.output_dir
    ensure_dir(outdir)

    paths: Dict[str, Optional[str]] = {}
    paths["confidence_heatmap"] = plot_heatmap(
        conn["confidence"], names,
        f"{group_key}: PAGA-like connectivity confidence",
        os.path.join(outdir, f"{prefix}_connectivity_confidence_heatmap.png"),
        cfg,
        cmap="viridis",
    )
    paths["enrichment_heatmap"] = plot_heatmap(
        conn["enrichment"], names,
        f"{group_key}: observed / degree-null expected connectivity",
        os.path.join(outdir, f"{prefix}_connectivity_enrichment_heatmap.png"),
        cfg,
        cmap="mako",
    )
    paths["zscore_heatmap"] = plot_heatmap(
        conn["zscore"], names,
        f"{group_key}: permutation-normalized connectivity z-score",
        os.path.join(outdir, f"{prefix}_connectivity_zscore_heatmap.png"),
        cfg,
        cmap="coolwarm",
        center=0,
    )
    paths["directed_flux_heatmap"] = plot_heatmap(
        F, names,
        f"{group_key}: vector-field directed flux",
        os.path.join(outdir, f"{prefix}_directed_flux_heatmap.png"),
        cfg,
        cmap="rocket_r",
    )
    paths["directionality_heatmap"] = plot_heatmap(
        D, names,
        f"{group_key}: antisymmetric directionality",
        os.path.join(outdir, f"{prefix}_directionality_heatmap.png"),
        cfg,
        cmap="coolwarm",
        center=0,
    )
    paths["abstracted_graph"] = plot_paga_like_graph_on_embedding(
        E=E,
        codes=codes,
        names=names,
        confidence=conn["confidence"],
        qvalue=conn["qvalue"],
        directed_flux=F,
        cfg=cfg,
        prefix=prefix,
    )
    paths["edge_significance"] = plot_edge_significance(
        conn["confidence"],
        conn["zscore"],
        conn["qvalue"],
        names,
        cfg,
        prefix,
    )
    paths["group_diagnostics"] = plot_group_diagnostics(group_summary, cfg, prefix)

    # Save tables
    group_summary_csv = os.path.join(outdir, f"{prefix}_group_summary.csv")
    group_summary.to_csv(group_summary_csv, index=False)
    paths["group_summary_csv"] = group_summary_csv

    edge_table = make_edge_table(names, conn, F, D)
    edge_table_csv = os.path.join(outdir, f"{prefix}_edge_table.csv")
    edge_table.to_csv(edge_table_csv, index=False)
    paths["edge_table_csv"] = edge_table_csv

    # Store in AnnData
    adata.uns[f"{group_key}_paga_like_connectivity"] = {
        "names": names,
        "confidence": conn["confidence"],
        "enrichment": conn["enrichment"],
        "observed_counts": conn["observed_counts"],
        "expected_degree_counts": conn["expected_degree_counts"],
        "zscore": conn["zscore"],
        "pvalue": conn["pvalue"],
        "qvalue": conn["qvalue"],
        "directed_flux": F,
        "directionality": D,
        "group_summary": group_summary,
        "edge_table": edge_table,
    }

    return {
        "group_key": group_key,
        "codes": codes,
        "names": names,
        "valid_mask": valid_mask,
        "connectivity": conn,
        "directed_transition": P_dir,
        "directed_flux": F,
        "directionality": D,
        "group_summary": group_summary,
        "edge_table": edge_table,
        "figure_paths": paths,
    }


def estimate_divergence_knn(Z: np.ndarray, V: np.ndarray, k: int = 30) -> np.ndarray:
    n = Z.shape[0]
    k = min(k, n)
    nn = NearestNeighbors(n_neighbors=k).fit(Z)
    _, ind = nn.kneighbors(Z)
    div = np.zeros(n, dtype=np.float32)
    for i in range(n):
        nb = ind[i]
        X = Z[nb] - Z[i]
        Y = V[nb] - V[i]
        try:
            B, *_ = np.linalg.lstsq(X, Y, rcond=1e-3)
            div[i] = np.trace(B)
        except Exception:
            div[i] = 0.0
    return zscore(div)


def estimate_embedding_curl(E: np.ndarray, V_emb: Optional[np.ndarray], k: int = 30) -> Optional[np.ndarray]:
    if V_emb is None:
        return None
    n = E.shape[0]
    k = min(k, n)
    nn = NearestNeighbors(n_neighbors=k).fit(E)
    _, ind = nn.kneighbors(E)
    curl = np.zeros(n, dtype=np.float32)
    for i in range(n):
        nb = ind[i]
        X = E[nb] - E[i]
        Y = V_emb[nb] - V_emb[i]
        try:
            B, *_ = np.linalg.lstsq(X, Y, rcond=1e-3)
            dVx_dy = B[1, 0]
            dVy_dx = B[0, 1]
            curl[i] = dVy_dx - dVx_dy
        except Exception:
            curl[i] = 0.0
    return zscore(np.abs(curl))


def make_edge_table(
    names: List[str],
    conn: Dict[str, np.ndarray],
    F: np.ndarray,
    D: np.ndarray,
) -> pd.DataFrame:
    rows = []
    K = len(names)
    for a in range(K):
        for b in range(K):
            if a == b:
                continue
            rows.append({
                "source": names[a],
                "target": names[b],
                "confidence": conn["confidence"][a, b],
                "enrichment": conn["enrichment"][a, b],
                "observed_count": conn["observed_counts"][a, b],
                "expected_degree_count": conn["expected_degree_counts"][a, b],
                "zscore": conn["zscore"][a, b],
                "pvalue": conn["pvalue"][a, b],
                "qvalue": conn["qvalue"][a, b],
                "directed_flux": F[a, b],
                "directionality": D[a, b],
            })
    return pd.DataFrame(rows).sort_values(
        ["confidence", "directed_flux"], ascending=False
    )


# =============================================================================
# Top-level runners
# =============================================================================

def run_velot_paga_like_connectivity_analysis(
    adata,
    cfg: Optional[PagaLikeConnectivityConfig] = None,
) -> Dict[str, object]:
    if cfg is None:
        cfg = PagaLikeConnectivityConfig()

    ensure_dir(cfg.output_dir)

    Z, latent_source = get_latent(adata, cfg)
    E, embedding_source = get_embedding(adata, Z, cfg)
    V, velocity_source, V_emb = get_velocity(adata, Z, E, cfg)
    pt = get_pseudotime(adata, cfg)

    print(f"[representation] latent={latent_source}, embedding={embedding_source}")
    print(f"[velocity] source={velocity_source}")
    if V is not None:
        print(f"[velocity] median norm={np.median(np.linalg.norm(V, axis=1)):.4f}")

    A, graph_source = build_or_get_knn_graph(adata, Z, cfg)
    print(f"[graph] source={graph_source}, shape={A.shape}, nnz={A.nnz}")

    shared_paths = {}
    print(adata)
    # shared_paths["vector_field_embedding"] = plot_vector_field_on_embedding(E, Z, V, cfg)
    shared_paths["vector_field_embedding"] = plot_vector_field_on_embedding(adata, cfg)

    results: Dict[str, object] = {
        "adata": adata,
        "latent": Z,
        "embedding": E,
        "velocity": V,
        "velocity_embedding": V_emb,
        "pseudotime": pt,
        "graph": A,
        "graph_source": graph_source,
        "latent_source": latent_source,
        "embedding_source": embedding_source,
        "velocity_source": velocity_source,
        "analyses": {},
        "figure_paths": {"shared": shared_paths},
    }

    for group_key in cfg.group_keys:
        if group_key not in adata.obs:
            warnings.warn(f"Skipping group_key={group_key!r}: not found in adata.obs.")
            continue
        res = run_paga_like_analysis_for_group_key(
            adata=adata,
            group_key=group_key,
            Z=Z,
            E=E,
            V=V,
            V_emb=V_emb,
            A=A,
            pt=pt,
            cfg=cfg,
        )
        results["analyses"][group_key] = res
        results["figure_paths"][group_key] = res["figure_paths"]

    # Compare VAMPFlow and QuantumMSM if both are present.
    if "velot_vampflow_state" in results["analyses"] and "velot_quantum_msm_state" in results["analyses"]:
        comp = compare_two_group_analyses(
            results["analyses"]["velot_vampflow_state"],
            results["analyses"]["velot_quantum_msm_state"],
            threshold=cfg.edge_confidence_threshold,
        )
        comp_path = plot_partition_comparison(comp, cfg)
        comp["figure_path"] = comp_path
        results["vamp_vs_quantum_comparison"] = comp

        comp_df = pd.DataFrame([comp])
        comp_csv = os.path.join(cfg.output_dir, "vamp_vs_quantum_connectivity_comparison.csv")
        comp_df.to_csv(comp_csv, index=False)
        results["figure_paths"]["vamp_vs_quantum_comparison"] = comp_path
        results["comparison_csv"] = comp_csv

    print("[done] PAGA-like connectivity analysis completed")
    return results


def run_full_metaflow_plus_connectivity(
    adata,
    metaflow_cfg=None,
    connectivity_cfg: Optional[PagaLikeConnectivityConfig] = None,
) -> Dict[str, object]:
    """
    Optional convenience wrapper.

    If velot_quantum_metaflow.py is importable, run the full VAMP/Quantum meta-state
    inference first, then run this PAGA-like quantitative connectivity analysis.

    Example:
        from velot_quantum_metaflow import VelOTQuantumMetaFlowConfig

        mf_cfg = VelOTQuantumMetaFlowConfig(estimator_mode="both")
        conn_cfg = PagaLikeConnectivityConfig()
        out = run_full_metaflow_plus_connectivity(adata, mf_cfg, conn_cfg)
    """
    try:
        from velot_quantum_metaflow import run_velot_quantum_metaflow
    except Exception as e:
        raise ImportError(
            "Could not import velot_quantum_metaflow.py. Put it in the same folder "
            "or run run_velot_paga_like_connectivity_analysis() after generating "
            "VAMP/Quantum meta-state outputs."
        ) from e

    mf_out = run_velot_quantum_metaflow(adata, metaflow_cfg)
    conn_out = run_velot_paga_like_connectivity_analysis(mf_out["adata"], connectivity_cfg)
    return {"metaflow": mf_out, "connectivity": conn_out, "adata": mf_out["adata"]}


# =============================================================================
# Example script entry point
# =============================================================================

if __name__ == "__main__":
    # Example:
    #
    # import scanpy as sc
    #
    # adata = sc.read_h5ad("adata_with_velot_quantum_metaflow.h5ad")
    #
    # cfg = PagaLikeConnectivityConfig(
    #     group_keys=["velot_vampflow_state", "velot_quantum_msm_state"],
    #     latent_key="X_velot_quantum_metaflow_latent_scaled",
    #     embedding_key="X_umap",
    #     velocity_key="velot_quantum_metaflow_velocity_latent_final",
    #     pseudotime_key="velot_quantum_metaflow_pseudotime",
    #     n_neighbors=30,
    #     directed_k=30,
    #     n_permutations=200,
    #     output_dir="velot_paga_like_connectivity_outputs",
    # )
    #
    # out = run_velot_paga_like_connectivity_analysis(adata, cfg)
    # out["adata"].write_h5ad("adata_with_paga_like_connectivity.h5ad")
    pass

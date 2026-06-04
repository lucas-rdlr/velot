
"""
General VelOT-MetaFlow meta-state summary module.

Purpose
-------
Create publication-style summary figures for any dataset after running a meta-state
method such as:

    run_velot_metaflow_with_velot_velocity(...)
    run_velot_quantum_metaflow(...)

The module is intentionally general:
    - detects meta-state columns;
    - detects or builds a transition matrix;
    - summarizes directionality: initial/source, terminal/sink, branching/saddle,
      cycling/recurrent, intermediate/transient;
    - overlays meta-states and transition directions on UMAP;
    - selects relevant genes automatically from expression according to detected
      meta-states and pseudotime;
    - creates path heatmaps similar in spirit to Paul-style lineage summaries.

Main outputs
------------
1. summary_umap_graph_diagnostics.png
   UMAP with meta-states + arrows + labels, coarse graph, diagnostic matrix.

2. state_gene_heatmap.png
   Mean expression of selected genes per meta-state ordered by inferred progression.

3. pseudotime_path_gene_heatmaps.png
   Relevant genes along inferred source-to-terminal paths.

4. CSV files:
   state_summary.csv
   state_top_genes.csv
   inferred_paths.csv
   selected_heatmap_genes.csv

Example
-------
from metaflow_general_summary import MetaFlowSummaryConfig, make_metaflow_summary

cfg = MetaFlowSummaryConfig(
    meta_state_key="velot_metaflow_state",
    transition_key="velot_metaflow_transition_matrix",
    pseudotime_key="velot_metaflow_pseudotime",
    cell_type_key="cell_type",
    embedding_key="X_umap",
    output_dir="pancreas_metaflow_summary",
    title="Pancreas VelOT-MetaFlow",
)

res = make_metaflow_summary(adata, cfg)
"""

from __future__ import annotations

import os
import re
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch

import seaborn as sns
import networkx as nx
from sklearn.neighbors import NearestNeighbors


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MetaFlowSummaryConfig:
    # Core keys
    embedding_key: str = "X_umap"
    meta_state_key: Optional[str] = None
    state_label_key: Optional[str] = None
    transition_key: Optional[str] = None
    pseudotime_key: Optional[str] = None
    cell_type_key: Optional[str] = "cell_type"

    # Direction/velocity visualization keys
    velocity_embedding_key: Optional[str] = "velot_velocity_umap"
    latent_key_for_velocity_projection: Optional[str] = "X_velot_metaflow_latent_scaled"
    future_latent_key: Optional[str] = "velot_metaflow_future_latent"

    # Expression source
    expression_layer: Optional[str] = None
    use_raw: bool = False

    # Optional marker genes. If None, genes are selected automatically.
    # Example:
    # marker_genes={"ductal": ["KRT19", "SOX9"], "acinar": ["PRSS1", "CPA1"]}
    marker_genes: Optional[Dict[str, List[str]]] = None

    # Gene selection
    n_candidate_genes: int = 3000
    max_cells_gene_selection: int = 12000
    top_genes_per_state: int = 5
    top_pseudotime_genes_per_path: int = 8
    max_heatmap_genes: int = 45
    min_pct_expressed: float = 0.03
    exclude_gene_prefixes: Tuple[str, ...] = ("MT-", "RPS", "RPL")

    # Path inference
    root_state: Optional[Union[str, int]] = None
    terminal_states: Optional[Sequence[Union[str, int]]] = None
    n_terminal_paths: int = 3
    edge_threshold: float = 0.06
    top_edges_per_state: int = 2
    max_states_on_path: int = 10

    # Heatmaps
    n_bins_per_path: int = 45
    min_cells_per_bin: int = 4
    zscore_heatmap: bool = True
    heatmap_clip: float = 2.5

    # Display
    title: str = "VelOT-MetaFlow meta-state summary"
    label_states_with_direction: bool = True
    label_states_with_celltype: bool = True
    max_celltype_label_chars: int = 13
    max_arrows: int = 900
    scatter_size: float = 7.0
    arrow_alpha: float = 0.42

    # Output
    output_dir: str = "metaflow_general_summary"
    dpi: int = 350
    random_state: int = 0


# =============================================================================
# Detection and small utilities
# =============================================================================

def _ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _natural_sort_key(x):
    x = str(x)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", x)]


def _state_to_int(x) -> int:
    if isinstance(x, (int, np.integer)):
        return int(x)
    s = str(x)
    m = re.search(r"(\d+)", s)
    if m:
        return int(m.group(1))
    raise ValueError(f"Could not parse state id from {x!r}.")


def _state_name(x: Union[int, str]) -> str:
    if isinstance(x, str) and x.startswith("M"):
        return x
    return f"M{_state_to_int(x)}"


def _row_normalize(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return M / (M.sum(axis=1, keepdims=True) + eps)


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=float))
    if np.std(x) > 0:
        return ((x - np.mean(x)) / (np.std(x) + 1e-8)).astype(np.float32)
    return np.zeros_like(x, dtype=np.float32)


def _detect_meta_state_key(adata, key: Optional[str]) -> str:
    if key is not None:
        if key not in adata.obs:
            raise KeyError(f"meta_state_key={key!r} not found in adata.obs.")
        return key

    candidates = [
        "velot_metaflow_state",
        "velot_quantum_msm_state",
        "velot_vampflow_state",
        "metaflow_state",
        "quantum_msm_state",
        "vampflow_state",
    ]
    for c in candidates:
        if c in adata.obs:
            return c

    possible = [c for c in adata.obs.columns if "state" in c.lower() or "meta" in c.lower()]
    raise KeyError(f"Could not auto-detect meta-state key. State-like obs columns: {possible}")


def _detect_state_label_key(adata, meta_state_key: str, key: Optional[str]) -> Optional[str]:
    if key is not None:
        return key if key in adata.obs else None

    candidates = [
        meta_state_key.replace("_state", "_state_label"),
        "velot_metaflow_state_label",
        "velot_quantum_msm_state_label",
        "velot_vampflow_state_label",
        "metaflow_state_label",
    ]
    for c in candidates:
        if c in adata.obs:
            return c
    return None


def _detect_transition_key(adata, key: Optional[str], meta_state_key: str) -> Optional[str]:
    if key is not None:
        if key not in adata.uns:
            raise KeyError(f"transition_key={key!r} not found in adata.uns.")
        return key

    candidates = [
        meta_state_key.replace("_state", "_transition_matrix"),
        "velot_metaflow_transition_matrix",
        "velot_quantum_msm_transition_matrix",
        "velot_vampflow_transition_matrix",
        "metaflow_transition_matrix",
    ]
    for c in candidates:
        if c in adata.uns:
            return c

    return None


def _detect_pseudotime_key(adata, key: Optional[str]) -> Optional[str]:
    if key is not None:
        if key not in adata.obs:
            warnings.warn(f"pseudotime_key={key!r} not found; continuing without pseudotime.")
            return None
        return key

    candidates = [
        "velot_metaflow_pseudotime",
        "velot_quantum_metaflow_pseudotime",
        "metaflow_pseudotime",
        "velot_pseudotime",
        "dpt_pseudotime",
        "pseudotime",
        "palantir_pseudotime",
    ]
    for c in candidates:
        if c in adata.obs:
            return c
    return None


def _get_embedding(adata, key: str) -> np.ndarray:
    if key not in adata.obsm:
        raise KeyError(f"embedding_key={key!r} not found in adata.obsm.")
    E = np.asarray(adata.obsm[key])
    if E.ndim != 2 or E.shape[1] < 2:
        raise ValueError(f"adata.obsm[{key!r}] must be n_cells x >=2.")
    return E[:, :2].astype(float)


def _get_state_vector(adata, meta_state_key: str) -> Tuple[pd.Series, np.ndarray, List[str]]:
    raw = adata.obs[meta_state_key].astype(str)
    state_int = np.array([_state_to_int(x) for x in raw], dtype=int)

    # Remap if states are not contiguous 0..K-1.
    unique = sorted(np.unique(state_int))
    remap = {old: new for new, old in enumerate(unique)}
    if unique != list(range(len(unique))):
        state_int = np.array([remap[x] for x in state_int], dtype=int)

    state_names = [f"M{k}" for k in range(len(np.unique(state_int)))]
    return raw, state_int, state_names


def _get_pseudotime(adata, key: Optional[str]) -> Optional[np.ndarray]:
    if key is None:
        return None
    pt = pd.to_numeric(adata.obs[key], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(pt)
    if finite.sum() < 10:
        return None
    pt[~finite] = np.nanmedian(pt[finite])
    pt = (pt - pt.min()) / (pt.max() - pt.min() + 1e-8)
    return pt.astype(float)


# =============================================================================
# Transition and state summary
# =============================================================================

def _transition_from_uns(adata, key: Optional[str], n_states: int) -> Optional[np.ndarray]:
    if key is None:
        return None
    T = np.asarray(adata.uns[key], dtype=float)
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        warnings.warn(f"adata.uns[{key!r}] is not a square matrix; ignoring it.")
        return None
    if T.shape[0] != n_states:
        warnings.warn(
            f"Transition matrix has shape {T.shape}, but detected {n_states} meta-states. "
            "Ignoring transition matrix and building pseudotime-based fallback."
        )
        return None
    T = np.nan_to_num(T)
    T[T < 0] = 0
    return _row_normalize(T)


def _fallback_transition_from_pseudotime_and_embedding(
    E: np.ndarray,
    state_int: np.ndarray,
    pt: Optional[np.ndarray],
    n_states: int,
) -> np.ndarray:
    """
    If no learned transition matrix is present, build a simple directed transition
    matrix from state centroids and median pseudotime.
    """
    C = np.zeros((n_states, 2), dtype=float)
    med_pt = np.zeros(n_states, dtype=float)

    for k in range(n_states):
        mask = state_int == k
        if mask.sum() == 0:
            C[k] = np.nan
            med_pt[k] = np.nan
        else:
            C[k] = np.median(E[mask], axis=0)
            med_pt[k] = np.nanmedian(pt[mask]) if pt is not None else k / max(n_states - 1, 1)

    med_pt = np.nan_to_num(med_pt, nan=np.nanmedian(med_pt))
    T = np.zeros((n_states, n_states), dtype=float)

    D = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
    sigma = np.nanmedian(D[D > 0]) if np.any(D > 0) else 1.0

    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                continue
            direction = med_pt[j] - med_pt[i]
            if direction < -0.05:
                continue
            T[i, j] = np.exp(-(D[i, j] ** 2) / (2 * sigma ** 2 + 1e-8)) * (0.2 + max(direction, 0))

        if T[i].sum() == 0:
            later = np.where(med_pt >= med_pt[i])[0]
            later = later[later != i]
            if len(later) > 0:
                j = later[np.argmin(D[i, later])]
                T[i, j] = 1.0
            else:
                T[i, i] = 1.0

    # Add some self-retention
    for i in range(n_states):
        T[i, i] = max(T[i, i], 0.15)

    return _row_normalize(T)


def _state_centroids(E: np.ndarray, state_int: np.ndarray, n_states: int) -> np.ndarray:
    C = np.zeros((n_states, 2), dtype=float)
    for k in range(n_states):
        mask = state_int == k
        C[k] = np.median(E[mask], axis=0) if mask.sum() else np.nan
    return C


def _dominant_celltypes(
    adata,
    state_int: np.ndarray,
    n_states: int,
    cell_type_key: Optional[str],
    max_chars: int = 13,
) -> Dict[int, str]:
    if cell_type_key is None or cell_type_key not in adata.obs:
        return {k: "" for k in range(n_states)}

    ct = adata.obs[cell_type_key].astype(str).to_numpy()
    out = {}
    for k in range(n_states):
        vals = ct[state_int == k]
        if len(vals) == 0:
            out[k] = ""
        else:
            lab = pd.Series(vals).value_counts().index[0]
            if len(lab) > max_chars:
                lab = lab[: max_chars - 1] + "…"
            out[k] = lab
    return out


def _score_and_classify_states(
    T: np.ndarray,
    state_int: np.ndarray,
    pt: Optional[np.ndarray],
    state_labels_from_obs: Optional[pd.Series] = None,
) -> pd.DataFrame:
    K = T.shape[0]

    self_t = np.diag(T)
    outgoing = T.sum(axis=1) - self_t
    incoming = T.sum(axis=0) - self_t
    entropy = -np.sum(T * np.log(T + 1e-12), axis=1) / (np.log(K) + 1e-12)

    if pt is None:
        med_pt = np.linspace(0, 1, K)
    else:
        med_pt = np.array([
            np.nanmedian(pt[state_int == k]) if np.any(state_int == k) else np.nan
            for k in range(K)
        ])
        med_pt = np.nan_to_num(med_pt, nan=np.nanmedian(med_pt))
        med_pt = (med_pt - med_pt.min()) / (med_pt.max() - med_pt.min() + 1e-8)

    source_score = outgoing - incoming + (1.0 - med_pt)
    sink_score = self_t + incoming - outgoing + med_pt
    branch_score = entropy + outgoing - self_t
    recurrent_score = self_t

    labels = []
    obs_label_by_state = None
    if state_labels_from_obs is not None:
        obs_label_by_state = {}
        for k in range(K):
            vals = state_labels_from_obs[state_int == k].astype(str)
            if len(vals) > 0:
                obs_label_by_state[k] = vals.value_counts().index[0]

    for k in range(K):
        if obs_label_by_state is not None:
            labels.append(obs_label_by_state.get(k, "intermediate/transient"))
            continue

        scores = {
            "initial/source": source_score[k],
            "terminal/sink": sink_score[k],
            "branching/saddle": branch_score[k],
            "cycling/recurrent": recurrent_score[k],
        }
        best = max(scores, key=scores.get)

        # Make labels conservative: only top quantiles get a special label.
        if best == "initial/source" and source_score[k] < np.quantile(source_score, 0.70):
            best = "intermediate/transient"
        if best == "terminal/sink" and sink_score[k] < np.quantile(sink_score, 0.70):
            best = "intermediate/transient"
        if best == "branching/saddle" and branch_score[k] < np.quantile(branch_score, 0.75):
            best = "intermediate/transient"
        if best == "cycling/recurrent" and recurrent_score[k] < np.quantile(recurrent_score, 0.75):
            best = "intermediate/transient"

        labels.append(best)

    occupancy = np.array([(state_int == k).mean() for k in range(K)])
    n_cells = np.array([(state_int == k).sum() for k in range(K)])

    return pd.DataFrame({
        "meta_state": [f"M{k}" for k in range(K)],
        "n_cells": n_cells,
        "occupancy": occupancy,
        "label": labels,
        "median_pseudotime": med_pt,
        "self_transition": self_t,
        "incoming": incoming,
        "outgoing": outgoing,
        "transition_entropy": entropy,
        "source_score": source_score,
        "sink_score": sink_score,
        "branch_score": branch_score,
        "recurrent_score": recurrent_score,
    })


# =============================================================================
# Path inference
# =============================================================================

def _build_transition_graph(T: np.ndarray, edge_threshold: float, top_edges_per_state: int) -> nx.DiGraph:
    K = T.shape[0]
    G = nx.DiGraph()
    for i in range(K):
        G.add_node(i)

    for i in range(K):
        row = T[i].copy()
        row[i] = 0.0

        candidates = set(np.where(row >= edge_threshold)[0].tolist())
        if top_edges_per_state > 0:
            candidates.update(np.argsort(row)[::-1][:top_edges_per_state].tolist())

        for j in candidates:
            if i == j or row[j] <= 0:
                continue
            G.add_edge(i, int(j), prob=float(row[j]), weight=float(-np.log(row[j] + 1e-12)))

    return G


def _infer_root_and_terminals(
    state_summary: pd.DataFrame,
    cfg: MetaFlowSummaryConfig,
) -> Tuple[int, List[int]]:
    if cfg.root_state is not None:
        root = _state_to_int(cfg.root_state)
    else:
        source_like = state_summary[state_summary["label"].astype(str).str.contains("initial|source", case=False, regex=True)]
        if len(source_like) > 0:
            root = _state_to_int(source_like.sort_values("source_score", ascending=False).iloc[0]["meta_state"])
        else:
            root = int(state_summary["source_score"].idxmax())

    if cfg.terminal_states is not None:
        terminals = [_state_to_int(x) for x in cfg.terminal_states]
    else:
        terminal_like = state_summary[state_summary["label"].astype(str).str.contains("terminal|sink", case=False, regex=True)]
        if len(terminal_like) > 0:
            candidates = [_state_to_int(x) for x in terminal_like.sort_values("sink_score", ascending=False)["meta_state"]]
        else:
            candidates = state_summary.sort_values("sink_score", ascending=False).index.tolist()

        terminals = [int(x) for x in candidates if int(x) != root][: cfg.n_terminal_paths]

    return int(root), terminals


def _infer_paths(
    T: np.ndarray,
    root: int,
    terminals: Sequence[int],
    cfg: MetaFlowSummaryConfig,
) -> Dict[int, List[int]]:
    G = _build_transition_graph(T, cfg.edge_threshold, cfg.top_edges_per_state)
    paths = {}

    for term in terminals:
        term = int(term)
        if term == root:
            continue

        try:
            p = nx.shortest_path(G, source=root, target=term, weight="weight")
        except Exception:
            # Greedy fallback.
            p = [root]
            current = root
            seen = {root}
            for _ in range(cfg.max_states_on_path - 1):
                row = T[current].copy()
                for s in seen:
                    row[s] = 0
                nxt = int(np.argmax(row))
                if row[nxt] <= 0:
                    break
                p.append(nxt)
                seen.add(nxt)
                current = nxt
                if current == term:
                    break
            if p[-1] != term:
                p.append(term)

        # Remove repeated states.
        clean = []
        for x in p:
            if int(x) not in clean:
                clean.append(int(x))
        paths[term] = clean[: cfg.max_states_on_path]

    return paths


# =============================================================================
# Expression utilities and automatic gene selection
# =============================================================================

def _get_var_names(adata, use_raw: bool = False) -> List[str]:
    if use_raw and adata.raw is not None:
        return list(map(str, adata.raw.var_names))
    return list(map(str, adata.var_names))


def _var_lookup(adata, use_raw: bool = False) -> Dict[str, str]:
    return {g.upper(): g for g in _get_var_names(adata, use_raw=use_raw)}


def _filter_gene_names(genes: List[str], exclude_prefixes: Tuple[str, ...]) -> List[str]:
    out = []
    for g in genes:
        gu = g.upper()
        if any(gu.startswith(p.upper()) for p in exclude_prefixes):
            continue
        out.append(g)
    return out


def _matrix_subset(adata, rows: Optional[np.ndarray], genes: List[str], layer: Optional[str], use_raw: bool):
    if use_raw and adata.raw is not None:
        X = adata.raw[:, genes].X
        if rows is not None:
            X = X[rows]
        return X

    idx = [adata.var_names.get_loc(g) for g in genes]
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer {layer!r} not found in adata.layers.")
        X = adata.layers[layer]
    else:
        X = adata.X

    if rows is not None:
        return X[rows][:, idx]
    return X[:, idx]


def _to_dense(X) -> np.ndarray:
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)


def _select_candidate_genes(adata, cfg: MetaFlowSummaryConfig) -> List[str]:
    var_names = _get_var_names(adata, use_raw=cfg.use_raw)
    var_names = _filter_gene_names(var_names, cfg.exclude_gene_prefixes)

    # If marker genes were provided, candidate set starts with them.
    marker_present = []
    if cfg.marker_genes is not None:
        lookup = _var_lookup(adata, use_raw=cfg.use_raw)
        for group, genes in cfg.marker_genes.items():
            for g in genes:
                real = lookup.get(str(g).upper())
                if real is not None and real not in marker_present:
                    marker_present.append(real)

    # Prefer highly variable genes if available.
    hvg = []
    if (not cfg.use_raw) and ("highly_variable" in adata.var.columns):
        hvg = adata.var_names[np.asarray(adata.var["highly_variable"].fillna(False), dtype=bool)].astype(str).tolist()
        hvg = _filter_gene_names(hvg, cfg.exclude_gene_prefixes)

    if len(hvg) >= min(100, cfg.n_candidate_genes // 10):
        candidates = marker_present + [g for g in hvg if g not in marker_present]
        return candidates[: cfg.n_candidate_genes]

    # Otherwise compute variance on a subsample.
    rng = np.random.default_rng(cfg.random_state)
    n = adata.n_obs
    rows = np.arange(n)
    if n > min(cfg.max_cells_gene_selection, 5000):
        rows = rng.choice(rows, size=min(cfg.max_cells_gene_selection, 5000), replace=False)

    # Compute variances over all genes without creating a huge all-cell matrix.
    if cfg.use_raw and adata.raw is not None:
        Xs = adata.raw.X[rows]
        names_all = list(map(str, adata.raw.var_names))
    else:
        Xall = adata.layers[cfg.expression_layer] if cfg.expression_layer is not None else adata.X
        Xs = Xall[rows]
        names_all = list(map(str, adata.var_names))

    if sp.issparse(Xs):
        mean = np.asarray(Xs.mean(axis=0)).ravel()
        mean2 = np.asarray(Xs.multiply(Xs).mean(axis=0)).ravel()
        var = mean2 - mean ** 2
    else:
        Xsd = np.asarray(Xs, dtype=float)
        var = Xsd.var(axis=0)

    names = np.array(names_all, dtype=object)
    keep_mask = np.ones(len(names), dtype=bool)
    for pref in cfg.exclude_gene_prefixes:
        keep_mask &= ~np.char.startswith(np.char.upper(names.astype(str)), pref.upper())

    order = np.argsort(var)[::-1]
    ordered = [str(names[i]) for i in order if keep_mask[i]]
    candidates = marker_present + [g for g in ordered if g not in marker_present]
    return candidates[: cfg.n_candidate_genes]


def _balanced_sample_cells(state_int: np.ndarray, max_cells: int, seed: int) -> np.ndarray:
    n = len(state_int)
    if n <= max_cells:
        return np.arange(n)

    rng = np.random.default_rng(seed)
    states = sorted(np.unique(state_int))
    per_state = max(20, max_cells // max(len(states), 1))
    selected = []

    for s in states:
        idx = np.where(state_int == s)[0]
        if len(idx) <= per_state:
            selected.extend(idx.tolist())
        else:
            selected.extend(rng.choice(idx, size=per_state, replace=False).tolist())

    selected = np.array(selected, dtype=int)
    if len(selected) > max_cells:
        selected = rng.choice(selected, size=max_cells, replace=False)
    return np.sort(selected)


def _compute_state_gene_scores(
    adata,
    genes: List[str],
    state_int: np.ndarray,
    cfg: MetaFlowSummaryConfig,
) -> pd.DataFrame:
    rows = _balanced_sample_cells(state_int, cfg.max_cells_gene_selection, cfg.random_state)
    X = _to_dense(_matrix_subset(adata, rows, genes, cfg.expression_layer, cfg.use_raw)).astype(np.float32)
    s = state_int[rows]
    K = int(s.max()) + 1

    # Gene-wide scale for standardized difference.
    gene_sd = X.std(axis=0) + 1e-8

    out = []
    for k in range(K):
        in_mask = s == k
        out_mask = ~in_mask
        if in_mask.sum() < 3 or out_mask.sum() < 3:
            continue

        mean_in = X[in_mask].mean(axis=0)
        mean_out = X[out_mask].mean(axis=0)
        pct_in = (X[in_mask] > 0).mean(axis=0)
        pct_out = (X[out_mask] > 0).mean(axis=0)

        score = (mean_in - mean_out) / gene_sd
        log2fc_like = np.log2((mean_in + 0.05) / (mean_out + 0.05))

        for j, g in enumerate(genes):
            if pct_in[j] < cfg.min_pct_expressed:
                continue
            out.append({
                "meta_state": f"M{k}",
                "gene": g,
                "score": float(score[j]),
                "mean_in": float(mean_in[j]),
                "mean_out": float(mean_out[j]),
                "pct_in": float(pct_in[j]),
                "pct_out": float(pct_out[j]),
                "log2fc_like": float(log2fc_like[j]),
            })

    df = pd.DataFrame(out)
    if len(df) == 0:
        return df
    df = df.sort_values(["meta_state", "score", "log2fc_like"], ascending=[True, False, False])
    return df


def _compute_path_pseudotime_gene_scores(
    adata,
    genes: List[str],
    paths: Dict[int, List[int]],
    state_int: np.ndarray,
    pt: Optional[np.ndarray],
    cfg: MetaFlowSummaryConfig,
) -> pd.DataFrame:
    if pt is None or len(paths) == 0:
        return pd.DataFrame(columns=["path_terminal", "gene", "corr", "abs_corr"])

    out = []
    rng = np.random.default_rng(cfg.random_state)

    for term, pstates in paths.items():
        idx = np.where(np.isin(state_int, pstates))[0]
        if len(idx) < 20:
            continue
        if len(idx) > cfg.max_cells_gene_selection:
            idx = rng.choice(idx, size=cfg.max_cells_gene_selection, replace=False)

        X = _to_dense(_matrix_subset(adata, idx, genes, cfg.expression_layer, cfg.use_raw)).astype(np.float32)
        t = pt[idx].astype(float)

        t = _zscore(t)
        X = X - X.mean(axis=0, keepdims=True)
        X = X / (X.std(axis=0, keepdims=True) + 1e-8)
        corr = (X.T @ t) / max(len(t) - 1, 1)

        for j, g in enumerate(genes):
            out.append({
                "path_terminal": f"M{term}",
                "gene": g,
                "corr": float(corr[j]),
                "abs_corr": float(abs(corr[j])),
            })

    df = pd.DataFrame(out)
    if len(df):
        df = df.sort_values(["path_terminal", "abs_corr"], ascending=[True, False])
    return df


def _select_heatmap_genes(
    adata,
    state_gene_scores: pd.DataFrame,
    path_gene_scores: pd.DataFrame,
    cfg: MetaFlowSummaryConfig,
) -> Tuple[List[str], List[str], pd.DataFrame]:
    genes = []
    groups = []

    # 1) User marker genes first, if provided.
    if cfg.marker_genes is not None:
        lookup = _var_lookup(adata, use_raw=cfg.use_raw)
        for group, gs in cfg.marker_genes.items():
            for g in gs:
                real = lookup.get(str(g).upper())
                if real is not None and real not in genes:
                    genes.append(real)
                    groups.append(group)

    # 2) Top genes per meta-state.
    if len(state_gene_scores) > 0:
        for state, sub in state_gene_scores.groupby("meta_state", sort=False):
            top = sub.sort_values(["score", "log2fc_like"], ascending=False).head(cfg.top_genes_per_state)
            for g in top["gene"]:
                if g not in genes:
                    genes.append(g)
                    groups.append(f"{state} enriched")

    # 3) Top pseudotime/path-correlated genes.
    if len(path_gene_scores) > 0:
        for term, sub in path_gene_scores.groupby("path_terminal", sort=False):
            top = sub.sort_values("abs_corr", ascending=False).head(cfg.top_pseudotime_genes_per_path)
            for g in top["gene"]:
                if g not in genes:
                    genes.append(g)
                    groups.append(f"{term} pseudotime")

    genes = genes[: cfg.max_heatmap_genes]
    groups = groups[: cfg.max_heatmap_genes]

    table = pd.DataFrame({"gene": genes, "group": groups})
    return genes, groups, table


# =============================================================================
# Heatmap construction
# =============================================================================

def _zscore_rows(M: np.ndarray, clip: float) -> np.ndarray:
    Z = M.astype(float).copy()
    mu = np.nanmean(Z, axis=1, keepdims=True)
    sd = np.nanstd(Z, axis=1, keepdims=True)
    Z = (Z - mu) / (sd + 1e-8)
    return np.clip(np.nan_to_num(Z), -clip, clip)


def _state_gene_heatmap_matrix(
    adata,
    genes: List[str],
    state_int: np.ndarray,
    state_summary: pd.DataFrame,
    cfg: MetaFlowSummaryConfig,
) -> Tuple[np.ndarray, List[int]]:
    order_states = [
        _state_to_int(x)
        for x in state_summary.sort_values("median_pseudotime")["meta_state"].tolist()
    ]

    X = _matrix_subset(adata, None, genes, cfg.expression_layer, cfg.use_raw)
    H = []
    for k in order_states:
        idx = np.where(state_int == k)[0]
        if len(idx) == 0:
            H.append(np.zeros(len(genes)))
        else:
            Xi = X[idx]
            if sp.issparse(Xi):
                mean = np.asarray(Xi.mean(axis=0)).ravel()
            else:
                mean = np.asarray(Xi).mean(axis=0)
            H.append(mean)
    H = np.vstack(H).T
    if cfg.zscore_heatmap:
        H = _zscore_rows(H, cfg.heatmap_clip)
    return H, order_states


def _path_heatmap_matrix(
    adata,
    genes: List[str],
    state_int: np.ndarray,
    pt: Optional[np.ndarray],
    path_states: List[int],
    cfg: MetaFlowSummaryConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    mask = np.isin(state_int, path_states)
    idx = np.where(mask)[0]
    n_bins = cfg.n_bins_per_path

    if len(idx) == 0:
        H = np.zeros((len(genes), n_bins))
        return H, np.zeros(n_bins, dtype=int), np.linspace(0, 1, n_bins), pd.DataFrame()

    # Ordering coordinate combines path-state order and pseudotime inside states.
    order_map = {s: i for i, s in enumerate(path_states)}
    coord = np.zeros(len(idx), dtype=float)

    for s in path_states:
        local = np.where(state_int[idx] == s)[0]
        if len(local) == 0:
            continue
        if pt is not None:
            vals = pt[idx[local]]
            ranks = pd.Series(vals).rank(method="average").to_numpy()
        else:
            ranks = np.arange(len(local), dtype=float)
        ranks = (ranks - ranks.min()) / (ranks.max() - ranks.min() + 1e-8)
        coord[local] = order_map[s] + ranks

    coord = (coord - coord.min()) / (coord.max() - coord.min() + 1e-8)
    edges = np.linspace(0, 1, n_bins + 1)
    bin_id = np.clip(np.digitize(coord, edges) - 1, 0, n_bins - 1)

    X = _to_dense(_matrix_subset(adata, idx, genes, cfg.expression_layer, cfg.use_raw)).astype(np.float32)

    H = np.zeros((len(genes), n_bins), dtype=float)
    bin_state = np.zeros(n_bins, dtype=int)
    bin_distance = np.linspace(0, 1, n_bins)
    rows = []

    for b in range(n_bins):
        local = np.where(bin_id == b)[0]
        if len(local) < cfg.min_cells_per_bin:
            center = 0.5 * (edges[b] + edges[b + 1])
            local = np.argsort(np.abs(coord - center))[: min(len(idx), cfg.min_cells_per_bin)]

        H[:, b] = X[local].mean(axis=0)

        if len(local) > 0:
            states_b = state_int[idx[local]]
            bin_state[b] = int(pd.Series(states_b).value_counts().index[0])
        else:
            bin_state[b] = path_states[min(len(path_states)-1, int(b / max(n_bins,1) * len(path_states)))]

        rows.append({
            "bin": b,
            "n_cells": int(len(local)),
            "dominant_state": f"M{bin_state[b]}",
            "distance": float(bin_distance[b]),
        })

    if cfg.zscore_heatmap:
        H = _zscore_rows(H, cfg.heatmap_clip)

    return H, bin_state, bin_distance, pd.DataFrame(rows)


# =============================================================================
# Plot helpers
# =============================================================================

def _savefig(path: Path, cfg: MetaFlowSummaryConfig) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close()


def _palette(K: int):
    return sns.color_palette("tab20", n_colors=max(K, 3))


def _short_direction(label: str) -> str:
    s = str(label).lower()
    if "initial" in s or "source" in s:
        return "source"
    if "terminal" in s or "sink" in s:
        return "sink"
    if "branch" in s or "saddle" in s:
        return "branch"
    if "cycling" in s or "recurrent" in s:
        return "cycle"
    return "transient"


def _project_velocity_to_embedding(adata, E: np.ndarray, cfg: MetaFlowSummaryConfig) -> Optional[np.ndarray]:
    # Prefer final latent future state projected to nearest observed latent point.
    if (
        cfg.latent_key_for_velocity_projection is not None
        and cfg.future_latent_key is not None
        and cfg.latent_key_for_velocity_projection in adata.obsm
        and cfg.future_latent_key in adata.obsm
    ):
        Z = np.asarray(adata.obsm[cfg.latent_key_for_velocity_projection])
        Zf = np.asarray(adata.obsm[cfg.future_latent_key])
        if Z.shape == Zf.shape and Z.shape[0] == E.shape[0]:
            nn = NearestNeighbors(n_neighbors=1).fit(Z)
            _, ind = nn.kneighbors(Zf)
            Ef = E[ind[:, 0]]
            return Ef - E

    if cfg.velocity_embedding_key is not None and cfg.velocity_embedding_key in adata.obsm:
        V = np.asarray(adata.obsm[cfg.velocity_embedding_key])
        if V.ndim == 2 and V.shape[0] == E.shape[0] and V.shape[1] >= 2:
            return V[:, :2].astype(float)

    return None


def _draw_transition_edges_on_embedding(ax, centroids, T, cfg, color="black", alpha=0.40):
    K = T.shape[0]
    for i in range(K):
        row = T[i].copy()
        row[i] = 0
        candidates = set(np.where(row >= cfg.edge_threshold)[0].tolist())
        candidates.update(np.argsort(row)[::-1][:cfg.top_edges_per_state].tolist())

        for j in candidates:
            if i == j or row[j] <= 0:
                continue
            if np.any(np.isnan(centroids[i])) or np.any(np.isnan(centroids[j])):
                continue
            arrow = FancyArrowPatch(
                tuple(centroids[i]),
                tuple(centroids[j]),
                arrowstyle="-|>",
                mutation_scale=8 + 28 * row[j],
                lw=0.4 + 4.5 * row[j],
                color=color,
                alpha=alpha,
                shrinkA=8,
                shrinkB=8,
                connectionstyle="arc3,rad=0.06",
            )
            ax.add_patch(arrow)


def _state_label_text(k: int, state_summary: pd.DataFrame, dominant_ct: Dict[int, str], cfg: MetaFlowSummaryConfig) -> str:
    pieces = [f"M{k}"]
    if cfg.label_states_with_direction:
        lab = state_summary.loc[k, "label"]
        pieces.append(_short_direction(lab))
    if cfg.label_states_with_celltype and dominant_ct.get(k, ""):
        pieces.append(dominant_ct[k])
    return "\n".join(pieces)


# =============================================================================
# Figures
# =============================================================================

def plot_summary_umap_graph_diagnostics(
    adata,
    E: np.ndarray,
    state_int: np.ndarray,
    T: np.ndarray,
    centroids: np.ndarray,
    state_summary: pd.DataFrame,
    dominant_ct: Dict[int, str],
    output_dir: Path,
    cfg: MetaFlowSummaryConfig,
) -> Path:
    path = output_dir / "summary_umap_graph_diagnostics.png"
    K = T.shape[0]
    pal = _palette(K)

    fig = plt.figure(figsize=(15.8, 5.3))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.25, 0.95, 1.05], wspace=0.28)

    # Panel A: UMAP
    ax = fig.add_subplot(gs[0, 0])
    colors = [pal[k % len(pal)] for k in state_int]
    ax.scatter(E[:, 0], E[:, 1], c=colors, s=cfg.scatter_size, alpha=0.58, linewidths=0)

    dE = _project_velocity_to_embedding(adata, E, cfg)
    if dE is not None:
        rng = np.random.default_rng(cfg.random_state)
        idx = np.arange(E.shape[0])
        if len(idx) > cfg.max_arrows:
            idx = rng.choice(idx, size=cfg.max_arrows, replace=False)
        ax.quiver(
            E[idx, 0], E[idx, 1], dE[idx, 0], dE[idx, 1],
            angles="xy", scale_units="xy", scale=1.0,
            width=0.002, alpha=cfg.arrow_alpha, color="black"
        )

    _draw_transition_edges_on_embedding(ax, centroids, T, cfg)

    for k in range(K):
        if np.any(np.isnan(centroids[k])):
            continue
        ax.text(
            centroids[k, 0], centroids[k, 1],
            _state_label_text(k, state_summary, dominant_ct, cfg),
            ha="center", va="center", fontsize=7.5, weight="bold",
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="black", lw=0.4, alpha=0.86),
        )

    ax.set_title("A. Meta-state landscape with inferred direction", fontsize=12, weight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    sns.despine(ax=ax)

    # Panel B: abstract transition graph
    ax = fig.add_subplot(gs[0, 1])
    G = _build_transition_graph(T, cfg.edge_threshold, cfg.top_edges_per_state)

    # Geometry-aware positions from UMAP centroids, normalized.
    C = centroids.copy()
    valid = np.all(np.isfinite(C), axis=1)
    if valid.sum() >= 2:
        C[:, 0] = (C[:, 0] - np.nanmean(C[:, 0])) / (np.nanstd(C[:, 0]) + 1e-8)
        C[:, 1] = (C[:, 1] - np.nanmean(C[:, 1])) / (np.nanstd(C[:, 1]) + 1e-8)
    pos = {k: C[k] if valid[k] else np.random.default_rng(cfg.random_state).normal(size=2) for k in range(K)}

    for i, j, dat in G.edges(data=True):
        if i == j:
            continue
        p = dat.get("prob", T[i, j])
        arrow = FancyArrowPatch(
            tuple(pos[i]), tuple(pos[j]),
            arrowstyle="-|>",
            mutation_scale=8 + 30 * p,
            lw=0.4 + 5.0 * p,
            color="black", alpha=0.72,
            shrinkA=14, shrinkB=14,
            connectionstyle="arc3,rad=0.05",
        )
        ax.add_patch(arrow)

    for k in range(K):
        x, y = pos[k]
        ax.scatter([x], [y], s=260, color=pal[k % len(pal)], edgecolor="black", linewidth=0.7, zorder=5)
        ax.text(x, y + 0.13, _state_label_text(k, state_summary, dominant_ct, cfg), ha="center", va="bottom", fontsize=7.2)

    ax.set_title("B. Coarse meta-state transition graph", fontsize=12, weight="bold")
    ax.axis("off")

    # Panel C: diagnostics heatmap
    ax = fig.add_subplot(gs[0, 2])
    metrics = [
        "median_pseudotime", "incoming", "outgoing", "self_transition",
        "source_score", "sink_score", "branch_score", "recurrent_score"
    ]
    D = state_summary[metrics].copy()
    D.index = state_summary["meta_state"]
    Dz = D.apply(lambda x: (x - x.mean()) / (x.std() + 1e-8), axis=0)
    sns.heatmap(
        Dz,
        cmap="coolwarm",
        center=0,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "z-scored diagnostic"},
        ax=ax,
    )
    ax.set_title("C. Directionality diagnostics", fontsize=12, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")

    fig.suptitle(cfg.title, fontsize=15, weight="bold", y=1.03)
    plt.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close()
    return path


def plot_state_gene_heatmap(
    adata,
    genes: List[str],
    gene_groups: List[str],
    state_int: np.ndarray,
    state_summary: pd.DataFrame,
    dominant_ct: Dict[int, str],
    output_dir: Path,
    cfg: MetaFlowSummaryConfig,
) -> Path:
    path = output_dir / "state_gene_heatmap.png"

    H, order_states = _state_gene_heatmap_matrix(adata, genes, state_int, state_summary, cfg)

    labels = []
    for k in order_states:
        lab = f"M{k}"
        if dominant_ct.get(k, ""):
            lab += f"\n{dominant_ct[k]}"
        labels.append(lab)

    fig, ax = plt.subplots(figsize=(max(6.5, 0.58 * len(order_states) + 3.5), max(5.5, 0.22 * len(genes) + 2)))
    im = ax.imshow(
        H,
        aspect="auto",
        cmap="Greys",
        vmin=-cfg.heatmap_clip if cfg.zscore_heatmap else None,
        vmax=cfg.heatmap_clip if cfg.zscore_heatmap else None,
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(order_states)))
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_yticks(np.arange(len(genes)))
    ax.set_yticklabels(genes, fontsize=7)
    ax.set_title("Relevant genes summarized across detected meta-states", fontsize=13, weight="bold")
    ax.set_xlabel("Meta-states ordered by median pseudotime")
    ax.set_ylabel("Selected genes")

    # Gene group separators
    prev = gene_groups[0] if gene_groups else None
    for i, g in enumerate(gene_groups):
        if i > 0 and g != prev:
            ax.axhline(i - 0.5, color="white", lw=0.9)
        prev = g

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("z-scored expression" if cfg.zscore_heatmap else "mean expression")
    _savefig(path, cfg)
    return path


def plot_pseudotime_path_gene_heatmaps(
    adata,
    genes: List[str],
    gene_groups: List[str],
    paths: Dict[int, List[int]],
    state_int: np.ndarray,
    pt: Optional[np.ndarray],
    state_summary: pd.DataFrame,
    dominant_ct: Dict[int, str],
    output_dir: Path,
    cfg: MetaFlowSummaryConfig,
) -> Tuple[Path, Dict[int, pd.DataFrame]]:
    path = output_dir / "pseudotime_path_gene_heatmaps.png"

    if len(paths) == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No paths inferred", ha="center", va="center")
        ax.axis("off")
        plt.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
        plt.close()
        return path, {}

    K = int(state_int.max()) + 1
    pal = _palette(K)

    heatmaps = {}
    bin_tables = {}
    for term, pstates in paths.items():
        H, bin_state, bin_distance, bin_table = _path_heatmap_matrix(
            adata, genes, state_int, pt, pstates, cfg
        )
        heatmaps[term] = {
            "H": H,
            "bin_state": bin_state,
            "bin_distance": bin_distance,
            "path_states": pstates,
        }
        bin_tables[term] = bin_table

    n_paths = len(heatmaps)
    fig = plt.figure(figsize=(max(10.0, 3.5 * n_paths + 2.0), max(5.8, 0.22 * len(genes) + 2.8)))
    gs = GridSpec(
        3, n_paths,
        figure=fig,
        height_ratios=[0.18, 1.0, 0.10],
        wspace=0.07,
        hspace=0.05,
    )

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.text(
        0.5, 0.50,
        "Relevant genes along inferred source-to-terminal meta-state paths",
        ha="center", va="center", fontsize=13, weight="bold",
    )
    ax_title.axis("off")

    for c, (term, hdat) in enumerate(heatmaps.items()):
        H = hdat["H"]
        bin_state = hdat["bin_state"]
        bin_distance = hdat["bin_distance"]
        pstates = hdat["path_states"]

        ax = fig.add_subplot(gs[1, c])
        im = ax.imshow(
            H,
            aspect="auto",
            cmap="Greys",
            vmin=-cfg.heatmap_clip if cfg.zscore_heatmap else None,
            vmax=cfg.heatmap_clip if cfg.zscore_heatmap else None,
            interpolation="nearest",
        )

        terminal_label = dominant_ct.get(term, "")
        title = f"{' → '.join([f'M{x}' for x in pstates])}"
        if terminal_label:
            title = f"{terminal_label} path\n{title}"
        else:
            title = f"M{term} path\n{title}"
        ax.set_title(title, fontsize=9)

        ax.set_xticks([])
        if c == 0:
            ax.set_yticks(np.arange(len(genes)))
            ax.set_yticklabels(genes, fontsize=7)
        else:
            ax.set_yticks([])

        prev = gene_groups[0] if gene_groups else None
        for i, g in enumerate(gene_groups):
            if i > 0 and g != prev:
                ax.axhline(i - 0.5, color="white", lw=0.9)
            prev = g

        # Annotation strip
        axb = fig.add_subplot(gs[2, c])
        state_rgb = np.array([pal[int(s) % len(pal)] for s in bin_state])[:, :3]
        dist_rgb = plt.cm.viridis(bin_distance)[:, :3]
        strip = np.stack([state_rgb, dist_rgb], axis=0)
        axb.imshow(strip, aspect="auto", interpolation="nearest")
        axb.set_xticks([])
        if c == 0:
            axb.set_yticks([0, 1])
            axb.set_yticklabels(["state", "pseudo"], fontsize=7)
        else:
            axb.set_yticks([])

    cax = fig.add_axes([0.92, 0.22, 0.012, 0.55])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("z-scored expression" if cfg.zscore_heatmap else "mean expression")

    plt.savefig(path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close()
    return path, bin_tables


# =============================================================================
# Main API
# =============================================================================

def make_metaflow_summary(
    adata,
    cfg: Optional[MetaFlowSummaryConfig] = None,
) -> Dict[str, object]:
    """
    Create a general meta-state summary with UMAP, directionality, transition graph,
    and relevant gene heatmaps according to pseudotime and detected meta-states.
    """
    if cfg is None:
        cfg = MetaFlowSummaryConfig()

    outdir = _ensure_dir(cfg.output_dir)

    # Keys and data
    meta_state_key = _detect_meta_state_key(adata, cfg.meta_state_key)
    state_label_key = _detect_state_label_key(adata, meta_state_key, cfg.state_label_key)
    transition_key = _detect_transition_key(adata, cfg.transition_key, meta_state_key)
    pseudotime_key = _detect_pseudotime_key(adata, cfg.pseudotime_key)

    E = _get_embedding(adata, cfg.embedding_key)
    raw_state, state_int, state_names = _get_state_vector(adata, meta_state_key)
    K = len(state_names)
    pt = _get_pseudotime(adata, pseudotime_key)

    T = _transition_from_uns(adata, transition_key, K)
    if T is None:
        warnings.warn("No valid transition matrix found; using pseudotime/embedding fallback transition.")
        T = _fallback_transition_from_pseudotime_and_embedding(E, state_int, pt, K)

    centroids = _state_centroids(E, state_int, K)
    dominant_ct = _dominant_celltypes(
        adata, state_int, K, cfg.cell_type_key, max_chars=cfg.max_celltype_label_chars
    )

    state_labels_obs = adata.obs[state_label_key].astype(str) if state_label_key is not None else None
    state_summary = _score_and_classify_states(T, state_int, pt, state_labels_obs)

    # Add dominant cell type to summary
    state_summary["dominant_cell_type"] = [dominant_ct.get(k, "") for k in range(K)]
    state_summary["meta_state_key"] = meta_state_key
    state_summary["transition_key"] = transition_key if transition_key is not None else "fallback"
    state_summary["pseudotime_key"] = pseudotime_key if pseudotime_key is not None else "none"

    root, terminals = _infer_root_and_terminals(state_summary, cfg)
    paths = _infer_paths(T, root, terminals, cfg)

    path_summary_rows = []
    for term, pstates in paths.items():
        path_summary_rows.append({
            "root_state": f"M{root}",
            "terminal_state": f"M{term}",
            "terminal_dominant_cell_type": dominant_ct.get(term, ""),
            "path": " -> ".join([f"M{x}" for x in pstates]),
            "path_length": len(pstates),
        })
    path_summary = pd.DataFrame(path_summary_rows)

    # Gene selection
    candidate_genes = _select_candidate_genes(adata, cfg)
    state_gene_scores = _compute_state_gene_scores(adata, candidate_genes, state_int, cfg)
    path_gene_scores = _compute_path_pseudotime_gene_scores(
        adata, candidate_genes, paths, state_int, pt, cfg
    )
    selected_genes, gene_groups, selected_gene_table = _select_heatmap_genes(
        adata, state_gene_scores, path_gene_scores, cfg
    )

    if len(selected_genes) == 0:
        raise RuntimeError("No selected genes available for heatmaps.")

    # Save tables
    state_summary_path = outdir / "state_summary.csv"
    path_summary_path = outdir / "inferred_paths.csv"
    state_gene_path = outdir / "state_top_genes.csv"
    path_gene_path = outdir / "path_pseudotime_gene_scores.csv"
    selected_gene_path = outdir / "selected_heatmap_genes.csv"
    transition_path = outdir / "transition_matrix.csv"

    state_summary.to_csv(state_summary_path, index=False)
    path_summary.to_csv(path_summary_path, index=False)
    state_gene_scores.to_csv(state_gene_path, index=False)
    path_gene_scores.to_csv(path_gene_path, index=False)
    selected_gene_table.to_csv(selected_gene_path, index=False)
    pd.DataFrame(T, index=state_names, columns=state_names).to_csv(transition_path)

    # Figures
    paths_out = {}
    paths_out["summary_umap_graph_diagnostics"] = plot_summary_umap_graph_diagnostics(
        adata, E, state_int, T, centroids, state_summary, dominant_ct, outdir, cfg
    )
    paths_out["state_gene_heatmap"] = plot_state_gene_heatmap(
        adata, selected_genes, gene_groups, state_int, state_summary, dominant_ct, outdir, cfg
    )
    path_heatmap_path, bin_tables = plot_pseudotime_path_gene_heatmaps(
        adata, selected_genes, gene_groups, paths, state_int, pt, state_summary, dominant_ct, outdir, cfg
    )
    paths_out["pseudotime_path_gene_heatmaps"] = path_heatmap_path

    # Save bin tables for path heatmaps
    for term, tab in bin_tables.items():
        tab.to_csv(outdir / f"path_M{term}_bin_summary.csv", index=False)

    print("\n[MetaFlow general summary]")
    print(f"Meta-state key: {meta_state_key}")
    print(f"Transition key: {transition_key if transition_key is not None else 'fallback'}")
    print(f"Pseudotime key: {pseudotime_key if pseudotime_key is not None else 'none'}")
    print(f"Root/source state: M{root}")
    print(f"Terminal states: {', '.join([f'M{x}' for x in terminals])}")
    print(f"Selected genes for heatmaps: {len(selected_genes)}")
    print("\nState summary:")
    cols = ["meta_state", "label", "dominant_cell_type", "n_cells", "median_pseudotime", "incoming", "outgoing", "self_transition"]
    print(state_summary[cols].round(3).to_string(index=False))
    print("\nInferred paths:")
    if len(path_summary):
        print(path_summary.to_string(index=False))
    else:
        print("No paths inferred.")

    return {
        "state_summary": state_summary,
        "path_summary": path_summary,
        "state_gene_scores": state_gene_scores,
        "path_gene_scores": path_gene_scores,
        "selected_genes": selected_gene_table,
        "transition_matrix": T,
        "root_state": root,
        "terminal_states": terminals,
        "paths": paths,
        "figure_paths": paths_out,
        "table_paths": {
            "state_summary": state_summary_path,
            "inferred_paths": path_summary_path,
            "state_top_genes": state_gene_path,
            "path_pseudotime_gene_scores": path_gene_path,
            "selected_heatmap_genes": selected_gene_path,
            "transition_matrix": transition_path,
        },
    }


if __name__ == "__main__":
    # Example:
    #
    # import scanpy as sc
    # adata = sc.read_h5ad("adata_with_velot_metaflow_velocity_metastates.h5ad")
    #
    # cfg = MetaFlowSummaryConfig(
    #     meta_state_key="velot_metaflow_state",
    #     transition_key="velot_metaflow_transition_matrix",
    #     pseudotime_key="velot_metaflow_pseudotime",
    #     cell_type_key="cell_type",
    #     embedding_key="X_umap",
    #     output_dir="pancreas_metaflow_summary",
    #     title="Pancreas VelOT-MetaFlow",
    # )
    #
    # res = make_metaflow_summary(adata, cfg)
    pass

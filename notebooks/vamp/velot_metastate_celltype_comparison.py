"""
Post-hoc comparison of VelOT-MetaFlow meta-states with real cell-type annotations.

Use this AFTER running:
    out = run_velot_metaflow_with_velot_velocity(adata, cfg)

or after loading the saved AnnData containing:
    adata.obs["velot_metaflow_state"]
    adata.obs["velot_metaflow_state_label"]

Also supports:
    adata.obs["velot_quantum_msm_state"]
    adata.obs["velot_vampflow_state"]

Outputs:
    - UMAP: meta-states vs real cell types
    - contingency count heatmap
    - per-meta-state cell-type composition heatmap
    - per-cell-type meta-state composition heatmap
    - log2 observed/expected enrichment heatmap
    - Fisher exact enrichment dotplot
    - stacked-bar plots
    - alluvial/ribbon plot
    - CSV tables with enrichment statistics, purity, entropy, ARI/NMI
"""

from __future__ import annotations

import re
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

import seaborn as sns
from scipy.stats import fisher_exact

try:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
except Exception:  # pragma: no cover
    adjusted_rand_score = None
    normalized_mutual_info_score = None


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _natural_sort_key(x):
    x = str(x)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", x)]


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction without statsmodels."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[order[i]] = prev
    return np.clip(q, 0, 1)


def detect_metastate_key(adata, preferred: Optional[str] = None) -> str:
    if preferred is not None:
        if preferred not in adata.obs:
            raise KeyError(
                f"Requested meta_state_key={preferred!r} not found in adata.obs. "
                f"Available columns include: {list(adata.obs.columns[:40])}"
            )
        return preferred

    candidates = [
        "velot_metaflow_state",
        "velot_quantum_msm_state",
        "velot_vampflow_state",
        "metaflow_state",
        "quantum_msm_state",
        "vampflow_state",
    ]
    for key in candidates:
        if key in adata.obs:
            return key

    possible = [c for c in adata.obs.columns if "state" in c.lower() or "meta" in c.lower()]
    raise KeyError(
        "Could not auto-detect a meta-state column. "
        f"Candidate state-like columns found: {possible}. "
        "Pass meta_state_key='your_column_name'."
    )


def detect_state_label_key(adata, meta_state_key: str, preferred: Optional[str] = None) -> Optional[str]:
    if preferred is not None:
        return preferred if preferred in adata.obs else None

    candidates = [
        meta_state_key.replace("_state", "_state_label"),
        "velot_metaflow_state_label",
        "velot_quantum_msm_state_label",
        "velot_vampflow_state_label",
        "metaflow_state_label",
    ]
    for key in candidates:
        if key in adata.obs:
            return key
    return None


def check_cell_type_key(adata, cell_type_key: str) -> str:
    if cell_type_key not in adata.obs:
        likely = [
            c for c in adata.obs.columns
            if any(s in c.lower() for s in ["cell", "type", "annotation", "cluster", "subtype", "lineage", "ductal", "acinar"])
        ]
        raise KeyError(
            f"cell_type_key={cell_type_key!r} not found in adata.obs. "
            f"Possible annotation columns: {likely}"
        )
    return cell_type_key


def clean_annotation_series(s: pd.Series, missing_label: str = "Unknown") -> pd.Series:
    out = s.astype("object").astype(str)
    bad = out.isin(["nan", "None", "NA", "NaN", "", "unknown"])
    out[bad] = missing_label
    return out.astype(str)


def build_contingency(
    adata,
    meta_state_key: Optional[str] = None,
    cell_type_key: str = "cell_type",
    state_label_key: Optional[str] = None,
    min_celltype_count: int = 10,
    max_celltypes: Optional[int] = None,
    other_label: str = "Other",
) -> Dict[str, object]:
    """Build state x cell-type contingency table and associated vectors."""
    meta_state_key = detect_metastate_key(adata, meta_state_key)
    state_label_key = detect_state_label_key(adata, meta_state_key, state_label_key)
    cell_type_key = check_cell_type_key(adata, cell_type_key)

    states = clean_annotation_series(adata.obs[meta_state_key])
    celltypes = clean_annotation_series(adata.obs[cell_type_key])

    # Collapse rare cell types if requested.
    ct_counts = celltypes.value_counts()
    keep = ct_counts.index[ct_counts >= min_celltype_count].tolist()

    if max_celltypes is not None and len(keep) > max_celltypes:
        keep = ct_counts.head(max_celltypes).index.tolist()

    celltypes2 = celltypes.where(celltypes.isin(keep), other_label)

    state_order = sorted(states.unique(), key=_natural_sort_key)
    celltype_order = celltypes2.value_counts().index.tolist()

    contingency = pd.crosstab(
        pd.Categorical(states, categories=state_order, ordered=True),
        pd.Categorical(celltypes2, categories=celltype_order, ordered=True),
        dropna=False,
    )
    contingency.index.name = "meta_state"
    contingency.columns.name = "cell_type"

    state_labels = None
    if state_label_key is not None:
        tmp = pd.DataFrame({"state": states, "label": clean_annotation_series(adata.obs[state_label_key])})
        state_labels = tmp.groupby("state")["label"].agg(lambda x: x.value_counts().index[0]).to_dict()

    return {
        "meta_state_key": meta_state_key,
        "state_label_key": state_label_key,
        "cell_type_key": cell_type_key,
        "states": states,
        "celltypes": celltypes2,
        "contingency": contingency,
        "state_order": state_order,
        "celltype_order": celltype_order,
        "state_labels": state_labels,
    }


# ---------------------------------------------------------------------
# Enrichment and summary statistics
# ---------------------------------------------------------------------

def compute_enrichment_statistics(contingency: pd.DataFrame) -> pd.DataFrame:
    """
    For each meta-state x cell-type pair:
      - observed count
      - expected count under independence
      - log2 enrichment observed/expected
      - Fisher exact p-value
      - BH-FDR q-value
      - odds ratio
      - proportions within state and within cell type
    """
    C = contingency.to_numpy(dtype=float)
    N = C.sum()
    state_totals = C.sum(axis=1)
    type_totals = C.sum(axis=0)

    rows = []
    for i, state in enumerate(contingency.index):
        for j, ct in enumerate(contingency.columns):
            obs = C[i, j]
            expected = state_totals[i] * type_totals[j] / max(N, 1)

            a = obs
            b = state_totals[i] - obs
            c = type_totals[j] - obs
            d = N - a - b - c

            table = np.array([[a, b], [c, d]], dtype=float)
            try:
                odds, p = fisher_exact(table, alternative="greater")
            except Exception:
                odds, p = np.nan, np.nan

            log2_enrichment = np.log2((obs + 0.5) / (expected + 0.5))
            prop_within_state = obs / state_totals[i] if state_totals[i] > 0 else 0
            prop_within_celltype = obs / type_totals[j] if type_totals[j] > 0 else 0

            rows.append({
                "meta_state": state,
                "cell_type": ct,
                "observed": int(obs),
                "expected": expected,
                "log2_enrichment": log2_enrichment,
                "odds_ratio": odds,
                "p_value": p,
                "prop_within_state": prop_within_state,
                "prop_within_celltype": prop_within_celltype,
                "state_total": int(state_totals[i]),
                "celltype_total": int(type_totals[j]),
            })

    df = pd.DataFrame(rows)
    df["q_value"] = _bh_fdr(df["p_value"].fillna(1).to_numpy())
    df["minus_log10_q"] = -np.log10(df["q_value"].clip(lower=1e-300))
    df["enriched"] = (df["q_value"] < 0.05) & (df["log2_enrichment"] > 0)
    return df


def compute_state_summary(
    contingency: pd.DataFrame,
    enrichment_df: pd.DataFrame,
    states: pd.Series,
    celltypes: pd.Series,
) -> pd.DataFrame:
    C = contingency.to_numpy(dtype=float)
    state_totals = C.sum(axis=1)

    rows = []
    for i, state in enumerate(contingency.index):
        row = C[i]
        total = state_totals[i]
        if total > 0:
            prop = row / total
            entropy = -np.sum(prop[prop > 0] * np.log2(prop[prop > 0]))
            norm_entropy = entropy / np.log2(len(prop)) if len(prop) > 1 else 0
            top_j = int(np.argmax(row))
            top_ct = contingency.columns[top_j]
            top_prop = prop[top_j]
        else:
            entropy, norm_entropy, top_ct, top_prop = np.nan, np.nan, None, np.nan

        sub = enrichment_df[enrichment_df["meta_state"].astype(str) == str(state)].sort_values(
            ["q_value", "log2_enrichment"], ascending=[True, False]
        )
        enriched_types = sub[(sub["q_value"] < 0.05) & (sub["log2_enrichment"] > 0)]["cell_type"].tolist()
        top_enriched = enriched_types[0] if len(enriched_types) else None

        rows.append({
            "meta_state": state,
            "n_cells": int(total),
            "dominant_cell_type": top_ct,
            "dominant_cell_type_fraction": top_prop,
            "composition_entropy": entropy,
            "normalized_composition_entropy": norm_entropy,
            "top_enriched_cell_type": top_enriched,
            "n_significantly_enriched_cell_types": len(enriched_types),
            "significantly_enriched_cell_types": "; ".join(enriched_types),
        })

    summary = pd.DataFrame(rows)

    if adjusted_rand_score is not None:
        summary.attrs["ARI_state_vs_celltype"] = adjusted_rand_score(states.astype(str), celltypes.astype(str))
    if normalized_mutual_info_score is not None:
        summary.attrs["NMI_state_vs_celltype"] = normalized_mutual_info_score(states.astype(str), celltypes.astype(str))

    return summary


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def _savefig(path: Path, dpi: int = 300):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def _get_embedding(adata, embedding_key: str = "X_umap") -> np.ndarray:
    if embedding_key not in adata.obsm:
        raise KeyError(f"embedding_key={embedding_key!r} not found in adata.obsm.")
    E = np.asarray(adata.obsm[embedding_key])
    if E.ndim != 2 or E.shape[1] < 2:
        raise ValueError(f"adata.obsm[{embedding_key!r}] must be n_cells x >=2.")
    return E[:, :2]


def plot_umap_metastate_vs_celltype(
    adata,
    states: pd.Series,
    celltypes: pd.Series,
    output_dir: Path,
    embedding_key: str = "X_umap",
    dpi: int = 300,
    max_legend_items: int = 30,
) -> Path:
    E = _get_embedding(adata, embedding_key)

    state_values = pd.Categorical(states, categories=sorted(states.unique(), key=_natural_sort_key), ordered=True)
    celltype_values = pd.Categorical(celltypes, categories=celltypes.value_counts().index.tolist(), ordered=True)

    path = output_dir / "01_umap_metastates_vs_real_celltypes.png"

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.7))

    state_palette = sns.color_palette("tab20", n_colors=max(len(state_values.categories), 3))
    state_lut = dict(zip(state_values.categories, state_palette))
    axes[0].scatter(E[:, 0], E[:, 1], c=[state_lut[x] for x in state_values], s=6, linewidths=0, alpha=0.85)
    axes[0].set_title("Detected VelOT-MetaFlow meta-states", fontsize=13, weight="bold")
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")

    ct_palette = sns.color_palette("tab20", n_colors=max(len(celltype_values.categories), 3))
    ct_lut = dict(zip(celltype_values.categories, ct_palette))
    axes[1].scatter(E[:, 0], E[:, 1], c=[ct_lut[x] for x in celltype_values], s=6, linewidths=0, alpha=0.85)
    axes[1].set_title("Reference / real cell-type annotations", fontsize=13, weight="bold")
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")

    for ax in axes:
        sns.despine(ax=ax)

    state_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=state_lut[x], markersize=6, label=str(x))
        for x in state_values.categories[:max_legend_items]
    ]
    ct_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=ct_lut[x], markersize=6, label=str(x))
        for x in celltype_values.categories[:max_legend_items]
    ]

    axes[0].legend(handles=state_handles, bbox_to_anchor=(0.5, -0.12), loc="upper center",
                   ncol=min(5, len(state_handles)), frameon=False, fontsize=8)
    axes[1].legend(handles=ct_handles, bbox_to_anchor=(0.5, -0.12), loc="upper center",
                   ncol=min(5, len(ct_handles)), frameon=False, fontsize=8)

    plt.subplots_adjust(bottom=0.22, wspace=0.15)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path


def plot_count_heatmap(contingency: pd.DataFrame, output_dir: Path, dpi: int = 300) -> Path:
    path = output_dir / "02_contingency_counts_heatmap.png"
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * contingency.shape[1] + 3), max(4.8, 0.45 * contingency.shape[0] + 2)))
    sns.heatmap(
        contingency,
        cmap="mako",
        annot=True if contingency.size <= 120 else False,
        fmt="d",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "number of cells"},
        ax=ax,
    )
    ax.set_title("Meta-state × real cell-type contingency counts", fontsize=13, weight="bold")
    ax.set_xlabel("Real cell type")
    ax.set_ylabel("Detected meta-state")
    _savefig(path, dpi)
    return path


def plot_proportion_heatmaps(contingency: pd.DataFrame, output_dir: Path, dpi: int = 300) -> Tuple[Path, Path]:
    state_prop = contingency.div(contingency.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    type_prop = contingency.div(contingency.sum(axis=0).replace(0, np.nan), axis=1).fillna(0)

    path1 = output_dir / "03_celltype_fraction_within_each_metastate_heatmap.png"
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * contingency.shape[1] + 3), max(4.8, 0.45 * contingency.shape[0] + 2)))
    sns.heatmap(
        state_prop,
        cmap="viridis",
        vmin=0,
        vmax=min(1.0, max(0.25, state_prop.to_numpy().max())),
        annot=True if contingency.size <= 120 else False,
        fmt=".2f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "fraction within meta-state"},
        ax=ax,
    )
    ax.set_title("Cell-type composition within each detected meta-state", fontsize=13, weight="bold")
    ax.set_xlabel("Real cell type")
    ax.set_ylabel("Detected meta-state")
    _savefig(path1, dpi)

    path2 = output_dir / "04_metastate_fraction_within_each_celltype_heatmap.png"
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * contingency.shape[1] + 3), max(4.8, 0.45 * contingency.shape[0] + 2)))
    sns.heatmap(
        type_prop,
        cmap="rocket_r",
        vmin=0,
        vmax=min(1.0, max(0.25, type_prop.to_numpy().max())),
        annot=True if contingency.size <= 120 else False,
        fmt=".2f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "fraction within cell type"},
        ax=ax,
    )
    ax.set_title("Meta-state distribution within each real cell type", fontsize=13, weight="bold")
    ax.set_xlabel("Real cell type")
    ax.set_ylabel("Detected meta-state")
    _savefig(path2, dpi)

    return path1, path2


def plot_enrichment_heatmap(enrichment_df: pd.DataFrame, contingency: pd.DataFrame, output_dir: Path, dpi: int = 300) -> Path:
    path = output_dir / "05_log2_observed_expected_enrichment_heatmap.png"

    mat = enrichment_df.pivot(index="meta_state", columns="cell_type", values="log2_enrichment")
    mat = mat.reindex(index=contingency.index, columns=contingency.columns)

    sig = enrichment_df.pivot(index="meta_state", columns="cell_type", values="enriched")
    sig = sig.reindex(index=contingency.index, columns=contingency.columns).fillna(False)

    annot = mat.copy().astype(object)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            star = "*" if bool(sig.iloc[i, j]) else ""
            annot.iloc[i, j] = f"{val:.2f}{star}"

    vmax = np.nanpercentile(np.abs(mat.to_numpy()), 97)
    vmax = max(vmax, 1.0)

    fig, ax = plt.subplots(figsize=(max(7, 0.58 * mat.shape[1] + 3), max(4.8, 0.45 * mat.shape[0] + 2)))
    sns.heatmap(
        mat,
        cmap="coolwarm",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=annot if mat.size <= 150 else False,
        fmt="",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "log2 observed / expected"},
        ax=ax,
    )
    ax.set_title("Meta-state enrichment for real cell types\n* FDR < 0.05 and log2 enrichment > 0", fontsize=13, weight="bold")
    ax.set_xlabel("Real cell type")
    ax.set_ylabel("Detected meta-state")
    _savefig(path, dpi)
    return path


def plot_enrichment_dotplot(
    enrichment_df: pd.DataFrame,
    contingency: pd.DataFrame,
    output_dir: Path,
    dpi: int = 300,
    q_threshold: float = 0.05,
) -> Path:
    path = output_dir / "06_fisher_enrichment_dotplot.png"

    df = enrichment_df.copy()
    df["meta_state"] = pd.Categorical(df["meta_state"], categories=contingency.index.tolist(), ordered=True)
    df["cell_type"] = pd.Categorical(df["cell_type"], categories=contingency.columns.tolist(), ordered=True)
    df["dot_size"] = np.clip(df["minus_log10_q"], 0, 20)
    df["significant"] = (df["q_value"] < q_threshold) & (df["log2_enrichment"] > 0)

    vmax = np.nanpercentile(np.abs(df["log2_enrichment"]), 97)
    vmax = max(vmax, 1.0)

    fig, ax = plt.subplots(figsize=(max(8, 0.65 * contingency.shape[1] + 3), max(5.2, 0.55 * contingency.shape[0] + 2)))
    sc = ax.scatter(
        x=df["cell_type"].cat.codes,
        y=df["meta_state"].cat.codes,
        s=20 + 18 * df["dot_size"],
        c=df["log2_enrichment"],
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        edgecolor=np.where(df["significant"], "black", "none"),
        linewidth=np.where(df["significant"], 0.7, 0.0),
        alpha=0.85,
    )

    ax.set_xticks(np.arange(len(contingency.columns)))
    ax.set_xticklabels(contingency.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(contingency.index)))
    ax.set_yticklabels(contingency.index)
    ax.set_xlabel("Real cell type")
    ax.set_ylabel("Detected meta-state")
    ax.set_title("Fisher enrichment of real cell types within detected meta-states", fontsize=13, weight="bold")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log2 observed / expected")

    sizes = [1, 2, 5, 10]
    handles = [plt.scatter([], [], s=20 + 18 * s, color="gray", alpha=0.7, label=f"{s}") for s in sizes]
    ax.legend(handles=handles, title="-log10 FDR", frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    sns.despine(ax=ax)
    _savefig(path, dpi)
    return path


def plot_stacked_bars(contingency: pd.DataFrame, output_dir: Path, dpi: int = 300) -> Tuple[Path, Path]:
    state_prop = contingency.div(contingency.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    type_prop = contingency.div(contingency.sum(axis=0).replace(0, np.nan), axis=1).fillna(0)

    palette = sns.color_palette("tab20", n_colors=max(len(contingency.columns), 3))
    color_map = dict(zip(contingency.columns, palette))

    path1 = output_dir / "07_stacked_bar_celltype_composition_by_metastate.png"
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(state_prop.index) + 4), 5.2))
    bottom = np.zeros(len(state_prop.index))
    x = np.arange(len(state_prop.index))
    for ct in state_prop.columns:
        vals = state_prop[ct].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=ct, color=color_map[ct], width=0.82)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(state_prop.index, rotation=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction within meta-state")
    ax.set_xlabel("Detected meta-state")
    ax.set_title("Real cell-type composition of each detected meta-state", fontsize=13, weight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=8)
    sns.despine(ax=ax)
    _savefig(path1, dpi)

    palette2 = sns.color_palette("tab20", n_colors=max(len(contingency.index), 3))
    state_colors = dict(zip(contingency.index, palette2))

    path2 = output_dir / "08_stacked_bar_metastate_composition_by_celltype.png"
    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(type_prop.columns) + 4), 5.2))
    bottom = np.zeros(len(type_prop.columns))
    x = np.arange(len(type_prop.columns))
    for state in type_prop.index:
        vals = type_prop.loc[state].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=state, color=state_colors[state], width=0.82)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(type_prop.columns, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction within real cell type")
    ax.set_xlabel("Real cell type")
    ax.set_title("Detected meta-state composition of each real cell type", fontsize=13, weight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=8)
    sns.despine(ax=ax)
    _savefig(path2, dpi)

    return path1, path2


def _ribbon(ax, x0, x1, y0a, y0b, y1a, y1b, color, alpha=0.55):
    dx = x1 - x0
    verts = [
        (x0, y0a),
        (x0 + dx * 0.45, y0a),
        (x1 - dx * 0.45, y1a),
        (x1, y1a),
        (x1, y1b),
        (x1 - dx * 0.45, y1b),
        (x0 + dx * 0.45, y0b),
        (x0, y0b),
        (x0, y0a),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    patch = PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha)
    ax.add_patch(patch)


def plot_alluvial_metastate_to_celltype(
    contingency: pd.DataFrame,
    output_dir: Path,
    dpi: int = 300,
    min_flow_fraction: float = 0.005,
) -> Path:
    path = output_dir / "09_alluvial_metastate_to_celltype.png"

    C = contingency.astype(float)
    N = C.to_numpy().sum()
    if N <= 0:
        raise ValueError("Empty contingency table.")

    state_tot = C.sum(axis=1) / N
    type_tot = C.sum(axis=0) / N

    state_y0 = {}
    y = 0.0
    gap = 0.012
    for state, frac in state_tot.items():
        state_y0[state] = y
        y += frac + gap

    type_y0 = {}
    y = 0.0
    for ct, frac in type_tot.items():
        type_y0[ct] = y
        y += frac + gap

    palette = sns.color_palette("tab20", n_colors=max(len(C.columns), 3))
    ct_colors = dict(zip(C.columns, palette))

    fig, ax = plt.subplots(figsize=(9.5, max(5.5, 0.35 * (len(C.index) + len(C.columns)))))

    x0, x1 = 0.25, 0.75
    bar_w = 0.045

    state_offset = {s: state_y0[s] for s in C.index}
    type_offset = {ct: type_y0[ct] for ct in C.columns}

    for state in C.index:
        for ct in C.columns:
            frac = C.loc[state, ct] / N
            if frac < min_flow_fraction:
                continue
            y0a = state_offset[state]
            y0b = y0a + frac
            y1a = type_offset[ct]
            y1b = y1a + frac
            _ribbon(ax, x0 + bar_w / 2, x1 - bar_w / 2, y0a, y0b, y1a, y1b, color=ct_colors[ct], alpha=0.45)
            state_offset[state] = y0b
            type_offset[ct] = y1b

    for state, frac in state_tot.items():
        y0 = state_y0[state]
        ax.add_patch(plt.Rectangle((x0 - bar_w / 2, y0), bar_w, frac, facecolor="gray", edgecolor="black", lw=0.4))
        ax.text(x0 - 0.06, y0 + frac / 2, str(state), va="center", ha="right", fontsize=9)

    for ct, frac in type_tot.items():
        y0 = type_y0[ct]
        ax.add_patch(plt.Rectangle((x1 - bar_w / 2, y0), bar_w, frac, facecolor=ct_colors[ct], edgecolor="black", lw=0.4))
        ax.text(x1 + 0.06, y0 + frac / 2, str(ct), va="center", ha="left", fontsize=9)

    ymax = max(max(state_y0.values()) + state_tot.iloc[-1] + 0.12, max(type_y0.values()) + type_tot.iloc[-1] + 0.12)
    ax.text(x0, ymax - 0.02, "Detected\nmeta-states", ha="center", va="top", fontsize=12, weight="bold")
    ax.text(x1, ymax - 0.02, "Real\ncell types", ha="center", va="top", fontsize=12, weight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    ax.set_title("Alluvial comparison: detected meta-states versus real cell types", fontsize=13, weight="bold")
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path


# ---------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------

def compare_metastates_to_celltypes(
    adata,
    meta_state_key: Optional[str] = None,
    cell_type_key: str = "cell_type",
    state_label_key: Optional[str] = None,
    embedding_key: str = "X_umap",
    output_dir: str | Path = "metastate_celltype_comparison",
    min_celltype_count: int = 10,
    max_celltypes: Optional[int] = None,
    dpi: int = 300,
) -> Dict[str, object]:
    """
    Complete post-hoc comparison of detected meta-states against real cell types.
    """
    output_dir = _ensure_dir(output_dir)

    data = build_contingency(
        adata=adata,
        meta_state_key=meta_state_key,
        cell_type_key=cell_type_key,
        state_label_key=state_label_key,
        min_celltype_count=min_celltype_count,
        max_celltypes=max_celltypes,
    )

    states = data["states"]
    celltypes = data["celltypes"]
    contingency = data["contingency"]

    enrichment_df = compute_enrichment_statistics(contingency)
    state_summary = compute_state_summary(contingency, enrichment_df, states, celltypes)

    paths = {}
    paths["contingency_csv"] = output_dir / "contingency_meta_state_by_celltype.csv"
    paths["state_prop_csv"] = output_dir / "celltype_fraction_within_each_metastate.csv"
    paths["celltype_prop_csv"] = output_dir / "metastate_fraction_within_each_celltype.csv"
    paths["enrichment_csv"] = output_dir / "fisher_enrichment_statistics.csv"
    paths["state_summary_csv"] = output_dir / "metastate_celltype_summary.csv"

    contingency.to_csv(paths["contingency_csv"])
    contingency.div(contingency.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).to_csv(paths["state_prop_csv"])
    contingency.div(contingency.sum(axis=0).replace(0, np.nan), axis=1).fillna(0).to_csv(paths["celltype_prop_csv"])
    enrichment_df.to_csv(paths["enrichment_csv"], index=False)
    state_summary.to_csv(paths["state_summary_csv"], index=False)

    paths["umap_metastates_vs_celltypes"] = plot_umap_metastate_vs_celltype(
        adata, states, celltypes, output_dir, embedding_key=embedding_key, dpi=dpi
    )
    paths["counts_heatmap"] = plot_count_heatmap(contingency, output_dir, dpi=dpi)
    p1, p2 = plot_proportion_heatmaps(contingency, output_dir, dpi=dpi)
    paths["celltype_fraction_within_metastate_heatmap"] = p1
    paths["metastate_fraction_within_celltype_heatmap"] = p2
    paths["enrichment_heatmap"] = plot_enrichment_heatmap(enrichment_df, contingency, output_dir, dpi=dpi)
    paths["enrichment_dotplot"] = plot_enrichment_dotplot(enrichment_df, contingency, output_dir, dpi=dpi)
    p1, p2 = plot_stacked_bars(contingency, output_dir, dpi=dpi)
    paths["stacked_bar_celltype_by_metastate"] = p1
    paths["stacked_bar_metastate_by_celltype"] = p2
    paths["alluvial"] = plot_alluvial_metastate_to_celltype(contingency, output_dir, dpi=dpi)

    print("\n[Meta-state vs real cell-type comparison]")
    print(f"Meta-state key: {data['meta_state_key']}")
    print(f"Cell-type key:  {data['cell_type_key']}")
    print(f"Number of cells: {int(contingency.to_numpy().sum())}")
    print(f"Number of meta-states: {contingency.shape[0]}")
    print(f"Number of cell types:  {contingency.shape[1]}")

    if adjusted_rand_score is not None:
        ari = adjusted_rand_score(states.astype(str), celltypes.astype(str))
        print(f"Adjusted Rand Index meta-state vs cell type: {ari:.3f}")
    else:
        ari = None

    if normalized_mutual_info_score is not None:
        nmi = normalized_mutual_info_score(states.astype(str), celltypes.astype(str))
        print(f"Normalized mutual information meta-state vs cell type: {nmi:.3f}")
    else:
        nmi = None

    print("\nDominant / enriched cell types per meta-state:")
    cols = [
        "meta_state",
        "n_cells",
        "dominant_cell_type",
        "dominant_cell_type_fraction",
        "top_enriched_cell_type",
        "n_significantly_enriched_cell_types",
    ]
    print(state_summary[cols].round(3).to_string(index=False))

    return {
        "contingency": contingency,
        "enrichment": enrichment_df,
        "state_summary": state_summary,
        "paths": paths,
        "meta_state_key": data["meta_state_key"],
        "cell_type_key": data["cell_type_key"],
        "state_label_key": data["state_label_key"],
        "ARI": ari,
        "NMI": nmi,
    }


if __name__ == "__main__":
    # Example:
    # import scanpy as sc
    # adata = sc.read_h5ad("adata_with_velot_metaflow_velocity_metastates.h5ad")
    # res = compare_metastates_to_celltypes(
    #     adata,
    #     meta_state_key="velot_metaflow_state",
    #     cell_type_key="cell_type",
    #     embedding_key="X_umap",
    #     output_dir="metastate_celltype_comparison",
    # )
    pass

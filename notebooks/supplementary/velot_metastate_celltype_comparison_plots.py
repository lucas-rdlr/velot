"""
Plotting-only module.  Reads CSVs + config.json produced by
velot_metastate_celltype_analysis.run_analysis() and regenerates every figure.

Usage
-----
    from velot_metastate_celltype_plots import replot_all
    replot_all("metastate_celltype_comparison_pancreas", dpi=300)

Or selectively:
    from velot_metastate_celltype_plots import load_analysis, plot_enrichment_dotplot
    tables = load_analysis("metastate_celltype_comparison_pancreas")
    plot_enrichment_dotplot(tables, "metastate_celltype_comparison_pancreas")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────

def load_analysis(output_dir: str | Path) -> Dict[str, object]:
    """
    Load every CSV and the config.json written by run_analysis().
    Returns a dict with keys:
        contingency, state_prop, celltype_prop, enrichment, state_summary,
        cell_data, config
    """
    d = Path(output_dir)

    contingency = pd.read_csv(d / "contingency.csv", index_col=0)
    contingency.index.name = "meta_state"
    contingency.columns.name = "cell_type"

    state_prop = pd.read_csv(d / "state_prop.csv", index_col=0)
    state_prop.index.name = "meta_state"
    state_prop.columns.name = "cell_type"

    celltype_prop = pd.read_csv(d / "celltype_prop.csv", index_col=0)
    celltype_prop.index.name = "meta_state"
    celltype_prop.columns.name = "cell_type"

    enrichment = pd.read_csv(d / "enrichment.csv")
    state_summary = pd.read_csv(d / "state_summary.csv")
    cell_data = pd.read_csv(d / "cell_data.csv", index_col=0)

    with open(d / "config.json") as f:
        config = json.load(f)

    return {
        "contingency": contingency,
        "state_prop": state_prop,
        "celltype_prop": celltype_prop,
        "enrichment": enrichment,
        "state_summary": state_summary,
        "cell_data": cell_data,
        "config": config,
    }


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _savefig(path: Path, dpi: int = 300):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────
# Individual plot functions
# ─────────────────────────────────────────────────────────────────────

def plot_umap_metastate_vs_celltype(
    tables: Dict,
    output_dir: str | Path,
    dpi: int = 300,
    max_legend_items: int = 30,
) -> Path:
    output_dir = _ensure_dir(output_dir)
    path = output_dir / "01_umap_metastates_vs_real_celltypes.png"

    cell_data = tables["cell_data"]
    if "umap_1" not in cell_data.columns or "umap_2" not in cell_data.columns:
        print("[skip] No UMAP coordinates in cell_data.csv — skipping UMAP plot.")
        return path

    E = cell_data[["umap_1", "umap_2"]].to_numpy()

    config = tables["config"]
    state_order = config["state_order"]
    celltype_order = config["celltype_order"]

    states = pd.Categorical(cell_data["meta_state"].astype(str), categories=state_order, ordered=True)
    celltypes = pd.Categorical(cell_data["cell_type"].astype(str), categories=celltype_order, ordered=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.7))

    # left: meta-states
    state_palette = sns.color_palette("tab20", n_colors=max(len(state_order), 3))
    state_lut = dict(zip(state_order, state_palette))
    axes[0].scatter(E[:, 0], E[:, 1], c=[state_lut.get(str(x), "gray") for x in states],
                    s=6, linewidths=0, alpha=0.85)
    axes[0].set_xlabel("UMAP 1", fontsize=12)
    axes[0].set_ylabel("UMAP 2", fontsize=12)
    axes[0].tick_params(labelsize=12)

    # right: cell types
    ct_palette = sns.color_palette("tab20", n_colors=max(len(celltype_order), 3))
    ct_lut = dict(zip(celltype_order, ct_palette))
    axes[1].scatter(E[:, 0], E[:, 1], c=[ct_lut.get(str(x), "gray") for x in celltypes],
                    s=6, linewidths=0, alpha=0.85)
    axes[1].set_xlabel("UMAP 1", fontsize=12)
    axes[1].set_ylabel("UMAP 2", fontsize=12)
    axes[1].tick_params(labelsize=12)

    for ax in axes:
        sns.despine(ax=ax)

    state_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=state_lut[x], markersize=6, label=str(x))
        for x in state_order[:max_legend_items]
    ]
    ct_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=ct_lut[x], markersize=6, label=str(x))
        for x in celltype_order[:max_legend_items]
    ]
    axes[0].legend(handles=state_handles, bbox_to_anchor=(0.5, -0.12), loc="upper center",
                   ncol=min(5, len(state_handles)), frameon=False, fontsize=8)
    axes[1].legend(handles=ct_handles, bbox_to_anchor=(0.5, -0.12), loc="upper center",
                   ncol=min(5, len(ct_handles)), frameon=False, fontsize=8)

    plt.subplots_adjust(bottom=0.22, wspace=0.15)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path


def plot_count_heatmap(
    tables: Dict,
    output_dir: str | Path,
    dpi: int = 300,
) -> Path:
    output_dir = _ensure_dir(output_dir)
    path = output_dir / "02_contingency_counts_heatmap.png"
    contingency = tables["contingency"]

    fig, ax = plt.subplots(figsize=(max(7, 0.55 * contingency.shape[1] + 3),
                                    max(4.8, 0.45 * contingency.shape[0] + 2)))
    sns.heatmap(
        contingency, cmap="viridis",
        annot=contingency.size <= 120, fmt="d",
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "number of cells"}, ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
    ax.set_xlabel("Real cell type", fontsize=14)
    ax.set_ylabel("Detected meta-state", fontsize=14)
    ax.set_title("")
    _savefig(path, dpi)
    return path


def plot_proportion_heatmaps(
    tables: Dict,
    output_dir: str | Path,
    dpi: int = 300,
) -> Tuple[Path, Path]:
    output_dir = _ensure_dir(output_dir)
    contingency = tables["contingency"]
    state_prop = tables["state_prop"]
    celltype_prop = tables["celltype_prop"]

    # ── cell-type fraction within each meta-state ──
    path1 = output_dir / "03_celltype_fraction_within_each_metastate_heatmap.png"
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * contingency.shape[1] + 3),
                                    max(4.8, 0.45 * contingency.shape[0] + 2)))
    sns.heatmap(
        state_prop, cmap="viridis", vmin=0,
        vmax=min(1.0, max(0.25, state_prop.to_numpy().max())),
        annot=contingency.size <= 120, fmt=".2f",
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "fraction within meta-state"}, ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
    ax.set_xlabel("Real cell type", fontsize=14)
    ax.set_ylabel("Detected meta-state", fontsize=14)
    ax.set_title("")
    _savefig(path1, dpi)

    # ── meta-state fraction within each cell type ──
    path2 = output_dir / "04_metastate_fraction_within_each_celltype_heatmap.png"
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * contingency.shape[1] + 3),
                                    max(4.8, 0.45 * contingency.shape[0] + 2)))
    sns.heatmap(
        celltype_prop, cmap="rocket_r", vmin=0,
        vmax=min(1.0, max(0.25, celltype_prop.to_numpy().max())),
        annot=contingency.size <= 120, fmt=".2f",
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "fraction within cell type"}, ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
    ax.set_xlabel("Real cell type", fontsize=14)
    ax.set_ylabel("Detected meta-state", fontsize=14)
    ax.set_title("")
    _savefig(path2, dpi)

    return path1, path2


def plot_enrichment_heatmap(
    tables: Dict,
    output_dir: str | Path,
    dpi: int = 300,
) -> Path:
    output_dir = _ensure_dir(output_dir)
    path = output_dir / "05_log2_observed_expected_enrichment_heatmap.png"

    contingency = tables["contingency"]
    enrichment_df = tables["enrichment"]

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

    vmax = max(np.nanpercentile(np.abs(mat.to_numpy()), 97), 1.0)

    fig, ax = plt.subplots(figsize=(max(7, 0.58 * mat.shape[1] + 3),
                                    max(4.8, 0.45 * mat.shape[0] + 2)))
    sns.heatmap(
        mat, cmap="coolwarm", center=0, vmin=-vmax, vmax=vmax,
        annot=annot if mat.size <= 150 else False, fmt="",
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "log2 observed / expected"}, ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
    ax.set_xlabel("Real cell type", fontsize=14)
    ax.set_ylabel("Detected meta-state", fontsize=14)
    ax.set_title("")
    _savefig(path, dpi)
    return path


def plot_enrichment_dotplot(
    tables: Dict,
    output_dir: str | Path,
    dpi: int = 300,
    q_threshold: float = 0.05,
) -> Path:
    output_dir = _ensure_dir(output_dir)
    path = output_dir / "06_fisher_enrichment_dotplot.png"

    contingency = tables["contingency"]
    df = tables["enrichment"].copy()

    df["meta_state"] = pd.Categorical(df["meta_state"],
                                      categories=contingency.index.tolist(), ordered=True)
    df["cell_type"] = pd.Categorical(df["cell_type"],
                                     categories=contingency.columns.tolist(), ordered=True)
    df["dot_size"] = np.clip(df["minus_log10_q"], 0, 20)
    df["significant"] = (df["q_value"] < q_threshold) & (df["log2_enrichment"] > 0)

    vmax = max(np.nanpercentile(np.abs(df["log2_enrichment"]), 97), 1.0)

    fig, ax = plt.subplots(figsize=(max(8, 0.65 * contingency.shape[1] + 3),
                                    max(5.2, 0.55 * contingency.shape[0] + 2)))
    sc = ax.scatter(
        x=df["cell_type"].cat.codes,
        y=df["meta_state"].cat.codes,
        s=20 + 18 * df["dot_size"],
        c=df["log2_enrichment"],
        cmap="coolwarm", vmin=-vmax, vmax=vmax,
        edgecolor=np.where(df["significant"], "black", "none"),
        linewidth=np.where(df["significant"], 0.7, 0.0),
        alpha=0.85,
    )

    ax.set_xticks(np.arange(len(contingency.columns)))
    ax.set_xticklabels(contingency.columns, rotation=45, ha="right", fontsize=12)
    ax.set_yticks(np.arange(len(contingency.index)))
    ax.set_yticklabels(contingency.index, rotation=0, fontsize=12)
    ax.set_xlabel("Real cell type", fontsize=14)
    ax.set_ylabel("Detected meta-state", fontsize=14)

    # colorbar: shrunk to bottom half so it doesn't collide with dot-size legend
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04,
                        shrink=0.8, anchor=(0.0, 0.0))
    cbar.set_label("log2 observed / expected", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # dot-size legend: top-right, well above the colorbar
    sizes = [1, 2, 5, 10]
    handles = [plt.scatter([], [], s=20 + 18 * s, color="gray", alpha=0.7, label=f"{s}")
               for s in sizes]
    ax.legend(handles=handles, title="-log10 FDR", frameon=False,
              loc="upper left", bbox_to_anchor=(1.0, 1.0))

    sns.despine(ax=ax)
    _savefig(path, dpi)
    return path


def plot_stacked_bars(
    tables: Dict,
    output_dir: str | Path,
    dpi: int = 300,
) -> Tuple[Path, Path]:
    output_dir = _ensure_dir(output_dir)
    contingency = tables["contingency"]
    state_prop = tables["state_prop"]
    celltype_prop = tables["celltype_prop"]

    palette = sns.color_palette("tab20", n_colors=max(len(contingency.columns), 3))
    color_map = dict(zip(contingency.columns, palette))

    # ── cell-type composition by meta-state ──
    path1 = output_dir / "07_stacked_bar_celltype_composition_by_metastate.png"
    # fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(state_prop.index) + 4), 5.2))
    fig, ax = plt.subplots(figsize=(10,5))
    bottom = np.zeros(len(state_prop.index))
    x = np.arange(len(state_prop.index))
    for ct in state_prop.columns:
        vals = state_prop[ct].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=ct, color=color_map[ct], width=0.82)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(state_prop.index, rotation=0, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction within meta-state", fontsize=14)
    ax.set_xlabel("Detected meta-state", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(bbox_to_anchor=(0.98, 1), loc="upper left", frameon=False, fontsize=10)
    sns.despine(ax=ax)
    _savefig(path1, dpi)

    # ── meta-state composition by cell type ──
    palette2 = sns.color_palette("tab20", n_colors=max(len(contingency.index), 3))
    state_colors = dict(zip(contingency.index, palette2))

    path2 = output_dir / "08_stacked_bar_metastate_composition_by_celltype.png"
    # fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(celltype_prop.columns) + 4), 5.2))
    fig, ax = plt.subplots(figsize=(10,5))
    bottom = np.zeros(len(celltype_prop.columns))
    x = np.arange(len(celltype_prop.columns))
    for state in celltype_prop.index:
        vals = celltype_prop.loc[state].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=state, color=state_colors[state], width=0.82)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(celltype_prop.columns, fontsize=12) #rotation=45, ha="right", fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction within real cell type", fontsize=14)
    ax.set_xlabel("Real cell type", fontsize=14)
    ax.tick_params(labelsize=14)
    ax.legend(bbox_to_anchor=(0.98, 1), loc="upper left", frameon=False, fontsize=10)
    sns.despine(ax=ax)
    _savefig(path2, dpi)

    return path1, path2


def _ribbon(ax, x0, x1, y0a, y0b, y1a, y1b, color, alpha=0.55):
    dx = x1 - x0
    verts = [
        (x0, y0a), (x0 + dx * 0.45, y0a), (x1 - dx * 0.45, y1a), (x1, y1a),
        (x1, y1b), (x1 - dx * 0.45, y1b), (x0 + dx * 0.45, y0b), (x0, y0b),
        (x0, y0a),
    ]
    codes = [
        MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    patch = PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha)
    ax.add_patch(patch)


# def plot_alluvial_metastate_to_celltype(
#     tables: Dict,
#     output_dir: str | Path,
#     dpi: int = 300,
#     min_flow_fraction: float = 0.005,
# ) -> Path:
#     output_dir = _ensure_dir(output_dir)
#     path = output_dir / "09_alluvial_metastate_to_celltype.png"

#     C = tables["contingency"].astype(float)
#     N = C.to_numpy().sum()
#     if N <= 0:
#         raise ValueError("Empty contingency table.")

#     state_tot = C.sum(axis=1) / N
#     type_tot = C.sum(axis=0) / N

#     gap = 0.012
#     state_y0 = {}
#     y = 0.0
#     for state, frac in state_tot.items():
#         state_y0[state] = y
#         y += frac + gap

#     type_y0 = {}
#     y = 0.0
#     for ct, frac in type_tot.items():
#         type_y0[ct] = y
#         y += frac + gap

#     palette = sns.color_palette("tab20", n_colors=max(len(C.columns), 3))
#     ct_colors = dict(zip(C.columns, palette))

#     fig, ax = plt.subplots(figsize=(9.5, max(5.5, 0.35 * (len(C.index) + len(C.columns)))))
#     x0, x1 = 0.25, 0.75
#     bar_w = 0.045

#     state_offset = {s: state_y0[s] for s in C.index}
#     type_offset = {ct: type_y0[ct] for ct in C.columns}

#     for state in C.index:
#         for ct in C.columns:
#             frac = C.loc[state, ct] / N
#             if frac < min_flow_fraction:
#                 continue
#             y0a = state_offset[state]
#             y0b = y0a + frac
#             y1a = type_offset[ct]
#             y1b = y1a + frac
#             _ribbon(ax, x0 + bar_w / 2, x1 - bar_w / 2, y0a, y0b, y1a, y1b,
#                     color=ct_colors[ct], alpha=0.45)
#             state_offset[state] = y0b
#             type_offset[ct] = y1b

#     for state, frac in state_tot.items():
#         y0 = state_y0[state]
#         ax.add_patch(plt.Rectangle((x0 - bar_w / 2, y0), bar_w, frac,
#                                    facecolor="gray", edgecolor="black", lw=0.4))
#         ax.text(x0 - 0.06, y0 + frac / 2, str(state), va="center", ha="right", fontsize=12)

#     for ct, frac in type_tot.items():
#         y0 = type_y0[ct]
#         ax.add_patch(plt.Rectangle((x1 - bar_w / 2, y0), bar_w, frac,
#                                    facecolor=ct_colors[ct], edgecolor="black", lw=0.4))
#         ax.text(x1 + 0.06, y0 + frac / 2, str(ct), va="center", ha="left", fontsize=12)

#     ymax = max(max(state_y0.values()) + state_tot.iloc[-1] + 0.12,
#                max(type_y0.values()) + type_tot.iloc[-1] + 0.12)
#     ax.text(x0, ymax - 0.02, "Detected\nmeta-states", ha="center", va="top", fontsize=11)
#     ax.text(x1, ymax - 0.02, "Real\ncell types", ha="center", va="top", fontsize=11)
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, ymax)
#     ax.axis("off")
#     plt.savefig(path, dpi=dpi, bbox_inches="tight")

#     return path


def plot_alluvial_metastate_to_celltype(
    tables: Dict,
    output_dir: str | Path,
    dpi: int = 300,
    min_flow_fraction: float = 0.005,
) -> Path:
    output_dir = _ensure_dir(output_dir)
    path = output_dir / "09_alluvial_metastate_to_celltype.png"

    C = tables["contingency"].astype(float)
    N = C.to_numpy().sum()
    if N <= 0:
        raise ValueError("Empty contingency table.")

    state_tot = C.sum(axis=1) / N
    type_tot = C.sum(axis=0) / N

    gap = 0.012
    state_y0 = {}
    y = 0.0
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

    # fig, ax = plt.subplots(figsize=(9.5, max(5.5, 0.35 * (len(C.index) + len(C.columns)))))
    fig, ax = plt.subplots(figsize=(7,6))

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
            _ribbon(ax, x0 + bar_w / 2, x1 - bar_w / 2, y0a, y0b, y1a, y1b,
                    color=ct_colors[ct], alpha=0.45)
            state_offset[state] = y0b
            type_offset[ct] = y1b

    # Measure how wide the text labels are (approximate) to set tight xlim
    # Find longest state label and longest cell-type label
    max_state_len = max(len(str(s)) for s in state_tot.index)
    max_ct_len = max(len(str(ct)) for ct in type_tot.index)

    # Approximate: each character ~ 0.012 in data coords at this scale
    left_margin = 0.012 * max_state_len + 0.08
    right_margin = 0.012 * max_ct_len + 0.08

    for state, frac in state_tot.items():
        y0 = state_y0[state]
        ax.add_patch(plt.Rectangle((x0 - bar_w / 2, y0), bar_w, frac,
                                   facecolor="gray", edgecolor="black", lw=0.4))
        ax.text(x0 - 0.06, y0 + frac / 2, str(state), va="center", ha="right", fontsize=12)

    for ct, frac in type_tot.items():
        y0 = type_y0[ct]
        ax.add_patch(plt.Rectangle((x1 - bar_w / 2, y0), bar_w, frac,
                                   facecolor=ct_colors[ct], edgecolor="black", lw=0.4))
        ax.text(x1 + 0.06, y0 + frac / 2, str(ct), va="center", ha="left", fontsize=12)

    ymax = max(max(state_y0.values()) + state_tot.iloc[-1],
               max(type_y0.values()) + type_tot.iloc[-1])
    ax.text(x0, -0.05, "Detected\nmeta-states", ha="center", va="top", fontsize=14)
    ax.text(x1, -0.05, "Real\ncell types", ha="center", va="top", fontsize=14)

    # Tight x-limits instead of (0, 1)
    ax.set_xlim(x0 - left_margin, x1 + right_margin)
    ax.set_ylim(0, ymax)
    ax.axis("off")

    # Remove all padding around the axes
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    return path

# ─────────────────────────────────────────────────────────────────────
# Convenience: replot everything at once
# ─────────────────────────────────────────────────────────────────────

def replot_all(output_dir: str | Path, dpi: int = 300) -> Dict[str, Path]:
    """Load analysis results and regenerate every figure."""
    tables = load_analysis(output_dir)
    output_dir = Path(output_dir)

    paths = {}
    paths["umap"] = plot_umap_metastate_vs_celltype(tables, output_dir, dpi=dpi)
    paths["counts_heatmap"] = plot_count_heatmap(tables, output_dir, dpi=dpi)
    p1, p2 = plot_proportion_heatmaps(tables, output_dir, dpi=dpi)
    paths["state_prop_heatmap"] = p1
    paths["celltype_prop_heatmap"] = p2
    paths["enrichment_heatmap"] = plot_enrichment_heatmap(tables, output_dir, dpi=dpi)
    paths["enrichment_dotplot"] = plot_enrichment_dotplot(tables, output_dir, dpi=dpi)
    p1, p2 = plot_stacked_bars(tables, output_dir, dpi=dpi)
    paths["stacked_bar_ct_by_state"] = p1
    paths["stacked_bar_state_by_ct"] = p2
    paths["alluvial"] = plot_alluvial_metastate_to_celltype(tables, output_dir, dpi=dpi)

    print(f"\n[plots] All figures regenerated in {output_dir}/")
    for name, p in paths.items():
        print(f"  {name}: {p.name}")
    return paths


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    import scanpy as sc
    import sys
    from pathlib import Path

    PROJECT_DIR = Path("/home/user/Documents/velot_agusti/notebooks/agusti")
    sys.path.insert(0, str(PROJECT_DIR))

    # replot_all("metastate_celltype_comparison_pancreas", dpi=300)

    tables = load_analysis("metastate_celltype_comparison_pancreas")
    plot_alluvial_metastate_to_celltype(tables, "metastate_celltype_comparison_pancreas", dpi=300)
"""
Visual and quantitative comparison of OT variants (and the kNN gradient)
on a synthetic bifurcation with a known velocity.

Run:  conda activate velot_test && python ot_variations.py
Outputs in results/OT_bifurcation/:
    data_overview.png             data + true velocity
    <method>.png                  stream / quiver, raw and smoothed
    metrics_vs_truth.csv          cosine to the true velocity per method
    agreement_raw.csv / _smooth   mean cosine between every pair of fields
    metrics.png                   per-cell distributions + agreement heatmaps

All metrics are computed in PCA space (the space the velocity lives in).
UMAP is only for display.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import scanpy as sc
import velot

import matplotlib.pyplot as plt

sc.settings.verbosity = 0

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "OT_tree", "cluster_agnostic")
os.makedirs(OUT, exist_ok=True)

# =====================================================================
# Ground truth
# =====================================================================
def true_velocity(adata, slopes=(2.0, -2.0)):
    """Analytic unit velocity for velot.datasets.synthetic_bifurcation,
    expressed in the SAME basis as adata.obsm['X_pca'].

    In feature space the true motion is (1, slope, 0, 0, ...): the extra
    dimensions are pure noise and carry no dynamics. When extra
    dimensions are present the dataset builder runs a real PCA, which
    rotates the axes, so the truth must be pushed through the loadings
    rather than compared to the first two PCs.
    """
    lab = adata.obs["celltype"].astype(str).values
    n_feat = adata.shape[1]
    Vf = np.zeros((adata.n_obs, n_feat))
    Vf[lab == "Root", 0] = 1.0
    for i, s in enumerate(slopes):
        m = lab == f"Branch_{i+1}"
        Vf[m, 0] = 1.0
        Vf[m, 1] = s

    if "PCs" in adata.varm:                      # extra_dimensions > 0
        V = Vf @ np.asarray(adata.varm["PCs"])   # (n_obs, n_comps)
    else:                                        # X_pca is X itself
        V = Vf[:, :adata.obsm["X_pca"].shape[1]]

    nrm = np.linalg.norm(V, axis=1, keepdims=True)
    return V / np.clip(nrm, 1e-12, None)

def create_data_synthetic_bifurcation():
    adata = velot.datasets.synthetic_bifurcation(
        root_density=400, branch_densities=(300, 300), branch_positions=(2, 2),
        branch_slopes=(2, -2), noise_level=0.05, extra_dimensions=0, n_neighbors=30, seed=0
    )
    adata.obs["clusters_id"] = adata.obs["celltype"].cat.codes
    adata.obs["pseudotime"] = adata.obs["true_pseudotime"].values

    # Stored under its own key so velocity() never overwrites it
    adata.obsm["true_velocity_pca"] = true_velocity(adata, (2, -2))
    velot.tl.project_to_umap(adata, "true_velocity_pca", "true_velocity_umap",
                             basis_umap="X_umap")
    return adata

def create_data_synthetic_tree():
    adata = velot.datasets.synthetic_tree(
        topology=[
            {"name": "Root", "parent": None, "n_cells": 400, "length": 1.0, "slope": 0.0},
            {"name": "Branch_1", "parent": "Root", "n_cells": 200, "length": 1, "slope": 2.0},
            {"name": "Branch_2", "parent": "Root", "n_cells": 300, "length": 1, "slope": -2.0},
            {"name": "Branch_3", "parent": "Branch_2", "n_cells": 150, "length": 0.5, "slope": 1},
        ],
        noise_level=0.05, extra_dimensions=0, n_neighbors=30, seed=0
    )
    adata.obs["clusters_id"] = adata.obs["celltype"].cat.codes
    adata.obs["pseudotime"] = adata.obs["true_pseudotime"].values

    # Stored under its own key so velocity() never overwrites it
    adata.obsm["true_velocity_pca"] = true_velocity(adata, (2, -2, 1))
    velot.tl.project_to_umap(adata, "true_velocity_pca", f"true_velocity_umap", basis_umap="X_umap")
    
    return adata

# =====================================================================
# Methods - identical windows and smoother for all of them
# =====================================================================
# spatial_key="clusters_id" builds windows INSIDE each cell type (Root,
# Branch_1, Branch_2), so no OT pair crosses the bifurcation and
# n_clusters is ignored. Set SPATIAL_KEY = None to build windows along
# pseudotime over all cells (n_clusters=1): windows past the branch point
# then mix both branches, which is where OT should differ most from the
# window mean.
SPATIAL_KEY = None

SHARED = dict(
    basis="X_pca",
    smooth=True,
    n_clusters=1,
    spatial_key=SPATIAL_KEY,
    window_size=50,
    min_window_size=20,
    overlap_fraction=0.0,
    tail_handling="drop",
    tail_threshold=20,
    n_epochs=100,
    lambda_smooth=0.1,
    lambda_curl=0.1,
    lambda_divergence=0.0,
    k_smooth=30,
    project_umap=True,
    verbose=True,
)

METHODS = {
    "original":     dict(reg=0.1),                                        # package
    "no_knn":       dict(reg=0.1, use_graph=False),                       # no kNN graph (no penalty, no zeroing)
    "cost_nn":      dict(reg=0.1, use_graph=False, cost_scale="nn"),      # eps in spacing units
    "argmax":       dict(reg=0.1, use_graph=False, ot_assignment="argmax"),
    "exact_OT":     dict(ot_solver="emd", use_graph=False),               # exact OT
    # reg_m is in units of the squared nearest-neighbour spacing of the
    # source window: mass is kept where a target is within ~sqrt(reg_m)
    # spacings and dropped where the nearest target is much further, which
    # is what a terminal state looks like. compute_ot_velocity prints the
    # typical move per pair in the same units, to pick the value.
    "unbalanced":   dict(reg=0.1, use_graph=False, cost_scale="nn",
                         unbalanced=True, reg_m=100.0),
    "geodesic":     dict(reg=0.1, use_graph=False, cost_scale="nn",       # distances along the kNN graph
                         cost_metric="geodesic", unbalanced=True, reg_m=100.0),
    # The two candidate fixes for the terminal-state failure that the
    # balanced variants show with cluster-agnostic windows. They differ in
    # WHERE they act: normalize_raw leaves the (wrong) assignments alone and
    # only stops long vectors from dominating the smoother's L2 loss;
    # unbalanced changes the assignments themselves, letting a terminal cell
    # keep its mass instead of being forced onto a far branch.
    "normraw":      dict(ot_solver="emd", use_graph=False, normalize_raw=True),
    "unbalanced_auto": dict(reg=0.1, use_graph=False, cost_scale="nn",
                            unbalanced=True, reg_m="auto"),
    "knn_gradient": dict(method="gradient", gradient_mode="knn", gradient_k=30),
}

# =====================================================================
# Plots
# =====================================================================
def plot_data_overview(adata, save=None, show=False):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    velot.pl.dataset_overview_simple(adata, "celltype", "pca", ax=axes[0])

    velot.pl.dataset_overview_simple(adata, "pseudotime", "pca", ax=axes[1])

    velot.pl.velocity_stream(adata, color="celltype",
        title="True velocity (PCA)",
        figsize=(5,5), show=False, basis="pca", velocity_key="true_velocity", ax=axes[2]
    )
    velot.pl.velocity_stream(adata, color="celltype",
        title="True velocity (UMAP)",
        figsize=(5,5), show=False, basis="umap", velocity_key="true_velocity", ax=axes[3]
    )

    plt.tight_layout()
    _finish(fig, save, show)

def plot_stream_quiver(adata, name, save=None, show=False):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    panels = [("raw", "PCA"), ("raw", "UMAP"), ("smooth", "PCA"), ("smooth", "UMAP")]
    for k, (field, space) in enumerate(panels):
        key = "velot_velocity_raw" if field == "raw" else "velot_velocity"
        basis = space.lower()
        velot.pl.velocity_stream(adata, color="celltype",
            title=f"{field} {space} stream",
            figsize=(5,5), show=False, basis=basis, velocity_key=key, ax=axes[k]
        )
        velot.pl.velocity_quiver(adata,
            color="celltype", basis=basis, subsample=500,
            velocity_key=f"{key}_{basis}",
            title=f"{field} {space} vectors",
            show=False, ax=axes[k + 4]
        )

    fig.suptitle(name, fontsize=20)
    plt.tight_layout()
    _finish(fig, save, show)

def _finish(fig, save, show):
    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.savefig(save, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

# =====================================================================
# Metrics
# =====================================================================
def cosine_rows(A, B):
    """Per-cell cosine; NaN where either vector is zero (no estimate)."""
    na, nb = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
    out = np.full(len(A), np.nan)
    ok = (na > 0) & (nb > 0)
    out[ok] = (A[ok] * B[ok]).sum(1) / (na[ok] * nb[ok])
    return out

def metrics_vs_truth(fields, truth, celltype):
    """One row per (raw/smooth, method).

    coverage   fraction of cells with a non-zero vector (raw OT leaves the
               last window of each cluster empty)
    cos_*      cosine to the true velocity, over covered cells
    frac_cos>0 fraction pointing in the right half-space
    """
    rows = []
    for (name, field), V in fields.items():
        c = cosine_rows(V, truth)
        ok = ~np.isnan(c)
        row = dict(field=field, method=name, coverage=ok.mean(),
                   cos_mean=np.nanmean(c), cos_median=np.nanmedian(c),
                   **{"frac_cos>0": np.mean(c[ok] > 0)})
        for ct in np.unique(celltype):
            row[f"cos_{ct}"] = np.nanmean(c[celltype == ct])
        rows.append(row)
    return pd.DataFrame(rows).set_index(["field", "method"])

def agreement(fields, truth, field):
    """Mean per-cell cosine between every pair of methods (and the truth)
    for one field ("raw" or "smooth")."""
    F = {"truth": truth, **{n: V for (n, f), V in fields.items() if f == field}}
    M = pd.DataFrame({a: {b: np.nanmean(cosine_rows(F[a], F[b])) for b in F} for a in F})
    return M.loc[list(F), list(F)]

def plot_metrics(fields, truth, save=None, show=False):
    names = list(dict.fromkeys(n for n, _ in fields))
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for r, field in enumerate(("raw", "smooth")):
        # per-cell cosine to truth
        ax = axes[r, 0]
        data = [cosine_rows(fields[(n, field)], truth) for n in names]
        ax.boxplot([d[~np.isnan(d)] for d in data], showfliers=False)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_ylim(top=1.02)
        ax.set_ylabel("cosine to true velocity")
        ax.set_title(f"{field}: per-cell cosine to truth")

        # method x method agreement (truth included)
        ax = axes[r, 1]
        M = agreement(fields, truth, field)
        lo = np.nanmin(M.values)
        im = ax.imshow(M.values, cmap="viridis", vmin=lo, vmax=1)
        ax.set_xticks(range(len(M)))
        ax.set_xticklabels(M.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(M)))
        ax.set_yticklabels(M.index)
        for i in range(len(M)):
            for j in range(len(M)):
                v = M.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v < (lo + 1) / 2 else "black")
        ax.set_title(f"{field}: mean cosine between fields")
        fig.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    _finish(fig, save, show)


# =====================================================================
def main():
    adata = create_data_synthetic_tree()
    plot_data_overview(adata, os.path.join(OUT, "data_overview.png"))

    fields = {}
    for name, kwargs in METHODS.items():
        ad = adata.copy()
        velot.tl.velocity(ad, **kwargs, **SHARED)
        plot_stream_quiver(ad, name, os.path.join(OUT, f"{name}.png"))
        fields[(name, "raw")] = ad.obsm["velot_velocity_raw_pca"]
        fields[(name, "smooth")] = ad.obsm["velot_velocity_pca"]

    truth = adata.obsm["true_velocity_pca"]
    celltype = adata.obs["celltype"].astype(str).values

    table = metrics_vs_truth(fields, truth, celltype)
    table.to_csv(os.path.join(OUT, "metrics_vs_truth.csv"))
    for field in ("raw", "smooth"):
        agreement(fields, truth, field).to_csv(os.path.join(OUT, f"agreement_{field}.csv"))
    plot_metrics(fields, truth, os.path.join(OUT, "metrics.png"))

    pd.set_option("display.width", 160)
    print("\nCosine to the true velocity (PCA space)")
    print(table.round(3).to_string())
    print(f"\nFigures and tables in {OUT}")

if __name__ == "__main__":
    main()

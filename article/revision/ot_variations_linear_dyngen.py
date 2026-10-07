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
OUT = os.path.join(HERE, "results", "OT_linear_dyngen")
os.makedirs(OUT, exist_ok=True)

def create_data_linear():

    adata = sc.read_h5ad("article/data/Synthetic/synthetic_linear.h5ad")

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    sc.pp.pca(adata, n_comps=5)
    sc.pp.neighbors(adata, n_neighbors=20)
    sc.tl.umap(adata)

    milestones = adata.uns['traj_progressions']['from'].values + '->' + adata.uns['traj_progressions']['to'].values
    for i in range(len(milestones)):
        
        state = milestones[i]
        if state == 'sA->sB':
            milestones[i] = 'A'
        
        elif state == 'sB->sC':
            milestones[i] = 'B'
        
        elif state == 'sC->sEndC':
            milestones[i] = 'C'
    adata.obs['milestone'] = milestones

    # velot.pp.pseudotime(adata, root_cluster="A", cluster_key="milestone")
    velot.pp.pseudotime(adata, key="sim_time")
    adata.obs["milestone"] = adata.obs["milestone"].astype("category")
    adata.obs["clusters_id"] = adata.obs["milestone"].cat.codes

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

WINDOWS_DICT = dict(
    basis="X_pca",
    n_clusters=1,
    spatial_key=SPATIAL_KEY,
    window_size=500,
    min_window_size=50,
    overlap_fraction=0.0,
    tail_handling="drop",
    tail_threshold=20
)

SHARED = dict(
    smooth=True,
    n_epochs=100,
    lambda_smooth=0.1,
    lambda_curl=0.1,
    lambda_divergence=0.0,
    k_smooth=30,
    project_umap=True,
    verbose=True,
    **WINDOWS_DICT
)

METHODS = {
    "velot":          dict(use_graph=False, ot_solver="emd"),
    "velot_gradient": dict(method="gradient", gradient_mode="knn",
                           gradient_k=30),
    "velot_no_knn":     dict(use_graph=False),
    # "velot_cost_nn":    dict(use_graph=False, cost_scale="nn"),
    # "velot_argmax":     dict(use_graph=False, cost_scale="nn",
    #                          ot_assignment="argmax"),
    # "velot_unbalanced": dict(use_graph=False, cost_scale="nn",
    #                          unbalanced=True, reg_m="auto"),
    # direction-only: raw vectors are scaled to unit length before the
    # MLP, so cells with long displacements stop dominating the fit
    "velot_normraw":  dict(use_graph=False, ot_solver="emd",
                           normalize_raw=True),
}

# =====================================================================
# Plots
# =====================================================================
def plot_data_overview(adata, save=None, show=False):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    velot.pl.dataset_overview_simple(adata, "milestone", "umap", ax=axes[0])

    velot.pl.dataset_overview_simple(adata, "pseudotime", "umap", ax=axes[1])

    plt.tight_layout()
    _finish(fig, save, show)

def plot_stream_quiver(adata, name, save=None, show=False):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    panels = [("raw", "UMAP"), ("smooth", "UMAP")]
    for k, (field, space) in enumerate(panels):
        key = "velot_velocity_raw" if field == "raw" else "velot_velocity"
        basis = space.lower()
        velot.pl.velocity_stream(adata, color="milestone",
            title=f"{field} {space} stream",
            figsize=(5,5), show=False, basis=basis, velocity_key=key, ax=axes[k]
        )
        velot.pl.velocity_quiver(adata,
            color="milestone", basis=basis, subsample=500,
            velocity_key=f"{key}_{basis}",
            title=f"{field} {space} vectors",
            show=False, ax=axes[k + 2]
        )

    fig.suptitle(name, fontsize=20)
    plt.tight_layout()
    _finish(fig, save, show)

def plot_windows(adata, save=None, show=False):
    n_total_pairs = adata.uns["velot_windows"]["n_pairs"]
    pairs_to_show = tuple([i*1 for i in range(n_total_pairs)])
    os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
    velot.pl.windows(adata, basis="umap",
        pairs_to_show=pairs_to_show[:], ncols=4,
        pair_title=True, figsize_per_panel=(5,5), show=show,
        save=True, save_path=save
    )

def _finish(fig, save, show):
    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.savefig(save, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# =====================================================================
def main():
    adata = create_data_linear()
    plot_data_overview(adata, os.path.join(OUT, "data_overview.png"))

    velot.tl.build_windows(adata, **WINDOWS_DICT)

    plot_windows(adata, save=os.path.join(OUT, f"windows.png"))

    fields = {}
    for name, kwargs in METHODS.items():
        ad = adata.copy()
        velot.tl.velocity(ad, **kwargs, **SHARED)
        plot_stream_quiver(ad, name, os.path.join(OUT, f"{name}.png"))
        fields[(name, "raw")] = ad.obsm["velot_velocity_raw_pca"]
        fields[(name, "smooth")] = ad.obsm["velot_velocity_pca"]

if __name__ == "__main__":
    main()
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
OUT = os.path.join(HERE, "figures2")
os.makedirs(OUT, exist_ok=True)

DATASET = "erythroid"
CLUSTER = "celltype"
BASIS = "umap"
ORDER = [
    "original",
    "no_knn",
    "cost_nn",
    "argmax",
    "exact",
    "unbalanced",
    "geodesic",
    "knn_gradient"
]
edges = [
    ('Blood progenitors 1', 'Blood progenitors 2'), 
    ('Blood progenitors 2', 'Erythroid1'),
    ('Erythroid1', 'Erythroid2'), 
    ('Erythroid2', 'Erythroid3')
]

def _finish(fig, save, show):
    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# =====================================================================
def main():
    # Load all adatas created in the benchmark run
    PATH = os.path.join(HERE, "benchmark_results/real/data")
    adatas = [file for file in os.listdir(PATH) if ("smooth" in file) & (DATASET in file)]
    if ORDER:
        adatas = sorted(
            adatas, 
            key=lambda x: next(i for i, keyword in enumerate(ORDER) if keyword in x)
        )

    n_rows, n_cols = len(adatas), 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*5))

    for i,file_name in enumerate(adatas):
        adata_path = os.path.join(PATH, file_name)
        adata = sc.read_h5ad(adata_path)

        modes = ("raw", "smooth")
        for k,mode in enumerate(modes):
            key = "velot_velocity_raw" if mode == "raw" else "velot_velocity"
            velot.pl.velocity_stream(adata, color=CLUSTER,
                title=f"{mode} stream",
                figsize=(5,5), show=False, basis=BASIS, velocity_key=key, ax=axes[i,k]
            )
            velot.pl.velocity_quiver(adata,
                color=CLUSTER, basis=BASIS, subsample=500,
                velocity_key=f"{key}_{BASIS}",
                title=f"{mode} vectors",
                show=False, ax=axes[i,k+2]
            )

        # --- NEW CODE: Add vertical text to the left of the first column ---
        row_name = file_name.split(f"_smooth")[0].split("velot_")[1]
        axes[i, 0].annotate(
            row_name,                   # The text to display
            xy=(-0.25, 0.5),            # X and Y position (relative to the axes bounds)
            xycoords='axes fraction',   # Use axes proportions for positioning
            rotation=90,                # Vertical rotation
            ha='center',                # Horizontal alignment
            va='center',                # Vertical alignment
            fontsize=20,                # Size of the text
            fontweight='bold',          # Bold text
            annotation_clip=False       # Make sure it isn't cropped outside the plot frame
        )

    fig.suptitle("VelOT stream comparison for different setting of OT and knn gradient", fontsize=22, fontweight="bold", y=0.99)
    # rect=[left, bottom, right, top]. 
    # Setting top to 0.98 leaves the top 2% of the figure empty for the suptitle.
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    _finish(fig, os.path.join(OUT, f"{DATASET}_velocity_benchmark.png"), False)

    adata_path = os.path.join(PATH, adatas[0])
    adata = sc.read_h5ad(adata_path)
    velot.pl.windows(
        adata, basis=BASIS,
        pairs_to_show=None, ncols=4, max_show=20,
        pair_title=True, figsize_per_panel=(5,5), show=False,
        save=True, save_path=os.path.join(OUT, f"{DATASET}_windows.png")
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    velot.pl.dataset_overview_simple(adata, color=CLUSTER,
        title="Cell-type clusters", inframe=True,
        show=False, ax=axes[0]
    )
    velot.pl.dataset_overview_simple(adata, color="pseudotime",
        title="Diffusion pseudotime",
        show=False, ax=axes[1]
    )

    plt.tight_layout()
    _finish(fig, os.path.join(OUT, f"{DATASET}_dataset_overview.png"), False)

    results = velot.metrics.summary(adata,
        cluster_edges=edges, cluster_key=CLUSTER,
        embedding_key=f"X_pca", velocity_key=f"velot_velocity__raw_pca",
        print_results=False
    )

    velot.pl.metric_summary(results,
        orientation="vertical", layout="column",
        figsize=(7,7), show=False, save=True, save_path=os.path.join(OUT, f"{DATASET}_metrics_raw.png")
    )

    results = velot.metrics.summary(adata,
        cluster_edges=edges, cluster_key=CLUSTER,
        embedding_key=f"X_pca", velocity_key=f"velot_velocity_pca",
        print_results=False
    )

    velot.pl.metric_summary(results,
        orientation="vertical", layout="column",
        figsize=(7,7), show=False, save=True, save_path=os.path.join(OUT, f"{DATASET}_metrics_smooth.png")
    )


if __name__ == "__main__":
    main()
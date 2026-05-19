import scanpy as sc
import scvelo as scv
import velot
from velot.benchmark import BenchmarkTimer, save_benchmark

import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)

# ── Config ──────────────────────────────────────────────────────
MODEL_NAME = "velot"
DATASET_NAME = "erythroid"
OUTPUT_DIR = "../benchmark_results/real"

basis = "pca"
project_umap = True if basis == "pca" else False
clusters_key = "celltype"
n_pcs = 10
n_neighs = 20

edges = [
    ('Blood progenitors 1', 'Blood progenitors 2'), 
    ('Blood progenitors 2', 'Erythroid1'),
    ('Erythroid1', 'Erythroid2'), 
    ('Erythroid2', 'Erythroid3')
]

# ── Timer ───────────────────────────────────────────────────────
timer = BenchmarkTimer()

# ── Load ────────────────────────────────────────────────────────
with timer("load"):
    adata = sc.read("/home/user/Documents/velot/article/data/Gastrulation/erythroid_lineage.h5ad")

# ── Preprocess ──────────────────────────────────────────────────
with timer("preprocess"):
    sc.pp.filter_cells(adata, min_counts=20)
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.neighbors(adata, n_neighs, use_rep="X_pca")
    velot.pp.pseudotime(adata, root_cluster="Blood progenitors 1", cluster_key=clusters_key)
    adata.obs["clusters_id"] = adata.obs[clusters_key].cat.codes
    adata.obsm["X_pca"] = adata.obsm["X_pca"][:,:n_pcs]

# ── Velocity ────────────────────────────────────────────────────
with timer("velocity"):
    velot.tl.velocity(
        adata=adata,
        basis=f"X_{basis}",
        smooth=True,
        n_clusters=1,
        window_size=200,
        min_window_size=20,
        overlap_fraction=0,
        # spatial_key="clusters_id",
        tail_handling="drop", tail_threshold=20,
        reg=0.2, lambda_time=1, n_epochs=100, lambda_smooth=0.8, lambda_curl=0.8, lambda_divergence=0, k_smooth=30,
        project_umap=project_umap
    )

# ── Evaluate ────────────────────────────────────────────────────
with timer("evaluate"):
    results = velot.metrics.summary(
        adata,
        cluster_edges=edges,
        cluster_key=clusters_key,
        embedding_key=f"X_{basis}",
        velocity_key=f"velot_velocity_{basis}"
    )


# ── Save ────────────────────────────────────────────────────────
print(timer)

save_benchmark(
    results=results,
    timer=timer,
    model_name=MODEL_NAME,
    dataset_name=DATASET_NAME,
    output_dir=OUTPUT_DIR,
    extra_info={
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "n_pcs": n_pcs,
        "n_neighbors": n_neighs
    }
)
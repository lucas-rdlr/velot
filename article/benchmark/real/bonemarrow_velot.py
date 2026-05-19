import scanpy as sc
import scvelo as scv
import velot
from velot.benchmark import BenchmarkTimer, save_benchmark

import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)

# ── Config ──────────────────────────────────────────────────────
MODEL_NAME = "velot"
DATASET_NAME = "murine"
OUTPUT_DIR = "../benchmark_results/real"

basis = "pca"
project_umap = True if basis == "pca" else False
clusters_key = "cell_type"
n_pcs = 20
n_neighs = 20

edges = [
    ('Stem cells', 'TA cells'),
    ('Stem cells', 'Goblet cells'), 
    ('Goblet cells', 'Paneth cells')
]

# ── Timer ───────────────────────────────────────────────────────
timer = BenchmarkTimer()

# ── Load ────────────────────────────────────────────────────────
with timer("load"):
    adata = sc.read("/home/user/Documents/velot/article/data/Murine/preprocessed.h5ad")

# ── Preprocess ──────────────────────────────────────────────────
with timer("preprocess"):
    # sc.pp.neighbors(adata, n_neighs, use_rep="X_pca")
    velot.pp.pseudotime(adata, root_cluster="Stem cells", cluster_key=clusters_key)
    adata.obs["clusters_id"] = adata.obs[clusters_key].cat.codes
    adata.obsm["X_pca"] = adata.obsm["X_pca"][:,:n_pcs]

# ── Velocity ────────────────────────────────────────────────────
with timer("velocity"):
    velot.tl.velocity(
        adata=adata,
        basis=f"X_{basis}",
        smooth=True,
        n_clusters=None,
        window_size=200,
        min_window_size=20,
        overlap_fraction=0,
        # spatial_key="clusters_id",
        tail_handling="drop", tail_threshold=20,
        reg=0.05, lambda_time=1, n_epochs=100, lambda_smooth=0.2, lambda_curl=0.2, lambda_divergence=0, k_smooth=15,
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
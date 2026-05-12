import scanpy as sc
import scvelo as scv
import velot
from velot.benchmark import BenchmarkTimer, save_benchmark

import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)

# ── Config ──────────────────────────────────────────────────────
MODEL_NAME = "velot"
DATASET_NAME = "pancreas"
OUTPUT_DIR = "benchmark_results"

basis = "pca"
project_umap = True if basis == "pca" else False
clusters_key = "clusters"
n_pcs = 10
n_neighs = 20

edges = [
    ("Ngn3 low EP", "Ngn3 high EP"),
    ("Ngn3 high EP", "Fev+"),
    ("Fev+", "Delta"),
    ("Fev+", "Beta"),
    ("Fev+", "Epsilon"),
    ("Fev+", "Alpha"),
]

# ── Timer ───────────────────────────────────────────────────────
timer = BenchmarkTimer()

# ── Load ────────────────────────────────────────────────────────
with timer("load"):
    adata = sc.read(
        "/home/user/Documents/velot/article/datasets/"
        "endocrinogenesis_day15.5_preprocessed.h5ad"
    )

# ── Preprocess ──────────────────────────────────────────────────
with timer("preprocess"):
    velot.pp.pca(adata, n_pcs=n_pcs)
    sc.pp.neighbors(adata, n_neighs, use_rep="X_pca")
    velot.pp.pseudotime(adata, root_cluster="Ngn3 low EP", cluster_key=clusters_key)
    adata.obs["clusters_id"] = adata.obs[clusters_key].cat.codes

# ── Velocity ────────────────────────────────────────────────────
with timer("velocity"):
    velot.tl.velocity(
        adata=adata,
        basis=f"X_{basis}",
        smooth=True,
        n_clusters=None,
        window_size=50,
        overlap_fraction=0,
        spatial_key="clusters_id",
        reg=0.1, lambda_time=1, n_epochs=150, lambda_smooth=0.7, lambda_curl=0.7, lambda_divergence=0,
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
import scanpy as sc
import scvelo as scv
import velot
from velot.benchmark import BenchmarkTimer, save_benchmark

import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)

# ── Config ──────────────────────────────────────────────────────
MODEL_NAME = "velot"
DATASET_NAME = "linear"
OUTPUT_DIR = "../benchmark_results/synthetic"

basis = "pca"
project_umap = True if basis == "pca" else False
clusters_key = "milestone"
n_pcs = 30
n_neighs = 30

edges = [
    ('A', 'B'),
    ('B', 'C')
]

# ── Timer ───────────────────────────────────────────────────────
timer = BenchmarkTimer()

# ── Load ────────────────────────────────────────────────────────
with timer("load"):
    adata = sc.read("/home/user/Documents/velot/article/data/Synthetic/synthetic_linear.h5ad")

# ── Preprocess ──────────────────────────────────────────────────
with timer("preprocess"):
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

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    velot.pp.pca(adata, n_pcs=n_pcs)
    sc.pp.neighbors(adata, n_neighs)
    velot.pp.pseudotime(adata, root_cluster="A", cluster_key=clusters_key)
    adata.obs["milestone"] = adata.obs["milestone"].astype("category")
    adata.obs["clusters_id"] = adata.obs["milestone"].cat.codes

# ── Velocity ────────────────────────────────────────────────────
with timer("velocity"):
    velot.tl.velocity(
        adata=adata,
        basis=f"X_{basis}",
        smooth=True,
        n_clusters=None,
        window_size=300,
        overlap_fraction=0,
        tail_handling="drop", tail_threshold=10,
        spatial_key="clusters_id",
        reg=0.1, lambda_time=1, n_epochs=150, lambda_smooth=0.5, lambda_curl=0.5, lambda_divergence=0,
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
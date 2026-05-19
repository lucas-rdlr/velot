import deepvelo
import scanpy as sc
import scvelo as scv
import velot
from velot.benchmark import BenchmarkTimer, save_benchmark

import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)

# ── Config ──────────────────────────────────────────────────────
MODEL_NAME = "deepvelo"
DATASET_NAME = "bifurcation"
OUTPUT_DIR = "../benchmark_results/synthetic"

basis = "hvg"
clusters_key = "milestone"
n_pcs = None
n_neighs = 30

edges = [
    ('A', 'B'),
    ('A', 'C'),
    ('B', 'D'),
    ('C', 'E')
]

# ── Timer ───────────────────────────────────────────────────────
timer = BenchmarkTimer()

# ── Load ────────────────────────────────────────────────────────
with timer("load"):
    adata = sc.read("/home/user/Documents/velot/article/data/Synthetic/synthetic_bifurcation.h5ad")

# ── Preprocess ──────────────────────────────────────────────────
with timer("preprocess"):
    milestones = adata.uns['traj_progressions']['from'].values + '->' + adata.uns['traj_progressions']['to'].values
    for i in range(len(milestones)):
        
        state = milestones[i]
        if state == 'sA->sB' or state == 'sB->sBmid':
            milestones[i] = 'A'
        
        elif state == 'sBmid->sC':
            milestones[i] = 'B'
        
        elif state == 'sBmid->sD':
            milestones[i] = 'C'
        
        elif state == 'sC->sEndC':
            milestones[i] = 'D'
        elif state == 'sD->sEndD': 
            milestones[i] = 'E'
    adata.obs['milestone'] = milestones

    adata.layers["spliced"] = adata.layers["counts_spliced"]
    adata.layers["unspliced"] = adata.layers["counts_unspliced"]

    scv.pp.filter_and_normalize(adata, min_shared_counts=20)
    scv.pp.moments(adata, n_neighbors=n_neighs, n_pcs=n_neighs)

# ── Velocity ────────────────────────────────────────────────────
with timer("velocity"):
    trainer = deepvelo.train(adata, deepvelo.Constants.default_configs)
    adata.obsm["velocity_hvg"] = adata.layers["velocity"].copy()
    adata.obsm["X_hvg"] = adata.X.copy()

# ── Evaluate ────────────────────────────────────────────────────
with timer("evaluate"):
    results = velot.metrics.summary(
        adata,
        cluster_edges=edges,
        cluster_key=clusters_key,
        embedding_key=f"X_{basis}",
        velocity_key=f"velocity_{basis}"
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
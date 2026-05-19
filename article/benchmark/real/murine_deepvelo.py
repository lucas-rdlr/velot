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
DATASET_NAME = "murine"
OUTPUT_DIR = "../benchmark_results/real"

basis = "hvg"
clusters_key = "cell_type"
n_pcs = None
n_neighs = 30

edges = [
    ('Stem cells', 'TA cells'),
    ('Stem cells', 'Goblet cells'), 
    ('Goblet cells', 'Paneth cells')
]

# ── Timer ───────────────────────────────────────────────────────
timer = BenchmarkTimer()

# ── Load ────────────────────────────────────────────────────────
with timer("load"):
    adata = sc.read("/home/user/Documents/velot/article/data/Murine/raw.h5ad")

# ── Preprocess ──────────────────────────────────────────────────
with timer("preprocess"):
    adata.layers["spliced"] = (
        adata.layers["labeled_spliced"] +
        adata.layers["unlabeled_spliced"]
    )

    adata.layers["unspliced"] = (
        adata.layers["labeled_unspliced"] +
        adata.layers["unlabeled_unspliced"]
    )
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
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
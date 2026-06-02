import scanpy as sc
import scvelo as scv
import velot
from velot.benchmark import BenchmarkTimer, save_benchmark

import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)

# ── Config ──────────────────────────────────────────────────────
MODEL_NAME = "scvelo_stochastic"
DATASET_NAME = "linear"
OUTPUT_DIR = "../benchmark_results/synthetic"

clusters_key = "milestone"
n_pcs = 50
n_neighs = 20

edges = [
    ('A', 'B'),
    ('B', 'C')
]

# ── Timer ───────────────────────────────────────────────────────
timer = BenchmarkTimer()

# ── Load ────────────────────────────────────────────────────────
with timer("load"):
    adata = sc.read("../../data/Synthetic/synthetic_linear_processed.h5ad")

# ── Preprocess ──────────────────────────────────────────────────
with timer("preprocess"):
    adata.layers["spliced"] = adata.layers["counts_spliced"]
    adata.layers["unspliced"] = adata.layers["counts_unspliced"]

    scv.pp.filter_genes(adata, min_shared_counts=20)
    scv.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)

    sc.pp.pca(adata, n_comps=n_pcs)
    sc.pp.neighbors(adata, n_pcs=n_neighs, n_neighbors=n_neighs)
    scv.pp.moments(adata, n_pcs=n_neighs, n_neighbors=n_neighs)

# ── Velocity ────────────────────────────────────────────────────
with timer("velocity"):
    scv.tl.velocity(adata, mode="stochastic", vkey="stocvelo", n_jobs=12)
    scv.tl.velocity_graph(adata, vkey="stocvelo", n_jobs=12)
    scv.tl.velocity_embedding(adata, vkey="stocvelo", basis="pca")

# ── Evaluate ────────────────────────────────────────────────────
with timer("evaluate"):
    results = velot.metrics.summary(
        adata,
        cluster_edges=edges,
        cluster_key=clusters_key,
        embedding_key="X_pca",
        velocity_key="stocvelo_pca",
    )

# ── Save ────────────────────────────────────────────────────────
print(timer)

save_benchmark(
    adata=adata,
    results=results,
    timer=timer,
    model_name=MODEL_NAME,
    dataset_name=DATASET_NAME,
    output_dir=OUTPUT_DIR,
    extra_info={
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "n_pcs": n_pcs,
        "n_neighbors": n_neighs,
        "n_jobs": 12,
    },
)
import scanpy as sc
import scvelo as scv
import velot
from velot.benchmark import BenchmarkTimer, save_benchmark

from scvelo.tools import flux_velocity

import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)

# ── Config ──────────────────────────────────────────────────────
MODEL_NAME = "flux_matching"
DATASET_NAME = "linear"
OUTPUT_DIR = "../benchmark_results/synthetic"

basis = "umap"
clusters_key = "milestone"
n_pcs = 50
n_neighs = 30

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

    sc.pp.filter_cells(adata, min_counts=1)
    scv.pp.filter_and_normalize(adata, min_shared_counts=20)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=n_pcs)
    sc.pp.neighbors(adata, n_pcs=n_neighs, n_neighbors=n_neighs)
    scv.pp.moments(adata, n_neighbors=None, n_pcs=None)
    sc.tl.umap(adata)

# ── Velocity ────────────────────────────────────────────────────
with timer("velocity"):
    scv.tl.velocity(adata, mode="dynamical", mask_zero=False)
    flux_velocity(
        adata,
        model_family="dynamical",
        lr=1e-3,
        epochs=100,
    )
    scv.tl.velocity_graph(adata, n_jobs=8)
    scv.tl.velocity_embedding(adata, basis=basis)

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
        "n_neighbors": n_neighs
    }
)
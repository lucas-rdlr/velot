import scanpy as sc
import scvelo as scv
import velot
from velot.benchmark import BenchmarkTimer, save_benchmark

import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)

# ── Config ──────────────────────────────────────────────────────
MODEL_NAME = "scvelo_stochastic"
DATASET_NAME = "pancreas"
OUTPUT_DIR = "benchmark_results"

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
    scv.pp.filter_genes(adata, min_shared_counts=20)
    scv.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var["highly_variable"]].copy()

    sc.pp.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

# ── Velocity ────────────────────────────────────────────────────
with timer("velocity"):
    # scv.tl.recover_dynamics(adata, n_jobs=12)
    scv.tl.velocity(adata, mode="stochastic", vkey="stocvelo", n_jobs=12)
    scv.tl.velocity_graph(adata, vkey="stocvelo", n_jobs=12)
    scv.tl.velocity_embedding(adata, vkey="stocvelo", basis="pca")

# ── Evaluate ────────────────────────────────────────────────────
with timer("evaluate"):
    results = velot.metrics.summary(
        adata,
        cluster_edges=edges,
        cluster_key="clusters",
        embedding_key="X_pca",
        velocity_key="stocvelo_pca",
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
        "n_pcs": 30,
        "n_neighbors": 30,
        "n_jobs": 12,
    },
)
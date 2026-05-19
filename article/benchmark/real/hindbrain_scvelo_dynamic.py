import scanpy as sc
import scvelo as scv
import velot
from velot.benchmark import BenchmarkTimer, save_benchmark

import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)

# ── Config ──────────────────────────────────────────────────────
MODEL_NAME = "scvelo_dynamical"
DATASET_NAME = "hindbrain"
OUTPUT_DIR = "../benchmark_results/real"

clusters_key = "Celltype"
n_pcs = 50
n_neighs = 20

edges = [
    ('Neural stem cells', 'Proliferating VZ progenitors'),
    ('Proliferating VZ progenitors', 'VZ progenitors'),
    ('VZ progenitors', 'Gliogenic progenitors'),
    ('VZ progenitors', 'Differentiating GABA interneurons'),
    ('Differentiating GABA interneurons', 'GABA interneurons')
]

# ── Timer ───────────────────────────────────────────────────────
timer = BenchmarkTimer()

# ── Load ────────────────────────────────────────────────────────
with timer("load"):
    adata = sc.read_h5ad("/home/user/Documents/velot/article/data/HindBrain/Hindbrain_GABA_Glio.h5ad")

# ── Preprocess ──────────────────────────────────────────────────
with timer("preprocess"):
    scv.pp.filter_genes(adata, min_shared_counts=20)
    scv.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)

    sc.pp.pca(adata, n_comps=n_pcs)
    sc.pp.neighbors(adata, n_pcs=n_neighs, n_neighbors=n_neighs)
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)

# ── Velocity ────────────────────────────────────────────────────
with timer("velocity"):
    scv.tl.recover_dynamics(adata, n_jobs=12)
    scv.tl.velocity(adata, mode="dynamical", vkey="dynvelo", n_jobs=12)
    scv.tl.velocity_graph(adata, vkey="dynvelo", n_jobs=12)
    scv.tl.velocity_embedding(adata, vkey="dynvelo", basis="pca")

# ── Evaluate ────────────────────────────────────────────────────
with timer("evaluate"):
    results = velot.metrics.summary(
        adata,
        cluster_edges=edges,
        cluster_key=clusters_key,
        embedding_key="X_pca",
        velocity_key="dynvelo_pca",
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
        "n_neighbors": n_neighs,
        "n_jobs": 12
    }
)
import scanpy as sc
import velot
from velot.benchmark import BenchmarkTimer, save_benchmark

import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)

# ── Config ──────────────────────────────────────────────────────
MODEL_NAME = "velot"
DATASET_NAME = "hindbrain"
OUTPUT_DIR = "../benchmark_results/real"

basis = "pca"
project_umap = True if basis == "pca" else False
clusters_key = "Celltype"
n_pcs = 10
n_neighs = 50

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
    adata = sc.read_h5ad("../../data/HindBrain/Hindbrain_GABA_Glio.h5ad")

# ── Preprocess ──────────────────────────────────────────────────
with timer("preprocess"):
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    velot.pp.pca(adata, n_pcs=n_pcs)
    sc.pp.neighbors(adata, n_neighs)
    velot.pp.pseudotime(adata, root_cluster="Neural stem cells", cluster_key=clusters_key)
    adata.obs["clusters_id"] = adata.obs[clusters_key].cat.codes

# ── Velocity ────────────────────────────────────────────────────
with timer("velocity"):
    velot.tl.velocity(
        adata=adata,
        basis=f"X_{basis}",
        smooth=True,
        n_clusters=None,
        window_size=50,
        min_window_size=20,
        overlap_fraction=0,
        # spatial_key="clusters_id",
        tail_handling="drop", tail_threshold=20,
        reg=0.1, lambda_time=1, n_epochs=150, lambda_smooth=0.5, lambda_curl=0.5, lambda_divergence=0,
        project_umap=project_umap, project_basis="X_tsne"
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
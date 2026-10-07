import os
from pathlib import Path

import velot
from velot.benchmark import benchmark_dotplot
import scanpy as sc
import matplotlib.pyplot as plt

HERE = os.chdir(Path(__file__).resolve().parent)

import string

METHODS = ["velot"]
SEED = "0"
FIELDS = ["raw", "smooth"]
DATASET = "erythroid"
CLUSTERS_KEY = "celltype"
chr_code = ord("a")
letter = chr(chr_code)
show, save = False, True

for method in METHODS:
    for field in FIELDS:
        adata_name = f"{method}_seed{SEED}_smooth_{DATASET}.h5ad"
        adata = sc.read_h5ad(f"results/data/{adata_name}")
        velocity_key = "velot_velocity_raw" if field == "raw" else "velot_velocity"

        if chr_code == ord("a"):
            velot.pl.dataset_overview_simple(adata, color=CLUSTERS_KEY, title=None, figsize=(5,5), inframe=True, show=show, save=save, save_path=f"figure4_{letter}.png")
            chr_code += 1
            letter = chr(chr_code)
            velot.pl.dataset_overview_simple(adata, color="pseudotime", title=None, show=show, save=save, save_path=f"figure4_{letter}.png")
            chr_code += 1
            letter = chr(chr_code)
            velot.pl.dataset_overview_simple(adata, color="velot_confidence", title="", show=show, save=save, save_path=f"figure4_{letter}.png")
            chr_code += 1
            letter = chr(chr_code)


        velot.pl.velocity_quiver(
            adata=adata, color=CLUSTERS_KEY, basis="umap",
            velocity_key=f"{velocity_key}_umap", title=None,
            show=False, save=True,
            save_path=f"figure4_{letter}.png",
            figsize=(5,5), subsample=500
        )
        chr_code += 1
        letter = chr(chr_code)

        velot.pl.velocity_stream(
            adata=adata, color=CLUSTERS_KEY, basis="umap",
            velocity_key=velocity_key, title=None,
            show=False, save=True,
            save_path=f"figure4_{letter}.png",
            figsize=(5,5)
        )
        chr_code += 1
        letter = chr(chr_code)

EDGES=[
    ("Blood progenitors 1", "Blood progenitors 2"),
    ("Blood progenitors 2", "Erythroid1"),
    ("Erythroid1", "Erythroid2"),
    ("Erythroid2", "Erythroid3")
]

for method in METHODS:
    adata = sc.read_h5ad(f"results/data/{method}_seed{SEED}_smooth_{DATASET}.h5ad")

    results = {}
    for field, vkey in (("raw", "velot_velocity_raw_pca"),
                        ("smooth", "velot_velocity_pca")):
        res = velot.metrics.summary(
            adata, cluster_edges=EDGES, cluster_key=CLUSTERS_KEY,
            embedding_key="X_pca", velocity_key=vkey, print_results=False)
        results[field] = res

    velot.pl.metric_summary(
        results, orientation="vertical", layout="column", figsize=(7,7),
        stats_loc="inside", stats_fontsize=13, legend_loc="lower right",
        ylim=(-0.5,1.05), show=show, save=save,
        save_path=f"figure4_{letter}.png")
    chr_code += 1
    letter = chr(chr_code)
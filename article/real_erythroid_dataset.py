import velot
import numpy as np
import scanpy as sc
import scvelo as scv

import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)

figures_path = "/home/user/Documents/velot/article/figures/figure 4"
show = False
save = True
basis = "pca"
project_umap = True if basis == "pca" else False
clusters_key = "celltype"

# Load data
adata = velot.datasets.erythroid()

# Preprocess
sc.pp.filter_cells(adata, min_counts=20)
sc.pp.filter_genes(adata, min_cells=10)

# velot.pp.pca(adata, n_pcs=20)
sc.pp.neighbors(adata, 20, use_rep="X_pca")
velot.pp.pseudotime(adata, root_cluster="Blood progenitors 1", cluster_key=clusters_key)
adata.obs["clusters_id"] = adata.obs[clusters_key].cat.codes
adata.obsm["X_pca"] = adata.obsm["X_pca"][:,:10]

# Compute velocity (full pipeline)
velot.tl.velocity(
    adata=adata,
    basis=f"X_{basis}",
    smooth=True,
    n_clusters=1,
    window_size=200,
    min_window_size=20,
    overlap_fraction=0,
    # spatial_key="clusters_id",
    tail_handling="drop", tail_threshold=20,
    reg=0.2, lambda_time=1, n_epochs=100, lambda_smooth=0.8, lambda_curl=0.8, lambda_divergence=0, k_smooth=30,
    project_umap=project_umap
)

# Visualize
velot.pl.dataset_overview_simple(adata, color=clusters_key, title="", figsize=(5,5), inframe=True, out_legend=False, show=show, save=save, save_path=f"{figures_path}/figure4_a.png")
velot.pl.dataset_overview_simple(adata, color="pseudotime", title="", show=show, save=save, save_path=f"{figures_path}/figure4_b.png")
velot.pl.dataset_overview_simple(adata, color="velot_confidence", title="", show=show, save=save, save_path=f"{figures_path}/figure4_c.png")
velot.pl.velocity_stream(adata, color=clusters_key, title=None, figsize=(5,5), show=show, save=save, save_path=f"{figures_path}/figure4_d.png")
velot.pl.velocity_stream(adata, color=clusters_key, title=None, figsize=(5,5), show=show, save=save, min_mass=0, density=2, cutoff_perc=0, linewidth=1.5, save_path=f"{figures_path}/figure4_d.png")
velot.pl.velocity_quiver(adata, color=clusters_key, basis="umap", title="", subsample=500, figsize=(5,5), velocity_key=f"velot_velocity_raw_umap", show=show, save=save, save_path=f"{figures_path}/figure4_e.png")
velot.pl.velocity_quiver(adata, color=clusters_key, basis="umap", title="", subsample=500, figsize=(5,5), velocity_key=f"velot_velocity_umap", show=show, save=save, save_path=f"{figures_path}/figure4_f.png")
velot.pl.windows(adata, basis="umap", pairs_to_show=tuple([i for i in range(adata.uns["velot_windows"]["n_pairs"])]), figsize_per_panel=(4,4), show=show, save=save, save_path=f"{figures_path}/figure4_j.png")
velot.pl.training_curves_single(adata, figsize=(7,7), show=show, save=save, save_path=f"{figures_path}/figure4_g.png")

# Evaluate
edges = [
    ('Blood progenitors 1', 'Blood progenitors 2'), 
    ('Blood progenitors 2', 'Erythroid1'),
    ('Erythroid1', 'Erythroid2'), 
    ('Erythroid2', 'Erythroid3')
]
results = velot.metrics.summary(adata, cluster_edges=edges, cluster_key=clusters_key, embedding_key=f"X_{basis}", velocity_key=f"velot_velocity_{basis}")
velot.pl.metric_summary(results, orientation="horizontal", layout="row", figsize=(10,9), show=show, save=save, save_path=f"{figures_path}/figure4_h.png")

# Trajectories using the continuous field with evolving pseudotime
velot.tl.compute_trajectories(
    adata,
    basis=f"X_{basis}",
    velocity_key=f"velot_velocity_{basis}",
    direction="forward",
    use_network=True,
    evolve_pseudotime=True,
    end_pseudotime=0.05,
    n_trajectories=30,
    cluster_key=clusters_key,
    n_steps=300, step_size=1
)

velot.pl.trajectories(
    adata, color=clusters_key, line_width=1.5, line_style="-", arrow_frequency=10, arrow_size=15,
    basis="umap", show=show, save=save, save_path=f"{figures_path}/figure4_i.png", trajectory_ids=[4])

import matplotlib.pyplot as plt

n = 30
cols = 5
rows = n // 5 + 1
s = 5
fig, axes = plt.subplots(rows, cols, figsize=(s*cols,s*rows))
axes = axes.flatten()

for i in range(n):
    axes[i] = velot.pl.trajectories(
        adata, color=clusters_key, line_width=1.5, line_style="-", arrow_frequency=10, arrow_size=15,
        show=show, basis="umap", trajectory_ids=[i], ax=axes[i]
    )

# Hide unused subplots
for j in range(n, len(axes)):
    axes[j].remove()

plt.tight_layout()
plt.show()

adata.write("/home/user/Documents/velot/article/data/Gastrulation/erythroid_velot.h5ad")
import velot
import numpy as np
import scanpy as sc
import scvelo as scv

import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)

figures_path = "figures/figure 2"
show = False
save = True
basis = "pca"
project_umap = True if basis == "pca" else False
clusters_key = "clusters"

# Load data
adata = sc.read_h5ad("datasets/endocrinogenesis_day15.5_preprocessed.h5ad")

# Preprocess
velot.pp.pca(adata, n_pcs=10)
sc.pp.neighbors(adata, 20, use_rep="X_pca")
velot.pp.pseudotime(adata, root_cluster="Ngn3 low EP", root_cell=77)
adata.obs["clusters_id"] = adata.obs[clusters_key].cat.codes

# Compute velocity (full pipeline)
velot.tl.velocity(
    adata=adata,
    basis=f"X_{basis}",
    smooth=True,
    n_clusters=None,
    window_size=50,
    overlap_fraction=0,
    # spatial_key="clusters_id",
    reg=0.1, lambda_time=1, n_epochs=150, lambda_smooth=0.7, lambda_curl=0.7, lambda_divergence=0,
    project_umap=project_umap
)

# Visualize
velot.pl.dataset_overview_simple(adata, color=clusters_key, title="", figsize=(5,5), inframe=True, out_legend=False, show=show, save=save, save_path=f"{figures_path}/figure2_a.png")
velot.pl.dataset_overview_simple(adata, color="pseudotime", title="", show=show, save=save, save_path=f"{figures_path}/figure2_b.png")
velot.pl.dataset_overview_simple(adata, color="velot_confidence", title="", show=show, save=save, save_path=f"{figures_path}/figure2_c.png")
velot.pl.velocity_stream(adata, color=clusters_key, title=None, figsize=(5,5), show=show, save=save, save_path=f"{figures_path}/figure2_d.png")
velot.pl.velocity_quiver(adata, color=clusters_key, basis="umap", title="", subsample=500, figsize=(5,5), velocity_key=f"velot_velocity_raw_umap", show=show, save=save, save_path=f"{figures_path}/figure2_e.png")
velot.pl.velocity_quiver(adata, color=clusters_key, basis="umap", title="", subsample=500, figsize=(5,5), velocity_key=f"velot_velocity_umap", show=show, save=save, save_path=f"{figures_path}/figure2_f.png")
velot.pl.windows(adata, pairs_to_show=[29], figsize_per_panel=(4,4), show=show, save=save, save_path=f"{figures_path}/figure2_j.png")
velot.pl.training_curves_single(adata, figsize=(7,7), show=show, save=save, save_path=f"{figures_path}/figure2_g.png")

# Evaluate
edges = [
    ('Ngn3 low EP', 'Ngn3 high EP'), 
    ('Ngn3 high EP', 'Fev+'),
    ('Fev+', 'Delta'), 
    ('Fev+', 'Beta'),
    ('Fev+','Epsilon'),
    ('Fev+','Alpha')
]
results = velot.metrics.summary(adata, cluster_edges=edges, cluster_key=clusters_key, embedding_key=f"X_{basis}", velocity_key=f"velot_velocity_{basis}")
velot.pl.metric_summary(results, orientation="vertical", layout="column", figsize=(7,7), show=show, save=save, save_path=f"{figures_path}/figure2_h.png")

# Trajectories using the continuous field with evolving pseudotime
start_clusters = ["Alpha", "Beta", "Delta", "Epsilon"]
trajectory_ids = [1, 1, 1, 29]
for start_cluster,trajectory_id in zip(start_clusters, trajectory_ids):
    velot.tl.compute_trajectories(
        adata,
        basis=f"X_{basis}",
        velocity_key=f"velot_velocity_{basis}",
        direction="backward",
        use_network=False,
        evolve_pseudotime=True,
        start_cluster=start_cluster,
        n_trajectories=30,
        cluster_key=clusters_key,
        n_steps=300, step_size=2
    )
    velot.pl.trajectories(
        adata, color=clusters_key, trajectory_ids=[trajectory_id], line_width=1.5, line_style="-", arrow_frequency=5, arrow_size=15,
        basis="umap", show=show, save=save, save_path=f"{figures_path}/figure2_i_{start_cluster.lower()}.png"
    )

if save:
    adata.write("datasets/pancreas_velot.h5ad")
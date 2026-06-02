import velot
import numpy as np
import scanpy as sc
import scvelo as scv

import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)

figures_path = "figures/figure 3"
show = False
save = True
basis = "pca"
project_umap = True if basis == "pca" else False
clusters_key = "celltype"

# Load data
try:
    adata = sc.read_h5ad("data/Synthetic/cycle.h5ad")
except:
    adata = velot.datasets.synthetic_cycle((2000, 1500, 1500, 1000), n_rotations=3, x_amplitude=3, z_drift=20, noise_level=0.3, extra_dimensions=1)
    adata.write("data/Synthetic/cycle.h5ad")

# Preprocess
velot.pp.pseudotime(adata, key="true_pseudotime")
adata.obs["clusters_id"] = adata.obs[clusters_key].cat.codes

# Compute velocity (full pipeline)
velot.tl.velocity(
    adata=adata,
    basis=f"X_{basis}",
    smooth=True,
    n_clusters=1,
    window_size=50,
    overlap_fraction=0.0,
    spatial_key=None,
    reg=0.05, lambda_time=1, n_epochs=150, lambda_smooth=0.2, lambda_curl=0.2, lambda_divergence=0,
    project_umap=project_umap
)

# Visualize
velot.pl.dataset_overview_simple(adata, color=clusters_key, title="", figsize=(5,5), inframe=True, out_legend=False, show=show, save=save, save_path=f"{figures_path}/figure3_a.png")
velot.pl.dataset_overview_simple(adata, color="pseudotime", title="", show=show, save=save, save_path=f"{figures_path}/figure3_b.png")
velot.pl.dataset_overview_simple(adata, color="velot_confidence", title="", show=show, save=save, save_path=f"{figures_path}/figure3_c.png")
velot.pl.velocity_stream(adata, color=clusters_key, title=None, figsize=(5,5), show=show, save=save, save_path=f"{figures_path}/figure3_d.png", min_mass=0, density=2, cutoff_perc=0, linewidth=1.2)
velot.pl.velocity_quiver(adata, color=clusters_key, basis="umap", title="", subsample=500, figsize=(5,5), velocity_key=f"velot_velocity_raw_umap", show=show, save=save, save_path=f"{figures_path}/figure3_e.png")
velot.pl.velocity_quiver(adata, color=clusters_key, basis="umap", title="", subsample=500, figsize=(5,5), velocity_key=f"velot_velocity_umap", show=show, save=save, save_path=f"{figures_path}/figure3_f.png")
velot.pl.windows(adata, pairs_to_show=tuple([3*i for i in range(adata.uns["velot_windows"]["n_pairs"]) if 3*i<adata.uns["velot_windows"]["n_pairs"]]), figsize_per_panel=(4,4), show=show, save=save, save_path=f"{figures_path}/figure3_j.png")
velot.pl.training_curves_single(adata, figsize=(7,7), show=show, save=save, save_path=f"{figures_path}/figure3_g.png")

# Evaluate
edges = [
    ('Cluster 1', 'Cluster 2'),
    ('Cluster 2', 'Cluster 3'), 
    ('Cluster 3', 'Cluster 4')
]
results = velot.metrics.summary(adata, cluster_edges=edges, cluster_key=clusters_key, embedding_key=f"X_{basis}", velocity_key=f"velot_velocity_{basis}")
velot.pl.metric_summary(results, orientation="vertical", layout="column", figsize=(7,7), show=show, save=save, save_path=f"{figures_path}/figure3_h.png")

# Trajectories using the continuous field with evolving pseudotime
velot.tl.compute_trajectories(
    adata,
    basis=f"X_{basis}",
    velocity_key=f"velot_velocity_{basis}",
    direction="forward",
    end_pseudotime=0.05,
    use_network=True,
    evolve_pseudotime=True,
    n_trajectories=1,
    cluster_key=clusters_key,
    n_steps=300, step_size=1.5
)
velot.pl.trajectories(
    adata, color=clusters_key, line_width=1.5, line_style="-", arrow_frequency=10, arrow_size=15,
    basis="umap", show=show, save=save, save_path=f"{figures_path}/figure3_i.png")

adata.write("/home/user/Documents/velot/article/data/Synthetic/cycle_velot.h5ad")
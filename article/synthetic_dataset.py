import velot
import numpy as np
import scanpy as sc
import scvelo as scv

figures_path = "/home/user/Documents/velot/figures/figure 3"

# Load data
try:
    adata = sc.read_h5ad("/home/user/Documents/velot/notebooks/data/Synthetic/cycle.h5ad")
except:
    adata = velot.datasets.synthetic_cycle((1000, 2000, 4000, 2000), n_rotations=3, x_amplitude=3, z_drift=20, noise_level=0.3, extra_dimensions=1)
    adata.write("/home/user/Documents/velot/notebooks/data/Synthetic/cycle.h5ad")

# Preprocess
velot.pp.pseudotime(adata, key="true_pseudotime")
adata.obs["clusters_id"] = adata.obs["celltype"].cat.codes

# Compute velocity (full pipeline)
velot.tl.velocity(
    adata=adata,
    basis="X_pca",
    smooth=True,
    n_clusters=1,
    window_size=100,
    overlap_fraction=0.1,
    spatial_key=None,
    reg=0.05, lambda_time=1, n_epochs=300, lambda_smooth=0.5, lambda_curl=0.5, lambda_divergence=0,
    project_umap=True
)

# Visualize
velot.pl.dataset_overview_simple(adata, color="celltype", title="", figsize=(5,5), inframe=True, out_legend=False, save=f"{figures_path}/figure3_a.png")
velot.pl.dataset_overview_simple(adata, color="pseudotime", title="", save=f"{figures_path}/figure3_b.png")
velot.pl.dataset_overview_simple(adata, color="velot_confidence", title="", save=f"{figures_path}/figure3_c.png")
velot.pl.velocity_stream(adata, color="celltype", title=None, figsize=(5,5), save=f"{figures_path}/figure3_d.png")
# velot.pl.windows(adata, pairs_to_show=(6, 7, 8, 9), figsize_per_panel=(4,4), show=True, save=f"{figures_path}/figure3_e.png")

velot.pl.velocity_quiver(adata, color="celltype", basis="pca", title="", subsample=500, figsize=(10,5), velocity_key="velot_velocity_raw", show=True, save=f"{figures_path}/figure3_e.png")
velot.pl.velocity_quiver(adata, color="celltype", basis="pca", title="", subsample=500, figsize=(10,5), velocity_key="velot_velocity", show=True, save=f"{figures_path}/figure3_f.png")

velot.pl.training_curves_single(adata, True, figsize=(7,7), save=f"{figures_path}/figure3_g.png")

# Evaluate
edges = [
    ('Phase_1', 'Phase_2'),
    ('Phase_2', 'Phase_3'), 
    ('Phase_3', 'Phase_4')
]
results = velot.metrics.summary(adata, cluster_edges=edges, cluster_key="celltype")
velot.pl.metric_summary(results, orientation="horizontal", layout="row", figsize=(10,9), save=f"{figures_path}/figure3_h.png")

# Trajectories using the continuous field with evolving pseudotime
velot.tl.compute_trajectories(
    adata,
    basis="X_pca",
    direction="forward",
    end_pseudotime=0.05,
    use_network=True,
    evolve_pseudotime=True,
    n_trajectories=1,
    cluster_key="celltype",
    n_steps=300, step_size=2
)
velot.pl.trajectories(
    adata, color="celltype", line_width=1.5, line_style="-", arrow_frequency=10, arrow_size=15,
    show=False, basis="umap", save=f"{figures_path}/figure3_i.png")
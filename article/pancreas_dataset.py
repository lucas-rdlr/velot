import velot
import numpy as np
import scanpy as sc
import scvelo as scv

figures_path = "/home/user/Documents/velot/figures/figure 2"

# Load data
adata = sc.read_h5ad("/home/user/Documents/velot/notebooks/datasets/endocrinogenesis_day15.5_preprocessed.h5ad")

# Preprocess
velot.pp.pca(adata, n_pcs=20)
velot.pp.pseudotime(adata, root_cluster="Ngn3 low EP", cluster_key="clusters")
adata.obs["clusters_id"] = adata.obs["clusters"].cat.codes

# Compute velocity (full pipeline)
velot.tl.velocity(
    adata=adata,
    basis="X_umap",
    smooth=True,
    n_clusters=None,
    window_size=20,
    overlap_fraction=0,
    spatial_key="clusters_id",
    reg=0.1, lambda_time=1, n_epochs=300, lambda_smooth=0.0, lambda_curl=1, lambda_divergence=0,
    project_umap=False
)

# # Visualize
# velot.pl.dataset_overview_simple(adata, color="clusters", title="", figsize=(5,5), inframe=True, out_legend=False, save=f"{figures_path}/figure2_a.png")
# velot.pl.dataset_overview_simple(adata, color="pseudotime", title="", save=f"{figures_path}/figure2_b.png")
# velot.pl.dataset_overview_simple(adata, color="velot_confidence", title="", save=f"{figures_path}/figure2_c.png")
# velot.pl.velocity_stream(adata, color="clusters", title=None, figsize=(5,5), save=f"{figures_path}/figure2_d.png")
# velot.pl.windows(adata, pairs_to_show=(6, 7), figsize_per_panel=(4,4), show=True, save=f"{figures_path}/figure2_e.png")
# velot.pl.training_curves_single(adata, True, figsize=(7,7), save=f"{figures_path}/figure2_f.png")

# # Evaluate
# edges = [
#     ('Ngn3 low EP', 'Ngn3 high EP'), 
#     ('Ngn3 high EP', 'Fev+'),
#     ('Fev+', 'Delta'), 
#     ('Fev+', 'Beta'),
#     ('Fev+','Epsilon'),
#     ('Fev+','Alpha')
# ]
# results = velot.metrics.summary(adata, cluster_edges=edges, cluster_key="clusters")
# velot.pl.metric_summary(results, orientation="horizontal", layout="row", figsize=(10,9), save=f"{figures_path}/figure2_g.png")

# # Trajectories using the continuous field with evolving pseudotime
# velot.tl.compute_trajectories(
#     adata,
#     basis="X_umap",
#     start_cluster="Beta",
#     target_cluster="Ngn3 high EP",
#     direction="backward",
#     use_network=True,
#     evolve_pseudotime=True,
#     n_trajectories=30,
#     cluster_key="clusters",
#     n_steps=300, step_size=2
# )
# velot.pl.trajectories(
#     adata, color="clusters", line_width=1.5, line_style="-", arrow_frequency=10, arrow_size=15,
#     show=True, basis="umap", save=f"{figures_path}/figure2_j.png")

adata.write("pancreas_velot.h5ad")
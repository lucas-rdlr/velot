import velot
import numpy as np
import scanpy as sc
import scvelo as scv

import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)

figures_path = "/home/user/Documents/velot/article/figures/figure 11"
show = False
save = True
basis = "pca"
project_umap = True if basis == "pca" else False
clusters_key = "clusters"

# Load data
adata = sc.read_h5ad("/home/user/Documents/velot/article/data/BoneMarrow/human_cd34_bone_marrow.h5ad")

# Preprocess
scv.pp.filter_genes(adata, min_shared_counts=20)
sc.pp.pca(adata, n_comps=20)
sc.pp.neighbors(adata, 20, use_rep="X_pca")
sc.tl.umap(adata)

velot.pp.pseudotime(adata, root_cluster="HSC_1", cluster_key=clusters_key)
adata.obs["clusters_id"] = adata.obs[clusters_key].cat.codes

# Compute velocity (full pipeline)
velot.tl.velocity(
    adata=adata,
    basis=f"X_{basis}",
    smooth=True,
    n_clusters=None,
    window_size=100,
    tail_handling="drop", tail_threshold=20,
    overlap_fraction=0,
    # spatial_key="clusters_id",
    reg=0.1, lambda_time=1, n_epochs=200, lambda_smooth=0.3, lambda_curl=0.3, lambda_divergence=0,
    project_umap=project_umap, project_basis="X_tsne"
)

# Visualize
velot.pl.dataset_overview_simple(adata, basis="tsne", color=clusters_key, title="", figsize=(5,5), inframe=True, out_legend=False, show=show, save=save, save_path=f"{figures_path}/figure10_a.png")
velot.pl.dataset_overview_simple(adata, basis="tsne", color="pseudotime", title="", show=show, save=save, save_path=f"{figures_path}/figure10_b.png")
velot.pl.dataset_overview_simple(adata, basis="tsne", color="velot_confidence", title="", show=show, save=save, save_path=f"{figures_path}/figure10_c.png")
velot.pl.velocity_stream(adata, basis="tsne", color=clusters_key, title=None, figsize=(5,5), show=show, save=save, save_path=f"{figures_path}/figure10_d.png")
velot.pl.velocity_quiver(adata, color=clusters_key, basis="tsne", title="", subsample=500, figsize=(5,5), velocity_key=f"velot_velocity_raw_tsne", show=show, save=save, save_path=f"{figures_path}/figure10_e.png")
velot.pl.velocity_quiver(adata, color=clusters_key, basis="tsne", title="", subsample=500, figsize=(5,5), velocity_key=f"velot_velocity_tsne", show=show, save=save, save_path=f"{figures_path}/figure10_f.png")
velot.pl.windows(adata, basis="tsne", pairs_to_show=tuple([i for i in range(adata.uns["velot_windows"]["n_pairs"])]), figsize_per_panel=(4,4), show=show, save=save, save_path=f"{figures_path}/figure10_j.png")
velot.pl.training_curves_single(adata, figsize=(7,7), show=show, save=save, save_path=f"{figures_path}/figure10_g.png")

# Evaluate
edges = [
    ('HSC_1', 'Mega'),
    ('HSC_1', 'CLP'),
    ('HSC_1', 'Ery_1'), 
    ('Ery_1', 'Ery_2'),
    ('HSC_1', 'HSC_2'),
    ('HSC_2','Precursors'),
    ('HSC_2', 'Mono_2'),
    ('Mono_2', 'Mono_1'),
    ('Precursors','DCs')
]
results = velot.metrics.summary(adata, cluster_edges=edges, cluster_key=clusters_key, embedding_key=f"X_{basis}", velocity_key=f"velot_velocity_{basis}")
velot.pl.metric_summary(results, orientation="horizontal", layout="row", figsize=(10,9), show=show, save=save, save_path=f"{figures_path}/figure10_h.png")

# # Trajectories using the continuous field with evolving pseudotime
# velot.tl.compute_trajectories(
#     adata,
#     basis=f"X_{basis}",
#     velocity_key=f"velot_velocity_{basis}",
#     direction="forward",
#     use_network=True,
#     evolve_pseudotime=True,
#     end_pseudotime=0.1,
#     n_trajectories=30,
#     cluster_key=clusters_key,
#     n_steps=300, step_size=2
# )
# velot.pl.trajectories(
#     adata, color=clusters_key, line_width=1.5, line_style="-", arrow_frequency=10, arrow_size=15,
#     basis="umap", show=show, save=save, save_path=f"{figures_path}/figure10_i.png")

# import matplotlib.pyplot as plt

# n = 30
# cols = 5
# rows = n // 5 + 1
# s = 5
# fig, axes = plt.subplots(rows, cols, figsize=(s*cols,s*rows))
# axes = axes.flatten()

# for i in range(n):
#     axes[i] = velot.pl.trajectories(
#         adata, color=clusters_key, line_width=1.5, line_style="-", arrow_frequency=10, arrow_size=15,
#         show=show, basis="umap", trajectory_ids=[i], ax=axes[i]
#     )

# # Hide unused subplots
# for j in range(n, len(axes)):
#     axes[j].remove()

# plt.tight_layout()
# plt.show()

# adata.write("/home/user/Documents/velot/article/data/BoneMarrow/bonemarrow_velot.h5ad")
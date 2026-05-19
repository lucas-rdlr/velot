import velot
import numpy as np
import scanpy as sc
import scvelo as scv

import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)

figures_path = "/home/user/Documents/velot/article/figures/figure 10"
show = False
save = True
basis = "pca"
project_umap = True if basis == "pca" else False
clusters_key = "Celltype"

# Load data
adata = sc.read_h5ad("/home/user/Documents/velot/article/data/HindBrain/Hindbrain_GABA_Glio.h5ad")

print(adata)
print(adata.obs["Celltype"].value_counts())

# Preprocess
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
velot.pp.pca(adata, n_pcs=10)
sc.pp.neighbors(adata, 50)
velot.pp.pseudotime(adata, root_cluster="Neural stem cells", cluster_key=clusters_key)
adata.obs["clusters_id"] = adata.obs[clusters_key].cat.codes

# Compute velocity (full pipeline)
velot.tl.velocity(
    adata=adata,
    basis=f"X_{basis}",
    smooth=True,
    n_clusters=1,
    window_size=500,
    overlap_fraction=0,
    # spatial_key="clusters_id",
    reg=0.1, lambda_time=1, n_epochs=150, lambda_smooth=0.5, lambda_curl=0.5, lambda_divergence=0,
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
    ('Neural stem cells', 'progenitors'),
    ('Proliferating VZ progenitors', 'VZ progenitors'),
    ('VZ progenitors', 'Gliogenic progenitors'),
    ('VZ progenitors', 'Differentiating GABA interneurons'),
    ('Differentiating GABA interneurons', 'GABA interneurons')
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

# adata.write("/home/user/Documents/velot/article/data/HindBrain/hindbrain_velot.h5ad")
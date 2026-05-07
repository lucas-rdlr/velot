import velot
import scvelo as scv

# Load data
adata = velot.datasets.synthetic_bifurcation(400, 3, (300, 400), (5, 6), (1, -2), 0.2, 1)

velot.pp.pseudotime(adata, key="true_pseudotime")
adata.obs["clusters_id"] = adata.obs["celltype"].cat.codes
# Compute velocity (full pipeline)
# velot.tl.velocity(adata, "X_umap", True, None, 50, 0.0, spatial_key="clusters_id", reg=0.1, n_epochs=300, project_umap=False)
velot.tl.velocity(adata, "X_pca", True, None, 50, 0.0, spatial_key="clusters_id", reg=0.1, n_epochs=300, project_umap=True, lambda_smooth=0.5, lambda_curl=0.1, lambda_divergence=0)
print(adata)
# Visualize
# velot.pl.dataset_overview(adata, color="celltype", title=True, vertical=False, figsize=(8,4), save="/home/user/Documents/velot/figures/figure1_b.png")

# velot.pl.spatial_clusters(adata)
# velot.pl.windows(adata, pairs_to_show=(6, 7), figsize_per_panel=(4,4), show=True, save="/home/user/Documents/velot/figures/figure1_c.png")

# Wider context (more padding)
# velot.pl.window_transport(adata, pair_index=6, figsize=(7,7), padding=0.6, show_top_k=3, cluster_key="celltype", save="/home/user/Documents/velot/figures/figure1_e.png")

# velot.pl.confidence(adata)
velot.pl.velocity_stream(adata, color="celltype")
velot.pl.velocity_stream(adata, basis="pca", color="celltype")
# velot.pl.velocity_stream(adata, color="celltype", figsize=(10,10), save="/home/user/Documents/velot/figures/figure1_d.png")

# velot.pl.training_curves(adata, True, figsize=(4,4), save="/home/user/Documents/velot/figures/figure1_f.png")
velot.pl.training_curves_single(adata, True, figsize=(7,7))

# Evaluate
edges = [
    ('Root', 'Branch_1'), 
    ('Root', 'Branch_2')
]

velot.metrics.summary(adata, cluster_edges=edges, cluster_key="celltype")



# # Query the continuous field at arbitrary points
# random_points = np.random.randn(100, 30) * 0.5
# V = velot.tl.query_velocity(adata, random_points)

# # Trajectories using the continuous field with evolving pseudotime
# velot.tl.compute_trajectories(
#     adata,
#     start_cluster="Branch_2",
#     direction="backward",
#     use_network=True,
#     evolve_pseudotime=True,
#     n_trajectories=30,
#     cluster_key="celltype",
# )
# velot.pl.trajectories(adata, color="celltype")

# # Compare: fixed vs evolving pseudotime
# velot.tl.compute_trajectories(
#     adata, start_cluster="Root",
#     direction="forward", evolve_pseudotime=False,
# )
# velot.pl.trajectories(adata, title="Fixed pseudotime")

# velot.tl.compute_trajectories(
#     adata, start_cluster="Root",
#     direction="forward", evolve_pseudotime=True,
# )
# velot.pl.trajectories(adata, title="Evolving pseudotime")




# # Drop 200 particles at the source and let them flow
# velot.tl.simulate_flow(
#     adata,
#     n_particles=100,
#     source_cluster="Root",
#     n_steps=300,
#     step_size=0.05,
#     noise_scale=0.05,
#     cluster_key="celltype",
# )

# # Visualize: color by pseudotime (blue=early, red=late)
# velot.pl.flow_simulation(adata, color="celltype", colorby="pseudotime")

# # Visualize: color by terminal fate
# velot.pl.flow_simulation(adata, color="celltype", colorby="fate")

# # Start from a specific progenitor population
# velot.tl.simulate_flow(
#     adata,
#     source_cluster="Root",
#     n_particles=300,
#     noise_scale=0.02,
#     cluster_key="celltype",
# )
# velot.pl.flow_simulation(adata, colorby="fate")

# Start from the earliest 5% of cells regardless of cluster
# velot.tl.simulate_flow(
#     adata,
#     source_pseudotime_max=0.05,
#     n_particles=10,
#     noise_scale=0.0,
#     cluster_key="celltype",
#     step_size=0.1
# )
# velot.pl.flow_simulation(adata, colorby="pseudotime")
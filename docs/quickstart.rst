Quick Start
===========

Minimal Example
---------------

.. code-block:: python

   import velot

   # Load a dataset
   adata = velot.datasets.pancreas()

   # Preprocess
   adata = velot.pp.prepare(
       adata, root_cluster="Ductal", cluster_key="clusters"
   )

   # Compute velocity
   velot.tl.velocity(adata)

   # Visualize
   velot.pl.velocity_stream(adata, color="clusters")

Step-by-Step Example
--------------------

For more control over individual pipeline stages:

.. code-block:: python

   import velot

   adata = velot.datasets.pancreas()

   # Step 1: Preprocessing
   adata = velot.pp.prepare(
       adata, root_cluster="Ductal", cluster_key="clusters"
   )

   # Step 2: Build spatial-temporal windows
   velot.tl.build_windows(adata, n_clusters=10, window_size=50)

   # Step 3: Compute OT velocity
   velot.tl.compute_ot_velocity(
       adata, reg=0.05, lambda_time=1.0, lambda_knn=1.0
   )

   # Step 4: Smooth with neural network
   velot.tl.smooth_velocity(
       adata, n_epochs=200, lambda_smooth=0.5
   )

   # Step 5: Project to UMAP for visualization
   velot.tl.project_to_umap(adata)

   # Visualize
   velot.pl.velocity_stream(adata, color="clusters")

Evaluating Velocity Quality
----------------------------

.. code-block:: python

   edges = [
       ("Ngn3 low EP", "Ngn3 high EP"),
       ("Ngn3 high EP", "Fev+"),
       ("Fev+", "Alpha"),
       ("Fev+", "Beta"),
   ]

   results = velot.metrics.summary(
       adata, cluster_edges=edges, cluster_key="clusters"
   )

   velot.pl.metric_summary(results)

Trajectory Analysis
--------------------

.. code-block:: python

   velot.tl.compute_trajectories(
       adata,
       start_cluster="Ngn3 low EP",
       direction="forward",
       n_trajectories=20,
   )

   velot.pl.trajectories(adata, color="clusters")
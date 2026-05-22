Tutorials
=========

.. note::

   Tutorials coming soon. In the meantime, see the :doc:`quickstart`
   guide and the scripts in the ``article/`` directory of the
   `GitHub repository <https://github.com/lucas-rdlr/velot>`_.

Synthetic Datasets
------------------

.. code-block:: python

   import velot

   # Linear trajectory
   adata = velot.datasets.synthetic_linear(
       densities=[200, 200, 200],
       positions=[1, 2, 3],
   )

   # Bifurcation
   adata = velot.datasets.synthetic_bifurcation(
       root_density=300,
       branch_densities=[200, 200],
   )

   # Cycle / cell cycle
   adata = velot.datasets.synthetic_cycle(
       densities=[200, 50, 200, 50],
       names=["G1", "S", "G2", "M"],
   )

Custom Datasets
---------------

VelOT works with any AnnData object. The minimum requirements are:

1. Gene expression in ``adata.X`` (raw counts or log-normalized)
2. Cluster labels in ``adata.obs`` (any column name)

.. code-block:: python

   import scanpy as sc
   import velot

   # Load your own data
   adata = sc.read_h5ad("my_data.h5ad")

   # Preprocess
   adata = velot.pp.prepare(
       adata,
       root_cluster="Progenitor",
       cluster_key="cell_type",
   )

   # Compute velocity
   velot.tl.velocity(adata)


.. toctree::
..    :maxdepth: 1

..    tutorials/quickstart
..    tutorials/synthetic_datasets
..    tutorials/custom_data
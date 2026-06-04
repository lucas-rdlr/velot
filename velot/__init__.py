"""
VelOT — Optimal Transport RNA Velocity with Flow Matching Smoothing.

API follows the scanpy/scvelo convention::

    import velot

    adata = velot.datasets.pancreas()
    velot.pp.prepare(adata, root_cluster="Ductal")
    velot.tl.velocity(adata)
    velot.pl.velocity_stream(adata)

    edges = [("Ductal", "Ngn3 low EP"), ("Ngn3 low EP", "Ngn3 high EP")]
    velot.metrics.summary(adata, cluster_edges=edges)
"""

from . import pp, tl, pl, metrics, datasets

__version__ = "1.0.0"

__all__ = ["pp", "tl", "pl", "metrics", "datasets"]
"""
Dataset registry for the revision benchmark.

Everything that differs between datasets lives here — path, labels, PCA
size, neighbours, root cluster, transition edges, window size — so the
runner and the sweeps are identical for every dataset. The preprocessing
of each dataset is the one used for the published benchmark, with a
single change: the PCA is truncated BEFORE the kNN graph is built, so the
graph, the velocity and the metrics all live in the same space.

The smoother settings are shared by every dataset and every method (see
SHARED in run_velot.py). Only ``window_size`` is per dataset, and it is
chosen from the data geometry with ``velot.tl.choose_window_size`` rather
than tuned against the metrics.

``basis`` names the representation everything runs in; it defaults to
``X_pca`` and only oligodendroglioma overrides it.

``pseudotime_key`` is optional: when set, that obs column is used as the
ordering and no root is chosen and no DPT is run. The dyngen synthetics
carry ``sim_time``, so setting it turns "how good is our pseudotime?"
into an experiment arm rather than a hidden assumption.

``generate`` is the alternative to ``path``: a callable returning a
fresh AnnData. The two hand-built topologies use it, so they need no
file on disk and are reproducible from the seed alone.

``truth_key`` marks a dataset that carries a known velocity field in
``obsm``. When present the runner also scores every field by cosine to
it - a metric with no term in the VelOT objective, which is what R1.1
and R3.4 ask for. It is NaN for every dataset without ground truth.

The three dyngen synthetics use the 5-component versions
(``*_processed_5PCA.h5ad``, 20 neighbours). Their dynamics are ~3-D: at
30 PCs more than 90 % of the dimensions carry no signal, the unaveraged
OT field is then noise-dominated, and both VelOT and the kNN-gradient
baseline score at the level of a random vector against the true field.
"""
from pathlib import Path

import scanpy as sc
import velot

ROOT = Path(__file__).resolve().parents[3]          # the velot repo root
DATA = ROOT / "article" / "data"
DATASETS_DIR = ROOT / "article" / "datasets"


# ---------------------------------------------------------------------
# per-dataset preprocessing
# ---------------------------------------------------------------------
def _prep_synthetic(adata, cfg):
    """The *_5PCA files already carry the PCA and the kNN graph built in
    process_5PCA.ipynb (5 components, 20 neighbours), so there is nothing
    left to do here."""
    return None


def _prep_erythroid(adata, cfg):
    sc.pp.filter_cells(adata, min_counts=20)
    sc.pp.filter_genes(adata, min_cells=10)
    adata.obsm["X_pca"] = adata.obsm["X_pca"][:, :cfg["n_pcs"]]
    sc.pp.neighbors(adata, cfg["n_neighbors"], use_rep="X_pca")


def _prep_pancreas(adata, cfg):
    velot.pp.pca(adata, n_pcs=cfg["n_pcs"])
    sc.pp.neighbors(adata, cfg["n_neighbors"], use_rep="X_pca")


def _prep_truncate_pca(adata, cfg):
    """Datasets shipped with a PCA already computed."""
    adata.obsm["X_pca"] = adata.obsm["X_pca"][:, :cfg["n_pcs"]]
    sc.pp.neighbors(adata, cfg["n_neighbors"], use_rep="X_pca")


def _prep_oligodendroglioma(adata, cfg):
    """Malignant oligodendroglioma cells (Tirosh et al. 2016).

    The published analysis works in the two-dimensional
    differentiation/stemness plane rather than in a PCA, and derives the
    three fate states from it, so this dataset keeps ``X_dif_stem`` as
    its basis: the graph, the velocity and the metrics all live there,
    exactly as they live in the PCA for the other datasets.
    """
    import numpy as np
    import pandas as pd

    keep = ~np.isnan(adata.obsm["X_dif_stem"]).any(axis=1)
    adata = adata[keep].copy()          # cells with no position in the plane

    xy = pd.DataFrame(adata.obsm["X_dif_stem"], index=adata.obs_names,
                      columns=["dif_stem_x", "dif_stem_y"])
    adata.obs[["dif_stem_x", "dif_stem_y"]] = xy
    # the four states of Tirosh et al. 2016 (PMID 27806376)
    state = pd.Series("Unassigned", index=adata.obs_names)
    state[xy["dif_stem_x"] <= 0.0] = "AC-like"
    state[xy["dif_stem_x"] >= 0.5] = "OC-like"
    state[xy["dif_stem_y"] >= 0.0] = "Stem-like"
    adata.obs["fate_state"] = state.values

    sc.pp.neighbors(adata, cfg["n_neighbors"], use_rep=cfg["basis"])
    return adata


def _prep_hindbrain(adata, cfg):
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    velot.pp.pca(adata, n_pcs=cfg["n_pcs"])
    sc.pp.neighbors(adata, cfg["n_neighbors"], use_rep="X_pca")


# ---------------------------------------------------------------------
# hand-built topologies with a known velocity
# ---------------------------------------------------------------------
# These are deliberately low-dimensional: they exist to show what OT does
# to a transport problem whose answer can be written down, not to stand
# in for real data. The dyngen sets are the high-dimensional counterpart.
def _true_velocity(adata, slopes):
    """Analytic unit velocity, expressed in the basis of ``X_pca``.

    In feature space the motion is (1, slope, 0, 0, ...): the extra
    dimensions are noise and carry no dynamics. When extra dimensions are
    present the builder runs a real PCA, which rotates the axes, so the
    truth is pushed through the loadings rather than compared to the
    first two components.
    """
    import numpy as np
    lab = adata.obs["celltype"].astype(str).values
    Vf = np.zeros((adata.n_obs, adata.shape[1]))
    names = [c for c in dict.fromkeys(lab) if c != "Root"]
    Vf[:, 0] = 1.0
    for name, slope in zip(names, slopes):
        Vf[lab == name, 1] = slope
    if "PCs" in adata.varm:
        V = Vf @ np.asarray(adata.varm["PCs"])
    else:
        V = Vf[:, :adata.obsm["X_pca"].shape[1]]
    return V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-12, None)


def _gen_bifurcation(cfg):
    adata = velot.datasets.synthetic_bifurcation(
        root_density=400, branch_densities=(300, 300),
        branch_positions=(2, 2), branch_slopes=(2, -2), noise_level=0.05,
        extra_dimensions=0, n_neighbors=cfg["n_neighbors"],
        seed=cfg.get("data_seed", 0))
    adata.obsm["true_velocity_pca"] = _true_velocity(adata, (2, -2))
    return adata


def _gen_tree(cfg):
    adata = velot.datasets.synthetic_tree(
        topology=[
            {"name": "Root", "parent": None, "n_cells": 400, "length": 1.0, "slope": 0.0},
            {"name": "Branch_1", "parent": "Root", "n_cells": 200, "length": 1, "slope": 2.0},
            {"name": "Branch_2", "parent": "Root", "n_cells": 300, "length": 1, "slope": -2.0},
            {"name": "Branch_3", "parent": "Branch_2", "n_cells": 150, "length": 0.5, "slope": 1},
        ],
        noise_level=0.05, extra_dimensions=0,
        n_neighbors=cfg["n_neighbors"], seed=cfg.get("data_seed", 0))
    adata.obsm["true_velocity_pca"] = _true_velocity(adata, (2, -2, 1))
    return adata


def _prep_generated(adata, cfg):
    """The builder already made the PCA, the UMAP and the graph."""
    return None


# ---------------------------------------------------------------------
# the datasets
# ---------------------------------------------------------------------
DATASETS = {
    # ---------------- real ----------------
    "erythroid": dict(
        kind="real",
        path=DATA / "Gastrulation" / "erythroid_lineage.h5ad",
        clusters_key="celltype",
        root_cluster="Blood progenitors 1",
        n_pcs=10, n_neighbors=20,
        n_clusters=1, spatial_key=None,
        window_size=500,
        preprocess=_prep_erythroid,
        edges=[("Blood progenitors 1", "Blood progenitors 2"),
               ("Blood progenitors 2", "Erythroid1"),
               ("Erythroid1", "Erythroid2"),
               ("Erythroid2", "Erythroid3")],
    ),
    "pancreas": dict(
        kind="real",
        path=DATASETS_DIR / "endocrinogenesis_day15.5_preprocessed.h5ad",
        clusters_key="clusters",
        root_cluster="Ngn3 low EP",
        n_pcs=10, n_neighbors=20,
        n_clusters=1, spatial_key=None,
        window_size=300,
        preprocess=_prep_pancreas,
        edges=[("Ngn3 low EP", "Ngn3 high EP"),
               ("Ngn3 high EP", "Fev+"),
               ("Fev+", "Delta"), ("Fev+", "Beta"),
               ("Fev+", "Epsilon"), ("Fev+", "Alpha")],
    ),
    "murine": dict(
        kind="real",
        path=DATA / "Murine" / "preprocessed.h5ad",
        clusters_key="cell_type",
        root_cluster="Stem cells",
        n_pcs=20, n_neighbors=20,
        n_clusters=1, spatial_key=None,
        window_size=800,
        preprocess=_prep_truncate_pca,
        edges=[("Stem cells", "TA cells"),
               ("Stem cells", "Goblet cells"),
               ("Goblet cells", "Paneth cells")],
    ),
    "hindbrain": dict(
        kind="real",
        path=DATA / "HindBrain" / "Hindbrain_GABA_Glio.h5ad",
        clusters_key="Celltype",
        root_cluster="Neural stem cells",
        n_pcs=10, n_neighbors=20,
        n_clusters=1, spatial_key=None,
        window_size=800,
        preprocess=_prep_hindbrain,
        project_basis="X_tsne",
        edges=[("Neural stem cells", "Proliferating VZ progenitors"),
               ("Proliferating VZ progenitors", "VZ progenitors"),
               ("VZ progenitors", "Gliogenic progenitors"),
               ("VZ progenitors", "Differentiating GABA interneurons"),
               ("Differentiating GABA interneurons", "GABA interneurons")],
    ),
    "oligodendroglioma": dict(
        kind="real",
        path=DATA / "Oligodendroglioma" / "OG_processed_data.h5ad",
        clusters_key="fate_state",
        root_cluster="Stem-like",
        basis="X_dif_stem",
        n_pcs=2, n_neighbors=20,
        n_clusters=1, spatial_key=None,
        window_size=100,
        preprocess=_prep_oligodendroglioma,
        project_basis="X_dif_stem",
        edges=[("Stem-like", "AC-like"), ("Stem-like", "OC-like")],
    ),
    # ---------------- synthetic ----------------
    # The published runs pinned a manual root cell here (bifurcation 166,
    # trifurcation 9). The revision uses the root CLUSTER with the
    # geometric rule instead, so no cell is hand-picked anywhere.
    "linear": dict(
        kind="synthetic",
        path=DATA / "Synthetic" / "synthetic_linear_processed.h5ad",
        clusters_key="milestone",
        root_cluster="A",
        n_pcs=5, n_neighbors=20,
        n_clusters=1, spatial_key=None, #"milestone",
        window_size=100,
        preprocess=_prep_synthetic,
        edges=[("A", "B"), ("B", "C")],
    ),
    "bifurcation": dict(
        kind="synthetic",
        path=DATA / "Synthetic" / "synthetic_bifurcation_processed.h5ad",
        clusters_key="milestone",
        root_cluster="A",
        n_pcs=5, n_neighbors=20,
        n_clusters=1, spatial_key=None, #"milestone",
        window_size=200,
        preprocess=_prep_synthetic,
        edges=[("A", "B"), ("A", "C"), ("B", "D"), ("C", "E")],
    ),
    "trifurcation": dict(
        kind="synthetic",
        path=DATA / "Synthetic" / "synthetic_trifurcation_processed.h5ad",
        clusters_key="milestone",
        root_cluster="A",
        n_pcs=5, n_neighbors=20,
        n_clusters=1, spatial_key=None, #"milestone",
        window_size=200,
        preprocess=_prep_synthetic,
        edges=[("A", "B"), ("B", "C"), ("C", "F"),
               ("B", "D"), ("D", "G"), ("E", "H")],
    ),
    "linear_simtime": dict(
        kind="synthetic",
        path=DATA / "Synthetic" / "synthetic_linear_processed_5PCA.h5ad",
        clusters_key="milestone",
        pseudotime_key="sim_time",
        n_pcs=5, n_neighbors=20,
        n_clusters=None,
        preprocess=_prep_synthetic,
        window_size=500,           # chosen with choose_window_size
        edges=[("A", "B"), ("B", "C")],
    ),
    "bifurcation_simtime": dict(
        kind="synthetic",
        path=DATA / "Synthetic" / "synthetic_bifurcation_processed_5PCA.h5ad",
        clusters_key="milestone",
        pseudotime_key="sim_time",
        n_pcs=5, n_neighbors=20,
        n_clusters=None,
        preprocess=_prep_synthetic,
        window_size=500,           # dense: no window reaches the target move
        edges=[("A", "B"), ("A", "C"), ("B", "D"), ("C", "E")],
    ),
    "trifurcation_simtime": dict(
        kind="synthetic",
        path=DATA / "Synthetic" / "synthetic_trifurcation_processed_5PCA.h5ad",
        clusters_key="milestone",
        pseudotime_key="sim_time",
        n_pcs=5, n_neighbors=20,
        n_clusters=None,
        preprocess=_prep_synthetic,
        window_size=100,           # chosen with choose_window_size
        edges=[("A", "B"), ("B", "C"), ("C", "F"),
               ("B", "D"), ("D", "G"), ("E", "H")],
    ),
    
    # ---------------- hand-built, ground truth known ----------------
    # Ordering comes from ``true_pseudotime``, so no root and no DPT: the
    # question these answer is what OT does to the transport, with the
    # pseudotime held exact.
    "gen_bifurcation": dict(
        kind="generated",
        generate=_gen_bifurcation,
        clusters_key="celltype",
        pseudotime_key="true_pseudotime",
        truth_key="true_velocity_pca",
        root_cluster="Root",
        n_pcs=2, n_neighbors=30, data_seed=0,
        n_clusters=1, spatial_key=None,      # unsupervised; override on the CLI
        window_size=50,
        preprocess=_prep_generated,
        edges=[("Root", "Branch_1"), ("Root", "Branch_2")],
    ),
    "gen_tree": dict(
        kind="generated",
        generate=_gen_tree,
        clusters_key="celltype",
        pseudotime_key="true_pseudotime",
        truth_key="true_velocity_pca",
        root_cluster="Root",
        n_pcs=2, n_neighbors=30, data_seed=0,
        n_clusters=1, spatial_key=None,      # unsupervised; override on the CLI
        window_size=50,
        preprocess=_prep_generated,
        edges=[("Root", "Branch_1"), ("Root", "Branch_2"),
               ("Branch_2", "Branch_3")],
    ),
}


# Every dataset runs in ``basis``; only oligodendroglioma overrides it.
for _cfg in DATASETS.values():
    _cfg.setdefault("basis", "X_pca")


def load(name):
    """Read a dataset and run its preprocessing. Returns (adata, cfg)."""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; "
                       f"available: {sorted(DATASETS)}")
    cfg = DATASETS[name]
    if cfg.get("generate") is not None:
        adata = cfg["generate"](cfg)
    else:
        adata = sc.read_h5ad(cfg["path"])
    # a preprocess fn mutates in place, or returns a new object if it
    # has to subset the cells
    adata = cfg["preprocess"](adata, cfg) or adata
    if cfg["clusters_key"] in adata.obs:
        adata.obs[cfg["clusters_key"]] = \
            adata.obs[cfg["clusters_key"]].astype("category")
        adata.obs["clusters_id"] = adata.obs[cfg["clusters_key"]].cat.codes
    return adata, cfg

"""
Ground-truth comparison on synthetic data, with a seed sweep.

This is the control the real-data benchmark cannot provide: on synthetic
topologies the true velocity is known, so every field can be scored
against it with a metric that is OUTSIDE the VelOT objective (R1.1,
R3.4), next to the two benchmark metrics computed on the same field.

    python synthetic_truth.py                      # default methods, 5 seeds
    python synthetic_truth.py --all-methods        # the full OT ablation
    python synthetic_truth.py -t tree -w agnostic  # one cell of the grid

The grid is topology x windowing x method x seed:

  topology    bifurcation (one split) and tree (a split plus a sub-split)
  windowing   "agnostic" = n_clusters=1, windows run along pseudotime over
              all cells and therefore MIX branches past the split - this is
              the fully unsupervised setting the revision benchmark uses.
              "labels"   = windows are built inside each cell type, so no
              OT pair ever crosses the split.
  method      the same METHODS dict the real benchmark uses, imported from
              run_velot.py so the names cannot drift apart.
  seed        random_state of the smoother, exactly as in the benchmark.

Writes results/synthetic_truth/{runs,summary}.csv. ``runs.csv`` has the
same column names as the real benchmark's runs.csv wherever they mean the
same thing (dataset, method, seed, field, cbdir_mean, iccoh_mean,
coverage), plus the truth-based columns, so the same plotting code works
on both.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import scanpy as sc
import velot

HERE = Path(__file__).resolve().parent
OUT = HERE / "benchmark" / "article_figures" / "figure2"

# The method definitions are the benchmark's, imported rather than copied
# so that "velot" means the same thing on synthetic and on real data.
# run_velot.py chdirs into its own directory at import time.
_cwd = Path.cwd()
sys.path.insert(0, str(HERE / "benchmark"))
from run_velot import METHODS                                   # noqa: E402
os.chdir(_cwd)

sc.settings.verbosity = 0

from ot_variations import true_velocity, cosine_rows            # noqa: E402

# Reported in the benchmark, plus the two candidate fixes for the
# terminal-state failure that balanced OT shows under agnostic windows.
DEFAULT_METHODS = ["velot", "velot_gradient", "velot_unbalanced"]

# Small datasets: the smoother settings are the benchmark's, only the
# window is scaled to the size of these toy problems.
SHARED = dict(
    basis="X_pca",
    smooth=True,
    n_clusters=1,
    window_size=50,
    min_window_size=20,
    overlap_fraction=0.0,
    tail_handling="drop",
    tail_threshold=20,
    n_epochs=100,
    lambda_smooth=0.1,
    lambda_curl=0.1,
    lambda_divergence=0.0,
    k_smooth=30,
    project_umap=False,
    verbose=False,
)


# ---------------------------------------------------------------------
# topologies
# ---------------------------------------------------------------------
def _bifurcation(seed=0):
    ad = velot.datasets.synthetic_bifurcation(
        root_density=400, branch_densities=(300, 300), branch_positions=(2, 2),
        branch_slopes=(2, -2), noise_level=0.05, extra_dimensions=0,
        n_neighbors=30, seed=seed)
    ad.obsm["true_velocity_pca"] = true_velocity(ad, (2, -2))
    return ad


def _tree(seed=0):
    ad = velot.datasets.synthetic_tree(
        topology=[
            {"name": "Root", "parent": None, "n_cells": 400, "length": 1.0, "slope": 0.0},
            {"name": "Branch_1", "parent": "Root", "n_cells": 200, "length": 1, "slope": 2.0},
            {"name": "Branch_2", "parent": "Root", "n_cells": 300, "length": 1, "slope": -2.0},
            {"name": "Branch_3", "parent": "Branch_2", "n_cells": 150, "length": 0.5, "slope": 1},
        ],
        noise_level=0.05, extra_dimensions=0, n_neighbors=30, seed=seed)
    ad.obsm["true_velocity_pca"] = true_velocity(ad, (2, -2, 1))
    return ad


TOPOLOGIES = {
    "bifurcation": dict(build=_bifurcation,
                        edges=[("Root", "Branch_1"), ("Root", "Branch_2")]),
    "tree": dict(build=_tree,
                 edges=[("Root", "Branch_1"), ("Root", "Branch_2"),
                        ("Branch_2", "Branch_3")]),
}

# n_clusters=1 windows mix branches past the split; label windows cannot.
WINDOWS = {"agnostic": None, "labels": "clusters_id"}


def build(topology, data_seed=0):
    ad = TOPOLOGIES[topology]["build"](data_seed)
    ad.obs["clusters_id"] = ad.obs["celltype"].cat.codes
    ad.obs["pseudotime"] = ad.obs["true_pseudotime"].values
    return ad


# ---------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------
def score(adata, V, truth, celltype, edges):
    """Truth-based scores plus the two benchmark metrics, same field."""
    c = cosine_rows(V, truth)
    ok = ~np.isnan(c)
    row = dict(coverage=float(ok.mean()),
               cos_mean=float(np.nanmean(c)),
               cos_median=float(np.nanmedian(c)),
               frac_pos=float(np.mean(c[ok] > 0)) if ok.any() else np.nan)
    for ct in np.unique(celltype):
        row[f"cos_{ct}"] = float(np.nanmean(c[celltype == ct]))

    # the benchmark metrics, on the very same vectors, so the two can be
    # compared cell for cell rather than across separate runs
    adata.obsm["_scoring"] = V
    m = velot.metrics.summary(adata, cluster_edges=edges,
                              cluster_key="celltype", embedding_key="X_pca",
                              velocity_key="_scoring")
    del adata.obsm["_scoring"]
    row.update(cbdir_mean=m["cbdir_mean"], iccoh_mean=m["iccoh_mean"])
    return row


def run(topologies, windowings, methods, seeds, data_seed):
    rows = []
    for topology in topologies:
        edges = TOPOLOGIES[topology]["edges"]
        for windowing in windowings:
            base = build(topology, data_seed)
            truth = base.obsm["true_velocity_pca"]
            celltype = base.obs["celltype"].astype(str).values
            for method in methods:
                for seed in seeds:
                    ad = base.copy()
                    velot.tl.velocity(
                        ad, **{**SHARED, **METHODS[method]},
                        spatial_key=WINDOWS[windowing], random_state=seed)
                    for field, key in (("raw", "velot_velocity_raw_pca"),
                                       ("smooth", "velot_velocity_pca")):
                        rows.append(dict(
                            dataset=topology, windowing=windowing,
                            method=method, seed=seed, field=field,
                            **score(ad, ad.obsm[key], truth, celltype, edges)))
                    print(f"  {topology}/{windowing}/{method}/seed{seed}: "
                          f"raw cos {rows[-2]['cos_mean']:.3f} -> "
                          f"smooth {rows[-1]['cos_mean']:.3f}", flush=True)
    return pd.DataFrame(rows)


def main(a):
    methods = sorted(METHODS) if a.all_methods else a.methods
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise SystemExit(f"unknown method(s): {sorted(unknown)}; "
                         f"available: {sorted(METHODS)}")
    OUT.mkdir(parents=True, exist_ok=True)
    runs = run(a.topologies, a.windowings, methods, a.seeds, a.data_seed)
    runs.to_csv(OUT / "runs.csv", index=False)

    keys = ["dataset", "windowing", "method", "field"]
    metrics = ["cos_mean", "cos_median", "frac_pos", "cbdir_mean",
               "iccoh_mean", "coverage"]
    summary = (runs.melt(id_vars=keys + ["seed"], value_vars=metrics,
                         var_name="metric", value_name="value")
               .groupby(keys + ["metric"])["value"]
               .agg(mean="mean", sd="std", n="count").reset_index())
    summary["sd"] = summary["sd"].fillna(0.0)
    summary.to_csv(OUT / "summary.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\n{len(runs)} rows -> {OUT}/runs.csv, summary.csv\n")
    piv = (summary[summary.metric.isin(["cos_mean", "cbdir_mean", "iccoh_mean"])]
           .pivot_table(index=["dataset", "windowing", "method"],
                        columns=["field", "metric"], values="mean"))
    print("Mean over seeds. cos_mean is the truth-based score; the other "
          "two are the benchmark metrics on the same vectors.")
    print(piv.round(3).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-t", "--topologies", nargs="+", default=sorted(TOPOLOGIES),
                   choices=sorted(TOPOLOGIES))
    p.add_argument("-w", "--windowings", nargs="+", default=sorted(WINDOWS),
                   choices=sorted(WINDOWS))
    p.add_argument("-m", "--methods", nargs="+", default=DEFAULT_METHODS)
    p.add_argument("-s", "--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                   help="smoother random_state, as in the real benchmark")
    p.add_argument("--data-seed", type=int, default=0,
                   help="seed of the data generator; fixed so that the "
                        "ground truth is the same across the sweep")
    p.add_argument("--all-methods", action="store_true",
                   help="every entry of the benchmark METHODS dict")
    main(p.parse_args())

"""
Figures for the benchmark, built from the saved AnnData objects.

    python run_plots.py                         # every dataset, every method
    python run_plots.py -d pancreas             # one dataset
    python run_plots.py --figures metrics       # iterate on one figure

Deliberately separate from run_velot.py: the figures are rebuilt from the
h5ad files that run already wrote, so changing a plot never means redoing
an experiment. It needs the runs that were saved with ``--save-adata``
(run_all.sh does that for seed 0 of every method).

Four figures per dataset, written to ``figures/<dataset>_*.png``:

``overview``  the embedding coloured by cluster and by pseudotime.
              Depends on the dataset, not on the method.
``windows``   the window pairs OT actually transported between. Also
              method-independent: windows come from the pseudotime and
              the window size. They are REBUILT here rather than read
              back from the h5ad - window pairs are ragged arrays and do
              not reliably survive the round trip.
``streams``   one row per method: raw and smoothed field, as a stream
              plot and as arrows.
``metrics``   one row per method, ICCoh and CBDir, with the raw and the
              smoothed field as grouped boxes.

The metrics figure recomputes the scores from the AnnData instead of
reading runs.csv, and uses the same basis, edges and keys run_velot.py
used. That is the point: two independent paths to the same numbers. If
the figure and the CSV disagree, one of them is wrong, and that is worth
knowing before either goes into the paper.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scanpy as sc
import velot

os.chdir(Path(__file__).resolve().parent)
import datasets as ds                                        # noqa: E402
from run_velot import SHARED, METHODS, _cosine_to_truth      # noqa: E402

sc.settings.verbosity = 0

RESULTS = "results"
FIGURES = "figures"
ALL_FIGURES = ("overview", "windows", "streams", "metrics")


# ---------------------------------------------------------------------
# locating the runs
# ---------------------------------------------------------------------
def _model(method, seed, tag=None):
    return method + (f"_{tag}" if tag else "") + f"_seed{seed}_smooth"


def _paths(results_dir, dataset, method, seed, tag):
    stem = f"{_model(method, seed, tag)}_{dataset}"
    return (Path(results_dir) / "data" / f"{stem}.h5ad",
            Path(results_dir) / f"{stem}.json")


def _load(results_dir, dataset, method, seed, tag):
    h5, js = _paths(results_dir, dataset, method, seed, tag)
    if not h5.exists():
        return None, None
    adata = sc.read_h5ad(h5)
    record = json.load(open(js)) if js.exists() else {}
    return adata, record.get("extra", {})


def _keys(cfg):
    """(plotting basis name, metric basis, metric key suffix)."""
    pbasis = cfg.get("project_basis", "X_umap")
    if pbasis not in ("X_umap", "X_tsne") and pbasis != cfg["basis"]:
        pbasis = cfg["basis"]
    return pbasis.split("X_", 1)[1], cfg["basis"], cfg["basis"].split("X_", 1)[1]


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    {path}")


# ---------------------------------------------------------------------
# the four figures
# ---------------------------------------------------------------------
def fig_overview(adata, cfg, dataset, out):
    pname, _, _ = _keys(cfg)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    velot.pl.dataset_overview_simple(
        adata, color=cfg["clusters_key"], basis=pname,
        title="Clusters", inframe=True, show=False, ax=axes[0])
    velot.pl.dataset_overview_simple(
        adata, color="pseudotime", basis=pname,
        title=f"Pseudotime ({cfg.get('pseudotime_key') or 'DPT'})",
        show=False, ax=axes[1])
    fig.suptitle(f"{dataset} — {adata.n_obs} cells, basis {cfg['basis']}",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save(fig, f"{out}/{dataset}_overview.png")


def fig_windows(adata, cfg, dataset, out, extra):
    pname, _, _ = _keys(cfg)
    # rebuild rather than trust the h5ad: ragged pairs do not serialise
    velot.tl.build_windows(
        adata, basis=cfg["basis"],
        window_size=int(extra.get("window_size") or cfg["window_size"]),
        n_clusters=extra.get("n_clusters", 1),
        spatial_key=extra.get("spatial_key"),
        min_window_size=SHARED["min_window_size"],
        overlap_fraction=SHARED["overlap_fraction"],
        tail_handling=SHARED["tail_handling"],
        tail_threshold=SHARED["tail_threshold"])
    n = adata.uns["velot_windows"]["n_pairs"]
    velot.pl.windows(
        adata, basis=pname, pairs_to_show=tuple(range(n)), max_show=n,
        ncols=4, pair_title=True, figsize_per_panel=(4, 4), show=False,
        save=True, save_path=f"{out}/{dataset}_windows.png")
    plt.close("all")
    print(f"    {out}/{dataset}_windows.png  ({n} pairs)")


def fig_streams(runs, cfg, dataset, out):
    pname, _, _ = _keys(cfg)
    nrow = len(runs)
    fig, axes = plt.subplots(nrow, 4, figsize=(20, 5 * nrow), squeeze=False)
    for i, (method, adata, _) in enumerate(runs):
        for k, field in enumerate(("raw", "smooth")):
            key = "velot_velocity_raw" if field == "raw" else "velot_velocity"
            velot.pl.velocity_stream(
                adata, color=cfg["clusters_key"], basis=pname,
                velocity_key=key, title=f"{field} stream",
                show=False, ax=axes[i, k])
            velot.pl.velocity_quiver(
                adata, color=cfg["clusters_key"], basis=pname, subsample=500,
                velocity_key=f"{key}_{pname}", title=f"{field} arrows",
                show=False, ax=axes[i, k + 2])
        axes[i, 0].annotate(method, xy=(-0.18, 0.5), xycoords="axes fraction",
                            rotation=90, ha="center", va="center",
                            fontsize=17, fontweight="bold",
                            annotation_clip=False)
    fig.suptitle(f"{dataset} — velocity field by method",
                 fontsize=20, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    _save(fig, f"{out}/{dataset}_streams.png")


def fig_metrics(runs, cfg, dataset, out):
    _, mbasis, suffix = _keys(cfg)
    truth_key = cfg.get("truth_key")
    nrow = len(runs)
    # datasets with a known velocity field get a third panel
    has_truth = bool(truth_key) and all(
        truth_key in a.obsm for _, a, _ in runs)
    ncol = 3 if has_truth else 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 5 * nrow),
                             squeeze=False)
    printed = []
    for i, (method, adata, _) in enumerate(runs):
        labels = adata.obs[cfg["clusters_key"]].astype(str).values
        scored = {}
        for field, key in (("raw", f"velot_velocity_raw_{suffix}"),
                           ("smooth", f"velot_velocity_{suffix}")):
            res = velot.metrics.summary(
                adata, cluster_edges=cfg["edges"],
                cluster_key=cfg["clusters_key"], embedding_key=mbasis,
                velocity_key=key, print_results=False)
            if has_truth:
                # recomputed here, exactly as run_velot.py computes it, so
                # the figure stays an independent check on the CSV
                res.update(_cosine_to_truth(
                    adata.obsm[key], adata.obsm[truth_key], labels))
            scored[field] = res
        velot.pl.metric_summary(scored, orientation="vertical",
                                ax=axes[i], legend=(i == 0))
        axes[i, 0].annotate(method, xy=(-0.20, 0.5), xycoords="axes fraction",
                            rotation=90, ha="center", va="center",
                            fontsize=16, fontweight="bold",
                            annotation_clip=False)
        printed.append((method, scored["raw"], scored["smooth"]))
    title = "ICCoh, CBDir and cosine to truth" if has_truth \
        else "ICCoh and CBDir"
    fig.suptitle(f"{dataset} — {title}, raw against smoothed",
                 fontsize=19, fontweight="bold", y=0.997)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    _save(fig, f"{out}/{dataset}_metrics.png")

    # echo the numbers so they can be diffed against runs.csv by eye
    cols = ["iccoh_mean", "cbdir_mean"] + (["cos_mean"] if has_truth else [])
    print("    recomputed from the h5ad (compare with runs_wide.csv):")
    header = "".join(f"{c.replace('_mean',''):>11s}{'':>11s}" for c in cols)
    print(f"      {'method':18s} " + "".join(
        f"{c.replace('_mean','')+'_raw':>12s}{c.replace('_mean','')+'_smo':>12s}"
        for c in cols))
    for m, raw, smo in printed:
        vals = "".join(f"{raw[c]:12.4f}{smo[c]:12.4f}" for c in cols)
        print(f"      {m:18s} " + vals)


# ---------------------------------------------------------------------
def main(a):
    datasets = a.datasets or sorted(ds.DATASETS)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    missing = []

    for dataset in datasets:
        cfg = ds.DATASETS[dataset]
        runs = []
        for method in a.methods:
            adata, extra = _load(a.results_dir, dataset, method, a.seed, a.tag)
            if adata is None:
                missing.append(f"{dataset}/{method}")
                continue
            runs.append((method, adata, extra))
        if not runs:
            print(f"\n{dataset}: no saved AnnData - skipped")
            continue

        print(f"\n{dataset}  ({len(runs)} method(s), seed {a.seed})")
        base_adata, base_extra = runs[0][1], runs[0][2]
        if "overview" in a.figures:
            fig_overview(base_adata, cfg, dataset, a.out_dir)
        if "windows" in a.figures:
            fig_windows(base_adata.copy(), cfg, dataset, a.out_dir, base_extra)
        if "streams" in a.figures:
            fig_streams(runs, cfg, dataset, a.out_dir)
        if "metrics" in a.figures:
            fig_metrics(runs, cfg, dataset, a.out_dir)

    if missing:
        print(f"\n{len(missing)} (dataset, method) pair(s) had no saved "
              f"AnnData at seed {a.seed}:")
        for m in missing[:20]:
            print("   ", m)
        print("   run run_all.sh with --save-adata for those, or pass "
              "-m with only the methods you saved.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-d", "--datasets", nargs="+", default=None,
                   choices=sorted(ds.DATASETS))
    p.add_argument("-m", "--methods", nargs="+",
                   default=["velot", "velot_gradient", "velot_graph", "velot_unbalanced"],
                   choices=sorted(METHODS))
    p.add_argument("-s", "--seed", type=int, default=0)
    p.add_argument("--tag", default=None)
    p.add_argument("--figures", nargs="+", default=list(ALL_FIGURES),
                   choices=list(ALL_FIGURES))
    p.add_argument("--results-dir", default=RESULTS)
    p.add_argument("--out-dir", default=FIGURES)
    main(p.parse_args())

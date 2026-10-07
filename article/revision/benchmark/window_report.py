"""
Window-size report: run tl.window_diagnostics on every dataset, with the
same preprocessing, the same ordering and the same unsupervised windowing
the benchmark uses (one spatial cluster, no labels), and show which size
the rule picks.

    python window_report.py [-d pancreas erythroid ...]
    python window_report.py -d bifurcation --pseudotime-key sim_time

Windows are built by sorting cells on the pseudotime, so the window size
depends on WHICH pseudotime. A dataset carrying ``pseudotime_key`` in the
registry is measured with that ordering; everything else is measured with
DPT from the dataset's root cluster, exactly as run_velot.py does it.
Measuring a dataset under one ordering and running it under another gives
window sizes that do not correspond to the run.

Prints one table per dataset and writes window_report.csv, keeping the
rows for datasets this call did not re-measure. Put the chosen value into
datasets.py so the run is reproducible.
"""
from __future__ import annotations
import argparse, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import scanpy as sc, velot
os.chdir(Path(__file__).resolve().parent)
import datasets as ds
sc.settings.verbosity = 0

CANDIDATES = (50, 100, 200, 300, 500, 800)


def _order(adata, cfg, a):
    """Apply the same pseudotime run_velot.py would. Returns a label for
    the ordering and the leading diffusion eigenvalue when DPT was used."""
    pkey = a.pseudotime_key or cfg.get("pseudotime_key")
    if pkey:
        velot.pp.pseudotime(adata, key=pkey)
        return f"key:{pkey}", np.nan

    velot.pp.pseudotime(
        adata,
        root_cluster=cfg["root_cluster"],
        cluster_key=cfg["clusters_key"],
        root_selection=a.root_selection,
        basis=cfg["basis"],
        dpt_impl=a.dpt_impl,
    )
    root = adata.uns["velot_root"]
    # lambda1 >= 0.9994 is where scanpy's sc.tl.dpt stops weighting the
    # leading component properly; surfacing it here means a dataset that
    # trips it is visible at the moment its windows are measured.
    lam1 = float(np.asarray(adata.uns["diffmap_evals"])[1])
    return f"dpt:{root['cluster']}->{root['cell_name']}", lam1


def main(a):
    out = []
    for name in a.datasets:
        adata, cfg = ds.load(name)
        source, lam1 = _order(adata, cfg, a)
        note = ""
        if np.isfinite(lam1) and lam1 >= 0.9994:
            note = f"  [!] lambda1={lam1:.5f} >= 0.9994 - scanpy's dpt would break here"
        print(f"\n=== {name}  ({adata.n_obs} cells, {cfg['n_pcs']} PCs, "
              f"basis {cfg['basis']}, ordering {source}){note}")
        choice, table = velot.tl.choose_window_size(
            adata, basis=cfg["basis"],
            candidates=CANDIDATES, min_window_size=50,
            n_clusters=1, spatial_key=None,
            tail_handling="drop", tail_threshold=20, verbose=True)
        table = table.reset_index()
        table.insert(0, "dataset", name)
        table["chosen"] = table["window_size"] == choice
        table["n_cells"] = adata.n_obs
        table["pseudotime"] = source
        table["lambda1"] = lam1
        out.append(table)

    df = pd.concat(out)
    if os.path.exists("window_report.csv"):
        prev = pd.read_csv("window_report.csv")
        df = pd.concat([prev[~prev.dataset.isin(df.dataset.unique())], df])
    df = df.sort_values(["dataset", "window_size"])
    # rows carried over from an earlier run predate these columns
    df["pseudotime"] = df.get("pseudotime", pd.Series(dtype=object)).fillna("not recorded")
    df.to_csv("window_report.csv", index=False)

    print("\nchosen:")
    for d, g in df.groupby("dataset"):
        sel = g.loc[g.chosen]
        if not len(sel):
            print(f"  {d:22s} no eligible window"); continue
        print(f"  {d:22s} {int(sel['window_size'].iloc[0]):5d}   "
              f"({sel['pseudotime'].iloc[0]})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-d", "--datasets", nargs="+", default=sorted(ds.DATASETS))
    p.add_argument("--pseudotime-key", default=None,
                   help="obs column holding a precomputed ordering; "
                        "overrides the dataset's own pseudotime_key")
    p.add_argument("--root-selection", default="distal",
                   choices=("distal", "medoid", "first"))
    p.add_argument("--dpt-impl", default="velot", choices=("velot", "scanpy"))
    main(p.parse_args())

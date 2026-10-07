"""
Collect the revision benchmark JSONs into two tables.

    python aggregate.py [--dir benchmark_results/revision]

Writes next to them:

``runs.csv``
    one row per run: dataset, method, tag, seed, field (raw/smooth),
    the metrics, and the run's provenance (window size, root cell and
    rule, coverage, plan sharpness, transported mass, runtime).
``summary.csv``
    mean and sd across seeds per (dataset, method, tag, field, metric) —
    what the benchmark figure needs, and what R1.1(b) asked for.

Model names are ``<method>[_<tag>]_seed<N>_<field>``; the tag is what the
sweeps set (``rootcluster-Alpha``, ``sup-kmeans``, ...), so the same two
tables cover the benchmark, the root sweep and the supervision runs.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

NAME = re.compile(r"^(?P<method>velot(?:_[a-z_]+)??)"
                  r"(?:_(?P<tag>[^/]*?))?_seed(?P<seed>\d+)_"
                  r"(?P<field>raw|smooth)$")
KNOWN = ("velot_original", "velot_gradient", "velot_no_knn", "velot_cost_nn",
         "velot_argmax", "velot_unbalanced", "velot_geodesic", "velot_normraw",
         "velot")


def _split(model):
    """<method>[_<tag>]_seed<N>_<field> -> (method, tag, seed, field)."""
    m = re.match(r"^(?P<rest>.+)_seed(?P<seed>\d+)_(?P<field>raw|smooth)$",
                 model)
    if not m:
        return None
    rest, seed, field = m["rest"], int(m["seed"]), m["field"]
    method = next((k for k in KNOWN if rest == k or rest.startswith(k + "_")),
                  rest)
    tag = rest[len(method) + 1:] if len(rest) > len(method) else ""
    return method, tag, seed, field


def collect(directory):
    rows = []
    bad = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        try:
            d = json.load(open(path))
        except (json.JSONDecodeError, OSError) as err:
            bad.append((os.path.basename(path), err))
            continue
        parts = _split(d["model"])
        if parts is None:                      # not a revision run
            continue
        method, tag, seed, field = parts
        e, m = d.get("extra", {}), d["metrics"]
        root = e.get("root") or {}
        rows.append(dict(
            dataset=d["dataset"], method=method, tag=tag, seed=seed,
            field=field,
            iccoh_mean=m["iccoh_mean"], iccoh_median=m["iccoh_median"],
            cbdir_mean=m["cbdir_mean"], cbdir_median=m["cbdir_median"],
            # only datasets with a known field (truth_key) have these
            cos_mean=m.get("cos_mean"), cos_median=m.get("cos_median"),
            cos_frac_pos=m.get("cos_frac_pos"),
            coverage=e.get("coverage"),
            window_size=e.get("window_size"),
            root_cell=root.get("cell"), root_cluster=root.get("cluster"),
            root_selection=root.get("selection"),
            spatial_key=e.get("spatial_key"), n_clusters=e.get("n_clusters"),
            pseudotime_key=e.get("pseudotime_key"), dpt_impl=e.get("dpt_impl"),
            n_eff=e.get("mean_n_eff"), T_eff=e.get("mean_T_eff"),
            typical_move=e.get("typical_move_median"),
            transported_mass=e.get("mean_transported_mass"),
            n_cells=e.get("n_cells"),
            seconds=d["timing"].get("velocity"),
        ))
    if bad:
        print(f"skipped {len(bad)} unreadable file(s):")
        for name, err in bad[:10]:
            print(f"  {name}: {err}")
        if len(bad) > 10:
            print(f"  ... and {len(bad) - 10} more")
    return pd.DataFrame(rows)


def widen(runs):
    """One row per run with raw and smoothed side by side.

    runs.csv is tidy (one row per field) which is what the plots want;
    this is the shape you want when checking a run by eye, because the
    smoother's contribution is a subtraction across two columns rather
    than across two rows.

    Only the identity columns index the pivot. Provenance columns such
    as ``seconds`` or ``n_eff`` differ between the raw and the smoothed
    row of the same run, so indexing on them would multiply the rows out
    instead of pairing them; they are merged back from the smoothed row.
    """
    metrics = ["iccoh_mean", "iccoh_median", "cbdir_mean", "cbdir_median",
               "cos_mean", "cos_median", "cos_frac_pos"]
    metrics = [c for c in metrics if c in runs.columns]
    ident = [c for c in ("dataset", "method", "tag", "seed")
             if c in runs.columns]

    runs = runs.copy()
    if "tag" in runs:
        runs["tag"] = runs["tag"].fillna("")

    wide = runs.pivot_table(index=ident, columns="field", values=metrics)
    wide.columns = [f"{m}_{f}" for m, f in wide.columns]
    wide = wide.reset_index()

    for m in ("iccoh_mean", "cbdir_mean", "cos_mean"):
        if f"{m}_raw" in wide and f"{m}_smooth" in wide:
            wide[f"{m}_gain"] = wide[f"{m}_smooth"] - wide[f"{m}_raw"]

    # provenance: one row per run, taken from the smoothed record
    prov = runs[runs["field"] == "smooth"].drop(
        columns=[c for c in metrics + ["field"] if c in runs.columns])
    prov = prov.drop_duplicates(subset=ident)
    wide = wide.merge(prov, on=ident, how="left")

    front = ["dataset", "method", "tag", "seed",
             "cos_mean_raw", "cos_mean_smooth", "cos_mean_gain",
             "iccoh_mean_raw", "iccoh_mean_smooth", "iccoh_mean_gain",
             "cbdir_mean_raw", "cbdir_mean_smooth", "cbdir_mean_gain"]
    front = [c for c in front if c in wide.columns]
    return wide[front + [c for c in wide.columns if c not in front]]


def summarize(runs):
    metrics = ["iccoh_mean", "iccoh_median", "cbdir_mean", "cbdir_median",
               "cos_mean", "cos_median", "cos_frac_pos", "coverage"]
    long = runs.melt(
        id_vars=["dataset", "method", "tag", "field", "seed"],
        value_vars=metrics, var_name="metric", value_name="value")
    g = long.groupby(["dataset", "method", "tag", "field", "metric"],
                     dropna=False)["value"]
    out = g.agg(mean="mean", sd="std", n="count").reset_index()
    out["sd"] = out["sd"].fillna(0.0)
    return out


def main(directory):
    runs = collect(directory)
    if runs.empty:
        print(f"no revision runs found in {directory}")
        return
    runs = runs.sort_values(["dataset", "method", "tag", "field", "seed"])
    runs.to_csv(os.path.join(directory, "runs.csv"), index=False)
    summary = summarize(runs)
    summary.to_csv(os.path.join(directory, "summary.csv"), index=False)
    wide = widen(runs)
    wide.to_csv(os.path.join(directory, "runs_wide.csv"), index=False)

    print(f"{len(runs)} rows -> runs.csv ({len(wide)} runs -> runs_wide.csv), "
          f"summary.csv\n")
    main_tbl = summary[(summary.tag == "") & (summary.field == "smooth")
                       & summary.metric.isin(["iccoh_mean", "cbdir_mean"])]
    if len(main_tbl):
        piv = main_tbl.pivot_table(index=["dataset", "method"],
                                   columns="metric", values=["mean", "sd"])
        print("Smoothed field, mean +- sd across seeds")
        print(piv.round(3).to_string())
    seeds = runs.groupby(["dataset", "method", "tag", "field"])["seed"].nunique()
    thin = seeds[seeds < 5]
    if len(thin):
        print(f"\n{len(thin)} groups have fewer than 5 seeds "
              f"(sweeps are single-seed by design).")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default="benchmark_results/revision")
    main(p.parse_args().dir)

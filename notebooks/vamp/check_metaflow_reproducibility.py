"""
Is the published pancreas MetaFlow partition reproducible under the
revised VelOT pipeline?  (R1.6 part 2, m8)

The article's Figure "VAMPFlow meta-states" was built from one MetaFlow
run. The revision re-runs VelOT with a different root rule, no window
overlap and a single spatial group, so the question is whether the
meta-state partition survives that change.

Meta-state numbers are arbitrary across runs - the VAMP network is
randomly initialised, so M3 in one run has nothing to do with M3 in
another. Everything below therefore matches states with the Hungarian
algorithm on the contingency table before counting anything.

    python check_metaflow_reproducibility.py

Writes metaflow_reproducibility/ with the comparison table and, if the
comparison module is importable, the full cell-type contingency for the
new run so it can be set beside the published one.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

HERE = os.chdir(Path(__file__).resolve().parent)

# the new reference run
NEW_H5AD = Path("results/vamp_robustness_pancreas2/reference_pancreas_vampflow.h5ad")
# the per-cell assignments the article figure was built from
OLD_CELLS = Path("../supplementary/metastate_celltype_comparison_pancreas/cell_data.csv")
OUT = Path("metaflow_reproducibility")
CELL_TYPE_KEY = "clusters"
OUT.mkdir(exist_ok=True)


def find_state_key(adata):
    """The meta-state column, whatever this version of the module called it."""
    for k in ("velot_vampflow_state", "velot_metaflow_state",
              "velot_quantum_metaflow_state", "metastate", "meta_state"):
        if k in adata.obs:
            return k
    cand = [c for c in adata.obs.columns if "state" in c.lower()]
    raise SystemExit(f"no meta-state column found. Candidates: {cand}")


print(f"[load] {NEW_H5AD}")
ad = sc.read_h5ad(NEW_H5AD)
state_key = find_state_key(ad)
print(f"       meta-state column: {state_key!r}   cells: {ad.n_obs}")

new = pd.Series(ad.obs[state_key].astype(str).values, index=ad.obs_names, name="new")
old = pd.read_csv(OLD_CELLS, index_col=0)["meta_state"].astype(str).rename("old")

both = pd.concat([old, new], axis=1, join="inner").dropna()
print(f"[match] {len(both)} cells in common "
      f"(published run {len(old)}, new run {len(new)})")
if len(both) == 0:
    raise SystemExit("no shared cell barcodes - the two runs are not comparable "
                     "cell by cell; compare the contingency tables instead.")

# ---- how similar are the two partitions, numbering aside ------------
ari = adjusted_rand_score(both["old"], both["new"])
nmi = normalized_mutual_info_score(both["old"], both["new"])
print(f"\n=== published partition vs new partition ===")
print(f"  states: {both['old'].nunique()} published, {both['new'].nunique()} new")
print(f"  ARI {ari:.3f}   NMI {nmi:.3f}")

# Hungarian match on the contingency, then count cells that keep their state
ct = pd.crosstab(both["old"], both["new"])
r, c = linear_sum_assignment(-ct.values)
mapping = {ct.index[i]: ct.columns[j] for i, j in zip(r, c)}
kept = sum(ct.values[i, j] for i, j in zip(r, c))
print(f"  best one-to-one matching keeps {kept}/{len(both)} cells "
      f"({kept/len(both):.1%})")
pd.Series(mapping, name="new_state").rename_axis("published_state").to_csv(
    OUT / "state_matching.csv")
ct.to_csv(OUT / "contingency_published_vs_new.csv")

# ---- how well does each run track the cell types --------------------
print(f"\n=== agreement with cell types (the published run scored "
      f"ARI 0.444 / NMI 0.640) ===")
if CELL_TYPE_KEY in ad.obs:
    types = ad.obs[CELL_TYPE_KEY].astype(str)
    for name, lab in (("published", old.reindex(ad.obs_names)),
                      ("new      ", new)):
        ok = lab.notna()
        print(f"  {name}: ARI {adjusted_rand_score(types[ok.values], lab[ok]):.3f}"
              f"   NMI {normalized_mutual_info_score(types[ok.values], lab[ok]):.3f}")
    cont = pd.crosstab(new, types)
    cont.to_csv(OUT / "contingency_new_by_celltype.csv")
    print(f"\n=== new run: cell-type composition of each meta-state ===")
    frac = cont.div(cont.sum(axis=1), axis=0)
    summary = pd.DataFrame({
        "n_cells": cont.sum(axis=1),
        "dominant_cell_type": frac.idxmax(axis=1),
        "dominant_fraction": frac.max(axis=1).round(3),
    }).sort_values("n_cells", ascending=False)
    summary.to_csv(OUT / "new_state_summary.csv")
    print(summary.to_string())
    mixed = summary[summary["dominant_fraction"] < 0.8]
    print(f"\n  {len(mixed)} of {len(summary)} states are below 80% pure "
          f"({', '.join(mixed.index)})" if len(mixed) else "\n  all states >=80% pure")
else:
    print(f"  '{CELL_TYPE_KEY}' not in obs; columns: {list(ad.obs.columns)[:12]}")

print(f"\n[done] wrote {OUT}/")

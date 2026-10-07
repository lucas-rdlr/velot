"""
Do the new MetaFlow meta-states correspond to known cell types?  (R1.6, m8)

This looks at the NEW reference run on its own terms. Meta-state numbers
are arbitrary - the VAMP network is randomly initialised, so M3 here has
nothing to do with M3 in any earlier run, and no attempt is made to match
them. The only question is whether each state is a recognisable
population.

Outputs, in metaflow_celltypes/:
    contingency.csv      states x cell types, counts
    state_summary.csv    per state: n, dominant type, purity, enrichment
    enrichment.csv       log2 observed/expected and Fisher BH-corrected q
and prints the same as a table.

    python check_metaflow_celltypes.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import fisher_exact
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

HERE = os.chdir(Path(__file__).resolve().parent)

H5AD = Path("results/vamp_robustness_pancreas2/reference_pancreas_vampflow.h5ad")
CELL_TYPE_KEY = "clusters"
OUT = Path("metaflow_celltypes")
# optional: per-cell states of an earlier run, only as a side note
OLD_CELLS = Path("../supplementary/metastate_celltype_comparison_pancreas/cell_data.csv")

OUT.mkdir(exist_ok=True)


def find_key(adata, names, what):
    for k in names:
        if k in adata.obs:
            return k
    raise SystemExit(f"no {what} column. obs has: {list(adata.obs.columns)}")


print(f"[load] {H5AD}")
ad = sc.read_h5ad(H5AD)
state_key = find_key(ad, ("velot_vampflow_state", "velot_metaflow_state",
                          "velot_quantum_metaflow_state", "metastate",
                          "meta_state"), "meta-state")
ct_key = find_key(ad, (CELL_TYPE_KEY, "clusters", "cell_type", "celltype"),
                  "cell-type")
states = ad.obs[state_key].astype(str)
types = ad.obs[ct_key].astype(str)
print(f"       states: {state_key!r} ({states.nunique()} distinct)   "
      f"cell types: {ct_key!r} ({types.nunique()})   cells: {ad.n_obs}")

# ---- contingency ----------------------------------------------------
cont = pd.crosstab(states, types)
cont.to_csv(OUT / "contingency.csv")

frac = cont.div(cont.sum(axis=1), axis=0)
N = cont.values.sum()
exp = np.outer(cont.sum(axis=1), cont.sum(axis=0)) / N
log2oe = pd.DataFrame(np.log2((cont.values + 0.5) / (exp + 0.5)),
                      index=cont.index, columns=cont.columns)

# Fisher per cell of the table, BH corrected, as the article reports
pv = np.ones(cont.shape)
for i in range(cont.shape[0]):
    for j in range(cont.shape[1]):
        a = cont.values[i, j]
        tab = [[a, cont.values[i].sum() - a],
               [cont.values[:, j].sum() - a,
                N - cont.values[i].sum() - cont.values[:, j].sum() + a]]
        pv[i, j] = fisher_exact(tab, alternative="greater")[1]
flat = pv.ravel()
order = np.argsort(flat)
q = np.empty_like(flat)
q[order] = np.minimum.accumulate(
    (flat[order] * len(flat) / (np.arange(len(flat)) + 1))[::-1])[::-1]
qv = pd.DataFrame(q.reshape(pv.shape), index=cont.index, columns=cont.columns)
pd.concat({"log2_obs_exp": log2oe, "q_value": qv}, axis=1).to_csv(
    OUT / "enrichment.csv")

summary = pd.DataFrame({
    "n_cells": cont.sum(axis=1),
    "dominant_cell_type": frac.idxmax(axis=1),
    "purity": frac.max(axis=1).round(3),
    "n_enriched_q<0.05": (qv < 0.05).sum(axis=1),
    "enriched_types": [", ".join(qv.columns[(qv < 0.05).loc[s]])
                       for s in cont.index],
}).sort_values("n_cells", ascending=False)
summary.to_csv(OUT / "state_summary.csv")

print("\n=== every meta-state, by the cell types it contains ===")
print(summary.to_string())

pure = (summary["purity"] >= 0.8).sum()
print(f"\n  {pure}/{len(summary)} states are >=80% one cell type; "
      f"{(summary['purity'] >= 0.9).sum()} are >=90%")
print(f"  agreement with the cell-type labels: "
      f"ARI {adjusted_rand_score(types, states):.3f}   "
      f"NMI {normalized_mutual_info_score(types, states):.3f}")

print("\n=== which cell types got their own state(s) ===")
for t in cont.columns:
    owners = summary.index[summary["dominant_cell_type"] == t].tolist()
    tot = cont[t].sum()
    got = cont.loc[owners, t].sum() if owners else 0
    print(f"  {t:16s} {tot:5d} cells -> "
          f"{'states ' + ', '.join(owners) if owners else 'NO dedicated state'}"
          f"  ({got/tot:.0%} of them)" if tot else f"  {t}: 0 cells")

# ---- side note only: how it compares with an earlier run ------------
if OLD_CELLS.exists():
    old = pd.read_csv(OLD_CELLS, index_col=0)["meta_state"].astype(str)
    both = pd.concat([old.rename("old"),
                      pd.Series(states.values, index=ad.obs_names, name="new")],
                     axis=1, join="inner").dropna()
    if len(both):
        print(f"\n[side note] against the earlier run ({len(both)} shared cells): "
              f"ARI {adjusted_rand_score(both['old'], both['new']):.3f}. "
              "Not a target - the pipeline changed and state numbering is "
              "arbitrary; reported only so the difference is on record.")

print(f"\n[done] wrote {OUT}/")

"""
R1.5: are MetaFlow terminal states stable when the field is restricted to
cells with raw-transport support?

Reviewer 1 asks us to *show* this, not argue it. MetaFlow is run twice
with everything else held fixed - same latent space, same seed, K fixed
at the same value - on

    full        all cells, as reported
    confident   only cells that received a raw OT vector

and the two are compared on the cells they share. Meta-state numbers are
arbitrary between runs, so the partitions are matched with the Hungarian
algorithm before anything is counted, and terminal designation is taken
from the state labels rather than from state indices.

    python run_metaflow_confident.py

Writes metaflow_confident/ and prints the agreement table.
Expect roughly 8-10 minutes for the two arms.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

os.chdir(Path(__file__).resolve().parent)

from run_pancreas_vamp_robustness import (  # noqa: E402
    ROBUST_CFG,
    ensure_basic_preprocessing_if_needed,
    ensure_pseudotime_if_missing,
    ensure_velot_velocity_key,
    infer_cell_type_key,
    infer_stemness_key,
    infer_cycle_key,
    pick_first_existing_obs_key,
    run_single_vampflow,
)

ADATA = Path("pancreas_velot.h5ad")
OUT = Path("metaflow_confident")
N_METASTATES = 12
SEED = 0
RAW_KEYS = ("velot_velocity_raw_umap", "velot_velocity_raw")
OUT.mkdir(exist_ok=True)

print(f"[load] {ADATA}")
adata = sc.read_h5ad(ADATA)
adata = ensure_basic_preprocessing_if_needed(adata)
smooth_key = ensure_velot_velocity_key(adata)
pseudotime_key = ensure_pseudotime_if_missing(
    adata, pick_first_existing_obs_key(
        adata, ["velot_pseudotime", "pseudotime", "dpt_pseudotime",
                "palantir_pseudotime", "latent_time"]))
cell_type_key = infer_cell_type_key(adata)

# ---- which cells have raw transport support ------------------------
if "velot_confidence" in adata.obs:
    confident = np.asarray(adata.obs["velot_confidence"], dtype=float) > 0
    basis = "obs['velot_confidence'] > 0"
else:
    raw_key = next((k for k in RAW_KEYS if k in adata.obsm), None)
    if raw_key is None:
        raise SystemExit(f"no confidence and no raw field; obsm: {sorted(adata.obsm)}")
    nv = np.linalg.norm(np.asarray(adata.obsm[raw_key]), axis=1)
    confident = np.isfinite(nv) & (nv > 0)
    basis = f"nonzero {raw_key}"
print(f"[confident] {confident.sum()}/{adata.n_obs} cells "
      f"({confident.mean():.1%}) by {basis}")

keys = dict(velocity_key=smooth_key, pseudotime_key=pseudotime_key,
            cell_type_key=cell_type_key,
            stemness_key=infer_stemness_key(adata),
            cycle_key=infer_cycle_key(adata))


def run(ad_in, run_id):
    print(f"\n{'='*62}\n[arm] {run_id}  ({ad_in.n_obs} cells)\n{'='*62}")
    res = run_single_vampflow(
        adata_input=ad_in.copy(), run_id=run_id, sweep_type="confident",
        sweep_value=run_id, keys=keys, cfg=ROBUST_CFG, output_dir=OUT,
        seed=SEED, n_metastates=N_METASTATES,
        velot_velocity_weight=ROBUST_CFG.base_velot_velocity_weight,
        graph_n_neighbors=ROBUST_CFG.n_neighbors_reference,
        rerun_metaflow=True)
    return res["adata"]


ad_full = run(adata, "full")
ad_conf = run(adata[confident].copy(), "confident")


def pull(ad):
    sk = next(k for k in ("velot_vampflow_state", "velot_metaflow_state",
                          "velot_quantum_metaflow_state") if k in ad.obs)
    lk = next((k for k in (sk + "_label", "velot_vampflow_state_label")
               if k in ad.obs), None)
    st = pd.Series(ad.obs[sk].astype(str).values, index=ad.obs_names)
    lab = (pd.Series(ad.obs[lk].astype(str).values, index=ad.obs_names)
           if lk else None)
    return st, lab


st_f, lab_f = pull(ad_full)
st_c, lab_c = pull(ad_conf)
shared = st_f.index.intersection(st_c.index)
print(f"\n[compare] {len(shared)} shared cells")

# ---- partition agreement, numbering aside --------------------------
a, b = st_f.loc[shared], st_c.loc[shared]
ari = adjusted_rand_score(a, b)
nmi = normalized_mutual_info_score(a, b)
ct = pd.crosstab(a, b)
r, c = linear_sum_assignment(-ct.values)
kept = sum(ct.values[i, j] for i, j in zip(r, c)) / len(shared)
ct.to_csv(OUT / "contingency_full_vs_confident.csv")

rows = [{"quantity": "meta-state partition ARI", "value": round(ari, 3)},
        {"quantity": "meta-state partition NMI", "value": round(nmi, 3)},
        {"quantity": "cells keeping their matched state", "value": round(kept, 3)}]

# ---- terminal-state agreement, which is what R1.5 asks about -------
if lab_f is not None and lab_c is not None:
    tf = lab_f.loc[shared].str.contains("terminal|sink", case=False)
    tc = lab_c.loc[shared].str.contains("terminal|sink", case=False)
    agree = (tf == tc).mean()
    jac = ((tf & tc).sum() / max((tf | tc).sum(), 1))
    rows += [
        {"quantity": "cells called terminal, full run", "value": int(tf.sum())},
        {"quantity": "cells called terminal, confident run", "value": int(tc.sum())},
        {"quantity": "terminal/non-terminal agreement", "value": round(agree, 3)},
        {"quantity": "Jaccard of the terminal cell sets", "value": round(jac, 3)},
    ]
else:
    print("  (no state-label column; terminal comparison skipped)")

df = pd.DataFrame(rows).set_index("quantity")
df.to_csv(OUT / "summary.csv")
print(f"\n{'='*62}\nR1.5 terminal-state stability on confident cells\n{'='*62}")
print(df.to_string())
print(f"""
Read it as: high agreement means dropping the {100*(1-confident.mean()):.0f}% of
cells without raw transport support does not change which cells MetaFlow
calls terminal. That is what Reviewer 1 asked to be shown.

[done] wrote {OUT}/summary.csv""")

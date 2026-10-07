"""
MetaFlow controls for R1.6: is the meta-state structure imported from the
velocity field, or would any field do?

Reviewer 1 objects that the transition matrix is built from
velocity-aligned transitions, so MetaFlow "re-imports the field it is
supposed to summarize", and asks for a comparison against a run on the
raw (unsmoothed) field or on a randomised field. This runs exactly that:

    smooth    the VelOT field, as reported          (the claim)
    raw       the raw OT field, no MLP smoothing    (control 1)
    shuffled  the VelOT field, cells permuted       (control 2)

Everything else is held fixed: same cells, same latent space, same graph,
same seed, and K fixed at the same value for all three so the comparison
is not confounded by a different number of states.

Each arm is scored the same way the article scores the meta-states: how
well the partition lines up with the annotated cell types. If the
reviewer's concern is right, the shuffled arm should score about as well
as the real one. If the velocity is doing work, it should collapse.

    python run_metaflow_controls.py

Writes metaflow_controls/<arm>/ per run and metaflow_controls/summary.csv.
Expect roughly 3-5 minutes per arm.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

os.chdir(Path(__file__).resolve().parent)

# Reuse the machinery already written for the robustness analysis: this
# brings in run_single_vampflow, the key detection, and the config.
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
OUT = Path("metaflow_controls")
N_METASTATES = 12          # fixed for every arm, so the arms are comparable
SEED = 0
RAW_KEY_CANDIDATES = ("velot_velocity_raw_umap", "velot_velocity_raw")

OUT.mkdir(exist_ok=True)

print(f"[load] {ADATA}")
adata = sc.read_h5ad(ADATA)
adata = ensure_basic_preprocessing_if_needed(adata)

smooth_key = ensure_velot_velocity_key(adata)
pseudotime_key = ensure_pseudotime_if_missing(
    adata,
    pick_first_existing_obs_key(
        adata, ["velot_pseudotime", "pseudotime", "dpt_pseudotime",
                "palantir_pseudotime", "latent_time"]))
cell_type_key = infer_cell_type_key(adata)

raw_key = next((k for k in RAW_KEY_CANDIDATES if k in adata.obsm), None)
if raw_key is None:
    raise SystemExit(
        f"no raw field in obsm. Looked for {RAW_KEY_CANDIDATES}; "
        f"obsm has {sorted(adata.obsm)}")

# the randomised control: same vectors, assigned to the wrong cells
rng = np.random.default_rng(SEED)
perm = rng.permutation(adata.n_obs)
adata.obsm["velot_velocity_shuffled_umap"] = np.asarray(
    adata.obsm[smooth_key])[perm].copy()

ARMS = {
    "smooth":   smooth_key,
    "raw":      raw_key,
    "shuffled": "velot_velocity_shuffled_umap",
}
print(f"[keys] smoothed={smooth_key!r}  raw={raw_key!r}  "
      f"cell types={cell_type_key!r}  pseudotime={pseudotime_key!r}")
print(f"[plan] {len(ARMS)} arms, K fixed at {N_METASTATES}, seed {SEED}")


def score(ad_out, arm):
    """How well does this partition line up with the annotated types?"""
    sk = next((k for k in ("velot_vampflow_state", "velot_metaflow_state",
                           "velot_quantum_metaflow_state")
               if k in ad_out.obs), None)
    if sk is None:
        raise SystemExit(f"[{arm}] no meta-state column in the output")
    st = ad_out.obs[sk].astype(str)
    ct = ad_out.obs[cell_type_key].astype(str)
    cont = pd.crosstab(st, ct)
    cont.to_csv(OUT / f"contingency_{arm}.csv")
    purity = (cont.max(axis=1) / cont.sum(axis=1))
    return dict(
        arm=arm,
        states=cont.shape[0],
        ARI=round(adjusted_rand_score(ct, st), 3),
        NMI=round(normalized_mutual_info_score(ct, st), 3),
        median_purity=round(float(purity.median()), 3),
        states_over_80pc=int((purity >= 0.8).sum()),
        types_with_own_state=int(cont.idxmax(axis=1).nunique()),
    )


def n_significant_edges(run_dir):
    f = run_dir / "connectivity" / "velot_vampflow_state_edge_table.csv"
    if not f.exists():
        return np.nan
    t = pd.read_csv(f)
    return int((t["qvalue"] < 0.05).sum()) if "qvalue" in t else np.nan


rows = []
for arm, key in ARMS.items():
    print(f"\n{'='*62}\n[arm] {arm}  (velocity key: {key})\n{'='*62}")
    keys = {
        "velocity_key": key,
        "pseudotime_key": pseudotime_key,
        "cell_type_key": cell_type_key,
        "stemness_key": infer_stemness_key(adata),
        "cycle_key": infer_cycle_key(adata),
    }
    res = run_single_vampflow(
        adata_input=adata.copy(),
        run_id=arm,
        sweep_type="field",
        sweep_value=arm,
        keys=keys,
        cfg=ROBUST_CFG,
        output_dir=OUT,
        seed=SEED,
        n_metastates=N_METASTATES,
        velot_velocity_weight=ROBUST_CFG.base_velot_velocity_weight,
        graph_n_neighbors=ROBUST_CFG.n_neighbors_reference,
        rerun_metaflow=True,
    )
    row = score(res["adata"], arm)
    row["significant_edges_q<0.05"] = n_significant_edges(OUT / arm)
    rows.append(row)
    print(f"[{arm}] {row}")

df = pd.DataFrame(rows).set_index("arm")
df.to_csv(OUT / "summary.csv")
print(f"\n{'='*62}\nMetaFlow controls (R1.6)\n{'='*62}")
print(df.to_string())
print(f"""
How to read this: 'smooth' is the reported configuration. If the
meta-states merely re-imported the velocity field, 'shuffled' would
score about the same. The gap between them is what the velocity
contributes; 'raw' sits in between and shows what the smoother adds.

[done] wrote {OUT}/summary.csv""")

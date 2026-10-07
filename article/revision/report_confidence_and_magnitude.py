"""
Two reporting numbers the response still needs.  (R1.5 part 1, R3.5)

R1.5 asks for the fraction of cells with zero or low OT confidence in
each dataset. R3.5 asks whether the vector norm means anything; we
answer that it tracks sampling sparsity rather than a rate, and so far
we can only show it on the synthetic tree. Both come from the saved
AnnData objects, so this is reading, not re-running.

For every dataset it prints:
    coverage          fraction of cells with a raw OT velocity
    confidence        distribution of obs['velot_confidence']
    |v| vs sparsity   Spearman of the norm against the mean kNN distance
                      (the R3.5 claim: magnitude follows sampling)
    |v| vs truth      where a ground-truth field exists

    python report_confidence_and_magnitude.py

Takes a couple of minutes; the objects are large but only obs/obsm are
touched.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr

os.chdir(Path(__file__).resolve().parent)

FIG = Path("article_figures")
DATASETS = {
    "pancreas":          FIG / "figure2/results/data/velot_seed0_smooth_pancreas.h5ad",
    "gen_tree":          FIG / "figure3/results/data/velot_unbalanced_seed0_smooth_gen_tree.h5ad",
    "gen_bifurcation":   FIG / "figure3/results/data/velot_unbalanced_seed0_smooth_gen_bifurcation.h5ad",
    "erythroid":         FIG / "figure5/results/data/velot_seed0_smooth_erythroid.h5ad",
    "hindbrain":         FIG / "figure5/results/data/velot_seed0_smooth_hindbrain.h5ad",
    "murine":            FIG / "figure5/results/data/velot_seed0_smooth_murine.h5ad",
    "oligodendroglioma": FIG / "supplementary/root_sweeps_oligo/results/data/"
                               "velot_unbalanced_root-Stem-like-1939_seed0_smooth_oligodendroglioma.h5ad",
}

rows = []
for name, path in DATASETS.items():
    if not path.exists():
        print(f"[skip] {name}: {path} not found")
        continue
    ad = sc.read_h5ad(path)

    # the basis this dataset was run in -> the obsm suffix velot used
    suffix = next((b.split("X_", 1)[1] for b in ("X_dif_stem", "X_pca")
                   if f"velot_velocity_raw_{b.split('X_',1)[1]}" in ad.obsm), None)
    if suffix is None:
        print(f"[skip] {name}: no velot_velocity_raw_* in {sorted(ad.obsm)}")
        continue

    Vr = np.asarray(ad.obsm[f"velot_velocity_raw_{suffix}"])
    nr = np.linalg.norm(Vr, axis=1)
    has_raw = np.isfinite(nr) & (nr > 0)
    row = {"dataset": name, "n_cells": ad.n_obs,
           "coverage": round(float(has_raw.mean()), 3)}

    if "velot_confidence" in ad.obs:
        c = np.asarray(ad.obs["velot_confidence"], dtype=float)
        row.update({
            "conf_zero": round(float((c == 0).mean()), 3),
            "conf_median": round(float(np.nanmedian(c)), 2),
            "conf_p10": round(float(np.nanpercentile(c, 10)), 2),
        })

    # R3.5: does the norm track how sparsely this region was sampled?
    if "distances" in ad.obsp:
        D = ad.obsp["distances"]
        sparsity = np.asarray(D.sum(axis=1)).ravel() / np.maximum(
            (D > 0).sum(axis=1).A.ravel(), 1)
        for key, tag in ((f"velot_velocity_raw_{suffix}", "raw"),
                         (f"velot_velocity_{suffix}", "smooth")):
            if key not in ad.obsm:
                continue
            nv = np.linalg.norm(np.asarray(ad.obsm[key]), axis=1)
            ok = np.isfinite(nv) & (nv > 0) & np.isfinite(sparsity)
            row[f"rho_norm_vs_sparsity_{tag}"] = round(
                float(spearmanr(nv[ok], sparsity[ok])[0]), 3)

    if "true_velocity_pca" in ad.obsm:
        nt = np.linalg.norm(np.asarray(ad.obsm["true_velocity_pca"]), axis=1)
        row["true_norm_cv"] = round(float(nt.std() / nt.mean()), 3)

    rows.append(row)
    print(f"[done] {name}")

df = pd.DataFrame(rows).set_index("dataset")
df.to_csv("confidence_and_magnitude.csv")
print("\n" + "=" * 78)
print(df.to_string())
print("=" * 78)
print("""
coverage                 R1.5: fraction of cells that were a transport source
                         at least once, i.e. that have a raw OT velocity.
conf_zero                fraction with zero confidence.
rho_norm_vs_sparsity_*   R3.5: Spearman of the vector norm against the mean
                         distance to a cell's neighbours. A clearly positive
                         value says the norm reports how sparsely the region
                         was sampled, not a rate of change.
true_norm_cv             spread of the ground-truth norm. ~0 means the
                         synthetic truth is a unit-length direction field and
                         cannot be used to validate magnitude.

wrote confidence_and_magnitude.csv""")

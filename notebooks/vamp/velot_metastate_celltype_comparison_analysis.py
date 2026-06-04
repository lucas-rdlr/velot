"""
Analysis-only module: builds contingency tables, enrichment statistics, and
summary tables, then saves them to disk. No plotting happens here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

try:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
except Exception:
    adjusted_rand_score = None
    normalized_mutual_info_score = None


# ── helpers ──────────────────────────────────────────────────────────

def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _natural_sort_key(x):
    x = str(x)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", x)]


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[order[i]] = prev
    return np.clip(q, 0, 1)


def clean_annotation_series(s: pd.Series, missing_label: str = "Unknown") -> pd.Series:
    out = s.astype("object").astype(str)
    bad = out.isin(["nan", "None", "NA", "NaN", "", "unknown"])
    out[bad] = missing_label
    return out.astype(str)


def detect_metastate_key(adata, preferred: Optional[str] = None) -> str:
    if preferred is not None:
        if preferred not in adata.obs:
            raise KeyError(f"meta_state_key={preferred!r} not in adata.obs.")
        return preferred
    candidates = [
        "velot_metaflow_state", "velot_quantum_msm_state", "velot_vampflow_state",
        "metaflow_state", "quantum_msm_state", "vampflow_state",
    ]
    for key in candidates:
        if key in adata.obs:
            return key
    possible = [c for c in adata.obs.columns if "state" in c.lower() or "meta" in c.lower()]
    raise KeyError(f"Cannot auto-detect meta-state column. Candidates: {possible}")


def detect_state_label_key(adata, meta_state_key: str, preferred: Optional[str] = None) -> Optional[str]:
    if preferred is not None:
        return preferred if preferred in adata.obs else None
    candidates = [
        meta_state_key.replace("_state", "_state_label"),
        "velot_metaflow_state_label", "velot_quantum_msm_state_label",
        "velot_vampflow_state_label", "metaflow_state_label",
    ]
    for key in candidates:
        if key in adata.obs:
            return key
    return None


def check_cell_type_key(adata, cell_type_key: str) -> str:
    if cell_type_key not in adata.obs:
        likely = [c for c in adata.obs.columns
                  if any(s in c.lower() for s in ["cell", "type", "annotation", "cluster", "subtype"])]
        raise KeyError(f"cell_type_key={cell_type_key!r} not found. Possible: {likely}")
    return cell_type_key


# ── core analysis ────────────────────────────────────────────────────

def build_contingency(
    adata,
    meta_state_key: Optional[str] = None,
    cell_type_key: str = "cell_type",
    state_label_key: Optional[str] = None,
    min_celltype_count: int = 10,
    max_celltypes: Optional[int] = None,
    other_label: str = "Other",
) -> Dict[str, object]:
    meta_state_key = detect_metastate_key(adata, meta_state_key)
    state_label_key = detect_state_label_key(adata, meta_state_key, state_label_key)
    cell_type_key = check_cell_type_key(adata, cell_type_key)

    states = clean_annotation_series(adata.obs[meta_state_key])
    celltypes = clean_annotation_series(adata.obs[cell_type_key])

    ct_counts = celltypes.value_counts()
    keep = ct_counts.index[ct_counts >= min_celltype_count].tolist()
    if max_celltypes is not None and len(keep) > max_celltypes:
        keep = ct_counts.head(max_celltypes).index.tolist()
    celltypes2 = celltypes.where(celltypes.isin(keep), other_label)

    state_order = sorted(states.unique(), key=_natural_sort_key)
    celltype_order = celltypes2.value_counts().index.tolist()

    contingency = pd.crosstab(
        pd.Categorical(states, categories=state_order, ordered=True),
        pd.Categorical(celltypes2, categories=celltype_order, ordered=True),
        dropna=False,
    )
    contingency.index.name = "meta_state"
    contingency.columns.name = "cell_type"

    state_labels = None
    if state_label_key is not None:
        tmp = pd.DataFrame({"state": states, "label": clean_annotation_series(adata.obs[state_label_key])})
        state_labels = tmp.groupby("state")["label"].agg(lambda x: x.value_counts().index[0]).to_dict()

    return {
        "meta_state_key": meta_state_key,
        "state_label_key": state_label_key,
        "cell_type_key": cell_type_key,
        "states": states,
        "celltypes": celltypes2,
        "contingency": contingency,
        "state_order": state_order,
        "celltype_order": celltype_order,
        "state_labels": state_labels,
    }


def compute_enrichment_statistics(contingency: pd.DataFrame) -> pd.DataFrame:
    C = contingency.to_numpy(dtype=float)
    N = C.sum()
    state_totals = C.sum(axis=1)
    type_totals = C.sum(axis=0)
    rows = []
    for i, state in enumerate(contingency.index):
        for j, ct in enumerate(contingency.columns):
            obs = C[i, j]
            expected = state_totals[i] * type_totals[j] / max(N, 1)
            a, b = obs, state_totals[i] - obs
            c = type_totals[j] - obs
            d = N - a - b - c
            try:
                odds, p = fisher_exact(np.array([[a, b], [c, d]]), alternative="greater")
            except Exception:
                odds, p = np.nan, np.nan
            rows.append({
                "meta_state": state, "cell_type": ct,
                "observed": int(obs), "expected": expected,
                "log2_enrichment": np.log2((obs + 0.5) / (expected + 0.5)),
                "odds_ratio": odds, "p_value": p,
                "prop_within_state": obs / state_totals[i] if state_totals[i] > 0 else 0,
                "prop_within_celltype": obs / type_totals[j] if type_totals[j] > 0 else 0,
                "state_total": int(state_totals[i]),
                "celltype_total": int(type_totals[j]),
            })
    df = pd.DataFrame(rows)
    df["q_value"] = _bh_fdr(df["p_value"].fillna(1).to_numpy())
    df["minus_log10_q"] = -np.log10(df["q_value"].clip(lower=1e-300))
    df["enriched"] = (df["q_value"] < 0.05) & (df["log2_enrichment"] > 0)
    return df


def compute_state_summary(contingency, enrichment_df, states, celltypes) -> pd.DataFrame:
    C = contingency.to_numpy(dtype=float)
    state_totals = C.sum(axis=1)
    rows = []
    for i, state in enumerate(contingency.index):
        row = C[i]
        total = state_totals[i]
        if total > 0:
            prop = row / total
            entropy = -np.sum(prop[prop > 0] * np.log2(prop[prop > 0]))
            norm_entropy = entropy / np.log2(len(prop)) if len(prop) > 1 else 0
            top_j = int(np.argmax(row))
            top_ct, top_prop = contingency.columns[top_j], prop[top_j]
        else:
            entropy, norm_entropy, top_ct, top_prop = np.nan, np.nan, None, np.nan
        sub = enrichment_df[enrichment_df["meta_state"].astype(str) == str(state)].sort_values(
            ["q_value", "log2_enrichment"], ascending=[True, False])
        enriched = sub[(sub["q_value"] < 0.05) & (sub["log2_enrichment"] > 0)]["cell_type"].tolist()
        rows.append({
            "meta_state": state, "n_cells": int(total),
            "dominant_cell_type": top_ct, "dominant_cell_type_fraction": top_prop,
            "composition_entropy": entropy, "normalized_composition_entropy": norm_entropy,
            "top_enriched_cell_type": enriched[0] if enriched else None,
            "n_significantly_enriched_cell_types": len(enriched),
            "significantly_enriched_cell_types": "; ".join(enriched),
        })
    return pd.DataFrame(rows)


# ── main entry point ─────────────────────────────────────────────────

def run_analysis(
    adata,
    meta_state_key: Optional[str] = None,
    cell_type_key: str = "cell_type",
    state_label_key: Optional[str] = None,
    embedding_key: str = "X_umap",
    output_dir: str | Path = "metastate_celltype_comparison",
    min_celltype_count: int = 10,
    max_celltypes: Optional[int] = None,
) -> Path:
    """
    Run the full analysis once, save everything to *output_dir*, return that path.
    """
    output_dir = _ensure_dir(output_dir)

    data = build_contingency(adata, meta_state_key, cell_type_key,
                             state_label_key, min_celltype_count, max_celltypes)
    states = data["states"]
    celltypes = data["celltypes"]
    contingency = data["contingency"]

    enrichment_df = compute_enrichment_statistics(contingency)
    state_summary = compute_state_summary(contingency, enrichment_df, states, celltypes)

    # ── save tables ──
    contingency.to_csv(output_dir / "contingency.csv")
    (contingency.div(contingency.sum(axis=1).replace(0, np.nan), axis=0)
     .fillna(0).to_csv(output_dir / "state_prop.csv"))
    (contingency.div(contingency.sum(axis=0).replace(0, np.nan), axis=1)
     .fillna(0).to_csv(output_dir / "celltype_prop.csv"))
    enrichment_df.to_csv(output_dir / "enrichment.csv", index=False)
    state_summary.to_csv(output_dir / "state_summary.csv", index=False)

    # ── save per-cell vectors for UMAP ──
    cell_df = pd.DataFrame({
        "meta_state": states.values,
        "cell_type": celltypes.values,
    }, index=adata.obs_names)

    if embedding_key in adata.obsm:
        E = np.asarray(adata.obsm[embedding_key])[:, :2]
        cell_df["umap_1"] = E[:, 0]
        cell_df["umap_2"] = E[:, 1]
    cell_df.to_csv(output_dir / "cell_data.csv")

    # ── save config / global metrics ──
    config = {
        "meta_state_key": data["meta_state_key"],
        "cell_type_key": data["cell_type_key"],
        "state_label_key": data["state_label_key"],
        "embedding_key": embedding_key,
        "state_order": data["state_order"],
        "celltype_order": data["celltype_order"],
        "state_labels": data["state_labels"],
    }
    if adjusted_rand_score is not None:
        config["ARI"] = float(adjusted_rand_score(states.astype(str), celltypes.astype(str)))
    if normalized_mutual_info_score is not None:
        config["NMI"] = float(normalized_mutual_info_score(states.astype(str), celltypes.astype(str)))

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    print(f"[analysis] All tables saved to {output_dir}/")
    print(f"  contingency.csv, enrichment.csv, state_summary.csv,")
    print(f"  state_prop.csv, celltype_prop.csv, cell_data.csv, config.json")
    if "ARI" in config:
        print(f"  ARI = {config['ARI']:.3f}")
    if "NMI" in config:
        print(f"  NMI = {config['NMI']:.3f}")

    return output_dir


if __name__ == "__main__":

    import scanpy as sc
    import sys
    from pathlib import Path

    PROJECT_DIR = Path("/home/user/Documents/velot_agusti/notebooks/agusti")
    sys.path.insert(0, str(PROJECT_DIR))

    adata = sc.read_h5ad("results/vamp_robustness_pancreas/reference_pancreas_vampflow.h5ad")

    run_analysis(
        adata,
        meta_state_key="velot_vampflow_state",
        cell_type_key="clusters",
        embedding_key="X_umap",
        output_dir="metastate_celltype_comparison_pancreas",
    )
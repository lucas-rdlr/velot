#!/usr/bin/env python
"""
End-to-end VelOT Quantum/VAMP MetaFlow + PAGA-like connectivity analysis
for the pancreas VelOT dataset.

This script is adapted to your local project structure:

    PROJECT_DIR = Path("/home/aalentorn/Projects/stemness_T_cell")
    ADATA_PATH  = PROJECT_DIR / "pancreas_velot.h5ad"

It assumes that these two modules are available either:
    1. in PROJECT_DIR/scripts/
    2. in PROJECT_DIR/
    3. in the current working directory

Required module files:
    velot_quantum_metaflow.py
    velot_paga_like_connectivity.py

These are the two previously provided scripts:
    - velot_quantum_metaflow.py
    - velot_paga_like_connectivity.py

Outputs:
    PROJECT_DIR/results/velot_quantum_paga_pancreas/
        01_metaflow/
        02_paga_like_connectivity/
        pancreas_with_velot_quantum_metaflow.h5ad
        pancreas_with_velot_quantum_paga_like_connectivity.h5ad
"""

from __future__ import annotations

import sys
from pathlib import Path
import warnings

import scanpy as sc
import pandas as pd
import numpy as np


# =============================================================================
# 0. Project paths
# =============================================================================

PROJECT_DIR = Path("/home/user/Documents/velot_agusti")
ADATA_PATH = PROJECT_DIR / "pancreas_velot.h5ad"

RESULTS_DIR = PROJECT_DIR / "results" / "velot_quantum_paga_pancreas"
METAFLOW_OUTDIR = RESULTS_DIR / "01_metaflow"
PAGA_OUTDIR = RESULTS_DIR / "02_paga_like_connectivity"

INTERMEDIATE_H5AD = RESULTS_DIR / "pancreas_with_velot_quantum_metaflow.h5ad"
FINAL_H5AD = RESULTS_DIR / "pancreas_with_velot_quantum_paga_like_connectivity.h5ad"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
METAFLOW_OUTDIR.mkdir(parents=True, exist_ok=True)
PAGA_OUTDIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. Make sure local modules can be imported
# =============================================================================

candidate_module_dirs = [
    PROJECT_DIR / "scripts",
    PROJECT_DIR,
    Path.cwd(),
]

for d in candidate_module_dirs:
    if d.exists():
        sys.path.insert(0, str(d))

try:
    from velot_quantum_metaflow import (
        VelOTQuantumMetaFlowConfig,
        run_velot_quantum_metaflow,
    )
except Exception as e:
    raise ImportError(
        "\nCould not import velot_quantum_metaflow.py.\n\n"
        "Please place velot_quantum_metaflow.py in one of these locations:\n"
        f"  - {PROJECT_DIR / 'scripts'}\n"
        f"  - {PROJECT_DIR}\n"
        f"  - {Path.cwd()}\n\n"
        "Then rerun this script.\n"
    ) from e

try:
    from velot_paga_like_connectivity import (
        PagaLikeConnectivityConfig,
        run_velot_paga_like_connectivity_analysis,
    )
except Exception as e:
    raise ImportError(
        "\nCould not import velot_paga_like_connectivity.py.\n\n"
        "Please place velot_paga_like_connectivity.py in one of these locations:\n"
        f"  - {PROJECT_DIR / 'scripts'}\n"
        f"  - {PROJECT_DIR}\n"
        f"  - {Path.cwd()}\n\n"
        "Then rerun this script.\n"
    ) from e


# =============================================================================
# 2. Helpers to infer keys robustly
# =============================================================================

def pick_first_existing_obs_key(adata, candidates, default=None):
    for key in candidates:
        if key in adata.obs:
            return key
    return default


def pick_first_existing_obsm_key(adata, candidates, default=None):
    for key in candidates:
        if key in adata.obsm:
            return key
    return default


def ensure_basic_preprocessing_if_needed(adata):
    """
    Only computes PCA/neighbors/UMAP if missing.
    Does not overwrite existing VelOT results.
    """
    if "X_pca" not in adata.obsm:
        print("[preprocess] X_pca not found. Computing PCA.")
        sc.pp.pca(adata, n_comps=50)

    if "neighbors" not in adata.uns:
        print("[preprocess] neighbors not found. Computing neighbors using X_pca.")
        sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=30)

    if "X_umap" not in adata.obsm:
        print("[preprocess] X_umap not found. Computing UMAP.")
        sc.tl.umap(adata)

    return adata


def ensure_pseudotime_if_missing(adata, pseudotime_key):
    """
    If no VelOT/pseudotime key is available, compute DPT using diffusion map.
    This is only a fallback; if velot_pseudotime exists it will be preferred.
    """
    if pseudotime_key is not None and pseudotime_key in adata.obs:
        return pseudotime_key

    candidate_existing = pick_first_existing_obs_key(
        adata,
        [
            "velot_pseudotime",
            "pseudotime",
            "dpt_pseudotime",
            "palantir_pseudotime",
            "latent_time",
        ],
    )
    if candidate_existing is not None:
        return candidate_existing

    print("[pseudotime] No pseudotime key found. Computing fallback DPT pseudotime.")
    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, use_rep="X_pca" if "X_pca" in adata.obsm else None)
    if "X_diffmap" not in adata.obsm:
        sc.tl.diffmap(adata)

    # Choose a root cell heuristically as the minimum of the first non-trivial diffusion component.
    diff = np.asarray(adata.obsm["X_diffmap"])
    root_dim = 1 if diff.shape[1] > 1 else 0
    adata.uns["iroot"] = int(np.nanargmin(diff[:, root_dim]))
    sc.tl.dpt(adata)

    return "dpt_pseudotime"


def ensure_velot_velocity_key(adata):
    """
    Ensures that adata.obsm["velot_velocity_umap"] exists.
    If not, tries common alternative velocity keys.
    """
    if "velot_velocity_umap" in adata.obsm:
        return "velot_velocity_umap"

    alternatives = [
        "velocity_umap",
        "velot_velocity_embedding",
        "velocity_embedding",
        "X_velocity_umap",
    ]
    alt = pick_first_existing_obsm_key(adata, alternatives)
    if alt is not None:
        print(f"[velocity] adata.obsm['velot_velocity_umap'] not found, using {alt!r} and copying it.")
        adata.obsm["velot_velocity_umap"] = np.asarray(adata.obsm[alt])[:, :2]
        return "velot_velocity_umap"

    raise KeyError(
        "Could not find adata.obsm['velot_velocity_umap'] or common alternatives:\n"
        f"{alternatives}\n\n"
        "Please make sure the pancreas_velot.h5ad file contains the VelOT UMAP vector field.\n"
    )


def infer_cell_type_key(adata):
    """
    Tries common annotation columns.
    Update this list if your pancreas object uses another annotation key.
    """
    return pick_first_existing_obs_key(
        adata,
        [
            "cell_type",
            "celltype",
            "celltypes",
            "CellType",
            "annotation",
            "annotations",
            "clusters",
            "leiden",
            "louvain",
            "Subtype",
            "subtype",
        ],
        default=None,
    )


def infer_stemness_key(adata):
    return pick_first_existing_obs_key(
        adata,
        [
            "velot_stemness",
            "stemness",
            "hv_stemness",
            "cytotrace_score",
            "CytoTRACE",
            "cytotrace",
        ],
        default=None,
    )


def infer_cycle_key(adata):
    return pick_first_existing_obs_key(
        adata,
        [
            "cell_cycle_score",
            "S_score",
            "G2M_score",
            "phase_score",
            "cycling_score",
        ],
        default=None,
    )


# =============================================================================
# 3. Load data
# =============================================================================

print(f"[load] Reading: {ADATA_PATH}")
if not ADATA_PATH.exists():
    raise FileNotFoundError(
        f"Input AnnData file not found:\n{ADATA_PATH}\n\n"
        "Please verify PROJECT_DIR and the filename pancreas_velot.h5ad."
    )

adata = sc.read_h5ad(ADATA_PATH)
print(adata)

adata = ensure_basic_preprocessing_if_needed(adata)

velot_velocity_key = ensure_velot_velocity_key(adata)
cell_type_key = infer_cell_type_key(adata)
stemness_key = infer_stemness_key(adata)
cycle_key = infer_cycle_key(adata)

pseudotime_key = pick_first_existing_obs_key(
    adata,
    [
        "velot_pseudotime",
        "pseudotime",
        "dpt_pseudotime",
        "palantir_pseudotime",
        "latent_time",
    ],
)
pseudotime_key = ensure_pseudotime_if_missing(adata, pseudotime_key)

print("\n[detected keys]")
print(f"  latent_key:         X_pca")
print(f"  embedding_key:      X_umap")
print(f"  velocity_key:       {velot_velocity_key}")
print(f"  pseudotime_key:     {pseudotime_key}")
print(f"  cell_type_key:      {cell_type_key}")
print(f"  stemness_key:       {stemness_key}")
print(f"  cycle_key:          {cycle_key}")

if cell_type_key is None:
    warnings.warn(
        "No cell-type annotation key was detected. "
        "Camembert plots will be skipped unless you set cell_type_key manually."
    )


# =============================================================================
# 4. Run VelOT Quantum/VAMP MetaFlow
# =============================================================================

metaflow_cfg = VelOTQuantumMetaFlowConfig(
    latent_key="X_pca",
    embedding_key="X_umap",
    velot_velocity_key=velot_velocity_key,
    pseudotime_key=pseudotime_key,
    cell_type_key=cell_type_key,

    stemness_key=stemness_key,
    cycle_key=cycle_key,

    # Run both estimators: previous VAMPFlow + new QuantumMSM
    estimator_mode="both",

    # Vector-field setup
    velocity_to_latent_method="local_linear",
    use_ot_flow_matching=True,
    final_velocity_mode="blend",
    velot_velocity_weight=0.50,
    flow_velot_alignment_weight=0.15,

    # Meta-state model
    n_metastates="auto",
    metastate_candidates=[5, 6, 7, 8, 9, 10, 12],

    # Reasonable first-run settings.
    # Increase epochs for final high-quality runs.
    flow_epochs=1200,
    vamp_epochs=1200,
    quantum_epochs=1400,

    output_dir=str(METAFLOW_OUTDIR),
    random_state=0,
)

print("\n[run] VelOT Quantum/VAMP MetaFlow")
metaflow_out = run_velot_quantum_metaflow(adata, metaflow_cfg)
adata_mf = metaflow_out["adata"]

print(f"\n[save] Intermediate MetaFlow AnnData: {INTERMEDIATE_H5AD}")
adata_mf.write_h5ad(INTERMEDIATE_H5AD)


# =============================================================================
# 5. Run PAGA-like quantitative connectivity analysis
# =============================================================================

connectivity_cfg = PagaLikeConnectivityConfig(
    group_keys=[
        "velot_vampflow_state",
        "velot_quantum_msm_state",
    ],
    latent_key="X_velot_quantum_metaflow_latent_scaled",
    embedding_key="X_umap",
    velocity_key="velot_quantum_metaflow_velocity_latent_final",
    pseudotime_key="velot_quantum_metaflow_pseudotime",

    # PAGA-like kNN graph
    use_existing_connectivities=True,
    connectivities_key="connectivities",
    n_neighbors=30,

    # Vector-field directed graph
    directed_k=30,
    directed_temperature=0.25,

    # Null model
    n_permutations=200,

    # For final figures, consider:
    # n_permutations=1000,

    output_dir=str(PAGA_OUTDIR),
    random_state=0,
)

print("\n[run] PAGA-like quantitative connectivity analysis")
conn_out = run_velot_paga_like_connectivity_analysis(adata_mf, connectivity_cfg)
adata_final = conn_out["adata"]

print(f"\n[save] Final AnnData with MetaFlow + PAGA-like connectivity: {FINAL_H5AD}")
adata_final.write_h5ad(FINAL_H5AD)


# =============================================================================
# 6. Print concise output summary
# =============================================================================

print("\n[complete]")
print(f"Results directory:       {RESULTS_DIR}")
print(f"MetaFlow figures:        {METAFLOW_OUTDIR}")
print(f"PAGA-like figures:       {PAGA_OUTDIR}")
print(f"Intermediate h5ad:       {INTERMEDIATE_H5AD}")
print(f"Final h5ad:              {FINAL_H5AD}")

print("\nGenerated key outputs in adata.obs:")
for key in [
    "velot_vampflow_state",
    "velot_vampflow_state_label",
    "velot_quantum_msm_state",
    "velot_quantum_msm_state_label",
    "velot_quantum_metaflow_pseudotime",
    "velot_quantum_metaflow_divergence",
    "velot_quantum_metaflow_embedding_curl",
]:
    if key in adata_final.obs:
        print(f"  - {key}")

print("\nGenerated key outputs in adata.uns:")
for key in [
    "velot_vampflow_state_paga_like_connectivity",
    "velot_quantum_msm_state_paga_like_connectivity",
    "velot_vampflow_transition_matrix",
    "velot_quantum_msm_transition_matrix",
    "velot_quantum_msm_state_summary",
    "velot_vampflow_state_summary",
]:
    if key in adata_final.uns:
        print(f"  - {key}")

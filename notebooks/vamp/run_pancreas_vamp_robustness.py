#!/usr/bin/env python
"""
Pancreas VelOT VAMPFlow robustness analysis
===========================================

This script runs a PAGA-robustness-inspired analysis for the VAMPFlow estimator
on your pancreas VelOT AnnData object.

It starts directly from:

    PROJECT_DIR = Path("/home/aalentorn/Projects/stemness_T_cell")
    ADATA_PATH  = PROJECT_DIR / "pancreas_velot.h5ad"

It does NOT require an existing adata_with_velot_quantum_metaflow.h5ad file.

What it assesses
----------------
For the VAMPFlow approach, it evaluates robustness of:

    1. VAMP meta-state assignments
    2. source/sink/branch/cycle annotations
    3. PAGA-like connectivity confidence
    4. permutation-supported significant edges
    5. vector-field-directed flux
    6. directionality
    7. stability across:
        - random seeds
        - cell bootstraps
        - kNN graph size
        - VelOT/OT velocity blending
        - VelOT velocity noise
        - optional K / number-of-meta-states sensitivity

Core idea
---------
A reference VAMPFlow run is computed first. Each subsequent run is aligned back
to the reference by maximal cell-overlap using the Hungarian algorithm. This is
important because state IDs such as M0/M1/M2 are arbitrary across independent
neural runs.

Required local module files
---------------------------
Place these two previously provided scripts in one of:

    /home/aalentorn/Projects/stemness_T_cell/scripts/
    /home/aalentorn/Projects/stemness_T_cell/
    current working directory

Required:
    velot_quantum_metaflow.py
    velot_paga_like_connectivity.py       # connectivity version 1 preferred

Run
---
cd /home/aalentorn/Projects/stemness_T_cell
python scripts/run_pancreas_vamp_robustness.py

Outputs
-------
PROJECT_DIR/results/vamp_robustness_pancreas/
    00_reference/
    01_runs/
    02_robustness_summary/
    reference_pancreas_vampflow.h5ad
    robustness_summary.csv
    edge_stability_undirected.csv
    edge_stability_directed.csv
    several publication-ready figures
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Sequence, Set

import numpy as np
import scanpy as sc

# =============================================================================
# 0. Project paths
# =============================================================================

PROJECT_DIR = Path("/home/user/Documents/velot_agusti")
ADATA_PATH = PROJECT_DIR / "pancreas_velot.h5ad"

RESULTS_DIR = PROJECT_DIR / "results" / "vamp_robustness_pancreas3"
REFERENCE_DIR = RESULTS_DIR / "00_reference"

for d in [RESULTS_DIR, REFERENCE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. Import local modules
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
        "Could not import velot_quantum_metaflow.py. "
        "Place it in PROJECT_DIR/scripts, PROJECT_DIR, or the current directory."
    ) from e

try:
    from velot_paga_like_connectivity import (
        PagaLikeConnectivityConfig,
        run_velot_paga_like_connectivity_analysis,
    )
except Exception as e:
    raise ImportError(
        "Could not import velot_paga_like_connectivity.py. "
        "Place the connectivity-version-1 file in PROJECT_DIR/scripts, PROJECT_DIR, or the current directory."
    ) from e


# =============================================================================
# 2. Robustness configuration
# =============================================================================

@dataclass
class VAMPRobustnessConfig:
    # Analysis profile
    # For final results, increase epochs and n_permutations below.
    flow_epochs: int = 800
    vamp_epochs: int = 900
    n_permutations: int = 200

    # Reference model
    reference_seed: int = 0
    reference_n_metastates: object = "auto"
    metastate_candidates: Sequence[int] = field(default_factory=lambda: [5, 6, 7, 8, 9, 10, 12])

    # Sweeps
    seed_sweep: Sequence[int] = field(default_factory=lambda: [1, 2, 3])
    bootstrap_seeds: Sequence[int] = field(default_factory=lambda: [11, 12, 13, 14, 15])
    bootstrap_fraction: float = 0.85

    graph_k_sweep: Sequence[int] = field(default_factory=lambda: [15, 20, 30, 40, 50])
    velot_weight_sweep: Sequence[float] = field(default_factory=lambda: [0.25, 0.50, 0.75, 1.00])
    velocity_noise_sweep: Sequence[float] = field(default_factory=lambda: [0.05, 0.10, 0.20])

    # Optional K sensitivity.
    run_k_sweep: bool = True
    k_sweep_delta: Sequence[int] = field(default_factory=lambda: [-2, -1, 0, 1, 2])

    # Connectivity thresholds
    edge_confidence_threshold: float = 0.15
    edge_qvalue_threshold: float = 0.10
    directed_flux_threshold: float = 0.05

    # Graph
    n_neighbors_reference: int = 30
    directed_k: int = 30

    # Vector field
    base_velot_velocity_weight: float = 0.50
    flow_velot_alignment_weight: float = 0.15

    # Computation options
    save_each_run_h5ad: bool = False
    device: Optional[str] = None

    # Plotting
    figure_dpi: int = 300


ROBUST_CFG = VAMPRobustnessConfig()


# =============================================================================
# 3. Helpers for key detection and preprocessing
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

    diff = np.asarray(adata.obsm["X_diffmap"])
    root_dim = 1 if diff.shape[1] > 1 else 0
    adata.uns["iroot"] = int(np.nanargmin(diff[:, root_dim]))
    sc.tl.dpt(adata)
    return "dpt_pseudotime"


def ensure_velot_velocity_key(adata):
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
        print(f"[velocity] velot_velocity_umap not found. Copying {alt!r} to velot_velocity_umap.")
        adata.obsm["velot_velocity_umap"] = np.asarray(adata.obsm[alt])[:, :2]
        return "velot_velocity_umap"

    raise KeyError(
        "Could not find adata.obsm['velot_velocity_umap'] or common alternatives. "
        "The robustness analysis needs a VelOT UMAP vector field."
    )


def infer_cell_type_key(adata):
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
# 4. Run VAMPFlow + PAGA-like connectivity
# =============================================================================

def add_velocity_noise_to_copy(adata, base_velocity_key: str, noise_level: float, seed: int) -> Tuple[object, str]:
    """
    Adds Gaussian noise to the UMAP velocity field for perturbation robustness.
    noise_level is relative to the median vector norm.
    """
    ad = adata.copy()
    V = np.asarray(ad.obsm[base_velocity_key], dtype=np.float32)
    rng = np.random.default_rng(seed)
    med = np.median(np.linalg.norm(V[:, :2], axis=1)) + 1e-8
    noise = rng.normal(0.0, noise_level * med, size=V[:, :2].shape).astype(np.float32)
    key = f"{base_velocity_key}_noise_{noise_level:.2f}".replace(".", "p")
    ad.obsm[key] = V[:, :2] + noise
    return ad, key


def bootstrap_adata(adata, fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    n = adata.n_obs
    size = max(10, int(round(fraction * n)))
    idx = rng.choice(np.arange(n), size=size, replace=False)
    return adata[idx].copy()


def run_single_vampflow(
    adata_input,
    run_id: str,
    sweep_type: str,
    sweep_value: object,
    keys: Dict[str, Optional[str]],
    cfg: VAMPRobustnessConfig,
    output_dir: Path,
    seed: int,
    n_metastates: object,
    velot_velocity_weight: float,
    velocity_noise: float = 0.0,
    bootstrap_fraction: Optional[float] = None,
    graph_n_neighbors: Optional[int] = None,
    rerun_metaflow: bool = True,
    reference_adata_for_graph_only=None,
) -> Dict[str, object]:
    """
    Runs VAMPFlow and PAGA-like connectivity.

    For graph-only robustness, rerun_metaflow=False and reference_adata_for_graph_only
    should be provided.
    """
    run_dir = output_dir / run_id
    mf_dir = run_dir / "metaflow"
    conn_dir = run_dir / "connectivity"
    mf_dir.mkdir(parents=True, exist_ok=True)
    conn_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[run] {run_id} | {sweep_type}={sweep_value}")

    if rerun_metaflow:
        ad_run = adata_input

        if bootstrap_fraction is not None:
            ad_run = bootstrap_adata(ad_run, bootstrap_fraction, seed)

        velocity_key = keys["velocity_key"]
        if velocity_noise > 0:
            ad_run, velocity_key = add_velocity_noise_to_copy(ad_run, velocity_key, velocity_noise, seed)

        mf_cfg = VelOTQuantumMetaFlowConfig(
            latent_key="X_pca",
            embedding_key="X_umap",
            velot_velocity_key=velocity_key,
            pseudotime_key=keys["pseudotime_key"],
            cell_type_key=keys["cell_type_key"],
            stemness_key=keys["stemness_key"],
            cycle_key=keys["cycle_key"],

            estimator_mode="vamp",

            velocity_to_latent_method="local_linear",
            use_ot_flow_matching=True,
            final_velocity_mode="blend",
            velot_velocity_weight=velot_velocity_weight,
            flow_velot_alignment_weight=cfg.flow_velot_alignment_weight,

            n_metastates=n_metastates,
            metastate_candidates=list(cfg.metastate_candidates),

            flow_epochs=cfg.flow_epochs,
            vamp_epochs=cfg.vamp_epochs,

            output_dir=str(mf_dir),
            random_state=seed,
            device=cfg.device,
        )

        mf_out = run_velot_quantum_metaflow(ad_run, mf_cfg)
        ad_mf = mf_out["adata"]

        if cfg.save_each_run_h5ad:
            ad_mf.write_h5ad(run_dir / f"{run_id}_vampflow.h5ad")

    else:
        if reference_adata_for_graph_only is None:
            raise ValueError("reference_adata_for_graph_only is required when rerun_metaflow=False.")
        ad_mf = reference_adata_for_graph_only.copy()

    conn_cfg = PagaLikeConnectivityConfig(
        group_keys=["velot_vampflow_state"],
        latent_key="X_velot_quantum_metaflow_latent_scaled",
        embedding_key="X_umap",
        velocity_key="velot_quantum_metaflow_velocity_latent_final",
        pseudotime_key="velot_quantum_metaflow_pseudotime",

        use_existing_connectivities=True,
        connectivities_key="connectivities",
        n_neighbors=graph_n_neighbors or cfg.n_neighbors_reference,

        directed_k=cfg.directed_k,
        directed_temperature=0.25,

        n_permutations=cfg.n_permutations,
        edge_confidence_threshold=cfg.edge_confidence_threshold,
        edge_qvalue_threshold=cfg.edge_qvalue_threshold,

        output_dir=str(conn_dir),
        random_state=seed,
    )

    conn_out = run_velot_paga_like_connectivity_analysis(ad_mf, conn_cfg)

    return {
        "run_id": run_id,
        "sweep_type": sweep_type,
        "sweep_value": sweep_value,
        "seed": seed,
        "adata": conn_out["adata"],
        "conn_out": conn_out,
        "run_dir": run_dir,
    }

# =============================================================================
# 7. Main
# =============================================================================

def main():
    print(f"[load] {ADATA_PATH}")
    if not ADATA_PATH.exists():
        raise FileNotFoundError(f"Could not find {ADATA_PATH}")

    adata = sc.read_h5ad(ADATA_PATH)
    print(adata)

    adata = ensure_basic_preprocessing_if_needed(adata)
    velocity_key = ensure_velot_velocity_key(adata)

    pseudotime_key = pick_first_existing_obs_key(
        adata,
        ["velot_pseudotime", "pseudotime", "dpt_pseudotime", "palantir_pseudotime", "latent_time"],
    )
    pseudotime_key = ensure_pseudotime_if_missing(adata, pseudotime_key)

    keys = {
        "velocity_key": velocity_key,
        "pseudotime_key": pseudotime_key,
        "cell_type_key": infer_cell_type_key(adata),
        "stemness_key": infer_stemness_key(adata),
        "cycle_key": infer_cycle_key(adata),
    }

    print("\n[detected keys]")
    for k, v in keys.items():
        print(f"  {k}: {v}")

    # -------------------------------------------------------------------------
    # Reference run
    # -------------------------------------------------------------------------
    ref_result = run_single_vampflow(
        adata_input=adata.copy(),
        run_id="reference",
        sweep_type="reference",
        sweep_value="reference",
        keys=keys,
        cfg=ROBUST_CFG,
        output_dir=REFERENCE_DIR,
        seed=ROBUST_CFG.reference_seed,
        n_metastates=ROBUST_CFG.reference_n_metastates,
        velot_velocity_weight=ROBUST_CFG.base_velot_velocity_weight,
        graph_n_neighbors=ROBUST_CFG.n_neighbors_reference,
        rerun_metaflow=True,
        reference_adata_for_graph_only=sc.read_h5ad("/home/user/Documents/velot_agusti/results/vamp_robustness_pancreas/reference_pancreas_vampflow.h5ad")
    )

    ref_adata = ref_result["adata"]
    ref_h5ad = RESULTS_DIR / "reference_pancreas_vampflow.h5ad"
    print(f"[save] reference AnnData: {ref_h5ad}")
    ref_adata.write_h5ad(ref_h5ad)

    print("\n[complete]")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Reference h5ad:    {ref_h5ad}")


if __name__ == "__main__":
    main()
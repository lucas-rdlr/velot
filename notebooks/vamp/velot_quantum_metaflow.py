
"""
VelOT Quantum-MetaFlow: GPCCA-free, CellRank-free meta-state discovery with two estimators.

Estimator modes
---------------
1) "vamp"
   The previous VAMPFlow-style neural metastability estimator:
       Z -> Z_future -> soft meta-states q(z) -> classical coarse transition matrix.

2) "quantum_msm"
   A quantum-inspired Markov-state estimator:
       Z -> Z_future -> complex amplitudes psi(z)
       -> density/coherence matrices
       -> quantum-channel-style transition matrix from amplitude overlaps |A_ab|^2
       -> source/sink/branch/cycle annotation.

3) "both"
   Runs both estimators with the same VelOT/OT vector field and writes both outputs.

The pipeline explicitly uses the VelOT vector field:
    adata.obsm["velot_velocity_umap"]

It can:
    - plot the raw VelOT UMAP velocity field;
    - lift VelOT UMAP velocity into latent space;
    - train an OT-flow-matching vector field aligned to VelOT velocity;
    - blend OT-flow and VelOT velocities;
    - infer neural meta-states using either VAMPFlow or QuantumMSM.

Dependencies
------------
pip install numpy pandas scipy scikit-learn matplotlib seaborn scanpy anndata torch networkx

CellRank is intentionally NOT used.

Minimal usage
-------------
import scanpy as sc
from velot_quantum_metaflow import VelOTQuantumMetaFlowConfig, run_velot_quantum_metaflow

adata = sc.read_h5ad("your_velot_dataset.h5ad")

cfg = VelOTQuantumMetaFlowConfig(
    latent_key="X_pca",
    embedding_key="X_umap",
    velot_velocity_key="velot_velocity_umap",
    pseudotime_key="velot_pseudotime",
    cell_type_key="cell_type",

    estimator_mode="quantum_msm",   # "vamp", "quantum_msm", or "both"

    final_velocity_mode="blend",
    velot_velocity_weight=0.50,
    use_ot_flow_matching=True,
    flow_velot_alignment_weight=0.15,

    n_metastates="auto",
    output_dir="velot_quantum_metaflow_outputs",
)

out = run_velot_quantum_metaflow(adata, cfg)
out["adata"].write_h5ad("adata_with_velot_quantum_metaflow.h5ad")
"""

from __future__ import annotations

import os
import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

try:
    import scanpy as sc
except Exception:
    sc = None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class VelOTQuantumMetaFlowConfig:
    # Data keys
    latent_key: Optional[str] = "X_pca"
    embedding_key: str = "X_umap"
    pseudotime_key: Optional[str] = None
    cell_type_key: Optional[str] = "cell_type"

    # VelOT velocity keys
    velot_velocity_key: str = "velot_velocity_umap"
    velot_velocity_latent_key: Optional[str] = None
    require_velot_velocity: bool = True

    # Optional biological covariates
    stemness_key: Optional[str] = None
    cycle_key: Optional[str] = None

    # Latent/embedding setup
    n_latent_pcs_if_needed: int = 30
    standardize_latent: bool = True

    # VelOT UMAP velocity -> latent velocity
    velocity_to_latent_method: str = "local_linear"   # "local_linear" or "nearest_future"
    velocity_to_latent_k: int = 30
    velocity_to_latent_ridge: float = 1e-3
    velot_embedding_dt_for_lift: float = 1.0
    rescale_velot_to_ot_norm: bool = True

    # Pseudotime bins and OT
    n_time_bins: int = 8
    min_cells_per_bin: int = 30
    max_cells_per_bin_ot: int = 800
    ot_epsilon: float = 0.05
    ot_iters: int = 80

    # Flow matching
    use_ot_flow_matching: bool = True
    flow_hidden: int = 256
    flow_layers: int = 4
    flow_epochs: int = 1200
    flow_batch_size: int = 1024
    flow_lr: float = 1e-3
    flow_noise: float = 0.01
    flow_velot_alignment_weight: float = 0.15
    flow_velot_alignment_mode: str = "cosine+mse"

    # Final velocity
    final_velocity_mode: str = "blend"                # "blend", "velot_only", "ot_only"
    velot_velocity_weight: float = 0.50
    future_dt: float = 0.15

    # Estimator choice
    estimator_mode: str = "quantum_msm"               # "vamp", "quantum_msm", "both"

    # Number of meta-states
    n_metastates: Union[int, str] = 8                 # int or "auto"
    metastate_candidates: Sequence[int] = field(default_factory=lambda: [5, 6, 7, 8, 9, 10, 12])

    # VAMP estimator
    vamp_hidden: int = 256
    vamp_layers: int = 3
    vamp_dropout: float = 0.05
    vamp_epochs: int = 1200
    vamp_pretrain_epochs_auto: int = 350
    vamp_batch_size: int = 2048
    vamp_lr: float = 1e-3
    vamp_lambda_balance: float = 0.20
    vamp_lambda_sharp: float = 0.02
    vamp_lambda_orth: float = 0.02

    # Quantum-inspired MSM estimator
    quantum_hidden: int = 256
    quantum_layers: int = 3
    quantum_dropout: float = 0.05
    quantum_epochs: int = 1400
    quantum_pretrain_epochs_auto: int = 400
    quantum_batch_size: int = 2048
    quantum_lr: float = 1e-3

    # QuantumMSM losses
    quantum_lambda_balance: float = 0.20
    quantum_lambda_sharp: float = 0.015
    quantum_lambda_density_decoherence: float = 0.015
    quantum_lambda_temporal_overlap: float = 0.05
    quantum_lambda_phase_smoothness: float = 0.005

    # Quantum transition matrix
    # T = (1 - blend) * T_amplitude + blend * T_classical_prob
    quantum_classical_transition_blend: float = 0.25

    # Quantum model velocity features
    quantum_append_velocity_features: bool = True
    quantum_velocity_feature_scale: float = 1.0

    # State annotation thresholds
    terminal_quantile: float = 0.75
    source_quantile: float = 0.75
    branch_quantile: float = 0.75
    cycle_quantile: float = 0.75

    # Plotting
    max_arrows_plot: int = 1500
    max_celltypes_camembert: int = 8
    max_membership_panels: int = 12
    output_dir: str = "velot_quantum_metaflow_outputs"
    figure_dpi: int = 300

    # Reproducibility and compute
    random_state: int = 0
    device: Optional[str] = None


# =============================================================================
# General utilities
# =============================================================================

def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: Optional[str] = None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_numeric_obs(adata, key: Optional[str], default: float = 0.0, zscore: bool = True) -> Optional[np.ndarray]:
    if key is None:
        return None
    if key not in adata.obs:
        warnings.warn(f"obs key {key!r} not found; ignoring it.")
        return None
    x = pd.to_numeric(adata.obs[key], errors="coerce").to_numpy(dtype=float)
    x = np.nan_to_num(x, nan=default, posinf=default, neginf=default)
    if zscore and np.std(x) > 0:
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)
    return x.astype(np.float32)


def robust_median_norm(V: np.ndarray, eps: float = 1e-8) -> float:
    n = np.linalg.norm(V, axis=1)
    n = n[np.isfinite(n)]
    if len(n) == 0:
        return 1.0
    return max(float(np.median(n)), eps)


def rescale_to_reference_norm(V: np.ndarray, V_ref: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    scale = robust_median_norm(V_ref, eps) / robust_median_norm(V, eps)
    return (V * scale).astype(np.float32)


def cosine_similarity_rows(A: np.ndarray, B: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    num = np.sum(A * B, axis=1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + eps
    return (num / den).astype(np.float32)


def zscore_vector(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=float))
    if x.std() > 0:
        x = (x - x.mean()) / (x.std() + 1e-8)
    return x.astype(np.float32)


def row_normalize(M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return M / (M.sum(axis=1, keepdims=True) + eps)


# =============================================================================
# Latent space, embedding and pseudotime
# =============================================================================

def get_or_compute_latent(adata, cfg: VelOTQuantumMetaFlowConfig) -> Tuple[np.ndarray, str, Optional[StandardScaler]]:
    if cfg.latent_key is not None and cfg.latent_key in adata.obsm:
        Z_raw = np.asarray(adata.obsm[cfg.latent_key], dtype=np.float32)
        used = cfg.latent_key
    else:
        if sc is None:
            raise ImportError("scanpy is needed to compute PCA automatically. Provide adata.obsm[latent_key] or install scanpy.")
        if "X_pca" not in adata.obsm:
            sc.tl.pca(adata, n_comps=cfg.n_latent_pcs_if_needed, svd_solver="arpack")
        Z_raw = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
        used = "X_pca"

    if cfg.standardize_latent:
        scaler = StandardScaler()
        Z = scaler.fit_transform(Z_raw).astype(np.float32)
    else:
        scaler = None
        Z = Z_raw.astype(np.float32)

    return Z, used, scaler


def get_or_compute_embedding(adata, Z: np.ndarray, cfg: VelOTQuantumMetaFlowConfig) -> Tuple[np.ndarray, str]:
    if cfg.embedding_key in adata.obsm:
        E = np.asarray(adata.obsm[cfg.embedding_key], dtype=np.float32)
        if E.ndim != 2 or E.shape[1] < 2:
            raise ValueError(f"adata.obsm[{cfg.embedding_key!r}] must be n_cells x >=2.")
        return E[:, :2], cfg.embedding_key

    if sc is None:
        warnings.warn("No embedding found and scanpy unavailable. Using first two latent dimensions.")
        return Z[:, :2].astype(np.float32), "latent_first2"

    if "neighbors" not in adata.uns:
        use_rep = cfg.latent_key if cfg.latent_key in adata.obsm else None
        sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.umap(adata)
    return np.asarray(adata.obsm["X_umap"], dtype=np.float32)[:, :2], "X_umap_auto"


def get_or_compute_pseudotime(adata, Z: np.ndarray, cfg: VelOTQuantumMetaFlowConfig) -> Tuple[np.ndarray, str]:
    candidate_keys: List[str] = []
    if cfg.pseudotime_key is not None:
        candidate_keys.append(cfg.pseudotime_key)
    candidate_keys += [
        "velot_pseudotime",
        "velot_metaflow_pseudotime",
        "metaflow_pseudotime",
        "dpt_pseudotime",
        "pseudotime",
        "palantir_pseudotime",
    ]

    for key in candidate_keys:
        if key in adata.obs:
            pt = pd.to_numeric(adata.obs[key], errors="coerce").to_numpy(dtype=float)
            source = key
            break
    else:
        if sc is not None:
            if "neighbors" not in adata.uns:
                use_rep = cfg.latent_key if cfg.latent_key in adata.obsm else None
                sc.pp.neighbors(adata, use_rep=use_rep)
            if "X_diffmap" not in adata.obsm:
                sc.tl.diffmap(adata)
            diff = np.asarray(adata.obsm["X_diffmap"])
            root = int(np.nanargmin(diff[:, 1] if diff.shape[1] > 1 else diff[:, 0]))
            adata.uns["iroot"] = root
            sc.tl.dpt(adata)
            pt = pd.to_numeric(adata.obs["dpt_pseudotime"], errors="coerce").to_numpy(dtype=float)
            source = "dpt_pseudotime_auto"
        else:
            pt = pd.Series(Z[:, 0]).rank(method="average").to_numpy(dtype=float)
            source = "latent_dim0_rank"

    pt = np.asarray(pt, dtype=float)
    finite = np.isfinite(pt)
    if finite.sum() < 10:
        raise ValueError("Pseudotime contains too few finite values.")
    pt[~finite] = np.nanmedian(pt[finite])
    pt = (pt - pt.min()) / (pt.max() - pt.min() + 1e-8)
    return pt.astype(np.float32), source


def make_pseudotime_bins(pt: np.ndarray, n_bins: int, min_cells: int) -> np.ndarray:
    bins = pd.qcut(pt, q=n_bins, labels=False, duplicates="drop")
    bins = np.asarray(bins, dtype=float)
    bins[np.isnan(bins)] = 0
    bins = bins.astype(int)

    uniq = np.unique(bins)
    remap = {u: i for i, u in enumerate(uniq)}
    bins = np.array([remap[x] for x in bins], dtype=int)

    counts = pd.Series(bins).value_counts().sort_index()
    if (counts < min_cells).any():
        warnings.warn(
            f"Some pseudotime bins have fewer than {min_cells} cells: {counts.to_dict()}. "
            "Consider reducing n_time_bins or improving pseudotime."
        )
    return bins


# =============================================================================
# VelOT velocity handling
# =============================================================================

def get_velot_velocity_embedding(adata, cfg: VelOTQuantumMetaFlowConfig) -> Optional[np.ndarray]:
    if cfg.velot_velocity_key in adata.obsm:
        V = np.asarray(adata.obsm[cfg.velot_velocity_key], dtype=np.float32)
        if V.ndim != 2 or V.shape[1] < 2:
            raise ValueError(f"adata.obsm[{cfg.velot_velocity_key!r}] must be n_cells x >=2.")
        return np.nan_to_num(V[:, :2].astype(np.float32))

    if cfg.require_velot_velocity:
        raise KeyError(
            f"Required key adata.obsm[{cfg.velot_velocity_key!r}] not found. "
            "Set require_velot_velocity=False or provide the key."
        )

    warnings.warn(f"VelOT velocity key {cfg.velot_velocity_key!r} not found; continuing without embedding velocity.")
    return None


def get_velot_velocity_latent_direct(adata, cfg: VelOTQuantumMetaFlowConfig, Z: np.ndarray) -> Optional[np.ndarray]:
    if cfg.velot_velocity_latent_key is None:
        return None
    if cfg.velot_velocity_latent_key not in adata.obsm:
        warnings.warn(f"velot_velocity_latent_key={cfg.velot_velocity_latent_key!r} not found; using embedding lift if possible.")
        return None
    V = np.asarray(adata.obsm[cfg.velot_velocity_latent_key], dtype=np.float32)
    if V.shape != Z.shape:
        raise ValueError(f"Latent velocity shape {V.shape} does not match latent shape {Z.shape}.")
    return np.nan_to_num(V.astype(np.float32))


def lift_embedding_velocity_to_latent_local_linear(
    E: np.ndarray,
    Z: np.ndarray,
    V_emb: np.ndarray,
    k: int = 30,
    ridge: float = 1e-3,
) -> np.ndarray:
    """
    Local-linear lift:
        fit local dZ ≈ dE @ B_i in an embedding neighborhood
        V_latent_i = V_embedding_i @ B_i.
    """
    n, d = Z.shape
    k = min(k, n)
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(E)
    _, ind = nn.kneighbors(E)
    V_lat = np.zeros((n, d), dtype=np.float32)
    I2 = np.eye(E.shape[1], dtype=np.float32)

    for i in range(n):
        nb = ind[i]
        X = E[nb] - E[i]   # k x 2
        Y = Z[nb] - Z[i]   # k x d
        XtX = X.T @ X + ridge * I2
        XtY = X.T @ Y
        try:
            B = np.linalg.solve(XtX, XtY)
        except np.linalg.LinAlgError:
            B = np.linalg.lstsq(XtX, XtY, rcond=1e-6)[0]
        V_lat[i] = V_emb[i] @ B

    return np.nan_to_num(V_lat.astype(np.float32))


def lift_embedding_velocity_to_latent_nearest_future(
    E: np.ndarray,
    Z: np.ndarray,
    V_emb: np.ndarray,
    dt: float = 1.0,
    k: int = 10,
) -> np.ndarray:
    E_future = E + dt * V_emb
    k = min(k, len(E))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(E)
    dist, ind = nn.kneighbors(E_future)
    sigma = np.median(dist[:, -1]) + 1e-8
    w = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    w = w / (w.sum(axis=1, keepdims=True) + 1e-8)
    Z_future = np.einsum("nk,nkd->nd", w, Z[ind])
    V_lat = (Z_future - Z) / max(dt, 1e-8)
    return np.nan_to_num(V_lat.astype(np.float32))


def obtain_velot_velocity_latent(
    adata,
    E: np.ndarray,
    Z: np.ndarray,
    cfg: VelOTQuantumMetaFlowConfig,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    V_direct = get_velot_velocity_latent_direct(adata, cfg, Z)
    V_emb = get_velot_velocity_embedding(adata, cfg)

    if V_direct is not None:
        return V_direct, V_emb, cfg.velot_velocity_latent_key or "direct_latent"

    if V_emb is None:
        return None, None, "none"

    method = cfg.velocity_to_latent_method.lower()
    if method == "local_linear":
        V_lat = lift_embedding_velocity_to_latent_local_linear(
            E, Z, V_emb,
            k=cfg.velocity_to_latent_k,
            ridge=cfg.velocity_to_latent_ridge,
        )
        source = f"{cfg.velot_velocity_key}->latent_local_linear"
    elif method == "nearest_future":
        V_lat = lift_embedding_velocity_to_latent_nearest_future(
            E, Z, V_emb,
            dt=cfg.velot_embedding_dt_for_lift,
            k=cfg.velocity_to_latent_k,
        )
        source = f"{cfg.velot_velocity_key}->latent_nearest_future"
    else:
        raise ValueError("velocity_to_latent_method must be 'local_linear' or 'nearest_future'.")

    return V_lat.astype(np.float32), V_emb.astype(np.float32), source


# =============================================================================
# OT couplings and flow matching
# =============================================================================

def pairwise_sqdist_torch(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    x2 = (X ** 2).sum(dim=1, keepdim=True)
    y2 = (Y ** 2).sum(dim=1, keepdim=True).T
    C = x2 + y2 - 2.0 * X @ Y.T
    return torch.clamp(C, min=0.0)


def sinkhorn_coupling_torch(
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float = 0.05,
    n_iter: int = 80,
    device: str = "cpu",
) -> np.ndarray:
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    C = pairwise_sqdist_torch(X_t, Y_t)
    med = torch.median(C.detach())
    C = C / (med + 1e-8)

    n, m = C.shape
    a = torch.full((n,), 1.0 / n, device=device)
    b = torch.full((m,), 1.0 / m, device=device)

    K = torch.exp(-C / epsilon).clamp_min(1e-30)
    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(n_iter):
        u = a / (K @ v + 1e-12)
        v = b / (K.T @ u + 1e-12)

    gamma = u[:, None] * K * v[None, :]
    gamma = gamma / (gamma.sum() + 1e-12)
    return gamma.detach().cpu().numpy().astype(np.float32)


@dataclass
class OTPairBank:
    source_indices: np.ndarray
    target_indices: np.ndarray
    gamma: np.ndarray
    t0_bin: int
    t1_bin: int


def build_ot_pair_banks(
    Z: np.ndarray,
    bins: np.ndarray,
    cfg: VelOTQuantumMetaFlowConfig,
    device: str,
) -> List[OTPairBank]:
    rng = np.random.default_rng(cfg.random_state)
    banks: List[OTPairBank] = []
    unique_bins = np.unique(bins)

    for b0, b1 in zip(unique_bins[:-1], unique_bins[1:]):
        idx0 = np.where(bins == b0)[0]
        idx1 = np.where(bins == b1)[0]
        if len(idx0) < 2 or len(idx1) < 2:
            continue
        if len(idx0) > cfg.max_cells_per_bin_ot:
            idx0 = rng.choice(idx0, cfg.max_cells_per_bin_ot, replace=False)
        if len(idx1) > cfg.max_cells_per_bin_ot:
            idx1 = rng.choice(idx1, cfg.max_cells_per_bin_ot, replace=False)

        gamma = sinkhorn_coupling_torch(
            Z[idx0], Z[idx1],
            epsilon=cfg.ot_epsilon,
            n_iter=cfg.ot_iters,
            device=device,
        )
        banks.append(OTPairBank(idx0, idx1, gamma, int(b0), int(b1)))

    if len(banks) == 0:
        raise RuntimeError("Could not build OT couplings between adjacent pseudotime bins.")
    return banks


def sample_ot_pairs(
    banks: List[OTPairBank],
    Z: np.ndarray,
    pt: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bank = banks[rng.integers(0, len(banks))]
    flat = bank.gamma.ravel().astype(np.float64)
    flat = flat / (flat.sum() + 1e-12)
    draw = rng.choice(flat.size, size=batch_size, replace=True, p=flat)
    i_local, j_local = np.unravel_index(draw, bank.gamma.shape)
    i = bank.source_indices[i_local]
    j = bank.target_indices[j_local]
    return Z[i], Z[j], pt[i], pt[j], i, j


class FlowField(nn.Module):
    def __init__(self, d_in: int, hidden: int = 256, n_layers: int = 4):
        super().__init__()
        layers: List[nn.Module] = []
        d = d_in + 1
        for _ in range(n_layers):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.SiLU())
            d = hidden
        layers.append(nn.Linear(d, d_in))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        if tau.ndim == 1:
            tau = tau[:, None]
        return self.net(torch.cat([z, tau], dim=1))


def velocity_alignment_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mode: str = "cosine+mse",
    eps: float = 1e-8,
) -> torch.Tensor:
    mode = mode.lower()
    losses = []
    if "cosine" in mode:
        pred_n = pred / (torch.norm(pred, dim=1, keepdim=True) + eps)
        targ_n = target / (torch.norm(target, dim=1, keepdim=True) + eps)
        losses.append((1.0 - torch.sum(pred_n * targ_n, dim=1)).mean())
    if "mse" in mode:
        pred_norm = torch.median(torch.norm(pred.detach(), dim=1)).clamp_min(eps)
        targ_norm = torch.median(torch.norm(target.detach(), dim=1)).clamp_min(eps)
        target_scaled = target * (pred_norm / targ_norm)
        losses.append(F.mse_loss(pred, target_scaled))
    if not losses:
        return torch.zeros((), device=pred.device)
    return sum(losses) / len(losses)


def train_flow_field(
    Z: np.ndarray,
    pt: np.ndarray,
    banks: List[OTPairBank],
    V_velot_latent: Optional[np.ndarray],
    cfg: VelOTQuantumMetaFlowConfig,
    device: str,
) -> FlowField:
    rng = np.random.default_rng(cfg.random_state + 17)
    model = FlowField(Z.shape[1], cfg.flow_hidden, cfg.flow_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.flow_lr, weight_decay=1e-4)

    V_velot_t = None
    if V_velot_latent is not None:
        V_velot_t = torch.tensor(V_velot_latent, dtype=torch.float32, device=device)

    for epoch in range(cfg.flow_epochs):
        x0, x1, t0, t1, idx0, _ = sample_ot_pairs(banks, Z, pt, cfg.flow_batch_size, rng)

        x0_t = torch.tensor(x0, dtype=torch.float32, device=device)
        x1_t = torch.tensor(x1, dtype=torch.float32, device=device)
        t0_t = torch.tensor(t0, dtype=torch.float32, device=device)
        t1_t = torch.tensor(t1, dtype=torch.float32, device=device)

        lam = torch.rand(x0_t.shape[0], device=device)
        lam_col = lam[:, None]
        tau = (1.0 - lam) * t0_t + lam * t1_t
        z_tau = (1.0 - lam_col) * x0_t + lam_col * x1_t
        if cfg.flow_noise > 0:
            z_tau = z_tau + cfg.flow_noise * torch.randn_like(z_tau)

        dt = (t1_t - t0_t).clamp_min(1e-3)[:, None]
        target_ot_v = (x1_t - x0_t) / dt
        pred_v = model(z_tau, tau)

        loss_fm = F.mse_loss(pred_v, target_ot_v)
        loss = loss_fm
        loss_align_value = torch.zeros((), device=device)

        if V_velot_t is not None and cfg.flow_velot_alignment_weight > 0:
            idx0_t = torch.tensor(idx0, dtype=torch.long, device=device)
            pred_source_v = model(x0_t, t0_t)
            target_velot_v = V_velot_t[idx0_t]
            loss_align_value = velocity_alignment_loss(
                pred_source_v,
                target_velot_v,
                mode=cfg.flow_velot_alignment_mode,
            )
            loss = loss + cfg.flow_velot_alignment_weight * loss_align_value

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if epoch % 200 == 0 or epoch == cfg.flow_epochs - 1:
            print(
                f"[OT-flow] epoch={epoch:04d} "
                f"fm={loss_fm.item():.5f} align={loss_align_value.item():.5f} total={loss.item():.5f}"
            )

    return model


@torch.no_grad()
def predict_flow(
    model: FlowField,
    Z: np.ndarray,
    pt: np.ndarray,
    device: str,
    batch_size: int = 8192,
) -> np.ndarray:
    model.eval()
    outs = []
    for start in range(0, Z.shape[0], batch_size):
        sl = slice(start, min(start + batch_size, Z.shape[0]))
        z_t = torch.tensor(Z[sl], dtype=torch.float32, device=device)
        pt_t = torch.tensor(pt[sl], dtype=torch.float32, device=device)
        outs.append(model(z_t, pt_t).cpu().numpy())
    return np.vstack(outs).astype(np.float32)


def combine_velocities(
    V_ot: Optional[np.ndarray],
    V_velot: Optional[np.ndarray],
    cfg: VelOTQuantumMetaFlowConfig,
) -> Tuple[np.ndarray, str, Optional[np.ndarray], Optional[np.ndarray]]:
    mode = cfg.final_velocity_mode.lower()

    if mode == "velot_only":
        if V_velot is None:
            raise ValueError("final_velocity_mode='velot_only' requires VelOT velocity.")
        return V_velot.astype(np.float32), "velot_only", V_ot, V_velot

    if mode == "ot_only":
        if V_ot is None:
            raise ValueError("final_velocity_mode='ot_only' requires OT-flow matching.")
        return V_ot.astype(np.float32), "ot_only", V_ot, V_velot

    if mode != "blend":
        raise ValueError("final_velocity_mode must be 'blend', 'velot_only', or 'ot_only'.")

    if V_ot is None and V_velot is None:
        raise ValueError("No velocity field available.")
    if V_ot is None:
        return V_velot.astype(np.float32), "blend_fallback_velot_only", V_ot, V_velot
    if V_velot is None:
        return V_ot.astype(np.float32), "blend_fallback_ot_only", V_ot, V_velot

    Vv = V_velot
    Vo = V_ot
    if cfg.rescale_velot_to_ot_norm:
        Vv = rescale_to_reference_norm(Vv, Vo)

    alpha = float(np.clip(cfg.velot_velocity_weight, 0.0, 1.0))
    V = (1.0 - alpha) * Vo + alpha * Vv
    return V.astype(np.float32), f"blend_ot_{1-alpha:.2f}_velot_{alpha:.2f}", Vo.astype(np.float32), Vv.astype(np.float32)


# =============================================================================
# Diagnostics
# =============================================================================

def estimate_divergence_knn(Z: np.ndarray, V: np.ndarray, k: int = 30) -> np.ndarray:
    n = Z.shape[0]
    k = min(k, n)
    nn = NearestNeighbors(n_neighbors=k).fit(Z)
    _, ind = nn.kneighbors(Z)
    div = np.zeros(n, dtype=np.float32)

    for i in range(n):
        nb = ind[i]
        X = Z[nb] - Z[i]
        Y = V[nb] - V[i]
        try:
            B, *_ = np.linalg.lstsq(X, Y, rcond=1e-3)
            div[i] = np.trace(B)
        except Exception:
            div[i] = 0.0

    return zscore_vector(div)


def estimate_embedding_curl(E: np.ndarray, V_emb: Optional[np.ndarray], k: int = 30) -> Optional[np.ndarray]:
    if V_emb is None:
        return None

    n = E.shape[0]
    k = min(k, n)
    nn = NearestNeighbors(n_neighbors=k).fit(E)
    _, ind = nn.kneighbors(E)

    curl = np.zeros(n, dtype=np.float32)
    for i in range(n):
        nb = ind[i]
        X = E[nb] - E[i]
        Y = V_emb[nb] - V_emb[i]
        try:
            B, *_ = np.linalg.lstsq(X, Y, rcond=1e-3)
            dVx_dy = B[1, 0]
            dVy_dx = B[0, 1]
            curl[i] = dVy_dx - dVx_dy
        except Exception:
            curl[i] = 0.0

    return zscore_vector(np.abs(curl))


def make_quantum_features(
    Z: np.ndarray,
    V_final: np.ndarray,
    divergence: Optional[np.ndarray],
    curl_score: Optional[np.ndarray],
    velocity_alignment: Optional[np.ndarray],
    cfg: VelOTQuantumMetaFlowConfig,
) -> np.ndarray:
    """
    Optional extra features for the quantum-inspired estimator:
        speed, divergence, curl, velocity alignment.
    The complex amplitude model still works with Z only; these features
    help encode phase/transport information.
    """
    if not cfg.quantum_append_velocity_features:
        return Z.astype(np.float32)

    feats = []
    speed = np.linalg.norm(V_final, axis=1)
    feats.append(zscore_vector(speed)[:, None])

    if divergence is not None:
        feats.append(zscore_vector(divergence)[:, None])
    if curl_score is not None:
        feats.append(zscore_vector(curl_score)[:, None])
    if velocity_alignment is not None:
        feats.append(zscore_vector(velocity_alignment)[:, None])

    if len(feats) == 0:
        return Z.astype(np.float32)

    F_extra = np.hstack(feats).astype(np.float32) * float(cfg.quantum_velocity_feature_scale)
    return np.hstack([Z, F_extra]).astype(np.float32)


def compute_row_entropy(T: np.ndarray) -> np.ndarray:
    H = -np.sum(T * np.log(T + 1e-8), axis=1)
    H = H / (np.log(T.shape[1]) + 1e-8)
    return H.astype(np.float32)


def compute_terminal_committors(T: np.ndarray, terminal_states: Sequence[int]) -> np.ndarray:
    K = T.shape[0]
    terminals = np.array(sorted(set(int(x) for x in terminal_states)), dtype=int)
    if len(terminals) == 0:
        return np.zeros((K, 0), dtype=np.float32)

    terminal_set = set(terminals.tolist())
    transient = np.array([i for i in range(K) if i not in terminal_set], dtype=int)

    B = np.zeros((K, len(terminals)), dtype=np.float32)
    for c, t in enumerate(terminals):
        B[t, c] = 1.0

    if len(transient) > 0:
        Q = T[np.ix_(transient, transient)]
        R = T[np.ix_(transient, terminals)]
        A = np.eye(len(transient)) - Q
        try:
            X = np.linalg.solve(A, R)
        except np.linalg.LinAlgError:
            X = np.linalg.lstsq(A, R, rcond=1e-6)[0]
        B[transient] = X

    B = np.clip(B, 0, 1)
    B = B / (B.sum(axis=1, keepdims=True) + 1e-8)
    return B.astype(np.float32)


def classify_metastates(
    T: np.ndarray,
    q: np.ndarray,
    divergence: Optional[np.ndarray],
    stemness: Optional[np.ndarray],
    cycle_score: Optional[np.ndarray],
    curl_score: Optional[np.ndarray],
    velocity_alignment: Optional[np.ndarray],
    cfg: VelOTQuantumMetaFlowConfig,
) -> pd.DataFrame:
    K = T.shape[0]
    self_t = np.diag(T)
    outgoing = T.sum(axis=1) - self_t
    incoming = T.sum(axis=0) - self_t
    row_entropy = compute_row_entropy(T)

    def weighted_mean(x: Optional[np.ndarray]) -> np.ndarray:
        if x is None:
            return np.zeros(K, dtype=np.float32)
        return np.array([np.average(x, weights=q[:, k] + 1e-8) for k in range(K)], dtype=np.float32)

    div_state = weighted_mean(divergence)
    stem_state = weighted_mean(stemness)
    cycle_state = weighted_mean(cycle_score)
    curl_state = weighted_mean(curl_score)
    align_state = weighted_mean(velocity_alignment)

    source_score = outgoing - incoming + 0.35 * div_state + 0.35 * stem_state
    sink_score = self_t + incoming - outgoing - 0.35 * div_state
    branch_score = row_entropy + outgoing - self_t
    recurrent_score = self_t + 0.30 * cycle_state + 0.30 * curl_state
    coherent_velocity_score = align_state

    thresholds = {
        "initial/source": np.quantile(source_score, cfg.source_quantile),
        "terminal/sink": np.quantile(sink_score, cfg.terminal_quantile),
        "branching/saddle": np.quantile(branch_score, cfg.branch_quantile),
        "cycling/recurrent": np.quantile(recurrent_score, cfg.cycle_quantile),
    }

    labels = []
    for k in range(K):
        candidates = {
            "initial/source": source_score[k] - thresholds["initial/source"],
            "terminal/sink": sink_score[k] - thresholds["terminal/sink"],
            "branching/saddle": branch_score[k] - thresholds["branching/saddle"],
            "cycling/recurrent": recurrent_score[k] - thresholds["cycling/recurrent"],
        }
        best = max(candidates, key=candidates.get)
        if candidates[best] < 0:
            best = "intermediate/transient"
        labels.append(best)

    return pd.DataFrame({
        "metastate": [f"M{k}" for k in range(K)],
        "label": labels,
        "occupancy": q.mean(axis=0),
        "self_transition": self_t,
        "incoming": incoming,
        "outgoing": outgoing,
        "row_entropy": row_entropy,
        "divergence": div_state,
        "stemness": stem_state,
        "cycle_score": cycle_state,
        "embedding_curl": curl_state,
        "velocity_alignment": coherent_velocity_score,
        "source_score": source_score,
        "sink_score": sink_score,
        "branch_score": branch_score,
        "recurrent_score": recurrent_score,
    })


# =============================================================================
# VAMPFlow estimator
# =============================================================================

class VAMPMetaStateNet(nn.Module):
    def __init__(self, d_in: int, n_states: int, hidden: int, n_layers: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        d = d_in
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, n_states))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(z), dim=1)


def inv_sqrtm_psd_real(C: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    C = 0.5 * (C + C.T)
    eigvals, eigvecs = torch.linalg.eigh(C)
    eigvals = torch.clamp(eigvals, min=eps)
    return eigvecs @ torch.diag(torch.rsqrt(eigvals)) @ eigvecs.T


def vamp2_score(q0: torch.Tensor, q1: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    q0c = q0 - q0.mean(dim=0, keepdim=True)
    q1c = q1 - q1.mean(dim=0, keepdim=True)
    n = q0.shape[0]
    k = q0.shape[1]
    eye = torch.eye(k, device=q0.device)
    C00 = (q0c.T @ q0c) / max(n - 1, 1) + eps * eye
    C11 = (q1c.T @ q1c) / max(n - 1, 1) + eps * eye
    C01 = (q0c.T @ q1c) / max(n - 1, 1)
    K = inv_sqrtm_psd_real(C00, eps) @ C01 @ inv_sqrtm_psd_real(C11, eps)
    return torch.sum(K ** 2)


def prob_balance_loss(q: torch.Tensor) -> torch.Tensor:
    k = q.shape[1]
    p = q.mean(dim=0)
    target = torch.full_like(p, 1.0 / k)
    return F.mse_loss(p, target)


def prob_sharpness_loss(q: torch.Tensor) -> torch.Tensor:
    return -torch.sum(q * torch.log(q + 1e-8), dim=1).mean()


def prob_orthogonality_loss(q: torch.Tensor) -> torch.Tensor:
    qn = q / (q.norm(dim=0, keepdim=True) + 1e-8)
    G = qn.T @ qn
    eye = torch.eye(q.shape[1], device=q.device)
    return ((G - eye) ** 2).mean()


def train_vamp_model(
    Z: np.ndarray,
    Z_future: np.ndarray,
    n_states: int,
    cfg: VelOTQuantumMetaFlowConfig,
    device: str,
    n_epochs: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[VAMPMetaStateNet, float]:
    if n_epochs is None:
        n_epochs = cfg.vamp_epochs

    Z_t = torch.tensor(Z, dtype=torch.float32, device=device)
    Zf_t = torch.tensor(Z_future, dtype=torch.float32, device=device)

    model = VAMPMetaStateNet(
        d_in=Z.shape[1],
        n_states=n_states,
        hidden=cfg.vamp_hidden,
        n_layers=cfg.vamp_layers,
        dropout=cfg.vamp_dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.vamp_lr, weight_decay=1e-4)
    n = Z.shape[0]
    last_score = np.nan

    for epoch in range(n_epochs):
        idx = torch.randint(0, n, size=(min(cfg.vamp_batch_size, n),), device=device)
        q0 = model(Z_t[idx])
        q1 = model(Zf_t[idx])

        score = vamp2_score(q0, q1)
        loss = (
            -score
            + cfg.vamp_lambda_balance * prob_balance_loss(q0)
            + cfg.vamp_lambda_sharp * prob_sharpness_loss(q0)
            + cfg.vamp_lambda_orth * prob_orthogonality_loss(q0)
        )

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        last_score = float(score.detach().cpu())
        if verbose and (epoch % 200 == 0 or epoch == n_epochs - 1):
            print(f"[VAMP K={n_states}] epoch={epoch:04d} VAMP2={last_score:.4f} loss={loss.item():.4f}")

    return model, last_score


@torch.no_grad()
def infer_vamp_states(
    model: VAMPMetaStateNet,
    Z: np.ndarray,
    Z_future: np.ndarray,
    device: str,
    batch_size: int = 8192,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    q0s, q1s = [], []
    for start in range(0, Z.shape[0], batch_size):
        sl = slice(start, min(start + batch_size, Z.shape[0]))
        z_t = torch.tensor(Z[sl], dtype=torch.float32, device=device)
        zf_t = torch.tensor(Z_future[sl], dtype=torch.float32, device=device)
        q0s.append(model(z_t).cpu().numpy())
        q1s.append(model(zf_t).cpu().numpy())
    q0 = np.vstack(q0s).astype(np.float32)
    q1 = np.vstack(q1s).astype(np.float32)
    return q0, q1, q0.argmax(axis=1)


def compute_soft_transition_matrix(q0: np.ndarray, q1: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    T = q0.T @ q1
    T = T / (q0.sum(axis=0, keepdims=True).T + eps)
    return row_normalize(T).astype(np.float32)


def auto_select_k_vamp(
    Z: np.ndarray,
    Z_future: np.ndarray,
    cfg: VelOTQuantumMetaFlowConfig,
    device: str,
) -> int:
    best_k, best_crit = None, -np.inf
    for k in cfg.metastate_candidates:
        model, score = train_vamp_model(
            Z, Z_future, int(k), cfg, device,
            n_epochs=cfg.vamp_pretrain_epochs_auto,
            verbose=False,
        )
        q0, _, _ = infer_vamp_states(model, Z, Z_future, device)
        occ = q0.mean(axis=0)
        min_occ = float(occ.min())
        H_occ = float(-(occ * np.log(occ + 1e-8)).sum() / np.log(k))
        criterion = score - 0.08 * k + 2.0 * min_occ + 0.50 * H_occ
        print(f"[auto-K VAMP] K={k:02d} VAMP2={score:.3f} min_occ={min_occ:.3f} H_occ={H_occ:.3f} criterion={criterion:.3f}")
        if criterion > best_crit:
            best_crit = criterion
            best_k = int(k)
    print(f"[auto-K VAMP] selected K={best_k}")
    return int(best_k)


# =============================================================================
# Quantum-inspired MSM estimator
# =============================================================================

class QuantumMSMNet(nn.Module):
    """
    Quantum-inspired meta-state network.

    It outputs:
        probs q_k(z) = |psi_k(z)|^2
        phases phi_k(z)
        complex amplitudes psi_k(z) = sqrt(q_k(z)) * exp(i * phi_k(z))

    The model is not a quantum-computing model. It is a classical neural network
    with complex-valued outputs inspired by density matrices, quantum channels,
    coherence/decoherence and open-system Markov semigroups.
    """
    def __init__(self, d_in: int, n_states: int, hidden: int, n_layers: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        d = d_in
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout)]
            d = hidden
        self.body = nn.Sequential(*layers)
        self.logit_head = nn.Linear(d, n_states)
        self.phase_head = nn.Linear(d, n_states)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.body(z)
        logits = self.logit_head(h)
        q = F.softmax(logits, dim=1)
        phase = math.pi * torch.tanh(self.phase_head(h))   # bounded phase in [-pi, pi]
        amp = torch.sqrt(q + 1e-8)
        psi = torch.complex(amp * torch.cos(phase), amp * torch.sin(phase))
        return q, phase, psi


def inv_sqrtm_psd_complex(C: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    C = 0.5 * (C + C.conj().T)
    eigvals, eigvecs = torch.linalg.eigh(C)
    eigvals = torch.clamp(eigvals.real, min=eps)
    D = torch.diag(torch.rsqrt(eigvals)).to(dtype=C.dtype, device=C.device)
    return eigvecs @ D @ eigvecs.conj().T


def quantum_vamp_score(psi0: torch.Tensor, psi1: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Complex analogue of a VAMP-like score:
        || rho00^{-1/2} C01 rho11^{-1/2} ||_F^2
    where psi are complex amplitudes.
    """
    n = psi0.shape[0]
    k = psi0.shape[1]
    eye = torch.eye(k, device=psi0.device, dtype=psi0.dtype)
    C00 = (psi0.conj().T @ psi0) / max(n, 1) + eps * eye
    C11 = (psi1.conj().T @ psi1) / max(n, 1) + eps * eye
    C01 = (psi0.conj().T @ psi1) / max(n, 1)

    S = inv_sqrtm_psd_complex(C00, eps) @ C01 @ inv_sqrtm_psd_complex(C11, eps)
    return torch.sum(torch.abs(S) ** 2).real


def density_decoherence_loss(psi: torch.Tensor) -> torch.Tensor:
    """
    Penalize excessive off-diagonal density-matrix mass.
    Low but nonzero weight keeps states interpretable while still allowing
    coherent transition amplitude effects.
    """
    n = psi.shape[0]
    rho = (psi.conj().T @ psi) / max(n, 1)
    diag = torch.diag(torch.diagonal(rho))
    off = rho - diag
    return torch.mean(torch.abs(off) ** 2).real


def quantum_temporal_overlap_loss(psi0: torch.Tensor, psi1: torch.Tensor) -> torch.Tensor:
    """
    Encourage coherent local temporal transport:
        maximize |<psi(z_t), psi(z_t+dt)>|^2 per cell.
    """
    overlap = torch.sum(psi0.conj() * psi1, dim=1)
    return -torch.mean(torch.abs(overlap) ** 2).real


def phase_smoothness_loss(phase0: torch.Tensor, phase1: torch.Tensor, q0: torch.Tensor) -> torch.Tensor:
    """
    Encourage phase increments to be smooth and membership-weighted.
    The wrapped circular difference avoids discontinuities at +/-pi.
    """
    dphi = torch.atan2(torch.sin(phase1 - phase0), torch.cos(phase1 - phase0))
    return torch.mean(torch.sum(q0.detach() * dphi ** 2, dim=1))


def train_quantum_msm_model(
    X: np.ndarray,
    X_future: np.ndarray,
    n_states: int,
    cfg: VelOTQuantumMetaFlowConfig,
    device: str,
    n_epochs: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[QuantumMSMNet, float]:
    if n_epochs is None:
        n_epochs = cfg.quantum_epochs

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    Xf_t = torch.tensor(X_future, dtype=torch.float32, device=device)

    model = QuantumMSMNet(
        d_in=X.shape[1],
        n_states=n_states,
        hidden=cfg.quantum_hidden,
        n_layers=cfg.quantum_layers,
        dropout=cfg.quantum_dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.quantum_lr, weight_decay=1e-4)
    n = X.shape[0]
    last_score = np.nan

    for epoch in range(n_epochs):
        idx = torch.randint(0, n, size=(min(cfg.quantum_batch_size, n),), device=device)

        q0, phase0, psi0 = model(X_t[idx])
        q1, phase1, psi1 = model(Xf_t[idx])

        score = quantum_vamp_score(psi0, psi1)
        loss_balance = prob_balance_loss(q0)
        loss_sharp = prob_sharpness_loss(q0)
        loss_decoh = 0.5 * (density_decoherence_loss(psi0) + density_decoherence_loss(psi1))
        loss_overlap = quantum_temporal_overlap_loss(psi0, psi1)
        loss_phase = phase_smoothness_loss(phase0, phase1, q0)

        loss = (
            -score
            + cfg.quantum_lambda_balance * loss_balance
            + cfg.quantum_lambda_sharp * loss_sharp
            + cfg.quantum_lambda_density_decoherence * loss_decoh
            + cfg.quantum_lambda_temporal_overlap * loss_overlap
            + cfg.quantum_lambda_phase_smoothness * loss_phase
        )

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        last_score = float(score.detach().cpu())
        if verbose and (epoch % 200 == 0 or epoch == n_epochs - 1):
            print(
                f"[QuantumMSM K={n_states}] epoch={epoch:04d} "
                f"QScore={last_score:.4f} decoh={loss_decoh.item():.4f} "
                f"overlap={-loss_overlap.item():.4f} loss={loss.item():.4f}"
            )

    return model, last_score


@torch.no_grad()
def infer_quantum_states(
    model: QuantumMSMNet,
    X: np.ndarray,
    X_future: np.ndarray,
    device: str,
    batch_size: int = 8192,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    q0s, q1s, phase0s, phase1s, psi0s, psi1s = [], [], [], [], [], []
    for start in range(0, X.shape[0], batch_size):
        sl = slice(start, min(start + batch_size, X.shape[0]))
        x_t = torch.tensor(X[sl], dtype=torch.float32, device=device)
        xf_t = torch.tensor(X_future[sl], dtype=torch.float32, device=device)
        q0, ph0, ps0 = model(x_t)
        q1, ph1, ps1 = model(xf_t)
        q0s.append(q0.cpu().numpy())
        q1s.append(q1.cpu().numpy())
        phase0s.append(ph0.cpu().numpy())
        phase1s.append(ph1.cpu().numpy())
        psi0s.append(ps0.cpu().numpy())
        psi1s.append(ps1.cpu().numpy())

    q0 = np.vstack(q0s).astype(np.float32)
    q1 = np.vstack(q1s).astype(np.float32)
    phase0 = np.vstack(phase0s).astype(np.float32)
    phase1 = np.vstack(phase1s).astype(np.float32)
    psi0 = np.vstack(psi0s).astype(np.complex64)
    psi1 = np.vstack(psi1s).astype(np.complex64)
    hard = q0.argmax(axis=1)
    return q0, q1, phase0, phase1, psi0, psi1, hard


def compute_quantum_transition_matrix(
    psi0: np.ndarray,
    psi1: np.ndarray,
    q0: np.ndarray,
    q1: np.ndarray,
    classical_blend: float = 0.25,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quantum-channel-style transition:
        A_ab = mean_i conj(psi_i,a(t)) psi_i,b(t+dt)
        T_amp_ab ∝ |A_ab|^2
    optionally blended with a classical soft-probability transition matrix.
    """
    n = psi0.shape[0]
    A = (psi0.conj().T @ psi1) / max(n, 1)
    T_amp = np.abs(A) ** 2
    T_amp = row_normalize(T_amp + eps)

    T_classical = compute_soft_transition_matrix(q0, q1, eps=eps)
    beta = float(np.clip(classical_blend, 0.0, 1.0))
    T = (1.0 - beta) * T_amp + beta * T_classical
    T = row_normalize(T + eps)
    return T.astype(np.float32), T_amp.astype(np.float32), A.astype(np.complex64)


def compute_density_matrix(psi: np.ndarray) -> np.ndarray:
    n = psi.shape[0]
    return ((psi.conj().T @ psi) / max(n, 1)).astype(np.complex64)


def quantum_coherence_per_cell(psi: np.ndarray) -> np.ndarray:
    """
    Per-cell coherence proxy:
        sum_{a != b} |psi_a psi_b*|
    Higher values mean the cell is represented as a superposition of several meta-states.
    """
    q = np.abs(psi) ** 2
    # For normalized psi: total pairwise coherence is roughly 1 - sum q^2.
    coh = 1.0 - np.sum(q ** 2, axis=1)
    return zscore_vector(coh)


def quantum_phase_resultant(psi: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Per-cell dominant-state phase, useful only as a diagnostic/visualization.
    """
    k = q.argmax(axis=1)
    phase = np.angle(psi[np.arange(len(k)), k])
    return phase.astype(np.float32)


def auto_select_k_quantum(
    X: np.ndarray,
    X_future: np.ndarray,
    cfg: VelOTQuantumMetaFlowConfig,
    device: str,
) -> int:
    best_k, best_crit = None, -np.inf
    for k in cfg.metastate_candidates:
        model, score = train_quantum_msm_model(
            X, X_future, int(k), cfg, device,
            n_epochs=cfg.quantum_pretrain_epochs_auto,
            verbose=False,
        )
        q0, q1, _, _, psi0, psi1, _ = infer_quantum_states(model, X, X_future, device)
        T, _, _ = compute_quantum_transition_matrix(
            psi0, psi1, q0, q1,
            classical_blend=cfg.quantum_classical_transition_blend,
        )
        occ = q0.mean(axis=0)
        min_occ = float(occ.min())
        H_occ = float(-(occ * np.log(occ + 1e-8)).sum() / np.log(k))
        self_t = float(np.mean(np.diag(T)))
        criterion = score - 0.08 * k + 2.0 * min_occ + 0.50 * H_occ + 0.20 * self_t
        print(
            f"[auto-K QuantumMSM] K={k:02d} QScore={score:.3f} "
            f"min_occ={min_occ:.3f} H_occ={H_occ:.3f} selfT={self_t:.3f} criterion={criterion:.3f}"
        )
        if criterion > best_crit:
            best_crit = criterion
            best_k = int(k)
    print(f"[auto-K QuantumMSM] selected K={best_k}")
    return int(best_k)


# =============================================================================
# Plotting
# =============================================================================

def savefig(path: str, cfg: VelOTQuantumMetaFlowConfig) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=cfg.figure_dpi, bbox_inches="tight")
    plt.close()


def subsample_indices(n: int, max_n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if n <= max_n:
        return np.arange(n)
    return rng.choice(np.arange(n), size=max_n, replace=False)


def plot_raw_velot_velocity_embedding(E: np.ndarray, V_emb: Optional[np.ndarray], cfg: VelOTQuantumMetaFlowConfig) -> Optional[str]:
    if V_emb is None:
        return None
    path = os.path.join(cfg.output_dir, "00_raw_velot_velocity_umap.png")
    idx = subsample_indices(E.shape[0], cfg.max_arrows_plot, cfg.random_state)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(E[:, 0], E[:, 1], s=4, alpha=0.22, linewidths=0)
    ax.quiver(E[idx, 0], E[idx, 1], V_emb[idx, 0], V_emb[idx, 1],
              angles="xy", scale_units="xy", scale=1.0, width=0.0022, alpha=0.65)
    ax.set_title('Input VelOT vector field: adata.obsm["velot_velocity_umap"]', fontsize=13, weight="bold")
    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")
    sns.despine(ax=ax)
    savefig(path, cfg)
    return path


def plot_final_velocity_embedding(E: np.ndarray, Z: np.ndarray, Z_future: np.ndarray, cfg: VelOTQuantumMetaFlowConfig) -> str:
    path = os.path.join(cfg.output_dir, "01_final_velocity_embedding.png")
    nn = NearestNeighbors(n_neighbors=1).fit(Z)
    _, ind = nn.kneighbors(Z_future)
    E_future = E[ind[:, 0]]
    dE = E_future - E
    idx = subsample_indices(E.shape[0], cfg.max_arrows_plot, cfg.random_state + 1)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(E[:, 0], E[:, 1], s=4, alpha=0.22, linewidths=0)
    ax.quiver(E[idx, 0], E[idx, 1], dE[idx, 0], dE[idx, 1],
              angles="xy", scale_units="xy", scale=1.0, width=0.0022, alpha=0.58)
    ax.set_title("Final latent velocity projected to embedding", fontsize=13, weight="bold")
    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")
    sns.despine(ax=ax)
    savefig(path, cfg)
    return path


def plot_metastates_embedding(
    E: np.ndarray,
    hard: np.ndarray,
    state_df: pd.DataFrame,
    cfg: VelOTQuantumMetaFlowConfig,
    prefix: str,
    title: str,
) -> str:
    path = os.path.join(cfg.output_dir, f"{prefix}_metastates_embedding.png")
    K = int(hard.max()) + 1
    palette = sns.color_palette("tab20", n_colors=max(K, 3))
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    sns.scatterplot(x=E[:, 0], y=E[:, 1], hue=hard, palette=palette, s=8, linewidth=0, legend=False, ax=ax)

    for k in range(K):
        mask = hard == k
        if mask.sum() == 0:
            continue
        x, y = np.median(E[mask, 0]), np.median(E[mask, 1])
        label = state_df.loc[k, "label"].split("/")[0]
        ax.text(x, y, f"M{k}\n{label}", ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.5, alpha=0.85))

    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")
    sns.despine(ax=ax)
    savefig(path, cfg)
    return path


def plot_transition_heatmap(T: np.ndarray, state_df: pd.DataFrame, cfg: VelOTQuantumMetaFlowConfig, prefix: str, title: str) -> str:
    path = os.path.join(cfg.output_dir, f"{prefix}_transition_heatmap.png")
    labels = [f"M{i}\n{state_df.loc[i, 'label'].split('/')[0]}" for i in range(T.shape[0])]
#     fig, ax = plt.subplots(figsize=(0.72 * T.shape[0] + 3.0, 0.72 * T.shape[0] + 2.5))
#     sns.heatmap(T, cmap="viridis", annot=True, fmt=".2f", square=True,
#                 xticklabels=labels, yticklabels=labels, cbar_kws={"label": "transition probability"}, ax=ax)
#     ax.set_title(title, fontsize=13, weight="bold")
#     ax.set_xlabel("future meta-state")
#     ax.set_ylabel("current meta-state")

    fig, ax = plt.subplots(
        figsize=(
            max(7, 0.55 * T.shape[1] + 3),
            max(4.8, 0.45 * T.shape[0] + 2),
        )
    )

    sns.heatmap(
        T,
        cmap="viridis",
        annot=True,
        # annot=T.size <= 120,
        fmt=".2f",
        square=True,
        linewidths=0.4,
        linecolor="white",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "transition probability"},
        ax=ax,
    )

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        fontsize=12,
    )

    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=12,
    )

    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)

    # Colorbar formatting
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("transition probability", fontsize=14)

    savefig(path, cfg)
    return path


def plot_flux_graph(T: np.ndarray, state_df: pd.DataFrame, cfg: VelOTQuantumMetaFlowConfig, prefix: str, title: str, min_edge: float = 0.08) -> str:
    path = os.path.join(cfg.output_dir, f"{prefix}_flux_graph.png")
    K = T.shape[0]
    G = nx.DiGraph()
    for k in range(K):
        G.add_node(k, label=f"M{k}\n{state_df.loc[k, 'label'].split('/')[0]}")
    for i in range(K):
        for j in range(K):
            if i != j and T[i, j] >= min_edge:
                G.add_edge(i, j, weight=float(T[i, j]))

    pos = nx.spring_layout(G, seed=cfg.random_state, weight="weight")
    fig, ax = plt.subplots(figsize=(7.6, 6.5))
    node_sizes = 850 + 8500 * state_df["occupancy"].to_numpy()
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.90, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, "label"), font_size=8, ax=ax)
    widths = [1.0 + 6.0 * G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=widths, arrows=True, arrowstyle="-|>", arrowsize=16, alpha=0.65, ax=ax)
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)
    ax.set_title(title, fontsize=13, weight="bold")
    ax.axis("off")
    savefig(path, cfg)
    return path


def plot_state_diagnostics(state_df: pd.DataFrame, cfg: VelOTQuantumMetaFlowConfig, prefix: str, title: str) -> str:
    path = os.path.join(cfg.output_dir, f"{prefix}_dynamical_diagnostics.png")
    metrics = ["source_score", "sink_score", "branch_score", "recurrent_score", "velocity_alignment"]
    df = state_df[["metastate", "label"] + metrics].melt(
        id_vars=["metastate", "label"], var_name="diagnostic", value_name="score"
    )
    fig, ax = plt.subplots(figsize=(10.2, 4.9))
    sns.barplot(data=df, x="metastate", y="score", hue="diagnostic", ax=ax)
    ax.axhline(0, lw=0.8, color="black")
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("diagnostic score")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    sns.despine(ax=ax)
    savefig(path, cfg)
    return path


def plot_camembert_celltype_by_metastate(
    hard: np.ndarray,
    adata,
    cfg: VelOTQuantumMetaFlowConfig,
    prefix: str,
    title: str,
) -> Optional[str]:
    if cfg.cell_type_key is None or cfg.cell_type_key not in adata.obs:
        warnings.warn("cell_type_key not found; skipping camembert plot.")
        return None

    path = os.path.join(cfg.output_dir, f"{prefix}_camembert_celltype_composition_by_metastate.png")
    celltypes = adata.obs[cfg.cell_type_key].astype(str).to_numpy()
    K = int(hard.max()) + 1
    global_counts = pd.Series(celltypes).value_counts()
    keep = set(global_counts.head(cfg.max_celltypes_camembert).index)
    celltypes2 = np.array([ct if ct in keep else "Other" for ct in celltypes])
    categories = list(pd.Series(celltypes2).value_counts().index)

    n_cols = min(4, K)
    n_rows = int(math.ceil(K / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.25 * n_cols, 3.05 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    palette = dict(zip(categories, sns.color_palette("tab20", n_colors=len(categories))))

    for k in range(K):
        ax = axes[k]
        vals = pd.Series(celltypes2[hard == k]).value_counts().reindex(categories, fill_value=0)
        vals = vals[vals > 0]
        if vals.sum() == 0:
            ax.axis("off")
            continue
        colors = [palette[c] for c in vals.index]
        ax.pie(vals.values, labels=None, colors=colors, startangle=90, counterclock=False,
               wedgeprops=dict(width=0.92, edgecolor="white", linewidth=0.7))
        ax.set_title(f"M{k}  n={int(vals.sum())}", fontsize=10, weight="bold")

    for ax in axes[K:]:
        ax.axis("off")

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[c], markersize=8) for c in categories]
    fig.legend(handles, categories, loc="lower center", ncol=min(5, len(categories)), frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=13, weight="bold", y=0.98)
    plt.subplots_adjust(bottom=0.12, top=0.88)
    plt.savefig(path, dpi=cfg.figure_dpi, bbox_inches="tight")
    plt.close()
    return path


def plot_committor_maps(E: np.ndarray, q: np.ndarray, B_meta: np.ndarray, terminal_states: Sequence[int], cfg: VelOTQuantumMetaFlowConfig, prefix: str, title: str) -> Optional[str]:
    if B_meta.shape[1] == 0:
        return None

    path = os.path.join(cfg.output_dir, f"{prefix}_terminal_committor_probability_maps.png")
    cell_B = q @ B_meta
    n_terms = len(terminal_states)
    n_cols = min(3, n_terms)
    n_rows = int(math.ceil(n_terms / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.1 * n_cols, 3.8 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for c, term in enumerate(terminal_states):
        ax = axes[c]
        sca = ax.scatter(E[:, 0], E[:, 1], c=cell_B[:, c], s=6, cmap="magma", linewidths=0, vmin=0, vmax=1)
        ax.set_title(f"Committor to terminal M{term}", fontsize=10, weight="bold")
        ax.set_xlabel("Embedding 1")
        ax.set_ylabel("Embedding 2")
        plt.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
        sns.despine(ax=ax)

    for ax in axes[n_terms:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=13, weight="bold")
    savefig(path, cfg)
    return path


# def plot_soft_memberships(E: np.ndarray, q: np.ndarray, cfg: VelOTQuantumMetaFlowConfig, prefix: str, title: str) -> str:
#     path = os.path.join(cfg.output_dir, f"{prefix}_soft_membership_maps.png")
#     K = min(q.shape[1], cfg.max_membership_panels)
#     n_cols = min(4, K)
#     n_rows = int(math.ceil(K / n_cols))
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.2 * n_rows))
#     axes = np.atleast_1d(axes).ravel()

#     for k in range(K):
#         ax = axes[k]
#         sca = ax.scatter(E[:, 0], E[:, 1], c=q[:, k], s=5, cmap="viridis", linewidths=0, vmin=0, vmax=1)
#         ax.set_title(f"q(M{k})", fontsize=12)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         plt.colorbar(sca, ax=ax, fraction=0.046, pad=0.03)

#     for ax in axes[K:]:
#         ax.axis("off")

#     fig.suptitle(title, fontsize=13, weight="bold")
#     savefig(path, cfg)
#     return path

def umap_axis(ax, pos=(0.02, 0.02), length=0.15, fontsize=9):
    x0, y0 = pos
    ax.axis("off")
    ax.annotate("", xy=(x0 + length, y0), xytext=(x0, y0),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))
    ax.annotate("", xy=(x0, y0 + length), xytext=(x0, y0),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))
    ax.text(x0 + length, y0 - 0.02, "UMAP1",
            transform=ax.transAxes, ha="right", va="top", fontsize=fontsize)
    ax.text(x0 - 0.02, y0 + length, "UMAP2",
            transform=ax.transAxes, ha="right", va="top",
            rotation=90, fontsize=fontsize)

def plot_soft_memberships(E: np.ndarray, q: np.ndarray, cfg: VelOTQuantumMetaFlowConfig,
                          prefix: str, title: str) -> str:

    path = os.path.join(cfg.output_dir, f"{prefix}_soft_membership_maps.png")

    K = min(q.shape[1], cfg.max_membership_panels)
    n_cols = min(4, K)
    n_rows = int(math.ceil(K / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 4 * n_rows)
    )
    axes = np.atleast_1d(axes).ravel()

    sca = None
    for k in range(K):
        ax = axes[k]

        sca = ax.scatter(
            E[:, 0], E[:, 1],
            c=q[:, k],
            s=5,
            cmap="viridis",
            linewidths=0,
            vmin=0,
            vmax=1,
        )

        ax.set_title(f"q(M{k})", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        umap_axis(ax)

    # Remove unused axes first
    for ax in axes[K:]:
        fig.delaxes(ax)

    # Shared colorbar with more space
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([1.01, 0.25, 0.01, 0.5])  # [left, bottom, width, height]
    cbar = fig.colorbar(sca, cax=cbar_ax)
    cbar.set_label("Membership probability")

    # fig.suptitle(title, fontsize=13, weight="bold")

    savefig(path, cfg)
    return path


def plot_velocity_alignment(E: np.ndarray, alignment: Optional[np.ndarray], cfg: VelOTQuantumMetaFlowConfig) -> Optional[str]:
    if alignment is None:
        return None
    path = os.path.join(cfg.output_dir, "02_ot_flow_vs_velot_velocity_alignment.png")
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    sca = ax.scatter(E[:, 0], E[:, 1], c=alignment, s=7, cmap="coolwarm", linewidths=0, vmin=-1, vmax=1)
    ax.set_title("Cosine alignment: OT-flow velocity vs lifted VelOT velocity", fontsize=13, weight="bold")
    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")
    plt.colorbar(sca, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
    sns.despine(ax=ax)
    savefig(path, cfg)
    return path


def plot_quantum_density_matrix(rho: np.ndarray, cfg: VelOTQuantumMetaFlowConfig, prefix: str) -> str:
    path = os.path.join(cfg.output_dir, f"{prefix}_quantum_density_matrix_abs.png")
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    sns.heatmap(np.abs(rho), cmap="mako", annot=True, fmt=".2f", square=True,
                cbar_kws={"label": "|rho_ab|"}, ax=ax)
    ax.set_title("Quantum-inspired density matrix magnitude", fontsize=13, weight="bold")
    ax.set_xlabel("meta-state b")
    ax.set_ylabel("meta-state a")
    savefig(path, cfg)
    return path


def plot_quantum_transition_amplitude_phase(A: np.ndarray, cfg: VelOTQuantumMetaFlowConfig, prefix: str) -> str:
    path = os.path.join(cfg.output_dir, f"{prefix}_transition_amplitude_phase.png")
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    sns.heatmap(np.angle(A), cmap="twilight", center=0, square=True,
                cbar_kws={"label": "phase angle"}, ax=ax)
    ax.set_title("Quantum-inspired transition amplitude phase", fontsize=13, weight="bold")
    ax.set_xlabel("future meta-state b")
    ax.set_ylabel("current meta-state a")
    savefig(path, cfg)
    return path


def plot_quantum_coherence_map(E: np.ndarray, coherence: np.ndarray, cfg: VelOTQuantumMetaFlowConfig, prefix: str) -> str:
    path = os.path.join(cfg.output_dir, f"{prefix}_quantum_cellular_coherence_map.png")
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    sca = ax.scatter(E[:, 0], E[:, 1], c=coherence, s=7, cmap="plasma", linewidths=0)
    ax.set_title("Quantum-inspired cellular coherence / superposition score", fontsize=13, weight="bold")
    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")
    plt.colorbar(sca, ax=ax, fraction=0.046, pad=0.04, label="coherence score")
    sns.despine(ax=ax)
    savefig(path, cfg)
    return path


def plot_quantum_phase_map(E: np.ndarray, phase: np.ndarray, cfg: VelOTQuantumMetaFlowConfig, prefix: str) -> str:
    path = os.path.join(cfg.output_dir, f"{prefix}_dominant_phase_map.png")
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    sca = ax.scatter(E[:, 0], E[:, 1], c=phase, s=7, cmap="twilight", linewidths=0, vmin=-np.pi, vmax=np.pi)
    ax.set_title("Dominant complex phase of quantum-inspired meta-state amplitude", fontsize=13, weight="bold")
    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")
    plt.colorbar(sca, ax=ax, fraction=0.046, pad=0.04, label="phase")
    sns.despine(ax=ax)
    savefig(path, cfg)
    return path


# =============================================================================
# Estimator runners
# =============================================================================

def run_vamp_estimator(
    adata,
    E: np.ndarray,
    Z: np.ndarray,
    Z_future: np.ndarray,
    T_diagnostics: Dict[str, Optional[np.ndarray]],
    cfg: VelOTQuantumMetaFlowConfig,
    device: str,
) -> Dict[str, object]:
    if cfg.n_metastates == "auto":
        K = auto_select_k_vamp(Z, Z_future, cfg, device)
    else:
        K = int(cfg.n_metastates)

    model, score = train_vamp_model(Z, Z_future, K, cfg, device, verbose=True)
    q0, q1, hard = infer_vamp_states(model, Z, Z_future, device)
    T = compute_soft_transition_matrix(q0, q1)

    state_df = classify_metastates(
        T=T,
        q=q0,
        divergence=T_diagnostics.get("divergence"),
        stemness=T_diagnostics.get("stemness"),
        cycle_score=T_diagnostics.get("cycle_score"),
        curl_score=T_diagnostics.get("curl_score"),
        velocity_alignment=T_diagnostics.get("velocity_alignment"),
        cfg=cfg,
    )

    terminal_states = state_df.index[state_df["label"].eq("terminal/sink")].to_list()
    if len(terminal_states) == 0:
        terminal_states = [int(state_df["sink_score"].idxmax())]
    B_meta = compute_terminal_committors(T, terminal_states)
    cell_committors = q0 @ B_meta if B_meta.shape[1] > 0 else np.zeros((adata.n_obs, 0), dtype=np.float32)

    prefix = "vamp"
    paths: Dict[str, Optional[str]] = {}
    paths["metastates_embedding"] = plot_metastates_embedding(E, hard, state_df, cfg, prefix, "VAMPFlow neural meta-states")
    paths["transition_heatmap"] = plot_transition_heatmap(T, state_df, cfg, prefix, "VAMPFlow coarse transition matrix")
    paths["flux_graph"] = plot_flux_graph(T, state_df, cfg, prefix, "VAMPFlow flux graph")
    paths["diagnostics"] = plot_state_diagnostics(state_df, cfg, prefix, "VAMPFlow dynamical diagnostics")
    paths["camembert_celltypes"] = plot_camembert_celltype_by_metastate(
        hard, adata, cfg, prefix,
        "Camembert plot: cell-type composition per VAMPFlow meta-state"
    )
    paths["committor_maps"] = plot_committor_maps(E, q0, B_meta, terminal_states, cfg, prefix, "VAMPFlow terminal committor maps")
    paths["soft_memberships"] = plot_soft_memberships(E, q0, cfg, prefix, "VAMPFlow soft meta-state memberships")

    summary_csv = os.path.join(cfg.output_dir, f"{prefix}_state_summary.csv")
    state_df.to_csv(summary_csv, index=False)
    paths["state_summary_csv"] = summary_csv

    return {
        "model": model,
        "score": score,
        "q": q0,
        "q_future": q1,
        "hard_state": hard,
        "T_meta": T,
        "state_summary": state_df,
        "terminal_states": terminal_states,
        "B_meta": B_meta,
        "cell_committors": cell_committors,
        "figure_paths": paths,
    }


def run_quantum_msm_estimator(
    adata,
    E: np.ndarray,
    Z: np.ndarray,
    Z_future: np.ndarray,
    X_quantum: np.ndarray,
    X_quantum_future: np.ndarray,
    T_diagnostics: Dict[str, Optional[np.ndarray]],
    cfg: VelOTQuantumMetaFlowConfig,
    device: str,
) -> Dict[str, object]:
    if cfg.n_metastates == "auto":
        K = auto_select_k_quantum(X_quantum, X_quantum_future, cfg, device)
    else:
        K = int(cfg.n_metastates)

    model, score = train_quantum_msm_model(X_quantum, X_quantum_future, K, cfg, device, verbose=True)
    q0, q1, phase0, phase1, psi0, psi1, hard = infer_quantum_states(model, X_quantum, X_quantum_future, device)

    T, T_amp, A = compute_quantum_transition_matrix(
        psi0, psi1, q0, q1,
        classical_blend=cfg.quantum_classical_transition_blend,
    )
    rho = compute_density_matrix(psi0)
    coherence = quantum_coherence_per_cell(psi0)
    dominant_phase = quantum_phase_resultant(psi0, q0)

    state_df = classify_metastates(
        T=T,
        q=q0,
        divergence=T_diagnostics.get("divergence"),
        stemness=T_diagnostics.get("stemness"),
        cycle_score=T_diagnostics.get("cycle_score"),
        curl_score=T_diagnostics.get("curl_score"),
        velocity_alignment=T_diagnostics.get("velocity_alignment"),
        cfg=cfg,
    )

    terminal_states = state_df.index[state_df["label"].eq("terminal/sink")].to_list()
    if len(terminal_states) == 0:
        terminal_states = [int(state_df["sink_score"].idxmax())]
    B_meta = compute_terminal_committors(T, terminal_states)
    cell_committors = q0 @ B_meta if B_meta.shape[1] > 0 else np.zeros((adata.n_obs, 0), dtype=np.float32)

    prefix = "quantum_msm"
    paths: Dict[str, Optional[str]] = {}
    paths["metastates_embedding"] = plot_metastates_embedding(E, hard, state_df, cfg, prefix, "QuantumMSM meta-states")
    paths["transition_heatmap"] = plot_transition_heatmap(T, state_df, cfg, prefix, "QuantumMSM transition matrix")
    paths["amplitude_transition_heatmap"] = plot_transition_heatmap(
        T_amp, state_df, cfg, f"{prefix}_amplitude_only",
        "QuantumMSM amplitude-only transition matrix"
    )
    paths["flux_graph"] = plot_flux_graph(T, state_df, cfg, prefix, "QuantumMSM flux graph")
    paths["diagnostics"] = plot_state_diagnostics(state_df, cfg, prefix, "QuantumMSM dynamical diagnostics")
    paths["camembert_celltypes"] = plot_camembert_celltype_by_metastate(
        hard, adata, cfg, prefix,
        "Camembert plot: cell-type composition per QuantumMSM meta-state"
    )
    paths["committor_maps"] = plot_committor_maps(E, q0, B_meta, terminal_states, cfg, prefix, "QuantumMSM terminal committor maps")
    paths["soft_memberships"] = plot_soft_memberships(E, q0, cfg, prefix, "QuantumMSM soft probability maps |psi|²")
    paths["density_matrix_abs"] = plot_quantum_density_matrix(rho, cfg, prefix)
    paths["transition_amplitude_phase"] = plot_quantum_transition_amplitude_phase(A, cfg, prefix)
    paths["coherence_map"] = plot_quantum_coherence_map(E, coherence, cfg, prefix)
    paths["dominant_phase_map"] = plot_quantum_phase_map(E, dominant_phase, cfg, prefix)

    summary_csv = os.path.join(cfg.output_dir, f"{prefix}_state_summary.csv")
    state_df.to_csv(summary_csv, index=False)
    paths["state_summary_csv"] = summary_csv

    return {
        "model": model,
        "score": score,
        "q": q0,
        "q_future": q1,
        "phase": phase0,
        "phase_future": phase1,
        "psi": psi0,
        "psi_future": psi1,
        "hard_state": hard,
        "T_meta": T,
        "T_amplitude": T_amp,
        "transition_amplitude": A,
        "density_matrix": rho,
        "cellular_coherence": coherence,
        "dominant_phase": dominant_phase,
        "state_summary": state_df,
        "terminal_states": terminal_states,
        "B_meta": B_meta,
        "cell_committors": cell_committors,
        "figure_paths": paths,
    }


# =============================================================================
# AnnData storage
# =============================================================================

def store_estimator_outputs_in_adata(
    adata,
    result: Dict[str, object],
    prefix: str,
) -> None:
    q = result["q"]
    qf = result["q_future"]
    hard = result["hard_state"]
    state_df = result["state_summary"]
    T = result["T_meta"]
    terminals = result["terminal_states"]
    comm = result["cell_committors"]

    adata.obs[f"{prefix}_state"] = pd.Categorical([f"M{x}" for x in hard])
    adata.obs[f"{prefix}_state_label"] = pd.Categorical([state_df.loc[x, "label"] for x in hard])

    for c, t in enumerate(terminals):
        adata.obs[f"{prefix}_committor_M{t}"] = comm[:, c]

    adata.obsm[f"{prefix}_soft_states"] = q
    adata.obsm[f"{prefix}_soft_states_future"] = qf
    adata.uns[f"{prefix}_transition_matrix"] = T
    adata.uns[f"{prefix}_state_summary"] = state_df
    adata.uns[f"{prefix}_terminal_states"] = terminals

    if "psi" in result:
        # AnnData HDF5 can be inconsistent with complex arrays across versions;
        # store real/imag separately for robust writing.
        psi = result["psi"]
        psi_f = result["psi_future"]
        adata.obsm[f"{prefix}_psi_real"] = psi.real.astype(np.float32)
        adata.obsm[f"{prefix}_psi_imag"] = psi.imag.astype(np.float32)
        adata.obsm[f"{prefix}_phase"] = result["phase"]
        adata.obs[f"{prefix}_cellular_coherence"] = result["cellular_coherence"]
        adata.obs[f"{prefix}_dominant_phase"] = result["dominant_phase"]
        adata.uns[f"{prefix}_transition_amplitude_real"] = result["transition_amplitude"].real.astype(np.float32)
        adata.uns[f"{prefix}_transition_amplitude_imag"] = result["transition_amplitude"].imag.astype(np.float32)
        adata.uns[f"{prefix}_density_matrix_real"] = result["density_matrix"].real.astype(np.float32)
        adata.uns[f"{prefix}_density_matrix_imag"] = result["density_matrix"].imag.astype(np.float32)
        adata.uns[f"{prefix}_amplitude_transition_matrix"] = result["T_amplitude"]


# =============================================================================
# Main runner
# =============================================================================

def run_velot_quantum_metaflow(
    adata,
    cfg: Optional[VelOTQuantumMetaFlowConfig] = None,
) -> Dict[str, object]:
    if cfg is None:
        cfg = VelOTQuantumMetaFlowConfig()

    set_seed(cfg.random_state)
    device = get_device(cfg.device)
    ensure_dir(cfg.output_dir)
    print(f"[VelOT Quantum-MetaFlow] device: {device}")
    print(f"[estimator_mode] {cfg.estimator_mode}")

    # 1) Latent, embedding, pseudotime
    Z, latent_used, scaler = get_or_compute_latent(adata, cfg)
    E, embedding_used = get_or_compute_embedding(adata, Z, cfg)
    pt, pt_source = get_or_compute_pseudotime(adata, Z, cfg)
    bins = make_pseudotime_bins(pt, cfg.n_time_bins, cfg.min_cells_per_bin)

    print(f"[latent] {latent_used}: {Z.shape}")
    print(f"[embedding] {embedding_used}: {E.shape}")
    print(f"[pseudotime] {pt_source}; bins={pd.Series(bins).value_counts().sort_index().to_dict()}")

    # 2) VelOT velocity
    V_velot_latent, V_velot_emb, velot_source = obtain_velot_velocity_latent(adata, E, Z, cfg)
    print(f"[VelOT velocity] source: {velot_source}")
    if V_velot_latent is not None:
        print(f"[VelOT velocity] latent median norm={robust_median_norm(V_velot_latent):.4f}")
    if V_velot_emb is not None:
        print(f"[VelOT velocity] embedding median norm={robust_median_norm(V_velot_emb):.4f}")

    # 3) OT-flow matching
    V_ot = None
    flow_model = None
    banks = None

    if cfg.use_ot_flow_matching:
        banks = build_ot_pair_banks(Z, bins, cfg, device=device)
        print(f"[OT] built {len(banks)} adjacent-bin coupling banks")
        flow_model = train_flow_field(Z, pt, banks, V_velot_latent, cfg, device=device)
        V_ot = predict_flow(flow_model, Z, pt, device=device)
        print(f"[OT-flow] median norm={robust_median_norm(V_ot):.4f}")

    # 4) Final vector field and future state
    V_final, velocity_description, V_ot_rescaled, V_velot_rescaled = combine_velocities(V_ot, V_velot_latent, cfg)
    Z_future = (Z + cfg.future_dt * V_final).astype(np.float32)

    print(f"[final velocity] {velocity_description}; median norm={robust_median_norm(V_final):.4f}")

    velocity_alignment = None
    if V_ot_rescaled is not None and V_velot_rescaled is not None:
        velocity_alignment = cosine_similarity_rows(V_ot_rescaled, V_velot_rescaled)
        print(f"[velocity alignment] median={np.median(velocity_alignment):.3f}, mean={np.mean(velocity_alignment):.3f}")

    # 5) Diagnostics and quantum features
    print("[diagnostics] estimating divergence/curl")
    divergence = estimate_divergence_knn(Z, V_final, k=30)
    curl_score = estimate_embedding_curl(E, V_velot_emb, k=30)
    stemness = safe_numeric_obs(adata, cfg.stemness_key)
    cycle_score = safe_numeric_obs(adata, cfg.cycle_key)

    diagnostics = {
        "divergence": divergence,
        "curl_score": curl_score,
        "stemness": stemness,
        "cycle_score": cycle_score,
        "velocity_alignment": velocity_alignment,
    }

    X_quantum = make_quantum_features(Z, V_final, divergence, curl_score, velocity_alignment, cfg)
    # Use the same additional features for future state, updating speed/div/curl features would require
    # a second field estimate. For stability we propagate latent coordinates and keep diagnostics fixed.
    X_quantum_future = make_quantum_features(Z_future, V_final, divergence, curl_score, velocity_alignment, cfg)

    # 6) Shared plots
    shared_paths: Dict[str, Optional[str]] = {}
    shared_paths["raw_velot_velocity_umap"] = plot_raw_velot_velocity_embedding(E, V_velot_emb, cfg)
    shared_paths["final_velocity_embedding"] = plot_final_velocity_embedding(E, Z, Z_future, cfg)
    shared_paths["velocity_alignment"] = plot_velocity_alignment(E, velocity_alignment, cfg)

    # 7) Estimators
    mode = cfg.estimator_mode.lower()
    if mode not in {"vamp", "quantum_msm", "both"}:
        raise ValueError("estimator_mode must be 'vamp', 'quantum_msm', or 'both'.")

    results: Dict[str, object] = {
        "adata": adata,
        "Z": Z,
        "embedding": E,
        "pseudotime": pt,
        "bins": bins,
        "V_velot_embedding": V_velot_emb,
        "V_velot_latent": V_velot_latent,
        "V_ot": V_ot,
        "V_final": V_final,
        "Z_future": Z_future,
        "velocity_alignment": velocity_alignment,
        "divergence": divergence,
        "curl_score": curl_score,
        "flow_model": flow_model,
        "figure_paths": {"shared": shared_paths},
        "config": cfg,
    }

    if mode in {"vamp", "both"}:
        print("[estimator] running VAMPFlow")
        vamp_result = run_vamp_estimator(adata, E, Z, Z_future, diagnostics, cfg, device)
        results["vamp"] = vamp_result
        results["figure_paths"]["vamp"] = vamp_result["figure_paths"]
        store_estimator_outputs_in_adata(adata, vamp_result, "velot_vampflow")

    if mode in {"quantum_msm", "both"}:
        print("[estimator] running QuantumMSM")
        q_result = run_quantum_msm_estimator(adata, E, Z, Z_future, X_quantum, X_quantum_future, diagnostics, cfg, device)
        results["quantum_msm"] = q_result
        results["figure_paths"]["quantum_msm"] = q_result["figure_paths"]
        store_estimator_outputs_in_adata(adata, q_result, "velot_quantum_msm")

    # 8) Global AnnData storage
    adata.obs["velot_quantum_metaflow_pseudotime"] = pt
    adata.obs["velot_quantum_metaflow_time_bin"] = pd.Categorical(bins.astype(str))
    adata.obs["velot_quantum_metaflow_divergence"] = divergence
    if curl_score is not None:
        adata.obs["velot_quantum_metaflow_embedding_curl"] = curl_score
    if velocity_alignment is not None:
        adata.obs["velot_quantum_metaflow_ot_velot_alignment"] = velocity_alignment

    adata.obsm["X_velot_quantum_metaflow_latent_scaled"] = Z
    adata.obsm["velot_quantum_metaflow_future_latent"] = Z_future
    adata.obsm["velot_quantum_metaflow_velocity_latent_final"] = V_final
    if V_velot_latent is not None:
        adata.obsm["velot_quantum_metaflow_velocity_latent_from_velot"] = V_velot_latent
    if V_ot is not None:
        adata.obsm["velot_quantum_metaflow_velocity_latent_ot_flow"] = V_ot
    if V_velot_rescaled is not None:
        adata.obsm["velot_quantum_metaflow_velocity_latent_from_velot_rescaled"] = V_velot_rescaled

    adata.uns["velot_quantum_metaflow_velocity_description"] = velocity_description
    adata.uns["velot_quantum_metaflow_config"] = dict(cfg.__dict__)

    print("[done] VelOT Quantum-MetaFlow completed")

    if "quantum_msm" in results:
        print("\n[QuantumMSM state summary]")
        print(results["quantum_msm"]["state_summary"].round(3).to_string(index=False))
    if "vamp" in results:
        print("\n[VAMPFlow state summary]")
        print(results["vamp"]["state_summary"].round(3).to_string(index=False))

    return results


# =============================================================================
# Script entry point
# =============================================================================

if __name__ == "__main__":
    # Example only:
    #
    # import scanpy as sc
    # adata = sc.read_h5ad("your_velot_dataset.h5ad")
    #
    # cfg = VelOTQuantumMetaFlowConfig(
    #     latent_key="X_pca",
    #     embedding_key="X_umap",
    #     velot_velocity_key="velot_velocity_umap",
    #     pseudotime_key="velot_pseudotime",
    #     cell_type_key="cell_type",
    #
    #     estimator_mode="both",             # "vamp", "quantum_msm", or "both"
    #
    #     use_ot_flow_matching=True,
    #     final_velocity_mode="blend",
    #     velot_velocity_weight=0.50,
    #     flow_velot_alignment_weight=0.15,
    #
    #     n_metastates="auto",
    #     output_dir="velot_quantum_metaflow_outputs",
    # )
    #
    # out = run_velot_quantum_metaflow(adata, cfg)
    # out["adata"].write_h5ad("adata_with_velot_quantum_metaflow.h5ad")
    pass

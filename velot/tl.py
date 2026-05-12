"""
velot.tl — Tools
================

Core analysis functions for the VelOT pipeline.

Follows the scanpy/scvelo convention: functions modify adata in place
and return it for optional chaining.

Quick usage::

    import velot
    velot.tl.velocity(adata)   # runs the full pipeline

Step-by-step usage::

    velot.tl.build_windows(adata)
    velot.tl.compute_ot_velocity(adata)
    velot.tl.smooth_velocity(adata)
    velot.tl.project_to_umap(adata)
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from anndata import AnnData
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

try:
    import ot as pot
except ImportError:
    raise ImportError(
        "The POT (Python Optimal Transport) library is required. "
        "Install it with: pip install POT"
    )

import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# 1. SPATIAL-TEMPORAL WINDOWING
# =====================================================================


def build_windows(
    adata: AnnData,
    basis: str = "X_pca",
    n_clusters: Optional[int] = None,
    window_size: Optional[int] = None,
    overlap_fraction: float = 0.5,
    min_window_size: int = 20,
    spatial_key: Optional[str] = None,
    tail_handling: str = "force",  # NEW: "force", "drop", or "split"
    tail_threshold: int = 10,      # NEW: Used only if tail_handling="drop"
    random_state: int = 42,
) -> AnnData:
    """
    Build spatial-temporal windows for local OT velocity computation.

    Instead of sorting all cells by pseudotime and creating global
    sequential windows (which fails when pseudotime does not correspond
    to spatial locality), this function:

      1. Clusters cells spatially in PCA space
      2. Within each cluster, sorts cells by pseudotime
      3. Creates overlapping temporal windows within each cluster
      4. Pairs consecutive windows for OT computation

    This ensures OT is only computed between spatially nearby cells,
    even when cells of very different pseudotime coexist in the same
    region of the embedding.

    Parameters
    ----------
    adata
        Must contain ``adata.obsm[basis]`` and ``adata.obs['pseudotime']``.
    basis
        Key in ``adata.obsm`` for the embedding to cluster in.
    n_clusters
        Number of spatial clusters. If None, chosen automatically
        based on dataset size.
    window_size
        Number of cells per temporal window. If None, chosen
        automatically.
    overlap_fraction
        Fraction of overlap between consecutive temporal windows
        within each cluster. 0.5 means 50% overlap.
    min_window_size
        Minimum number of cells to form a valid window.
        Clusters smaller than 2 * min_window_size are skipped.
    spatial_key
        Column in ``adata.obs`` with precomputed spatial cluster
        labels. If provided, skips KMeans clustering.
    random_state
        Random seed for KMeans.

    Returns
    -------
    adata, modified in place with:
      - ``adata.obs['velot_spatial_cluster']`` : spatial cluster labels
      - ``adata.uns['velot_windows']`` : dict with window pair info
    """
    _check_fields(adata, obsm_keys=[basis], obs_keys=["pseudotime"])

    X = adata.obsm[basis]
    pseudotime = adata.obs["pseudotime"].values
    n_cells = adata.n_obs

    # ------------------------------------------------------------------
    # Step 1: Spatial clustering
    # ------------------------------------------------------------------
    if spatial_key is not None:
        labels = adata.obs[spatial_key].values.astype(int)
        n_clust = len(np.unique(labels))
        print(f"  Using precomputed spatial labels from '{spatial_key}': "
              f"{n_clust} clusters")
    else:
        if n_clusters is None:
            n_clusters = max(5, min(30, n_cells // 100))

        km = KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init=10,
        ).fit(X)
        labels = km.labels_
        n_clust = n_clusters
        print(f"  Spatial clustering: {n_clust} clusters via KMeans "
              f"on {basis} ({X.shape[1]}D)")

    adata.obs["velot_spatial_cluster"] = labels

    # ------------------------------------------------------------------
    # Step 2: Within each cluster, create temporal windows
    # ------------------------------------------------------------------
    if window_size is None:
        # Aim for ~3-5 windows per cluster on average
        mean_cluster_size = n_cells / n_clust
        window_size = max(min_window_size, int(mean_cluster_size / 4))

    step_size = max(1, int(window_size * (1.0 - overlap_fraction)))

    window_pairs = []
    skipped_clusters = 0

    for c in range(n_clust):
        cluster_mask = labels == c
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) < 2 * min_window_size:
            skipped_clusters += 1
            continue

        # Sort by pseudotime within this cluster
        pt_local = pseudotime[cluster_indices]
        order = np.argsort(pt_local)
        sorted_indices = cluster_indices[order]

        # Adapt window size for small clusters
        local_ws = min(window_size, len(sorted_indices) // 2)
        local_ws = max(local_ws, min_window_size)
        local_step = max(1, int(local_ws * (1.0 - overlap_fraction)))

        # Create windows
        windows = []
        last_start = 0
        for start in range(0, len(sorted_indices) - local_ws + 1, local_step):
            windows.append(sorted_indices[start : start + local_ws])
            last_start = start

        if len(windows) > 0:
            next_start = last_start + local_step
            remaining_count = len(sorted_indices) - next_start

            if remaining_count > 0:
                if tail_handling == "split" and remaining_count >= 2:
                    # Distribute the remaining cells across two full-sized 
                    # windows to smooth the overlap and add an extra OT step
                    step_B = remaining_count // 2
                    
                    end_A = len(sorted_indices) - step_B
                    start_A = max(0, end_A - local_ws)
                    
                    end_B = len(sorted_indices)
                    start_B = max(0, end_B - local_ws)
                    
                    w_A = sorted_indices[start_A : end_A]
                    w_B = sorted_indices[start_B : end_B]
                    
                    # Avoid duplicates if they perfectly overlap
                    if not np.array_equal(windows[-1], w_A):
                        windows.append(w_A)
                    if not np.array_equal(windows[-1], w_B):
                        windows.append(w_B)

                elif tail_handling == "drop":
                    # Only append the final forced window if the remaining
                    # cells meet the user's noise threshold
                    if remaining_count >= tail_threshold:
                        windows.append(sorted_indices[-local_ws:])

                else: 
                    # "force" (Default behavior): Force the last window 
                    # to capture the tail, regardless of overlap spike
                    windows.append(sorted_indices[-local_ws:])

        # Pair consecutive windows
        for w_src, w_tgt in zip(windows[:-1], windows[1:]):
            window_pairs.append((w_src, w_tgt))

    # ------------------------------------------------------------------
    # Store results
    # ------------------------------------------------------------------
    adata.uns["velot_windows"] = {
        "pairs": window_pairs,
        "n_pairs": len(window_pairs),
        "n_clusters": n_clust,
        "window_size": window_size,
        "overlap_fraction": overlap_fraction,
    }

    print(f"  Built {len(window_pairs)} window pairs across "
          f"{n_clust - skipped_clusters} clusters "
          f"(skipped {skipped_clusters} small clusters)")
    print(f"  Window size: {window_size}, step: {step_size}")

    return adata


# =====================================================================
# 2. OPTIMAL TRANSPORT VELOCITY
# =====================================================================


def _ot_velocity_pair(
    X: np.ndarray,
    idx_source: np.ndarray,
    idx_target: np.ndarray,
    pseudotime: np.ndarray,
    knn_adj,
    reg: float = 0.05,
    lambda_time: float = 1.0,
    lambda_knn: float = 1.0,
) -> np.ndarray:
    """
    Compute OT-based velocity for one window pair.

    Returns velocity vectors for cells in the source window.
    Adapted from the original VelOT compute_ot_velocity.
    """
    X1 = X[idx_source]
    X2 = X[idx_target]

    n1 = X1.shape[0]
    n2 = X2.shape[0]

    # Uniform marginals
    a = np.ones(n1, dtype=np.float64) / n1
    b = np.ones(n2, dtype=np.float64) / n2

    # Cost matrix: squared Euclidean distance, normalized
    C = pot.dist(X1, X2, metric="sqeuclidean")
    C_max = C.max()
    if C_max > 0:
        C = C / C_max

    # Pseudotime penalty: penalize backward transport
    t1 = pseudotime[idx_source]
    t2 = pseudotime[idx_target]
    time_diff = t1[:, None] - t2[None, :]
    C[time_diff > 0] += lambda_time

    # KNN locality penalty: penalize transport to non-neighbors
    if knn_adj is not None:
        local_adj = knn_adj[idx_source][:, idx_target]
        if hasattr(local_adj, "toarray"):
            local_adj = local_adj.toarray()
        C[local_adj == 0] += lambda_knn

    # Sinkhorn OT
    try:
        P = pot.sinkhorn(a, b, C, reg=reg, numItermax=500, stopThr=1e-6)
    except Exception:
        # Fallback to exact OT if Sinkhorn diverges
        try:
            P = pot.emd(a, b, C)
        except Exception:
            return np.zeros_like(X1)

    # Check for numerical issues
    if not np.isfinite(P).all() or P.sum() < 1e-10:
        return np.zeros_like(X1)

    # Velocity = weighted displacement
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    V = (P @ X2) / row_sums - X1

    # Zero out velocity for cells with no KNN neighbors in target
    if knn_adj is not None:
        has_neighbors = local_adj.sum(axis=1) > 0
        V[~has_neighbors] = 0.0

    return V


def compute_ot_velocity(
    adata: AnnData,
    basis: str = "X_pca",
    reg: float = 0.05,
    lambda_time: float = 1.0,
    lambda_knn: float = 1.0,
) -> AnnData:
    """
    Compute raw OT velocity from spatial-temporal windows.

    For each window pair, Sinkhorn optimal transport is used to
    match cells in the source window to cells in the target window.
    The velocity for each cell is the weighted displacement under
    the transport plan.

    Results are aggregated across all window pairs where a cell
    appears as a source. A per-cell confidence score tracks how
    many windows contributed to each cell's velocity estimate.

    Parameters
    ----------
    adata
        Must contain windows from ``velot.tl.build_windows()``.
    basis
        Embedding key in ``adata.obsm``.
    reg
        Sinkhorn entropy regularization. Smaller = sharper plans.
    lambda_time
        Penalty added to cost for backward-in-time transport.
    lambda_knn
        Penalty added to cost for transport between non-neighbors.

    Returns
    -------
    adata, modified in place with:
      - ``adata.obsm['velot_velocity']`` : raw OT velocity (PCA space)
      - ``adata.obs['velot_confidence']`` : contribution count per cell
    """
    _check_fields(adata, obsm_keys=[basis], uns_keys=["velot_windows"])

    X = adata.obsm[basis]
    pseudotime = adata.obs["pseudotime"].values
    n_cells = adata.n_obs
    dim = X.shape[1]

    window_info = adata.uns["velot_windows"]
    window_pairs = window_info["pairs"]

    # Get KNN adjacency for locality penalty
    knn_adj = None
    if "connectivities" in adata.obsp:
        knn_adj = adata.obsp["connectivities"]

    # Accumulate velocity across all window pairs
    V = np.zeros((n_cells, dim), dtype=np.float64)
    counts = np.zeros(n_cells, dtype=np.float64)

    for pair_i, (idx_src, idx_tgt) in enumerate(window_pairs):
        v_local = _ot_velocity_pair(
            X, idx_src, idx_tgt, pseudotime, knn_adj,
            reg=reg, lambda_time=lambda_time, lambda_knn=lambda_knn,
        )

        for i, cell_idx in enumerate(idx_src):
            V[cell_idx] += v_local[i]
            counts[cell_idx] += 1

    # Average velocity across contributing windows
    mask = counts > 0
    V[mask] /= counts[mask, None]

    # Confidence: normalized contribution count
    max_count = counts.max() if counts.max() > 0 else 1.0
    confidence = counts / max_count

    if basis == "X_pca":
        adata.obsm["velot_velocity_raw_pca"] = V
    elif basis == "X_umap":
        adata.obsm["velot_velocity_raw_umap"] = V
    else:
        raise NotImplemented

    adata.obs["velot_confidence"] = confidence

    n_with_velocity = mask.sum()
    n_zero = (~mask).sum()
    print(f"  OT velocity computed: {n_with_velocity} cells with velocity, "
          f"{n_zero} cells without (will be filled by smoothing)")

    return adata


# =====================================================================
# 3. NEURAL VELOCITY FIELD SMOOTHING
# =====================================================================


def _build_knn_index(X: np.ndarray, k: int = 15) -> np.ndarray:
    """
    Build a fixed-k nearest neighbor index using cKDTree.

    Returns
    -------
    knn_indices : ndarray of shape (n_cells, k)
        For each cell, the indices of its k nearest neighbors.
    """
    tree = cKDTree(X)
    _, indices = tree.query(X, k=k + 1)  # +1 because self is included
    return indices[:, 1:]  # exclude self


class _VelocityNet(nn.Module):
    """Small MLP that predicts velocity from PCA coordinates."""

    def __init__(self, dim: int, hidden: int = 128, use_pseudotime: bool = True):
        super().__init__()
        self.use_pseudotime = use_pseudotime
        in_dim = dim + (1 if use_pseudotime else 0)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, pt: torch.Tensor = None) -> torch.Tensor:
        if self.use_pseudotime and pt is not None:
            x = torch.cat([x, pt], dim=-1)
        return self.net(x)

    def forward_with_jacobian(self, x: torch.Tensor, pt: torch.Tensor = None):
        """
        Forward pass that also computes the Jacobian dv/dx.
        Used for curl and divergence regularization.
        """
        x_in = x.detach().clone().requires_grad_(True)
        if self.use_pseudotime and pt is not None:
            inp = torch.cat([x_in, pt], dim=-1)
        else:
            inp = x_in
        v = self.net(inp)
        return v, x_in


def smooth_velocity(
    adata: AnnData,
    basis: str = "pca",
    velocity_key: str = "velot_velocity_raw",
    n_epochs: int = 200,
    hidden_dim: int = 128,
    lr: float = 1e-3,
    batch_size: int = 256,
    lambda_smooth: float = 0.5,
    lambda_curl: float = 0.0,
    lambda_divergence: float = 0.0,
    k_smooth: int = 15,
    use_pseudotime: bool = True,
    random_state: int = 42,
    verbose: bool = True,
) -> AnnData:
    """
    Smooth the raw OT velocity field using a neural network.

    Three types of regularization are available:

    - **Smoothness** (``lambda_smooth``): neighboring cells should
      have similar velocities. Reduces noise and zig-zagging.

    - **Curl penalty** (``lambda_curl``): penalizes rotational
      components of the velocity field. Encourages irrotational
      flow where streamlines do not form loops. This is the
      strongest constraint for preventing crossing field lines.

    - **Divergence penalty** (``lambda_divergence``): penalizes
      the divergence of the velocity field. Encourages
      incompressible-like flow where cells neither accumulate
      nor deplete locally.

    Parameters
    ----------
    adata
        Must contain ``adata.obsm['velot_velocity']``.
    basis
        Embedding key for cell coordinates.
    n_epochs
        Training epochs.
    hidden_dim
        Network hidden layer width.
    lr
        Learning rate.
    batch_size
        Cells per batch.
    lambda_smooth
        Weight of KNN smoothness loss.
    lambda_curl
        Weight of curl penalty. Higher values produce flow
        with fewer crossing streamlines. Recommended range:
        0.0 (off) to 0.5.
    lambda_divergence
        Weight of divergence penalty. Higher values produce
        more volume-preserving flow. Recommended range:
        0.0 (off) to 0.5.
    k_smooth
        Number of neighbors for smoothness.
    use_pseudotime
        Condition network on pseudotime.
    random_state
        Random seed.
    verbose
        Print progress.

    Returns
    -------
    adata with smoothed velocity.
    """    
    _check_fields(
        adata, obsm_keys=[f"X_{basis}", velocity_key],
        obs_keys=["velot_confidence"],
    )

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    X_np = adata.obsm[f"X_{basis}"].astype(np.float32)
    V_np = adata.obsm[velocity_key].astype(np.float32)
    conf_np = adata.obs["velot_confidence"].values.astype(np.float32)
    pt_np = adata.obs["pseudotime"].values.astype(np.float32)

    n_cells, dim = X_np.shape

    # adata.obsm[f"{velocity_key}_raw"] = V_np.copy()

    knn_indices = _build_knn_index(X_np, k=k_smooth)

    X_t = torch.tensor(X_np, device=DEVICE)
    V_t = torch.tensor(V_np, device=DEVICE)
    conf_t = torch.tensor(conf_np, device=DEVICE)
    pt_t = torch.tensor(pt_np, device=DEVICE).unsqueeze(-1)
    knn_t = torch.tensor(knn_indices, dtype=torch.long, device=DEVICE)

    net = _VelocityNet(
        dim=dim, hidden=hidden_dim, use_pseudotime=use_pseudotime,
    ).to(DEVICE)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    use_jacobian = (lambda_curl > 0) or (lambda_divergence > 0)

    net.train()
    losses_reg = []
    losses_smooth = []
    losses_curl = []
    losses_div = []
    losses_total = []

    # Track initial loss scales for normalization
    init_scales = {}
    warmup_epochs = 20  # use raw weights during warmup

    for epoch in range(n_epochs):
        idx = torch.randint(0, n_cells, (min(batch_size, n_cells),), device=DEVICE)

        B = idx.shape[0]
        k = knn_indices.shape[1]

        nbr_idx = knn_t[idx]
        all_idx = torch.cat([idx, nbr_idx.reshape(-1)])

        X_all = X_t[all_idx]
        pt_all = pt_t[all_idx] if use_pseudotime else None

        V_all = net(X_all, pt_all)

        V_batch = V_all[:B]
        V_nbr = V_all[B:].reshape(B, k, dim)

        V_target = V_t[idx]
        conf_batch = conf_t[idx]

        sq_err = ((V_batch - V_target) ** 2).sum(dim=-1)
        loss_reg = (conf_batch * sq_err).sum() / (conf_batch.sum() + 1e-8)

        diff = V_batch.unsqueeze(1) - V_nbr
        loss_smooth = (diff ** 2).mean()

        # Curl and divergence
        loss_curl_val = torch.tensor(0.0, device=DEVICE)
        loss_div_val = torch.tensor(0.0, device=DEVICE)

        if use_jacobian:
            x_jac = X_t[idx].detach().clone().requires_grad_(True)
            pt_jac = pt_t[idx] if use_pseudotime else None

            if use_pseudotime and pt_jac is not None:
                inp_jac = torch.cat([x_jac, pt_jac], dim=-1)
            else:
                inp_jac = x_jac

            v_jac = net.net(inp_jac)

            jac_rows = []
            for d in range(min(dim, 3)):
                grad_d = torch.autograd.grad(
                    v_jac[:, d].sum(), x_jac,
                    create_graph=True, retain_graph=True,
                )[0]
                jac_rows.append(grad_d)

            if len(jac_rows) >= 2:
                if lambda_divergence > 0:
                    divergence = sum(
                        jac_rows[d][:, d] for d in range(min(dim, len(jac_rows)))
                    )
                    loss_div_val = (divergence ** 2).mean()

                if lambda_curl > 0:
                    curl_components = []
                    n_curl_dims = min(dim, len(jac_rows))
                    for d1 in range(n_curl_dims):
                        for d2 in range(d1 + 1, n_curl_dims):
                            curl_d1d2 = jac_rows[d2][:, d1] - jac_rows[d1][:, d2]
                            curl_components.append(curl_d1d2)
                    if curl_components:
                        curl_stack = torch.stack(curl_components, dim=-1)
                        loss_curl_val = (curl_stack ** 2).mean()

        # Record initial scales after warmup
        if epoch == warmup_epochs:
            init_scales["reg"] = max(loss_reg.item(), 1e-6)
            init_scales["smooth"] = max(loss_smooth.item(), 1e-6)
            if lambda_curl > 0:
                init_scales["curl"] = max(loss_curl_val.item(), 1e-6)
            if lambda_divergence > 0:
                init_scales["div"] = max(loss_div_val.item(), 1e-6)

        # Compute total loss with scale normalization
        if epoch < warmup_epochs or not init_scales:
            # During warmup: just regression + smoothness with raw weights
            loss = loss_reg + lambda_smooth * loss_smooth
            if lambda_curl > 0:
                loss = loss + lambda_curl * loss_curl_val
            if lambda_divergence > 0:
                loss = loss + lambda_divergence * loss_div_val
        else:
            # After warmup: normalize each loss by its initial scale
            # so that lambda values have comparable effect
            loss = loss_reg / init_scales["reg"]
            loss = loss + lambda_smooth * loss_smooth / init_scales["smooth"]
            if lambda_curl > 0:
                loss = loss + lambda_curl * loss_curl_val / init_scales["curl"]
            if lambda_divergence > 0:
                loss = loss + lambda_divergence * loss_div_val / init_scales["div"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses_reg.append(loss_reg.item())
        losses_smooth.append(loss_smooth.item())
        losses_curl.append(loss_curl_val.item())
        losses_div.append(loss_div_val.item())
        losses_total.append(loss.item())

        if verbose and (epoch + 1) % 50 == 0:
            msg = (f"    epoch {epoch+1}/{n_epochs}: "
                   f"total={loss.item():.4f} "
                   f"reg={loss_reg.item():.4f} "
                   f"smooth={loss_smooth.item():.4f}")
            if lambda_curl > 0:
                msg += f" curl={loss_curl_val.item():.4f}"
            if lambda_divergence > 0:
                msg += f" div={loss_div_val.item():.4f}"
            print(msg)

    # Extract smoothed velocity
    net.eval()
    with torch.no_grad():
        chunk_size = 2048
        V_smooth = []
        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            X_chunk = X_t[start:end]
            pt_chunk = pt_t[start:end] if use_pseudotime else None
            V_smooth.append(net(X_chunk, pt_chunk).cpu().numpy())
        V_smooth = np.concatenate(V_smooth, axis=0)

    adata.obsm[f"velot_velocity_{basis}"] = V_smooth

    adata.uns["velot_smoothing"] = {
        "n_epochs": n_epochs,
        "losses_regression": losses_reg,
        "losses_smoothness": losses_smooth,
        "losses_curl": losses_curl,
        "losses_divergence": losses_div,
        "losses_total": losses_total,
        "lambda_smooth": lambda_smooth,
        "lambda_curl": lambda_curl,
        "lambda_divergence": lambda_divergence,
        "network": net,
        "device": str(DEVICE),
        "dim": dim,
        "use_pseudotime": use_pseudotime,
    }

    if verbose:
        print(f"  Smoothing complete: final total={losses_total[-1]:.4f}")

    return adata


def query_velocity(
    adata: AnnData,
    positions: np.ndarray,
    pseudotime_values: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Query the learned continuous velocity field at arbitrary positions.

    Unlike the stored velocity vectors in ``adata.obsm['velot_velocity']``
    which are only defined at cell locations, this function evaluates
    the trained neural velocity field at any point in PCA space.

    This is what makes VelOT a true velocity FIELD rather than just
    a set of velocity vectors: given any point on the manifold, you
    can ask "what is the velocity here?"

    Parameters
    ----------
    positions
        Array of shape (n_points, D) with PCA coordinates to query.
    pseudotime_values
        Array of shape (n_points,) with pseudotime values for each
        query point. If None, uses the median pseudotime from the
        dataset.

    Returns
    -------
    Array of shape (n_points, D) with velocity vectors at each
    query position.

    Example
    -------
    ::

        # Query at cell positions (same as stored vectors)
        V = velot.tl.query_velocity(adata, adata.obsm["X_pca"])

        # Query at arbitrary positions
        import numpy as np
        grid_points = np.random.randn(100, 30) * 0.1
        V_grid = velot.tl.query_velocity(adata, grid_points)
    """
    if "velot_smoothing" not in adata.uns:
        raise ValueError(
            "No smoothing network found. Run velot.tl.smooth_velocity() first."
        )

    info = adata.uns["velot_smoothing"]
    net = info["network"]
    use_pt = info["use_pseudotime"]

    positions = np.asarray(positions, dtype=np.float32)
    n_points = positions.shape[0]

    if pseudotime_values is None:
        median_pt = float(np.median(adata.obs["pseudotime"].values))
        pseudotime_values = np.full(n_points, median_pt, dtype=np.float32)
    else:
        pseudotime_values = np.asarray(pseudotime_values, dtype=np.float32)

    X_t = torch.tensor(positions, device=DEVICE)
    pt_t = torch.tensor(pseudotime_values, device=DEVICE).unsqueeze(-1)

    net.eval()
    with torch.no_grad():
        chunk_size = 2048
        V_parts = []
        for start in range(0, n_points, chunk_size):
            end = min(start + chunk_size, n_points)
            X_chunk = X_t[start:end]
            pt_chunk = pt_t[start:end] if use_pt else None
            V_parts.append(net(X_chunk, pt_chunk).cpu().numpy())

    return np.concatenate(V_parts, axis=0)


# =====================================================================
# 4. PROJECTION TO UMAP
# =====================================================================


def project_to_umap(
    adata: AnnData,
    velocity_key: str = "velot_velocity_pca",
    velocity_key_umap: str = "velot_velocity_umap",
    basis_pca: str = "X_pca",
    basis_umap: str = "X_umap",
    n_neighbors: int = 30,
) -> AnnData:
    """
    Project velocity from PCA space to UMAP space for visualization.

    Uses a local linear approximation: for each cell, the Jacobian
    of the PCA→UMAP mapping is estimated from its KNN neighborhood
    via least-squares, and the PCA velocity is transformed accordingly.

    Note: UMAP is for visualization only. The velocity in PCA space
    (``adata.obsm['velot_velocity']``) is the primary output.

    Parameters
    ----------
    adata
        Must contain PCA and UMAP embeddings, and computed velocity.

    Returns
    -------
    adata with ``adata.obsm['velocity_umap']`` populated.
    """
    _check_fields(
        adata,
        obsm_keys=[velocity_key, basis_pca, basis_umap],
    )

    X_pca = adata.obsm[basis_pca]
    X_umap = adata.obsm[basis_umap]
    V_pca = adata.obsm[velocity_key]

    n_cells = X_pca.shape[0]

    # Build KNN in PCA space for the local linear approximation
    knn_indices = _build_knn_index(X_pca, k=n_neighbors)

    V_umap = np.zeros_like(X_umap)

    for i in range(n_cells):
        nbrs = knn_indices[i]

        # Local displacement in PCA and UMAP
        dX = X_pca[nbrs] - X_pca[i]       # (k, d_pca)
        dU = X_umap[nbrs] - X_umap[i]     # (k, 2)

        # Least-squares: dU ≈ dX @ A  →  A = (dX^T dX)^{-1} dX^T dU
        A, _, _, _ = np.linalg.lstsq(dX, dU, rcond=None)

        # Project PCA velocity through the local Jacobian
        V_umap[i] = V_pca[i] @ A

    adata.obsm[velocity_key_umap] = V_umap

    print(f"  Velocity projected to UMAP ({n_cells} cells)")

    return adata


# =====================================================================
# 5. ORCHESTRATOR
# =====================================================================


def velocity(
    adata: AnnData,
    basis: str = "X_pca",
    smooth: bool = True,
    # Windowing params
    n_clusters: Optional[int] = None,
    window_size: Optional[int] = None,
    overlap_fraction: float = 0.5,
    min_window_size: int = 20,
    spatial_key: Optional[str] = None,
    tail_handling: str = "force",
    tail_threshold: int = 10,
    # OT params
    reg: float = 0.05,
    lambda_time: float = 1.0,
    lambda_knn: float = 1.0,
    # Smoothing params
    n_epochs: int = 200,
    hidden_dim: int = 128,
    lambda_smooth: float = 0.5,
    lambda_curl: float = 0.0,
    lambda_divergence: float = 0.0,
    k_smooth: int = 15,
    use_pseudotime: bool = True,
    # Output
    project_umap: bool = True,
    random_state: int = 42,
    verbose: bool = True,
) -> AnnData:
    """
    Run the full VelOT velocity pipeline.

    This is a convenience function that calls, in order:
      1. ``build_windows()`` — spatial-temporal windowing
      2. ``compute_ot_velocity()`` — local OT velocity
      3. ``smooth_velocity()`` — neural smoothing (optional)
      4. ``project_to_umap()`` — PCA → UMAP projection (optional)

    Parameters
    ----------
    adata
        Preprocessed AnnData (run ``velot.pp.prepare()`` first).
    basis
        Embedding key for velocity computation.
    smooth
        Whether to apply neural smoothing.
    project_umap
        Whether to project velocity to UMAP for visualization.
    verbose
        Whether to print progress.

    Returns
    -------
    adata with velocity fields computed.

    Example
    -------
    ::

        import velot
        import scvelo as scv

        adata = scv.datasets.pancreas()
        velot.pp.prepare(adata, root_cluster="Ductal")
        velot.tl.velocity(adata)
        velot.pl.velocity_stream(adata)
    """
    _check_fields(adata, obsm_keys=[basis], obs_keys=["pseudotime"])

    if verbose:
        print("VelOT: Computing velocity field")
        print(f"  Basis: {basis} ({adata.obsm[basis].shape[1]}D)")
        print(f"  Cells: {adata.n_obs}")
        print(f"  Smoothing: {'ON' if smooth else 'OFF'}")
        print()

    # Step 1: Windowing
    if verbose:
        print("[1/4] Building spatial-temporal windows...")
    build_windows(
        adata,
        basis=basis,
        n_clusters=n_clusters,
        window_size=window_size,
        overlap_fraction=overlap_fraction,
        min_window_size=min_window_size,
        spatial_key=spatial_key,
        tail_handling=tail_handling,
        tail_threshold=tail_threshold,
        random_state=random_state,
    )

    # Step 2: OT velocity
    if verbose:
        print("\n[2/4] Computing OT velocity...")
    compute_ot_velocity(
        adata,
        basis=basis,
        reg=reg,
        lambda_time=lambda_time,
        lambda_knn=lambda_knn,
    )

    # Step 3: Smoothing
    if smooth:
        if verbose:
            print("\n[3/4] Smoothing velocity field...")
        v_key = "velot_velocity_raw_pca" if basis == "X_pca" else "velot_velocity_raw_umap"
        smooth_velocity(
            adata,
            basis=basis.split("X_")[1],
            velocity_key=v_key,
            n_epochs=n_epochs,
            hidden_dim=hidden_dim,
            lambda_smooth=lambda_smooth,
            lambda_curl=lambda_curl,
            lambda_divergence=lambda_divergence,
            k_smooth=k_smooth,
            use_pseudotime=use_pseudotime,
            random_state=random_state,
            verbose=verbose,
        )
    else:
        if verbose:
            print("\n[3/4] Smoothing: SKIPPED")

    # Step 4: Project to UMAP
    if project_umap and "X_umap" in adata.obsm:
        if verbose:
            print("\n[4/4] Projecting to UMAP...")
        project_to_umap(adata, "velot_velocity_raw_pca", "velot_velocity_raw_umap")
        if smooth:
            project_to_umap(adata, "velot_velocity_pca", "velot_velocity_umap")
    else:
        if verbose:
            print("\n[4/4] UMAP projection: SKIPPED")

    if verbose:
        print("\nVelOT: Done.")

    return adata


# =====================================================================
# INTERNAL HELPERS
# =====================================================================


def _check_fields(
    adata: AnnData,
    obsm_keys: list = None,
    obs_keys: list = None,
    uns_keys: list = None,
):
    """Validate that required fields exist in adata."""
    obsm_keys = obsm_keys or []
    obs_keys = obs_keys or []
    uns_keys = uns_keys or []

    for key in obsm_keys:
        if key not in adata.obsm:
            raise ValueError(
                f"'{key}' not found in adata.obsm. "
                f"Run velot.pp.prepare() first. "
                f"Available keys: {list(adata.obsm.keys())}"
            )
    for key in obs_keys:
        if key not in adata.obs:
            raise ValueError(
                f"'{key}' not found in adata.obs. "
                f"Run velot.pp.prepare() first. "
                f"Available columns: {list(adata.obs.columns)}"
            )
    for key in uns_keys:
        if key not in adata.uns:
            raise ValueError(
                f"'{key}' not found in adata.uns. "
                f"Run the required upstream step first."
            )


# =====================================================================
# 6. GRID SEARCH
# =====================================================================


def gridsearch(
    adata: AnnData,
    param_grid: dict,
    cluster_edges: Optional[list] = None,
    cluster_key: str = "clusters",
    velocity_metric_key: str = "velocity_umap",
    fixed_params: Optional[dict] = None,
    pseudotime_key: str = "pseudotime",
    verbose: bool = True,
):
    """
    Run the VelOT pipeline across multiple parameter combinations
    and collect evaluation metrics for each.

    Parameters
    ----------
    adata
        Preprocessed AnnData. Must already have PCA, neighbors,
        UMAP, and pseudotime computed (everything from ``velot.pp``).
        A fresh copy is used for each parameter combination.
    param_grid
        Dictionary mapping parameter names to lists of values.
        Parameter names must match arguments of ``velot.tl.velocity()``.
        Example::

            param_grid = {
                "basis": ["X_pca"],
                "reg": [0.01, 0.05, 0.1],
                "lambda_smooth": [0.1, 0.5, 1.0],
                "n_clusters": [10, 20],
            }

    cluster_edges
        Transition edges for CBDir metric. If None, only ICCoh
        is computed.
    cluster_key
        Column in adata.obs with cluster labels.
    velocity_metric_key
        Key in adata.obsm to evaluate metrics on.
    fixed_params
        Parameters passed to ``velot.tl.velocity()`` for every run
        that are NOT part of the grid. Example::

            fixed_params = {"smooth": True, "n_epochs": 200}

    pseudotime_key
        Column in adata.obs with pseudotime. Used to verify it
        exists before running.
    verbose
        Whether to print progress.

    Returns
    -------
    pandas DataFrame with one row per parameter combination and
    columns for each parameter, ICCoh mean, CBDir mean, and
    per-cluster/per-edge scores.

    Example
    -------
    ::

        import velot

        adata = velot.datasets.dentategyrus()
        # ... preprocessing ...

        param_grid = {
            "reg": [0.01, 0.05, 0.1, 0.5],
            "lambda_smooth": [0.1, 0.5, 1.0],
            "n_clusters": [10, 20, 30],
        }

        edges = [
            ("OPC", "OL"),
            ("Neuroblast", "Granule immature"),
        ]

        results = velot.tl.gridsearch(
            adata,
            param_grid,
            cluster_edges=edges,
            cluster_key="clusters",
            fixed_params={"basis": "X_pca", "n_epochs": 200},
        )

        # Best by ICCoh
        print(results.sort_values("iccoh_mean", ascending=False).head())

        # Best by CBDir
        print(results.sort_values("cbdir_mean", ascending=False).head())
    """
    import pandas as pd
    from itertools import product
    import time as _time

    # Validate
    if pseudotime_key not in adata.obs:
        raise ValueError(
            f"'{pseudotime_key}' not found in adata.obs. "
            f"Run velot.pp.pseudotime() first."
        )

    fixed_params = fixed_params or {}

    # Build all combinations
    param_names = sorted(param_grid.keys())
    param_values = [param_grid[k] for k in param_names]
    combinations = list(product(*param_values))
    n_combos = len(combinations)

    if verbose:
        print(f"VelOT Grid Search: {n_combos} combinations")
        print(f"  Parameters: {param_names}")
        print(f"  Fixed: {fixed_params}")
        print()

    # Run each combination
    rows = []

    try:
        from tqdm.auto import tqdm
        iterator = tqdm(
            enumerate(combinations),
            total=n_combos,
            desc="VelOT Grid Search",
            disable=not verbose,
        )
    except ImportError:
        iterator = enumerate(combinations)
        if verbose:
            warnings.warn(
                "Install tqdm for progress bars: pip install tqdm",
                stacklevel=2,
            )

    for combo_i, combo in iterator:
        params = dict(zip(param_names, combo))
        run_params = {**fixed_params, **params}

        # Update progress bar description
        if hasattr(iterator, "set_postfix"):
            short_params = {k: v for k, v in params.items()}
            iterator.set_postfix(short_params, refresh=True)

        # Work on a fresh copy each time
        ad = adata.copy()

        t0 = _time.time()
        try:
            velocity(
                ad,
                verbose=False,
                **run_params,
            )
            elapsed = _time.time() - t0

            # Compute metrics
            from . import metrics as _metrics

            iccoh_scores, iccoh_mean = _metrics.inner_cluster_coherence(
                ad,
                cluster_key=cluster_key,
                velocity_key=velocity_metric_key,
            )

            row = {**params}
            row["iccoh_mean"] = iccoh_mean
            for cat, score in iccoh_scores.items():
                row[f"iccoh_{cat}"] = score

            if cluster_edges is not None:
                cbdir_scores, cbdir_mean = _metrics.cross_boundary_correctness(
                    ad,
                    cluster_edges,
                    cluster_key=cluster_key,
                    velocity_key=velocity_metric_key,
                )
                row["cbdir_mean"] = cbdir_mean
                for (u, v), score in cbdir_scores.items():
                    row[f"cbdir_{u}_to_{v}"] = score
            else:
                row["cbdir_mean"] = float("nan")

            row["elapsed_seconds"] = elapsed
            row["status"] = "ok"

            # Update progress bar with latest metrics
            if hasattr(iterator, "set_postfix"):
                post = {**short_params}
                post["ICCoh"] = f"{iccoh_mean:.3f}"
                if not np.isnan(row.get("cbdir_mean", float("nan"))):
                    post["CBDir"] = f"{row['cbdir_mean']:.3f}"
                iterator.set_postfix(post, refresh=True)

        except Exception as e:
            row = {**params}
            row["iccoh_mean"] = float("nan")
            row["cbdir_mean"] = float("nan")
            row["elapsed_seconds"] = _time.time() - t0
            row["status"] = f"error: {str(e)[:80]}"

            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix({"status": "FAILED"}, refresh=True)

        rows.append(row)

        if verbose and row["status"] == "ok":
            msg = f"    ICCoh={row['iccoh_mean']:.3f}"
            if "cbdir_mean" in row and not np.isnan(row.get("cbdir_mean", float("nan"))):
                msg += f"  CBDir={row['cbdir_mean']:.3f}"
            msg += f"  ({elapsed:.1f}s)"
            print(msg)

    # Build DataFrame
    df = pd.DataFrame(rows)

    # Reorder columns: params first, then summary metrics, then details
    param_cols = param_names
    summary_cols = ["iccoh_mean", "cbdir_mean", "elapsed_seconds", "status"]
    detail_cols = [c for c in df.columns if c not in param_cols + summary_cols]
    col_order = param_cols + summary_cols + sorted(detail_cols)
    df = df[[c for c in col_order if c in df.columns]]

    if verbose:
        print()
        print("=" * 60)
        print("Grid Search Complete")
        print("=" * 60)

        best_iccoh = df.loc[df["iccoh_mean"].idxmax()]
        print(f"\nBest ICCoh ({best_iccoh['iccoh_mean']:.3f}):")
        for p in param_names:
            print(f"  {p}: {best_iccoh[p]}")

        if cluster_edges is not None:
            best_cbdir = df.loc[df["cbdir_mean"].idxmax()]
            print(f"\nBest CBDir ({best_cbdir['cbdir_mean']:.3f}):")
            for p in param_names:
                print(f"  {p}: {best_cbdir[p]}")

    return df

# =====================================================================
# 7. CELL TRAJECTORY TRACING
# =====================================================================

def compute_trajectories(
    adata: AnnData,
    start_cells: Optional[np.ndarray] = None,
    start_cluster: Optional[str] = None,
    target_cluster: Optional[str] = None,
    start_pseudotime: Optional[float] = None,
    end_pseudotime: Optional[float] = None,
    n_trajectories: int = 20,
    n_steps: int = 200,
    step_size: float = 0.05,
    direction: str = "forward",
    velocity_key: str = "velot_velocity_pca",
    basis: str = "X_pca",
    cluster_key: str = "clusters",
    k_velocity: int = 15,
    use_network: bool = True,
    evolve_pseudotime: bool = True,
    max_attempts_factor: int = 10,
    random_state: int = 42,
) -> AnnData:
    """
    Compute cell trajectories by integrating the velocity field.

    Supports three modes:
      - ``"forward"``: follow the flow from starting cells.
      - ``"backward"``: go against the flow to trace origins.
      - ``"both"``: compute both directions.

    Starting cells can be selected by:
      - Explicit indices (``start_cells``).
      - Cluster name (``start_cluster``).
      - Pseudotime range (``start_pseudotime`` / ``end_pseudotime``).
      - Any combination: cluster + pseudotime range narrows the selection.

    When ``target_cluster`` is specified, only trajectories that
    reach the target are kept. The function will attempt up to
    ``n_trajectories * max_attempts_factor`` integrations to find
    enough successful trajectories.

    Parameters
    ----------
    adata
        Must have velocity computed.
    start_cells
        Array of cell indices to start from. If provided,
        ``start_cluster`` and pseudotime range are ignored.
    start_cluster
        Cluster name to select starting cells from.
        Can be combined with ``start_pseudotime`` / ``end_pseudotime``
        to further narrow the selection.
    target_cluster
        If provided, only keep trajectories whose terminal cluster
        (forward) or origin cluster (backward) matches this value.
    start_pseudotime
        Lower bound of pseudotime for selecting starting cells.
        Defaults to ``None`` (no lower bound). Can be used alone
        or combined with ``start_cluster``.
    end_pseudotime
        Upper bound of pseudotime for selecting starting cells.
        Defaults to ``None`` (no upper bound). Can be used alone
        or combined with ``start_cluster``.
    n_trajectories
        Number of successful trajectories to collect.
    n_steps
        Maximum integration steps.
    step_size
        Euler step size as fraction of mean velocity magnitude.
    direction
        ``"forward"``, ``"backward"``, or ``"both"``.
    velocity_key
        Key in adata.obsm with velocity vectors.
    basis
        Embedding key for cell coordinates.
    cluster_key
        Cluster annotation column.
    k_velocity
        KNN for velocity interpolation (fallback mode).
    use_network
        If True and the smoothing network is available, query
        the continuous velocity field directly.
    evolve_pseudotime
        If True, pseudotime evolves along the trajectory.
    max_attempts_factor
        When ``target_cluster`` is set, try up to
        ``n_trajectories * max_attempts_factor`` starting cells
        to find enough trajectories that reach the target.
    random_state
        Random seed.

    Returns
    -------
    adata with ``adata.uns['velot_trajectories']``.

    The stored metadata for each trajectory includes an ``"id"``
    field that can be used with ``velot.pl.trajectories(trajectory_ids=...)``
    to plot specific trajectories.

    Examples
    --------
    All backward trajectories from Alpha::

        velot.tl.compute_trajectories(
            adata, start_cluster="Alpha", direction="backward",
        )

    Only backward trajectories from Alpha that reach Ngn3 low EP::

        velot.tl.compute_trajectories(
            adata, start_cluster="Alpha", direction="backward",
            target_cluster="Ngn3 low EP", n_trajectories=5,
        )

    Forward from progenitors that reach Epsilon::

        velot.tl.compute_trajectories(
            adata, start_cluster="Ngn3 low EP", direction="forward",
            target_cluster="Epsilon", n_trajectories=10,
        )

    Forward from early cells (pseudotime 0 to 0.1) regardless of cluster::

        velot.tl.compute_trajectories(
            adata, start_pseudotime=0.0, end_pseudotime=0.1,
            direction="forward",
        )

    Backward from Alpha cells with pseudotime > 0.8::

        velot.tl.compute_trajectories(
            adata, start_cluster="Alpha", start_pseudotime=0.8,
            direction="backward",
        )
    """
    _check_fields(adata, obsm_keys=[basis, velocity_key])

    if direction not in ("forward", "backward", "both"):
        raise ValueError(
            f"direction must be 'forward', 'backward', or 'both', "
            f"got '{direction}'."
        )

    X = adata.obsm[basis]
    V = adata.obsm[velocity_key]
    pseudotime = adata.obs["pseudotime"].values
    n_cells, dim = X.shape

    tree = cKDTree(X)

    # Check if continuous field is available
    has_network = (
        use_network
        and "velot_smoothing" in adata.uns
        and "network" in adata.uns["velot_smoothing"]
    )

    if has_network:
        net = adata.uns["velot_smoothing"]["network"]
        use_pt = adata.uns["velot_smoothing"]["use_pseudotime"]
        net.eval()

    # ------------------------------------------------------------------
    # Select candidate starting cells
    # Priority: start_cells > (start_cluster and/or pseudotime range)
    # ------------------------------------------------------------------
    has_pt_range = (start_pseudotime is not None or end_pseudotime is not None)

    if start_cells is not None:
        # Explicit indices — use as-is, no further filtering
        all_candidates = np.asarray(start_cells)

    elif start_cluster is not None or has_pt_range:
        # Start with all cells, then apply filters
        if start_cluster is not None:
            # Cluster filter
            if cluster_key not in adata.obs:
                raise ValueError(f"'{cluster_key}' not found in adata.obs.")
            cluster_mask = adata.obs[cluster_key].astype(str) == str(start_cluster)
            all_candidates = np.where(cluster_mask)[0]
            if len(all_candidates) == 0:
                available = sorted(
                    adata.obs[cluster_key].astype(str).unique().tolist()
                )
                raise ValueError(
                    f"Cluster '{start_cluster}' not found. "
                    f"Available: {available}"
                )
        else:
            # No cluster filter — all cells are candidates
            all_candidates = np.arange(n_cells)

        # Apply pseudotime range filter on top
        if has_pt_range:
            pt_min = start_pseudotime if start_pseudotime is not None else -np.inf
            pt_max = end_pseudotime if end_pseudotime is not None else np.inf
            pt_vals = pseudotime[all_candidates]
            pt_mask = (pt_vals >= pt_min) & (pt_vals <= pt_max)
            all_candidates = all_candidates[pt_mask]

            if len(all_candidates) == 0:
                # Helpful error message
                if start_cluster is not None:
                    cluster_pts = pseudotime[
                        adata.obs[cluster_key].astype(str) == str(start_cluster)
                    ]
                    raise ValueError(
                        f"No cells in cluster '{start_cluster}' with "
                        f"pseudotime in [{pt_min}, {pt_max}]. "
                        f"Cluster pseudotime range: "
                        f"[{cluster_pts.min():.3f}, {cluster_pts.max():.3f}]"
                    )
                else:
                    raise ValueError(
                        f"No cells with pseudotime in [{pt_min}, {pt_max}]. "
                        f"Data pseudotime range: "
                        f"[{pseudotime.min():.3f}, {pseudotime.max():.3f}]"
                    )
    else:
        raise ValueError(
            "Provide at least one of: start_cells, start_cluster, "
            "or a pseudotime range (start_pseudotime / end_pseudotime)."
        )

    # Report selection
    selection_desc = []
    if start_cluster is not None:
        selection_desc.append(f"cluster='{start_cluster}'")
    if has_pt_range:
        pt_min_str = f"{start_pseudotime:.3f}" if start_pseudotime is not None else "min"
        pt_max_str = f"{end_pseudotime:.3f}" if end_pseudotime is not None else "max"
        selection_desc.append(f"pseudotime=[{pt_min_str}, {pt_max_str}]")
    if selection_desc:
        print(f"  Starting cells: {len(all_candidates)} candidates "
              f"({', '.join(selection_desc)})")

    rng = np.random.RandomState(random_state)

    # Step size: proportional to mean inter-cell distance
    sample_idx = np.random.choice(n_cells, min(500, n_cells), replace=False)
    mean_nn_dist = np.mean(tree.query(X[sample_idx], k=2)[0][:, 1])
    dt = step_size * mean_nn_dist

    # Manifold boundary: generous threshold
    boundary_threshold = 10 * mean_nn_dist

    # UMAP availability
    has_umap = "X_umap" in adata.obsm
    if has_umap:
        X_umap = adata.obsm["X_umap"]

    def _velocity_at(pos, pt_value):
        if has_network:
            pos_t = torch.tensor(
                pos.reshape(1, -1).astype(np.float32), device=DEVICE
            )
            pt_t = torch.tensor(
                [[pt_value]], dtype=torch.float32, device=DEVICE
            )
            with torch.no_grad():
                v = net(pos_t, pt_t if use_pt else None)
            return v.cpu().numpy().flatten()
        else:
            dists, indices = tree.query(pos, k=k_velocity)
            weights = 1.0 / (dists + 1e-10)
            weights = weights / weights.sum()
            return np.sum(V[indices] * weights[:, None], axis=0)

    def _estimate_pseudotime_at(pos):
        dists, indices = tree.query(pos, k=min(10, n_cells - 1))
        weights = 1.0 / (dists + 1e-10)
        weights = weights / weights.sum()
        return float(np.sum(pseudotime[indices] * weights))

    def _project_to_umap(pos):
        dists, indices = tree.query(pos, k=min(15, n_cells - 1))
        dX = X[indices] - pos
        dU = X_umap[indices] - X_umap[indices[0]]
        try:
            A, _, _, _ = np.linalg.lstsq(dX, dU, rcond=None)
            return X_umap[indices[0]] + (pos - X[indices[0]]) @ A
        except np.linalg.LinAlgError:
            return X_umap[indices[0]]

    def _nearest_cluster(pos):
        _, idx = tree.query(pos, k=1)
        return str(adata.obs[cluster_key].iloc[idx])

    def _integrate(cell_idx, sign=1.0):
        """Integrate by always stepping from real cell positions."""

        current_cell = cell_idx
        path_pca = [X[current_cell].copy()]
        clusters_visited = [_nearest_cluster(X[current_cell])]
        pseudotimes_along = [float(pseudotime[current_cell])]

        visited_cells = {current_cell}
        stall_count = 0

        snap_dt = max(dt, mean_nn_dist * 0.5)

        for step in range(n_steps):
            pos = X[current_cell]
            pt_current = float(pseudotime[current_cell])

            v = _velocity_at(pos, pt_current)
            v_mag = np.linalg.norm(v)

            if v_mag < 1e-8:
                break

            v_unit = v / v_mag
            pos_candidate = pos + sign * snap_dt * v_unit

            k_snap = min(5, n_cells - 1)
            dists_snap, idx_snap = tree.query(pos_candidate, k=k_snap)

            best_cell = None
            best_score = -np.inf

            for i in range(k_snap):
                candidate_cell = idx_snap[i]

                displacement = X[candidate_cell] - X[current_cell]
                disp_mag = np.linalg.norm(displacement)

                if disp_mag < 1e-10:
                    continue

                alignment = np.dot(sign * v_unit, displacement / disp_mag)

                proximity = 1.0 / (dists_snap[i] + 1e-10)
                score = alignment * proximity

                if score > best_score:
                    best_score = score
                    best_cell = candidate_cell

            if best_cell is None or best_cell == current_cell:
                pos_candidate2 = pos + sign * snap_dt * 2 * v_unit
                _, idx2 = tree.query(pos_candidate2, k=k_snap)
                for i in range(k_snap):
                    if idx2[i] != current_cell:
                        best_cell = idx2[i]
                        break

            if best_cell is None or best_cell == current_cell:
                stall_count += 1
                if stall_count > 5:
                    break
                continue

            stall_count = 0
            current_cell = best_cell
            path_pca.append(X[current_cell].copy())

            pseudotimes_along.append(float(pseudotime[current_cell]))
            clusters_visited.append(
                str(adata.obs[cluster_key].iloc[current_cell])
            )

            # Boundary check
            if evolve_pseudotime:
                pt_now = float(pseudotime[current_cell])
                if pt_now < -0.1 or pt_now > 1.1:
                    break

            if current_cell in visited_cells:
                pass
            visited_cells.add(current_cell)

        return np.array(path_pca), clusters_visited, pseudotimes_along

    def _matches_target(clusters_visited, sign):
        """Check if this trajectory reached the target cluster."""
        if target_cluster is None:
            return True
        if sign > 0:
            return clusters_visited[-1] == target_cluster
        else:
            return clusters_visited[-1] == target_cluster

    # ------------------------------------------------------------------
    # Integrate trajectories with target filtering
    # ------------------------------------------------------------------
    trajectories_pca = []
    trajectories_umap = []
    trajectory_metadata = []

    directions_to_run = []
    if direction == "forward":
        directions_to_run = [("forward", 1.0)]
    elif direction == "backward":
        directions_to_run = [("backward", -1.0)]
    elif direction == "both":
        directions_to_run = [("forward", 1.0), ("backward", -1.0)]

    for dir_name, sign in directions_to_run:
        collected = 0
        attempts = 0
        max_attempts = n_trajectories * max_attempts_factor

        shuffled = rng.permutation(all_candidates)
        candidate_idx = 0

        while collected < n_trajectories and attempts < max_attempts:
            cell_idx = shuffled[candidate_idx % len(shuffled)]
            candidate_idx += 1
            attempts += 1

            path_pca, clusters_visited, pt_along = _integrate(cell_idx, sign)

            if not _matches_target(clusters_visited, sign):
                continue

            if len(path_pca) < 5:
                continue

            traj_id = len(trajectory_metadata)
            trajectories_pca.append(path_pca)

            if has_umap:
                path_umap = np.array(
                    [_project_to_umap(p) for p in path_pca]
                )
                trajectories_umap.append(path_umap)

            if sign > 0:
                origin_cluster = clusters_visited[0]
                terminal_cluster = clusters_visited[-1]
            else:
                origin_cluster = clusters_visited[-1]
                terminal_cluster = clusters_visited[0]

            trajectory_metadata.append({
                "id": traj_id,
                "start_cell": int(cell_idx),
                "start_cluster": str(
                    adata.obs[cluster_key].iloc[cell_idx]
                ),
                "direction": dir_name,
                "n_steps": len(path_pca),
                "clusters_visited": clusters_visited,
                "pseudotime_along": pt_along,
                "origin_cluster": origin_cluster,
                "terminal_cluster": terminal_cluster,
            })

            collected += 1

        if target_cluster is not None:
            print(
                f"  {dir_name.capitalize()}: found "
                f"{collected}/{n_trajectories} "
                f"trajectories reaching '{target_cluster}' "
                f"({attempts} attempts)"
            )

    adata.uns["velot_trajectories"] = {
        "paths_pca": trajectories_pca,
        "paths_umap": trajectories_umap if has_umap else [],
        "metadata": trajectory_metadata,
        "start_cluster": start_cluster,
        "target_cluster": target_cluster,
        "start_pseudotime": start_pseudotime,
        "end_pseudotime": end_pseudotime,
        "direction": direction,
        "n_trajectories": len(trajectory_metadata),
        "n_steps": n_steps,
        "step_size": step_size,
        "used_network": has_network,
        "evolved_pseudotime": evolve_pseudotime,
    }

    # Summary
    print(
        f"  Total: {len(trajectory_metadata)} trajectories "
        f"({direction}) from '{start_cluster}'"
        + (f" to '{target_cluster}'" if target_cluster else "")
    )

    if has_network:
        print(f"  Using continuous velocity field (neural network)")

    for dir_name, _ in directions_to_run:
        dir_meta = [
            m for m in trajectory_metadata if m["direction"] == dir_name
        ]
        if not dir_meta:
            continue
        if dir_name == "forward":
            endpoints = [m["terminal_cluster"] for m in dir_meta]
            label = "Terminal"
        else:
            endpoints = [m["origin_cluster"] for m in dir_meta]
            label = "Origin"

        unique, counts = np.unique(endpoints, return_counts=True)
        summary = dict(zip(unique, counts))
        print(f"  {dir_name.capitalize()} — {label} clusters: {summary}")

    return adata

def simulate_flow(
    adata: AnnData,
    n_particles: int = 200,
    source_cluster: Optional[str] = None,
    source_pseudotime_min: float = 0.0,
    source_pseudotime_max: float = 0.1,
    source_x_lim: tuple = (-np.inf, np.inf),
    source_y_lim: tuple = (-np.inf, np.inf),
    n_steps: int = 300,
    step_size: float = 0.05,
    diffusion: float = 0.5,
    basis: str = "X_pca",
    cluster_key: str = "clusters",
    noise_scale: float = 0.0,
    random_state: int = 42,
) -> AnnData:
    """
    Simulate a flow of particles through the learned velocity field.

    Drops particles at the "top of the river" (low pseudotime or
    a source cluster) and integrates them forward through the
    velocity field with optional stochastic diffusion.

    The integration follows a stochastic differential equation:

        x_{n+1} = x_n + dt * v(x_n, τ_n) + sqrt(2 * D * dt) * η_n

    where D is the diffusion coefficient and η is Gaussian noise.
    The diffusion allows particles starting from the same region
    to explore different branches at bifurcation points, producing
    realistic fate distributions.

    Parameters
    ----------
    adata
        Must have the smoothing network trained.
    n_particles
        Number of particles to simulate.
    source_cluster
        Cluster to seed particles from. If None, uses cells
        with lowest pseudotime.
    source_pseudotime_max
        If source_cluster is None, seed from cells with
        pseudotime below this value.
    n_steps
        Maximum number of integration steps.
    step_size
        Euler step size as fraction of mean velocity magnitude.
    diffusion
        Diffusion coefficient controlling stochasticity.
        0.0 = deterministic (all trajectories converge).
        0.1-0.5 = mild stochasticity (some branching exploration).
        1.0+ = strong stochasticity (wide exploration).
        The noise is scaled relative to the local velocity
        magnitude so it is adaptive.
    basis
        PCA embedding key.
    cluster_key
        Cluster annotation column.
    noise_scale
        Gaussian noise added to INITIAL positions only.
        Separate from diffusion which is added at every step.
    random_state
        Random seed.

    Returns
    -------
    adata with ``adata.uns['velot_flow']``.

    Examples
    --------
    Deterministic flow (trajectories will converge)::

        velot.tl.simulate_flow(adata, diffusion=0.0, ...)

    Stochastic flow (trajectories explore branches)::

        velot.tl.simulate_flow(adata, diffusion=0.3, ...)

    Strong diffusion (wide exploration)::

        velot.tl.simulate_flow(adata, diffusion=1.0, ...)
    """
    _check_fields(adata, obsm_keys=[basis], obs_keys=["pseudotime"])

    if "velot_smoothing" not in adata.uns or "network" not in adata.uns["velot_smoothing"]:
        raise ValueError(
            "Smoothing network not found. Run velot.tl.velocity(smooth=True) first."
        )

    X = adata.obsm[basis]
    V = adata.obsm["velot_velocity"]
    pseudotime = adata.obs["pseudotime"].values
    n_cells, dim = X.shape

    net = adata.uns["velot_smoothing"]["network"]
    use_pt = adata.uns["velot_smoothing"]["use_pseudotime"]
    net.eval()

    tree = cKDTree(X)
    rng = np.random.RandomState(random_state)

    # Manifold boundary
    sample_dists = tree.query(X[:min(200, n_cells)], k=2)[0][:, 1]
    mean_nn_dist = np.mean(sample_dists)

    # UMAP
    has_umap = "X_umap" in adata.obsm
    if has_umap:
        X_umap = adata.obsm["X_umap"]

    # Replace step size calibration:
    sample_idx = np.random.choice(n_cells, min(500, n_cells), replace=False)
    mean_nn_dist = np.mean(tree.query(X[sample_idx], k=2)[0][:, 1])
    dt = step_size * mean_nn_dist
    diffusion_scale = diffusion * mean_nn_dist * np.sqrt(dt)
    boundary_threshold = 10 * mean_nn_dist

    # ------------------------------------------------------------------
    # Seed particles
    # ------------------------------------------------------------------
    if source_cluster is not None:
        mask = adata.obs[cluster_key].astype(str) == str(source_cluster)
        candidates = np.where(mask)[0]
        if len(candidates) == 0:
            raise ValueError(f"Cluster '{source_cluster}' not found.")
        selected = rng.choice(
            candidates,
            size=min(n_particles, len(candidates)),
            replace=(n_particles > len(candidates)),
        )
        source_label = source_cluster
    elif (source_pseudotime_min is not None) & (source_pseudotime_max is not None):
        mask = (source_pseudotime_min <= pseudotime) & (pseudotime <= source_pseudotime_max)
        candidates = np.where(mask)[0]
        if len(candidates) == 0:
            candidates = np.argsort(pseudotime)[:max(10, n_particles)]
        selected = rng.choice(candidates, size=min(n_particles, len(candidates)), replace=True)
        source_label = f"{source_pseudotime_min} ≤ pseudotime ≤ {source_pseudotime_max}"
    else:
        mask = ((X[:, 0] >= source_x_lim[0]) & (X[:, 0] <= source_x_lim[1]) &
        (X[:, 1] >= source_y_lim[0]) & (X[:, 1] <= source_y_lim[1]))

        candidates = np.where(mask)[0]
        if len(candidates) == 0:
            candidates = np.argsort(pseudotime)[:max(10, n_particles)]
        selected = rng.choice(candidates, size=min(n_particles, len(candidates)), replace=True)
        source_label = f"x lim $\in$ {source_x_lim}; x lim $\in$ {source_y_lim}"

    # Initial positions
    initial_positions = X[selected].copy()
    if noise_scale > 0:
        initial_positions += rng.normal(0, noise_scale, initial_positions.shape)

    initial_pseudotimes = pseudotime[selected].copy()

    print(f"  Seeding {len(selected)} particles from {source_label}")
    print(f"  Diffusion coefficient: {diffusion} "
          f"(scale={diffusion_scale:.4f} per step)")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _velocity_at(pos, pt_value):
        pos_t = torch.tensor(
            pos.reshape(1, -1).astype(np.float32), device=DEVICE
        )
        pt_t = torch.tensor(
            [[pt_value]], dtype=torch.float32, device=DEVICE
        )
        with torch.no_grad():
            v = net(pos_t, pt_t if use_pt else None)
        return v.cpu().numpy().flatten()

    def _estimate_pseudotime_at(pos):
        dists, indices = tree.query(pos, k=min(10, n_cells - 1))
        weights = 1.0 / (dists + 1e-10)
        weights = weights / weights.sum()
        return float(np.sum(pseudotime[indices] * weights))

    def _nearest_cluster(pos):
        _, idx = tree.query(pos, k=1)
        return str(adata.obs[cluster_key].iloc[idx])

    def _project_to_umap(pos):
        dists, indices = tree.query(pos, k=min(15, n_cells - 1))
        dX = X[indices] - pos
        dU = X_umap[indices] - X_umap[indices[0]]
        try:
            A, _, _, _ = np.linalg.lstsq(dX, dU, rcond=None)
            return X_umap[indices[0]] + (pos - X[indices[0]]) @ A
        except np.linalg.LinAlgError:
            return X_umap[indices[0]]

    # ------------------------------------------------------------------
    # Integrate all particles (SDE)
    # ------------------------------------------------------------------
    all_paths = []
    all_paths_umap = []
    all_pseudotimes = []
    all_clusters = []

    for p_idx in range(len(selected)):
        pos = initial_positions[p_idx].copy()
        pt_current = float(initial_pseudotimes[p_idx])

        path = [pos.copy()]
        pt_along = [pt_current]
        cl_along = [_nearest_cluster(pos)]

        for step in range(n_steps):
            # Deterministic drift
            v = _velocity_at(pos, pt_current)
            v_mag = np.linalg.norm(v)

            if v_mag < 1e-8:
                break

            # Stochastic diffusion
            # Scale noise by local velocity magnitude so that
            # fast-moving regions get proportionally less noise
            # and bifurcation points (where velocity is ambiguous)
            # get relatively more exploration
            if diffusion > 0:
                noise = rng.normal(0, 1, dim)
                # Adaptive: noise is perpendicular-biased
                # Project out the velocity direction to get noise
                # mostly perpendicular to the flow
                v_unit = v / (v_mag + 1e-10)
                parallel = np.dot(noise, v_unit) * v_unit
                perpendicular = noise - parallel
                # Keep 80% perpendicular, 20% parallel
                # This lets particles spread across branches
                # without fighting the main flow direction
                noise = 0.2 * parallel + 0.8 * perpendicular
                noise = noise * diffusion_scale
            else:
                noise = 0.0

            # SDE step: drift + diffusion
            v_mag = np.linalg.norm(v)
            v_unit = v / (v_mag + 1e-10)
            pos = pos + dt * v_unit + noise
            path.append(pos.copy())

            # Update pseudotime
            pt_current = _estimate_pseudotime_at(pos)
            pt_along.append(pt_current)
            cl_along.append(_nearest_cluster(pos))

            # Stop conditions
            dist_to_nearest, _ = tree.query(pos, k=1)
            if dist_to_nearest > boundary_threshold:
                break
            if pt_current > 0.99:
                break

        path = np.array(path)
        all_paths.append(path)
        all_pseudotimes.append(pt_along)
        all_clusters.append(cl_along)

        if has_umap:
            path_umap = np.array([_project_to_umap(p) for p in path])
            all_paths_umap.append(path_umap)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    terminal_clusters = [cl[-1] for cl in all_clusters]
    unique_terminals, terminal_counts = np.unique(
        terminal_clusters, return_counts=True
    )
    terminal_fractions = dict(zip(
        unique_terminals,
        (terminal_counts / terminal_counts.sum()).round(3),
    ))

    final_pseudotimes = [pt[-1] for pt in all_pseudotimes]
    mean_steps = np.mean([len(p) for p in all_paths])

    adata.uns["velot_flow"] = {
        "particles": all_paths,
        "particles_umap": all_paths_umap if has_umap else [],
        "pseudotime_along": all_pseudotimes,
        "cluster_along": all_clusters,
        "initial_cells": selected,
        "source": source_label,
        "n_particles": len(selected),
        "diffusion": diffusion,
        "metadata": {
            "terminal_fractions": terminal_fractions,
            "mean_final_pseudotime": float(np.mean(final_pseudotimes)),
            "mean_trajectory_length": float(mean_steps),
        },
    }

    print(f"  Flow simulation complete:")
    print(f"    Mean trajectory length: {mean_steps:.0f} steps")
    print(f"    Mean final pseudotime: {np.mean(final_pseudotimes):.3f}")
    print(f"    Terminal fate distribution:")
    for cl, frac in sorted(terminal_fractions.items(), key=lambda x: -x[1]):
        count = int(terminal_counts[unique_terminals == cl][0])
        print(f"      {cl}: {frac:.1%} ({count} particles)")

    return adata
"""
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
from typing import Optional, Sequence

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

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    DEVICE = None


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
# 1b. WINDOW DIAGNOSTICS (how to choose window_size, reg and reg_m)
# =====================================================================


def _basis_suffix(adata, basis):
    """``"X_pca"`` -> ``"pca"``: the tag VelOT names its obsm keys with.

    Any embedding in ``adata.obsm`` may be used as the working basis, not
    just the PCA and the UMAP, so the tag is derived from the name rather
    than picked from a fixed pair.
    """
    if not isinstance(basis, str) or not basis.startswith("X_"):
        raise ValueError(f"basis must be an obsm key of the form 'X_<name>', "
                         f"got {basis!r}")
    if basis not in adata.obsm:
        raise KeyError(f"basis {basis!r} not found in adata.obsm; "
                       f"available: {sorted(adata.obsm)}")
    return basis.split("X_", 1)[1]


def window_diagnostics(
    adata: AnnData,
    basis: str = "X_pca",
    window_sizes: Sequence[int] = (50, 100, 200, 400, 800),
    overlap_fraction: float = 0.0,
    n_clusters: Optional[int] = 1,
    spatial_key: Optional[str] = None,
    min_window_size: int = 20,
    tail_handling: str = "drop",
    tail_threshold: int = 10,
    random_state: int = 42,
    verbose: bool = True,
):
    """
    Report, per candidate ``window_size``, how far consecutive windows sit
    apart compared with the spacing between neighbouring cells. No OT is
    solved, so this is cheap.

    Everything is measured in units of ``delta`` = the median distance
    from a source cell to its nearest other source cell (the sampling
    resolution of the data). For each window pair,

        move = distance from a source cell to its CHEAPEST target,

    and ``move / delta`` is the number of spacings a cell has to travel.
    That single number drives the three settings that matter:

    * **window_size** sets the signal. At ``move/delta`` near 1 the real
      displacement is the size of the local scatter, so which target a
      sharp plan picks is mostly sampling noise: the raw field is noisy
      (low ICCoh) even when it is right on average (CBDir near 1).
      Larger windows move the targets further away and raise that ratio,
      but each window then spans more pseudotime, so genuine change gets
      averaged inside it. Aim for the smallest window that gets
      ``move/delta`` clearly above 1 while each window still covers a
      small slice of pseudotime (``dtau`` column).
    * **reg** (with ``cost_scale="nn"``) is the blur of the plan in the
      same units: the plan spreads over roughly ``sqrt(reg)`` spacings,
      so keep ``reg`` at or below ``move2_median`` - beyond that the plan
      washes out into the window mean.
    * **reg_m** (unbalanced OT) is the give-up threshold in those units.
      A cell keeps its mass when its cheapest move costs less than about
      ``reg_m`` and drops it when the cost is far beyond. Put it between
      the bulk of the moves and the dead ends: a few times
      ``move2_p90``, and well below ``move2_max`` if that maximum comes
      from terminal windows you want the model to recognise.

    So the natural ordering is

        reg  <=  move2_median  <  reg_m  <<  (dead-end moves)

    Parameters
    ----------
    adata
        Needs ``adata.obsm[basis]`` and ``adata.obs['pseudotime']``.
    window_sizes
        Candidate window sizes to report.
    overlap_fraction, n_clusters, spatial_key, min_window_size,
    tail_handling, tail_threshold, random_state
        Passed to :func:`build_windows`, so the windows are exactly the
        ones the pipeline would use.

    Returns
    -------
    pandas.DataFrame, one row per window size:

    ``n_pairs``
        number of window pairs.
    ``no_source``
        fraction of cells that are never a source, so get no raw velocity.
    ``dtau``
        median pseudotime span of a window (fraction of the full range).
    ``move2_median`` / ``move2_p90`` / ``move2_max``
        SQUARED cheapest move in units of delta^2, summarised over pairs.
        These are the units of ``reg`` and ``reg_m``.
    ``move_median``
        the same as a distance (``sqrt`` of the above): "how many spacings
        a cell travels between consecutive windows".
    """
    import pandas as pd

    _check_fields(adata, obsm_keys=[basis], obs_keys=["pseudotime"])
    X = np.asarray(adata.obsm[basis], dtype=np.float64)
    tau = adata.obs["pseudotime"].values.astype(np.float64)
    tau_range = float(np.ptp(tau)) or 1.0

    keep_windows = adata.uns.get("velot_windows", None)
    rows = []
    for ws in window_sizes:
        ad = adata  # build_windows only writes uns/obs keys
        build_windows(
            ad, basis=basis, n_clusters=n_clusters, window_size=int(ws),
            overlap_fraction=overlap_fraction,
            min_window_size=min_window_size, spatial_key=spatial_key,
            tail_handling=tail_handling, tail_threshold=tail_threshold,
            random_state=random_state,
        )
        pairs = ad.uns["velot_windows"]["pairs"]
        moves, dtaus, sources = [], [], set()
        for idx_s, idx_t in pairs:
            idx_s, idx_t = np.asarray(idx_s), np.asarray(idx_t)
            sources.update(idx_s.tolist())
            delta2 = _pair_nn_scale(X[idx_s], fallback=1.0)
            d2 = pot.dist(X[idx_s], X[idx_t], metric="sqeuclidean")
            moves.append(float(np.median(d2.min(axis=1))) / max(delta2, 1e-12))
            dtaus.append(float(np.ptp(tau[idx_s])) / tau_range)
        rows.append(dict(
            window_size=int(ws), n_pairs=len(pairs),
            no_source=1.0 - len(sources) / adata.n_obs,
            dtau=float(np.median(dtaus)) if dtaus else float("nan"),
            move2_median=float(np.median(moves)) if moves else float("nan"),
            move2_p90=float(np.percentile(moves, 90)) if moves else float("nan"),
            move2_max=float(np.max(moves)) if moves else float("nan"),
            move_median=float(np.sqrt(np.median(moves))) if moves else float("nan"),
        ))

    if keep_windows is not None:
        adata.uns["velot_windows"] = keep_windows
    else:
        adata.uns.pop("velot_windows", None)

    df = pd.DataFrame(rows).set_index("window_size")
    if verbose:
        print("\nWindow diagnostics (distances in units of the nearest-"
              "neighbour spacing)")
        print(df.round(3).to_string())
        print("  reg <= move2_median  <  reg_m  <<  dead-end moves")
    return df


def choose_window_size(
    adata: AnnData,
    candidates: Sequence[int] = (50, 100, 200, 300, 500, 800),
    target_move: float = 2.0,
    max_dtau: float = 0.10,
    max_no_source: float = 0.10,
    basis: str = "X_pca",
    verbose: bool = True,
    **window_kwargs,
):
    """
    Pick ``window_size`` from the geometry of the data, not from a metric.

    A window is ELIGIBLE when it spans at most ``max_dtau`` of the
    pseudotime range and leaves at most ``max_no_source`` of the cells
    without a raw estimate (the cells of the final window of each
    cluster, which are never a source). Among those, the choice is the
    SMALLEST window whose typical move reaches ``target_move``, in units
    of the squared nearest-neighbour spacing — 2.0 is about 1.4 spacings,
    where the drift between consecutive windows starts to exceed the
    local scatter.

    In a dense dataset no window reaches that: the nearest cell of the
    next window stays about one spacing away however wide the window is.
    The raw field is then noise-dominated whatever you choose, and the
    function takes the eligible window with the largest typical move,
    reporting that the target was not met. Treat a low raw ICCoh on such
    a dataset as a property of the data, not of the estimator.

    Choosing the window this way keeps the parameter out of the metrics
    it is later judged by.

    Returns
    -------
    (window_size, table) - the choice and the full diagnostics frame.
    The table carries ``eligible`` and ``chosen`` columns and the
    attribute ``table.attrs["target_met"]``.
    """
    table = window_diagnostics(adata, basis=basis, window_sizes=candidates,
                               verbose=False, **window_kwargs)
    eligible = ((table["dtau"] <= max_dtau)
                & (table["no_source"] <= max_no_source))
    table["eligible"] = eligible
    ok = table[eligible & (table["move2_median"] >= target_move)]
    target_met = bool(len(ok))
    if target_met:
        choice = int(ok.index[0])
        why = (f"smallest eligible window with move2 >= {target_move} "
               f"(dtau <= {max_dtau}, no_source <= {max_no_source})")
    elif eligible.any():
        choice = int(table[eligible]["move2_median"].idxmax())
        why = ("no eligible window reaches move2 >= "
               f"{target_move}: the data are dense and the raw field is "
               "noise-dominated at every window size; took the eligible "
               "window with the largest typical move")
    else:
        choice = int(table.index[0])
        why = "no eligible window at all; took the smallest candidate"
    table["chosen"] = table.index == choice
    table.attrs["target_met"] = target_met
    if verbose:
        print(table.round(3).to_string())
        print(f"  chosen window_size = {choice}\n    {why}")
    return choice, table


# =====================================================================
# 2. OPTIMAL TRANSPORT VELOCITY
# =====================================================================


# On a GPU the convergence test forces a synchronisation, so it is only
# made every few iterations; the loop may overshoot by that many.
_SINKHORN_CHECK_EVERY = 10
# Below this many entries the cost matrix is too small for a GPU to pay
# for its launch overhead, and the numpy path is used instead.
_SINKHORN_GPU_MIN_SIZE = 10_000


def _lse(M, axis):
    """log-sum-exp with the maximum shifted out.

    ``scipy.special.logsumexp`` computes the same thing, but carries
    enough extra machinery (weights, sign handling, input checking) to
    dominate the cost on the small matrices a window pair produces: it
    is ~4x slower at 50x50 and ~3x at 300x300, for differences of 1e-15.
    An all -inf slice (a disconnected pair under cost_metric="geodesic")
    gives a non-finite maximum, so the shift falls back to zero there
    and the slice correctly returns -inf.
    """
    m = M.max(axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return (m + np.log(np.exp(M - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def _log_sinkhorn_unbalanced(a, b, C, reg, reg_m, n_iter=5000, tol=1e-6):
    """Log-domain Sinkhorn for entropic unbalanced OT with KL marginal
    penalties (Chizat et al., 2018), stable for any cost range:

        min_P <P, C> + reg KL(P | ab) + rho_a KL(P1 | a) + rho_b KL(P^T 1 | b)

    ``reg_m`` is rho (same for both marginals) or (rho_a, rho_b); an
    infinite value keeps that marginal exact.

    Both backends run the identical recursion in float64 and agree to
    ~1e-14. The work is O(n^2) per iteration, so the GPU is used only
    for windows large enough to pay for the transfer; on CPU torch is
    no faster than numpy, so numpy is used there.

    ``tol`` is on the dual potential f, in units of ``reg``. The solver
    is run to a direction, not to machine precision: against a 1e-9
    solve, 1e-6 leaves the resulting velocity unchanged to 10 decimal
    places. Pass a smaller value to recover the old behaviour exactly.
    """
    rho_a, rho_b = (reg_m, reg_m) if np.ndim(reg_m) == 0 else reg_m
    ka = 1.0 if np.isinf(rho_a) else rho_a / (rho_a + reg)
    kb = 1.0 if np.isinf(rho_b) else rho_b / (rho_b + reg)
    thr = tol * reg

    if (_HAS_TORCH and DEVICE is not None and DEVICE.type == "cuda"
            and np.asarray(C).size >= _SINKHORN_GPU_MIN_SIZE):
        opt = dict(dtype=torch.float64, device=DEVICE)
        Ct = torch.as_tensor(np.asarray(C), **opt)
        la = torch.as_tensor(np.log(a), **opt)
        lb = torch.as_tensor(np.log(b), **opt)
        f = torch.zeros_like(la)
        g = torch.zeros_like(lb)
        for i in range(n_iter):
            f_old = f
            f = -ka * reg * torch.logsumexp(
                (g[None, :] - Ct) / reg + lb[None, :], dim=1)
            g = -kb * reg * torch.logsumexp(
                (f[:, None] - Ct) / reg + la[:, None], dim=0)
            if (i % _SINKHORN_CHECK_EVERY == 0
                    and torch.max(torch.abs(f - f_old)).item() < thr):
                break
        P = torch.exp((f[:, None] + g[None, :] - Ct) / reg
                      + la[:, None] + lb[None, :])
        return P.cpu().numpy()

    la, lb = np.log(a), np.log(b)
    f = np.zeros_like(a)
    g = np.zeros_like(b)
    for _ in range(n_iter):
        f_old = f
        f = -ka * reg * _lse((g[None, :] - C) / reg + lb[None, :], 1)
        g = -kb * reg * _lse((f[:, None] - C) / reg + la[:, None], 0)
        if np.max(np.abs(f - f_old)) < thr:
            break
    return np.exp((f[:, None] + g[None, :] - C) / reg + la[:, None] + lb[None, :])


def _graph_weights(X: np.ndarray, knn_adj):
    """Weighted kNN graph: every edge of the connectivity graph carries
    the Euclidean distance between its endpoints. Used by
    ``cost_metric="geodesic"``."""
    from scipy import sparse
    A = sparse.csr_matrix(knn_adj)
    A = ((A + A.T) > 0).tocoo()
    d = np.linalg.norm(X[A.row] - X[A.col], axis=1)
    return sparse.csr_matrix((d, (A.row, A.col)), shape=A.shape)


def _geodesic_distances(W, idx_source):
    """Shortest-path distances from each source cell to every cell along
    the kNN graph. Disconnected pairs come back as inf."""
    from scipy.sparse.csgraph import dijkstra
    return dijkstra(W, directed=False, indices=idx_source)


def _pair_cell_moves(X, idx_source, idx_target, cost_metric="euclidean",
                     graph_weights=None, knn_adj=None) -> np.ndarray:
    """Per source cell, the squared distance to its cheapest target, in
    units of the squared nearest-neighbour spacing of the source window.
    This is the scale ``reg_m`` is measured in."""
    X1 = X[idx_source]
    if cost_metric == "geodesic":
        W = graph_weights if graph_weights is not None \
            else _graph_weights(X, knn_adj)
        D = _geodesic_distances(W, idx_source) ** 2
        finite = np.isfinite(D)
        D[~finite] = 10.0 * D[finite].max() if finite.any() else 1.0
        C = D[:, idx_target]
        Dss = D[:, idx_source]
        np.fill_diagonal(Dss, np.inf)
        delta2 = float(np.median(Dss.min(axis=1))) if len(idx_source) > 1 else 1.0
    else:
        C = pot.dist(X1, X[idx_target], metric="sqeuclidean")
        delta2 = _pair_nn_scale(X1, fallback=1.0)
    if delta2 <= 0 or not np.isfinite(delta2):
        return np.full(len(idx_source), np.nan)
    return C.min(axis=1) / delta2


def _pair_typical_move(X, idx_source, idx_target, **kw) -> float:
    """Median over source cells of :func:`_pair_cell_moves`."""
    return float(np.median(_pair_cell_moves(X, idx_source, idx_target, **kw)))


def _pair_nn_scale(X1: np.ndarray, fallback: float) -> float:
    """Median squared distance from each source cell to its nearest other
    source cell. Used by ``cost_scale="nn"``."""
    if X1.shape[0] < 2:
        return fallback
    D = pot.dist(X1, X1, metric="sqeuclidean")
    np.fill_diagonal(D, np.inf)
    nn = D.min(axis=1)
    delta2 = float(np.median(nn))
    if not np.isfinite(delta2) or delta2 <= 0:
        pos = nn[np.isfinite(nn) & (nn > 0)]
        delta2 = float(pos.min()) if pos.size else fallback
    return delta2


def _ot_velocity_pair(
    X: np.ndarray,
    idx_source: np.ndarray,
    idx_target: np.ndarray,
    pseudotime: np.ndarray,
    knn_adj,
    reg: float = 0.05,
    lambda_time: float = 1.0,
    lambda_knn: float = 1.0,
    unbalanced: bool = False,
    reg_m: float = 1.0,
    mask_self: bool = True,
    use_graph: bool = True,
    cost_metric: str = "euclidean",
    cost_scale="max",
    solver: str = "sinkhorn",
    assignment: str = "barycentric",
    graph_weights=None,
    return_self_mass: bool = False,
    return_info: bool = False,
):
    """
    Compute OT-based velocity for one window pair.

    Returns velocity vectors for cells in the source window.

    The cost between source cell i and target cell j is

        C_ij = d2_ij / s
               + (d2_max / s) * ( lambda_time * [tau_i > tau_j]
                                  + lambda_knn * [j not a kNN neighbour of i] )

    where d2 is the squared Euclidean distance and d2_max its maximum over
    the pair. The penalties are always expressed relative to the largest
    geometric cost, so ``lambda = 1`` means "as costly as the farthest
    pair" whatever ``cost_scale`` is. With the default ``cost_scale="max"``
    (s = d2_max) this is exactly the original VelOT cost.

    Parameters
    ----------
    unbalanced
        If True, solve the unbalanced entropic OT problem
        (``ot.unbalanced.sinkhorn_unbalanced``) instead of the balanced
        one. Unbalanced OT relaxes the requirement that all source mass
        be transported and all target mass be received, which is the
        appropriate model when the two windows differ in size because of
        proliferation, cell death, or uneven lineage sampling rather
        than because of pure displacement.
    reg_m
        Marginal relaxation strength for the unbalanced problem, in units
        of the squared nearest-neighbour spacing INSIDE the source window
        (``delta2``): ``reg_m=25`` means "giving up a unit of mass costs
        about as much as moving 5 spacings". Mass is then kept wherever an
        ordinary target is within that reach and dropped where the nearest
        target is much further - a terminal state. The unit is local to the
        source window, so it does not depend on ``cost_scale`` or
        ``cost_metric``, and it does not rescale itself away in a pair
        whose targets are all far. To choose it, look at the per-pair
        ``typical_move`` diagnostic (the median cheapest move, in the same
        spacing units), printed by :func:`compute_ot_velocity`: pick a
        ``reg_m`` above the usual value and below the dead-end ones.
        A tuple ``(reg_m_source, reg_m_target)`` relaxes the two marginals
        separately (``float("inf")`` keeps one exact).
    mask_self
        If True, forbid transport from a cell to itself. This matters
        only when source and target windows overlap: an identical pair
        has zero displacement and incurs no backward-pseudotime
        penalty, so it is maximally favorable under the OT objective
        and would bias the expected displacement toward zero.
    cost_metric
        How the geometric cost d2 is measured.

        ``"euclidean"`` (default)
            Squared straight-line distance, as in the original estimator.
        ``"geodesic"``
            Squared shortest-path distance along the kNN graph of ALL
            cells. Reaching a cell on another branch then costs the whole
            path back through the branch point, which makes such transport
            expensive without forbidding it: unlike ``lambda_knn`` it
            leaves far targets along the same branch reachable, so the
            estimator does not collapse onto each cell's neighbours.
            Needs ``knn_adj`` (or ``graph_weights``). Pairs with no path
            get 10x the largest finite distance.
    use_graph
        If True (default), the kNN graph ``knn_adj`` is used in two ways:
        (1) ``lambda_knn`` is added to the cost of every target that is
        not a graph neighbour of the source, and (2) a source cell with no
        graph neighbour in the target window gets zero velocity. Setting
        ``lambda_knn=0`` removes (1) but not (2); ``use_graph=False``
        removes both.
    cost_scale
        What the squared distances are divided by, which sets the meaning
        of ``reg``.

        ``"max"`` (default)
            s = d2_max of the pair. ``reg`` is relative to the largest
            squared distance, so the same ``reg`` gives a blurrier plan
            the further apart the two windows are.
        ``"nn"``
            s = median squared distance from each source cell to its
            nearest other source cell. ``reg`` is then in units of the
            within-window spacing (effective temperature T_eff = reg),
            independent of how far the target window is displaced.
        ``"none"``
            s = 1 (raw squared distances).
        float
            s = that value.
    solver
        ``"sinkhorn"`` (default): entropic OT. With ``cost_scale="max"``
        it uses exactly the original call (``ot.sinkhorn``, 500
        iterations). With any other scale it uses the log-domain solver,
        which stays stable when ``reg`` is small relative to the cost
        range. ``"emd"``: exact unregularised OT (``reg`` is ignored);
        with equal window sizes the plan is a permutation. Not available
        with ``unbalanced=True``.
    graph_weights
        Optional precomputed weighted graph from ``_graph_weights``
        (``cost_metric="geodesic"`` only), so it is not rebuilt per pair.
    assignment
        ``"barycentric"`` (default): v_i = sum_j P_ij (x_j - x_i) / sum_j P_ij.
        ``"argmax"``: v_i = x_j* - x_i with j* = argmax_j P_ij, the single
        target receiving most of cell i's mass.
    return_self_mass
        If True, also return the fraction of transported mass that was
        assigned to identical source/target pairs (computed before
        masking). Used as a diagnostic. Ignored if ``return_info``.
    return_info
        If True, return ``(V, info)`` where ``info`` holds
        ``self_mass``, ``T_eff`` (reg * s / nn-spacing; NaN for EMD),
        ``n_eff`` (mean over source cells of 1 / sum_j w_ij^2, the
        effective number of targets per cell), ``marginal_err`` (balanced
        OT only: largest relative deviation of the plan's column sums from
        1/n_target; a large value means Sinkhorn did not reach the
        balanced solution, so the plan behaves like unbalanced OT),
        ``row_mass`` (per source cell, fraction of its mass that was
        transported; 1 for balanced OT, below 1 where unbalanced OT
        destroys mass), ``typical_move`` (median cheapest move, in units
        of the squared spacing inside the source window - the scale
        ``reg_m`` is measured in), ``scale`` and ``delta2``.
    """
    if solver not in ("sinkhorn", "emd"):
        raise ValueError(f"solver must be 'sinkhorn' or 'emd', got {solver!r}")
    if assignment not in ("barycentric", "argmax"):
        raise ValueError(
            f"assignment must be 'barycentric' or 'argmax', got {assignment!r}")
    if solver == "emd" and unbalanced:
        raise ValueError("solver='emd' is only available for balanced OT")
    if cost_metric not in ("euclidean", "geodesic"):
        raise ValueError("cost_metric must be 'euclidean' or 'geodesic', "
                         f"got {cost_metric!r}")
    if cost_metric == "geodesic" and knn_adj is None and graph_weights is None:
        raise ValueError("cost_metric='geodesic' needs the kNN graph")

    X1 = X[idx_source]
    X2 = X[idx_target]

    n1 = X1.shape[0]
    n2 = X2.shape[0]

    # Uniform marginals
    a = np.ones(n1, dtype=np.float64) / n1
    b = np.ones(n2, dtype=np.float64) / n2

    # Geometric cost: squared distance, divided by the scale
    Dss = None
    if cost_metric == "geodesic":
        W = graph_weights if graph_weights is not None \
            else _graph_weights(X, knn_adj)
        D = _geodesic_distances(W, idx_source) ** 2      # (n1, n_cells)
        finite = np.isfinite(D)
        D[~finite] = 10.0 * D[finite].max() if finite.any() else 1.0
        C = D[:, idx_target].copy()
        Dss = D[:, idx_source]
        np.fill_diagonal(Dss, np.inf)
    else:
        C = pot.dist(X1, X2, metric="sqeuclidean")
    C_max = C.max()
    need_delta = return_info or cost_scale == "nn" or unbalanced
    if not need_delta:
        delta2 = np.nan
    elif Dss is not None:
        delta2 = float(np.median(Dss.min(axis=1))) if n1 > 1 \
            else (C_max if C_max > 0 else 1.0)
    else:
        delta2 = _pair_nn_scale(X1, fallback=C_max if C_max > 0 else 1.0)

    if cost_scale == "max":
        scale = C_max if C_max > 0 else 1.0
        if C_max > 0:
            C = C / C_max
    else:
        if cost_scale == "nn":
            scale = delta2
        elif cost_scale == "none":
            scale = 1.0
        else:
            scale = float(cost_scale)
            if scale <= 0:
                raise ValueError("a numeric cost_scale must be positive")
        C = C / scale
    # Penalty unit: the largest geometric cost in scaled units (1 for "max")
    pen = (C_max / scale) if C_max > 0 else 1.0
    # A typical forward move of this pair, in spacing units: small in an
    # ordinary pair, large when the only targets are on another branch.
    typical_move = (float(np.median(C.min(axis=1))) * scale / delta2
                    if need_delta and delta2 > 0 and n1 > 0 and n2 > 0
                    else float("nan"))
    # Unit for the marginal relaxation: the spacing inside the source window
    unit_m = (delta2 / scale) if (need_delta and delta2 > 0) else pen

    # Pseudotime penalty: penalize backward transport
    t1 = pseudotime[idx_source]
    t2 = pseudotime[idx_target]
    time_diff = t1[:, None] - t2[None, :]
    C[time_diff > 0] += lambda_time * pen

    # KNN locality penalty: penalize transport to non-neighbors
    local_adj = None
    if use_graph and knn_adj is not None:
        local_adj = knn_adj[idx_source][:, idx_target]
        if hasattr(local_adj, "toarray"):
            local_adj = local_adj.toarray()
        C[local_adj == 0] += lambda_knn * pen

    # Identical-cell pairs: only possible when windows overlap.
    self_pairs = idx_source[:, None] == idx_target[None, :]
    has_self = bool(self_pairs.any())
    if mask_self and has_self:
        # Large finite penalty rather than inf: keeps exp(-C/reg) at
        # exactly 0 for these entries without producing NaNs.
        C = C.copy()
        C[self_pairs] += 1e3 * pen

    def _fail():
        V0 = np.zeros_like(X1)
        if return_info:
            return V0, dict(self_mass=float("nan"), T_eff=float("nan"),
                            n_eff=float("nan"), marginal_err=float("nan"),
                            row_mass=np.zeros(n1),
                            typical_move=float("nan"), scale=scale,
                            delta2=delta2)
        if return_self_mass:
            return V0, float("nan")
        return V0

    # Marginal relaxation, in units of a typical forward move
    if np.ndim(reg_m) == 0:
        reg_m_eff = reg_m * unit_m
    else:
        reg_m_eff = tuple(r * unit_m for r in reg_m)

    # Solve
    original_path = (solver == "sinkhorn" and cost_scale == "max")
    try:
        if solver == "emd":
            P = pot.emd(a, b, C, numItermax=1_000_000)
        elif original_path:
            if unbalanced:
                P = pot.unbalanced.sinkhorn_unbalanced(
                    a, b, C, reg=reg, reg_m=reg_m_eff,
                    numItermax=500, stopThr=1e-6,
                )
            else:
                P = pot.sinkhorn(a, b, C, reg=reg, numItermax=500,
                                 stopThr=1e-6)
        else:
            if unbalanced:
                P = _log_sinkhorn_unbalanced(a, b, C, reg, reg_m_eff)
            else:
                P = pot.sinkhorn(a, b, C, reg=reg, method="sinkhorn_log",
                                 numItermax=2000, stopThr=1e-9)
    except Exception:
        # Fallback to exact OT if Sinkhorn diverges
        try:
            P = pot.emd(a, b, C)
        except Exception:
            return _fail()

    # Check for numerical issues. Under unbalanced OT a (near) empty plan
    # is a legitimate answer - every target was too expensive - so only the
    # balanced problem treats it as a failure.
    if not np.isfinite(P).all() or (P.sum() < 1e-10 and not unbalanced):
        return _fail()

    self_mass = float("nan")
    if has_self:
        total = P.sum()
        self_mass = float(P[self_pairs].sum() / total) if total > 0 else 0.0
    else:
        self_mass = 0.0

    # Velocity from the plan. A source row that carries (numerically) no
    # mass has no estimate: its velocity is set to 0, not to -x_i.
    row_mass = P.sum(axis=1)
    empty = row_mass <= 1e-12 * a
    row_sums = row_mass[:, None].copy()
    row_sums[empty] = 1.0
    if assignment == "argmax":
        j_star = np.argmax(P, axis=1)
        V = X2[j_star] - X1
        W = np.zeros_like(P)
        W[np.arange(n1), j_star] = 1.0
    else:
        W = P / row_sums
        V = W @ X2 - X1
    V[empty] = 0.0
    W[empty] = 0.0

    # Zero out velocity for cells with no KNN neighbors in target
    if local_adj is not None:
        has_neighbors = local_adj.sum(axis=1) > 0
        V[~has_neighbors] = 0.0
        W[~has_neighbors] = 0.0

    if return_info:
        wsq = (W ** 2).sum(axis=1)
        ok = wsq > 0
        info = dict(
            self_mass=self_mass,
            T_eff=(float(reg * scale / delta2)
                   if solver == "sinkhorn" and delta2 > 0 else float("nan")),
            n_eff=float(np.mean(1.0 / wsq[ok])) if ok.any() else float("nan"),
            marginal_err=(float(np.abs(P.sum(axis=0) - b).max() / b[0])
                          if not unbalanced else float("nan")),
            row_mass=row_mass / a,
            typical_move=typical_move,
            scale=float(scale),
            delta2=float(delta2),
        )
        return V, info
    if return_self_mass:
        return V, self_mass
    return V


def compute_ot_velocity(
    adata: AnnData,
    basis: str = "X_pca",
    reg: float = 0.05,
    lambda_time: float = 1.0,
    lambda_knn: float = 1.0,
    unbalanced: bool = False,
    reg_m: float = 1.0,
    mask_self: bool = True,
    use_graph: bool = True,
    cost_metric: str = "euclidean",
    cost_scale="max",
    solver: str = "sinkhorn",
    assignment: str = "barycentric",
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
    unbalanced, reg_m, mask_self, use_graph, cost_metric, cost_scale,
    solver, assignment
        See :func:`_ot_velocity_pair`. The defaults reproduce the
        original VelOT estimator exactly. In short:
        ``use_graph=False`` drops both the kNN penalty and the zeroing
        of cells with no graph neighbour in the target window;
        ``cost_metric="geodesic"`` measures distances along the kNN graph
        instead of straight through empty space;
        ``reg_m="auto"`` sets the give-up threshold from the windows
        themselves (ten times the ordinary move, i.e. the 75th percentile
        of the per-cell cheapest moves, floored at one spacing), which
        keeps ordinary transitions and drops only the dead ends;
        ``cost_scale="nn"`` makes ``reg`` a temperature in units of the
        within-window spacing; ``solver="emd"`` solves exact OT;
        ``assignment="argmax"`` uses only the highest-mass target.

    Returns
    -------
    adata, modified in place with:
      - ``adata.obsm['velot_velocity_raw_pca']`` : raw OT velocity
      - ``adata.obs['velot_confidence']`` : contribution count per cell
      - ``adata.obs['velot_transported_mass']`` : fraction of each cell's
        mass transported (1 for balanced OT)
      - ``adata.uns['velot_raw_velocity_params']`` : settings plus
        per-pair diagnostics averaged over pairs (``mean_T_eff``,
        ``mean_n_eff``, ``mean_self_transport_mass``)
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
    mass = np.zeros(n_cells, dtype=np.float64)

    graph_weights = None
    if cost_metric == "geodesic":
        if knn_adj is None:
            raise ValueError(
                "cost_metric='geodesic' needs adata.obsp['connectivities']")
        graph_weights = _graph_weights(X, knn_adj)

    # reg_m="auto": put the give-up threshold above ordinary moves but
    # below the dead ends, from the geometry of the windows themselves.
    if unbalanced and isinstance(reg_m, str):
        if reg_m != "auto":
            raise ValueError(f"reg_m must be a number, a pair, or 'auto', "
                             f"got {reg_m!r}")
        moves = np.concatenate([
            _pair_cell_moves(X, np.asarray(s_), np.asarray(t_),
                             cost_metric=cost_metric,
                             graph_weights=graph_weights, knn_adj=knn_adj)
            for s_, t_ in window_pairs]) if window_pairs else np.array([])
        moves = moves[np.isfinite(moves)]
        # Ordinary transport scale: the bulk of the cells, floored at one
        # spacing (a cheapest move below the local spacing is not a cost
        # worth relaxing). Ten times that keeps ordinary cells at ~90% of
        # their mass while a dead end, which is orders of magnitude more
        # expensive, keeps almost none.
        ordinary = max(float(np.percentile(moves, 75)), 1.0) if moves.size else 1.0
        reg_m = 10.0 * ordinary
        print(f"  reg_m='auto' -> {reg_m:.3g}  (10x the ordinary move "
              f"{ordinary:.3g}; p50/p90/max of the per-cell moves = "
              f"{np.median(moves):.3g}/{np.percentile(moves, 90):.3g}/"
              f"{moves.max():.3g}, in spacing units)")

    self_mass_log = []
    t_eff_log = []
    n_eff_log = []
    marg_log = []
    move_log = []

    for pair_i, (idx_src, idx_tgt) in enumerate(window_pairs):
        v_local, info = _ot_velocity_pair(
            X, idx_src, idx_tgt, pseudotime, knn_adj,
            reg=reg, lambda_time=lambda_time, lambda_knn=lambda_knn,
            unbalanced=unbalanced, reg_m=reg_m, mask_self=mask_self,
            use_graph=use_graph, cost_metric=cost_metric,
            cost_scale=cost_scale, solver=solver, assignment=assignment,
            graph_weights=graph_weights, return_info=True,
        )
        self_mass = info["self_mass"]
        if np.isfinite(self_mass):
            self_mass_log.append(self_mass)
        if np.isfinite(info["T_eff"]):
            t_eff_log.append(info["T_eff"])
        if np.isfinite(info["n_eff"]):
            n_eff_log.append(info["n_eff"])
        if np.isfinite(info["marginal_err"]):
            marg_log.append(info["marginal_err"])
        if np.isfinite(info["typical_move"]):
            move_log.append(info["typical_move"])

        for i, cell_idx in enumerate(idx_src):
            V[cell_idx] += v_local[i]
            counts[cell_idx] += 1
            mass[cell_idx] += info["row_mass"][i]

    # Average velocity across contributing windows
    mask = counts > 0
    V[mask] /= counts[mask, None]
    mass[mask] /= counts[mask]

    # Cells that actually carry a vector. A cell can be a source in some
    # window and still end up at zero, because the kNN-graph rule blanks
    # sources with no graph neighbour in the target window, so counting
    # sources alone would overstate the coverage.
    nonzero = np.linalg.norm(V, axis=1) > 0

    # Confidence: normalized contribution count
    max_count = counts.max() if counts.max() > 0 else 1.0
    confidence = counts / max_count

    adata.obsm[f"velot_velocity_raw_{_basis_suffix(adata, basis)}"] = V

    adata.obs["velot_confidence"] = confidence
    # Fraction of each cell's mass that was transported (1 for balanced
    # OT; low values under unbalanced OT flag cells with nowhere to go,
    # e.g. terminal states). NaN for cells that were never a source.
    adata.obs["velot_transported_mass"] = np.where(mask, mass, np.nan)

    adata.uns["velot_raw_velocity_params"] = {
        "estimator": "ot",
        "reg": reg,
        "lambda_time": lambda_time,
        "lambda_knn": lambda_knn,
        "unbalanced": unbalanced,
        "reg_m": reg_m if unbalanced else None,
        "mask_self": mask_self,
        "use_graph": use_graph,
        "cost_metric": cost_metric,
        "cost_scale": cost_scale,
        "solver": solver,
        "assignment": assignment,
        "mean_self_transport_mass": (
            float(np.mean(self_mass_log)) if self_mass_log else 0.0
        ),
        "mean_T_eff": float(np.mean(t_eff_log)) if t_eff_log else None,
        "mean_n_eff": float(np.mean(n_eff_log)) if n_eff_log else None,
        "max_marginal_err": float(np.max(marg_log)) if marg_log else None,
        "typical_move_median": float(np.median(move_log)) if move_log else None,
        "typical_move_max": float(np.max(move_log)) if move_log else None,
        "typical_move_per_pair": np.asarray(move_log, dtype=float),
        "coverage": float(nonzero.mean()),
    }

    n_with_velocity = int(nonzero.sum())
    n_never_source = int((~mask).sum())
    n_blanked = int((mask & ~nonzero).sum())
    print(f"  OT velocity computed: {n_with_velocity} cells with velocity, "
          f"{n_cells - n_with_velocity} without "
          f"({n_never_source} never a source"
          + (f", {n_blanked} blanked by the kNN-graph rule" if n_blanked else "")
          + ") (will be filled by smoothing)")
    if self_mass_log and max(self_mass_log) > 0:
        print(f"  Mean transport mass on identical source/target pairs: "
              f"{np.mean(self_mass_log):.4f}")
    if t_eff_log or n_eff_log:
        msg = "  Plan sharpness:"
        if t_eff_log:
            msg += f" mean T_eff = {np.mean(t_eff_log):.3g}"
        if n_eff_log:
            msg += f"{',' if t_eff_log else ''} mean targets per cell (n_eff) = {np.mean(n_eff_log):.3g}"
        print(msg)
    if move_log:
        print(f"  Typical move per pair (spacing units, the unit of reg_m): "
              f"median {np.median(move_log):.3g}, max {np.max(move_log):.3g}")
    if marg_log and max(marg_log) > 0.05:
        n_bad = int(np.sum(np.asarray(marg_log) > 0.05))
        warnings.warn(
            f"Sinkhorn did not reach the balanced plan in {n_bad}/"
            f"{len(marg_log)} window pairs (max column-marginal error "
            f"{max(marg_log):.0%}). Those plans behave like unbalanced OT. "
            "Increase reg, or use solver='emd' or unbalanced=True "
            "explicitly.", RuntimeWarning)
        print(f"  WARNING: balanced marginals not reached in {n_bad}/"
              f"{len(marg_log)} pairs (max error {max(marg_log):.0%})")

    return adata


# =====================================================================
# 2b. PSEUDOTIME-GRADIENT VELOCITY (NO-OT CONTROL BASELINE)
# =====================================================================


def gradient_velocity(
    adata: AnnData,
    basis: str = "X_pca",
    mode: str = "knn",
    k: int = 30,
    weighting: str = "uniform",
    temperature: float = 0.1,
    min_pseudotime_gap: float = 0.0,
    use_windows_from_uns: bool = True,
) -> AnnData:
    """
    Estimate a raw velocity field as a local pseudotime gradient, without
    solving any optimal transport problem.

    This is the transport-free control for the VelOT pipeline. It writes
    exactly the same keys as :func:`compute_ot_velocity`, so it can be
    swapped in underneath an otherwise identical neural smoothing and
    projection stage. Any difference in the final field is then
    attributable to the transport step alone.

    Two modes are available, answering two different questions.

    ``mode="knn"`` (default)
        For each cell, the raw velocity is the (weighted) mean
        displacement toward those of its ``k`` nearest neighbors that
        have a larger pseudotime. No windows, no clusters, no coupling.
        This is the minimal "smoothed pseudotime gradient" baseline: it
        asks whether the OT machinery contributes anything beyond
        pseudotime plus neural smoothing.

    ``mode="window"``
        Reuses the window pairs already built by
        :func:`build_windows`, but replaces the Sinkhorn coupling with
        the uniform coupling: each source cell's velocity is the
        displacement toward the unweighted mean of the target window.
        Spatial clustering, temporal windowing, and the aggregation over
        overlapping windows are all held fixed, so the contrast against
        :func:`compute_ot_velocity` isolates the transport plan itself.

    Parameters
    ----------
    adata
        Must contain ``adata.obsm[basis]`` and ``adata.obs['pseudotime']``.
        For ``mode="window"`` it must also contain windows from
        :func:`build_windows`.
    basis
        Embedding key in ``adata.obsm``.
    mode
        ``"knn"`` or ``"window"``.
    k
        Number of nearest neighbors (``mode="knn"`` only). Set this to
        the same value used for the smoothing kNN so the two stages see
        the same neighborhood scale.
    weighting
        ``"uniform"`` weights all forward neighbors equally.
        ``"softmax"`` weights neighbor ``j`` by
        ``exp((tau_j - tau_i) / temperature)``, giving more influence to
        neighbors further ahead in pseudotime.
    temperature
        Softmax temperature (``weighting="softmax"`` only).
    min_pseudotime_gap
        Only neighbors with ``tau_j - tau_i > min_pseudotime_gap``
        contribute. The default of 0 uses every strictly-forward
        neighbor.
    use_windows_from_uns
        ``mode="window"`` only; kept for symmetry with
        :func:`compute_ot_velocity`.

    Returns
    -------
    adata, modified in place with the same keys as
    :func:`compute_ot_velocity`:
      - ``adata.obsm['velot_velocity_raw_pca']`` (or ``..._umap``)
      - ``adata.obs['velot_confidence']``
      - ``adata.uns['velot_raw_velocity_params']``

    Example
    -------
    ::

        velot.pp.prepare(adata, root_cluster="Ngn3 low EP")
        velot.tl.velocity(adata, method="gradient")   # control
        velot.tl.velocity(adata, method="ot")         # VelOT
    """
    if mode not in ("knn", "window"):
        raise ValueError(f"mode must be 'knn' or 'window', got {mode!r}")
    if weighting not in ("uniform", "softmax"):
        raise ValueError(
            f"weighting must be 'uniform' or 'softmax', got {weighting!r}"
        )

    _check_fields(adata, obsm_keys=[basis], obs_keys=["pseudotime"])

    X = np.asarray(adata.obsm[basis], dtype=np.float64)
    pseudotime = adata.obs["pseudotime"].values.astype(np.float64)
    n_cells, dim = X.shape

    V = np.zeros((n_cells, dim), dtype=np.float64)
    counts = np.zeros(n_cells, dtype=np.float64)

    if mode == "knn":
        k_eff = int(min(k, max(1, n_cells - 1)))
        knn_indices = _build_knn_index(X, k=k_eff)

        for i in range(n_cells):
            nbrs = knn_indices[i]
            gaps = pseudotime[nbrs] - pseudotime[i]
            forward = gaps > min_pseudotime_gap

            n_forward = int(forward.sum())
            if n_forward == 0:
                continue

            nbrs_f = nbrs[forward]
            disp = X[nbrs_f] - X[i]

            if weighting == "softmax":
                g = gaps[forward]
                w = np.exp((g - g.max()) / max(temperature, 1e-12))
                w = w / w.sum()
                V[i] = (w[:, None] * disp).sum(axis=0)
            else:
                V[i] = disp.mean(axis=0)

            # Confidence: how much of the local neighborhood is
            # forward in pseudotime. Plays the same role as the OT
            # confidence, i.e. how well supported this estimate is.
            counts[i] = n_forward

        max_count = counts.max() if counts.max() > 0 else 1.0
        confidence = counts / max_count

    else:  # mode == "window"
        _check_fields(adata, uns_keys=["velot_windows"])
        window_pairs = adata.uns["velot_windows"]["pairs"]

        knn_adj = adata.obsp["connectivities"] if "connectivities" in adata.obsp else None

        for idx_src, idx_tgt in window_pairs:
            idx_src = np.asarray(idx_src)
            idx_tgt = np.asarray(idx_tgt)

            # Uniform coupling: every source cell moves toward the
            # unweighted barycenter of the target window. Exclude
            # identical cells, mirroring mask_self in the OT path.
            target_mean = X[idx_tgt].mean(axis=0)
            v_local = target_mean[None, :] - X[idx_src]

            overlap = np.isin(idx_src, idx_tgt)
            if overlap.any():
                n_tgt = len(idx_tgt)
                if n_tgt > 1:
                    sum_tgt = X[idx_tgt].sum(axis=0)
                    for pos in np.where(overlap)[0]:
                        cell = idx_src[pos]
                        adj_mean = (sum_tgt - X[cell]) / (n_tgt - 1)
                        v_local[pos] = adj_mean - X[cell]

            if knn_adj is not None:
                local_adj = knn_adj[idx_src][:, idx_tgt]
                if hasattr(local_adj, "toarray"):
                    local_adj = local_adj.toarray()
                has_neighbors = np.asarray(local_adj).sum(axis=1) > 0
                v_local[~has_neighbors] = 0.0

            for i, cell_idx in enumerate(idx_src):
                V[cell_idx] += v_local[i]
                counts[cell_idx] += 1

        max_count = counts.max() if counts.max() > 0 else 1.0
        confidence = counts / max_count

    mask = counts > 0
    if mode == "window":
        # Average over the windows in which each cell was a source,
        # exactly as compute_ot_velocity does.
        V[mask] /= counts[mask, None]

    adata.obsm[f"velot_velocity_raw_{_basis_suffix(adata, basis)}"] = V

    adata.obs["velot_confidence"] = confidence

    adata.uns["velot_raw_velocity_params"] = {
        "estimator": f"gradient::{mode}",
        "k": int(k) if mode == "knn" else None,
        "weighting": weighting if mode == "knn" else "uniform",
        "temperature": temperature if weighting == "softmax" else None,
        "min_pseudotime_gap": min_pseudotime_gap,
    }

    n_with_velocity = int(mask.sum())
    n_zero = int((~mask).sum())
    print(f"  Gradient velocity ({mode}) computed: {n_with_velocity} cells "
          f"with velocity, {n_zero} cells without "
          f"(will be filled by smoothing)")

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
    lambda_curl: float = 0.5,
    lambda_divergence: float = 0.0,
    k_smooth: int = 15,
    use_pseudotime: bool = True,
    normalize_raw: bool = False,
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

    if not _HAS_TORCH:
        raise ImportError(
            "PyTorch is required for velocity smoothing."
            "Install with: pip3 install torch"
        )
    else:
        print(f"Found torch compatible version. Running on {DEVICE} device")

    _check_fields(
        adata, obsm_keys=[f"X_{basis}", velocity_key],
        obs_keys=["velot_confidence"],
    )

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    X_np = adata.obsm[f"X_{basis}"].astype(np.float32)
    V_np = adata.obsm[velocity_key].astype(np.float32)
    if normalize_raw:
        # Fit direction only. The regression loss is an unweighted L2 on
        # the raw vectors, so cells whose displacement happens to be long
        # dominate the fit; scaling every target to unit length removes
        # that weighting. The smoothed field then carries the magnitudes
        # the network produces, not the window-to-window displacements.
        nrm = np.linalg.norm(V_np, axis=1, keepdims=True)
        keep = nrm[:, 0] > 0
        V_np = V_np.copy()
        V_np[keep] = V_np[keep] / nrm[keep]
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
    
def project_to_embedding(
    adata: AnnData,
    velocity_key: str = "velot_velocity_pca",
    velocity_key_umap: str = "velot_velocity_umap",
    basis_pca: str = "X_pca",
    basis_embedding: str = "X_umap",
    n_neighbors: int = 30,
) -> AnnData:
    """
    Project velocity from PCA space to any 2D embedding space for visualization.

    Uses a local linear approximation: for each cell, the Jacobian
    of the PCA→2D (UMAP, TSNE...) mapping is estimated from its KNN neighborhood
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
        obsm_keys=[velocity_key, basis_pca, basis_embedding],
    )

    if basis_embedding == "X_tsne":
        import warnings
        warnings.warn(
            "t-SNE distorts global distances. Velocity arrows show "
            "local directions correctly but arrow lengths and "
            "cross-cluster directions may be misleading. "
            "Consider using UMAP for velocity visualization.",
            UserWarning,
        )

    X_pca = adata.obsm[basis_pca]
    X_umap = adata.obsm[basis_embedding]
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

    print(f"  Velocity projected to 2D embedding {basis_embedding} ({n_cells} cells)")

    return adata


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
    # Raw velocity estimator
    method: str = "ot",
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
    unbalanced: bool = False,
    reg_m: float = 1.0,
    mask_self: bool = True,
    use_graph: bool = True,
    cost_metric: str = "euclidean",
    cost_scale="max",
    ot_solver: str = "sinkhorn",
    ot_assignment: str = "barycentric",
    # Gradient-baseline params (method="gradient")
    gradient_mode: str = "knn",
    gradient_k: int = 30,
    gradient_weighting: str = "uniform",
    gradient_temperature: float = 0.1,
    # Smoothing params
    n_epochs: int = 200,
    hidden_dim: int = 128,
    lambda_smooth: float = 0.5,
    lambda_curl: float = 0.0,
    lambda_divergence: float = 0.0,
    k_smooth: int = 15,
    use_pseudotime: bool = True,
    normalize_raw: bool = False,
    # Output
    project_umap: bool = True,
    project_basis: str = "X_umap",
    random_state: int = 42,
    verbose: bool = True,
) -> AnnData:
    """
    Run the full VelOT velocity pipeline.

    This is a convenience function that calls, in order:
      1. ``build_windows()`` — spatial-temporal windowing
      2. ``compute_ot_velocity()`` or ``gradient_velocity()`` — raw field
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
    method
        Estimator for the raw velocity field.

        ``"ot"`` (default)
            The VelOT estimator: entropy-regularized optimal transport
            between consecutive pseudotime windows within each spatial
            cluster.
        ``"gradient"``
            The transport-free control baseline
            (:func:`gradient_velocity`). Everything downstream —
            smoothing, projection, metrics — is identical, so comparing
            the two isolates the contribution of the transport step.
            See ``gradient_mode`` for the two available controls.

    unbalanced, reg_m, mask_self, use_graph, cost_metric, cost_scale
        Passed to :func:`compute_ot_velocity` (``method="ot"`` only).
    ot_solver, ot_assignment
        Passed to :func:`compute_ot_velocity` as ``solver`` and
        ``assignment`` (``method="ot"`` only). All OT options default to
        the original estimator.
    gradient_mode, gradient_k, gradient_weighting, gradient_temperature
        Passed to :func:`gradient_velocity` (``method="gradient"``
        only) as ``mode``, ``k``, ``weighting`` and ``temperature``.
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
        label = "VelOT" if method == "ot" else "VelOT (gradient control)"
        print(f"{label}: Computing velocity field")
        print(f"  Estimator: {method}"
              + (f" / {gradient_mode}" if method == "gradient" else "")
              + (" / unbalanced" if (method == "ot" and unbalanced) else "")
              + (f" / {ot_solver}, {cost_metric} cost/{cost_scale}, {ot_assignment}"
                 + ("" if use_graph else ", no kNN graph")
                 if method == "ot" and (ot_solver != "sinkhorn"
                                        or cost_metric != "euclidean"
                                        or cost_scale != "max"
                                        or ot_assignment != "barycentric"
                                        or not use_graph) else ""))
        print(f"  Basis: {basis} ({adata.obsm[basis].shape[1]}D)")
        print(f"  Cells: {adata.n_obs}")
        print(f"  Smoothing: {'ON' if smooth else 'OFF'}")
        print()

    if method not in ("ot", "gradient"):
        raise ValueError(f"method must be 'ot' or 'gradient', got {method!r}")

    needs_windows = (method == "ot") or (gradient_mode == "window")

    # Step 1: Windowing
    if needs_windows:
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
    elif verbose:
        print("[1/4] Windowing: SKIPPED (gradient_mode='knn')")

    # Step 2: raw velocity field
    if method == "ot":
        if verbose:
            print("\n[2/4] Computing OT velocity...")
        compute_ot_velocity(
            adata,
            basis=basis,
            reg=reg,
            lambda_time=lambda_time,
            lambda_knn=lambda_knn,
            unbalanced=unbalanced,
            reg_m=reg_m,
            mask_self=mask_self,
            use_graph=use_graph,
            cost_metric=cost_metric,
            cost_scale=cost_scale,
            solver=ot_solver,
            assignment=ot_assignment,
        )
    else:
        if verbose:
            print(f"\n[2/4] Computing gradient velocity "
                  f"(control baseline, mode='{gradient_mode}')...")
        gradient_velocity(
            adata,
            basis=basis,
            mode=gradient_mode,
            k=gradient_k,
            weighting=gradient_weighting,
            temperature=gradient_temperature,
        )

    # Step 3: Smoothing
    if smooth:
        if verbose:
            print("\n[3/4] Smoothing velocity field...")
        v_suffix = _basis_suffix(adata, basis)
        v_key = f"velot_velocity_raw_{v_suffix}"
        smooth_velocity(
            adata,
            basis=v_suffix,
            velocity_key=v_key,
            n_epochs=n_epochs,
            hidden_dim=hidden_dim,
            lambda_smooth=lambda_smooth,
            lambda_curl=lambda_curl,
            lambda_divergence=lambda_divergence,
            k_smooth=k_smooth,
            use_pseudotime=use_pseudotime,
            normalize_raw=normalize_raw,
            random_state=random_state,
            verbose=verbose,
        )
    else:
        if verbose:
            print("\n[3/4] Smoothing: SKIPPED")

    # Step 4: Project to UMAP
    if project_umap and project_basis in adata.obsm and project_basis != basis:
        basis_name = project_basis.split("X_")[1]
        src = _basis_suffix(adata, basis)
        if verbose:
            print("\n[4/4] Projecting to UMAP...")
        project_to_umap(adata, f"velot_velocity_raw_{src}",
                        f"velot_velocity_raw_{basis_name}",
                        basis_umap=project_basis)
        if smooth:
            project_to_umap(adata, f"velot_velocity_{src}",
                            f"velot_velocity_{basis_name}",
                            basis_umap=project_basis)
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

    if not _HAS_TORCH:
        raise ImportError(
            "PyTorch is required for velocity smoothing."
            "Install with: pip3 install torch"
        )
    else:
        print(f"Found torch compatible version. Running on {DEVICE} device")

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
"""
Toy window: look inside a single VelOT transport step, cell by cell.

Run:  conda activate velot_test && python toy_window_transport.py
      (or import it from a notebook, see the example at the bottom)

WHAT THIS IS FOR
----------------
Every raw estimator in VelOT has the same form

    v_i = sum_j w_ij (x_j - x_i),

and the estimators differ only in the weights w_ij. This module builds a
tiny two-gene dataset (default 10 + 10 cells) where the answer is known,
runs the three raw estimators on it, and exposes every w_ij so you can see
where each cell's velocity comes from:

  * "ot"      Sinkhorn plan between the source and target window,
              row-normalised: w_ij = P_ij / sum_j P_ij.
              Same cost, penalties and solver call as
              velot.tl._ot_velocity_pair.
  * "window"  gradient_velocity(mode="window"): uniform coupling, every
              source cell moves to the mean of the target window,
              w_ij = 1 / n_target.
  * "knn"     gradient_velocity(mode="knn"): uniform weights over the k
              nearest neighbours (of ALL cells) that are ahead in
              pseudotime. With a small k most neighbours of a source cell
              are in its own cluster, so only a few target cells count.

Note on "knn": it is not the mean of the future cluster. That is the
"window" mode. The knn baseline uses only the forward cells that happen to
be among the k nearest neighbours of each cell, so it can have one, a few,
or zero targets per cell depending on k and the geometry.

THE TOY DATA
------------
Cluster A: n cells around `center` (gaussian, ring, grid or line).
Cluster B: exactly the same coordinates displaced by `shift`, plus
optional `target_noise`. So every cell i in A has a "twin" i + n in B, and
the true displacement of every cell is `shift`.

For a pure translation the squared-Euclidean OT map is the translation
itself (Brenier), so with small epsilon OT should send each cell to its
twin and recover v_i = shift exactly. The window gradient instead sends
every cell to the centroid of B, v_i = shift + (mean(A) - x_i), which
contracts the cloud: the spread of A leaks into the velocities. The knn
gradient lies somewhere in between depending on k. That contrast is what
the plots are meant to show.

Every decomposition is checked against the package functions
(check=True), so what you see is what the pipeline computes.

MAIN ENTRY POINTS
-----------------
    adata = make_toy_adata(...)
    res   = ot_decomposition(adata, reg=0.1)
    res   = window_decomposition(adata)
    res   = knn_decomposition(adata, k=5)
    summarize(res, adata)            # one row per source cell
    to_long(res)                     # one row per (source, target) pair
    received_mass(res)               # mass arriving at each target
    plot_cell(res, adata, cell=0)    # one cell: weights + composite + final
    plot_all_cells(res, adata)       # all sources, all couplings
    plot_weights(res, adata)         # w_ij heatmap, twins outlined
    compare(adata, ...)              # all of the above, side by side
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anndata import AnnData

import ot as pot
from velot import tl as vtl

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "toy_window")

# Colours: categorical slots 1 and 2 of the plotting palette used for the
# revision figures. Final vector in ink, true shift in grey.
C_SRC = "#2a78d6"
C_TGT = "#eb6834"
C_INK = "#1f1f1e"
C_TRUE = "#8a8a85"


# =====================================================================
# Data
# =====================================================================
def make_toy_adata(
    n_per_cluster: int = 10,
    center=(0.0, 0.0),
    shift=(1.0, 0.4),
    spread: float = 0.5,
    shape: str = "gaussian",
    target_noise: float = 0.0,
    target: str = "twin",
    pseudotime: str = "cluster",
    n_neighbors: int | None = 10,
    window_overlap: float = 0.0,
    seed: int = 0,
) -> AnnData:
    """
    Two-cluster toy dataset with 2 genes.

    Parameters
    ----------
    n_per_cluster
        Cells per cluster (total = 2 * n_per_cluster).
    center, shift
        Centre of cluster A and the displacement A -> B.
    spread
        Scale of cluster A (sd for "gaussian", radius for "ring",
        half-width for "grid" and "line").
    shape
        "gaussian", "ring", "grid" or "line". "line" lies perpendicular to
        the shift, which makes the contraction of the window gradient
        easy to see.
    target_noise
        Extra isotropic noise added to B after the shift. 0 gives exact
        twins.
    target
        "twin": B is A displaced by `shift` (each A cell has an exact
        copy in B; best case for OT).
        "resample": B is a NEW sample from the same distribution,
        displaced by `shift`. This is the realistic case: the cells of
        the next window are different cells, not the same cells moved,
        so there is no exact partner and obs["twin"] is -1. The true
        velocity is still `shift` for every cell.
    pseudotime
        "cluster": A = 0, B = 1 (no backward pairs inside the window).
        "projection": projection onto the shift direction, rescaled to
        [0, 1]. Cells of A can then be ahead of each other, and with a
        large spread some A cells can be ahead of some B cells, which
        activates the lambda_time penalty and adds forward A neighbours
        to the knn gradient.
    n_neighbors
        Build adata.obsp["connectivities"] with scanpy using this many
        neighbours (self included, as in scanpy). This graph feeds the
        lambda_knn penalty in OT and the no-neighbour zeroing. None
        skips it, which removes the kNN penalty from OT entirely.

        Watch the ratio |shift| / spread. Consecutive windows in real
        data overlap in space, so the default keeps |shift| close to the
        spread. If the shift is much larger, a cell's kNN are all in its
        own cluster, it has no neighbour in the target window, and the
        package sets its velocity to zero (both OT and window modes).
        compare() prints how many cells this hits.
    window_overlap
        Passed to velot.tl.build_windows as overlap_fraction. 0 gives
        exactly two windows (A and B) and one pair. This is membership
        overlap; spatial overlap between the clouds is set by
        |shift| / spread.
    seed
        RNG seed.

    Windows
    -------
    Windows are built by velot.tl.build_windows itself, with ALL cells in
    one spatial cluster and window_size = n_per_cluster. Sorting by
    pseudotime (A = 0, B = 1) then gives window 1 = A, window 2 = B.
    obs["cluster"] (A/B) is only a display label: if A and B were given
    to build_windows as two spatial clusters, windows would be built
    inside each cluster separately and no A -> B pair would exist.

    Returns
    -------
    AnnData with
      obsm["X_pca"]            the 2-D coordinates (basis used everywhere)
      obs["cluster"]           "A" / "B"
      obs["pseudotime"]
      obs["twin"]              index of the displaced copy (-1 for B)
      obs["velot_toy_space"]   0 for every cell (the single spatial cluster)
      uns["velot_windows"]     written by velot.tl.build_windows
      uns["toy"]               the generating parameters
    """
    rng = np.random.default_rng(seed)
    n = int(n_per_cluster)
    center = np.asarray(center, dtype=np.float64)
    shift = np.asarray(shift, dtype=np.float64)

    if shape == "gaussian":
        XA = center + rng.normal(scale=spread, size=(n, 2))
    elif shape == "ring":
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        XA = center + spread * np.c_[np.cos(ang), np.sin(ang)]
    elif shape == "grid":
        side = int(np.ceil(np.sqrt(n)))
        g = np.linspace(-spread, spread, side)
        gx, gy = np.meshgrid(g, g)
        XA = center + np.c_[gx.ravel(), gy.ravel()][:n]
    elif shape == "line":
        perp = np.array([-shift[1], shift[0]])
        perp = perp / (np.linalg.norm(perp) + 1e-12)
        s = np.linspace(-spread, spread, n)
        XA = center + s[:, None] * perp[None, :]
    else:
        raise ValueError(f"unknown shape {shape!r}")

    if target == "twin":
        XB = XA + shift
    elif target == "resample":
        if shape == "gaussian":
            XB = center + rng.normal(scale=spread, size=(n, 2)) + shift
        else:
            # deterministic shapes: resample by jittering the positions
            XB = XA + shift + rng.normal(scale=0.25 * spread, size=XA.shape)
    else:
        raise ValueError(f"unknown target {target!r}")
    if target_noise > 0:
        XB = XB + rng.normal(scale=target_noise, size=XB.shape)
    X = np.vstack([XA, XB]).astype(np.float64)

    if pseudotime == "cluster":
        tau = np.r_[np.zeros(n), np.ones(n)]
    elif pseudotime == "projection":
        u = shift / (np.linalg.norm(shift) + 1e-12)
        proj = (X - center) @ u
        tau = (proj - proj.min()) / (proj.max() - proj.min() + 1e-12)
    else:
        raise ValueError(f"unknown pseudotime {pseudotime!r}")

    obs = pd.DataFrame(
        {
            "cluster": pd.Categorical(["A"] * n + ["B"] * n),
            "pseudotime": tau,
            "twin": (np.r_[np.arange(n, 2 * n), -np.ones(n, dtype=int)]
                     if target == "twin" else -np.ones(2 * n, dtype=int)),
            "velot_toy_space": np.zeros(2 * n, dtype=int),
        },
        index=[f"A{i}" for i in range(n)] + [f"B{i}" for i in range(n)],
    )
    adata = AnnData(
        X=X.copy(), obs=obs,
        var=pd.DataFrame(index=["gene_x", "gene_y"]),
    )
    adata.obsm["X_pca"] = X.copy()

    adata.uns["toy"] = dict(
        n_per_cluster=n, center=center, shift=shift, spread=spread,
        shape=shape, target_noise=target_noise, target=target,
        pseudotime=pseudotime, n_neighbors=n_neighbors,
        window_overlap=window_overlap, seed=seed,
    )

    vtl.build_windows(
        adata, basis="X_pca", spatial_key="velot_toy_space",
        window_size=n, overlap_fraction=window_overlap,
        min_window_size=n if window_overlap == 0 else max(2, n // 2),
        tail_handling="force",
    )
    pairs = adata.uns["velot_windows"]["pairs"]
    if window_overlap == 0:
        (src, tgt), = pairs
        assert set(src) == set(range(n)) and set(tgt) == set(range(n, 2 * n))

    if n_neighbors is not None:
        set_knn_graph(adata, n_neighbors)
    return adata


def set_knn_graph(adata: AnnData, n_neighbors: int) -> AnnData:
    """(Re)build the scanpy kNN graph on X_pca. Pass a large value
    (>= n_cells) to make every pair a neighbour, i.e. switch off the
    effect of lambda_knn without removing the graph."""
    import scanpy as sc
    k = int(min(n_neighbors, adata.n_obs))
    sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=k)
    adata.uns["toy"]["n_neighbors"] = k
    return adata


# =====================================================================
# Decompositions
# =====================================================================
@dataclass
class Decomposition:
    """Everything needed to see how each source cell's velocity is built.

    weights[i, j]   w_ij, row sums to 1 (or 0 for a cell with no targets)
    mass[i, j]      transport mass. For OT this is the plan P (sums to 1
                    overall). For the gradients it is weights / n_source,
                    i.e. the implied plan if every source carries 1/n.
    contrib[i,j,:]  w_ij * (x_j - x_i): the piece of v_i coming from j
    V[i, :]         the raw velocity, sum_j contrib[i, j, :]
    """
    method: str
    idx_source: np.ndarray
    idx_target: np.ndarray
    weights: np.ndarray
    mass: np.ndarray
    contrib: np.ndarray
    V: np.ndarray
    params: dict = field(default_factory=dict)
    cost: np.ndarray | None = None
    cost_parts: dict | None = None

    @property
    def label(self) -> str:
        p = self.params
        if self.method == "ot":
            if p.get("solver") == "emd":
                s = "OT exact (EMD)"
            else:
                s = f"OT eps={p['reg']:g}"
                if p.get("cost_scale", "max") != "max":
                    s += f" [cost/{p['cost_scale']}]"
            if not p.get("has_graph"):
                s += " no graph"
            elif p.get("lambda_knn", 0):
                s += f" λknn={p['lambda_knn']:g}"
            else:
                s += " λknn=0 (zeroing on)"
            if p.get("assignment") == "argmax":
                s += " argmax"
            if p.get("unbalanced"):
                s += f" unbal reg_m={p['reg_m']:g}"
            if p.get("solver") != "emd":
                s += f"  (T_eff={p['T_eff']:.2g})"
            return s
        if self.method == "knn":
            return f"kNN gradient  k={p['k']}"
        return "Window gradient (mean of target)"


def _window(adata, pair):
    idx_s, idx_t = adata.uns["velot_windows"]["pairs"][pair]
    return np.asarray(idx_s), np.asarray(idx_t)


def _local_adj(adata, idx_s, idx_t):
    if "connectivities" not in adata.obsp:
        return None
    A = adata.obsp["connectivities"][idx_s][:, idx_t]
    return np.asarray(A.toarray() if hasattr(A, "toarray") else A)


def ot_decomposition(
    adata: AnnData,
    pair: int = 0,
    basis: str = "X_pca",
    reg: float = 0.1,
    lambda_time: float = 1.0,
    lambda_knn: float = 1.0,
    unbalanced: bool = False,
    reg_m: float = 1.0,
    mask_self: bool = True,
    use_graph: bool = True,
    cost_scale="max",
    solver: str = "sinkhorn",
    assignment: str = "barycentric",
    check: bool = True,
) -> Decomposition:
    """Transport plan for one window pair, decomposed per (source, target).

    With the defaults it rebuilds the cost exactly as
    velot.tl._ot_velocity_pair does, keeping each term separately in
    `cost_parts`, and (check=True) asserts that the velocities equal the
    package's. The four options below mirror the same arguments of
    velot.tl._ot_velocity_pair (also exposed in compute_ot_velocity and
    velocity); penalties are expressed relative to the largest geometric
    cost, as in the package, and every setting is checked against it.

    use_graph
        True (package): the kNN graph in obsp["connectivities"] is used
        twice: (1) lambda_knn is added to the cost of every target that is
        not a graph neighbour of the source, and (2) a source with NO
        graph neighbour in the target window gets v = 0. Setting
        lambda_knn = 0 removes (1) but not (2). use_graph=False removes
        both.
    cost_scale
        What the squared distances are divided by before the penalties
        are added and before eps is applied.
        "max"    d2 / max(d2)  (package). eps is then relative to the
                 LARGEST squared distance, which grows with the shift.
        "nn"     d2 / delta2, delta2 = median squared distance from each
                 source cell to its nearest other source cell. eps is then
                 in units of the within-window spacing and the plan does
                 not depend on the shift at all (see T_eff below).
        "none"   raw d2.
        float    d2 / value.
        lambda_time and lambda_knn are multiplied by max(d2) / scale, so
        lambda = 1 always means "as costly as the farthest pair".
    solver
        "sinkhorn" (package) or "emd" (exact, unregularised OT; eps is
        ignored). With equal window sizes and uniform marginals the exact
        plan is a permutation: every cell sends all its mass to one
        target.
    assignment
        "barycentric" (package): v_i = sum_j w_ij (x_j - x_i).
        "argmax": v_i = x_j* - x_i with j* = argmax_j P_ij, i.e. only the
        target that receives most of i's mass.

    params["T_eff"] = eps * scale / delta2 is the entropic temperature in
    units of the within-window spacing. T_eff << 1 gives a sharp plan
    (close to exact OT); T_eff >> 1 gives a blurred plan that tends to
    the window gradient (every cell to the target mean).
    """
    if solver not in ("sinkhorn", "emd"):
        raise ValueError(f"solver must be 'sinkhorn' or 'emd'")
    if assignment not in ("barycentric", "argmax"):
        raise ValueError(f"assignment must be 'barycentric' or 'argmax'")
    X = np.asarray(adata.obsm[basis], dtype=np.float64)
    tau = adata.obs["pseudotime"].values.astype(np.float64)
    idx_s, idx_t = _window(adata, pair)
    X1, X2 = X[idx_s], X[idx_t]
    n1, n2 = len(idx_s), len(idx_t)

    a = np.ones(n1) / n1
    b = np.ones(n2) / n2

    d2 = pot.dist(X1, X2, metric="sqeuclidean")
    d2_max = d2.max()
    dss = pot.dist(X1, X1, metric="sqeuclidean")
    np.fill_diagonal(dss, np.inf)
    delta2 = float(np.median(dss.min(axis=1))) if n1 > 1 else 1.0
    if cost_scale == "max":
        scale = d2_max if d2_max > 0 else 1.0
    elif cost_scale == "nn":
        scale = delta2
    elif cost_scale == "none":
        scale = 1.0
    else:
        scale = float(cost_scale)
    geo = d2 / scale
    pen = d2_max / scale if d2_max > 0 else 1.0   # penalties relative to max

    backward = (tau[idx_s][:, None] - tau[idx_t][None, :]) > 0
    time_pen = lambda_time * pen * backward

    local_adj = _local_adj(adata, idx_s, idx_t) if use_graph else None
    knn_pen = np.zeros_like(geo)
    if local_adj is not None:
        knn_pen = lambda_knn * pen * (local_adj == 0)

    self_pairs = idx_s[:, None] == idx_t[None, :]
    self_pen = np.zeros_like(geo)
    if mask_self and self_pairs.any():
        self_pen[self_pairs] = 1e3 * pen

    C = geo + time_pen + knn_pen + self_pen

    package_mode = (cost_scale == "max" and solver == "sinkhorn"
                    and assignment == "barycentric" and use_graph)
    if solver == "emd":
        P = pot.emd(a, b, C, numItermax=1_000_000)
    elif cost_scale == "max":
        try:
            if unbalanced:
                P = pot.unbalanced.sinkhorn_unbalanced(
                    a, b, C, reg=reg, reg_m=reg_m, numItermax=500,
                    stopThr=1e-6)
            else:
                P = pot.sinkhorn(a, b, C, reg=reg, numItermax=500,
                                 stopThr=1e-6)
        except Exception:
            P = pot.emd(a, b, C)
    else:
        # log-domain: stable for small eps relative to the cost range
        if unbalanced:
            P = pot.unbalanced.sinkhorn_unbalanced(
                a, b, C, reg=reg, reg_m=reg_m, method="sinkhorn_stabilized",
                numItermax=2000, stopThr=1e-9)
        else:
            P = pot.sinkhorn(a, b, C, reg=reg, method="sinkhorn_log",
                             numItermax=2000, stopThr=1e-9)

    row = P.sum(axis=1, keepdims=True)
    row_safe = np.where(row == 0, 1.0, row)
    W = P / row_safe
    if assignment == "argmax":
        W1 = np.zeros_like(W)
        has = row[:, 0] > 0
        W1[np.where(has)[0], np.argmax(P[has], axis=1)] = 1.0
        W = W1
    D = X2[None, :, :] - X1[:, None, :]
    contrib = W[..., None] * D
    V = contrib.sum(axis=1)

    if local_adj is not None:
        no_nb = local_adj.sum(axis=1) == 0
        V[no_nb] = 0.0
        W[no_nb] = 0.0
        contrib[no_nb] = 0.0

    params = dict(reg=reg, lambda_time=lambda_time, lambda_knn=lambda_knn,
                  unbalanced=unbalanced, reg_m=reg_m, mask_self=mask_self,
                  pair=pair, cost_scale=cost_scale, solver=solver,
                  assignment=assignment, scale=scale, delta2=delta2,
                  T_eff=reg * scale / delta2, has_graph=local_adj is not None)

    if check:
        V_pkg = vtl._ot_velocity_pair(
            X, idx_s, idx_t, tau,
            adata.obsp["connectivities"] if "connectivities" in adata.obsp
            else None,
            reg=reg, lambda_time=lambda_time, lambda_knn=lambda_knn,
            unbalanced=unbalanced, reg_m=reg_m, mask_self=mask_self,
            use_graph=use_graph, cost_scale=cost_scale, solver=solver,
            assignment=assignment)
        tol = 1e-8 if package_mode else 1e-6
        if not np.allclose(V, V_pkg, atol=tol, equal_nan=True):
            raise AssertionError(
                "OT decomposition drifted from velot.tl._ot_velocity_pair; "
                f"max |diff| = {np.nanmax(np.abs(V - V_pkg)):.3g}")

    return Decomposition(
        method="ot", idx_source=idx_s, idx_target=idx_t, weights=W, mass=P,
        contrib=contrib, V=V, params=params, cost=C,
        cost_parts=dict(geometry=geo, time=time_pen, knn=knn_pen,
                        self=self_pen, d2_max=d2_max, scale=scale,
                        delta2=delta2))


def window_decomposition(
    adata: AnnData,
    pair: int = 0,
    basis: str = "X_pca",
    check: bool = True,
) -> Decomposition:
    """gradient_velocity(mode="window") for one pair: uniform weights over
    the target window (identical cells excluded when windows overlap)."""
    X = np.asarray(adata.obsm[basis], dtype=np.float64)
    idx_s, idx_t = _window(adata, pair)
    X1, X2 = X[idx_s], X[idx_t]
    n1, n2 = len(idx_s), len(idx_t)

    W = np.full((n1, n2), 1.0 / n2)
    self_pairs = idx_s[:, None] == idx_t[None, :]
    for i in np.where(self_pairs.any(axis=1))[0]:
        if n2 > 1:
            W[i] = np.where(self_pairs[i], 0.0, 1.0 / (n2 - 1))

    local_adj = _local_adj(adata, idx_s, idx_t)
    if local_adj is not None:
        W[local_adj.sum(axis=1) == 0] = 0.0

    D = X2[None, :, :] - X1[:, None, :]
    contrib = W[..., None] * D
    V = contrib.sum(axis=1)

    if check:
        if adata.uns["velot_windows"]["n_pairs"] != 1:
            pass  # package averages over all pairs; only comparable for 1
        else:
            ad = adata.copy()
            vtl.gradient_velocity(ad, basis=basis, mode="window")
            V_pkg = ad.obsm["velot_velocity_raw_pca"][idx_s]
            if not np.allclose(V, V_pkg, atol=1e-8):
                raise AssertionError(
                    "window decomposition drifted from gradient_velocity")

    return Decomposition(
        method="window", idx_source=idx_s, idx_target=idx_t, weights=W,
        mass=W / n1, contrib=contrib, V=V, params=dict(pair=pair))


def knn_decomposition(
    adata: AnnData,
    k: int = 5,
    basis: str = "X_pca",
    weighting: str = "uniform",
    temperature: float = 0.1,
    min_pseudotime_gap: float = 0.0,
    sources=None,
    check: bool = True,
) -> Decomposition:
    """gradient_velocity(mode="knn"), decomposed. Columns are ALL cells,
    because a forward neighbour need not be in the target window.

    sources
        Indices of the cells to show as rows. Default: the source window
        of pair 0 (cluster A).
    """
    X = np.asarray(adata.obsm[basis], dtype=np.float64)
    tau = adata.obs["pseudotime"].values.astype(np.float64)
    n = adata.n_obs
    idx_s = (_window(adata, 0)[0] if sources is None
             else np.asarray(sources))
    idx_t = np.arange(n)

    k_eff = int(min(k, max(1, n - 1)))
    nbr_all = vtl._build_knn_index(X, k=k_eff)

    W = np.zeros((len(idx_s), n))
    for r, i in enumerate(idx_s):
        nbrs = nbr_all[i]
        gaps = tau[nbrs] - tau[i]
        fwd = gaps > min_pseudotime_gap
        if not fwd.any():
            continue
        if weighting == "softmax":
            g = gaps[fwd]
            w = np.exp((g - g.max()) / max(temperature, 1e-12))
            W[r, nbrs[fwd]] = w / w.sum()
        else:
            W[r, nbrs[fwd]] = 1.0 / fwd.sum()

    D = X[None, :, :] - X[idx_s][:, None, :]
    contrib = W[..., None] * D
    V = contrib.sum(axis=1)

    if check:
        ad = adata.copy()
        vtl.gradient_velocity(ad, basis=basis, mode="knn", k=k,
                              weighting=weighting, temperature=temperature,
                              min_pseudotime_gap=min_pseudotime_gap)
        V_pkg = ad.obsm["velot_velocity_raw_pca"][idx_s]
        if not np.allclose(V, V_pkg, atol=1e-8):
            raise AssertionError(
                "knn decomposition drifted from gradient_velocity")

    return Decomposition(
        method="knn", idx_source=idx_s, idx_target=idx_t, weights=W,
        mass=W / len(idx_s), contrib=contrib, V=V,
        params=dict(k=k, weighting=weighting, temperature=temperature,
                    min_pseudotime_gap=min_pseudotime_gap))


# =====================================================================
# Tables
# =====================================================================
def summarize(res: Decomposition, adata: AnnData) -> pd.DataFrame:
    """One row per source cell.

    n_eff       1 / sum_j w_ij^2: how many targets the cell effectively
                spreads over (1 = a single target).
    w_twin      weight on the cell's own displaced copy.
    err_vs_true |v_i - shift|, the error against the known displacement.
    """
    shift = np.asarray(adata.uns["toy"]["shift"], dtype=np.float64)
    twin = adata.obs["twin"].values
    names = adata.obs_names.values
    rows = []
    for r, i in enumerate(res.idx_source):
        w = res.weights[r]
        v = res.V[r]
        nv = np.linalg.norm(v)
        top = int(np.argmax(w)) if w.sum() > 0 else -1
        tw = twin[i]
        col_twin = np.where(res.idx_target == tw)[0] if tw >= 0 else []
        w_tw = float(w[col_twin[0]]) if len(col_twin) else np.nan
        rows.append(dict(
            cell=names[i],
            v_x=v[0], v_y=v[1], norm=nv,
            cos_true=float(v @ shift / (nv * np.linalg.norm(shift)))
            if nv > 0 else np.nan,
            err_vs_true=float(np.linalg.norm(v - shift)),
            n_eff=float(1.0 / (w ** 2).sum()) if w.sum() > 0 else 0.0,
            n_targets=int((w > 1e-3).sum()),
            top_target=names[res.idx_target[top]] if top >= 0 else "",
            w_top=float(w[top]) if top >= 0 else 0.0,
            w_twin=w_tw,
        ))
    return pd.DataFrame(rows).set_index("cell")


def to_long(res: Decomposition, adata: AnnData | None = None,
            tol: float = 1e-6) -> pd.DataFrame:
    """One row per (source, target) pair with non-negligible weight:
    mass, weight, full displacement and the weighted contribution."""
    rows = []
    for r, i in enumerate(res.idx_source):
        for c, j in enumerate(res.idx_target):
            w = res.weights[r, c]
            if w <= tol:
                continue
            ci = res.contrib[r, c]
            d = ci / w
            rows.append(dict(source=i, target=j, mass=res.mass[r, c],
                             weight=w, disp_x=d[0], disp_y=d[1],
                             contrib_x=ci[0], contrib_y=ci[1]))
    df = pd.DataFrame(rows)
    if adata is not None and len(df):
        nm = adata.obs_names.values
        df["source"] = nm[df["source"].values]
        df["target"] = nm[df["target"].values]
    return df


def received_mass(res: Decomposition, adata: AnnData | None = None
                  ) -> pd.Series:
    """Mass arriving at each target cell. Balanced OT fixes this at
    1/n_target for every cell (the column marginal). The gradients have
    no such constraint, so attractive targets can collect more."""
    m = res.mass.sum(axis=0)
    idx = res.idx_target
    names = adata.obs_names.values[idx] if adata is not None else idx
    s = pd.Series(m, index=names, name=f"received_mass[{res.method}]")
    return s[s > 0] if res.method == "knn" else s


# =====================================================================
# Plots
# =====================================================================
def _base_scatter(ax, adata, basis="X_pca", annotate=True):
    X = adata.obsm[basis]
    cl = adata.obs["cluster"].values
    for lab, col in (("A", C_SRC), ("B", C_TGT)):
        m = cl == lab
        ax.scatter(X[m, 0], X[m, 1], s=46, color=col, edgecolor="white",
                   linewidth=1.2, zorder=3, label=f"cluster {lab}")
    if annotate:
        for i, nm in enumerate(adata.obs_names):
            ax.annotate(nm, X[i], xytext=(4, 4), textcoords="offset points",
                        fontsize=7, color="#555", zorder=4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(color="#e6e6e3", linewidth=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("gene_x")
    ax.set_ylabel("gene_y")


def _arrow(ax, x0, v, color, lw=2.0, ls="-", z=5, alpha=1.0, label=None):
    ax.annotate("", xy=x0 + v, xytext=x0, zorder=z,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=ls, alpha=alpha,
                                shrinkA=0, shrinkB=0, mutation_scale=12))
    if label:
        ax.plot([], [], color=color, lw=lw, ls=ls, label=label)


def plot_cell(res: Decomposition, adata: AnnData, cell=0, ax=None,
              basis="X_pca", compose=True, show_true=True, tol=1e-3,
              label_weights=True):
    """How one source cell's velocity is assembled.

    * thin lines from x_i to every target j, width and opacity ~ w_ij,
      labelled with w_ij in % (the mass each target receives from i)
    * if compose=True, the weighted pieces w_ij (x_j - x_i) drawn head to
      tail, largest first: the path they trace ends at x_i + v_i
    * bold arrow: the final raw velocity v_i
    * dashed grey arrow: the true displacement (shift)

    cell may be a row index into res.idx_source or an obs name.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.4))
    X = adata.obsm[basis]
    if isinstance(cell, str):
        gi = adata.obs_names.get_loc(cell)
        r = int(np.where(res.idx_source == gi)[0][0])
    else:
        r = int(cell)
        gi = res.idx_source[r]
    x0 = X[gi]
    w = res.weights[r]

    _base_scatter(ax, adata, basis)
    ax.scatter(*x0, s=150, facecolor="none", edgecolor=C_INK, lw=1.6,
               zorder=6)

    wmax = w.max() if w.max() > 0 else 1.0
    for c in np.where(w > tol)[0]:
        xj = X[res.idx_target[c]]
        a = 0.15 + 0.75 * w[c] / wmax
        ax.plot([x0[0], xj[0]], [x0[1], xj[1]], color=C_TGT,
                lw=0.6 + 4.0 * w[c] / wmax, alpha=a, zorder=2,
                solid_capstyle="round")
        if label_weights:
            mid = x0 + 0.72 * (xj - x0)
            ax.text(*mid, f"{100 * w[c]:.0f}%", fontsize=7, color=C_INK,
                    ha="center", va="center", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="none", alpha=0.8))

    if compose:
        order = np.argsort(-w)
        p = x0.copy()
        for c in order:
            if w[c] <= tol:
                break
            piece = res.contrib[r, c]
            _arrow(ax, p, piece, C_SRC, lw=1.0, alpha=0.7, z=5)
            p = p + piece

    if show_true:
        shift = np.asarray(adata.uns["toy"]["shift"], dtype=np.float64)
        _arrow(ax, x0, shift, C_TRUE, lw=1.6, ls="--", z=6,
               label="true shift")
    _arrow(ax, x0, res.V[r], C_INK, lw=2.4, z=8, label="final v")
    if compose:
        ax.plot([], [], color=C_SRC, lw=1.0, alpha=0.7,
                label="weighted pieces")

    nm = adata.obs_names[gi]
    s = summarize(res, adata).loc[nm]
    ax.set_title(f"{res.label}\n{nm}: n_eff={s.n_eff:.1f}, "
                 f"w_twin={s.w_twin:.2f}, |v-shift|={s.err_vs_true:.2f}",
                 fontsize=9)
    return ax


def plot_all_cells(res: Decomposition, adata: AnnData, ax=None,
                   basis="X_pca", tol=1e-3, show_true=True):
    """All source cells at once: every coupling as a line (opacity ~ w_ij)
    and every final raw velocity as an arrow."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.4))
    X = adata.obsm[basis]
    _base_scatter(ax, adata, basis, annotate=False)
    for r, i in enumerate(res.idx_source):
        w = res.weights[r]
        for c in np.where(w > tol)[0]:
            xj = X[res.idx_target[c]]
            ax.plot([X[i, 0], xj[0]], [X[i, 1], xj[1]], color=C_TGT,
                    lw=0.4 + 2.2 * w[c], alpha=0.1 + 0.6 * w[c], zorder=2)
    if show_true:
        shift = np.asarray(adata.uns["toy"]["shift"], dtype=np.float64)
        for i in res.idx_source:
            _arrow(ax, X[i], shift, C_TRUE, lw=1.0, ls="--", z=4)
    for r, i in enumerate(res.idx_source):
        _arrow(ax, X[i], res.V[r], C_INK, lw=1.6, z=6)
    s = summarize(res, adata)
    ax.set_title(f"{res.label}\nmean |v-shift|={s.err_vs_true.mean():.3f}, "
                 f"mean n_eff={s.n_eff.mean():.1f}", fontsize=9)
    return ax


def plot_weights(res: Decomposition, adata: AnnData, ax=None,
                 only_nonzero_cols=True):
    """Heatmap of w_ij (rows: sources, columns: targets). Each cell's twin
    is outlined, so a perfect translation-recovering plan is the outlined
    diagonal."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 4.4))
    W = res.weights
    cols = np.arange(W.shape[1])
    if only_nonzero_cols and res.method == "knn":
        cols = np.where(W.sum(axis=0) > 0)[0]
        if len(cols) == 0:
            cols = np.arange(W.shape[1])
    Wv = W[:, cols]
    im = ax.imshow(Wv, cmap="Blues", vmin=0, vmax=max(Wv.max(), 1e-12),
                   aspect="auto")
    names = adata.obs_names.values
    ax.set_yticks(range(len(res.idx_source)))
    ax.set_yticklabels(names[res.idx_source], fontsize=7)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(names[res.idx_target[cols]], fontsize=7, rotation=90)
    twin = adata.obs["twin"].values
    for r, i in enumerate(res.idx_source):
        hit = np.where(res.idx_target[cols] == twin[i])[0]
        if len(hit):
            ax.add_patch(plt.Rectangle((hit[0] - 0.5, r - 0.5), 1, 1,
                                       fill=False, ec=C_TGT, lw=1.4))
    ax.set_xlabel("target")
    ax.set_ylabel("source")
    ax.set_title(f"{res.label}\nweights w_ij (twin outlined)", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_received_mass(results, adata: AnnData, ax=None):
    """Mass each target of the window receives, per method. The flat line
    at 1/n is the OT column marginal."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    idx_t = _window(adata, 0)[1]
    names = adata.obs_names.values[idx_t]
    x = np.arange(len(idx_t))
    width = 0.8 / max(1, len(results))
    greys = ["#1f1f1e", "#6f6f6a", "#b4b4ae", "#d8d8d2"]
    for k, res in enumerate(results):
        m = res.mass.sum(axis=0)
        if res.method == "knn":
            m = m[np.searchsorted(res.idx_target, idx_t)]
        ax.bar(x + (k - (len(results) - 1) / 2) * width, m, width * 0.9,
               color=greys[k % len(greys)], label=res.label)
    ax.axhline(1 / len(idx_t), color=C_TGT, lw=1, ls="--",
               label="1 / n_target")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("received mass")
    ax.legend(fontsize=7, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


# =====================================================================
# Convenience
# =====================================================================
def compare(adata: AnnData, reg=0.1, k=5, cells=(0,), ot_variants=None,
            include=("window", "knn"), knn_kwargs=None,
            save: str | None = None, show=True):
    """One row per estimator: all cells | weight heatmap | selected cells.

    ot_variants
        List of kwargs dicts for ot_decomposition, one row each.
        Default: [dict(reg=reg)] (the package's OT). Example:
            [dict(reg=0.1),                         # package
             dict(reg=0.1, use_graph=False),           # no kNN graph
             dict(reg=0.1, use_graph=False, cost_scale="nn"),
             dict(solver="emd", use_graph=False)]
    include
        Gradient baselines to add below the OT rows.

    Returns dict(label -> Decomposition) and prints the per-cell tables.
    """
    knn_kwargs = dict(knn_kwargs or {})
    ot_variants = ot_variants or [dict(reg=reg)]
    results = {}
    for kw in ot_variants:
        r = ot_decomposition(adata, **{"reg": reg, **kw})
        results[r.label] = r
    if "window" in include:
        r = window_decomposition(adata)
        results[r.label] = r
    if "knn" in include:
        r = knn_decomposition(adata, k=k, **knn_kwargs)
        results[r.label] = r

    nrow, ncol = len(results), 2 + len(cells)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.15 * nrow),
                             squeeze=False)
    for row, res in enumerate(results.values()):
        plot_all_cells(res, adata, ax=axes[row, 0])
        plot_weights(res, adata, ax=axes[row, 1])
        for c, cell in enumerate(cells):
            plot_cell(res, adata, cell=cell, ax=axes[row, 2 + c])
    h, lab = axes[0, min(2, ncol - 1)].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=max(1, len(lab)),
               frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    pd.set_option("display.width", 160)
    for name, res in results.items():
        print(f"\n=== {name} ===")
        print(summarize(res, adata).round(3).to_string())
        n_zero = int((res.weights.sum(axis=1) == 0).sum())
        if n_zero:
            why = ("no forward cell among its k nearest neighbours"
                   if res.method == "knn" else
                   "no graph-graph neighbour in the target window")
            print(f"  -> {n_zero}/{len(res.idx_source)} source cells have "
                  f"v = 0: {why}.")

    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.savefig(save, dpi=130, bbox_inches="tight")
        print(f"\nsaved {save}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return results


def error_table(adata: AnnData, variants: dict) -> pd.DataFrame:
    """Mean |v - shift| and mean cosine to the shift for several OT
    variants (name -> kwargs for ot_decomposition). Quick way to compare
    many settings without drawing them."""
    rows = []
    for name, kw in variants.items():
        res = ot_decomposition(adata, **kw)
        smm = summarize(res, adata)
        rows.append(dict(variant=name,
                         T_eff=(np.nan if res.params["solver"] == "emd"
                                else res.params["T_eff"]),
                         mean_err=smm.err_vs_true.mean(),
                         mean_cos=smm.cos_true.mean(),
                         mean_n_eff=smm.n_eff.mean(),
                         twin_weight=smm.w_twin.mean(),
                         n_zero=int((res.weights.sum(1) == 0).sum())))
    return pd.DataFrame(rows).set_index("variant")


# =====================================================================
if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    VARIANTS = [
        dict(reg=0.1),                                   # package
        dict(reg=0.1, use_graph=False),                     # no kNN graph (no penalty, no zeroing)
        dict(reg=0.1, use_graph=False, cost_scale="nn"),    # eps in spacing units
        dict(reg=0.1, use_graph=False, assignment="argmax"),
        dict(solver="emd", use_graph=False),                # exact OT
    ]
    SCENARIOS = {
        # partial spatial overlap: shift smaller than the cloud span
        "overlap": dict(shift=(1.0, 0.4), spread=0.5),
        # no spatial overlap
        "bigshift": dict(shift=(3.0, 1.0), spread=0.6),
    }
    for scen, geo in SCENARIOS.items():
        for target in ("twin", "resample"):
            ad = make_toy_adata(seed=0, target=target, **geo)
            tag = f"{scen}_{target}"
            print("\n" + "#" * 74 + f"\n# {tag}\n" + "#" * 74)
            compare(ad, reg=0.1, k=5, cells=(0, 3), ot_variants=VARIANTS,
                    save=os.path.join(OUT, f"{tag}.png"), show=False)
            tab = error_table(ad, {
                "package eps=0.1": dict(reg=0.1),
                "package eps=0.01": dict(reg=0.01),
                "no graph eps=0.1": dict(reg=0.1, use_graph=False),
                "no graph eps=0.01": dict(reg=0.01, use_graph=False),
                "no graph, nn-scale eps=0.1": dict(reg=0.1, use_graph=False,
                                                 cost_scale="nn"),
                "no graph, nn-scale eps=1": dict(reg=1.0, use_graph=False,
                                               cost_scale="nn"),
                "no graph argmax eps=0.1": dict(reg=0.1, use_graph=False,
                                              assignment="argmax"),
                "exact EMD": dict(solver="emd", use_graph=False),
            })
            tab.to_csv(os.path.join(OUT, f"{tag}_errors.csv"))
            print("\n" + tab.round(3).to_string())

    print(f"\nFigures and tables in {OUT}")

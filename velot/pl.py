"""
velot.pl — Plotting
===================

Visualization functions for the VelOT pipeline.
Follows the scanpy/scvelo convention.

All functions accept ``show=True`` (display immediately) and
``save=None`` (path to save the figure).

Usage::

    import velot

    velot.pl.velocity_stream(adata, color="clusters")
    velot.pl.windows(adata)
    velot.pl.training_curves(adata)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData

try:
    import scvelo as scv

    _HAS_SCVELO = True
except ImportError:
    _HAS_SCVELO = False


# =====================================================================
# Main plotting functions
# =====================================================================

def dataset_overview_simple(
    adata: AnnData,
    color: str = "clusters",
    basis: str = "umap",
    title: Optional[str] = "",
    show: bool = True,
    save: bool = False,
    save_path: Optional[str] = None,
    figsize: tuple = (5,5),
    inframe: bool = False,
    out_legend: bool = False
) -> plt.Figure:
    """
    Overview of the dataset: embedding colored by clusters and pseudotime.

    Parameters
    ----------
    adata
        Annotated data matrix with UMAP and pseudotime computed.
    color
        Column in adata.obs for cluster coloring.
    basis
        Embedding to plot (``'umap'`` or ``'pca'``).
    show
        Whether to display the figure.
    save
        File path to save the figure. None to skip.
    figsize
        Figure size.

    Returns
    -------
    matplotlib Figure.
    """

    # extra_width = 0
    # if out_legend:
    #     extra_width += 0

    # fig, axes = plt.subplots(
    #     figsize=(figsize[0] + extra_width, figsize[1]),
    #     constrained_layout=True
    # )

    fig, axes = plt.subplots(figsize=figsize)

    if inframe:
        scv.pl.scatter(
            adata, basis=basis, color=color,
            ax=axes, show=False, title="",
            frameon=False, legend_loc="on data"
        )
    
    else:
        sc.pl.embedding(
            adata, basis=basis, color=color,
            ax=axes, show=False, title="",
            frameon=False
        )

        # Get existing legend
        leg = axes.get_legend()
        if leg is not None:
            handles = leg.legend_handles
            labels = [t.get_text() for t in leg.get_texts()]
            leg.remove()  # remove original legend

            if out_legend:
                args = {"loc": "center left", "bbox_to_anchor": (1,0.5)}
            else:
                args = {"loc": "lower right"}


            # Create new legend (customized)
            # axes.legend(
            #     handles,
            #     labels,
            #     # title="Sample",          # or color name
            #     fontsize=14,             # bigger text
            #     # title_fontsize=11,
            #     markerscale=1.5,           # bigger markers
            #     frameon=False,
            #     **args
            # )

            fig.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=14,
                markerscale=1.5,
                frameon=False
            )

    axes.set_title(title, fontsize=18, fontfamily="sans serif")
    add_umap_axis(axes)

    # plt.tight_layout()
    _finish(fig, show=show, save=save, save_path=save_path)
    return fig if not show else None

def dataset_overview(
    adata: AnnData,
    color: str = "clusters",
    basis: str = "umap",
    title: bool = True,
    show: bool = True,
    vertical: bool = False,
    save: Optional[str] = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Overview of the dataset: embedding colored by clusters and pseudotime.

    Parameters
    ----------
    adata
        Annotated data matrix with UMAP and pseudotime computed.
    color
        Column in adata.obs for cluster coloring.
    basis
        Embedding to plot (``'umap'`` or ``'pca'``).
    show
        Whether to display the figure.
    save
        File path to save the figure. None to skip.
    figsize
        Figure size.

    Returns
    -------
    matplotlib Figure.
    """
    if vertical:
        fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize)

    titles = ("", "")
    if title:
        titles = (color, "Diffusion Pseudotime")

    sc.pl.embedding(
        adata, basis=basis, color=color,
        ax=axes[0], show=False, title="",
        frameon=False
    )
    axes[0].set_title(titles[0], fontsize=18, fontfamily="sans serif")

    ax = axes[0]
    add_umap_axis(ax)

    # Get existing legend
    leg = ax.get_legend()
    if leg is not None:
        handles = leg.legend_handles
        labels = [t.get_text() for t in leg.get_texts()]
        leg.remove()  # remove original legend

        # Create new legend (customized)
        ax.legend(
            handles,
            labels,
            # title="Sample",          # or color name
            loc="lower right",       # change position here
            fontsize=14,             # bigger text
            # title_fontsize=11,
            markerscale=1.5,           # bigger markers
            frameon=False,
        )

    if "pseudotime" in adata.obs:
        sc.pl.embedding(
            adata, basis=basis, color="pseudotime",
            ax=axes[1], show=False, title="",
            frameon=False, color_map="viridis",
        )
    else:
        axes[1].text(
            0.5, 0.5, "No pseudotime computed",
            ha="center", va="center", transform=axes[1].transAxes,
        )
        axes[1].set_title("Pseudotime")
    axes[1].set_title(titles[1], fontsize=18, fontfamily="sans serif")
    add_umap_axis(axes[1])

    plt.tight_layout()
    _finish(fig, show=show, save=save, save_path=save_path)
    return fig if not show else None

def add_umap_axis(ax, basis='umap', pos=(0, 0), length=0.2, fontsize=10):
    """
    Draw equal-length axis arrows on a scatter plot.

    The arrows are guaranteed to be the same physical length
    on screen regardless of the axes aspect ratio.

    Parameters
    ----------
    ax
        Matplotlib Axes to draw on.
    basis
        ``'umap'`` or ``'pca'`` — controls the labels.
    pos
        Position of the arrow origin in axes fraction (0–1).
    length
        Arrow length in axes-fraction units (applied to the
        x-direction; the y-direction is adjusted to match).
    fontsize
        Label font size.
    """
    x0, y0 = pos

    # ------------------------------------------------------------------
    # Compute the physical aspect ratio of the axes so both arrows
    # have the same length in display (screen/paper) space
    # ------------------------------------------------------------------
    fig = ax.figure

    # Try renderer-based measurement first (most accurate after draw)
    try:
        renderer = fig.canvas.get_renderer()
        bbox = ax.get_window_extent(renderer=renderer)
        ax_width = bbox.width   # pixels
        ax_height = bbox.height  # pixels
    except (AttributeError, TypeError):
        # Fallback: compute from figure size and axes position
        bbox = ax.get_position()
        fig_w, fig_h = fig.get_size_inches()
        ax_width = bbox.width * fig_w    # inches
        ax_height = bbox.height * fig_h  # inches

    aspect = ax_width / max(ax_height, 1e-10)

    # length_x is the reference (in axes fraction)
    # length_y is adjusted so physical (display) length matches
    # 
    # Physical length of x-arrow = length_x * ax_width
    # Physical length of y-arrow = length_y * ax_height
    # Set equal: length_y * ax_height = length_x * ax_width
    #         → length_y = length_x * (ax_width / ax_height)
    #         → length_y = length_x * aspect
    length_x = length
    length_y = length * aspect

    # Arrows
    ax.annotate(
        "", xy=(x0 + length_x, y0), xytext=(x0, y0),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
    )
    ax.annotate(
        "", xy=(x0, y0 + length_y), xytext=(x0, y0),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
    )

    # Labels
    if basis == 'umap':
        texts = ["UMAP1", "UMAP2"]
    elif basis == 'pca':
        texts = ["PCA1", "PCA2"]
    else:
        texts = [f"{basis.upper()}1", f"{basis.upper()}2"]

    ax.text(
        x0 + length_x, y0 - 0.02, texts[0],
        transform=ax.transAxes, ha="right", va="top", fontsize=fontsize,
    )
    ax.text(
        x0 - 0.02, y0 + length_y, texts[1],
        transform=ax.transAxes, ha="right", va="top",
        rotation=90, fontsize=fontsize,
    )

def windows(
    adata: AnnData,
    basis: str = "umap",
    pairs_to_show: Optional[Sequence[int]] = None,
    max_show: int = 12,
    ncols: int = 4,
    point_size: int = 20,
    title: str = None,
    frameon: bool = False,
    show: bool = True,
    save: bool = False,
    save_path: Optional[str] = None,
    figsize_per_panel: tuple = (3, 3),
) -> plt.Figure:
    """
    Visualize the spatial-temporal windows.

    Each panel shows one window pair: source cells in blue, target
    cells in orange. Overlap is shown in green.

    Parameters
    ----------
    adata
        Must have ``adata.uns['velot_windows']`` from
        ``velot.tl.build_windows()``.
    basis
        Embedding to plot in.
    pairs_to_show
        Specific pair indices to display. For example,
        ``[10, 13, 14, 15]`` shows only those four pairs.
        If None, shows the first ``max_show`` pairs.
    max_show
        Maximum number of pairs when ``pairs_to_show`` is None.
    ncols
        Number of columns in the grid.
    point_size
        Scatter point size.
    show
        Whether to display.
    save
        File path to save.

    Returns
    -------
    matplotlib Figure.

    Examples
    --------
    Show first 8 pairs::

        velot.pl.windows(adata, max_show=8)

    Show specific pairs::

        velot.pl.windows(adata, pairs_to_show=[10, 13, 14, 15])
    """
    if "velot_windows" not in adata.uns:
        raise ValueError("No windows found. Run velot.tl.build_windows() first.")

    pairs = adata.uns["velot_windows"]["pairs"]
    n_total = len(pairs)

    # Determine which pairs to show
    if pairs_to_show is not None:
        indices = list(pairs_to_show)
        # Validate
        for idx in indices:
            if idx < 0 or idx >= n_total:
                raise ValueError(
                    f"Pair index {idx} out of range. "
                    f"Dataset has {n_total} pairs (0 to {n_total - 1})."
                )
    else:
        indices = list(range(min(max_show, n_total)))

    n_show = len(indices)
    nrows = int(np.ceil(n_show / ncols))

    coords = adata.obsm.get(f"X_{basis}")
    if coords is None:
        raise ValueError(f"Embedding 'X_{basis}' not found in adata.obsm.")

    figsize = (ncols * figsize_per_panel[0], nrows * figsize_per_panel[1])
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()

    for panel_i, pair_idx in enumerate(indices):
        ax = axes[panel_i]
        idx_src, idx_tgt = pairs[pair_idx]

        # Background: all cells in light gray
        ax.scatter(
            coords[:, 0], coords[:, 1],
            s=1, c="lightgray", alpha=0.3,
        )

        # Find overlap
        overlap = np.intersect1d(idx_src, idx_tgt)
        src_only = np.setdiff1d(idx_src, overlap)
        tgt_only = np.setdiff1d(idx_tgt, overlap)

        # Plot source, target, overlap
        if len(tgt_only) > 0:
            ax.scatter(
                coords[tgt_only, 0], coords[tgt_only, 1],
                s=point_size, c="tab:orange", alpha=0.6, label="target",
            )
        if len(src_only) > 0:
            ax.scatter(
                coords[src_only, 0], coords[src_only, 1],
                s=point_size, c="tab:blue", alpha=0.6, label="source",
            )
        if len(overlap) > 0:
            ax.scatter(
                coords[overlap, 0], coords[overlap, 1],
                s=point_size, c="tab:green", alpha=0.6, label="overlap",
            )

        # ax.set_title(f"Pair {pair_idx}", fontsize=18, fontfamily="sans serif")

        if not frameon:
            ax.axis("off")
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        # if panel_i == 0:
        #     ax.legend(fontsize=6, loc="lower right")

    add_umap_axis(axes[0])
    
    # Turn off unused axes
    for j in range(n_show, len(axes)):
        # axes[j].axis("off")
        fig.delaxes(axes[j])

    if title is not None:
        fig.suptitle(
            f"Window pairs ({n_show} of {n_total} shown)",
            fontsize=18, fontweight="bold",
        )
    
    handles, labels = axes[panel_i].get_legend_handles_labels()
    axes[panel_i].legend(
        handles,
        labels,
        # title="Sample",
        fontsize=14,
        # title_fontsize=8,
        markerscale=2,
        loc="lower right",
        frameon=False
    )

    plt.tight_layout()
    _finish(fig, show=show, save=save, save_path=save_path)
    return fig if not show else None


def window_transport(
    adata: AnnData,
    pair_index: int,
    basis: str = "umap",
    velocity_basis: str = "X_pca",
    reg: float = 0.05,
    lambda_time: float = 1.0,
    lambda_knn: float = 1.0,
    show_top_k: Optional[int] = None,
    padding: float = 0.3,
    background_alpha: float = 0.4,
    background_size: int = 15,
    background_color: str = "lightgray",
    point_size: int = 60,
    arrow_alpha: float = 0.4,
    arrow_width: float = 0.003,
    arrow_color: str = "black",
    colorby_weight: bool = True,
    cluster_key: Optional[str] = None,
    frameon: bool = False,
    show: bool = True,
    save: bool = False,
    save_path: Optional[str] = None,
    figsize: tuple = (4,4),
) -> plt.Figure:
    """
    Zoom into a window pair showing OT transport on the full dataset.

    All cells are shown as context (gray or colored by cluster),
    with source and target cells highlighted and transport arrows
    overlaid. The axes are zoomed to the window region.

    Parameters
    ----------
    adata
        Must have windows computed.
    pair_index
        Index of the window pair to visualize.
    basis
        Embedding for visualization.
    velocity_basis
        Embedding where OT is computed (for recomputing the plan).
    reg
        Sinkhorn regularization.
    lambda_time
        Pseudotime backward penalty.
    lambda_knn
        Non-neighbor penalty.
    show_top_k
        Show arrows only to top-k targets per source cell.
        None shows all above a threshold.
    padding
        Fractional padding around the window for zoom.
    background_alpha
        Transparency of background (non-window) cells.
    background_size
        Point size of background cells.
    background_color
        Color for background cells when ``cluster_key`` is None.
    point_size
        Size of source/target cells.
    arrow_alpha
        Arrow transparency.
    arrow_width
        Arrow line width.
    arrow_color
        Arrow color when ``colorby_weight=False``.
    colorby_weight
        Color arrows by transport weight.
    cluster_key
        If provided, background cells are colored by cluster
        instead of uniform gray. Gives spatial context.
    frameon
        Show axes frame.
    show
        Display the figure.
    save
        File path to save.

    Examples
    --------
    ::

        # With cluster context
        velot.pl.window_transport(adata, 5, cluster_key="clusters")

        # Clean, gray background
        velot.pl.window_transport(adata, 5)

        # Top-3 connections only
        velot.pl.window_transport(adata, 5, show_top_k=3)
    """
    import ot as pot

    if "velot_windows" not in adata.uns:
        raise ValueError("No windows found. Run velot.tl.build_windows() first.")

    pairs = adata.uns["velot_windows"]["pairs"]
    n_total = len(pairs)

    if pair_index < 0 or pair_index >= n_total:
        raise ValueError(
            f"pair_index {pair_index} out of range (0 to {n_total - 1})."
        )

    idx_src, idx_tgt = pairs[pair_index]

    # Visualization coordinates
    coords_key = f"X_{basis}"
    if coords_key not in adata.obsm:
        raise ValueError(f"'{coords_key}' not found in adata.obsm.")
    coords = adata.obsm[coords_key]

    # OT computation coordinates
    if velocity_basis not in adata.obsm:
        raise ValueError(f"'{velocity_basis}' not found in adata.obsm.")
    X = adata.obsm[velocity_basis]
    pseudotime = adata.obs["pseudotime"].values

    # ------------------------------------------------------------------
    # Recompute transport plan
    # ------------------------------------------------------------------
    X_src = X[idx_src]
    X_tgt = X[idx_tgt]
    n_src = X_src.shape[0]
    n_tgt = X_tgt.shape[0]

    a = np.ones(n_src, dtype=np.float64) / n_src
    b = np.ones(n_tgt, dtype=np.float64) / n_tgt

    C = pot.dist(X_src, X_tgt, metric="sqeuclidean")
    C_max = C.max()
    if C_max > 0:
        C = C / C_max

    t_src = pseudotime[idx_src]
    t_tgt = pseudotime[idx_tgt]
    time_diff = t_src[:, None] - t_tgt[None, :]
    C[time_diff > 0] += lambda_time

    if "connectivities" in adata.obsp:
        knn_adj = adata.obsp["connectivities"]
        local_adj = knn_adj[idx_src][:, idx_tgt]
        if hasattr(local_adj, "toarray"):
            local_adj = local_adj.toarray()
        C[local_adj == 0] += lambda_knn

    try:
        P = pot.sinkhorn(a, b, C, reg=reg, numItermax=300, stopThr=1e-6)
    except Exception:
        try:
            P = pot.emd(a, b, C)
        except Exception:
            raise RuntimeError("OT computation failed for this pair.")

    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P_norm = P / row_sums

    # ------------------------------------------------------------------
    # Compute zoom region from window cells
    # ------------------------------------------------------------------
    coords_src = coords[idx_src]
    coords_tgt = coords[idx_tgt]
    all_window_coords = np.vstack([coords_src, coords_tgt])

    x_min, y_min = all_window_coords.min(axis=0)
    x_max, y_max = all_window_coords.max(axis=0)
    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)

    xlim = (x_min - padding * x_range, x_max + padding * x_range)
    ylim = (y_min - padding * y_range, y_max + padding * y_range)

    # ------------------------------------------------------------------
    # Identify which cells are in the zoomed region (for background)
    # ------------------------------------------------------------------
    in_view = (
        (coords[:, 0] >= xlim[0]) & (coords[:, 0] <= xlim[1]) &
        (coords[:, 1] >= ylim[0]) & (coords[:, 1] <= ylim[1])
    )

    # Cells that are in view but NOT in the window pair
    all_window_idx = np.union1d(idx_src, idx_tgt)
    is_window_cell = np.zeros(adata.n_obs, dtype=bool)
    is_window_cell[all_window_idx] = True
    bg_mask = in_view & ~is_window_cell

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # Layer 1: Background cells in the zoomed region
    bg_indices = np.where(bg_mask)[0]

    if cluster_key is not None and cluster_key in adata.obs:
        categories = adata.obs[cluster_key].astype("category")
        cat_names = categories.cat.categories.tolist()
        codes = categories.cat.codes.values
        cmap = plt.cm.tab20
        colors_array = cmap(np.linspace(0, 1, len(cat_names)))

        for ci, cat_name in enumerate(cat_names):
            cat_bg = bg_indices[codes[bg_indices] == ci]
            if len(cat_bg) > 0:
                ax.scatter(
                    coords[cat_bg, 0], coords[cat_bg, 1],
                    s=background_size,
                    c=[colors_array[ci]],
                    alpha=background_alpha,
                    label=cat_name,
                    zorder=1,
                )
    else:
        if len(bg_indices) > 0:
            ax.scatter(
                coords[bg_indices, 0], coords[bg_indices, 1],
                s=background_size, c=background_color,
                alpha=background_alpha, zorder=1,
            )

    # Layer 2: Transport arrows (behind the highlighted cells)
    w_max = P_norm.max() if P_norm.max() > 0 else 1.0

    for i in range(n_src):
        src_coord = coords_src[i]

        if show_top_k is not None:
            top_indices = np.argsort(P_norm[i])[::-1][:show_top_k]
        else:
            threshold = 1.0 / (n_tgt * 5)
            top_indices = np.where(P_norm[i] > threshold)[0]

        for j in top_indices:
            weight = P_norm[i, j]
            if weight < 1e-10:
                continue

            tgt_coord = coords_tgt[j]
            dx = tgt_coord[0] - src_coord[0]
            dy = tgt_coord[1] - src_coord[1]

            if colorby_weight:
                intensity = min(1.0, weight / (w_max * 0.5))
                color = plt.cm.Reds(0.3 + 0.7 * intensity)
                alpha = max(0.1, min(0.8, arrow_alpha + 0.4 * intensity))
                width = arrow_width * (0.5 + 1.5 * intensity)
            else:
                color = arrow_color
                alpha = arrow_alpha
                width = arrow_width

            ax.annotate(
                "",
                xy=(tgt_coord[0], tgt_coord[1]),
                xytext=(src_coord[0], src_coord[1]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=width * 300,
                    alpha=alpha,
                    shrinkA=2, shrinkB=2,
                    mutation_scale=12,
                ),
                zorder=2,
            )

    # Layer 3: Source and target cells on top
    overlap = np.intersect1d(idx_src, idx_tgt)
    src_only = np.setdiff1d(idx_src, overlap)
    tgt_only = np.setdiff1d(idx_tgt, overlap)

    if len(tgt_only) > 0:
        ax.scatter(
            coords[tgt_only, 0], coords[tgt_only, 1],
            s=point_size, c="tab:orange", alpha=0.85,
            edgecolors="black", linewidths=0.4,
            label="target", zorder=4,
        )
    if len(src_only) > 0:
        ax.scatter(
            coords[src_only, 0], coords[src_only, 1],
            s=point_size, c="tab:blue", alpha=0.85,
            edgecolors="black", linewidths=0.4,
            label="source", zorder=4,
        )
    if len(overlap) > 0:
        ax.scatter(
            coords[overlap, 0], coords[overlap, 1],
            s=point_size, c="tab:green", alpha=0.85,
            edgecolors="black", linewidths=0.4,
            label="overlap", zorder=4,
        )

    # ------------------------------------------------------------------
    # Zoom and formatting
    # ------------------------------------------------------------------
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Legend without duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(), by_label.keys(),
        fontsize=20, loc="lower right", frameon=False, #framealpha=0.9,
    )

    # ax.set_title(
    #     f"Window pair {pair_index} — OT transport\n"
    #     f"src={n_src} cells (blue), tgt={n_tgt} cells (orange), "
    #     f"reg={reg}",
    #     fontsize=11,
    # )

    ax.set_title(
        f"OT transport for pair {pair_index}", fontsize=26, fontfamily="sans serif"
    )

    if not frameon:
        ax.axis("off")

    plt.tight_layout()
    _finish(fig, show=show, save=save, save_path=save_path)
    return fig if not show else None


def velocity_stream(
    adata: AnnData,
    color: str = "clusters",
    basis: str = "umap",
    velocity_key: str = "velot_velocity",
    title: Optional[str] = "VelOT velocity",
    show: bool = True,
    save: bool = False,
    save_path: Optional[str] = None,
    figsize: tuple = (4,4),
    **kwargs,
) -> plt.Figure:
    """
    Stream plot of the velocity field.

    Uses scVelo's stream plotting if available, falls back to a quiver
    plot otherwise.

    Parameters
    ----------
    adata
        Must have ``adata.obsm['velocity_umap']``.
    color
        Column in adata.obs for cell coloring.
    basis
        Embedding basis.
    velocity_key
        Key in adata.obsm containing the velocity to plot.
    title
        Plot title. Defaults to "VelOT velocity".
    show
        Whether to display.
    save
        File path to save.
    **kwargs
        Passed to scvelo.pl.velocity_embedding_stream or quiver.

    Returns
    -------
    matplotlib Figure.
    """
    # if velocity_key not in adata.obsm:
    #     raise ValueError(
    #         f"'{velocity_key}' not found in adata.obsm. "
    #         f"Run velot.tl.velocity() first."
    #     )

    if _HAS_SCVELO:
        # scVelo expects the velocity key to follow its naming convention
        # velocity_umap is already the standard name
        fig, ax = plt.subplots(figsize=figsize)
        # if basis == "umap" and "velocity_umap" not in adata.obsm:
        #     adata.obsm["velocity_umap"] = adata.obsm["velot_velocity"]
        # elif basis =="pca":
        #     adata.obsm["velocity_pca"] = adata.obsm["velot_velocity"]
        # else:
        #     raise ValueError(
        #         f"'{basis}' not implemented"
        #     )

        scv.pl.velocity_embedding_stream(
            adata, basis=basis, vkey=velocity_key, color=color,
            title="", ax=ax, show=False, **kwargs,
        )
        if title is not None:
            ax.set_title(title, fontsize=18, fontfamily="sans serif")
        add_umap_axis(ax)
        _finish(fig, show=show, save=save, save_path=save_path)
        return fig if not show else None
    else:
        return velocity_quiver(
            adata, color=color, basis=basis, velocity_key=velocity_key,
            title=title, show=show, save=save, figsize=figsize,
        )

def velocity_quiver(
    adata: AnnData,
    color: str = "clusters",
    basis: str = "umap",
    velocity_key: str = "velocity_umap",
    title: Optional[str] = None,
    spot_size=50,
    arrow_scale: float = 1.0,
    subsample: Optional[int] = None,
    arrow_color: str = "black",
    arrow_alpha: float = 0.7,
    normalize_arrows: bool = False,
    show: bool = True,
    save: bool = False,
    save_path: Optional[str] = None,
    figsize: tuple = (7, 6),
    **scatter_kwargs,
) -> plt.Figure:
    """
    Quiver plot of velocity arrows overlaid on a scVelo/scanpy scatter.

    Uses scVelo's scatter as the base plot for consistent styling
    (colors, legends, layout). Falls back to scanpy, then to plain
    matplotlib if neither is available.

    Parameters
    ----------
    adata
        Must have embedding and velocity computed.
    color
        Column in ``adata.obs`` for coloring cells (passed to the
        base scatter plot).
    basis
        Embedding to use: ``"umap"``, ``"pca"``, etc.
        Looks for ``adata.obsm[f"X_{basis}"]``.
    velocity_key
        Key in ``adata.obsm`` with velocity vectors **in the same
        coordinate system as the embedding**. For UMAP basis, use
        projected UMAP velocities. For PCA basis, use PCA velocities
        (only the first 2 components are plotted).
    title
        Plot title.
    arrow_scale
        Multiplier for arrow length. Increase to make arrows longer,
        decrease to shorten them.
    subsample
        Number of cells to show arrows for. ``None`` for all cells.
        Recommended ~500-1000 for readability.
    arrow_color
        Color of the quiver arrows. Can be any matplotlib color.
    arrow_alpha
        Transparency of arrows (0–1).
    normalize_arrows
        If True, all arrows have the same length (unit vectors),
        showing direction only. Useful when velocity magnitudes vary
        wildly.
    show
        Whether to display the plot.
    save
        Path to save the figure.
    figsize
        Figure size in inches.
    **scatter_kwargs
        Extra keyword arguments passed to the base scatter plot
        (e.g., ``legend_loc``, ``size``, ``alpha``, ``palette``).

    Returns
    -------
    matplotlib Figure.

    Examples
    --------
    Basic quiver on UMAP::

        velot.pl.velocity_quiver(adata, velocity_key="velocity_umap")

    Quiver on PCA (first 2 components), subsampled::

        velot.pl.velocity_quiver(
            adata, basis="pca", velocity_key="velot_velocity",
            subsample=500,
        )

    Normalized arrows colored red::

        velot.pl.velocity_quiver(
            adata, normalize_arrows=True, arrow_color="red",
            arrow_scale=0.5, subsample=800,
        )
    """
    coords_key = f"X_{basis}"
    if coords_key not in adata.obsm:
        raise ValueError(f"'{coords_key}' not found in adata.obsm.")
    if velocity_key not in adata.obsm:
        raise ValueError(f"'{velocity_key}' not found in adata.obsm.")

    coords = adata.obsm[coords_key][:, :2]  # always first 2 dims for plot
    vel = adata.obsm[velocity_key]

    # Only use first 2 dimensions of velocity for 2D plot
    if vel.shape[1] > 2:
        vel_2d = vel[:, :2]
    else:
        vel_2d = vel

    plot_title = title
    if title == "auto":
        plot_title = f"VelOT velocity (quiver — {basis.upper()})"

    # ------------------------------------------------------------------
    # Base scatter: try scvelo → scanpy → matplotlib
    # ------------------------------------------------------------------
    ax = None

    # --- Try scVelo ---
    try:
        ax = scv.pl.scatter(
            adata,
            basis=basis,
            color=color,
            size=spot_size,
            title=plot_title,
            frameon=False,
            legend_loc="on data",
            figsize=figsize,
            show=False,
            **scatter_kwargs,
        )
        # scv.pl.scatter can return a list if color is a list
        if isinstance(ax, (list, np.ndarray)):
            ax = ax[0]
    except Exception:
        ax = None

    # --- Try scanpy ---
    if ax is None:
        try:
            import scanpy as sc
            ax = sc.pl.embedding(
                adata,
                basis=basis,
                color=color,
                title=plot_title,
                figsize=figsize,
                show=False,
                return_fig=False,
                **scatter_kwargs,
            )
            # sc.pl.embedding may return axes or None
            if ax is None:
                ax = plt.gca()
            if isinstance(ax, (list, np.ndarray)):
                ax = ax[0]
        except Exception:
            ax = None

    # --- Fallback: plain matplotlib ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        if color in adata.obs:
            categories = adata.obs[color].astype("category")
            codes = categories.cat.codes.values
            n_cats = len(categories.cat.categories)
            cmap = plt.cm.tab20 if n_cats <= 20 else plt.cm.tab20b
            scatter = ax.scatter(
                coords[:, 0], coords[:, 1],
                s=10, c=codes, cmap=cmap, alpha=0.5, zorder=1,
            )
            # Manual legend
            handles = []
            for i, cat in enumerate(categories.cat.categories):
                handles.append(
                    plt.Line2D(
                        [0], [0],
                        marker="o", color="w",
                        markerfacecolor=cmap(i / max(1, n_cats - 1)),
                        markersize=6, label=cat,
                    )
                )
            ax.legend(
                handles=handles,
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                frameon=False,
                fontsize=8,
            )
        else:
            ax.scatter(
                coords[:, 0], coords[:, 1],
                s=10, c="lightgray", alpha=0.5, zorder=1,
            )

        ax.set_title(plot_title, fontsize=13)
        ax.set_xlabel(f"{basis.upper()}1")
        ax.set_ylabel(f"{basis.upper()}2")

    fig = ax.figure

    # ------------------------------------------------------------------
    # Subsample for readability
    # ------------------------------------------------------------------
    if subsample is not None and subsample < adata.n_obs:
        rng = np.random.RandomState(42)
        idx = rng.choice(adata.n_obs, subsample, replace=False)
    else:
        idx = np.arange(adata.n_obs)

    # ------------------------------------------------------------------
    # Velocity arrows
    # ------------------------------------------------------------------
    v = vel_2d[idx].copy()

    if normalize_arrows:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        v = v / (norms + 1e-10)

    # Auto-scale: make arrows visible relative to the embedding range
    # The user's arrow_scale multiplies this base scale
    coord_range = np.ptp(coords, axis=0).mean()  # mean span of embedding
    vel_mag = np.linalg.norm(v, axis=1).mean()

    if vel_mag > 1e-10:
        # Base: arrows span ~3% of the plot range on average
        auto_scale = (coord_range * 0.03) / vel_mag
    else:
        auto_scale = 1.0

    scale_factor = auto_scale * arrow_scale

    ax.quiver(
        coords[idx, 0],
        coords[idx, 1],
        v[:, 0] * scale_factor,
        v[:, 1] * scale_factor,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.002 * coord_range / 20,  # scale width to plot range
        headwidth=4,
        headlength=5,
        color=arrow_color,
        alpha=arrow_alpha,
        zorder=10,
    )
    add_umap_axis(ax, basis=basis)

    plt.tight_layout()
    _finish(fig, show=show, save=save, save_path=save_path)
    return fig if not show else None

def confidence(
    adata: AnnData,
    basis: str = "umap",
    show: bool = True,
    save: bool = False,
    save_path: Optional[str] = None,
    figsize: tuple = (7, 6),
) -> plt.Figure:
    """
    Plot per-cell velocity confidence (how many windows contributed).

    Low confidence regions are where the OT velocity is uncertain
    or missing, and the smoothing network is interpolating.

    Returns
    -------
    matplotlib Figure.
    """
    if "velot_confidence" not in adata.obs:
        raise ValueError("No confidence data. Run velot.tl.compute_ot_velocity() first.")

    fig, ax = plt.subplots(figsize=figsize)

    sc.pl.embedding(
        adata, basis=basis, color="velot_confidence",
        ax=ax, show=False, title="VelOT confidence",
        color_map="YlOrRd", frameon=False,
    )

    plt.tight_layout()
    _finish(fig, show=show, save=save, save_path=save_path)
    return fig if not show else None


def training_curves(
    adata: AnnData,
    vertical: bool=False,
    show: bool = True,
    save: bool = False,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """
    Plot training loss curves from the smoothing step.

    Returns
    -------
    matplotlib Figure.
    """
    if "velot_smoothing" not in adata.uns:
        raise ValueError("No smoothing data. Run velot.tl.smooth_velocity() first.")

    info = adata.uns["velot_smoothing"]

    if vertical:
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    axes[0].plot(info["losses_regression"], color="tab:blue", alpha=0.8)
    axes[0].set_title("Regression loss", fontsize=16)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")

    axes[1].plot(info["losses_smoothness"], color="tab:orange", alpha=0.8)
    # axes[1].set_title(f"Smoothness loss (λ={info['lambda_smooth']})")
    axes[1].set_title(f"Smoothness loss", fontsize=16)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_yscale("log")

    # fig.suptitle("VelOT smoothing training", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _finish(fig, show=show, save=save, save_path=save_path)
    return fig if not show else None

def training_curves_single(
    adata: AnnData,
    show: bool = True,
    save: bool = False,
    save_path: Optional[str] = None,
    figsize: tuple = (7,7),
) -> plt.Figure:
    """
    Plot training loss curves from the smoothing step.

    Returns
    -------
    matplotlib Figure.
    """
    if "velot_smoothing" not in adata.uns:
        raise ValueError("No smoothing data. Run velot.tl.smooth_velocity() first.")

    info = adata.uns["velot_smoothing"]

    fig, axes = plt.subplots(figsize=figsize)

    losses = ["losses_regression", "losses_smoothness", "losses_curl"]
    titles = ["Regression", "Smoothness", "Curl"]

    for loss,title in zip(losses,titles):
        axes.plot(info[loss], alpha=0.8, label=title)
        axes.set_xlabel("Epoch")
        axes.set_ylabel("Loss")
        axes.set_yscale("log")
    
    axes.legend(loc="lower right")
    _finish(fig, show=show, save=save, save_path=save_path)
    return fig if not show else None


def spatial_clusters(
    adata: AnnData,
    basis: str = "umap",
    show: bool = True,
    save: bool = False,
    save_path: Optional[str] = None,
    figsize: tuple = (7, 6),
) -> plt.Figure:
    """
    Visualize the spatial clusters used for windowing.

    Returns
    -------
    matplotlib Figure.
    """
    if "velot_spatial_cluster" not in adata.obs:
        raise ValueError(
            "No spatial clusters. Run velot.tl.build_windows() first."
        )

    fig, ax = plt.subplots(figsize=figsize)

    sc.pl.embedding(
        adata, basis=basis, color="velot_spatial_cluster",
        ax=ax, show=False, title="VelOT spatial clusters",
        frameon=False, legend_loc="on data",
    )

    plt.tight_layout()
    _finish(fig, show=show, save=save, save_path=save_path)
    return fig if not show else None


def velocity_comparison(
    adata: AnnData,
    velocity_keys: Sequence[str],
    titles: Optional[Sequence[str]] = None,
    color: str = "clusters",
    basis: str = "umap",
    show: bool = True,
    save: bool = False,
    save_path: Optional[str] = None,
    figsize_per_panel: tuple = (6, 5),
) -> plt.Figure:
    """
    Side-by-side stream plots comparing multiple velocity fields.

    Parameters
    ----------
    adata
        Annotated data matrix.
    velocity_keys
        List of keys in adata.obsm with velocity fields to compare.
    titles
        Optional titles for each panel.

    Returns
    -------
    matplotlib Figure.

    Example
    -------
    ::

        # Compare raw vs smoothed velocity
        velot.pl.velocity_comparison(
            adata,
            velocity_keys=["velot_velocity_raw_umap", "velocity_umap"],
            titles=["Raw OT", "Smoothed"],
        )
    """
    if not _HAS_SCVELO:
        raise ImportError(
            "velocity_comparison requires scVelo for stream plots. "
            "Install with: pip install scvelo"
        )

    n = len(velocity_keys)
    titles = titles or velocity_keys

    figsize = (figsize_per_panel[0] * n, figsize_per_panel[1])
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    backup = adata.obsm.get("velocity_umap", None)

    for i, (vkey, title) in enumerate(zip(velocity_keys, titles)):
        if vkey not in adata.obsm:
            axes[i].text(
                0.5, 0.5, f"'{vkey}' not found",
                ha="center", va="center", transform=axes[i].transAxes,
            )
            continue

        adata.obsm["velocity_umap"] = adata.obsm[vkey]
        scv.pl.velocity_embedding_stream(
            adata, basis=basis, color=color,
            title=title, ax=axes[i], show=False,
        )

    # Restore original
    if backup is not None:
        adata.obsm["velocity_umap"] = backup

    plt.tight_layout()
    _finish(fig, show=show, save=save, save_path=save_path)
    return fig if not show else None


# =====================================================================
# Internal helpers
# =====================================================================


def _finish(fig: plt.Figure, show: bool, save: bool, save_path: Optional[str]):
    """Handle show/save logic for all plot functions."""
    if save:
        # Fallback if they say save=True but forget to give a path
        if save_path is None:
            save_path = "velot_figure.png" 
            print(f"Warning: save=True but no save_path provided. Saving to {save_path}")
            
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    
    plt.close()


def gridsearch_results(
    df,
    metric: str = "iccoh_mean",
    param_x: Optional[str] = None,
    param_hue: Optional[str] = None,
    show: bool = True,
    save: bool = False,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Visualize grid search results.

    Parameters
    ----------
    df
        DataFrame returned by ``velot.tl.gridsearch()``.
    metric
        Column to plot as the y-axis.
    param_x
        Parameter for x-axis grouping. If None, auto-detected.
    param_hue
        Parameter for color grouping. If None, auto-detected.
    show
        Whether to display.
    save
        File path to save.

    Returns
    -------
    matplotlib Figure.

    Example
    -------
    ::

        results = velot.tl.gridsearch(adata, param_grid, ...)
        velot.pl.gridsearch_results(results, metric="cbdir_mean")
    """
    # Auto-detect parameters (columns that aren't metrics)
    non_param_cols = {
        "iccoh_mean", "cbdir_mean", "elapsed_seconds", "status",
    }
    param_cols = [
        c for c in df.columns
        if c not in non_param_cols and not c.startswith(("iccoh_", "cbdir_"))
    ]

    if not param_cols:
        raise ValueError("No parameter columns detected in DataFrame.")

    if param_x is None:
        param_x = param_cols[0]
    if param_hue is None and len(param_cols) > 1:
        param_hue = param_cols[1]

    # Filter to successful runs
    df_ok = df[df["status"] == "ok"].copy()
    if len(df_ok) == 0:
        raise ValueError("No successful runs to plot.")

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Panel 1: metric vs param_x, colored by param_hue
    ax = axes[0]
    if param_hue is not None:
        hue_values = sorted(df_ok[param_hue].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(hue_values)))

        for hue_val, color in zip(hue_values, colors):
            mask = df_ok[param_hue] == hue_val
            subset = df_ok[mask].sort_values(param_x)
            ax.plot(
                subset[param_x].astype(str), subset[metric],
                "o-", label=f"{param_hue}={hue_val}",
                color=color, markersize=8, alpha=0.8,
            )
        ax.legend(fontsize=8)
    else:
        df_sorted = df_ok.sort_values(param_x)
        ax.plot(
            df_sorted[param_x].astype(str), df_sorted[metric],
            "o-", color="tab:blue", markersize=8,
        )

    ax.set_xlabel(param_x)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs {param_x}")
    ax.tick_params(axis="x", rotation=45)

    # Panel 2: summary bar chart of top 10 runs
    ax = axes[1]
    top = df_ok.nlargest(min(10, len(df_ok)), metric)

    labels = []
    for _, row in top.iterrows():
        parts = [f"{p}={row[p]}" for p in param_cols]
        labels.append("\n".join(parts))

    bars = ax.barh(
        range(len(top)), top[metric].values,
        color="tab:green", alpha=0.7,
    )
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(metric)
    ax.set_title(f"Top {len(top)} configurations")
    ax.invert_yaxis()

    fig.suptitle("VelOT Grid Search Results", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _finish(fig, show=show, save=save, save_path=save_path)
    return fig if not show else None

def trajectories(
    adata: AnnData,
    trajectory_ids: Optional[Sequence[int]] = None,
    color: str = "clusters",
    basis: str = "umap",
    line_color: str = "black",
    line_alpha: float = 0.7,
    line_width: float = 1,
    line_style: str = "--",
    start_marker: str = "o",
    start_size: int = 50,
    start_color: str = "gray",
    end_marker: str = "o",
    end_size: int = 50,
    end_color: str = "black",
    arrow_frequency: int = 30,
    arrow_size: float = 10,
    show_start: bool = True,
    show_end: bool = True,
    show_arrows: bool = True,
    frameon: bool = False,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    save: bool = False,
    save_path: Optional[str] = None,
    figsize: tuple = (5,5),
    **scanpy_kwargs,
) -> plt.Axes:
    """
    Plot cell trajectories on top of a scanpy embedding plot.

    Parameters
    ----------
    adata
        Must have ``adata.uns['velot_trajectories']``.
    trajectory_ids
        List of trajectory IDs to plot. If None, plots all.
        IDs are stored in each trajectory's metadata ``"id"`` field.
        Use this to select specific trajectories after inspecting
        the metadata.
    color
        Column passed to ``sc.pl.embedding``.
    basis
        Embedding basis.
    line_color
        Color of trajectory lines.
    line_alpha
        Line transparency.
    line_width
        Line width.
    line_style
        ``"-"`` solid, ``"--"`` dashed, ``":"`` dotted.
    start_marker, start_size, start_color
        Start point appearance.
    end_marker, end_size, end_color
        End point appearance.
    arrow_frequency
        Arrow every N steps.
    arrow_size
        Arrowhead size.
    show_start, show_end, show_arrows
        Toggle visual elements.
    frameon
        Show axes frame.
    title
        Plot title.
    ax
        Pre-existing axes. If None, creates new figure with scatter
        background. If provided, draws scatter background and
        trajectories on it — useful for subplot grids.
    show
        Display the plot. Ignored when ``ax`` is provided
        (caller controls display).
    save
        Save path. Ignored when ``ax`` is provided.
    **scanpy_kwargs
        Passed to ``scv.pl.scatter`` for the background.

    Returns
    -------
    matplotlib Axes.

    Examples
    --------
    Plot all trajectories::

        velot.pl.trajectories(adata)

    Plot specific trajectories by ID::

        velot.pl.trajectories(adata, trajectory_ids=[0, 3, 7])

    Grid of individual trajectories::

        n = 10
        cols = 5
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        for i in range(n):
            velot.pl.trajectories(
                adata, trajectory_ids=[i], ax=axes.flat[i],
                show=False, title=f"Trajectory {i}",
            )
        for j in range(n, len(axes.flat)):
            axes.flat[j].axis("off")
        plt.tight_layout()
        plt.show()

    Inspect metadata to choose which to plot::

        meta = adata.uns["velot_trajectories"]["metadata"]
        for m in meta:
            print(f"ID {m['id']}: {m['origin_cluster']} → "
                  f"{m['terminal_cluster']} ({m['n_steps']} steps)")
        velot.pl.trajectories(adata, trajectory_ids=[2, 5])
    """
    if "velot_trajectories" not in adata.uns:
        raise ValueError(
            "No trajectories found. Run velot.tl.compute_trajectories() first."
        )

    traj_data = adata.uns["velot_trajectories"]
    all_metadata = traj_data["metadata"]

    if basis == "umap" and len(traj_data.get("paths_umap", [])) > 0:
        all_paths = traj_data["paths_umap"]
    elif basis == "pca" and len(traj_data.get("paths_pca", [])) > 0:
        all_paths = [p[:, :2] for p in traj_data["paths_pca"]]
    else:
        raise ValueError(
            f"No trajectory paths available for basis='{basis}'. "
            f"Available: paths_umap={len(traj_data.get('paths_umap', []))}, "
            f"paths_pca={len(traj_data.get('paths_pca', []))}"
        )

    if not all_paths:
        raise ValueError("No UMAP-projected trajectories available.")

    # Filter by trajectory_ids if specified
    if trajectory_ids is not None:
        id_set = set(trajectory_ids)
        selected = [
            (path, meta)
            for path, meta in zip(all_paths, all_metadata)
            if meta["id"] in id_set
        ]
        if not selected:
            available_ids = [m["id"] for m in all_metadata]
            raise ValueError(
                f"No trajectories found with IDs {trajectory_ids}. "
                f"Available IDs: {available_ids}"
            )
        paths = [s[0] for s in selected]
        metadata = [s[1] for s in selected]
    else:
        paths = all_paths
        metadata = all_metadata

    # ------------------------------------------------------------------
    # Background scatter — always drawn, whether ax is provided or not
    # ------------------------------------------------------------------
    owns_figure = ax is None

    if owns_figure:
        fig, ax = plt.subplots(figsize=figsize)

    # Draw scatter on the axes (provided or newly created)
    try:
        scv.pl.scatter(
            adata,
            basis=basis,
            color=color,
            ax=ax,
            show=False,
            frameon=frameon,
            legend_loc="on data",
            **scanpy_kwargs,
        )
    except Exception:
        print("Trajectories on manual scatter...")
        # Fallback: manual scatter
        coords_key = f"X_{basis}"
        if coords_key in adata.obsm:
            coords = adata.obsm[coords_key][:, :2]
            if color in adata.obs:
                cats = adata.obs[color].astype("category")
                ax.scatter(
                    coords[:, 0], coords[:, 1],
                    s=10, c=cats.cat.codes.values,
                    cmap="tab20", alpha=0.5, zorder=1,
                )
            else:
                ax.scatter(
                    coords[:, 0], coords[:, 1],
                    s=10, c="lightgray", alpha=0.5, zorder=1,
                )

    fig = ax.get_figure()

    # ------------------------------------------------------------------
    # Overlay trajectories
    # ------------------------------------------------------------------
    for traj_i, (path, meta) in enumerate(zip(paths, metadata)):
        if len(path) < 2:
            continue

        is_backward = meta["direction"] == "backward"

        if is_backward:
            display_path = path[::-1]
        else:
            display_path = path

        # Trajectory line
        ax.plot(
            display_path[:, 0], display_path[:, 1],
            color=line_color, alpha=line_alpha,
            linewidth=line_width, linestyle=line_style,
            zorder=10,
        )

        # Start point
        if show_start:
            ax.scatter(
                display_path[0, 0], display_path[0, 1],
                s=start_size, c=start_color,
                marker=start_marker,
                edgecolors="black", linewidths=0.5,
                zorder=12,
            )

        # End point
        if show_end:
            ax.scatter(
                display_path[-1, 0], display_path[-1, 1],
                s=end_size, c=end_color,
                marker=end_marker,
                edgecolors="white", linewidths=0.5,
                zorder=12,
            )

        # Direction arrows
        if show_arrows:
            for step in range(
                arrow_frequency, len(display_path) - 1, arrow_frequency
            ):
                dx = display_path[step + 1, 0] - display_path[step, 0]
                dy = display_path[step + 1, 1] - display_path[step, 1]
                mag = np.sqrt(dx**2 + dy**2)
                if mag > 1e-10:
                    ax.annotate(
                        "",
                        xy=(
                            display_path[step, 0] + dx * 0.5,
                            display_path[step, 1] + dy * 0.5,
                        ),
                        xytext=(
                            display_path[step, 0],
                            display_path[step, 1],
                        ),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color=line_color,
                            lw=line_width * 0.6,
                            alpha=line_alpha,
                            mutation_scale=arrow_size,
                        ),
                        zorder=11,
                    )

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    direction = traj_data.get("direction", "forward")
    start = traj_data.get("start_cluster", "?")
    target = traj_data.get("target_cluster", None)
    n_shown = len(paths)

    if title is None:
        title = ""
    if title == "auto":
        parts = []
        if direction == "backward":
            parts.append(f"Backward from '{start}'")
        else:
            parts.append(f"Forward from '{start}'")
        if target:
            parts.append(f"→ '{target}'")
        parts.append(f"(n={n_shown})")
        title = " ".join(parts)

    ax.set_title(title, fontsize=12)
    add_umap_axis(ax, basis=basis)

    # ------------------------------------------------------------------
    # Finish — only when we own the figure
    # ------------------------------------------------------------------
    if owns_figure:
        plt.tight_layout()
        _finish(fig, show=show, save=save, save_path=save_path)

    return ax


def fate_summary(
    adata: AnnData,
    show: bool = True,
    save: bool = False,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 4),
) -> plt.Figure:
    """
    Bar chart summarizing trajectory fate or origin.

    For forward trajectories: shows terminal cluster distribution.
    For backward trajectories: shows origin cluster distribution.

    Returns
    -------
    matplotlib Figure.
    """
    if "velot_trajectories" not in adata.uns:
        raise ValueError(
            "No trajectories found. Run velot.tl.compute_trajectories() first."
        )

    traj_data = adata.uns["velot_trajectories"]
    metadata = traj_data["metadata"]
    start_cluster = traj_data.get("start_cluster", "unknown")
    direction = traj_data.get("direction", "forward")

    # Separate by direction if "both"
    groups = {}
    for m in metadata:
        d = m["direction"]
        if d not in groups:
            groups[d] = []
        groups[d].append(m)

    n_panels = len(groups)
    fig, axes = plt.subplots(1, n_panels, figsize=(figsize[0] * n_panels / 2, figsize[1]))
    if n_panels == 1:
        axes = [axes]

    for ax, (dir_name, dir_meta) in zip(axes, groups.items()):
        if dir_name == "forward":
            values = [m["terminal_cluster"] for m in dir_meta]
            ylabel = "Fraction of trajectories"
            subtitle = f"Forward from '{start_cluster}': terminal fates"
        else:
            values = [m["origin_cluster"] for m in dir_meta]
            ylabel = "Fraction of trajectories"
            subtitle = f"Backward from '{start_cluster}': traced origins"

        unique, counts = np.unique(values, return_counts=True)
        fractions = counts / counts.sum()

        order = np.argsort(-fractions)
        unique = unique[order]
        fractions = fractions[order]
        counts = counts[order]

        bars = ax.bar(
            range(len(unique)), fractions,
            color=plt.cm.tab20(np.linspace(0, 1, len(unique))),
            edgecolor="black", linewidth=0.5,
        )

        ax.set_xticks(range(len(unique)))
        ax.set_xticklabels(unique, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle, fontsize=11)

        for bar, count, frac in zip(bars, counts, fractions):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{count} ({frac:.0%})",
                ha="center", va="bottom", fontsize=8,
            )

        ax.set_ylim(0, min(1.0, fractions.max() * 1.3))

    fig.suptitle("VelOT trajectory fate analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _finish(fig, show=show, save=save, save_path=save_path)
    return fig if not show else None


def flow_simulation(
    adata: AnnData,
    color: str = "clusters",
    basis: str = "umap",
    line_alpha: float = 0.3,
    line_width: float = 0.8,
    colorby: str = "pseudotime",
    show_cells: bool = True,
    cell_alpha: float = 0.2,
    cell_size: int = 8,
    show_start: bool = True,
    show_end: bool = True,
    frameon: bool = False,
    title: Optional[str] = None,
    show: bool = True,
    save: bool = False,
    save_path: Optional[str] = None,
    figsize: tuple = (9, 8),
) -> plt.Figure:
    """
    Visualize the flow simulation — particles flowing through
    the velocity field like water through a river.

    Parameters
    ----------
    adata
        Must have ``adata.uns['velot_flow']`` from
        ``velot.tl.simulate_flow()``.
    color
        Column for background cell coloring.
    basis
        Embedding basis.
    colorby
        How to color trajectories:
          - ``"pseudotime"``: color by pseudotime along trajectory
            (blue=early, red=late)
          - ``"cluster"``: color by starting cluster
          - ``"fate"``: color by terminal cluster
    show_cells
        Show background cells.
    cell_alpha
        Background cell transparency.
    cell_size
        Background cell size.
    show_start
        Mark starting positions with stars.
    show_end
        Mark ending positions with dots.
    frameon
        Show axes frame.
    title
        Plot title.
    show
        Display the plot.
    save
        File path to save.

    Returns
    -------
    matplotlib Figure.
    """
    if "velot_flow" not in adata.uns:
        raise ValueError(
            "No flow data. Run velot.tl.simulate_flow() first."
        )

    flow = adata.uns["velot_flow"]
    paths = flow["particles_umap"]
    pt_along = flow["pseudotime_along"]
    cl_along = flow["cluster_along"]

    if not paths:
        raise ValueError("No UMAP-projected flow paths available.")

    coords_key = f"X_{basis}"
    coords = adata.obsm[coords_key]

    fig, ax = plt.subplots(figsize=figsize)

    # Background cells
    if show_cells and color in adata.obs:
        categories = adata.obs[color].astype("category")
        cat_names = categories.cat.categories.tolist()
        codes = categories.cat.codes.values
        cmap_bg = plt.cm.tab20
        colors_array = cmap_bg(np.linspace(0, 1, len(cat_names)))

        for ci, cat_name in enumerate(cat_names):
            mask = codes == ci
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                s=cell_size, c=[colors_array[ci]],
                alpha=cell_alpha, label=cat_name, zorder=1,
            )
    elif show_cells:
        ax.scatter(
            coords[:, 0], coords[:, 1],
            s=cell_size, c="lightgray", alpha=cell_alpha, zorder=1,
        )

    # Draw particle trajectories
    for p_idx, path in enumerate(paths):
        if len(path) < 2:
            continue

        if colorby == "pseudotime":
            # Color each segment by local pseudotime
            pts = pt_along[p_idx]
            for step in range(len(path) - 1):
                pt_val = pts[min(step, len(pts) - 1)]
                seg_color = plt.cm.coolwarm(pt_val)
                ax.plot(
                    path[step:step+2, 0], path[step:step+2, 1],
                    color=seg_color, alpha=line_alpha,
                    linewidth=line_width, zorder=2,
                )

        elif colorby == "fate":
            terminal = cl_along[p_idx][-1]
            if color in adata.obs and terminal in cat_names:
                lc = colors_array[cat_names.index(terminal)]
            else:
                lc = "gray"
            ax.plot(
                path[:, 0], path[:, 1],
                color=lc, alpha=line_alpha,
                linewidth=line_width, zorder=2,
            )

        else:
            ax.plot(
                path[:, 0], path[:, 1],
                color="steelblue", alpha=line_alpha,
                linewidth=line_width, zorder=2,
            )

        # Start marker
        if show_start:
            ax.scatter(
                path[0, 0], path[0, 1],
                s=30, c="blue", marker="*",
                edgecolors="white", linewidths=0.3,
                alpha=0.7, zorder=4,
            )

        # End marker
        if show_end:
            ax.scatter(
                path[-1, 0], path[-1, 1],
                s=15, c="red", marker="o",
                edgecolors="white", linewidths=0.3,
                alpha=0.7, zorder=4,
            )

    # Legend for background
    if show_cells and color in adata.obs:
        ax.legend(fontsize=7, loc="best", markerscale=2, framealpha=0.8)

    # Title with summary
    meta = flow["metadata"]
    source = flow["source"]
    n = flow["n_particles"]
    if title is None:
        title = (
            f"Flow simulation: {n} particles from {source}\n"
            f"Mean final τ = {meta['mean_final_pseudotime']:.2f}, "
            f"mean {meta['mean_trajectory_length']:.0f} steps"
        )

    ax.set_title(title, fontsize=11)

    if not frameon:
        ax.axis("off")

    plt.tight_layout()
    _finish(fig, show=show, save=save, save_path=save_path)
    return fig if not show else None


def metric_summary(
    metrics: dict,
    orientation: str = "horizontal",
    layout: str = "row",
    figsize: Optional[tuple] = None,
    palette_name: str = "Set2",
    median_color: str = "black",
    frameon: bool = True,
    show: bool = True,
    save: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Box plots of ICCoh and CBDir metrics from a precomputed summary.

    Parameters
    ----------
    metrics
        Dictionary returned by ``velot.metrics.summary()``.
    orientation
        ``"horizontal"`` or ``"vertical"`` box plots.
    layout
        ``"row"`` or ``"column"``.
    figsize
        Custom figure size.
    palette_name
        Seaborn color palette name. First color for ICCoh,
        second for CBDir.
    median_color
        Median line color.
    frameon
        Show axes frame.
    show
        Display the figure.
    save
        File path to save.

    Returns
    -------
    matplotlib Figure.
    """
    import seaborn as sns
    from matplotlib.colors import ListedColormap

    if "iccoh" not in metrics:
        raise ValueError(
            "metrics dict must contain 'iccoh'. "
            "Pass the output of velot.metrics.summary()."
        )

    palette = sns.color_palette(palette_name, 8)
    color_iccoh = palette[0]
    color_cbdir = palette[1]

    iccoh = metrics["iccoh"]
    sample_val = next(iter(iccoh.values()))
    iccoh_is_raw = isinstance(sample_val, (list, np.ndarray))

    has_cbdir = "cbdir" in metrics and len(metrics["cbdir"]) > 0
    n_panels = 2 if has_cbdir else 1

    if layout == "row":
        nrows, ncols = 1, n_panels
        default_figsize = (6 * n_panels, 5)
    else:
        nrows, ncols = n_panels, 1
        default_figsize = (7, 5 * n_panels)

    figsize = figsize or default_figsize
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_panels == 1:
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()

    horiz = orientation == "horizontal"

    medianprops = dict(color=median_color, linewidth=2)
    whiskerprops = dict(color="black")
    capprops = dict(color="black")
    meanprops = dict(
        marker="o", markerfacecolor="white",
        markeredgecolor="black", markersize=6,
    )

    def _draw_panel(ax, data_dict, labels_str, panel_color, title, mean_val, xlabel):
        data_list = list(data_dict.values())
        positions = list(range(len(labels_str)))

        boxprops = dict(facecolor=panel_color, edgecolor="black", alpha=0.6)

        bp = ax.boxplot(
            data_list,
            positions=positions,
            vert=not horiz,
            patch_artist=True,
            showmeans=True,
            showfliers=False,
            boxprops=boxprops,
            medianprops=medianprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            meanprops=meanprops,
            widths=0.6,
        )

        if horiz:
            ax.set_yticks(positions)
            ax.set_yticklabels(labels_str, fontsize=12)
            ax.set_xlim(-1.05, 1.05)
            ax.set_xlabel(xlabel, fontsize=14)
            ax.axvline(0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        else:
            ax.set_xticks(positions)
            ax.set_xticklabels(labels_str, fontsize=12, rotation=45, ha="right")
            ax.set_ylim(-1.05, 1.05)
            ax.set_ylabel(xlabel, fontsize=14)
            ax.axhline(0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

        ax.set_title(title, fontsize=16) #, fontweight="bold")

        # Mean as text in upper left corner
        # ax.text(
        #     0.05, 0.95,
        #     f"mean = {mean_val:.3f}",
        #     transform=ax.transAxes,
        #     fontsize=10, fontweight="bold",
        #     va="top", ha="left",
        #     bbox=dict(
        #         boxstyle="round,pad=0.3",
        #         facecolor=panel_color, alpha=0.3,
        #         edgecolor="black", linewidth=0.5,
        #     ),
        # )
        ax.text(
            0.05, 0.95,
            f"mean = {mean_val:.2f}",
            transform=ax.transAxes,
            fontsize=14, fontweight="bold",
            va="top", ha="left",
        )

    # ------------------------------------------------------------------
    # Panel 1: ICCoh
    # ------------------------------------------------------------------
    iccoh_labels = sorted(iccoh.keys(), key=str)
    if iccoh_is_raw:
        iccoh_ordered = {k: iccoh[k] for k in iccoh_labels}
    else:
        iccoh_ordered = {k: [iccoh[k]] for k in iccoh_labels}

    iccoh_labels_str = [str(k) for k in iccoh_labels]
    iccoh_mean = metrics.get(
        "iccoh_mean",
        np.mean([np.mean(v) for v in iccoh_ordered.values() if len(v) > 0]),
    )

    _draw_panel(
        axes[0], iccoh_ordered, iccoh_labels_str,
        color_iccoh, "",
        iccoh_mean, "ICCoh score",
    )

    # ------------------------------------------------------------------
    # Panel 2: CBDir
    # ------------------------------------------------------------------
    if has_cbdir:
        cbdir = metrics["cbdir"]
        sample_cbdir_val = next(iter(cbdir.values()))
        cbdir_is_raw = isinstance(sample_cbdir_val, (list, np.ndarray))

        cbdir_labels = list(cbdir.keys())
        if cbdir_is_raw:
            cbdir_ordered = {k: cbdir[k] for k in cbdir_labels}
        else:
            cbdir_ordered = {k: [cbdir[k]] for k in cbdir_labels}

        cbdir_labels_str = []
        for k in cbdir_labels:
            if isinstance(k, tuple):
                cbdir_labels_str.append(f"{k[0]} → {k[1]}")
            else:
                cbdir_labels_str.append(str(k))

        cbdir_mean = metrics.get(
            "cbdir_mean",
            np.mean([np.mean(v) for v in cbdir_ordered.values() if len(v) > 0]),
        )

        _draw_panel(
            axes[1], cbdir_ordered, cbdir_labels_str,
            color_cbdir, "",
            cbdir_mean, "CBDir score",
        )

    # fig.suptitle("VelOT Velocity Metrics", fontsize=24, fontfamily="sans serif")
    plt.tight_layout()
    _finish(fig, show=show, save=save, save_path=save_path)
    return fig if not show else None
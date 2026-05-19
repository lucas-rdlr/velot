from scipy.stats import mannwhitneyu
import os
from typing import List
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

def _significance_str(pval):
    """Convert p-value to significance annotation string."""
    if pval is None or np.isnan(pval):
        return ""
    if pval > 0.05:
        return "ns"
    elif pval > 0.01:
        return "*"
    elif pval > 0.001:
        return "**"
    elif pval > 0.0001:
        return "***"
    else:
        return "****"


def _draw_significance(
    ax,
    model_positions,
    reference_model,
    fontsize=9,
    min_samples=5,
):
    """
    Draw significance brackets from reference_model to each other model.

    Parameters
    ----------
    ax
        Matplotlib Axes.
    model_positions
        Dict of ``{model_name: (x_position, data_array)}``.
    reference_model
        Name of the reference model to compare against.
    fontsize
        Font size for the significance text.
    min_samples
        Minimum number of data points required in both groups
        to perform the test.

    Returns
    -------
    Maximum y coordinate used by brackets (for axis limit adjustment).
    """
    if reference_model not in model_positions:
        return None

    ref_pos, ref_data = model_positions[reference_model]
    if len(ref_data) < min_samples:
        return None

    # Collect all visible data to determine bracket placement
    all_visible = []
    for _, (_, d) in model_positions.items():
        if len(d) > 0:
            # Match boxplot whiskers (showfliers=False):
            # upper whisker = min(max, Q3 + 1.5*IQR)
            q1, q3 = np.percentile(d, [25, 75])
            iqr = q3 - q1
            upper_whisker = min(np.max(d), q3 + 1.5 * iqr)
            all_visible.append(upper_whisker)

    if not all_visible:
        return None

    y_top = max(all_visible)
    y_min = ax.get_ylim()[0]
    y_range = y_top - y_min if y_top > y_min else 1.0

    h = y_range * 0.03       # bracket tick height
    gap = y_range * 0.06     # vertical gap between stacked brackets
    margin = y_range * 0.08  # initial margin above data

    max_y = y_top
    bracket_idx = 0

    # Sort other models by position so brackets don't cross
    others = sorted(
        [
            (model, pos, data)
            for model, (pos, data) in model_positions.items()
            if model != reference_model and len(data) >= min_samples
        ],
        key=lambda x: abs(x[1] - ref_pos),
    )

    for model, other_pos, other_data in others:
        # Mann-Whitney U test (two-sided)
        try:
            _, pval = mannwhitneyu(
                ref_data, other_data, alternative="two-sided"
            )
        except ValueError:
            continue

        sig_str = _significance_str(pval)

        # Bracket y position
        y_bar = y_top + margin + bracket_idx * gap

        # Draw bracket: two ticks and a horizontal bar
        left = min(ref_pos, other_pos)
        right = max(ref_pos, other_pos)

        ax.plot(
            [left, left, right, right],
            [y_bar - h, y_bar, y_bar, y_bar - h],
            lw=0.8,
            color="black",
            clip_on=False,
        )

        # Significance text
        color = "black" if sig_str == "ns" else "black"
        weight = "normal" if sig_str == "ns" else "bold"
        ax.text(
            (left + right) / 2,
            y_bar + h * 0.3,
            sig_str,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=color,
            fontweight=weight,
        )

        max_y = max(max_y, y_bar + h * 2)
        bracket_idx += 1

    # Expand y-axis to fit brackets
    if bracket_idx > 0:
        current_ylim = ax.get_ylim()
        ax.set_ylim(current_ylim[0], max_y + y_range * 0.05)

    return max_y

def _plot_aggregated(ax, df_m, all_models, model_colors, reference_model=None):
    """
    One boxplot per model, pooling all groups and datasets.
    """
    model_positions = {}

    for m_idx, model in enumerate(all_models):
        data = df_m.loc[df_m["model"] == model, "value"].dropna().values

        if len(data) == 0:
            continue

        bp = ax.boxplot(
            [data],
            positions=[m_idx],
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.5),
        )
        bp["boxes"][0].set_facecolor(model_colors[model])
        bp["boxes"][0].set_alpha(0.7)

        _overlay_points(ax, data, m_idx, model_colors[model])
        model_positions[model] = (m_idx, data)

    ax.set_xticks(range(len(all_models)))
    ax.set_xticklabels(all_models, rotation=30, ha="right", fontsize=9)

    # Significance brackets
    if reference_model is not None:
        _draw_significance(ax, model_positions, reference_model, fontsize=10)


def _plot_per_dataset(ax, df_m, all_models, model_colors, reference_model=None):
    """
    Grouped by dataset, one boxplot per model within each group.
    """
    dataset_list = df_m["dataset"].unique()
    n_models = len(all_models)
    group_width = n_models + 1.5

    tick_positions = []
    tick_labels = []

    # Collect positions per dataset for significance testing
    dataset_model_positions = {ds: {} for ds in dataset_list}

    for d_idx, dataset in enumerate(dataset_list):
        base_pos = d_idx * group_width
        df_d = df_m[df_m["dataset"] == dataset]

        for m_idx, model in enumerate(all_models):
            data = df_d.loc[
                df_d["model"] == model, "value"
            ].dropna().values

            if len(data) == 0:
                continue

            pos = base_pos + m_idx
            bp = ax.boxplot(
                [data],
                positions=[pos],
                widths=0.6,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="black", linewidth=1.5),
            )
            bp["boxes"][0].set_facecolor(model_colors[model])
            bp["boxes"][0].set_alpha(0.7)

            _overlay_points(ax, data, pos, model_colors[model])
            dataset_model_positions[dataset][model] = (pos, data)

        tick_positions.append(base_pos + (n_models - 1) / 2)
        tick_labels.append(dataset)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=9)

    # Vertical separators between datasets
    for d_idx in range(1, len(dataset_list)):
        sep_x = d_idx * group_width - group_width / 2 + (n_models - 1) / 2
        ax.axvline(
            sep_x, color="gray", linewidth=0.5, linestyle="--", alpha=0.5
        )

    # Significance brackets per dataset
    if reference_model is not None:
        for dataset in dataset_list:
            _draw_significance(
                ax,
                dataset_model_positions[dataset],
                reference_model,
                fontsize=9,
            )


def _plot_per_group(ax, df_m, all_models, model_colors, reference_model=None):
    """
    Full detail: one boxplot per (model, group), organized by dataset.
    """
    dataset_list = df_m["dataset"].unique()
    n_models = len(all_models)
    model_width = 1.0
    group_gap = 1.5
    dataset_gap = 3.0

    current_pos = 0.0
    tick_positions = []
    tick_labels = []
    dataset_centers = []
    dataset_boundaries = []

    # Collect positions per (dataset, group) for significance
    group_model_positions = {}

    for d_idx, dataset in enumerate(dataset_list):
        df_d = df_m[df_m["dataset"] == dataset]
        groups = df_d["group"].unique()

        if d_idx > 0:
            dataset_boundaries.append(current_pos - dataset_gap / 2)
            current_pos += dataset_gap

        dataset_start = current_pos

        for g_idx, group in enumerate(groups):
            if g_idx > 0:
                current_pos += group_gap

            group_start = current_pos
            group_key = (dataset, group)
            group_model_positions[group_key] = {}

            for m_idx, model in enumerate(all_models):
                data = df_d.loc[
                    (df_d["group"] == group) & (df_d["model"] == model),
                    "value",
                ].dropna().values

                pos = current_pos

                if len(data) > 0:
                    bp = ax.boxplot(
                        [data],
                        positions=[pos],
                        widths=0.7,
                        patch_artist=True,
                        showfliers=False,
                        medianprops=dict(color="black", linewidth=1.5),
                    )
                    bp["boxes"][0].set_facecolor(model_colors[model])
                    bp["boxes"][0].set_alpha(0.7)

                    group_model_positions[group_key][model] = (pos, data)

                current_pos += model_width

            # Group label
            group_center = (group_start + current_pos - model_width) / 2
            label = group
            if " → " in label:
                parts = label.split(" → ")
                label = "→".join(parts)
            tick_positions.append(group_center)
            tick_labels.append(label)

        dataset_end = current_pos - model_width
        dataset_centers.append((dataset_start + dataset_end) / 2)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)

    # Dataset separators
    for boundary in dataset_boundaries:
        ax.axvline(
            boundary, color="black", linewidth=1.0, linestyle="-", alpha=0.4
        )

    # Dataset names
    y_min, y_max = ax.get_ylim()
    for center, dataset in zip(dataset_centers, dataset_list):
        ax.text(
            center,
            y_min - (y_max - y_min) * 0.15,
            dataset,
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            transform=ax.transData,
        )

    # Significance brackets per group
    if reference_model is not None:
        for group_key, model_pos in group_model_positions.items():
            _draw_significance(
                ax,
                model_pos,
                reference_model,
                fontsize=7,
            )

def benchmark_comparison(
    output_dir: str = "benchmark_results",
    models: Optional[List[str]] = None,
    models_order: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    detail: str = "aggregated",
    reference_model: Optional[str] = "velot",
    show_significance: bool = True,
    show_timing: bool = True,
    figsize_per_panel: tuple = (5, 4),
    show: bool = True,
    save: Optional[str] = None,
) -> plt.Figure:
    """
    Compare benchmark results across models and datasets.

    Parameters
    ----------
    output_dir
        Directory with saved JSON benchmark files.
    models
        Which models to include. None for all.
    datasets
        Which datasets to include. None for all.
    metrics
        Which metrics to plot (e.g., ``["cbdir", "iccoh"]``).
        None to auto-detect all metrics with per-cell data.
    detail
        Level of detail for the metric panels:

        - ``"aggregated"`` (default): one boxplot per model, pooling
          all groups and all datasets.
        - ``"per_dataset"``: one boxplot per (model, dataset),
          grouped by dataset, colored by model.
        - ``"per_group"``: one boxplot per (model, dataset, group),
          every edge/cluster visible, organized by dataset.
    reference_model
        Model to compare others against for significance testing.
        Set to None to disable significance brackets entirely.
    show_significance
        Whether to draw significance brackets. Only applies when
        ``reference_model`` is not None.
    show_timing
        Whether to include a timing comparison panel.
    figsize_per_panel
        Size of each subplot.
    show
        Display the plot.
    save
        Path to save.

    Returns
    -------
    matplotlib Figure.

    Examples
    --------
    Default — pooled, with significance vs velot::

        velot.pl.benchmark_comparison("benchmark_results")

    Without significance::

        velot.pl.benchmark_comparison(
            "benchmark_results", show_significance=False,
        )

    Compare against a different reference::

        velot.pl.benchmark_comparison(
            "benchmark_results", reference_model="scvelo_dynamical",
        )

    Per dataset::

        velot.pl.benchmark_comparison(
            "benchmark_results", detail="per_dataset",
        )

    Full detail::

        velot.pl.benchmark_comparison(
            "benchmark_results", detail="per_group",
            figsize_per_panel=(12, 4),
        )
    """
    from velot.benchmark import load_benchmarks, load_benchmarks_per_group

    if detail not in ("aggregated", "per_dataset", "per_group"):
        raise ValueError(
            f"detail must be 'aggregated', 'per_dataset', or "
            f"'per_group', got '{detail}'."
        )

    df_summary = load_benchmarks(output_dir, models, datasets)
    df_cells = load_benchmarks_per_group(output_dir, models, datasets)

    if len(df_summary) == 0:
        raise ValueError(f"No benchmark results found in '{output_dir}'.")

    # Resolve reference model
    ref = None
    if show_significance and reference_model is not None:
        available_models = df_summary["model"].unique()
        if reference_model in available_models:
            ref = reference_model
        else:
            import warnings
            warnings.warn(
                f"Reference model '{reference_model}' not found in results. "
                f"Available: {list(available_models)}. "
                f"Significance brackets disabled.",
                UserWarning,
            )

    # Auto-detect metrics
    if metrics is None:
        if len(df_cells) > 0:
            metrics = sorted(df_cells["metric"].unique().tolist())
        else:
            metrics = []

    n_metric_panels = len(metrics)
    n_panels = n_metric_panels + (1 if show_timing else 0)

    if n_panels == 0:
        raise ValueError("No metrics or timing data to plot.")

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(figsize_per_panel[0] * n_panels, figsize_per_panel[1]),
        squeeze=False,
    )
    axes = axes.flatten()

        # Color palette for models
    if models_order is not None:
        all_models = models_order
    else:
        all_models = df_summary["model"].unique()
    n_models = len(all_models)
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_models, 1)))
    model_colors = dict(zip(all_models, colors))

    # ── Metric panels ───────────────────────────────────────────
    for i, metric_name in enumerate(metrics):
        ax = axes[i]
        df_m = df_cells[df_cells["metric"] == metric_name]

        if len(df_m) == 0:
            ax.set_title(f"{metric_name}\n(no data)")
            continue

        if detail == "aggregated":
            _plot_aggregated(ax, df_m, all_models, model_colors, ref)
        elif detail == "per_dataset":
            _plot_per_dataset(ax, df_m, all_models, model_colors, ref)
        elif detail == "per_group":
            _plot_per_group(ax, df_m, all_models, model_colors, ref)

        ax.set_title(metric_name.upper(), fontsize=12, fontweight="bold")
        ax.set_ylabel(metric_name)
        ax.grid(axis="y", alpha=0.3)

    # ── Timing ──────────────────────────────────────────────────
    if show_timing and "time_total" in df_summary.columns:
        ax = axes[n_metric_panels]
        _plot_timing(ax, df_summary, all_models, model_colors)

    # ── Legend ──────────────────────────────────────────────────
    handles = [
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor=model_colors[m],
            edgecolor="black",
            alpha=0.7,
        )
        for m in all_models
    ]
    labels = list(all_models)

    # Mark reference model in legend
    if ref is not None:
        labels = [
            f"{m} (ref)" if m == ref else m for m in all_models
        ]

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(len(all_models), 6),
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig

def _plot_timing(ax, df_summary, all_models, model_colors):
    """Grouped bar chart of execution time."""
    dataset_list = df_summary["dataset"].unique()
    x = np.arange(len(dataset_list))
    n_models = len(all_models)
    bar_width = 0.8 / max(n_models, 1)

    for m_idx, model in enumerate(all_models):
        times = []
        for dataset in dataset_list:
            mask = (
                (df_summary["model"] == model)
                & (df_summary["dataset"] == dataset)
            )
            t = df_summary.loc[mask, "time_total"].values
            times.append(t[0] if len(t) > 0 else 0)

        offset = (m_idx - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            times,
            bar_width * 0.9,
            label=model,
            color=model_colors[model],
            edgecolor="black",
            linewidth=0.5,
        )

        for bar, t in zip(bars, times):
            if t > 0:
                label = f"{t:.1f}s" if t < 60 else f"{t / 60:.1f}m"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    label,
                    ha="center", va="bottom",
                    fontsize=12, rotation=0,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        dataset_list, rotation=30, ha="right", fontsize=12
    )
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Execution time", fontsize=12, fontweight="bold")

    # Log scale if there's a large spread
    times_all = df_summary["time_total"].dropna()
    if len(times_all) > 1 and times_all.max() / max(times_all.min(), 0.1) > 10:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)


def _overlay_points(ax, data, position, color, max_points=200):
    """Overlay jittered individual points on a boxplot."""
    n_show = min(len(data), max_points)
    if n_show < len(data):
        show_data = np.random.RandomState(42).choice(
            data, n_show, replace=False
        )
    else:
        show_data = data

    jitter = np.random.RandomState(42).normal(0, 0.08, len(show_data))
    ax.scatter(
        np.full(len(show_data), position) + jitter,
        show_data,
        s=8,
        c=[color],
        edgecolors="black",
        linewidths=0.2,
        zorder=5,
        alpha=0.5,
    )
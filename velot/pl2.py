from scipy.stats import mannwhitneyu
import os
from typing import List, Dict
import numpy as np
import pandas as pd
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


def _plot_per_dataset(ax, df_m, all_models, model_colors, datasets_order=None, reference_model=None):
    """
    Grouped by dataset, one boxplot per model within each group.
    """
    if datasets_order is not None:
        dataset_list = datasets_order
    else:
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
    datasets_order: Optional[List[str]] = None,
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

    if models_order is not None:
        all_models = models_order
    else:
        all_models = df_summary["model"].unique()
    n_models = len(all_models)

    cmap = plt.get_cmap("Set2")
    colors = [
        cmap(0),
        cmap(1),
        cmap(2),
        cmap(3),
        cmap(5)
    ]
    # colors = plt.cm.Set2(np.linspace(0, 1, max(n_models, 1)))
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
            _plot_per_dataset(ax, df_m, all_models, model_colors, datasets_order, ref)
        elif detail == "per_group":
            _plot_per_group(ax, df_m, all_models, model_colors, ref)

        ax.set_title(metric_name.upper(), fontsize=12, fontweight="bold")
        ax.set_ylabel(metric_name)
        ax.grid(axis="y", alpha=0.3)

    # ── Timing ──────────────────────────────────────────────────
    if show_timing and "time_total" in df_summary.columns:
        ax = axes[n_metric_panels]
        _plot_timing(ax, df_summary, all_models, model_colors, datasets_order)

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

def _plot_timing(ax, df_summary, all_models, model_colors, datasets_order=None):
    """Grouped bar chart of execution time."""
    if datasets_order is not None:
        dataset_list = datasets_order
    else:
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
            alpha=0.9
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

def benchmark_summary_table(
    output_dir: str = "benchmark_results",
    models: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    reference_model: str = "velot",
    detail: str = "aggregated",
    save_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a summary DataFrame with statistics and p-values.

    Parameters
    ----------
    output_dir
        Directory containing JSON result files.
    models
        Filter to specific models. None for all.
    datasets
        Filter to specific datasets. None for all.
    metrics
        Which metrics to include. None for all.
    reference_model
        Model to compare others against. P-values are computed
        between this model and each other model.
    detail
        Aggregation level:

        - ``"aggregated"``: one row per (model, metric), pooling
          all datasets and groups.
        - ``"per_dataset"``: one row per (model, dataset, metric),
          pooling groups within each dataset.
        - ``"per_group"``: one row per (model, dataset, group, metric),
          no aggregation.
    save_csv
        Path to save the DataFrame as CSV. None to skip saving.

    Returns
    -------
    DataFrame with columns including mean, median, std, n, and
    p-value vs the reference model.

    Examples
    --------
    Quick overview::

        df = velot.benchmark.benchmark_summary_table("benchmark_results")
        print(df)

    Per dataset with CSV export::

        df = velot.benchmark.benchmark_summary_table(
            "benchmark_results",
            detail="per_dataset",
            save_csv="benchmark_results/summary_per_dataset.csv",
        )

    Full detail::

        df = velot.benchmark.benchmark_summary_table(
            "benchmark_results",
            detail="per_group",
        )
    """
    from scipy.stats import mannwhitneyu

    if detail not in ("aggregated", "per_dataset", "per_group"):
        raise ValueError(
            f"detail must be 'aggregated', 'per_dataset', or "
            f"'per_group', got '{detail}'."
        )
    
    from velot.benchmark import load_benchmarks, load_benchmarks_per_group

    df_cells = load_benchmarks_per_group(output_dir, models, datasets)
    df_summary = load_benchmarks(output_dir, models, datasets)

    if len(df_cells) == 0:
        raise ValueError(f"No benchmark results found in '{output_dir}'.")

    # Auto-detect metrics
    if metrics is None:
        metrics = sorted(df_cells["metric"].unique().tolist())
    else:
        df_cells = df_cells[df_cells["metric"].isin(metrics)]

    # Add timing info per (model, dataset) for inclusion in the table
    timing_map = {}
    if len(df_summary) > 0 and "time_total" in df_summary.columns:
        for _, row in df_summary.iterrows():
            timing_map[(row["model"], row["dataset"])] = {
                "time_total": row.get("time_total", np.nan),
                "time_velocity": row.get("time_velocity", np.nan),
            }

    # ── Define grouping keys ────────────────────────────────────
    if detail == "aggregated":
        group_keys = ["model", "metric"]
    elif detail == "per_dataset":
        group_keys = ["model", "dataset", "metric"]
    elif detail == "per_group":
        group_keys = ["model", "dataset", "metric", "group"]

    # ── Compute summary stats per group ─────────────────────────
    records = []

    for keys, grp in df_cells.groupby(group_keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_keys, keys))

        values = grp["value"].dropna().values
        row["mean"] = float(np.nanmean(values))
        row["median"] = float(np.nanmedian(values))
        row["std"] = float(np.nanstd(values))
        row["n"] = int(len(values))

        # Store raw values for p-value computation later
        row["_values"] = values

        records.append(row)

    df = pd.DataFrame(records)

    # ── Compute p-values vs reference ───────────────────────────
    df["pvalue"] = np.nan
    df["significance"] = ""

    if reference_model is not None and reference_model in df["model"].values:

        # Define the matching keys (everything except model)
        match_keys = [k for k in group_keys if k != "model"]

        # Build lookup for reference model data
        ref_mask = df["model"] == reference_model
        ref_rows = df[ref_mask]

        ref_lookup = {}
        for idx, row in ref_rows.iterrows():
            match_key = tuple(row[k] for k in match_keys)
            ref_lookup[match_key] = row["_values"]

        # Compute p-values for non-reference models
        for idx, row in df.iterrows():
            if row["model"] == reference_model:
                continue

            match_key = tuple(row[k] for k in match_keys)
            ref_data = ref_lookup.get(match_key)

            if ref_data is None or len(ref_data) < 5:
                continue

            other_data = row["_values"]
            if len(other_data) < 5:
                continue

            try:
                _, pval = mannwhitneyu(
                    ref_data, other_data, alternative="two-sided"
                )
                df.at[idx, "pvalue"] = pval
                df.at[idx, "significance"] = _significance_str(pval)
            except ValueError:
                continue

        # Also mark the reference rows
        df.loc[ref_mask, "significance"] = "ref"

    # ── Clean up ────────────────────────────────────────────────
    df = df.drop(columns=["_values"])

    # Add timing columns for per_dataset and per_group detail
    if detail in ("per_dataset", "per_group") and timing_map:
        df["time_total"] = df.apply(
            lambda r: timing_map.get(
                (r["model"], r.get("dataset", "")), {}
            ).get("time_total", np.nan),
            axis=1,
        )

    # For aggregated, add mean timing across datasets
    if detail == "aggregated" and timing_map:
        model_times = {}
        for (model, _), t in timing_map.items():
            model_times.setdefault(model, []).append(t["time_total"])
        df["time_total_mean"] = df["model"].map(
            {m: np.nanmean(ts) for m, ts in model_times.items()}
        )

    # Sort for readability
    sort_cols = [k for k in group_keys if k in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Reorder columns for readability
    front_cols = [k for k in group_keys if k in df.columns]
    stat_cols = ["mean", "median", "std", "n", "pvalue", "significance"]
    time_cols = [c for c in df.columns if c.startswith("time_")]
    other_cols = [
        c for c in df.columns
        if c not in front_cols + stat_cols + time_cols
    ]
    col_order = front_cols + stat_cols + time_cols + other_cols
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    # ── Save ────────────────────────────────────────────────────
    if save_csv is not None:
        os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)
        df.to_csv(save_csv, index=False, float_format="%.6f")
        print(f"  Saved summary table: {save_csv}")

    return df



def benchmark_heatmaps(
    output_dir: str = "benchmark_results",
    models: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    stat: str = "mean",
    annot_fmt: str = ".3f",
    cmap: str = "viridis",
    figsize_per_panel: tuple = (5, 3),
    show: bool = True,
    save: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of model × dataset for each metric.

    Parameters
    ----------
    output_dir
        Directory with saved JSON benchmark files.
    models
        Which models to include. None for all.
    datasets
        Which datasets to include. None for all.
    metrics
        Which metrics to plot. None for all.
    stat
        Statistic to display in each cell:

        - ``"mean"``: mean of per-cell values.
        - ``"median"``: median of per-cell values.
        - ``"rank"``: average rank across groups within each dataset.
          Rank 1 = best. Averaged across all edges/clusters.
    annot_fmt
        Format string for cell annotations.
    cmap
        Colormap name. For rank, this is automatically reversed
        (lower rank = better = greener).
    figsize_per_panel
        Size per metric panel.
    show
        Display the plot.
    save
        Path to save.

    Returns
    -------
    matplotlib Figure.

    Examples
    --------
    Mean performance heatmap::

        velot.pl.benchmark_heatmaps("benchmark_results", stat="mean")

    Rank heatmap::

        velot.pl.benchmark_heatmaps("benchmark_results", stat="rank")

    Specific metrics::

        velot.pl.benchmark_heatmaps(
            "benchmark_results", metrics=["cbdir"], stat="median",
        )
    """
    from velot.benchmark import load_benchmarks_per_group

    if stat not in ("mean", "median", "rank"):
        raise ValueError(
            f"stat must be 'mean', 'median', or 'rank', got '{stat}'."
        )

    df_cells = load_benchmarks_per_group(output_dir, models, datasets)

    if len(df_cells) == 0:
        raise ValueError(f"No benchmark results found in '{output_dir}'.")

    if metrics is None:
        metrics = sorted(df_cells["metric"].unique().tolist())

    n_metrics = len(metrics)
    if n_metrics == 0:
        raise ValueError("No metrics to plot.")

    # Include timing as an extra panel
    from velot.benchmark import load_benchmarks
    df_summary = load_benchmarks(output_dir, models, datasets)
    has_timing = "time_total" in df_summary.columns

    n_panels = n_metrics + (1 if has_timing else 0)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(figsize_per_panel[0] * n_panels, figsize_per_panel[1]),
        squeeze=False,
    )
    axes = axes.flatten()

    for i, metric_name in enumerate(metrics):
        ax = axes[i]
        df_m = df_cells[df_cells["metric"] == metric_name]

        if len(df_m) == 0:
            ax.set_title(f"{metric_name}\n(no data)")
            continue

        if stat == "rank":
            pivot = _compute_rank_pivot(df_m)
            _draw_heatmap(
                ax, pivot,
                title=f"{metric_name.upper()} (avg rank)",
                cmap=cmap + "_r",  # reverse: low rank = good = green
                annot_fmt=".2f",
                vmin=1,
                vmax=len(pivot.index),
                lower_is_better=True,
            )
        else:
            pivot = _compute_stat_pivot(df_m, stat)
            _draw_heatmap(
                ax, pivot,
                title=f"{metric_name.upper()} ({stat})",
                cmap=cmap,
                annot_fmt=annot_fmt,
                lower_is_better=False,
            )

    # Timing heatmap
    if has_timing:
        ax = axes[n_metrics]
        pivot_time = df_summary.pivot_table(
            values="time_total",
            index="model",
            columns="dataset",
            aggfunc="first",
        )
        if len(pivot_time) > 0:
            _draw_heatmap(
                ax, pivot_time,
                title="Time (seconds)",
                cmap=cmap + "_r",  # lower = better = green
                annot_fmt=".1f",
                lower_is_better=True,
            )

    plt.tight_layout()

    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def benchmark_ranking(
    output_dir: str = "benchmark_results",
    models: Optional[List[str]] = None,
    models_order: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    include_timing: bool = True,
    figsize: tuple = (8, 5),
    show: bool = True,
    save: Optional[str] = None,
) -> plt.Figure:
    """
    Overall ranking summary: average rank across all datasets and groups.

    For each (dataset, group, metric) combination, models are ranked
    1 to N (1 = best). Ranks are then averaged to produce a single
    score per model. Lower is better.

    Optionally includes execution time as a ranked criterion.

    Parameters
    ----------
    output_dir
        Directory with saved JSON benchmark files.
    models
        Which models to include. None for all.
    datasets
        Which datasets to include. None for all.
    metrics
        Which metrics to include. None for all.
    include_timing
        Whether to include execution time as a ranking criterion.
    figsize
        Figure size.
    show
        Display the plot.
    save
        Path to save.

    Returns
    -------
    matplotlib Figure.

    Examples
    --------
    ::

        velot.pl.benchmark_ranking("benchmark_results")
    """
    from velot.benchmark import load_benchmarks_per_group, load_benchmarks

    df_cells = load_benchmarks_per_group(output_dir, models, datasets)
    df_summary = load_benchmarks(output_dir, models, datasets)

    if len(df_cells) == 0:
        raise ValueError(f"No benchmark results found in '{output_dir}'.")

    if metrics is None:
        metrics = sorted(df_cells["metric"].unique().tolist())

    # ── Compute ranks per (dataset, group, metric) ──────────────
    rank_records = []

    for metric_name in metrics:
        df_m = df_cells[df_cells["metric"] == metric_name]

        for (dataset, group), grp in df_m.groupby(["dataset", "group"]):
            # Compute mean per model for this (dataset, group, metric)
            model_means = grp.groupby("model")["value"].mean()

            # Rank: higher value = better → rank ascending=False
            ranks = model_means.rank(ascending=False, method="average")

            for model, rank in ranks.items():
                rank_records.append({
                    "model": model,
                    "dataset": dataset,
                    "group": group,
                    "metric": metric_name,
                    "criterion": metric_name,
                    "rank": rank,
                })

    # ── Add timing ranks per dataset ────────────────────────────
    if include_timing and "time_total" in df_summary.columns:
        for dataset in df_summary["dataset"].unique():
            df_d = df_summary[df_summary["dataset"] == dataset]
            times = df_d.set_index("model")["time_total"]

            # Lower time = better → rank ascending=True
            ranks = times.rank(ascending=True, method="average")

            for model, rank in ranks.items():
                rank_records.append({
                    "model": model,
                    "dataset": dataset,
                    "group": "_timing_",
                    "metric": "time",
                    "criterion": "time",
                    "rank": rank,
                })

    df_ranks = pd.DataFrame(rank_records)

    if len(df_ranks) == 0:
        raise ValueError("No ranking data computed.")

    # ── Aggregate ───────────────────────────────────────────────
    # Average rank per (model, criterion)
    avg_per_criterion = (
        df_ranks.groupby(["model", "criterion"])["rank"]
        .mean()
        .reset_index()
        .rename(columns={"rank": "avg_rank"})
    )

    # Overall average rank per model
    avg_overall = (
        df_ranks.groupby("model")["rank"]
        .mean()
        .reset_index()
        .rename(columns={"rank": "avg_rank"})
        .sort_values("avg_rank")
    )

    # ── Plot ────────────────────────────────────────────────────
    all_models = avg_overall["model"].values
    n_models = len(all_models)
    criteria = sorted(avg_per_criterion["criterion"].unique())
    n_criteria = len(criteria)

    fig, axes = plt.subplots(
        1, 2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [2, 3]},
    )

    # --- Left panel: overall average rank bar chart ---
    ax = axes[0]
    cmap = plt.get_cmap("Set2")
    colors = [
        cmap(0),
        cmap(1),
        cmap(2),
        cmap(3),
        cmap(5)
    ]
    # colors = plt.cm.Set2(np.linspace(0, 1, max(n_models, 1)))
    if models_order is not None:
        model_colors = dict(zip(models_order, colors))
    else:
        model_colors = dict(zip(all_models, colors))
    # colors = plt.cm.Set2(np.linspace(0, 1, max(n_models, 1)))
    # model_colors = dict(zip(all_models, colors))

    bars = ax.barh(
        range(n_models),
        avg_overall["avg_rank"].values,
        color=[model_colors[m] for m in all_models],
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_yticks(range(n_models))
    ax.set_yticklabels(all_models, fontsize=10)
    ax.set_xlabel("Average rank (lower = better)", fontsize=10)
    ax.set_title("Overall ranking", fontsize=12, fontweight="bold")
    ax.invert_yaxis()  # best (lowest rank) at top

    # Annotate bars
    for bar, val in zip(bars, avg_overall["avg_rank"].values):
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            ha="left", va="center", fontsize=9, fontweight="bold",
        )

    ax.set_xlim(0, n_models + 0.5)
    ax.axvline(1, color="green", linewidth=0.8, linestyle="--", alpha=0.5,
               label="Best possible (1.0)")
    ax.grid(axis="x", alpha=0.3)

    # --- Right panel: rank breakdown by criterion ---
    ax = axes[1]

    pivot_rank = avg_per_criterion.pivot_table(
        values="avg_rank",
        index="model",
        columns="criterion",
    )
    # Reorder models by overall rank
    pivot_rank = pivot_rank.loc[all_models]

    x = np.arange(n_criteria)
    bar_width = 0.8 / max(n_models, 1)

    for m_idx, model in enumerate(all_models):
        if model not in pivot_rank.index:
            continue
        values = pivot_rank.loc[model].values
        offset = (m_idx - n_models / 2 + 0.5) * bar_width

        ax.bar(
            x + offset,
            values,
            bar_width * 0.9,
            color=model_colors[model],
            edgecolor="black",
            linewidth=0.5,
            label=model,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [c.upper() for c in criteria],
        rotation=30, ha="right", fontsize=9,
    )
    ax.set_ylabel("Average rank (lower = better)", fontsize=10)
    ax.set_title("Rank by criterion", fontsize=12, fontweight="bold")
    ax.axhline(1, color="green", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    plt.tight_layout()

    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


# ── Internal helpers ────────────────────────────────────────────

def _compute_stat_pivot(df_m, stat="mean"):
    """
    Pivot: model × dataset, values = mean or median pooling all groups.
    """
    agg_func = "mean" if stat == "mean" else "median"
    pivot = df_m.pivot_table(
        values="value",
        index="model",
        columns="dataset",
        aggfunc=agg_func,
    )
    return pivot


def _compute_rank_pivot(df_m):
    """
    Pivot: model × dataset, values = average rank across groups.

    For each (dataset, group), rank models (1=best). Then average
    across groups within each dataset.
    """
    rank_records = []

    for (dataset, group), grp in df_m.groupby(["dataset", "group"]):
        model_means = grp.groupby("model")["value"].mean()
        # Higher value = better → rank ascending=False
        ranks = model_means.rank(ascending=False, method="average")

        for model, rank in ranks.items():
            rank_records.append({
                "model": model,
                "dataset": dataset,
                "rank": rank,
            })

    df_ranks = pd.DataFrame(rank_records)

    pivot = df_ranks.pivot_table(
        values="rank",
        index="model",
        columns="dataset",
        aggfunc="mean",
    )
    return pivot


def _draw_heatmap(
    ax,
    pivot,
    title="",
    cmap="viridis",
    annot_fmt=".3f",
    vmin=None,
    vmax=None,
    lower_is_better=False,
):
    """
    Draw an annotated heatmap on the given axes.

    Highlights the best value in each column (dataset) with a
    bold border.
    """
    data = pivot.values
    n_rows, n_cols = data.shape

    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    # Draw the heatmap
    im = ax.imshow(
        data,
        cmap=cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )

    # Find best row per column FIRST
    best_rows = {}
    for c in range(n_cols):
        col_data = data[:, c]
        valid = ~np.isnan(col_data)
        if not valid.any():
            continue
        if lower_is_better:
            best_rows[c] = int(np.nanargmin(col_data))
        else:
            best_rows[c] = int(np.nanargmax(col_data))

    # Annotate cells
    for r in range(n_rows):
        for c in range(n_cols):
            val = data[r, c]
            if np.isnan(val):
                text = "—"
                color = "gray"
                weight = "normal"
            else:
                text = f"{val:{annot_fmt}}"
                normalized = (val - vmin) / max(vmax - vmin, 1e-10)
                if lower_is_better:
                    normalized = 1 - normalized
                color = "black" if normalized < 0.5 else "white"
                weight = "bold" if best_rows.get(c) == r else "normal"

            ax.text(
                c, r, text,
                ha="center", va="center",
                fontsize=10, fontweight=weight,
                color=color,
            )

    # Labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        pivot.columns, rotation=30, ha="right", fontsize=9,
    )
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)



def benchmark_dotplot(
    output_dir: str = "benchmark_results",
    models: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    datasets_order: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    include_timing: bool = True,
    normalize_mode: str = "global",
    stat: str = "mean",
    metric_colors: Optional[Dict[str, str]] = None,
    metric_labels: Optional[Dict[str, str]] = None,
    model_labels: Optional[Dict[str, str]] = None,
    higher_is_better: Optional[Dict[str, bool]] = None,
    figsize: Optional[tuple] = None,
    legend: bool = True,
    summary: bool = True,
    show: bool = True,
    save: Optional[str] = None,
) -> plt.Figure:
    """
    Integrated dotplot benchmark summary.

    Circle area reflects per-column normalised score (for visual
    comparison only). Displayed values match the heatmap values
    for the chosen ``stat``.

    Parameters
    ----------
    output_dir
        Directory with saved JSON benchmark files.
    models
        Which models to include. None for all.
    datasets
        Which datasets to include. None for all.
    metrics
        Which metrics to include. None for all found.
    include_timing
        Whether to include execution time as a metric group.
    stat
        Aggregation statistic — same as in ``benchmark_heatmaps``:

        - ``"mean"``: mean of per-cell values per (model, dataset).
        - ``"median"``: median of per-cell values.
        - ``"rank"``: average rank across groups within each dataset
          (1 = best). Consistent with heatmap ``stat="rank"``.
    metric_colors
        Dict mapping metric name → color.
    metric_labels
        Dict mapping metric name → display label.
    model_labels
        Dict mapping model name → display label.
    higher_is_better
        Dict mapping metric name → bool. Only used for normalisation
        direction and ``stat="rank"``. Default True for accuracy
        metrics, False for time.
    figsize
        Figure size. Auto-computed if None.
    show
        Display the plot.
    save
        Path to save.

    Returns
    -------
    matplotlib Figure.
    """
    from velot.benchmark import load_benchmarks, load_benchmarks_per_group

    df_cells = load_benchmarks_per_group(output_dir, models, datasets)
    df_summary = load_benchmarks(output_dir, models, datasets)

    if len(df_cells) == 0:
        raise ValueError(f"No benchmark results found in '{output_dir}'.")

    # ── Resolve metrics ─────────────────────────────────────────
    available_metrics = sorted(df_cells["metric"].unique().tolist())
    if metrics is None:
        metrics = available_metrics
    else:
        metrics = [m for m in metrics if m in available_metrics]

    if len(metrics) == 0:
        raise ValueError("No metrics found to plot.")

    all_models = sorted(df_summary["model"].unique().tolist())
    if datasets_order is not None:
        all_datasets = datasets_order
    else:
        all_datasets = sorted(df_summary["dataset"].unique().tolist())
    n_models = len(all_models)
    n_datasets = len(all_datasets)

    # ── Defaults ────────────────────────────────────────────────
    default_palette = ["#c0392b", "#2874a6", "#8e44ad", "#d35400",
                       "#16a085", "#7d3c98", "#2c3e50"]
    if metric_colors is None:
        metric_colors = {}
    for i, m in enumerate(metrics):
        if m not in metric_colors:
            metric_colors[m] = default_palette[i % len(default_palette)]
    if include_timing:
        metric_colors.setdefault("time", "#1e8449")

    if higher_is_better is None:
        higher_is_better = {}
    for m in metrics:
        higher_is_better.setdefault(m, True)
    higher_is_better.setdefault("time", False)

    if metric_labels is None:
        metric_labels = {}
    for m in metrics:
        arrow = "↑" if higher_is_better.get(m, True) else "↓"
        metric_labels.setdefault(m, f"{m.upper()} {arrow}")
    metric_labels.setdefault("time", "Time ↓ (seconds)")

    if model_labels is None:
        model_labels = {}
    for m in all_models:
        model_labels.setdefault(m, m)

    # ── Build RAW data matrices (same as heatmaps) ──────────────
    raw_matrices = {}  # metric → (n_models, n_datasets)

    for metric_name in metrics:
        df_m = df_cells[df_cells["metric"] == metric_name]

        if stat == "rank":
            # Rank per (dataset, group), then average within dataset
            # Same logic as _compute_rank_pivot in heatmaps
            mat = np.full((n_models, n_datasets), np.nan)
            for di, dataset in enumerate(all_datasets):
                df_d = df_m[df_m["dataset"] == dataset]
                groups = df_d["group"].unique()

                model_ranks_all = {m: [] for m in all_models}
                for group in groups:
                    df_g = df_d[df_d["group"] == group]
                    model_means = df_g.groupby("model")["value"].mean()
                    hb = higher_is_better.get(metric_name, True)
                    ranks = model_means.rank(
                        ascending=not hb, method="average"
                    )
                    for model in all_models:
                        if model in ranks.index:
                            model_ranks_all[model].append(ranks[model])

                for mi, model in enumerate(all_models):
                    if model_ranks_all[model]:
                        mat[mi, di] = np.mean(model_ranks_all[model])
            raw_matrices[metric_name] = mat
        else:
            # Mean or median of per-cell values
            agg_func = np.nanmean if stat == "mean" else np.nanmedian
            mat = np.full((n_models, n_datasets), np.nan)
            for mi, model in enumerate(all_models):
                for di, dataset in enumerate(all_datasets):
                    mask = (
                        (df_m["model"] == model)
                        & (df_m["dataset"] == dataset)
                    )
                    vals = df_m.loc[mask, "value"].dropna().values
                    if len(vals) > 0:
                        mat[mi, di] = agg_func(vals)
            raw_matrices[metric_name] = mat

    # Time matrix
    has_timing = include_timing and "time_total" in df_summary.columns
    if has_timing:
        if stat == "rank":
            time_mat = np.full((n_models, n_datasets), np.nan)
            for di, dataset in enumerate(all_datasets):
                df_d = df_summary[df_summary["dataset"] == dataset]
                times = df_d.set_index("model")["time_total"]
                ranks = times.rank(ascending=True, method="average")
                for mi, model in enumerate(all_models):
                    if model in ranks.index:
                        time_mat[mi, di] = ranks[model]
        else:
            time_mat = np.full((n_models, n_datasets), np.nan)
            for mi, model in enumerate(all_models):
                for di, dataset in enumerate(all_datasets):
                    mask = (
                        (df_summary["model"] == model)
                        & (df_summary["dataset"] == dataset)
                    )
                    vals = df_summary.loc[mask, "time_total"].values
                    if len(vals) > 0:
                        time_mat[mi, di] = vals[0]
        raw_matrices["time"] = time_mat
        all_metric_keys = metrics + ["time"]
    else:
        all_metric_keys = list(metrics)

    # ── Compute OVERALL per metric (same as heatmap) ────────────
    # For mean/median: average of raw values across datasets
    # For rank: average of ranks across datasets
    raw_overall = {}
    for mk in all_metric_keys:
        raw_overall[mk] = np.nanmean(raw_matrices[mk], axis=1)

    # ── Grand score (for ranking methods) ───────────────────────
    if stat == "rank":
        # Average of average ranks — lower is better
        grand = np.nanmean(
            np.column_stack([raw_overall[mk] for mk in all_metric_keys]),
            axis=1,
        )
        order = np.argsort(grand)  # lower rank = better = first
        grand_higher_better = False
    else:
        # Normalise each metric's overall to [0,1] then average
        # This is needed because metrics have different scales
        norm_overalls = []
        for mk in all_metric_keys:
            ov = raw_overall[mk].copy()
            valid = ~np.isnan(ov)
            if valid.sum() == 0:
                norm_overalls.append(np.full_like(ov, np.nan))
                continue
            vmin, vmax = ov[valid].min(), ov[valid].max()
            hb = higher_is_better.get(mk, True)
            if mk == "time" and stat != "rank":
                # Log transform time before normalising
                ov_t = np.full_like(ov, np.nan)
                ov_t[valid] = np.log(ov[valid])
                vmin, vmax = ov_t[valid].min(), ov_t[valid].max()
                ov = ov_t
                hb = False
            if vmax == vmin:
                norm_overalls.append(np.where(valid, 1.0, np.nan))
                continue
            normed = np.full_like(ov, np.nan)
            normed[valid] = (ov[valid] - vmin) / (vmax - vmin)
            if not hb:
                normed[valid] = 1.0 - normed[valid]
            norm_overalls.append(normed)

        grand = np.nanmean(np.column_stack(norm_overalls), axis=1)
        order = np.argsort(-grand)  # higher = better = first
        grand_higher_better = True

    models_sorted = [all_models[i] for i in order]

    # ── Normalise for DOT SIZING only ───────────────────────────
    def _norm_for_sizing(arr, hb=True, log_transform=False, mode="global"):
        """
        Map values to [0, 1] for circle area.

        mode: "column" → per-column, "row" → per-row, "global" → whole matrix.
        """
        out = np.full_like(arr, np.nan, dtype=float)

        # Apply log transform first if needed
        work = arr.astype(float).copy()
        if log_transform:
            valid_all = ~np.isnan(work)
            work[valid_all] = np.log(work[valid_all])

        if mode == "column":
            for j in range(work.shape[1]):
                col = work[:, j]
                valid = ~np.isnan(col)
                if valid.sum() == 0:
                    continue
                vmin, vmax = col[valid].min(), col[valid].max()
                if vmax == vmin:
                    out[valid, j] = 1.0
                    continue
                s = (col[valid] - vmin) / (vmax - vmin)
                if not hb:
                    s = 1.0 - s
                out[valid, j] = s

        elif mode == "row":
            for i in range(work.shape[0]):
                row = work[i, :]
                valid = ~np.isnan(row)
                if valid.sum() == 0:
                    continue
                vmin, vmax = row[valid].min(), row[valid].max()
                if vmax == vmin:
                    out[i, valid] = 1.0
                    continue
                s = (row[valid] - vmin) / (vmax - vmin)
                if not hb:
                    s = 1.0 - s
                out[i, valid] = s

        elif mode == "global":
            valid = ~np.isnan(work)
            if valid.sum() == 0:
                return out
            vmin, vmax = work[valid].min(), work[valid].max()
            if vmax == vmin:
                out[valid] = 1.0
                return out
            s = (work[valid] - vmin) / (vmax - vmin)
            if not hb:
                s = 1.0 - s
            out[valid] = s

        return out

    size_matrices = {}
    for mk in all_metric_keys:
        hb = higher_is_better.get(mk, True)
        if stat == "rank":
            size_matrices[mk] = _norm_for_sizing(
                raw_matrices[mk], hb=False,
                log_transform=False, mode=normalize_mode,
            )
        elif mk == "time":
            size_matrices[mk] = _norm_for_sizing(
                raw_matrices[mk], hb=False,
                log_transform=True, mode=normalize_mode,
            )
        else:
            size_matrices[mk] = _norm_for_sizing(
                raw_matrices[mk], hb=hb,
                log_transform=False, mode=normalize_mode,
            )

    # Overall sizing
    size_overall = {}
    for mk in all_metric_keys:
        size_overall[mk] = np.nanmean(size_matrices[mk], axis=1)

    # ── Build column layout ─────────────────────────────────────
    cols = []
    group_spans = []
    x = 0

    # Determine annotation format
    if stat == "rank":
        annot_fmt = ".2f"  # ranks like 1.24
    else:
        annot_fmt = ".2f"  # raw values like 0.829

    time_annot_fmt = ".1f" if stat != "rank" else ".2f"

    for mk in all_metric_keys:
        gcolor = metric_colors[mk]
        glabel = metric_labels[mk]
        g_start = x

        for di, dataset in enumerate(all_datasets):
            cols.append({
                "x": x,
                "color": gcolor,
                "label": dataset,
                "kind": "dataset",
                "raw_values": raw_matrices[mk][:, di],
                "size_values": size_matrices[mk][:, di],
                "fmt": time_annot_fmt if mk == "time" else annot_fmt,
            })
            x += 1

        cols.append({
            "x": x,
            "color": gcolor,
            "label": "Overall",
            "kind": "overall",
            "raw_values": raw_overall[mk],
            "size_values": size_overall[mk],
            "fmt": time_annot_fmt if mk == "time" else annot_fmt,
        })
        x += 1
        group_spans.append((glabel, gcolor, g_start - 0.45, x - 1 + 0.45))
        x += 0.7

    m2i = {m: i for i, m in enumerate(all_models)}

    # ── Figure setup ────────────────────────────────────────────
    if figsize is None:
        width = max(12, len(cols) * 0.9 + 6)
        height = max(4, n_models * 0.9 + 2.5)
        figsize = (width, height)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.14, 0.16, 0.68, 0.64])

    # ── Zebra striping ──────────────────────────────────────────
    for row in range(n_models):
        if row % 2 == 0:
            ax.add_patch(plt.Rectangle(
                (-0.7, row - 0.48), x + 8, 0.96,
                color="#f6f6f6", zorder=0, lw=0,
            ))

    # ── Find best value per column for highlighting ─────────────
    best_per_col = {}
    for ci, col_info in enumerate(cols):
        raw_vals = col_info["raw_values"]
        valid = ~np.isnan(raw_vals)
        if not valid.any():
            continue
        mk = None
        for mk_name in all_metric_keys:
            if col_info["color"] == metric_colors[mk_name]:
                mk = mk_name
                break
        hb = higher_is_better.get(mk, True)
        if stat == "rank":
            hb = False  # lower rank = better
        if mk == "time" and stat != "rank":
            hb = False  # lower time = better

        if hb:
            best_per_col[ci] = np.nanmax(raw_vals)
        else:
            best_per_col[ci] = np.nanmin(raw_vals)

    for row, m in enumerate(models_sorted):
        mi = m2i[m]
        for ci, col_info in enumerate(cols):
            cx = col_info["x"]
            gcolor = col_info["color"]
            kind = col_info["kind"]
            raw_val = col_info["raw_values"][mi]
            size_val = col_info["size_values"][mi]
            fmt = col_info["fmt"]

            if np.isnan(raw_val):
                ax.plot(
                    cx, row, marker="x", color="#888",
                    markersize=11, mew=2, zorder=3,
                )
                continue

            # Check if this is the best in its column
            is_best = (
                ci in best_per_col
                and np.isclose(raw_val, best_per_col[ci], rtol=1e-6)
            )

            dot_size = 60 + 1100 * max(size_val, 0)
            ax.scatter(
                cx, row, s=dot_size,
                color=gcolor, #color="gold" if is_best else gcolor,
                alpha=1 if is_best else 0.75,
                edgecolor="black",
                linewidth=1.0,
                zorder=4 if is_best else 3,
            )

            # Annotation
            text = f"{raw_val:{fmt}}"
            if size_val >= 0.30:
                ax.text(
                    cx, row, text,
                    ha="center", va="center",
                    fontsize=7.5,
                    color="white", #"black" if is_best else "white",
                    fontweight="bold", zorder=5,
                )
    #         else:
    #             ax.text(
    #                 cx + 0.35, row, text,
    #                 ha="left", va="center",
    #                 fontsize=7.5, color="black",
    #                 fontweight="bold", zorder=5,
    #             )

    # # ── Draw circles ────────────────────────────────────────────
    # for row, m in enumerate(models_sorted):
    #     mi = m2i[m]
    #     for col_info in cols:
    #         cx = col_info["x"]
    #         gcolor = col_info["color"]
    #         kind = col_info["kind"]
    #         raw_val = col_info["raw_values"][mi]
    #         size_val = col_info["size_values"][mi]
    #         fmt = col_info["fmt"]

    #         if np.isnan(raw_val):
    #             ax.plot(
    #                 cx, row, marker="x", color="#888",
    #                 markersize=11, mew=2, zorder=3,
    #             )
    #             continue

    #         # Dot size from normalised score
    #         dot_size = 60 + 1100 * max(size_val, 0)
    #         ax.scatter(
    #             cx, row, s=dot_size, color=gcolor, alpha=0.88,
    #             edgecolor="black", linewidth=0.6, zorder=3,
    #         )

    #         # Annotation for "Overall" columns: show RAW value
    #         # if kind == "overall":
    #         text = f"{raw_val:{fmt}}"
    #         if size_val >= 0.30:
    #             ax.text(
    #                 cx, row, text,
    #                 ha="center", va="center",
    #                 fontsize=7.5, color="white",
    #                 fontweight="bold", zorder=4,
    #             )
    #         # else:
    #         #     ax.text(
    #         #         cx, row-0.1, text,
    #         #         ha="left", va="center",
    #         #         fontsize=7.5, color="black",
    #         #         fontweight="bold", zorder=4,
    #         #     )

    # ── Group header bars ───────────────────────────────────────
    header_y = -1.05
    for gtitle, gcolor, gs, ge in group_spans:
        ax.add_patch(plt.Rectangle(
            (gs, header_y - 0.32), ge - gs, 0.5,
            color=gcolor, alpha=0.95, zorder=2,
        ))
        ax.text(
            (gs + ge) / 2, header_y - 0.07, gtitle,
            ha="center", va="center",
            color="white", fontsize=10.5, fontweight="bold",
        )

    # ── Right-hand ranking panel ────────────────────────────────
    bar_x0 = x
    bar_w = 3.0
    rank_x = bar_x0 + bar_w + 1.1

    # For bar length, normalise grand to [0,1] for display
    grand_valid = grand[~np.isnan(grand)]
    if len(grand_valid) > 0:
        if grand_higher_better:
            g_min, g_max = grand_valid.min(), grand_valid.max()
        else:
            # Invert for bar: lower rank → longer bar
            g_min, g_max = grand_valid.min(), grand_valid.max()
    else:
        g_min, g_max = 0, 1

    for row, m in enumerate(models_sorted):
        mi = m2i[m]
        val = grand[mi]
        rank = row + 1
        bcolor = "#1b2631" if rank == 1 else "#566573"

        # Bar length proportional to score
        if grand_higher_better:
            bar_frac = (val - g_min) / max(g_max - g_min, 1e-10)
        else:
            # For ranks: lower is better → invert bar
            bar_frac = 1.0 - (val - g_min) / max(g_max - g_min, 1e-10)

        bar_frac = max(bar_frac, 0.02)  # minimum visible bar

        ax.add_patch(plt.Rectangle(
            (bar_x0, row - 0.28), bar_frac * bar_w, 0.56,
            color=bcolor, alpha=0.92, zorder=3,
        ))

        # Show the actual grand value
        if stat == "rank":
            grand_text = f"{val:.2f}"
        else:
            grand_text = f"{val:.2f}"

        ax.text(
            bar_x0 + bar_frac * bar_w + 0.08, row,
            grand_text, va="center",
            fontsize=10, fontweight="bold",
        )

    # Ranking panel header
    ax.add_patch(plt.Rectangle(
        (bar_x0, header_y - 0.32),
        (rank_x - bar_x0) + 0.7, 0.5,
        color="#1b2631", alpha=0.95, zorder=2,
    ))
    ax.text(
        bar_x0 + ((rank_x - bar_x0) + 0.7) / 2, header_y - 0.07,
        "Overall ranking",
        ha="center", va="center",
        color="white", fontsize=10.5, fontweight="bold",
    )

    # ── Axes formatting ─────────────────────────────────────────
    ax.set_xticks([c["x"] for c in cols])
    ax.set_xticklabels(
        [
            c["label"].capitalize() if c["kind"] != "overall"
            else "Overall"
            for c in cols
        ],
        rotation=40, fontsize=9.5, ha="right",
    )

    ax.set_yticks(range(n_models))
    ax.set_yticklabels(
        [model_labels.get(m, m) for m in models_sorted],
        fontsize=11, fontweight="bold",
    )
    ax.invert_yaxis()

    ax.set_xlim(-0.9, rank_x + 1.0)
    ax.set_ylim(n_models - 0.5, header_y - 0.55)
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(left=False, bottom=False)

    if legend:
        # ── Legend: circle size scale ───────────────────────────────
        leg = fig.add_axes([0.14, -0.05, 0.38, 0.1])
        leg.axis("off")
        leg.set_xlim(0, 5.5)
        leg.set_ylim(0, 1)
        leg.text(
            -0.05, 0.6, "Norm. score:",
            ha="right", va="center",
            fontsize=9.5, fontweight="bold",
        )
        for i, s in enumerate([0.1, 0.3, 0.5, 0.7, 0.9, 1.0]):
            leg.scatter(
                i * 0.85 + 0.25, 0.6, s=60 + 1100 * s,
                color="#7f8c8d", alpha=0.78,
                edgecolor="black", linewidth=0.5,
            )
            leg.text(
                i * 0.85 + 0.25, 0, f"{s:.1f}",
                ha="center", fontsize=8.5,
            )

        # ── Legend: metric colours ──────────────────────────────────
        cl = fig.add_axes([0.55, -0.05, 0.35, 0.1])
        cl.axis("off")
        n_legend = len(all_metric_keys)
        cl.set_xlim(0, max(n_legend * 1.3, 2))
        cl.set_ylim(0, 1)
        for i, mk in enumerate(all_metric_keys):
            cl.scatter(
                i * 1.25 + 0.2, 0.6, s=380,
                color=metric_colors[mk], alpha=0.88,
                edgecolor="black", linewidth=0.5,
            )
            cl.text(
                i * 1.25 + 0.45, 0.6,
                metric_labels.get(mk, mk.upper()),
                va="center", fontsize=9.5, fontweight="bold",
            )
    
    if summary:
        # ── Print numeric summary ───────────────────────────────────
        print(f"\nBenchmark ranking summary (stat={stat}):")
        print("-" * 70)
        header_parts = ["Rank", f"{'Model':<24}"]
        for mk in all_metric_keys:
            header_parts.append(f"{mk.upper():>8}")
        header_parts.append(f"{'GRAND':>8}")
        print("  ".join(header_parts))
        print("-" * 70)

        for rk, m in enumerate(models_sorted, 1):
            mi = m2i[m]
            parts = [f"  #{rk}", f"  {model_labels.get(m, m):<24}"]
            for mk in all_metric_keys:
                ov = raw_overall[mk][mi]
                if mk == "time" and stat != "rank":
                    parts.append(f"{ov:>8.1f}")
                else:
                    parts.append(f"{ov:>8.3f}")
            parts.append(f"{grand[mi]:>8.3f}")
            print("  ".join(parts))

    # ── Save / show ─────────────────────────────────────────────
    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.savefig(save, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"\n  Saved: {save}")
    if show:
        plt.show()

    return fig
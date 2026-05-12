import json
import time
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd


class BenchmarkTimer:
    """Context manager to time pipeline stages."""

    def __init__(self):
        self.stages: Dict[str, float] = {}
        self._current_stage: Optional[str] = None
        self._start_time: Optional[float] = None
        self._total_start: Optional[float] = None
        self._total_end: Optional[float] = None

    def start(self, stage: str):
        self._current_stage = stage
        self._start_time = time.perf_counter()
        if self._total_start is None:
            self._total_start = self._start_time

    def stop(self):
        if self._current_stage is None:
            return
        elapsed = time.perf_counter() - self._start_time
        self.stages[self._current_stage] = elapsed
        self._total_end = time.perf_counter()
        self._current_stage = None

    def __call__(self, stage: str):
        return _TimerContext(self, stage)

    @property
    def total(self) -> float:
        if self._total_start is None:
            return 0.0
        end = self._total_end or time.perf_counter()
        return end - self._total_start

    def summary(self) -> Dict[str, float]:
        result = dict(self.stages)
        result["total"] = self.total
        return result

    def __repr__(self):
        lines = [f"  {k}: {v:.2f}s" for k, v in self.stages.items()]
        lines.append(f"  total: {self.total:.2f}s")
        return "BenchmarkTimer(\n" + "\n".join(lines) + "\n)"


class _TimerContext:
    def __init__(self, timer: BenchmarkTimer, stage: str):
        self.timer = timer
        self.stage = stage

    def __enter__(self):
        self.timer.start(self.stage)
        return self.timer

    def __exit__(self, *args):
        self.timer.stop()


def save_benchmark(
    results: Dict[str, Any],
    timer: BenchmarkTimer,
    model_name: str,
    dataset_name: str,
    output_dir: str = "benchmark_results",
    extra_info: Optional[Dict] = None,
) -> str:
    """
    Save benchmark results for one (model, dataset) run.

    Parameters
    ----------
    results
        Output of ``velot.metrics.summary()``.
    timer
        BenchmarkTimer with timing information.
    model_name
        Name of the model (e.g., ``"scvelo_dynamical"``, ``"velot"``).
    dataset_name
        Name of the dataset (e.g., ``"pancreas"``).
    output_dir
        Directory to save results.
    extra_info
        Any additional metadata.

    Returns
    -------
    Path to the saved JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)

    record = {
        "model": model_name,
        "dataset": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "timing": timer.summary(),
        "extra": extra_info or {},
        "metrics": _to_serializable(results),
    }

    filename = f"{model_name}_{dataset_name}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(record, f, indent=2)

    print(f"  Saved: {filepath}")
    return filepath


def load_benchmarks(
    output_dir: str = "benchmark_results",
    models: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load benchmark summaries: one row per (model, dataset).

    Extracts scalar metrics (``*_mean``) and timing information.

    Returns
    -------
    DataFrame with columns: model, dataset, time_*, metric_mean, ...
    """
    records = []

    for filepath in sorted(Path(output_dir).glob("*.json")):
        with open(filepath) as f:
            data = json.load(f)

        if models and data["model"] not in models:
            continue
        if datasets and data["dataset"] not in datasets:
            continue

        row = {
            "model": data["model"],
            "dataset": data["dataset"],
            "timestamp": data["timestamp"],
        }

        # Timing
        for stage, seconds in data.get("timing", {}).items():
            row[f"time_{stage}"] = seconds

        # Extra info
        for k, v in data.get("extra", {}).items():
            if isinstance(v, (int, float)):
                row[k] = v

        # Scalar metrics (e.g., iccoh_mean, cbdir_mean)
        metrics = data.get("metrics", {})
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                row[k] = v

        records.append(row)

    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df.sort_values(["dataset", "model"]).reset_index(drop=True)
    return df


def load_benchmarks_per_group(
    output_dir: str = "benchmark_results",
    models: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    metric: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load per-cell metric values for boxplots.

    Parses the nested structure in the JSON where each metric
    (e.g., ``"cbdir"``, ``"iccoh"``) contains a dict of
    group → array-of-values. Groups can be edges
    (``"Fev+ → Alpha"``) or clusters (``"Alpha"``).

    Parameters
    ----------
    output_dir
        Directory containing JSON result files.
    models
        Filter to specific models. None for all.
    datasets
        Filter to specific datasets. None for all.
    metric
        Specific metric to load (e.g., ``"cbdir"``). If None,
        loads all metrics that contain per-cell arrays.

    Returns
    -------
    Long-format DataFrame with columns:
        model, dataset, metric, group, value

    Each row is one cell's value for one (model, dataset, metric, group).
    """
    records = []

    for filepath in sorted(Path(output_dir).glob("*.json")):
        with open(filepath) as f:
            data = json.load(f)

        if models and data["model"] not in models:
            continue
        if datasets and data["dataset"] not in datasets:
            continue

        model = data["model"]
        dataset = data["dataset"]
        metrics = data.get("metrics", {})

        for metric_name, metric_val in metrics.items():
            # Skip scalars (like iccoh_mean) — we want the dicts of arrays
            if not isinstance(metric_val, dict):
                continue

            # Skip if a specific metric was requested and this isn't it
            if metric is not None and metric_name != metric:
                continue

            # metric_val is like {"Alpha": [0.6, 0.8, ...], "Beta": [...]}
            for group_name, values in metric_val.items():
                if not isinstance(values, list):
                    continue

                for v in values:
                    if isinstance(v, (int, float)) and not np.isnan(v):
                        records.append({
                            "model": model,
                            "dataset": dataset,
                            "metric": metric_name,
                            "group": group_name,
                            "value": float(v),
                        })

    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df.sort_values(
            ["dataset", "metric", "model", "group"]
        ).reset_index(drop=True)
    return df


def load_benchmarks_per_group_summary(
    output_dir: str = "benchmark_results",
    models: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load one row per (model, dataset, metric, group) with summary stats.

    Useful for bar charts or compact comparisons where per-cell
    resolution is not needed.

    Returns
    -------
    DataFrame with columns:
        model, dataset, metric, group, mean, median, std, n
    """
    records = []

    for filepath in sorted(Path(output_dir).glob("*.json")):
        with open(filepath) as f:
            data = json.load(f)

        if models and data["model"] not in models:
            continue
        if datasets and data["dataset"] not in datasets:
            continue

        model = data["model"]
        dataset = data["dataset"]
        metrics = data.get("metrics", {})

        for metric_name, metric_val in metrics.items():
            if not isinstance(metric_val, dict):
                continue

            for group_name, values in metric_val.items():
                if not isinstance(values, list) or len(values) == 0:
                    continue

                arr = np.array([v for v in values if isinstance(v, (int, float))])
                if len(arr) == 0:
                    continue

                records.append({
                    "model": model,
                    "dataset": dataset,
                    "metric": metric_name,
                    "group": group_name,
                    "mean": float(np.nanmean(arr)),
                    "median": float(np.nanmedian(arr)),
                    "std": float(np.nanstd(arr)),
                    "n": int(np.sum(~np.isnan(arr))),
                })

    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df.sort_values(
            ["dataset", "metric", "model", "group"]
        ).reset_index(drop=True)
    return df


def _to_serializable(obj):
    """Convert numpy types and tuple keys to JSON-compatible types."""
    if isinstance(obj, dict):
        return {
            _key_to_str(k): _to_serializable(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _key_to_str(key):
    """Convert dictionary keys to JSON-safe strings."""
    if isinstance(key, tuple):
        return " → ".join(str(k) for k in key)
    return str(key)
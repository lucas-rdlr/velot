"""
One runner for every VelOT-family benchmark run in the revision.

    python run_velot.py --dataset pancreas --method velot --seed 0

Everything that varies between runs is a flag, so the seed sweep, the
root sweep and the supervision comparison all use this same script and
therefore the same windows, smoother and metrics.

    # main benchmark: one configuration per method, five seeds
    python run_velot.py -d erythroid -m velot          -s 0
    python run_velot.py -d erythroid -m velot_gradient -s 0

    # root sweep: same everything, a different root cluster or cell
    python run_velot.py -d pancreas -m velot --root-cluster Alpha --tag root-Alpha
    python run_velot.py -d pancreas -m velot --root-cell 1234     --tag rootcell-1234

    # supervision comparison
    python run_velot.py -d pancreas -m velot --spatial-key clusters_id --tag labels
    python run_velot.py -d pancreas -m velot --n-clusters 10          --tag kmeans

Results are written as ``<model>_<raw|smooth>_<dataset>.json`` where
``<model>`` is ``<method>[_<tag>]_seed<seed>``; ``aggregate.py`` turns
them into a tidy table with mean +- sd across seeds. The AnnData is only
written with ``--save-adata`` (they are large and mostly identical).
"""
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import scanpy as sc
import velot
from velot.benchmark import BenchmarkTimer, save_benchmark

os.chdir(Path(__file__).resolve().parent)
import datasets as ds                                   # noqa: E402

sc.settings.verbosity = 0
OUTPUT_DIR = "results"

# ---------------------------------------------------------------------
# One smoother for every dataset and every method. Only window_size is
# per dataset, and it comes from the geometry (see datasets.py).
# ---------------------------------------------------------------------
SHARED = dict(
    smooth=True,
    min_window_size=50,
    overlap_fraction=0.0,
    tail_handling="drop",
    tail_threshold=20,
    n_epochs=100,
    lambda_smooth=0.1,
    lambda_curl=0.1,
    lambda_divergence=0.0,
    k_smooth=30,
    project_umap=True,
    verbose=True,
)

# The benchmark reports the first three. The rest are the OT ablation and
# stay available for the variant figure.
METHODS = {
    "velot":          dict(use_graph=False, ot_solver="emd"),
    "velot_graph": dict(use_graph=True, ot_solver="emd"),
    "velot_original": dict(),                     # the published configuration
    "velot_gradient": dict(method="gradient", gradient_mode="knn",
                           gradient_k=30),
    "velot_no_knn":     dict(use_graph=False),
    "velot_cost_nn":    dict(use_graph=False, cost_scale="nn"),
    "velot_argmax":     dict(use_graph=False, cost_scale="nn",
                             ot_assignment="argmax"),
    "velot_unbalanced": dict(use_graph=False, cost_scale="nn",
                             unbalanced=True, reg_m="auto"),
    "velot_geodesic":   dict(use_graph=False, cost_scale="nn",
                             cost_metric="geodesic", unbalanced=True,
                             reg_m="auto"),
    # direction-only: raw vectors are scaled to unit length before the
    # MLP, so cells with long displacements stop dominating the fit
    "velot_normraw":  dict(use_graph=False, ot_solver="emd",
                           normalize_raw=True),
}


# Bump whenever preprocessing or root selection changes meaning, so that
# caches written by an older version are ignored instead of silently
# reused. v2: root selection moved from diffusion space to the working
# basis and gained the neighbourhood-purity filter (25 Sep).
CACHE_VERSION = 2


def _resolve_pseudotime_key(a, cfg):
    """The registry's key, unless the CLI overrides or disables it.

    ``--pseudotime-key none`` turns the key OFF and falls back to DPT.
    That is what makes a root sweep possible on a dataset that normally
    reads its ordering from a column: with the key in place the root is
    never used, so every root would give an identical run.
    """
    if a.pseudotime_key is not None:
        if a.pseudotime_key.strip().lower() in ("none", "off", ""):
            return None
        return a.pseudotime_key
    return cfg.get("pseudotime_key")


def _check_root_is_used(a, cfg, pkey):
    """A root option together with a pseudotime column is a no-op.

    The ordering then comes from the column and the root is never read,
    so a sweep over roots produces N identical runs with no indication
    that anything was ignored. Refuse instead.
    """
    asked = (a.root_cell is not None or a.root_cluster is not None
             or a.root_selection != "distal")
    if pkey and asked:
        raise SystemExit(
            f"\n{a.dataset!r} takes its ordering from obs[{pkey!r}], so the "
            "root is never used and the root options you passed would be "
            "silently ignored.\n"
            "Pass --pseudotime-key none to switch the column off and use "
            "DPT from the root instead.\n")


def _cosine_to_truth(V, truth, labels):
    """Cosine of every cell's vector to the known field.

    Only defined for datasets carrying ``truth_key``. This is the metric
    that sits OUTSIDE the VelOT objective: unlike ICCoh it is not a
    smoothness term, and unlike CBDir it is not what the pseudotime
    penalty imposes, so it is the one number that can say whether a
    smoother field is also a more correct one.
    """
    na = np.linalg.norm(V, axis=1)
    nb = np.linalg.norm(truth, axis=1)
    ok = (na > 0) & (nb > 0)
    cos = np.full(len(V), np.nan)
    cos[ok] = (V[ok] * truth[ok]).sum(1) / (na[ok] * nb[ok])
    per_cluster = {str(c): cos[labels == c][~np.isnan(cos[labels == c])].tolist()
                   for c in np.unique(labels)}
    valid = cos[~np.isnan(cos)]
    return {
        "cos": per_cluster,                       # per-cell, for box plots
        "cos_mean": float(np.mean(valid)) if len(valid) else float("nan"),
        "cos_median": float(np.median(valid)) if len(valid) else float("nan"),
        "cos_frac_pos": float(np.mean(valid > 0)) if len(valid) else float("nan"),
        "cos_coverage": float(ok.mean()),
    }


def _cache_path(a, cfg):
    pk = _resolve_pseudotime_key(a, cfg)
    if pk:                       # pseudotime is read, not computed
        return (Path(a.cache_dir) /
                f"v{CACHE_VERSION}__{a.dataset}__key-{pk}.h5ad")
    root = (f"cell{a.root_cell}" if a.root_cell is not None
            else (a.root_cluster or cfg["root_cluster"]).replace(" ", "_"))
    return (Path(a.cache_dir) /
            f"v{CACHE_VERSION}__{a.dataset}__{root}__{a.root_selection}.h5ad")


def main(a):
    cfg = ds.DATASETS[a.dataset]
    timer = BenchmarkTimer()

    # Preprocessing and pseudotime depend on the dataset and the root, not
    # on the method or the seed, so a sweep reuses one cached object.
    cache = _cache_path(a, cfg)
    if a.cache and cache.exists():
        with timer("load"):
            adata = sc.read_h5ad(cache)
        print(f"  reusing {cache}")
    else:
        with timer("load"):
            adata, cfg = ds.load(a.dataset)
        with timer("preprocess"):
            pkey = _resolve_pseudotime_key(a, cfg)
            _check_root_is_used(a, cfg, pkey)
            if pkey:
                # an externally supplied ordering (e.g. the simulation
                # time of a synthetic dataset): no root, no DPT
                velot.pp.pseudotime(adata, key=pkey)
            else:
                velot.pp.pseudotime(
                    adata,
                    root_cluster=None if a.root_cell is not None
                    else (a.root_cluster or cfg["root_cluster"]),
                    root_cell=a.root_cell,
                    cluster_key=cfg["clusters_key"],
                    root_selection=a.root_selection,
                    basis=cfg["basis"],
                    dpt_impl=a.dpt_impl,
                )
        if a.cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            adata.write(cache)

    # The registry holds the default windowing; the CLI overrides it when
    # given, so the supervised and unsupervised arms are one flag apart
    # rather than an edit to datasets.py. Resolved into locals: DATASETS
    # is module state and must not be mutated by a run.
    spatial_key = a.spatial_key if a.spatial_key is not None else cfg["spatial_key"]
    n_clusters = a.n_clusters if a.n_clusters is not None else cfg["n_clusters"]

    window_size = a.window_size or cfg.get("window_size")
    if window_size is None:
        window_size, _ = velot.tl.choose_window_size(
            adata, basis=cfg["basis"],
            min_window_size=SHARED["min_window_size"],
            n_clusters=n_clusters, spatial_key=spatial_key,
            tail_handling=SHARED["tail_handling"],
            tail_threshold=SHARED["tail_threshold"])

    with timer("velocity"):
        if spatial_key is not None:
            adata.obs["clusters_id"] = \
                adata.obs[spatial_key].astype("category").cat.codes
            spatial_key = "clusters_id"
        velot.tl.velocity(
            adata,
            **{**SHARED, **METHODS[a.method]},
            basis=cfg["basis"],
            window_size=window_size,
            n_clusters=n_clusters,
            spatial_key=spatial_key,
            project_basis=cfg.get("project_basis", "X_umap"),
            random_state=a.seed,
        )

    with timer("evaluate"):
        results = {}
        suffix = cfg["basis"].split("X_", 1)[1]     # velot names keys by basis
        truth_key = cfg.get("truth_key")
        labels = adata.obs[cfg["clusters_key"]].astype(str).values
        for field, key in (("raw", f"velot_velocity_raw_{suffix}"),
                           ("smooth", f"velot_velocity_{suffix}")):
            res = velot.metrics.summary(
                adata, cluster_edges=cfg["edges"],
                cluster_key=cfg["clusters_key"],
                embedding_key=cfg["basis"], velocity_key=key)
            if truth_key and truth_key in adata.obsm:
                res.update(_cosine_to_truth(
                    adata.obsm[key], adata.obsm[truth_key], labels))
                print(f"  {field}: cosine to truth "
                      f"{res['cos_mean']:.3f} (median {res['cos_median']:.3f}, "
                      f"{res['cos_frac_pos']:.0%} in the right half-space)")
            results[field] = res

    print(timer)

    params = adata.uns.get("velot_raw_velocity_params", {})
    pkey = _resolve_pseudotime_key(a, cfg)
    if pkey:
        root = {"selection": f"key:{pkey}"}
    else:
        root = dict(adata.uns.get(
            "velot_root",
            {"cell": adata.uns.get("iroot"), "selection": "root_cell"}))
    # after an h5ad cache round-trip these come back as numpy scalars
    root = {k: (v.item() if hasattr(v, "item") else v) for k, v in root.items()}
    mass = adata.obs.get("velot_transported_mass")
    extra = dict(
        seed=a.seed, method=a.method, tag=a.tag,
        window_size=int(window_size),
        n_clusters=n_clusters, spatial_key=spatial_key,
        root=root, pseudotime_key=pkey, dpt_impl=a.dpt_impl,
        coverage=params.get("coverage"),
        mean_T_eff=params.get("mean_T_eff"),
        mean_n_eff=params.get("mean_n_eff"),
        typical_move_median=params.get("typical_move_median"),
        reg_m=params.get("reg_m"),
        mean_transported_mass=(float(np.nanmean(mass))
                               if mass is not None else None),
        n_cells=int(adata.n_obs), n_genes=int(adata.n_vars),
        n_pcs=cfg["n_pcs"], n_neighbors=cfg["n_neighbors"],
        velot_params={k: v for k, v in params.items()
                      if not isinstance(v, np.ndarray)},
    )

    name = a.method + (f"_{a.tag}" if a.tag else "") + f"_seed{a.seed}"
    for field, res in results.items():
        save_benchmark(
            adata=adata, results=res, timer=timer,
            model_name=f"{name}_{field}", dataset_name=a.dataset,
            output_dir=a.output_dir, extra_info=extra,
            save_adata=a.save_adata and field == "smooth",
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-d", "--dataset", required=True, choices=sorted(ds.DATASETS))
    p.add_argument("-m", "--method", default="velot", choices=sorted(METHODS))
    p.add_argument("-s", "--seed", type=int, default=0)
    p.add_argument("--window-size", type=int, default=None,
                   help="override the dataset's window size")
    p.add_argument("--root-cluster", default=None,
                   help="override the dataset's root cluster (root sweep)")
    p.add_argument("--root-cell", type=int, default=None,
                   help="use this exact cell as root (root perturbation)")
    p.add_argument("--root-selection", default="distal",
                   choices=("distal", "medoid", "first"))
    p.add_argument("--pseudotime-key", default=None,
                   help="obs column holding a precomputed ordering (e.g. "
                        "sim_time). Skips the root and DPT entirely; "
                        "overrides the dataset's own pseudotime_key. Pass "
                        "'none' to disable the dataset's key and use DPT, "
                        "which is what a root sweep needs.")
    p.add_argument("--dpt-impl", default="velot", choices=("velot", "scanpy"),
                   help="'velot' uses the unmodified diffusion distance; "
                        "'scanpy' reproduces sc.tl.dpt, which gives unit "
                        "weight to components with eigenvalue >= 0.9994")
    p.add_argument("--spatial-key", default=None,
                   help="obs column with spatial groups, e.g. clusters_id")
    p.add_argument("--n-clusters", type=int, default=None,
                   help="KMeans groups when --spatial-key is not given; "
                        "overrides the dataset's own n_clusters")
    p.add_argument("--tag", default=None, help="suffix for the model name")
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--cache-dir", default="results/cache")
    p.add_argument("--no-cache", dest="cache", action="store_false",
                   help="re-run preprocessing and DPT instead of reusing "
                        "the cached object for this (dataset, root)")
    p.add_argument("--save-adata", action="store_true")
    main(p.parse_args())

# ============================================================
# Mouse VelOT erythroid differentiation — individual plot export
#
# Saves each subplot as a standalone 300 DPI PNG so panels can
# be assembled externally (e.g. in LaTeX, Inkscape, PowerPoint).
#
# Outputs in OUTDIR/:
#   heatmap_programs_flow.png
#   umap_vector_field.png
#   trend_<Gene>.png          (one per trend gene)
#   violin_forward_velocity.png
#   violin_speed.png
#   violin_coherence.png
#   corr_gene_vs_flow_features.png
#   corr_gene_vs_pseudotime_by_state.png
#   legend_cell_states.png
# ============================================================

import os
from pathlib import Path
import sys
import re
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap


# ============================================================
# User parameters
# ============================================================

PROJECT_DIR = Path("/home/user/Documents/velot_agusti/notebooks/agusti")
sys.path.insert(0, str(PROJECT_DIR))

ADATA_PATH = "erythroid_velot.h5ad"

OUTDIR = "pseudotime_erythroid"
os.makedirs(OUTDIR, exist_ok=True)

DPI = 300
RANDOM_STATE = 7

cluster_key = None
pt_key = "pseudotime"

basis_key = "X_umap"

PREFERRED_VELOCITY_OBSM = "velot_velocity_umap"
FALLBACK_VELOCITY_OBSM = "velocity_umap"

EXPRESSION_LAYER = None
RUN_SCVELO_PREPROCESS_IF_NEEDED = False

N_HEATMAP_BINS = 95
N_TREND_BINS = 28
N_FLOW_BINS = 80

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 120,
    "savefig.dpi": DPI,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# ============================================================
# Load data
# ============================================================

adata = sc.read_h5ad(ADATA_PATH)
adata.var_names_make_unique()

print(adata)
print("\nobs columns:", list(adata.obs.columns))
print("obsm keys:", list(adata.obsm.keys()))
print("layers:", list(adata.layers.keys()))


# ============================================================
# Basic helpers
# ============================================================

def to_dense(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def get_matrix(adata, genes=None, layer=None):
    a = adata[:, genes] if genes is not None else adata
    X = a.layers[layer] if (layer is not None and layer in a.layers) else a.X
    return np.asarray(to_dense(X), dtype=np.float32)


def clean_label(x):
    return re.sub(r"[^a-z0-9]+", " ", str(x).lower()).strip()


def unique_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def available_genes(adata, genes):
    lookup = {str(g).upper(): str(g) for g in adata.var_names}
    out = []
    for g in genes:
        k = str(g).upper()
        if k in lookup:
            out.append(lookup[k])
    return unique_preserve_order(out)


def minmax(x):
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return np.zeros_like(x)
    lo, hi = np.nanmin(x[ok]), np.nanmax(x[ok])
    return (x - lo) / (hi - lo + 1e-12)


def robust_interp(y):
    y = np.asarray(y, dtype=float)
    if np.all(~np.isfinite(y)):
        return np.zeros_like(y)
    y2 = pd.Series(y).interpolate(limit_direction="both").values
    if np.any(~np.isfinite(y2)):
        finite = y2[np.isfinite(y2)]
        fill = np.nanmedian(finite) if len(finite) > 0 else 0.0
        y2 = np.nan_to_num(y2, nan=fill)
    return y2


def symmetric_scale(x, q=95):
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return np.zeros_like(x)
    m = max(np.nanpercentile(np.abs(x[ok]), q), 1e-12)
    return np.clip(x / m, -1, 1)


def safe_spearman(x, y, min_n=10):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < min_n:
        return np.nan, np.nan
    if np.nanstd(x[ok]) < 1e-12 or np.nanstd(y[ok]) < 1e-12:
        return np.nan, np.nan
    rho, p = spearmanr(x[ok], y[ok])
    return float(rho), float(p)


# ============================================================
# Infer cell-state annotation
# ============================================================

def infer_cluster_key(adata):
    expected_labels = {
        "blood progenitors 1", "blood progenitors 2",
        "erythroid1", "erythroid2", "erythroid3",
    }
    candidates = [
        "cell_type", "celltype", "cell_types",
        "annotation", "annotations", "annot",
        "clusters", "cluster", "CellType",
        "leiden", "louvain",
    ]
    for k in candidates:
        if k not in adata.obs.columns:
            continue
        vals = set(clean_label(v) for v in adata.obs[k].astype(str).unique())
        if len(vals.intersection(expected_labels)) >= 3:
            print(f"Using cluster_key = '{k}'")
            return k

    best_key, best_score = None, -1
    for k in adata.obs.columns:
        vals_raw = adata.obs[k].astype(str)
        nunique = vals_raw.nunique()
        if not (2 <= nunique <= 80):
            continue
        score = 0
        for v in vals_raw.unique():
            lab = clean_label(v)
            if "erythroid" in lab:
                score += 3
            if "blood progenitor" in lab:
                score += 3
            if lab in expected_labels:
                score += 5
        if score > best_score:
            best_score, best_key = score, k
    if best_key is not None and best_score > 0:
        print(f"Using cluster_key = '{best_key}' with score {best_score}")
        return best_key

    for k in adata.obs.columns:
        if 2 <= adata.obs[k].astype(str).nunique() <= 80:
            print(f"Using fallback cluster_key = '{k}'")
            return k

    raise ValueError("Could not infer cell-state column.")


if cluster_key is None:
    cluster_key = infer_cluster_key(adata)

adata.obs[cluster_key] = adata.obs[cluster_key].astype(str)
available_clusters = list(pd.Categorical(adata.obs[cluster_key]).categories)
print("\nDetected cell states:")
for c in available_clusters:
    print("  -", c)


# ============================================================
# Mouse erythroid labels, colors, and marker panels
# ============================================================

def normalize_known_label(label):
    lab = clean_label(label)
    if "blood progenitors 1" in lab or "blood progenitor 1" in lab or lab == "bp1":
        return "Blood progenitors 1"
    if "blood progenitors 2" in lab or "blood progenitor 2" in lab or lab == "bp2":
        return "Blood progenitors 2"
    if "erythroid1" in lab or "erythroid 1" in lab or lab == "ery 1":
        return "Erythroid1"
    if "erythroid2" in lab or "erythroid 2" in lab or lab == "ery 2":
        return "Erythroid2"
    if "erythroid3" in lab or "erythroid 3" in lab or lab == "ery 3":
        return "Erythroid3"
    return str(label)


canonical_state_order = [
    "Blood progenitors 1", "Blood progenitors 2",
    "Erythroid1", "Erythroid2", "Erythroid3",
]

state_color_map_canonical = {
    "Blood progenitors 1": "#f6d8c8",
    "Blood progenitors 2": "#c8ad9c",
    "Erythroid1": "#b2182b",
    "Erythroid2": "#ef6548",
    "Erythroid3": "#f03b20",
}

mouse_marker_candidates = {
    "Blood progenitors 1": [
        "Cd34", "Kit", "Procr", "Hlf", "Meis1", "Gata2",
        "Runx1", "Mllt3", "Lmo2", "Flt3", "Ly6a", "Spi1",
    ],
    "Blood progenitors 2": [
        "Kit", "Cd34", "Gata2", "Tal1", "Lmo2", "Myb",
        "Gfi1b", "Nfe2", "Epor", "Mpl", "Klf1",
    ],
    "Erythroid1": [
        "Gata1", "Klf1", "Tal1", "Gfi1b", "Epor",
        "Tfrc", "Cd36", "Alas2", "Slc25a37", "Fech",
        "Hmbs", "Urod", "Gypa",
    ],
    "Erythroid2": [
        "Tfrc", "Cd36", "Alas2", "Fech", "Ahsp", "Car1",
        "Gypa", "Slc4a1", "Bpgm", "Blvrb",
        "Hba-a1", "Hba-a2", "Hbb-bs", "Hbb-bt",
    ],
    "Erythroid3": [
        "Ahsp", "Car1", "Slc4a1", "Rhag", "Ank1",
        "Spta1", "Sptb", "Bcl2l1", "Gypa",
        "Hba-a1", "Hba-a2", "Hbb-bs", "Hbb-bt",
        "Alas2", "Bpgm",
    ],
}

program_panels = {
    "Stem/progenitor": [
        "Cd34", "Kit", "Procr", "Hlf", "Meis1", "Gata2",
        "Runx1", "Mllt3", "Lmo2", "Flt3", "Ly6a", "Spi1",
    ],
    "Erythroid priming": [
        "Tal1", "Gata1", "Klf1", "Gfi1b", "Epor", "Nfe2", "Myb",
    ],
    "Iron/heme synthesis": [
        "Tfrc", "Cd36", "Alas2", "Slc25a37", "Fech", "Hmbs", "Urod",
    ],
    "Membrane/remodeling": [
        "Gypa", "Ahsp", "Car1", "Slc4a1", "Rhag", "Ank1", "Spta1", "Sptb",
        "Bcl2l1", "Bpgm", "Blvrb",
    ],
    "Globin/terminal": [
        "Hba-a1", "Hba-a2", "Hbb-bs", "Hbb-bt", "Hbb-bh1", "Hba-x",
    ],
}

program_colors = {
    "Stem/progenitor": "#2b8cbe",
    "Erythroid priming": "#7bccc4",
    "Iron/heme synthesis": "#fdae61",
    "Membrane/remodeling": "#f46d43",
    "Globin/terminal": "#a50026",
    "Early dynamic": "#91bfdb",
    "Intermediate dynamic": "#fee090",
    "Late dynamic": "#d73027",
}

BROAD_MOUSE_ERYTHROID_GENES = available_genes(
    adata,
    unique_preserve_order(
        sum(mouse_marker_candidates.values(), [])
        + sum(program_panels.values(), [])
    ),
)
print("\nDetected canonical mouse erythroid genes:", BROAD_MOUSE_ERYTHROID_GENES)


def find_actual_cluster_for_canonical(canonical, available_clusters):
    for cl in available_clusters:
        if normalize_known_label(cl) == canonical:
            return cl
    return None


actual_state_order = []
for c in canonical_state_order:
    actual = find_actual_cluster_for_canonical(c, available_clusters)
    if actual is not None:
        actual_state_order.append(actual)
if len(actual_state_order) < 2:
    actual_state_order = available_clusters

state_color_map = {}
for cl in available_clusters:
    canonical = normalize_known_label(cl)
    state_color_map[cl] = state_color_map_canonical.get(canonical, "#999999")


def build_marker_candidates_by_actual_cluster(adata, available_clusters):
    out = {}
    for cl in available_clusters:
        canonical = normalize_known_label(cl)
        if canonical in mouse_marker_candidates:
            out[cl] = available_genes(adata, mouse_marker_candidates[canonical])
        else:
            out[cl] = BROAD_MOUSE_ERYTHROID_GENES
    return out


marker_candidates_by_cluster = build_marker_candidates_by_actual_cluster(
    adata, available_clusters,
)


# ============================================================
# Neighbor, UMAP, velocity, pseudotime
# ============================================================

def ensure_neighbors_and_umap(adata):
    if "neighbors" not in adata.uns:
        print("No neighbors found. Computing PCA/neighbors.")
        if "X_pca" not in adata.obsm:
            sc.pp.pca(adata, n_comps=min(50, adata.n_vars - 1))
        sc.pp.neighbors(adata, n_pcs=min(30, adata.obsm["X_pca"].shape[1]))
    if basis_key not in adata.obsm:
        print("No UMAP found. Computing UMAP.")
        sc.tl.umap(adata)


def ensure_velocity_embedding(adata):
    ensure_neighbors_and_umap(adata)
    if PREFERRED_VELOCITY_OBSM in adata.obsm:
        print(f"Using vector field: adata.obsm['{PREFERRED_VELOCITY_OBSM}']")
        return PREFERRED_VELOCITY_OBSM
    if FALLBACK_VELOCITY_OBSM in adata.obsm:
        print(f"Using vector field: adata.obsm['{FALLBACK_VELOCITY_OBSM}']")
        return FALLBACK_VELOCITY_OBSM
    if not RUN_SCVELO_PREPROCESS_IF_NEEDED:
        raise ValueError(
            f"No velocity found. Expected adata.obsm['{PREFERRED_VELOCITY_OBSM}'] "
            f"or adata.obsm['{FALLBACK_VELOCITY_OBSM}']."
        )
    if "spliced" not in adata.layers or "unspliced" not in adata.layers:
        raise ValueError("Cannot compute scVelo fallback: spliced/unspliced layers missing.")
    scv.pp.filter_and_normalize(adata, min_shared_counts=20)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    scv.tl.velocity(adata, mode="stochastic")
    scv.tl.velocity_graph(adata)
    scv.tl.velocity_embedding(adata, basis="umap")
    return FALLBACK_VELOCITY_OBSM


def score_marker_set_per_cell(adata, genes, layer=None):
    genes = available_genes(adata, genes)
    if len(genes) == 0:
        return np.zeros(adata.n_obs, dtype=float)
    return np.nanmean(get_matrix(adata, genes=genes, layer=layer), axis=1)


def pick_root_cluster(adata, cluster_key):
    cl = adata.obs[cluster_key].astype(str).values
    cats = list(pd.Categorical(cl).categories)
    for c in cats:
        if normalize_known_label(c) == "Blood progenitors 1":
            return c
    progenitor_genes = available_genes(
        adata,
        mouse_marker_candidates["Blood progenitors 1"]
        + mouse_marker_candidates["Blood progenitors 2"],
    )
    if len(progenitor_genes) > 0:
        scores = score_marker_set_per_cell(adata, progenitor_genes, EXPRESSION_LAYER)
        cluster_scores = {}
        for c in cats:
            m = cl == c
            if m.sum() > 0:
                cluster_scores[c] = float(np.nanmean(scores[m]))
        best = max(cluster_scores, key=cluster_scores.get)
        print(f"Root chosen by progenitor marker score: {best}")
        return best
    return cats[0]


def compute_or_load_pseudotime(adata, cluster_key, pt_key="pseudotime"):
    candidate_pt_keys = [
        pt_key, "dpt_pseudotime", "diffusion_pseudotime",
        "velot_pseudotime", "VelOT_pseudotime",
        "velocity_pseudotime", "latent_time", "pseudotime_velot",
    ]
    for k in candidate_pt_keys:
        if k in adata.obs.columns:
            vals = pd.to_numeric(adata.obs[k], errors="coerce").values
            if np.isfinite(vals).sum() > 10:
                adata.obs[pt_key] = minmax(vals)
                print(f"Using existing pseudotime: adata.obs['{k}'] → adata.obs['{pt_key}']")
                return adata.obs[pt_key].values
    print("No usable pseudotime found. Computing DPT.")
    ensure_neighbors_and_umap(adata)
    sc.tl.diffmap(adata)
    root_cluster = pick_root_cluster(adata, cluster_key)
    clusters = adata.obs[cluster_key].astype(str).values
    root_cells = np.where(clusters == root_cluster)[0]
    root_idx = int(root_cells[0]) if len(root_cells) > 0 else 0
    adata.uns["iroot"] = root_idx
    sc.tl.dpt(adata, n_dcs=10)
    adata.obs[pt_key] = minmax(
        pd.to_numeric(adata.obs["dpt_pseudotime"], errors="coerce").values
    )
    print(f"Computed DPT pseudotime with root cluster: {root_cluster}")
    return adata.obs[pt_key].values


velocity_key = ensure_velocity_embedding(adata)
compute_or_load_pseudotime(adata, cluster_key, pt_key)


# ============================================================
# Main path
# ============================================================

def build_main_path(adata, cluster_key):
    clusters = list(pd.Categorical(adata.obs[cluster_key].astype(str)).categories)
    path = []
    for c in canonical_state_order:
        actual = find_actual_cluster_for_canonical(c, clusters)
        if actual is not None:
            path.append(actual)
    if len(path) >= 2:
        return "Blood progenitors → erythroid maturation", path
    pt = adata.obs[pt_key].astype(float).values
    cl = adata.obs[cluster_key].astype(str).values
    med = {}
    for c in clusters:
        med[c] = float(np.nanmedian(pt[cl == c]))
    path = sorted(clusters, key=lambda x: med.get(x, np.inf))
    return "All clusters ordered by pseudotime", path


main_path_name, main_path_clusters = build_main_path(adata, cluster_key)
print("\nMain path:", main_path_name)
for c in main_path_clusters:
    print("  -", c)


# ============================================================
# Gene selection
# ============================================================

def top_markers_for_cluster(adata, cluster, cluster_key, n=25, layer=None, min_cells=10):
    groups = adata.obs[cluster_key].astype(str).values
    mask = groups == str(cluster)
    if mask.sum() < min_cells or (~mask).sum() < min_cells:
        return []
    X = get_matrix(adata, layer=layer)
    mu_in = X[mask].mean(axis=0)
    mu_out = X[~mask].mean(axis=0)
    frac_in = (X[mask] > 0).mean(axis=0)
    score = (mu_in - mu_out) * np.sqrt(frac_in + 1e-6)
    order = np.argsort(score)[::-1]
    genes = []
    for idx in order:
        if score[idx] <= 0:
            break
        genes.append(str(adata.var_names[idx]))
        if len(genes) >= n:
            break
    return genes


def select_lineage_genes(adata, lineage_clusters, marker_candidates,
                         cluster_key, n_auto_per_cluster=25, max_genes=160):
    genes = []
    for cl in lineage_clusters:
        genes += marker_candidates.get(cl, [])
        genes += top_markers_for_cluster(adata, cl, cluster_key,
                                         n=n_auto_per_cluster, layer=EXPRESSION_LAYER)
    genes = available_genes(adata, unique_preserve_order(genes))
    return genes[:max_genes]


heatmap_genes = select_lineage_genes(
    adata, main_path_clusters, marker_candidates_by_cluster,
    cluster_key, n_auto_per_cluster=26, max_genes=160,
)
if len(heatmap_genes) < 5:
    raise ValueError("Too few heatmap genes selected.")

trend_gene_candidates = [
    "Kit", "Gata2", "Gata1", "Klf1",
    "Tfrc", "Alas2", "Gypa", "Hbb-bs",
    "Hbb-bt", "Hba-a1", "Slc4a1", "Ahsp",
]
trend_genes = available_genes(adata, trend_gene_candidates)
if len(trend_genes) < 8:
    trend_genes = unique_preserve_order(
        trend_genes + available_genes(adata, BROAD_MOUSE_ERYTHROID_GENES)
    )
trend_genes = trend_genes[:8]

label_genes = available_genes(adata, [
    "Cd34", "Kit", "Hlf", "Meis1", "Runx1", "Gata2", "Lmo2",
    "Tal1", "Gata1", "Klf1", "Epor", "Tfrc", "Cd36",
    "Alas2", "Fech", "Gypa", "Ahsp", "Car1", "Slc4a1",
    "Rhag", "Ank1", "Hba-a1", "Hba-a2", "Hbb-bs", "Hbb-bt",
])

print("\nHeatmap genes:", len(heatmap_genes))
print("Trend genes:", trend_genes)
print("Label genes:", label_genes)


# ============================================================
# Vector-field summaries and per-cell flow features
# ============================================================

def binned_gene_matrix_with_flow(
    adata, cell_mask, genes, velocity_key,
    n_bins=90, smooth_sigma=1.25, layer=None, z_clip=2.5,
):
    obs_idx = np.where(cell_mask)[0]
    pt = adata.obs[pt_key].astype(float).values[obs_idx]
    X_emb = np.asarray(adata.obsm[basis_key], dtype=float)[obs_idx]
    V_emb = np.asarray(adata.obsm[velocity_key], dtype=float)[obs_idx]

    ok = (np.isfinite(pt)
          & np.all(np.isfinite(X_emb), axis=1)
          & np.all(np.isfinite(V_emb), axis=1))
    obs_idx, pt, X_emb, V_emb = obs_idx[ok], pt[ok], X_emb[ok], V_emb[ok]
    order = np.argsort(pt)
    obs_idx, pt, X_emb, V_emb = obs_idx[order], pt[order], X_emb[order], V_emb[order]

    X_expr = get_matrix(adata[obs_idx, :], genes=genes, layer=layer)

    n_bins_eff = min(n_bins, max(15, len(obs_idx) // 6))
    edges = np.linspace(0, len(obs_idx), n_bins_eff + 1).astype(int)

    mats, centers, mean_xy, mean_v, bin_slices = [], [], [], [], []
    for i in range(n_bins_eff):
        s, e = edges[i], edges[i + 1]
        if e <= s:
            continue
        bin_slices.append((s, e))
        mats.append(X_expr[s:e].mean(axis=0))
        centers.append(np.nanmean(pt[s:e]))
        mean_xy.append(np.nanmean(X_emb[s:e], axis=0))
        mean_v.append(np.nanmean(V_emb[s:e], axis=0))

    M = np.vstack(mats).T
    centers = np.asarray(centers)
    mean_xy = np.vstack(mean_xy)
    mean_v = np.vstack(mean_v)

    tangent = np.gradient(mean_xy, axis=0)
    tangent_unit = tangent / (np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-12)

    speed, forward, coherence = [], [], []
    for i, (s, e) in enumerate(bin_slices):
        Vb = V_emb[s:e]
        Vnorm = np.linalg.norm(Vb, axis=1)
        Vunit = Vb / (Vnorm[:, None] + 1e-12)
        speed.append(np.nanmean(Vnorm))
        forward.append(np.nanmean(Vb @ tangent_unit[i]))
        coherence.append(np.linalg.norm(np.nanmean(Vunit, axis=0)))

    speed, forward = robust_interp(speed), robust_interp(forward)
    coherence = np.clip(robust_interp(coherence), 0, 1)

    if smooth_sigma and smooth_sigma > 0:
        M = gaussian_filter1d(M, sigma=smooth_sigma, axis=1, mode="nearest")
        speed = gaussian_filter1d(speed, sigma=smooth_sigma, mode="nearest")
        forward = gaussian_filter1d(forward, sigma=smooth_sigma, mode="nearest")
        coherence = gaussian_filter1d(coherence, sigma=smooth_sigma, mode="nearest")

    Z = (M - M.mean(axis=1, keepdims=True)) / (M.std(axis=1, keepdims=True) + 1e-6)
    Z = np.clip(Z, -z_clip, z_clip)

    flow = {
        "centers": centers,
        "forward": np.asarray(forward),
        "forward_scaled": symmetric_scale(forward),
        "speed": minmax(speed),
        "speed_raw": np.asarray(speed),
        "coherence": np.clip(coherence, 0, 1),
        "mean_xy": mean_xy,
        "mean_v": mean_v,
        "tangent_unit": tangent_unit,
    }
    return M, Z, centers, flow


def compute_per_cell_flow_features(adata, cell_mask, velocity_key, n_bins=80):
    pt_all = adata.obs[pt_key].astype(float).values
    X_all = np.asarray(adata.obsm[basis_key], dtype=float)
    V_all = np.asarray(adata.obsm[velocity_key], dtype=float)

    valid = (cell_mask & np.isfinite(pt_all)
             & np.all(np.isfinite(X_all), axis=1)
             & np.all(np.isfinite(V_all), axis=1))
    idx = np.where(valid)[0]
    pt, X, V = pt_all[idx], X_all[idx], V_all[idx]
    order = np.argsort(pt)
    idx_sorted, pt_sorted = idx[order], pt[order]
    X_sorted, V_sorted = X[order], V[order]

    edges = np.linspace(0, len(idx_sorted), n_bins + 1).astype(int)
    centers, mean_xy, bin_slices = [], [], []
    for i in range(n_bins):
        s, e = edges[i], edges[i + 1]
        if e <= s:
            continue
        bin_slices.append((s, e))
        centers.append(np.nanmean(pt_sorted[s:e]))
        mean_xy.append(np.nanmean(X_sorted[s:e], axis=0))
    centers = np.asarray(centers)
    mean_xy = np.vstack(mean_xy)
    tangent = np.gradient(mean_xy, axis=0)
    tangent_unit = tangent / (np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-12)

    per_cell_forward = np.full(adata.n_obs, np.nan)
    per_cell_speed = np.full(adata.n_obs, np.nan)
    per_cell_coherence = np.full(adata.n_obs, np.nan)

    for i, (s, e) in enumerate(bin_slices):
        cells = idx_sorted[s:e]
        Vb = V_sorted[s:e]
        Vnorm = np.linalg.norm(Vb, axis=1)
        Vunit = Vb / (Vnorm[:, None] + 1e-12)
        t = tangent_unit[i]
        per_cell_forward[cells] = Vb @ t
        per_cell_speed[cells] = Vnorm
        per_cell_coherence[cells] = np.linalg.norm(np.nanmean(Vunit, axis=0))

    adata.obs["velot_v_to_pt"] = per_cell_forward
    adata.obs["velot_v_to_pt_scaled"] = symmetric_scale(per_cell_forward)
    adata.obs["velot_speed"] = per_cell_speed
    adata.obs["velot_speed_scaled"] = minmax(per_cell_speed)
    adata.obs["velot_local_coherence"] = per_cell_coherence


main_mask = adata.obs[cluster_key].astype(str).isin(main_path_clusters).values

compute_per_cell_flow_features(
    adata, cell_mask=main_mask,
    velocity_key=velocity_key, n_bins=N_FLOW_BINS,
)


# ============================================================
# Biological program annotation of heatmap modules
# ============================================================

def order_genes_by_modules(Z, n_modules=5):
    n_genes = Z.shape[0]
    if n_genes < 5:
        return np.arange(n_genes), np.ones(n_genes, dtype=int)
    k = min(n_modules, max(2, n_genes // 14))
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=30)
    labels0 = km.fit_predict(Z)
    module_peak = {m: int(np.argmax(Z[labels0 == m].mean(axis=0)))
                   for m in np.unique(labels0)}
    module_order = sorted(module_peak, key=lambda m: module_peak[m])
    ordered_rows = []
    relabeled = np.zeros_like(labels0)
    for new_label, old_label in enumerate(module_order, start=1):
        rows = np.where(labels0 == old_label)[0]
        rows = rows[np.argsort(np.argmax(Z[rows], axis=1))]
        ordered_rows.extend(rows)
        relabeled[labels0 == old_label] = new_label
    return np.asarray(ordered_rows), relabeled


def infer_program_label_for_module(module_genes, module_profile, centers):
    genes_upper = {str(g).upper() for g in module_genes}
    scores = {}
    for pname, pgenes in program_panels.items():
        p_upper = {str(g).upper() for g in available_genes(adata, pgenes)}
        scores[pname] = len(genes_upper.intersection(p_upper))
    best_program = max(scores, key=scores.get)
    if scores[best_program] > 0:
        return best_program
    peak_pt = centers[int(np.argmax(module_profile))]
    if peak_pt < 0.25:
        return "Early dynamic"
    if peak_pt < 0.65:
        return "Intermediate dynamic"
    return "Late dynamic"


def annotate_ordered_modules(Z_ord, genes_ord, modules_ord, centers):
    module_info = []
    for m in np.unique(modules_ord):
        rows = np.where(modules_ord == m)[0]
        module_genes = genes_ord[rows]
        profile = Z_ord[rows, :].mean(axis=0)
        label = infer_program_label_for_module(module_genes, profile, centers)
        module_info.append({
            "module": int(m),
            "label": label,
            "color": program_colors.get(label, "#bdbdbd"),
            "row_start": int(rows.min()),
            "row_end": int(rows.max()),
            "row_mid": float((rows.min() + rows.max()) / 2),
            "n_genes": int(len(rows)),
            "peak_pt": float(centers[int(np.argmax(profile))]),
        })
    return module_info


# ============================================================
# Plot helpers (each draws on a provided Axes)
# ============================================================

def plot_vector_field_inset(ax, adata, cell_mask, velocity_key, flow=None,
                            grid_n=20, min_cells_per_bin=3):
    X = np.asarray(adata.obsm[basis_key], dtype=float)
    V = np.asarray(adata.obsm[velocity_key], dtype=float)
    pt = adata.obs[pt_key].astype(float).values
    mask = (cell_mask & np.isfinite(pt)
            & np.all(np.isfinite(X), axis=1)
            & np.all(np.isfinite(V), axis=1))
    cl = adata.obs[cluster_key].astype(str).values

    ax.scatter(X[:, 0], X[:, 1], s=1.2, color="0.88", alpha=0.30,
               linewidths=0, rasterized=True)
    for state in main_path_clusters:
        m = mask & (cl == state)
        if m.sum() == 0:
            continue
        ax.scatter(X[m, 0], X[m, 1], s=2.6,
                   color=state_color_map.get(state, "#999999"),
                   alpha=0.75, linewidths=0, rasterized=True)

    Xm, Vm = X[mask], V[mask]
    if Xm.shape[0] > 0:
        x_edges = np.linspace(np.nanmin(Xm[:, 0]), np.nanmax(Xm[:, 0]), grid_n + 1)
        y_edges = np.linspace(np.nanmin(Xm[:, 1]), np.nanmax(Xm[:, 1]), grid_n + 1)
        qx, qy, qu, qv = [], [], [], []
        for i in range(grid_n):
            for j in range(grid_n):
                b = ((Xm[:, 0] >= x_edges[i]) & (Xm[:, 0] < x_edges[i + 1])
                     & (Xm[:, 1] >= y_edges[j]) & (Xm[:, 1] < y_edges[j + 1]))
                if b.sum() < min_cells_per_bin:
                    continue
                qx.append(np.nanmean(Xm[b, 0]))
                qy.append(np.nanmean(Xm[b, 1]))
                qu.append(np.nanmean(Vm[b, 0]))
                qv.append(np.nanmean(Vm[b, 1]))
        qx, qy = np.asarray(qx), np.asarray(qy)
        qu, qv = np.asarray(qu), np.asarray(qv)
        if len(qx) > 0:
            norms = np.sqrt(qu ** 2 + qv ** 2)
            span = max(np.ptp(Xm[:, 0]), np.ptp(Xm[:, 1]), 1e-6)
            med_norm = np.nanmedian(norms[norms > 0]) if np.any(norms > 0) else 1.0
            ax.quiver(qx, qy, qu, qv, angles="xy", scale_units="xy",
                      scale=med_norm / (0.05 * span),
                      width=0.0055, color="black", alpha=0.55, zorder=5)

    if flow is not None and "mean_xy" in flow:
        path = flow["mean_xy"]
        ax.plot(path[:, 0], path[:, 1], color="black", lw=1.7, alpha=0.92, zorder=6)
        ax.scatter(path[0, 0], path[0, 1], s=24, color="white",
                   edgecolor="black", linewidth=0.8, zorder=7)
        ax.scatter(path[-1, 0], path[-1, 1], s=25, color="black",
                   edgecolor="black", linewidth=0.8, zorder=7)

    # ax.set_title("UMAP VelOT vector field", fontsize=9, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def place_spaced_gene_labels(ax, genes_ord, label_genes, min_sep=4):
    pos = {g: i for i, g in enumerate(genes_ord)}
    candidates = sorted([(pos[g], g) for g in label_genes if g in pos])
    placed = []
    for y, g in candidates:
        if len(placed) == 0 or y - placed[-1][0] >= min_sep:
            placed.append((y, g))
    ax.set_xlim(0, 1)
    ax.set_ylim(len(genes_ord) - 0.5, -0.5)
    ax.axis("off")
    for y, g in placed:
        ax.plot([0.00, 0.08], [y, y], color="0.25", lw=0.4)
        ax.text(0.10, y, g, va="center", ha="left", fontsize=10, fontstyle="italic")


def flow_profile_by_pseudotime(adata, cell_mask, velocity_key, n_bins=28):
    pt = adata.obs[pt_key].astype(float).values
    X_emb = np.asarray(adata.obsm[basis_key], dtype=float)
    V_emb = np.asarray(adata.obsm[velocity_key], dtype=float)
    mask = (cell_mask & np.isfinite(pt)
            & np.all(np.isfinite(X_emb), axis=1)
            & np.all(np.isfinite(V_emb), axis=1))
    x, xe, ve = pt[mask], X_emb[mask], V_emb[mask]
    bins = np.linspace(0, 1, n_bins + 1)
    mean_xy, bin_indices = [], []
    for i in range(n_bins):
        bmask = (x >= bins[i]) & (x <= bins[i + 1] if i == n_bins - 1 else x < bins[i + 1])
        if bmask.sum() < 5:
            mean_xy.append([np.nan, np.nan])
            bin_indices.append(None)
        else:
            mean_xy.append(np.nanmean(xe[bmask], axis=0))
            bin_indices.append(np.where(bmask)[0])
    mean_xy = np.asarray(mean_xy, dtype=float)
    for d in range(2):
        mean_xy[:, d] = robust_interp(mean_xy[:, d])
    tangent = np.gradient(mean_xy, axis=0)
    tangent_unit = tangent / (np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-12)
    forward, speed, coherence = [], [], []
    for i in range(n_bins):
        idx = bin_indices[i]
        if idx is None:
            forward.append(np.nan); speed.append(np.nan); coherence.append(np.nan)
            continue
        Vb = ve[idx]
        Vnorm = np.linalg.norm(Vb, axis=1)
        Vunit = Vb / (Vnorm[:, None] + 1e-12)
        forward.append(np.nanmean(Vb @ tangent_unit[i]))
        speed.append(np.nanmean(Vnorm))
        coherence.append(np.linalg.norm(np.nanmean(Vunit, axis=0)))
    return {
        "centers": 0.5 * (bins[:-1] + bins[1:]),
        "forward_scaled": symmetric_scale(robust_interp(forward)),
        "speed": minmax(robust_interp(speed)),
        "coherence": np.clip(robust_interp(coherence), 0, 1),
    }


def plot_gene_trend_with_error_and_flow(ax, adata, gene, groups_to_show,
                                         velocity_key, n_bins=28, smooth_sigma=1.0):
    expr = get_matrix(adata, genes=[gene], layer=EXPRESSION_LAYER).ravel()
    pt = adata.obs[pt_key].astype(float).values
    group_values = adata.obs[cluster_key].astype(str).values

    for group in groups_to_show:
        color = state_color_map.get(group, "#999999")
        mask = ((group_values == str(group)) & np.isfinite(pt) & np.isfinite(expr))
        if mask.sum() < 10:
            continue
        x, y = pt[mask], expr[mask]
        bins = np.linspace(0, 1, n_bins + 1)
        centers, means, sems = [], [], []
        for i in range(n_bins):
            bmask = (x >= bins[i]) & (x <= bins[i + 1] if i == n_bins - 1 else x < bins[i + 1])
            centers.append((bins[i] + bins[i + 1]) / 2)
            if bmask.sum() < 4:
                means.append(np.nan); sems.append(np.nan)
            else:
                vals = y[bmask]
                means.append(np.nanmean(vals))
                sems.append(np.nanstd(vals) / np.sqrt(len(vals)))
        centers = np.asarray(centers)
        means, sems = robust_interp(means), robust_interp(sems)
        if smooth_sigma and smooth_sigma > 0:
            means = gaussian_filter1d(means, sigma=smooth_sigma, mode="nearest")
            sems = gaussian_filter1d(sems, sigma=smooth_sigma, mode="nearest")
        ax.plot(centers, means, lw=1.9, color=color, label=group)
        ax.fill_between(centers, means - sems, means + sems,
                        color=color, alpha=0.17, linewidth=0)

    show_mask = np.isin(group_values, [str(g) for g in groups_to_show])
    flow = flow_profile_by_pseudotime(adata, show_mask, velocity_key, n_bins)
    ax_strip = ax.inset_axes([0.0, 1.035, 1.0, 0.075])
    ax_strip.imshow(flow["forward_scaled"][None, :], aspect="auto", cmap="coolwarm",
                    interpolation="nearest", vmin=-1, vmax=1, extent=[0, 1, 0, 1])
    ax_strip.set_xticks([]); ax_strip.set_yticks([])
    ax_strip.text(-0.018, 0.5, "v→pt", transform=ax_strip.transAxes,
                  ha="right", va="center", fontsize=6)
    for spine in ax_strip.spines.values():
        spine.set_visible(False)
    ax.set_title(gene, fontstyle="italic")
    ax.set_xlabel("pseudotime")
    ax.set_ylabel("expression")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_violin_by_state(ax, adata, value_key, groups_to_show,
                          ylabel, title, add_zero=True):
    cl = adata.obs[cluster_key].astype(str).values
    data, colors = [], []
    for group in groups_to_show:
        vals = pd.to_numeric(adata.obs[value_key], errors="coerce").values
        m = (cl == group) & np.isfinite(vals)
        data.append(vals[m] if m.sum() > 0 else np.array([np.nan]))
        colors.append(state_color_map.get(group, "#999999"))
    parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color); body.set_edgecolor("black")
        body.set_alpha(0.75); body.set_linewidth(0.5)
    if "cmedians" in parts:
        parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(1.0)
    for i, vals in enumerate(data, start=1):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        rng = np.random.default_rng(RANDOM_STATE + i)
        vals_show = rng.choice(vals, size=min(500, len(vals)), replace=False) if len(vals) > 0 else vals
        jitter = rng.normal(0, 0.035, len(vals_show))
        ax.scatter(np.full(len(vals_show), i) + jitter, vals_show,
                   s=3, color="black", alpha=0.12, linewidths=0, rasterized=True)
    if add_zero:
        ax.axhline(0, color="black", lw=0.8, alpha=0.35)
    ax.set_xticks(np.arange(1, len(groups_to_show) + 1))
    ax.set_xticklabels(groups_to_show, rotation=35, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def build_gene_feature_correlation_table(adata, genes, cell_mask):
    pt = adata.obs[pt_key].astype(float).values
    features = {
        "pseudotime": pt,
        "v→pt": pd.to_numeric(adata.obs["velot_v_to_pt"], errors="coerce").values,
        "|v|": pd.to_numeric(adata.obs["velot_speed"], errors="coerce").values,
        "coh.": pd.to_numeric(adata.obs["velot_local_coherence"], errors="coerce").values,
    }
    X = get_matrix(adata, genes=genes, layer=EXPRESSION_LAYER)
    mat = np.zeros((len(genes), len(features)), dtype=float)
    rows = []
    for i, gene in enumerate(genes):
        for j, (fname, fvals) in enumerate(features.items()):
            rho, p = safe_spearman(X[:, i][cell_mask], fvals[cell_mask], min_n=20)
            mat[i, j] = rho
            rows.append({"gene": gene, "feature": fname, "spearman_rho": rho, "p_value": p})
    return mat, list(features.keys()), pd.DataFrame(rows)


def build_celltype_pseudotime_correlation_table(adata, genes, groups_to_show):
    pt = adata.obs[pt_key].astype(float).values
    cl = adata.obs[cluster_key].astype(str).values
    X = get_matrix(adata, genes=genes, layer=EXPRESSION_LAYER)
    mat = np.zeros((len(genes), len(groups_to_show)), dtype=float)
    records = []
    for i, gene in enumerate(genes):
        for j, group in enumerate(groups_to_show):
            m = (cl == group) & np.isfinite(pt) & np.isfinite(X[:, i])
            rho, p = safe_spearman(X[:, i][m], pt[m], min_n=10)
            mat[i, j] = rho
            records.append({"gene": gene, "cell_state": group,
                            "spearman_rho": rho, "p_value": p, "n_cells": int(m.sum())})
    return mat, pd.DataFrame(records)


def plot_correlation_heatmap(ax, mat, row_labels, col_labels, title,
                              cbar_label="Spearman ρ"):
    mat = np.asarray(mat, dtype=float)
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1,
                   interpolation="nearest")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontstyle="italic")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color="black")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(cbar_label, fontsize=7)
    cb.ax.tick_params(labelsize=6)


# ============================================================
# Save: heatmap panel (standalone figure with its own GridSpec)
# ============================================================

def save_heatmap_panel(adata, title, lineage_clusters, genes, velocity_key,
                       label_genes, output_path, dpi=300):
    cell_mask = adata.obs[cluster_key].astype(str).isin(lineage_clusters).values

    _, Z, centers, flow = binned_gene_matrix_with_flow(
        adata, cell_mask=cell_mask, genes=genes, velocity_key=velocity_key,
        n_bins=N_HEATMAP_BINS, smooth_sigma=1.25, layer=EXPRESSION_LAYER,
    )

    row_order, modules = order_genes_by_modules(Z, n_modules=5)
    Z_ord = Z[row_order, :]
    genes_ord = np.asarray(genes)[row_order]
    modules_ord = modules[row_order]
    module_info = annotate_ordered_modules(Z_ord, genes_ord, modules_ord, centers)

    fig = plt.figure(figsize=(20,8))
    gs = GridSpec(
        5, 4, figure=fig,
        height_ratios=[0.06, 0.06, 0.06, 0.06, 0.76],
        width_ratios=[0.12, 0.82, 0.03, 0.02],
        hspace=0.025, wspace=0.04,
    )

    ax_program = fig.add_subplot(gs[4, 0])
    # ax_left = fig.add_subplot(gs[4, 1])
    ax_hm = fig.add_subplot(gs[4, 1])
    ax_lab = fig.add_subplot(gs[4, 2])
    ax_cb = fig.add_subplot(gs[4, 3])
    ax_pt = fig.add_subplot(gs[0, 1])
    ax_forward = fig.add_subplot(gs[1, 1])
    ax_speed = fig.add_subplot(gs[2, 1])
    ax_coh = fig.add_subplot(gs[3, 1])

    # --- flow bars ---
    ax_pt.imshow(centers[None, :], aspect="auto", cmap="YlGnBu", interpolation="nearest")
    # ax_pt.set_title(title, fontsize=10, pad=4)
    ax_forward.imshow(flow["forward_scaled"][None, :], aspect="auto", cmap="coolwarm",
                      interpolation="nearest", vmin=-1, vmax=1)
    ax_speed.imshow(flow["speed"][None, :], aspect="auto", cmap="magma",
                    interpolation="nearest", vmin=0, vmax=1)
    ax_coh.imshow(flow["coherence"][None, :], aspect="auto", cmap="viridis",
                  interpolation="nearest", vmin=0, vmax=1)
    for ax_bar, lab in zip([ax_pt, ax_forward, ax_speed, ax_coh],
                           ["pt", "v→pt", "|v|", "coh."]):
        ax_bar.set_xticks([]); ax_bar.set_yticks([])
        ax_bar.text(-0.018, 0.5, lab, transform=ax_bar.transAxes,
                    ha="right", va="center", fontsize=10)
        for spine in ax_bar.spines.values():
            spine.set_visible(False)

    # --- program block labels ---
    ax_program.set_xlim(0, 1)
    ax_program.set_ylim(len(genes_ord) - 0.5, -0.5)
    ax_program.axis("off")
    program_color_vector = []
    for g, mod in zip(genes_ord, modules_ord):
        info = [x for x in module_info if x["module"] == int(mod)][0]
        program_color_vector.append(info["color"])
    for info in module_info:
        ax_program.axhspan(info["row_start"] - 0.5, info["row_end"] + 0.5,
                           color=info["color"], alpha=0.90, linewidth=0)
        ax_program.text(0.98, info["row_mid"],
                        f"{info['label']}\n({info['n_genes']})",
                        ha="right", va="center", fontsize=10, color="black")

    # --- thin color bar ---
    # ax_left.imshow(np.arange(len(genes_ord))[:, None], aspect="auto",
    #                cmap=ListedColormap(program_color_vector), interpolation="nearest")
    # ax_left.set_xticks([]); ax_left.set_yticks([])

    # --- main heatmap ---
    im = ax_hm.imshow(Z_ord, aspect="auto", cmap="RdYlBu_r",
                      interpolation="nearest", vmin=-2.5, vmax=2.5)
    for info in module_info:
        ax_hm.axhline(info["row_start"] - 0.5, color="black", lw=0.45, alpha=0.45)
    ax_hm.set_xticks([]); ax_hm.set_yticks([])
    ax_hm.set_xlabel("Pseudotime (Blood progenitors → erythroid maturation)", fontsize=10)
    for frac in [0.25, 0.5, 0.75]:
        ax_hm.axvline(frac * (Z_ord.shape[1] - 1), color="white", lw=0.35, alpha=0.75)

    # --- gene labels ---
    place_spaced_gene_labels(ax_lab, genes_ord, label_genes, min_sep=4)

    # --- colorbar ---
    cb = fig.colorbar(im, cax=ax_cb)
    # cb.set_label("z-scored\nexpression", fontsize=7)
    cb.ax.set_title("z-scored\nexpression", fontsize=7, pad=4)
    cb.ax.tick_params(labelsize=6)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")

    return Z_ord, genes_ord, modules_ord, module_info, flow


# ============================================================
# Save all individual plots
# ============================================================

print("\n" + "=" * 60)
print("Saving individual plots...")
print("=" * 60)

# --- 1. Heatmap with program annotations and flow bars ---
Z_ord, genes_ord, modules_ord, module_info, flow = save_heatmap_panel(
    adata=adata,
    title=main_path_name,
    lineage_clusters=main_path_clusters,
    genes=heatmap_genes,
    velocity_key=velocity_key,
    label_genes=label_genes,
    output_path=os.path.join(OUTDIR, "pseudotime_a.png"),
    dpi=DPI,
)

# # --- 2. UMAP vector field ---
# fig, ax = plt.subplots(figsize=(5,5))
# plot_vector_field_inset(ax, adata, main_mask, velocity_key, flow)
# fig.savefig(os.path.join(OUTDIR, "pseudotime_b.png"),
#             dpi=DPI, bbox_inches="tight")
# plt.close(fig)
# print(f"Saved: {os.path.join(OUTDIR, 'umap_vector_field.png')}")

# # --- 3. Gene trends (individual + combined 2×4 panel) ---

# # # Individual plots
# # for gene in trend_genes:
# #     fig, ax = plt.subplots(figsize=(4.5, 3.5))
# #     plot_gene_trend_with_error_and_flow(
# #         ax, adata, gene, main_path_clusters,
# #         velocity_key, n_bins=N_TREND_BINS, smooth_sigma=1.0,
# #     )
# #     ax.legend(frameon=False, fontsize=7, loc="best")
# #     path = os.path.join(OUTDIR, f"trend_{gene}.png")
# #     fig.savefig(path, dpi=DPI, bbox_inches="tight")
# #     plt.close(fig)
# #     print(f"Saved: {path}")

# # Combined 2×4 panel
# n_cols, n_rows = 4, 2
# fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3*n_rows))
# axes_flat = axes.flatten()

# for idx, gene in enumerate(trend_genes[:n_rows * n_cols]):
#     ax = axes_flat[idx]
#     plot_gene_trend_with_error_and_flow(
#         ax, adata, gene, main_path_clusters,
#         velocity_key, n_bins=N_TREND_BINS, smooth_sigma=1.0,
#     )

# # Hide unused axes
# for idx in range(len(trend_genes), n_rows * n_cols):
#     axes_flat[idx].set_visible(False)

# # Single shared legend at the bottom
# handles = [plt.Line2D([0], [0], color=state_color_map.get(s, "#999"), lw=2.5, label=s)
#            for s in main_path_clusters]
# fig.legend(
#     handles, main_path_clusters,
#     title="cell state",
#     frameon=False,
#     loc="lower center",
#     bbox_to_anchor=(0.5, -0.02),
#     ncol=len(main_path_clusters),
#     fontsize=9,
#     title_fontsize=10,
# )

# fig.subplots_adjust(hspace=0.5, wspace=0.2)
# path = os.path.join(OUTDIR, "pseudotime_c.png")
# fig.savefig(path, dpi=DPI, bbox_inches="tight")
# plt.close(fig)
# print(f"Saved: {path}")

# # --- 4. Violin plots ---
# violin_specs = [
#     ("velot_v_to_pt", "v→pt",
#      "", True,
#      "pseudotime_d.png"),
#     ("velot_speed", "|v|",
#      "", False,
#      "pseudotime_e.png"),
#     ("velot_local_coherence", "coherence",
#      "", False,
#      "pseudotime_f.png"),
# ]
# for value_key, ylabel, title, add_zero, filename in violin_specs:
#     fig, ax = plt.subplots(figsize=(6,4))
#     plot_violin_by_state(ax, adata, value_key, main_path_clusters,
#                          ylabel, title, add_zero)
#     path = os.path.join(OUTDIR, filename)
#     fig.savefig(path, dpi=DPI, bbox_inches="tight")
#     plt.close(fig)
#     print(f"Saved: {path}")

# # --- 5. Correlation heatmaps ---
# corr_genes = trend_genes

# feature_corr_mat, feature_names, feature_corr_df = \
#     build_gene_feature_correlation_table(adata, corr_genes, main_mask)

# celltype_pt_corr_mat, celltype_pt_corr_df = \
#     build_celltype_pseudotime_correlation_table(adata, corr_genes, main_path_clusters)

# fig, ax = plt.subplots(figsize=(7, 4))
# plot_correlation_heatmap(
#     ax, feature_corr_mat, corr_genes, feature_names,
#     title="",
# )
# path = os.path.join(OUTDIR, "pseudotime_g.png")
# fig.savefig(path, dpi=DPI, bbox_inches="tight")
# plt.close(fig)
# print(f"Saved: {path}")

# fig, ax = plt.subplots(figsize=(7, 4))
# plot_correlation_heatmap(
#     ax, celltype_pt_corr_mat, corr_genes, main_path_clusters,
#     title="",
# )
# path = os.path.join(OUTDIR, "pseudotime_h.png")
# fig.savefig(path, dpi=DPI, bbox_inches="tight")
# plt.close(fig)
# print(f"Saved: {path}")

# # --- 6. Cell-state legend (reusable for panel assembly) ---
# fig_leg, ax_leg = plt.subplots(figsize=(4, 1.5))
# ax_leg.axis("off")
# handles = [plt.Line2D([0], [0], color=state_color_map.get(s, "#999"), lw=4, label=s)
#            for s in main_path_clusters]
# ax_leg.legend(handles=handles, title="Cell state", frameon=False,
#               loc="center", fontsize=9, title_fontsize=10)
# path = os.path.join(OUTDIR, "legend_cell_states.png")
# fig_leg.savefig(path, dpi=DPI, bbox_inches="tight")
# plt.close(fig_leg)
# print(f"Saved: {path}")

# --- Done ---
print(f"\nAll individual plots saved to: {OUTDIR}/")
# VelOT — RNA Velocity via Optimal Transport

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%E2%80%933.11-informational.svg" />
  <img src="https://img.shields.io/badge/AnnData-.h5ad%20native-blueviolet.svg" />
  <img src="https://img.shields.io/badge/Scanpy-compatible-brightgreen.svg" />
  <a href="https://pypi.org/project/velot/">
    <img src="https://img.shields.io/pypi/v/velot.svg" alt="PyPI">
  </a>
</p>

<p align="center">
  <a href="LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg">
  </a>
  <a href="https://velot.readthedocs.io">
    <img alt="Documentation" src="https://readthedocs.org/projects/velot/badge/?version=latest">
  </a>
  <a href="https://doi.org/10.5281/zenodo.20542878">
    <img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.20542878.svg">
  </a>
  <a href="https://doi.org/10.5281/zenodo.20530116">
    <img alt="Data DOI" src="https://img.shields.io/badge/Zenodo-data-blue.svg">
  </a>
</p>

**VelOT** is a kinetic-free framework for estimating RNA velocity from
single-cell transcriptomic data. Instead of modeling the molecular
kinetics of splicing and degradation, VelOT treats velocity inference as
an optimal transport problem in gene expression space, producing smooth,
continuous velocity fields without requiring spliced/unspliced count
matrices.

> **Paper:** Rincón de la Rosa L, Pérez García D, Alentorn A.
> *VelOT: kinetic-free RNA velocity inference via optimal transport,
> flow-field smoothing, and VAMP coarse-graining of cellular dynamics.*
> bioRxiv (2026).
> [https://doi.org/10.64898/2026.06.04.730132](https://doi.org/10.64898/2026.06.04.730132)

<p align="center">
  <img src="article/figures/figure 1/panel1.png" width="800">
</p>

## Highlights

- **No kinetic assumptions** — does not require spliced/unspliced quantification or kinetic parameter fitting.
- **Optimal transport** — estimates cell-to-cell transitions via entropy-regularized OT within spatial-temporal windows.
- **Continuous velocity field** — a neural network learns a smooth, queryable velocity field over the full expression manifold.
- **Fast** — full pipeline runs in seconds to minutes, significantly faster than dynamical scVelo or other deep-learning methods.
- **Scanpy-compatible** — follows the `pp` / `tl` / `pl` API convention. Works with standard AnnData objects.

## Installation

> PyTorch is installed separately to allow users to choose the appropriate
> CUDA version for their system. See [pytorch.org](https://pytorch.org)
> for installation options.

### Quick install (recommended)

```bash
conda create -n velot python=3.10 -y
conda activate velot

# Install PyTorch (adjust for your CUDA version)
# GPU:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CPU only:
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install VelOT
pip install velot
```

### Developer installation

For contributing, modifying the code, or reproducing the paper:

```bash
# Create environment and install PyTorch first
conda create -n velot python=3.10 -y
conda activate velot

# Install PyTorch (adjust for your CUDA version — see https://pytorch.org)
# GPU:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CPU only:
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install VelOT
git clone https://github.com/lucas-rdlr/velot.git
cd velot
pip install -e .
```

## Quick Start

```python
import velot

# Load a dataset
adata = velot.datasets.pancreas()

# Preprocess: normalize, HVG, PCA, KNN, UMAP, pseudotime
adata = velot.pp.prepare(adata, root_cluster="Ductal", cluster_key="clusters")

# Compute velocity (windows → OT → smoothing → UMAP projection)
velot.tl.velocity(adata)

# Visualize
velot.pl.velocity_stream(adata, color="clusters")
```

## Pipeline Overview

VelOT proceeds in four stages, all operating in PCA space:

| Stage | Function | Description |
|-------|----------|-------------|
| **1. Preprocessing** | `velot.pp.prepare()` | Normalization, HVG selection, PCA, KNN graph, pseudotime via DPT |
| **2. Windowing** | `velot.tl.build_windows()` | Spatial clustering + temporal windowing for local OT |
| **3. OT Velocity** | `velot.tl.compute_ot_velocity()` | Entropy-regularized OT within each window pair |
| **4. Smoothing** | `velot.tl.smooth_velocity()` | Neural network learns a continuous velocity field |

The convenience function `velot.tl.velocity()` runs stages 2–4 in one call.

### Step-by-Step Usage

```python
import velot

adata = velot.datasets.pancreas()

# Step 1: Preprocessing
adata = velot.pp.prepare(adata, root_cluster="Ductal", cluster_key="clusters")

# Step 2: Build spatial-temporal windows
velot.tl.build_windows(adata, n_clusters=10, window_size=50)

# Step 3: Compute OT velocity
velot.tl.compute_ot_velocity(adata, reg=0.05, lambda_time=1.0, lambda_knn=1.0)

# Step 4: Smooth with neural network
velot.tl.smooth_velocity(adata, n_epochs=200, lambda_smooth=0.5)

# Step 5: Project to UMAP for visualization
velot.tl.project_to_umap(adata)

# Visualize
velot.pl.velocity_stream(adata, color="clusters")
velot.pl.confidence(adata)
velot.pl.training_curves(adata)
```

## Evaluation

VelOT includes built-in metrics for velocity quality assessment:

```python
edges = [
    ("Ngn3 low EP", "Ngn3 high EP"),
    ("Ngn3 high EP", "Fev+"),
    ("Fev+", "Alpha"),
    ("Fev+", "Beta"),
    ("Fev+", "Delta"),
    ("Fev+", "Epsilon"),
]

results = velot.metrics.summary(
    adata,
    cluster_edges=edges,
    cluster_key="clusters",
)

velot.pl.metric_summary(results)
```

**Metrics:**
- **ICCoh** (Inner-Cluster Coherence) — consistency of velocity directions within clusters.
- **CBDir** (Cross-Boundary Direction Correctness) — whether velocities at cluster boundaries point in the expected direction.

## Trajectory Analysis

```python
# Forward trajectories from progenitors
velot.tl.compute_trajectories(
    adata,
    start_cluster="Ngn3 low EP",
    direction="forward",
    n_trajectories=20,
)

velot.pl.trajectories(adata, color="clusters")
velot.pl.fate_summary(adata)
```

## Benchmarking

VelOT includes a benchmarking framework for comparing velocity methods:

```python
from velot.benchmark import BenchmarkTimer, save_benchmark

timer = BenchmarkTimer()

with timer("velocity"):
    velot.tl.velocity(adata)

with timer("evaluate"):
    results = velot.metrics.summary(adata, cluster_edges=edges)

save_benchmark(results, timer, model_name="velot", dataset_name="pancreas")
```

```python
# Compare across methods and datasets
from velot.benchmark import benchmark_dotplot
benchmark_dotplot("benchmark_results")
```

## Built-in Datasets

```python
# Real datasets (via scVelo)
adata = velot.datasets.pancreas()
adata = velot.datasets.erythroid()
adata = velot.datasets.dentategyrus()

# Synthetic datasets
adata = velot.datasets.synthetic_linear(densities=[200, 200, 200])
adata = velot.datasets.synthetic_bifurcation()
adata = velot.datasets.synthetic_cycle()
```

## Documentation

Full API documentation: [velot.readthedocs.io](https://velot.readthedocs.io)

For the detailed mathematical methodology, see the
[Methods section](https://velot.readthedocs.io/en/latest/methodology.html)
of the documentation or the accompanying paper.

## Data and Reproducibility

Scripts to reproduce all figures and benchmarks from the paper are in
the [`article/`](article/) directory.

| Resource | Link |
|----------|------|
| **Preprint** | [![bioRxiv](https://img.shields.io/badge/bioRxiv-2026.06.04.730132-b31b1b.svg)](https://doi.org/10.64898/2026.06.04.730132) |
| **Benchmark data, model weights, and frozen outputs** | [![Data DOI](https://img.shields.io/badge/Zenodo_data-10.5281%2Fzenodo.20542878-blue)](https://doi.org/10.5281/zenodo.20542878) |
| **Archived source code** | [![Code DOI](https://zenodo.org/badge/1231780756.svg)](https://doi.org/10.5281/zenodo.20542877) |
| **Documentation** | [![Docs](https://readthedocs.org/projects/velot/badge/?version=latest)](https://velot.readthedocs.io) |

## Citation

If you use VelOT in your research, please cite:

```bibtex
@article {velot2026,
	author = {Rincon de la Rosa, Lucas and Perez Garcia, David and Alentorn, Agusti},
	title = {VelOT: kinetic-free RNA velocity inference via optimal transport,
          flow-field smoothing, and VAMP coarse-graining of cellular dynamics},
	elocation-id = {2026.06.04.730132},
	year = {2026},
	doi = {10.64898/2026.06.04.730132},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2026/06/06/2026.06.04.730132},
	eprint = {https://www.biorxiv.org/content/early/2026/06/06/2026.06.04.730132.full.pdf},
	journal = {bioRxiv}
}
```

## License

VelOT is released under the [MIT License](LICENSE).

> **Note on torch**: Not listed in dependencies since it requires
> platform-specific installation (CUDA version). Users install it
> separately before installing VelOT. An import-time error in `tl.py`
> already catches this if torch is missing.

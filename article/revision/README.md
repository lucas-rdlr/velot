# Revision analyses (Patterns PATTERNS-D-26-00419)

Scripts written for the major-revision response. Run them in the
`velot_test` environment so they pick up the editable install.

```bash
conda activate velot_test
python ot_vs_gradient_diagnostics.py      # ~5-10 min
python metric_dimension_sensitivity.py    # ~1 min
python toy_window_transport.py            # ~10 s, figures in results/toy_window/
```

| script | reviewer comment | question it answers |
|---|---|---|
| `ot_vs_gradient_diagnostics.py` | R1.2, R2.2, R3.3, R3.5 | Why do OT and the kNN gradient agree, and where should they diverge? |
| `toy_window_transport.py` | R1.2, R2.2, R3.3 | Inside one window pair: which targets does each cell send mass to under OT, the window gradient and the kNN gradient? |
| `metric_dimension_sensitivity.py` | R1.1, R2.3, R3.4 | How much of an ICCoh score is set by the dimensionality of the space it was measured in? |

---

## `ot_vs_gradient_diagnostics.py`

Both estimators have the same form,

    v_i = sum_j w_ij (x_j - x_i)

and differ only in the weights `w`. The gradient uses uniform weights over
forward kNN neighbours. OT uses the row-normalised Sinkhorn plan with

    K_ij = exp(-C_ij / eps)
    C_ij = d2_ij/d2_max + lambda_time*1[tau_i > tau_j] + lambda_knn*1[j not in kNN(i)]

With `lambda_time = lambda_knn = 1` and `eps = 0.1`, each indicator term
multiplies the kernel by `exp(-10) ~ 5e-5`, so backward and off-manifold
targets are effectively excluded. **That leaves OT's support equal to the
gradient's support** — forward, on-manifold neighbours. Inside that set the
only remaining variation is `exp(-d2/(d2_max*eps))`, which is nearly flat
when the window is compact.

So the null hypothesis is that OT *is* the gradient on this data, and the
one thing it still does differently is enforce the **column marginal**:
every target cell must receive a fixed share of mass, which stops many
sources from piling onto the same attractive target. That constraint is
slack under uniform density and binds under imbalance.

Parts of the script:

* **A. Raw-level agreement.** Cosine between the two raw fields before any
  MLP. If this is ~1, the question is settled at the estimator level and
  nothing downstream can separate them.
* **B. Plan structure vs epsilon.** Effective support `exp(row entropy)`,
  top-1 mass, and `KL(plan || uniform-on-support)`. KL -> 0 means OT has
  literally collapsed onto the mean assignment.
* **C1-C5. Sweeps** over epsilon, ambient dimension, cell density, branch
  imbalance and gradient `k`. C4 is the one that matters: it is the regime
  where the marginal constraint should give OT an edge.
* **D. Magnitude.** Nothing is normalised anywhere in the pipeline, and the
  MLP regression term is an unweighted L2 on raw magnitudes, so a heavy
  tail lets a few cells dominate the fit. Reports `p99/median`,
  `max/median` and the share of the regression loss carried by the
  heaviest 1% of cells, then re-runs with unit-normalised raw vectors to
  see whether the tail was masking a real difference.

Everything is scored against the **analytic ground-truth velocity** of
`velot.datasets.synthetic_bifurcation`, not against CBDir/ICCoh, so the
comparison does not inherit the metric-objective coupling the reviewers
object to. Pseudotime is set to the true pseudotime, which removes DPT
quality as a confound and isolates the estimator.

Note: the truth is defined in feature space as `(1, slope, 0, ...)` and
pushed through `adata.varm['PCs']`. Comparing against the first two PCs
directly is wrong whenever `extra_dimensions > 0`, because PCA rotates the
axes — that version returns cos ~ 0 (pure noise).

---

## `metric_dimension_sensitivity.py`

### What "nuisance fraction" means

A velocity vector can be split into two parts: the component lying in the
subspace where the biology actually varies (**signal**), and the component
pointing into directions that carry no dynamics (**nuisance**) — the tail
PCs, or the thousands of low-variance gene directions in an HVG space.

    nuisance fraction = ||v_nuisance|| / ||v_signal||

A value of 0 means the method only ever points along real dynamics. A value
of 1 means it puts as much magnitude into no-signal directions as into real
ones. Every real method has some; none report it.

### Why it interacts with the evaluation dimension

ICCoh is a cosine between neighbouring cells' velocities. Two neighbours'
**signal** components agree, because both follow the same local flow. Their
**nuisance** components are independent draws, so they contribute ~0 to the
dot product but still contribute to both norms. The cosine is therefore
diluted by roughly `1 / (1 + nuisance_fraction^2)`.

The more ambient dimensions the metric is computed in, the more of that
independent noise it includes. Measure in the signal subspace only and you
never see it. This matters because the submitted benchmark measured each
method in a different space: VelOT in a 10-30 D PCA, scVelo in a 50 D PCA,
DeepVelo in the 2,000 D HVG gene space, FluxMatching in the 2 D UMAP.

### What the script does

Builds a synthetic bifurcation whose true dynamics live in a 2-D plane,
embeds it in 2,000 ambient dimensions, and constructs velocity fields with
**identical angular accuracy** but different nuisance fractions. It then
computes ICCoh and CBDir in the first `d` dimensions for
`d = 2, 5, 10, ..., 2000` and reports how far the score moves.

Findings:

* **CBDir is essentially dimension-invariant** (1.000 at 2 D -> 0.998 at
  2,000 D). The submitted CBDir comparison is not compromised.
* **ICCoh is conditionally affected**, entirely through the nuisance
  fraction:

  | nuisance fraction | ICCoh @2 D | @2000 D | shift |
  |---|---|---|---|
  | 0.10 | 0.916 | 0.908 | 0.009 |
  | 0.25 | 0.916 | 0.863 | 0.054 |
  | 0.50 | 0.916 | 0.733 | 0.183 |
  | 1.00 | 0.916 | 0.458 | 0.458 (ranking flips) |

* **The null scale always differs.** An ICCoh of 0.5 sits 0.7 sd from a
  random field at 2 D and 22 sd from random at 2,000 D, so the same number
  means different things in the two spaces.

Practical consequence: running each method in its own recommended
representation is correct and should not change. What should change is that
ICCoh is reported **both** in the native and in one shared space, with each
method's nuisance fraction stated, so a reader can see whether the
comparison is affected. That number is measurable from the shipped
`benchmark_results/real/data/*.h5ad` files.


---

## `toy_window_transport.py`

A 2-gene toy (default 10 + 10 cells). Cluster B is cluster A moved by a known
`shift`: either the same cells (`target="twin"`, exact translation) or a
fresh sample from the same distribution (`target="resample"`, the realistic
case, since the next window holds different cells). Windows come from
`velot.tl.build_windows` with every cell in one spatial cluster and
`window_size = n_per_cluster`, which gives exactly two windows (A, B) and
one pair.

Every raw estimator is written as `v_i = sum_j w_ij (x_j - x_i)` and every
`w_ij` is exposed:

* `ot_decomposition(adata, reg=...)`: exactly `tl._ot_velocity_pair`, and
  every setting is checked against the package. The same four options are
  arguments of `tl._ot_velocity_pair`, `tl.compute_ot_velocity` and
  `tl.velocity` (there as `use_graph`, `cost_scale`, `ot_solver`,
  `ot_assignment`), all defaulting to the original estimator:
  * `use_graph=False` removes both uses of the kNN graph: the `lambda_knn`
    penalty, and the rule that sets v = 0 for a cell with no graph
    neighbour in the target window. Setting `lambda_knn=0` removes only the
    penalty. Penalties are always multiplied by max(d2) / scale, so
    `lambda = 1` means "as costly as the farthest pair" under any scale.
  * `cost_scale="max" | "nn" | "none" | float`: what d² is divided by.
    `"max"` is the package's choice. `"nn"` divides by the median
    squared nearest-neighbour distance inside the source window.
  * `solver="emd"`: exact, unregularised OT.
  * `assignment="argmax"`: use only the target with the most mass.
* `window_decomposition(adata)`: `gradient_velocity(mode="window")`.
* `knn_decomposition(adata, k=...)`: `gradient_velocity(mode="knn")`.

```python
from toy_window_transport import *
ad  = make_toy_adata(shift=(3, 1), spread=0.6, target="twin")
res = compare(ad, cells=(0, 3), ot_variants=[
        dict(reg=0.1),                                   # package
        dict(reg=0.1, use_graph=False),
        dict(reg=0.1, use_graph=False, cost_scale="nn"),
        dict(solver="emd", use_graph=False)])
error_table(ad, {"package": dict(reg=0.1), "EMD": dict(solver="emd", use_graph=False)})
```

`T_eff = eps * scale / delta2` is the entropic temperature in units of the
within-window spacing, and it appears in every plot title. The plan is
invariant to adding any constant per row or per column of the cost, and
|x_i - (x_j + s)|^2 differs from |x_i - x_j|^2 only by such terms, so the
unscaled plan does not depend on the shift at all. Dividing by
`max(d2)` makes the effective temperature grow roughly with |shift|^2,
which blurs the plan toward the window gradient.

Mean |v - shift| (running the script reproduces these):

| variant | overlap, twin | overlap, resample | big shift, twin | big shift, resample |
|---|---|---|---|---|
| package, eps 0.1 | 0.373 | 0.547 | 2.601 (7/10 zeroed) | 3.005 (9/10 zeroed) |
| no graph, eps 0.1 | 0.296 | 0.415 | 0.478 | 0.579 |
| no graph, cost/nn, eps 0.1 | 0.003 | 0.294 | 0.004 | 0.353 |
| no graph, argmax, eps 0.1 | 0.251 | 0.311 | 0.282 | 0.399 |
| exact EMD, no graph | 0.000 | 0.296 | 0.000 | 0.355 |

---

## Comparing OT settings on real or synthetic data

```python
settings = {
    "original":        dict(),
    "no graph":        dict(use_graph=False),
    "no graph, nn":    dict(use_graph=False, cost_scale="nn"),
    "no graph, argmax":dict(use_graph=False, ot_assignment="argmax"),
    "exact OT":        dict(use_graph=False, ot_solver="emd"),
}
for name, kw in settings.items():
    a = adata.copy()
    velot.tl.velocity(a, reg=0.1, **kw)          # everything else unchanged
    print(name, a.uns["velot_raw_velocity_params"]["mean_T_eff"],
          a.uns["velot_raw_velocity_params"]["mean_n_eff"])
```

`uns["velot_raw_velocity_params"]` records the settings plus two
sharpness diagnostics averaged over window pairs: `mean_T_eff`
(temperature in units of the within-window spacing; NaN for EMD) and
`mean_n_eff` (effective number of targets per cell; 1 = one target).
With `cost_scale="nn"`, `reg` equals T_eff; values around 0.1-1 are the
range explored in the toy. EMD cost grows as n^3, fine for windows of a
few hundred cells.

#!/bin/bash
set -u
cd "$(dirname "$0")/../../benchmark"

PYTHON=${PYTHON:-/home/user/miniforge3/envs/velot_test/bin/python}
DATASETS=${DATASETS:-"gen_bifurcation gen_tree"}
METHODS=${METHODS:-"velot_gradient velot_unbalanced"}
SEEDS=${SEEDS:-"0"}
OUT=${OUT:-"article_figures/figure3/results"}
FIGURES=${FIGURES:-"0"}
FIGURES_OUT=${FIGURES_OUT:-"article_figures/figure3/figures"}
SAVE_METHODS=${SAVE_METHODS:-"$METHODS"}
# Windows built inside each cell type instead of one window over all
# cells - the supervised arm:
#   SPATIAL_KEY=clusters_id TAG=labels ./run_all.sh
# Leave both unset for the unsupervised default (n_clusters=1).
SPATIAL_KEY=${SPATIAL_KEY:-}
TAG=${TAG:-}
opts=""
[ -n "$SPATIAL_KEY" ] && opts="$opts --spatial-key $SPATIAL_KEY"
[ -n "$TAG" ] && opts="$opts --tag $TAG"

mkdir -p "$OUT"
LOG="$OUT/run_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "log: $LOG"
echo "datasets: $DATASETS"
echo "methods:  $METHODS"
echo "seeds:    $SEEDS"
echo

failed=()
for dataset in $DATASETS; do
  for method in $METHODS; do
    for seed in $SEEDS; do
      echo "=== $dataset / $method / seed $seed"
      extra=""
      if [ "$seed" = "0" ] && [[ " $SAVE_METHODS " == *" $method "* ]]; then
        extra="--save-adata"
      fi
      if ! $PYTHON run_velot.py -d "$dataset" -m "$method" -s "$seed" --output-dir "$OUT" --cache-dir "$OUT/cache" $opts $extra; then
        echo "FAILED: $dataset $method seed $seed"
        failed+=("$dataset/$method/seed$seed")
      fi
    done
  done
done

echo
if [ ${#failed[@]} -gt 0 ]; then
  echo "### ${#failed[@]} run(s) FAILED:"
  printf '  %s\n' "${failed[@]}"
else
  echo "### all runs completed"
fi

# Completeness check: every (dataset, method, seed) must have a readable
# raw and smooth JSON. This is what catches a run that died between the
# two saves, or a file that was written but cannot be parsed.
echo
DATASETS="$DATASETS" METHODS="$METHODS" SEEDS="$SEEDS" OUT="$OUT" TAG="$TAG" $PYTHON - <<'PY'
import json, os
out = os.environ["OUT"]
missing, corrupt = [], []
for d in os.environ["DATASETS"].split():
    for m in os.environ["METHODS"].split():
        for s in os.environ["SEEDS"].split():
            for f in ("raw", "smooth"):
                tag = os.environ.get("TAG", "")
                nm = m + (f"_{tag}" if tag else "")
                p = os.path.join(out, f"{nm}_seed{s}_{f}_{d}.json")
                if not os.path.exists(p):
                    missing.append(os.path.basename(p)); continue
                try:
                    json.load(open(p))
                except Exception:
                    corrupt.append(os.path.basename(p))
if missing or corrupt:
    print(f"### INCOMPLETE: {len(missing)} missing, {len(corrupt)} unreadable")
    for name in (missing + corrupt)[:20]:
        print("  ", name)
    if len(missing) + len(corrupt) > 20:
        print(f"   ... and {len(missing) + len(corrupt) - 20} more")
else:
    print("### all expected result files present and readable")
PY

echo
$PYTHON aggregate.py --dir "$OUT"

# Figures for seed 0, rebuilt from the saved AnnData. Separate script, so
# changing a plot never means re-running an experiment: just call
# run_plots.py again. Set FIGURES=0 to skip.
if [ "${FIGURES:-1}" != "0" ]; then
  echo
  echo "### figures"
  plotopts=""
  [ -n "$TAG" ] && plotopts="--tag $TAG"
  $PYTHON run_plots.py -d $DATASETS -m $METHODS -s 0 $plotopts \
    --results-dir $OUT --out-dir $FIGURES_OUT || echo "FIGURES FAILED"
fi

echo
echo "log: $LOG"
[ ${#failed[@]} -eq 0 ]

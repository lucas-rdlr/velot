#!/bin/bash
# Root sweep and supervision comparison (R1.4, R2.3, R3.2). Run after
# run_all.sh: the intended-root runs are already there, these add the
# alternatives. Single seed by design - these vary the prior, not the
# initialisation.
#
#   ./run_sweeps.sh
#   SWEEP_DATASETS="pancreas" ./run_sweeps.sh
#
# The root arms run with --no-cache on purpose: every root needs its own
# preprocessed object, each is used exactly once, and caching ~50 of them
# would write tens of GB. The supervision arm reuses the intended root,
# so it does use the cache.
set -u
cd "$(dirname "$0")"

PYTHON=${PYTHON:-/home/user/miniforge3/envs/velot_test/bin/python}
METHOD=${METHOD:-velot}
# R1.4 asks for the root-perturbation figure on pancreas AND
# oligodendroglioma; erythroid is the third panel for the supplement.
SWEEP_DATASETS=${SWEEP_DATASETS:-"pancreas oligodendroglioma erythroid"}
N_PERTURB=${N_PERTURB:-20}
SUP_DATASET=${SUP_DATASET:-pancreas}
OUT=${OUT:-"benchmark_results/revision"}

mkdir -p "$OUT"
LOG="$OUT/run_sweeps_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
FAILLOG=$(mktemp)                     # a pipe runs in a subshell: use a file
trap 'rm -f "$FAILLOG"' EXIT
echo "log: $LOG"

run() {                               # run(label, args...)
  local label=$1; shift
  echo "=== $label"
  $PYTHON run_velot.py "$@" || echo "$label" >> "$FAILLOG"
}

for dataset in $SWEEP_DATASETS; do
  echo "### $dataset: root rules"
  for rule in distal medoid first; do
    run "$dataset rule-$rule" -d "$dataset" -m "$METHOD" -s 0 --no-cache \
      --root-selection "$rule" --tag "rule-$rule"
  done

  echo "### $dataset: every cluster as root"
  # Go through ds.load, not sc.read_h5ad: some datasets derive their
  # labels during preprocessing, so the raw file has no cluster column.
  # Write to a file first - a pipeline would swallow a failure here and
  # the sweep would silently run no roots at all.
  LIST=$(mktemp)
  if ! $PYTHON -c "
import datasets as ds
adata, cfg = ds.load('$dataset')
for c in sorted(set(adata.obs[cfg['clusters_key']].astype(str))):
    print(c)" > "$LIST"; then
    echo "$dataset: could not list clusters" >> "$FAILLOG"
  fi
  # cluster names contain spaces, so read them line by line
  while IFS= read -r cluster; do
    [ -z "$cluster" ] && continue
    tag="rootcluster-$(echo "$cluster" | tr ' /' '__')"
    run "$dataset $tag" -d "$dataset" -m "$METHOD" -s 0 --no-cache \
      --root-cluster "$cluster" --tag "$tag"
  done < "$LIST"
  rm -f "$LIST"

  echo "### $dataset: perturbed root cells inside the intended cluster"
  LIST=$(mktemp)
  if ! $PYTHON -c "
import datasets as ds, numpy as np
adata, cfg = ds.load('$dataset')
idx = np.where(adata.obs[cfg['clusters_key']].astype(str) == cfg['root_cluster'])[0]
rng = np.random.default_rng(0)
for c in rng.choice(idx, min($N_PERTURB, len(idx)), replace=False):
    print(int(c))" > "$LIST"; then
    echo "$dataset: could not list root cells" >> "$FAILLOG"
  fi
  while IFS= read -r cell; do
    [ -z "$cell" ] && continue
    run "$dataset rootcell-$cell" -d "$dataset" -m "$METHOD" -s 0 --no-cache \
      --root-cell "$cell" --tag "rootcell-$cell"
  done < "$LIST"
  rm -f "$LIST"
done

echo "### supervision comparison ($SUP_DATASET)"
for seed in 0 1 2 3 4; do
  run "$SUP_DATASET sup-labels seed$seed" -d "$SUP_DATASET" -m "$METHOD" \
    -s "$seed" --spatial-key clusters_id --tag "sup-labels"
  run "$SUP_DATASET sup-kmeans seed$seed" -d "$SUP_DATASET" -m "$METHOD" \
    -s "$seed" --n-clusters 10 --tag "sup-kmeans"
done
# the "single cluster" arm of that comparison is the benchmark run itself

echo
if [ -s "$FAILLOG" ]; then
  echo "### $(wc -l < "$FAILLOG") run(s) FAILED:"
  sed 's/^/  /' "$FAILLOG"
else
  echo "### all sweep runs completed"
fi

echo
$PYTHON aggregate.py --dir "$OUT"
echo
echo "log: $LOG"

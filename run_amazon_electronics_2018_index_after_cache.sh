#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/users/atavory/scratch/wsdm_experiments/rqd}
PY=${PY:-/data/users/atavory/scratch/wsdm_experiments/venv/bin/python}
CACHE=${CACHE:-/data/users/atavory/scratch/wsdm_experiments/cache/amazon2018_electronics_full5core64.npz}
OUT=${OUT:-/data/users/atavory/scratch/wsdm_experiments/results/amazon_tierb_20260812}
PREP_PID_FILE=${PREP_PID_FILE:-$OUT/logs/electronics2018_cache.pid}

cd "$ROOT"

while [[ ! -s "$CACHE" ]]; do
  if [[ -s "$PREP_PID_FILE" ]] && ! ps -p "$(cat "$PREP_PID_FILE")" >/dev/null 2>&1; then
    echo "cache prep exited before writing $CACHE" >&2
    exit 1
  fi
  sleep 30
done

for arch in uniform16 funnel16 uniform20 funnel20 balanced24 funnel24; do
  for seed in 0 1 2; do
    "$PY" run_wsdm_index_sweep.py \
      --cache "$CACHE" \
      --arch "$arch" \
      --seed "$seed" \
      --freeze-depths 1,2,3 \
      --kmeans-iterations 20 \
      --output "$OUT/index_amazon2018_electronics_${arch}_seed${seed}.json" \
      --overwrite
  done
done

"$PY" summarize_wsdm_results.py --results-dir "$OUT"

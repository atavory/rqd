#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/users/atavory/scratch/wsdm_experiments/rqd}
PY=${PY:-/data/users/atavory/scratch/wsdm_experiments/venv/bin/python}
DATA=${DATA:-/data/users/atavory/scratch/wsdm_experiments/data/amazon/Electronics_5.json.gz}
CACHE=${CACHE:-/data/users/atavory/scratch/wsdm_experiments/cache/amazon2018_electronics_full5core64.npz}
OUT=${OUT:-/data/users/atavory/scratch/wsdm_experiments/results/amazon_tierb_20260812}
LOG="$OUT/logs"

mkdir -p "$(dirname "$DATA")" "$(dirname "$CACHE")" "$LOG"

if [[ ! -s "$DATA" ]]; then
  curl -fL \
    https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Electronics_5.json.gz \
    -o "$DATA"
fi

cd "$ROOT"
"$PY" run_wsdm_web_recsys.py \
  --dataset amazon \
  --amazon-data "$DATA" \
  --amazon-core-passes 0 \
  --max-train-sequences 50000 \
  --max-eval-sequences 10000 \
  --cache "$CACHE" \
  --prepare-only \
  --embedding-dim 64 \
  --overwrite

#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/users/atavory/scratch/wsdm_experiments/rqd}
PY=${PY:-/data/users/atavory/scratch/wsdm_experiments/venv/bin/python}
BASE_URL=${BASE_URL:-https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main}

DATA_ROOT=${DATA_ROOT:-/data/users/atavory/scratch/wsdm_experiments/data/amazon2023}
CACHE_ROOT=${CACHE_ROOT:-/data/users/atavory/scratch/wsdm_experiments/cache}
OUT_ROOT=${OUT_ROOT:-/data/users/atavory/scratch/wsdm_experiments/results/amazon2023_5core_index_20260812}
QUEUE_ROOT=${QUEUE_ROOT:-/data/users/atavory/scratch/wsdm_experiments/results/amazon2023_acquire_prepare_index_20260812}
LOG="$QUEUE_ROOT/logs"

ARCHES=${ARCHES:-funnel24}
SEEDS=${SEEDS:-0 1 2 3 4}
FREEZE_DEPTHS=${FREEZE_DEPTHS:-1,2,3}
KMEANS_ITERATIONS=${KMEANS_ITERATIONS:-20}
EMBEDDING_DIM=${EMBEDDING_DIM:-64}
MAX_TRAIN_SEQUENCES=${MAX_TRAIN_SEQUENCES:-50000}
MAX_EVAL_SEQUENCES=${MAX_EVAL_SEQUENCES:-10000}
WAIT_FOR_CURRENT_QUEUE_BEFORE_PREP=${WAIT_FOR_CURRENT_QUEUE_BEFORE_PREP:-1}
CURRENT_QUEUE_PID=${CURRENT_QUEUE_PID:-/data/users/atavory/scratch/wsdm_experiments/results/wsdm_available_full_queue_20260812/logs/master.pid}
INCLUDE_ZERO_CORE_BOOKS=${INCLUDE_ZERO_CORE_BOOKS:-1}

mkdir -p "$DATA_ROOT/5core" "$DATA_ROOT/0core" "$CACHE_ROOT" "$OUT_ROOT" "$LOG"

exec 9>"$QUEUE_ROOT/queue.lock"
if ! flock -n 9; then
  echo "amazon2023 acquire/prepare/index queue already running"
  exit 0
fi

ts() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(ts)] $*"
}

sha_ok() {
  local path=$1
  local expected=$2
  [[ -s "$path" ]] && [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]]
}

download_one() {
  local rel=$1
  local dest=$2
  local sha=$3
  local url="$BASE_URL/$rel?download=true"

  mkdir -p "$(dirname "$dest")"
  if sha_ok "$dest" "$sha"; then
    log "skip verified download $dest"
    return
  fi

  log "download $rel -> $dest"
  curl -fL --retry 12 --retry-delay 20 --continue-at - "$url" -o "$dest"
  if ! sha_ok "$dest" "$sha"; then
    echo "sha256 mismatch for $dest" >&2
    echo "expected $sha" >&2
    sha256sum "$dest" >&2 || true
    exit 1
  fi
}

wait_for_current_queue() {
  if [[ "$WAIT_FOR_CURRENT_QUEUE_BEFORE_PREP" != "1" ]]; then
    return
  fi
  while [[ -s "$CURRENT_QUEUE_PID" ]] && ps -p "$(cat "$CURRENT_QUEUE_PID")" >/dev/null 2>&1; do
    log "waiting for current GPU queue before cache prep: $(cat "$CURRENT_QUEUE_PID")"
    sleep 300
  done
}

prepare_cache() {
  local source=$1
  local cache=$2
  local variant=$3

  if [[ -s "$cache" && -s "${cache%.npz}.metadata.json" ]]; then
    log "skip prepared cache $cache"
    return
  fi

  log "prepare cache variant=$variant source=$source"
  (
    cd "$ROOT"
    "$PY" run_wsdm_web_recsys.py \
      --dataset amazon \
      --amazon-data "$source" \
      --amazon-core-passes 0 \
      --max-train-sequences "$MAX_TRAIN_SEQUENCES" \
      --max-eval-sequences "$MAX_EVAL_SEQUENCES" \
      --cache "$cache" \
      --prepare-only \
      --embedding-dim "$EMBEDDING_DIM" \
      --overwrite
  )
}

json_ok() {
  [[ -s "$1" ]] && "$PY" - "$1" >/dev/null 2>&1 <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    json.load(handle)
PY
}

run_index_matrix() {
  local cache=$1
  local label=$2

  for arch in $ARCHES; do
    for seed in $SEEDS; do
      local output="$OUT_ROOT/index_${label}_${arch}_seed${seed}.json"
      if json_ok "$output"; then
        log "skip complete index $output"
        continue
      fi
      log "index label=$label arch=$arch seed=$seed"
      (
        cd "$ROOT"
        "$PY" run_wsdm_index_sweep.py \
          --cache "$cache" \
          --arch "$arch" \
          --seed "$seed" \
          --freeze-depths "$FREEZE_DEPTHS" \
          --kmeans-iterations "$KMEANS_ITERATIONS" \
          --output "$output" \
          --overwrite
      )
    done
  done
}

summarize() {
  log "summarize $OUT_ROOT"
  (
    cd "$ROOT"
    "$PY" summarize_wsdm_results.py --results-dir "$OUT_ROOT"
  )
}

download_all() {
  download_one benchmark/5core/rating_only/Books.csv \
    "$DATA_ROOT/5core/Books.csv" \
    6e65a282b42894099a105f3a30dae09354e6cf423b8b81f0014b21a1fd8d3add
  download_one benchmark/5core/rating_only/Electronics.csv \
    "$DATA_ROOT/5core/Electronics.csv" \
    70bc096a7c908c90b1da5b8ab9d880f0ed9590fb1b523d10374b977cd1996257
  download_one benchmark/5core/rating_only/Clothing_Shoes_and_Jewelry.csv \
    "$DATA_ROOT/5core/Clothing_Shoes_and_Jewelry.csv" \
    dd1ea1d6cbdf4c061067648fb978fd4412eded0d59b48232e5db7b1057da7a12
  download_one benchmark/5core/rating_only/Home_and_Kitchen.csv \
    "$DATA_ROOT/5core/Home_and_Kitchen.csv" \
    455f85aac9d8e3eb2892fca2fe370696b89ffba5f118e4d83bb21a3e0e0f448d

  download_one benchmark/5core/rating_only/Beauty_and_Personal_Care.csv \
    "$DATA_ROOT/5core/Beauty_and_Personal_Care.csv" \
    14a2bacd6b4e18521ba45705d0a0e672ea82d2878fff2c8897ec8056c24cd6c1
  download_one benchmark/5core/rating_only/Tools_and_Home_Improvement.csv \
    "$DATA_ROOT/5core/Tools_and_Home_Improvement.csv" \
    f49770e5b6781e88c49a49b925b5420551f7d87b703175856a5590639cebcc47
  download_one benchmark/5core/rating_only/Toys_and_Games.csv \
    "$DATA_ROOT/5core/Toys_and_Games.csv" \
    64c2e01a0402010477fb5fc35b43508a8df9e2bb208e0e6f763d93477bfb7a07

  if [[ "$INCLUDE_ZERO_CORE_BOOKS" == "1" ]]; then
    download_one benchmark/0core/rating_only/Books.csv \
      "$DATA_ROOT/0core/Books.csv" \
      068388cec9a5ee1b56e6858dad021d1e2154dedbbda65b5db30c634eccc874ba
  fi
}

prepare_and_index_all() {
  prepare_cache "$DATA_ROOT/5core/Books.csv" \
    "$CACHE_ROOT/amazon2023_5core_books64.npz" amazon2023_5core_books
  run_index_matrix "$CACHE_ROOT/amazon2023_5core_books64.npz" amazon2023_5core_books
  summarize

  prepare_cache "$DATA_ROOT/5core/Electronics.csv" \
    "$CACHE_ROOT/amazon2023_5core_electronics64.npz" amazon2023_5core_electronics
  run_index_matrix "$CACHE_ROOT/amazon2023_5core_electronics64.npz" amazon2023_5core_electronics
  summarize

  prepare_cache "$DATA_ROOT/5core/Clothing_Shoes_and_Jewelry.csv" \
    "$CACHE_ROOT/amazon2023_5core_clothing_shoes_and_jewelry64.npz" amazon2023_5core_clothing
  run_index_matrix "$CACHE_ROOT/amazon2023_5core_clothing_shoes_and_jewelry64.npz" amazon2023_5core_clothing
  summarize

  prepare_cache "$DATA_ROOT/5core/Home_and_Kitchen.csv" \
    "$CACHE_ROOT/amazon2023_5core_home_and_kitchen64.npz" amazon2023_5core_home
  run_index_matrix "$CACHE_ROOT/amazon2023_5core_home_and_kitchen64.npz" amazon2023_5core_home
  summarize

  prepare_cache "$DATA_ROOT/5core/Beauty_and_Personal_Care.csv" \
    "$CACHE_ROOT/amazon2023_5core_beauty_and_personal_care64.npz" amazon2023_5core_beauty
  prepare_cache "$DATA_ROOT/5core/Tools_and_Home_Improvement.csv" \
    "$CACHE_ROOT/amazon2023_5core_tools_and_home_improvement64.npz" amazon2023_5core_tools
  prepare_cache "$DATA_ROOT/5core/Toys_and_Games.csv" \
    "$CACHE_ROOT/amazon2023_5core_toys_and_games64.npz" amazon2023_5core_toys

  if [[ "$INCLUDE_ZERO_CORE_BOOKS" == "1" ]]; then
    prepare_cache "$DATA_ROOT/0core/Books.csv" \
      "$CACHE_ROOT/amazon2023_0core_books64.npz" amazon2023_0core_books
    run_index_matrix "$CACHE_ROOT/amazon2023_0core_books64.npz" amazon2023_0core_books
    summarize
  fi
}

main() {
  log "amazon2023 acquire/prepare/index queue start"
  log "data_root=$DATA_ROOT cache_root=$CACHE_ROOT out_root=$OUT_ROOT"
  download_all
  wait_for_current_queue
  prepare_and_index_all
  log "amazon2023 acquire/prepare/index queue done"
}

main "$@"

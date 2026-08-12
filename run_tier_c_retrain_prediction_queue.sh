#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/users/atavory/scratch/wsdm_experiments/rqd}
PY=${PY:-/data/users/atavory/scratch/wsdm_experiments/venv/bin/python}
CACHE_ML1M=${CACHE_ML1M:-/data/users/atavory/scratch/wsdm_experiments/cache/movielens64.npz}
CACHE_AMZ18=${CACHE_AMZ18:-/data/users/atavory/scratch/wsdm_experiments/cache/amazon2018_electronics_full5core64.npz}

OUT_ROOT=${OUT_ROOT:-/data/users/atavory/scratch/wsdm_experiments/results/tier_c_retrain_prediction_20260812}
QUEUE_ROOT=${QUEUE_ROOT:-$OUT_ROOT/queue}
LOG="$QUEUE_ROOT/logs"

CURRENT_QUEUE_PID=${CURRENT_QUEUE_PID:-/data/users/atavory/scratch/wsdm_experiments/results/wsdm_available_full_queue_20260812/logs/master.pid}
AMAZON2023_QUEUE_PID=${AMAZON2023_QUEUE_PID:-/data/users/atavory/scratch/wsdm_experiments/results/amazon2023_acquire_prepare_index_20260812/logs/master.pid}

SYNTHETIC_OUT=${SYNTHETIC_OUT:-$OUT_ROOT/synthetic}
REAL_INDEX_OUT=${REAL_INDEX_OUT:-$OUT_ROOT/real_index}
PREDICTION_OUT=${PREDICTION_OUT:-$OUT_ROOT/predictions}

ML1M_ARCHES=${ML1M_ARCHES:-funnel24 balanced24 uniform24}
AMZ18_ARCHES=${AMZ18_ARCHES:-funnel24 balanced24 uniform24}
SEEDS=${SEEDS:-0 1 2 3 4}
FREEZE_DEPTHS=${FREEZE_DEPTHS:-1,2,3}
KMEANS_ITERATIONS=${KMEANS_ITERATIONS:-20}

AMZ18_DOWN_OUT=${AMZ18_DOWN_OUT:-/data/users/atavory/scratch/wsdm_experiments/results/amazon2018_electronics_downstream_funnel24_fd23_seeds012_20260812}
AMZ18_INDEX_OUT=${AMZ18_INDEX_OUT:-/data/users/atavory/scratch/wsdm_experiments/results/amazon_tierb_20260812}
ML1M_SEED0_OUT=${ML1M_SEED0_OUT:-/data/users/atavory/scratch/wsdm_experiments/results/phase0_ml1m_primary24_20260812}
ML1M_SEEDS12_OUT=${ML1M_SEEDS12_OUT:-/data/users/atavory/scratch/wsdm_experiments/results/phase0_ml1m_primary24_seeds12_20260812}
ML1M_SEEDS34_OUT=${ML1M_SEEDS34_OUT:-/data/users/atavory/scratch/wsdm_experiments/results/phase0_ml1m_primary24_seeds34_20260812}
AMAZON2023_INDEX_OUT=${AMAZON2023_INDEX_OUT:-/data/users/atavory/scratch/wsdm_experiments/results/amazon2023_5core_index_20260812}

mkdir -p "$LOG" "$SYNTHETIC_OUT" "$REAL_INDEX_OUT" "$PREDICTION_OUT"

exec 9>"$QUEUE_ROOT/queue.lock"
if ! flock -n 9; then
  echo "Tier-C retrain-prediction queue already running"
  exit 0
fi

ts() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(ts)] $*"
}

json_ok() {
  [[ -s "$1" ]] && "$PY" - "$1" >/dev/null 2>&1 <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    json.load(handle)
PY
}

wait_for_pid_file() {
  local pid_file=$1
  local label=$2
  while [[ -s "$pid_file" ]] && ps -p "$(cat "$pid_file")" >/dev/null 2>&1; do
    log "waiting for $label pid $(cat "$pid_file")"
    sleep 300
  done
}

run_synthetic() {
  local output="$SYNTHETIC_OUT/tier_c_synthetic_retrain_predictions.json"
  local csv="$SYNTHETIC_OUT/tier_c_synthetic_retrain_predictions.csv"
  if json_ok "$output" && [[ -s "$csv" ]]; then
    log "skip complete synthetic Tier-C predictions"
    return
  fi
  log "run synthetic Tier-C retrain-prediction matrix"
  (
    cd "$ROOT"
    "$PY" run_tier_c_retrain_prediction.py \
      --output "$output" \
      --csv-output "$csv" \
      --arches funnel24 \
      --seeds 0,1,2,3,4 \
      --freeze-depths 1,2,3 \
      --magnitudes 0.0,0.05,0.15,0.35,0.7
  )
}

run_index() {
  local label=$1
  local cache=$2
  local arch=$3
  local seed=$4
  local output="$REAL_INDEX_OUT/index_${label}_${arch}_seed${seed}.json"

  if json_ok "$output"; then
    log "skip complete diagnostic index $output"
    return
  fi
  log "run diagnostic index label=$label arch=$arch seed=$seed"
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
}

run_available_real_index() {
  if [[ -s "$CACHE_ML1M" ]]; then
    for arch in $ML1M_ARCHES; do
      for seed in $SEEDS; do
        run_index movielens "$CACHE_ML1M" "$arch" "$seed"
      done
    done
  else
    log "missing ML-1M cache: $CACHE_ML1M"
  fi

  if [[ -s "$CACHE_AMZ18" ]]; then
    for arch in $AMZ18_ARCHES; do
      for seed in $SEEDS; do
        run_index amazon2018_electronics "$CACHE_AMZ18" "$arch" "$seed"
      done
    done
  else
    log "missing Amazon2018 cache: $CACHE_AMZ18"
  fi
}

summarize_dir() {
  local dir=$1
  if [[ -d "$dir" ]]; then
    log "summarize WSDM result dir $dir"
    (
      cd "$ROOT"
      "$PY" summarize_wsdm_results.py --results-dir "$dir"
    )
  fi
}

prediction_args_for_existing_dirs() {
  local args=()
  for dir in "$REAL_INDEX_OUT" "$AMZ18_INDEX_OUT" "$AMZ18_DOWN_OUT" \
    "$ML1M_SEED0_OUT" "$ML1M_SEEDS12_OUT" "$ML1M_SEEDS34_OUT" \
    "$AMAZON2023_INDEX_OUT"; do
    if [[ -d "$dir" ]]; then
      args+=(--results-dir "$dir")
    fi
  done
  printf '%q ' "${args[@]}"
}

summarize_predictions() {
  summarize_dir "$REAL_INDEX_OUT"
  summarize_dir "$AMZ18_INDEX_OUT"
  summarize_dir "$AMZ18_DOWN_OUT"
  summarize_dir "$ML1M_SEED0_OUT"
  summarize_dir "$ML1M_SEEDS12_OUT"
  summarize_dir "$ML1M_SEEDS34_OUT"
  summarize_dir "$AMAZON2023_INDEX_OUT"

  local output="$PREDICTION_OUT/tier_c_real_retrain_predictions.csv"
  local args
  args=$(prediction_args_for_existing_dirs)
  if [[ -z "$args" ]]; then
    log "no result dirs available for Tier-C prediction summary"
    return
  fi
  log "write Tier-C real prediction summary $output"
  (
    cd "$ROOT"
    # shellcheck disable=SC2086
    "$PY" summarize_tier_c_retrain_predictions.py $args --output "$output"
  )
}

main() {
  log "Tier-C retrain-prediction queue start"
  log "out_root=$OUT_ROOT"

  run_synthetic >"$LOG/synthetic.log" 2>&1 &
  local synthetic_pid=$!
  run_available_real_index >"$LOG/real_index.log" 2>&1 &
  local real_index_pid=$!

  log "workers launched synthetic=$synthetic_pid real_index=$real_index_pid"
  local synthetic_status=0
  local real_index_status=0
  wait "$synthetic_pid" || synthetic_status=$?
  wait "$real_index_pid" || real_index_status=$?
  log "workers finished synthetic_status=$synthetic_status real_index_status=$real_index_status"

  wait_for_pid_file "$CURRENT_QUEUE_PID" "available-data GPU queue"
  wait_for_pid_file "$AMAZON2023_QUEUE_PID" "Amazon2023 queue"
  summarize_predictions >"$LOG/summarize_predictions.log" 2>&1

  log "Tier-C retrain-prediction queue done"
}

main "$@"

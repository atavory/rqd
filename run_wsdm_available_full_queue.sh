#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/users/atavory/scratch/wsdm_experiments/rqd}
PY=${PY:-/data/users/atavory/scratch/wsdm_experiments/venv/bin/python}
CACHE_ML1M=${CACHE_ML1M:-/data/users/atavory/scratch/wsdm_experiments/cache/movielens64.npz}
CACHE_AMZ18=${CACHE_AMZ18:-/data/users/atavory/scratch/wsdm_experiments/cache/amazon2018_electronics_full5core64.npz}

OUT_ROOT=${OUT_ROOT:-/data/users/atavory/scratch/wsdm_experiments/results}
QUEUE_ROOT=${QUEUE_ROOT:-$OUT_ROOT/wsdm_available_full_queue_20260812}
LOG="$QUEUE_ROOT/logs"

AMZ18_INDEX_OUT=${AMZ18_INDEX_OUT:-$OUT_ROOT/amazon_tierb_20260812}
AMZ18_DOWN_OUT=${AMZ18_DOWN_OUT:-$OUT_ROOT/amazon2018_electronics_downstream_funnel24_fd23_seeds012_20260812}
ML1M_SEED0_OUT=${ML1M_SEED0_OUT:-$OUT_ROOT/phase0_ml1m_primary24_20260812}
ML1M_SEEDS12_OUT=${ML1M_SEEDS12_OUT:-$OUT_ROOT/phase0_ml1m_primary24_seeds12_20260812}
ML1M_SEEDS34_OUT=${ML1M_SEEDS34_OUT:-$OUT_ROOT/phase0_ml1m_primary24_seeds34_20260812}

mkdir -p "$LOG" "$AMZ18_INDEX_OUT" "$AMZ18_DOWN_OUT"

exec 9>"$QUEUE_ROOT/queue.lock"
if ! flock -n 9; then
  echo "queue already running; lock=$QUEUE_ROOT/queue.lock"
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

wait_for_output_process() {
  local output=$1
  local pattern=$2
  while pgrep -f -- "$pattern.*$output" >/dev/null 2>&1 || pgrep -f -- "$output.*$pattern" >/dev/null 2>&1; do
    log "waiting for existing process writing $output"
    sleep 60
  done
}

run_downstream() {
  local dataset=$1
  local cache=$2
  local arch=$3
  local freeze_depth=$4
  local seed=$5
  local gpu=$6
  local output=$7

  mkdir -p "$(dirname "$output")"
  wait_for_output_process "$output" "run_wsdm_web_recsys.py"
  if json_ok "$output"; then
    log "skip complete downstream $output"
    return
  fi

  log "run downstream dataset=$dataset arch=$arch fd=$freeze_depth seed=$seed gpu=$gpu"
  (
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" run_wsdm_web_recsys.py \
      --dataset "$dataset" \
      --cache "$cache" \
      --arch "$arch" \
      --seed "$seed" \
      --freeze-depth "$freeze_depth" \
      --epochs 50 \
      --n-beams 10 \
      --device cuda:0 \
      --output "$output" \
      --overwrite
  )
}

run_index() {
  local cache=$1
  local arch=$2
  local seed=$3
  local output=$4

  mkdir -p "$(dirname "$output")"
  wait_for_output_process "$output" "run_wsdm_index_sweep.py"
  if json_ok "$output"; then
    log "skip complete index $output"
    return
  fi

  log "run index arch=$arch seed=$seed"
  (
    cd "$ROOT"
    "$PY" run_wsdm_index_sweep.py \
      --cache "$cache" \
      --arch "$arch" \
      --seed "$seed" \
      --freeze-depths 1,2,3 \
      --kmeans-iterations 20 \
      --output "$output" \
      --overwrite
  )
}

summarize_dir() {
  local dir=$1
  if [[ -d "$dir" ]]; then
    log "summarize $dir"
    (
      cd "$ROOT"
      "$PY" summarize_wsdm_results.py --results-dir "$dir"
    )
  fi
}

ml1m_downstream_output() {
  local arch=$1
  local seed=$2

  if [[ "$seed" == "0" ]]; then
    echo "$ML1M_SEED0_OUT/downstream_movielens_${arch}_fd2_seed${seed}.json"
  elif [[ "$seed" == "1" || "$seed" == "2" ]]; then
    echo "$ML1M_SEEDS12_OUT/downstream_movielens_${arch}_fd2_seed${seed}.json"
  else
    echo "$ML1M_SEEDS34_OUT/downstream_movielens_${arch}_fd2_seed${seed}.json"
  fi
}

gpu0_worker() {
  log "gpu0 worker start"

  for seed in 0 1 2; do
    run_downstream amazon "$CACHE_AMZ18" funnel24 2 "$seed" 0 \
      "$AMZ18_DOWN_OUT/downstream_amazon2018_electronics_funnel24_fd2_seed${seed}.json"
  done

  for arch in balanced24 uniform24; do
    for seed in 0 1 2; do
      run_downstream amazon "$CACHE_AMZ18" "$arch" 2 "$seed" 0 \
        "$AMZ18_DOWN_OUT/downstream_amazon2018_electronics_${arch}_fd2_seed${seed}.json"
    done
  done

  for seed in 0 1 2 3 4; do
    run_downstream movielens "$CACHE_ML1M" funnel24 2 "$seed" 0 \
      "$(ml1m_downstream_output funnel24 "$seed")"
  done

  log "gpu0 worker done"
}

gpu1_worker() {
  log "gpu1 worker start"

  for seed in 0 1 2; do
    run_downstream amazon "$CACHE_AMZ18" funnel24 3 "$seed" 1 \
      "$AMZ18_DOWN_OUT/downstream_amazon2018_electronics_funnel24_fd3_seed${seed}.json"
  done

  for arch in balanced24 uniform24; do
    for freeze_depth in 3; do
      for seed in 0 1 2; do
        run_downstream amazon "$CACHE_AMZ18" "$arch" "$freeze_depth" "$seed" 1 \
          "$AMZ18_DOWN_OUT/downstream_amazon2018_electronics_${arch}_fd${freeze_depth}_seed${seed}.json"
      done
    done
  done

  for arch in balanced24 uniform24; do
    for seed in 0 1 2 3 4; do
      run_downstream movielens "$CACHE_ML1M" "$arch" 2 "$seed" 1 \
        "$(ml1m_downstream_output "$arch" "$seed")"
    done
  done

  log "gpu1 worker done"
}

cpu_index_worker() {
  log "cpu index worker start"

  for arch in funnel24 balanced24 uniform24; do
    for seed in 0 1 2 3 4; do
      run_index "$CACHE_AMZ18" "$arch" "$seed" \
        "$AMZ18_INDEX_OUT/index_amazon2018_electronics_${arch}_seed${seed}.json"
    done
  done

  log "cpu index worker done"
}

main() {
  log "queue start root=$QUEUE_ROOT"
  log "existing Amazon2018 downstream manual jobs will be waited on and skipped by output"

  gpu0_worker >"$LOG/gpu0_worker.log" 2>&1 &
  local gpu0_pid=$!
  gpu1_worker >"$LOG/gpu1_worker.log" 2>&1 &
  local gpu1_pid=$!
  cpu_index_worker >"$LOG/cpu_index_worker.log" 2>&1 &
  local cpu_pid=$!

  log "workers launched gpu0=$gpu0_pid gpu1=$gpu1_pid cpu=$cpu_pid"

  wait "$gpu0_pid"
  wait "$gpu1_pid"
  wait "$cpu_pid"

  summarize_dir "$AMZ18_INDEX_OUT"
  summarize_dir "$AMZ18_DOWN_OUT"
  summarize_dir "$ML1M_SEED0_OUT"
  summarize_dir "$ML1M_SEEDS12_OUT"
  summarize_dir "$ML1M_SEEDS34_OUT"

  log "queue done"
}

main "$@"

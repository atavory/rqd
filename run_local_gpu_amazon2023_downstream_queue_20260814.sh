#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/users/atavory/scratch/wsdm_experiments/rqd}
PY=${PY:-/data/users/atavory/scratch/wsdm_experiments/venv/bin/python}
CACHE_ROOT=${CACHE_ROOT:-/data/users/atavory/scratch/wsdm_experiments/cache}
OUT_ROOT=${OUT_ROOT:-/data/users/atavory/scratch/wsdm_experiments/results/amazon2023_downstream_rung_funnel24_20260814}
LOG="$OUT_ROOT/logs"

CATEGORIES=${CATEGORIES:-books electronics}
ARCHES=${ARCHES:-funnel24}
FREEZE_DEPTHS=${FREEZE_DEPTHS:-1 2 3}
SEEDS=${SEEDS:-0}
EPOCHS=${EPOCHS:-50}
N_BEAMS=${N_BEAMS:-10}

mkdir -p "$LOG" "$OUT_ROOT"

exec 9>"$OUT_ROOT/queue.lock"
if ! flock -n 9; then
  echo "local Amazon2023 downstream queue already running; lock=$OUT_ROOT/queue.lock"
  exit 0
fi

ts() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(ts)] $*"
}

cache_for_category() {
  case "$1" in
    books) echo "$CACHE_ROOT/amazon2023_5core_books64.npz" ;;
    electronics) echo "$CACHE_ROOT/amazon2023_5core_electronics64.npz" ;;
    clothing) echo "$CACHE_ROOT/amazon2023_5core_clothing_shoes_and_jewelry64.npz" ;;
    home) echo "$CACHE_ROOT/amazon2023_5core_home_and_kitchen64.npz" ;;
    beauty) echo "$CACHE_ROOT/amazon2023_5core_beauty_and_personal_care64.npz" ;;
    tools) echo "$CACHE_ROOT/amazon2023_5core_tools_and_home_improvement64.npz" ;;
    toys) echo "$CACHE_ROOT/amazon2023_5core_toys_and_games64.npz" ;;
    *) echo "unknown category: $1" >&2; exit 2 ;;
  esac
}

downstream_ok() {
  [[ -s "$1" ]] && "$PY" - "$1" >/dev/null 2>&1 <<'PY'
import json
import sys

EXPECTED = {
    "frozen",
    "stratified",
    "warm_start_full_old_generator",
    "ema_streaming_vq_old_generator",
    "full_old_generator",
    "full_old_generator_centroid_relabel",
    "full_old_generator_assignment_relabel",
    "grm_only_retrained_generator",
    "full_retrained_generator",
}

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
strategies = payload.get("strategies")
names = {row.get("strategy") for row in strategies or []}
raise SystemExit(0 if EXPECTED <= names else 1)
PY
}

run_downstream() {
  local category=$1
  local arch=$2
  local freeze_depth=$3
  local seed=$4
  local gpu=$5
  local cache
  local output

  cache=$(cache_for_category "$category")
  output="$OUT_ROOT/downstream_${category}_${arch}_fd${freeze_depth}_seed${seed}.json"

  if [[ ! -s "$cache" ]]; then
    echo "missing cache $cache" >&2
    exit 1
  fi

  if downstream_ok "$output"; then
    log "skip complete $output"
    return
  fi

  log "run category=$category arch=$arch fd=$freeze_depth seed=$seed gpu=$gpu"
  (
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" run_wsdm_web_recsys.py \
      --dataset amazon \
      --cache "$cache" \
      --arch "$arch" \
      --seed "$seed" \
      --freeze-depth "$freeze_depth" \
      --epochs "$EPOCHS" \
      --n-beams "$N_BEAMS" \
      --device cuda:0 \
      --output "$output" \
      --overwrite
  )
}

summarize() {
  log "summarize $OUT_ROOT"
  (
    cd "$ROOT"
    "$PY" summarize_wsdm_results.py --results-dir "$OUT_ROOT"
  )
}

worker() {
  local gpu=$1
  local parity=$2
  local i=0

  log "worker gpu=$gpu parity=$parity start"
  for category in $CATEGORIES; do
    for arch in $ARCHES; do
      for freeze_depth in $FREEZE_DEPTHS; do
        for seed in $SEEDS; do
          if (( i % 2 == parity )); then
            run_downstream "$category" "$arch" "$freeze_depth" "$seed" "$gpu"
          fi
          i=$((i + 1))
        done
      done
    done
  done
  log "worker gpu=$gpu parity=$parity done"
}

main() {
  log "local Amazon2023 downstream queue start"
  log "out_root=$OUT_ROOT categories=$CATEGORIES arches=$ARCHES freeze_depths=$FREEZE_DEPTHS seeds=$SEEDS"

  worker 0 0 >"$LOG/gpu0.log" 2>&1 &
  local gpu0_pid=$!
  worker 1 1 >"$LOG/gpu1.log" 2>&1 &
  local gpu1_pid=$!

  echo "$gpu0_pid" >"$LOG/gpu0.pid"
  echo "$gpu1_pid" >"$LOG/gpu1.pid"
  log "workers launched gpu0=$gpu0_pid gpu1=$gpu1_pid"

  local status0=0
  local status1=0
  wait "$gpu0_pid" || status0=$?
  wait "$gpu1_pid" || status1=$?
  log "workers finished status0=$status0 status1=$status1"

  if [[ "$status0" != "0" || "$status1" != "0" ]]; then
    exit 1
  fi
  summarize
  log "local Amazon2023 downstream queue done"
}

main "$@"

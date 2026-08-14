#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -z "${BUNDLE_ROOT:-}" ]]; then
  if [[ -d "$SCRIPT_DIR/rqd" && -d "$SCRIPT_DIR/cache" ]]; then
    BUNDLE_ROOT=$SCRIPT_DIR
  else
    BUNDLE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
  fi
fi
ROOT=${ROOT:-$BUNDLE_ROOT/rqd}
PY=${PY:-python3}
CACHE_ROOT=${CACHE_ROOT:-$BUNDLE_ROOT/cache}
OUT_ROOT=${OUT_ROOT:-$BUNDLE_ROOT/results_remote_gpu_20260814}
LOG="$OUT_ROOT/logs"

CATEGORIES=${CATEGORIES:-beauty tools toys}
ARCHES=${ARCHES:-funnel24}
FREEZE_DEPTHS=${FREEZE_DEPTHS:-1 2 3}
SEEDS=${SEEDS:-0 1 2}
EPOCHS=${EPOCHS:-50}
N_BEAMS=${N_BEAMS:-10}

mkdir -p "$LOG" "$OUT_ROOT"

exec 9>"$OUT_ROOT/remote_gpu_queue.lock"
if ! flock -n 9; then
  echo "remote GPU queue already running; lock=$OUT_ROOT/remote_gpu_queue.lock"
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
    beauty) echo "$CACHE_ROOT/amazon2023_5core_beauty_and_personal_care64.npz" ;;
    tools) echo "$CACHE_ROOT/amazon2023_5core_tools_and_home_improvement64.npz" ;;
    toys) echo "$CACHE_ROOT/amazon2023_5core_toys_and_games64.npz" ;;
    amazon2018) echo "$CACHE_ROOT/amazon2018_electronics_full5core64.npz" ;;
    movielens) echo "$CACHE_ROOT/movielens64.npz" ;;
    *) echo "unknown category: $1" >&2; exit 2 ;;
  esac
}

dataset_for_category() {
  case "$1" in
    movielens) echo "movielens" ;;
    *) echo "amazon" ;;
  esac
}

downstream_ok() {
  [[ -s "$1" ]] && "$PY" - "$1" >/dev/null 2>&1 <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
strategies = payload.get("strategies")
raise SystemExit(0 if isinstance(strategies, list) and len(strategies) > 0 else 1)
PY
}

run_downstream() {
  local category=$1
  local arch=$2
  local freeze_depth=$3
  local seed=$4
  local gpu=$5
  local dataset
  local cache
  local output

  dataset=$(dataset_for_category "$category")
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
      --dataset "$dataset" \
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
  log "remote GPU queue start"
  log "bundle_root=$BUNDLE_ROOT root=$ROOT cache_root=$CACHE_ROOT out_root=$OUT_ROOT"
  log "categories=$CATEGORIES arches=$ARCHES freeze_depths=$FREEZE_DEPTHS seeds=$SEEDS"

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
  log "remote GPU queue done"
}

main "$@"

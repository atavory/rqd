#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/users/atavory/scratch/wsdm_experiments/rqd}
PY=${PY:-/data/users/atavory/scratch/wsdm_experiments/venv/bin/python}
OUT_ROOT=${OUT_ROOT:-/data/users/atavory/scratch/wsdm_experiments/results/amazon2023_downstream_rung_funnel24_20260814}
LOG_ROOT=${LOG_ROOT:-$OUT_ROOT/followon_logs}

CATEGORIES=${CATEGORIES:-books electronics clothing home}
ARCHES=${ARCHES:-funnel24}
FREEZE_DEPTHS=${FREEZE_DEPTHS:-1 2 3}
SEEDS=${SEEDS:-0 1 2 3 4}
EPOCHS=${EPOCHS:-50}
N_BEAMS=${N_BEAMS:-10}
WAIT_SECONDS=${WAIT_SECONDS:-300}

mkdir -p "$LOG_ROOT"
echo "$$" >"$LOG_ROOT/followon.pid"

ts() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(ts)] $*"
}

queue_lock_available() {
  (
    exec 8>"$OUT_ROOT/queue.lock"
    flock -n 8
  )
}

wait_for_current_queue() {
  while ! queue_lock_available; do
    log "waiting for active downstream queue lock: $OUT_ROOT/queue.lock"
    sleep "$WAIT_SECONDS"
  done
}

main() {
  log "Amazon2023 downstream full follow-on start"
  log "out_root=$OUT_ROOT categories=$CATEGORIES arches=$ARCHES freeze_depths=$FREEZE_DEPTHS seeds=$SEEDS"
  wait_for_current_queue
  log "downstream lock available; launching full skip-aware queue"
  (
    cd "$ROOT"
    PY="$PY" \
      OUT_ROOT="$OUT_ROOT" \
      CATEGORIES="$CATEGORIES" \
      ARCHES="$ARCHES" \
      FREEZE_DEPTHS="$FREEZE_DEPTHS" \
      SEEDS="$SEEDS" \
      EPOCHS="$EPOCHS" \
      N_BEAMS="$N_BEAMS" \
      ./run_local_gpu_amazon2023_downstream_queue_20260814.sh
  )
  log "Amazon2023 downstream full follow-on done"
}

main "$@"

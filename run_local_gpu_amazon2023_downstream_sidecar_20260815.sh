#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/users/atavory/scratch/wsdm_experiments/rqd}
PY=${PY:-/data/users/atavory/scratch/wsdm_experiments/venv/bin/python}
CACHE_ROOT=${CACHE_ROOT:-/data/users/atavory/scratch/wsdm_experiments/cache}
OUT_ROOT=${OUT_ROOT:-/data/users/atavory/scratch/wsdm_experiments/results/amazon2023_downstream_rung_funnel24_20260814}
TMP_ROOT=${TMP_ROOT:-$OUT_ROOT/sidecar_tmp}
LOG_ROOT=${LOG_ROOT:-$OUT_ROOT/sidecar_logs}
EPOCHS=${EPOCHS:-50}
N_BEAMS=${N_BEAMS:-10}

mkdir -p "$TMP_ROOT" "$LOG_ROOT" "$OUT_ROOT"

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

install_complete() {
  local tmp_output=$1
  local final_output=$2

  if ! downstream_ok "$tmp_output"; then
    log "not installing incomplete tmp_output=$tmp_output"
    return 1
  fi
  if downstream_ok "$final_output"; then
    log "canonical already complete; leaving tmp_output=$tmp_output final_output=$final_output"
    return 0
  fi
  if pgrep -af "run_wsdm_web_recsys.py.*--output ${final_output}" >/dev/null; then
    log "canonical row currently active elsewhere; leaving tmp_output=$tmp_output final_output=$final_output"
    return 0
  fi
  install -m 0644 "$tmp_output" "$final_output"
  log "installed complete sidecar row final_output=$final_output"
}

run_row() {
  local category=$1
  local arch=$2
  local freeze_depth=$3
  local seed=$4
  local gpu=$5
  local cache
  local name
  local tmp_output
  local final_output

  cache=$(cache_for_category "$category")
  name="downstream_${category}_${arch}_fd${freeze_depth}_seed${seed}.json"
  tmp_output="$TMP_ROOT/$name"
  final_output="$OUT_ROOT/$name"

  if [[ ! -s "$cache" ]]; then
    echo "missing cache $cache" >&2
    exit 1
  fi
  if downstream_ok "$final_output"; then
    log "skip canonical complete $final_output"
    return
  fi
  if downstream_ok "$tmp_output"; then
    install_complete "$tmp_output" "$final_output"
    return
  fi

  log "sidecar run category=$category arch=$arch fd=$freeze_depth seed=$seed gpu=$gpu tmp_output=$tmp_output"
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
      --output "$tmp_output" \
      --overwrite
  )
  install_complete "$tmp_output" "$final_output"
}

main() {
  log "Amazon2023 downstream sidecar start"
  log "tmp_root=$TMP_ROOT out_root=$OUT_ROOT"
  echo "$$" >"$LOG_ROOT/sidecar.pid"

  run_row clothing funnel24 1 0 0 >"$LOG_ROOT/clothing_fd1_seed0.log" 2>&1 &
  local pid0=$!
  run_row home funnel24 1 0 1 >"$LOG_ROOT/home_fd1_seed0.log" 2>&1 &
  local pid1=$!
  log "sidecar workers launched clothing=$pid0 home=$pid1"

  local status0=0
  local status1=0
  wait "$pid0" || status0=$?
  wait "$pid1" || status1=$?
  log "sidecar workers finished status0=$status0 status1=$status1"
  if [[ "$status0" != "0" || "$status1" != "0" ]]; then
    exit 1
  fi
  log "Amazon2023 downstream sidecar done"
}

main "$@"

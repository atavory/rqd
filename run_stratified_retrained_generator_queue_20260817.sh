#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/users/atavory/scratch/wsdm_experiments
PY=${PY:-$ROOT/venv/bin/python}
OUT=${OUT:-$ROOT/results/stratified_retrained_generator_confirm_20260817}
DEVICE=${DEVICE:-cuda:0}
ARCH=${ARCH:-funnel24}
ARCHES=${ARCHES:-$ARCH}
FREEZE_DEPTHS=${FREEZE_DEPTHS:-2 3}
SEEDS=${SEEDS:-0 1 2 3 4}
EPOCHS=${EPOCHS:-50}
N_BEAMS=${N_BEAMS:-10}
RUN_TRAIN_SEQUENCE_LIMIT=${RUN_TRAIN_SEQUENCE_LIMIT:-0}
RUN_EVAL_SEQUENCE_LIMIT=${RUN_EVAL_SEQUENCE_LIMIT:-0}
INCLUDE_MOVIELENS=${INCLUDE_MOVIELENS:-0}
INCLUDE_AMAZON2018=${INCLUDE_AMAZON2018:-1}
INCLUDE_AMAZON2023_ELECTRONICS=${INCLUDE_AMAZON2023_ELECTRONICS:-1}
INCLUDE_AMAZON2023_BOOKS=${INCLUDE_AMAZON2023_BOOKS:-0}
INCLUDE_AMAZON2023_BEAUTY=${INCLUDE_AMAZON2023_BEAUTY:-0}
WAIT_FOR_EXISTING=${WAIT_FOR_EXISTING:-1}

mkdir -p "$OUT"
LOG="$OUT/master.log"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$LOG"
}

is_complete() {
  local path=$1
  "$PY" - "$path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
try:
    payload = json.loads(path.read_text())
except Exception:
    raise SystemExit(1)
names = {row.get("strategy") for row in payload.get("strategies", [])}
required = {"grm_only_retrained_generator", "stratified_retrained_generator"}
raise SystemExit(0 if required <= names else 1)
PY
}

summarize() {
  "$PY" summarize_wsdm_results.py --results-dir "$OUT" --output-dir "$OUT/csvs"
}

wait_for_existing() {
  if [[ "$WAIT_FOR_EXISTING" != "1" ]]; then
    return
  fi
  while pgrep -f 'run_wsdm_web_recsys.py .*--device '"$DEVICE" >/dev/null; do
    log "waiting for existing run_wsdm_web_recsys.py on $DEVICE"
    sleep 300
  done
}

run_one() {
  local label=$1
  local dataset=$2
  local cache=$3
  local arch=$4
  local fd=$5
  local seed=$6
  local output="$OUT/downstream_${label}_${arch}_fd${fd}_seed${seed}_stratified_retrained_generator.json"

  if is_complete "$output"; then
    log "skip complete $output"
    return
  fi

  wait_for_existing
  log "run stratified-retrained-generator label=$label arch=$arch fd=$fd seed=$seed"
  "$PY" run_wsdm_web_recsys.py \
    --dataset "$dataset" \
    --cache "$cache" \
    --arch "$arch" \
    --freeze-depth "$fd" \
    --seed "$seed" \
    --epochs "$EPOCHS" \
    --n-beams "$N_BEAMS" \
    --run-train-sequence-limit "$RUN_TRAIN_SEQUENCE_LIMIT" \
    --run-eval-sequence-limit "$RUN_EVAL_SEQUENCE_LIMIT" \
    --device "$DEVICE" \
    --stratified-retrained-only \
    --output "$output" \
    --overwrite \
    >>"$LOG" 2>&1
  summarize >>"$LOG" 2>&1
}

log "stratified-retrained-generator queue start OUT=$OUT DEVICE=$DEVICE"

for seed in $SEEDS; do
  for arch in $ARCHES; do
    for fd in $FREEZE_DEPTHS; do
      if [[ "$INCLUDE_MOVIELENS" == "1" ]]; then
        run_one \
          movielens \
          movielens \
          "$ROOT/cache/movielens64.npz" \
          "$arch" \
          "$fd" \
          "$seed"
      fi
      if [[ "$INCLUDE_AMAZON2018" == "1" ]]; then
        run_one \
          amazon2018_electronics \
          amazon \
          "$ROOT/cache/amazon2018_electronics_full5core64.npz" \
          "$arch" \
          "$fd" \
          "$seed"
      fi
      if [[ "$INCLUDE_AMAZON2023_ELECTRONICS" == "1" ]]; then
        run_one \
          amazon2023_electronics \
          amazon \
          "$ROOT/cache/amazon2023_5core_electronics64.npz" \
          "$arch" \
          "$fd" \
          "$seed"
      fi
      if [[ "$INCLUDE_AMAZON2023_BOOKS" == "1" ]]; then
        run_one \
          amazon2023_books \
          amazon \
          "$ROOT/cache/amazon2023_5core_books64.npz" \
          "$arch" \
          "$fd" \
          "$seed"
      fi
      if [[ "$INCLUDE_AMAZON2023_BEAUTY" == "1" ]]; then
        run_one \
          amazon2023_beauty \
          amazon \
          "$ROOT/cache/amazon2023_5core_beauty_and_personal_care64.npz" \
          "$arch" \
          "$fd" \
          "$seed"
      fi
    done
  done
done

summarize >>"$LOG" 2>&1
log "stratified-retrained-generator queue done"

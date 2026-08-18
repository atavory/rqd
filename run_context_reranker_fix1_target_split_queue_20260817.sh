#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/users/atavory/scratch/wsdm_experiments
PY=${PY:-$ROOT/venv/bin/python}
OUT=${OUT:-$ROOT/results/context_reranker_fix1_target_split_20260817}
DEVICE=${DEVICE:-cuda:1}
ARCH=${ARCH:-funnel24}
ARCHES=${ARCHES:-$ARCH}
FREEZE_DEPTHS=${FREEZE_DEPTHS:-2 3}
SEEDS=${SEEDS:-0 1 2 3 4}
EPOCHS=${EPOCHS:-50}
SCORER_EPOCHS=${SCORER_EPOCHS:-20}
SCORER_NEGATIVES=${SCORER_NEGATIVES:-2048}
N_BEAMS=${N_BEAMS:-10}
RUN_TRAIN_SEQUENCE_LIMIT=${RUN_TRAIN_SEQUENCE_LIMIT:-0}
RUN_EVAL_SEQUENCE_LIMIT=${RUN_EVAL_SEQUENCE_LIMIT:-0}
FIX1_TARGET_SPLITS=${FIX1_TARGET_SPLITS:-new_source_zero_target_nonzero}
INCLUDE_AMAZON2018=${INCLUDE_AMAZON2018:-1}
INCLUDE_AMAZON2023_ELECTRONICS=${INCLUDE_AMAZON2023_ELECTRONICS:-1}
INCLUDE_AMAZON2023_BOOKS=${INCLUDE_AMAZON2023_BOOKS:-0}
INCLUDE_AMAZON2023_BEAUTY=${INCLUDE_AMAZON2023_BEAUTY:-0}
INCLUDE_AMAZON2023_CLOTHING=${INCLUDE_AMAZON2023_CLOTHING:-0}
INCLUDE_AMAZON2023_HOME=${INCLUDE_AMAZON2023_HOME:-0}
INCLUDE_AMAZON2023_TOOLS=${INCLUDE_AMAZON2023_TOOLS:-0}
INCLUDE_AMAZON2023_TOYS=${INCLUDE_AMAZON2023_TOYS:-0}
SCORER_FINETUNE_MODEL=${SCORER_FINETUNE_MODEL:-0}
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
rows = payload.get("context_reranker_target_item_split_rows", [])
raise SystemExit(0 if len(rows) >= 6 else 1)
PY
}

summarize() {
  "$PY" summarize_wsdm_results.py --results-dir "$OUT" --output-dir "$OUT/csvs"
}

wait_for_existing() {
  if [[ "$WAIT_FOR_EXISTING" != "1" ]]; then
    return
  fi
  while pgrep -f 'run_context_reranker_recsys.py .*--device '"$DEVICE" >/dev/null; do
    log "waiting for existing run_context_reranker_recsys.py on $DEVICE"
    sleep 300
  done
}

run_one() {
  local label=$1
  local cache=$2
  local arch=$3
  local fd=$4
  local seed=$5
  local suffix=fix1_new
  local finetune_args=()

  if [[ "$SCORER_FINETUNE_MODEL" == "1" ]]; then
    suffix=fix1_new_finetuned
    finetune_args=(--scorer-finetune-model)
  fi

  local output="$OUT/context_${label}_${arch}_fd${fd}_seed${seed}_${suffix}.json"

  if is_complete "$output"; then
    log "skip complete $output"
    return
  fi

  wait_for_existing
  log "run context FIX-1 label=$label arch=$arch fd=$fd seed=$seed split=$FIX1_TARGET_SPLITS finetune=$SCORER_FINETUNE_MODEL"
  "$PY" run_context_reranker_recsys.py \
    --cache "$cache" \
    --arch "$arch" \
    --freeze-depth "$fd" \
    --seed "$seed" \
    --epochs "$EPOCHS" \
    --scorer-epochs "$SCORER_EPOCHS" \
    --scorer-negatives "$SCORER_NEGATIVES" \
    --n-beams "$N_BEAMS" \
    --run-train-sequence-limit "$RUN_TRAIN_SEQUENCE_LIMIT" \
    --run-eval-sequence-limit "$RUN_EVAL_SEQUENCE_LIMIT" \
    --device "$DEVICE" \
    --fix1-target-split-only \
    --fix1-target-splits "$FIX1_TARGET_SPLITS" \
    "${finetune_args[@]}" \
    --output "$output" \
    --overwrite \
    >>"$LOG" 2>&1
  summarize >>"$LOG" 2>&1
}

log "context FIX-1 target-split queue start OUT=$OUT DEVICE=$DEVICE"

for seed in $SEEDS; do
  for arch in $ARCHES; do
    for fd in $FREEZE_DEPTHS; do
      if [[ "$INCLUDE_AMAZON2018" == "1" ]]; then
        run_one \
          amazon2018_electronics \
          "$ROOT/cache/amazon2018_electronics_full5core64.npz" \
          "$arch" \
          "$fd" \
          "$seed"
      fi
      if [[ "$INCLUDE_AMAZON2023_ELECTRONICS" == "1" ]]; then
        run_one \
          amazon2023_electronics \
          "$ROOT/cache/amazon2023_5core_electronics64.npz" \
          "$arch" \
          "$fd" \
          "$seed"
      fi
      if [[ "$INCLUDE_AMAZON2023_BOOKS" == "1" ]]; then
        run_one \
          amazon2023_books \
          "$ROOT/cache/amazon2023_5core_books64.npz" \
          "$arch" \
          "$fd" \
          "$seed"
      fi
      if [[ "$INCLUDE_AMAZON2023_BEAUTY" == "1" ]]; then
        run_one \
          amazon2023_beauty \
          "$ROOT/cache/amazon2023_5core_beauty_and_personal_care64.npz" \
          "$arch" \
          "$fd" \
          "$seed"
      fi
      if [[ "$INCLUDE_AMAZON2023_CLOTHING" == "1" ]]; then
        run_one \
          amazon2023_clothing \
          "$ROOT/cache/amazon2023_5core_clothing_shoes_and_jewelry64.npz" \
          "$arch" \
          "$fd" \
          "$seed"
      fi
      if [[ "$INCLUDE_AMAZON2023_HOME" == "1" ]]; then
        run_one \
          amazon2023_home \
          "$ROOT/cache/amazon2023_5core_home_and_kitchen64.npz" \
          "$arch" \
          "$fd" \
          "$seed"
      fi
      if [[ "$INCLUDE_AMAZON2023_TOOLS" == "1" ]]; then
        run_one \
          amazon2023_tools \
          "$ROOT/cache/amazon2023_5core_tools_and_home_improvement64.npz" \
          "$arch" \
          "$fd" \
          "$seed"
      fi
      if [[ "$INCLUDE_AMAZON2023_TOYS" == "1" ]]; then
        run_one \
          amazon2023_toys \
          "$ROOT/cache/amazon2023_5core_toys_and_games64.npz" \
          "$arch" \
          "$fd" \
          "$seed"
      fi
    done
  done
done

summarize >>"$LOG" 2>&1
log "context FIX-1 target-split queue done"

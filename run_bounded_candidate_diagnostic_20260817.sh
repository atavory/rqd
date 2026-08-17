#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/users/atavory/scratch/wsdm_experiments
PY=${PY:-$ROOT/venv/bin/python}
OUT=${OUT:-$ROOT/results/bounded_candidate_diagnostic_20260817}
DEVICE=${DEVICE:-cuda:1}
EPOCHS=${EPOCHS:-50}
N_BEAMS=${N_BEAMS:-10}
CANDIDATE_BUDGET_VALUES=${CANDIDATE_BUDGET_VALUES:-200,500,1000}

mkdir -p "$OUT"
LOG="$OUT/master.log"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$LOG"
}

is_complete() {
  local path=$1
  local expected=$2
  "$PY" - "$path" "$expected" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
if not path.exists():
    raise SystemExit(1)
try:
    payload = json.loads(path.read_text())
except Exception:
    raise SystemExit(1)
rows = payload.get("candidate_budget_sweep", [])
raise SystemExit(0 if len(rows) >= expected else 1)
PY
}

summarize() {
  "$PY" summarize_wsdm_results.py --results-dir "$OUT" --output-dir "$OUT/csvs"
}

run_one() {
  local dataset=$1
  local cache=$2
  local arch=$3
  local freeze_depth=$4
  local seed=$5
  local train_limit=$6
  local eval_limit=$7
  local label=$8
  local output="$OUT/downstream_${label}_${arch}_fd${freeze_depth}_seed${seed}.json"
  local expected=$((7 * 3))

  if is_complete "$output" "$expected"; then
    log "skip complete $output"
    return
  fi

  log "run dataset=$dataset label=$label arch=$arch fd=$freeze_depth seed=$seed budgets=$CANDIDATE_BUDGET_VALUES"
  "$PY" run_wsdm_web_recsys.py \
    --dataset "$dataset" \
    --cache "$cache" \
    --arch "$arch" \
    --freeze-depth "$freeze_depth" \
    --seed "$seed" \
    --epochs "$EPOCHS" \
    --n-beams "$N_BEAMS" \
    --candidate-budget-values "$CANDIDATE_BUDGET_VALUES" \
    --run-train-sequence-limit "$train_limit" \
    --run-eval-sequence-limit "$eval_limit" \
    --device "$DEVICE" \
    --skip-rebuilt-consumer \
    --output "$output" \
    --overwrite \
    >>"$LOG" 2>&1
  summarize >>"$LOG" 2>&1
}

log "bounded-candidate diagnostic start OUT=$OUT DEVICE=$DEVICE"

for seed in 0 1 2 3 4; do
  for fd in 2 3; do
    run_one \
      movielens \
      "$ROOT/cache/movielens64.npz" \
      funnel24 \
      "$fd" \
      "$seed" \
      0 \
      1000 \
      movielens
  done
done

for seed in 0 1 2; do
  for fd in 2 3; do
    run_one \
      amazon \
      "$ROOT/cache/amazon2018_electronics_full5core64.npz" \
      funnel24 \
      "$fd" \
      "$seed" \
      10000 \
      2000 \
      amazon2018_electronics
  done
done

summarize >>"$LOG" 2>&1
log "bounded-candidate diagnostic done"

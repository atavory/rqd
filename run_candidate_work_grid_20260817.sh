#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/users/atavory/scratch/wsdm_experiments
PY=${PY:-$ROOT/venv/bin/python}
OUT=${OUT:-$ROOT/results/candidate_work_grid_20260817}
DEVICE=${DEVICE:-cuda:1}
EPOCHS=${EPOCHS:-50}
N_BEAMS=${N_BEAMS:-10}
CANDIDATE_BUDGET_VALUES=${CANDIDATE_BUDGET_VALUES:-50,100,200,500,1000}
CANDIDATE_GRID_BEAM_VALUES=${CANDIDATE_GRID_BEAM_VALUES:-1,2,5,10}
TRAIN_LIMIT=${TRAIN_LIMIT:-0}
MOVIELENS_EVAL_LIMIT=${MOVIELENS_EVAL_LIMIT:-1000}
AMAZON2018_TRAIN_LIMIT=${AMAZON2018_TRAIN_LIMIT:-10000}
AMAZON2018_EVAL_LIMIT=${AMAZON2018_EVAL_LIMIT:-1000}
WAIT_FOR_BOUNDED_QUEUE=${WAIT_FOR_BOUNDED_QUEUE:-1}

mkdir -p "$OUT"
LOG="$OUT/master.log"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$LOG"
}

count_values() {
  local values=$1
  "$PY" - "$values" <<'PY'
import sys

values = [value for value in sys.argv[1].split(",") if value]
print(len(set(values)))
PY
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
rows = payload.get("candidate_grid_sweep", [])
raise SystemExit(0 if len(rows) >= expected else 1)
PY
}

summarize() {
  "$PY" summarize_wsdm_results.py --results-dir "$OUT" --output-dir "$OUT/csvs"
}

wait_for_bounded_queue() {
  if [[ "$WAIT_FOR_BOUNDED_QUEUE" != "1" ]]; then
    return
  fi
  while pgrep -f 'run_bounded_candidate_diagnostic_20260817.sh' >/dev/null; do
    log "waiting for bounded-candidate queue to release $DEVICE"
    sleep 300
  done
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
  local n_budgets
  local n_grid_beams
  n_budgets=$(count_values "$CANDIDATE_BUDGET_VALUES")
  n_grid_beams=$(count_values "$CANDIDATE_GRID_BEAM_VALUES")
  local expected=$((7 * n_budgets * n_grid_beams))

  if is_complete "$output" "$expected"; then
    log "skip complete $output"
    return
  fi

  log "run grid dataset=$dataset label=$label arch=$arch fd=$freeze_depth seed=$seed beams=$CANDIDATE_GRID_BEAM_VALUES budgets=$CANDIDATE_BUDGET_VALUES"
  "$PY" run_wsdm_web_recsys.py \
    --dataset "$dataset" \
    --cache "$cache" \
    --arch "$arch" \
    --freeze-depth "$freeze_depth" \
    --seed "$seed" \
    --epochs "$EPOCHS" \
    --n-beams "$N_BEAMS" \
    --candidate-budget-values "$CANDIDATE_BUDGET_VALUES" \
    --candidate-grid-beam-values "$CANDIDATE_GRID_BEAM_VALUES" \
    --run-train-sequence-limit "$train_limit" \
    --run-eval-sequence-limit "$eval_limit" \
    --device "$DEVICE" \
    --skip-rebuilt-consumer \
    --output "$output" \
    --overwrite \
    >>"$LOG" 2>&1
  summarize >>"$LOG" 2>&1
}

log "candidate work grid start OUT=$OUT DEVICE=$DEVICE"
wait_for_bounded_queue

for seed in 0 1 2 3 4; do
  for fd in 2 3; do
    run_one \
      movielens \
      "$ROOT/cache/movielens64.npz" \
      funnel24 \
      "$fd" \
      "$seed" \
      "$TRAIN_LIMIT" \
      "$MOVIELENS_EVAL_LIMIT" \
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
      "$AMAZON2018_TRAIN_LIMIT" \
      "$AMAZON2018_EVAL_LIMIT" \
      amazon2018_electronics
  done
done

summarize >>"$LOG" 2>&1
log "candidate work grid done"

#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/users/atavory/scratch/wsdm_experiments
PY=${PY:-$ROOT/venv/bin/python}
OUT=${OUT:-$ROOT/results/context_reranker_20260817}
DEVICE=${DEVICE:-cuda:1}
EPOCHS=${EPOCHS:-50}
SCORER_EPOCHS=${SCORER_EPOCHS:-20}
SCORER_NEGATIVES=${SCORER_NEGATIVES:-2048}
N_BEAMS=${N_BEAMS:-10}
CANDIDATE_BUDGET_VALUES=${CANDIDATE_BUDGET_VALUES:-200,500,1000}
CANDIDATE_GRID_BEAM_VALUES=${CANDIDATE_GRID_BEAM_VALUES:-10}
MOVIELENS_EVAL_LIMIT=${MOVIELENS_EVAL_LIMIT:-1000}
AMAZON2018_TRAIN_LIMIT=${AMAZON2018_TRAIN_LIMIT:-10000}
AMAZON2018_EVAL_LIMIT=${AMAZON2018_EVAL_LIMIT:-1000}
INCLUDE_MOVIELENS=${INCLUDE_MOVIELENS:-1}
INCLUDE_AMAZON2018=${INCLUDE_AMAZON2018:-1}
MOVIELENS_SEEDS=${MOVIELENS_SEEDS:-0 1 2 3 4}
AMAZON2018_SEEDS=${AMAZON2018_SEEDS:-0 1 2}
WAIT_FOR_BOUNDED_QUEUE=${WAIT_FOR_BOUNDED_QUEUE:-1}

mkdir -p "$OUT"
LOG="$OUT/master.log"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$LOG"
}

expected_grid_rows() {
  "$PY" - "$CANDIDATE_BUDGET_VALUES" "$CANDIDATE_GRID_BEAM_VALUES" <<'PY'
import sys

budgets = {value for value in sys.argv[1].split(",") if value}
beams = {value for value in sys.argv[2].split(",") if value}
print(7 * len(budgets) * len(beams))
PY
}

is_complete() {
  local path=$1
  local expected_grid=$2
  "$PY" - "$path" "$expected_grid" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_grid = int(sys.argv[2])
if not path.exists():
    raise SystemExit(1)
try:
    payload = json.loads(path.read_text())
except Exception:
    raise SystemExit(1)
rows = payload.get("context_reranker_rows", [])
grid = payload.get("context_reranker_grid", [])
raise SystemExit(0 if len(rows) >= 7 and len(grid) >= expected_grid else 1)
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
  local label=$1
  local cache=$2
  local arch=$3
  local fd=$4
  local seed=$5
  local train_limit=$6
  local eval_limit=$7
  local output="$OUT/context_${label}_${arch}_fd${fd}_seed${seed}.json"
  local expected_grid
  expected_grid=$(expected_grid_rows)

  if is_complete "$output" "$expected_grid"; then
    log "skip complete $output"
    return
  fi

  log "run context reranker label=$label arch=$arch fd=$fd seed=$seed beams=$CANDIDATE_GRID_BEAM_VALUES budgets=$CANDIDATE_BUDGET_VALUES"
  "$PY" run_context_reranker_recsys.py \
    --cache "$cache" \
    --arch "$arch" \
    --freeze-depth "$fd" \
    --seed "$seed" \
    --epochs "$EPOCHS" \
    --scorer-epochs "$SCORER_EPOCHS" \
    --scorer-negatives "$SCORER_NEGATIVES" \
    --n-beams "$N_BEAMS" \
    --candidate-budget-values "$CANDIDATE_BUDGET_VALUES" \
    --candidate-grid-beam-values "$CANDIDATE_GRID_BEAM_VALUES" \
    --run-train-sequence-limit "$train_limit" \
    --run-eval-sequence-limit "$eval_limit" \
    --device "$DEVICE" \
    --output "$output" \
    --overwrite \
    >>"$LOG" 2>&1
  summarize >>"$LOG" 2>&1
}

log "context-reranker queue start OUT=$OUT DEVICE=$DEVICE"
wait_for_bounded_queue

if [[ "$INCLUDE_MOVIELENS" == "1" ]]; then
  for seed in $MOVIELENS_SEEDS; do
    for fd in 2 3; do
      run_one \
        movielens \
        "$ROOT/cache/movielens64.npz" \
        funnel24 \
        "$fd" \
        "$seed" \
        0 \
        "$MOVIELENS_EVAL_LIMIT"
    done
  done
fi

if [[ "$INCLUDE_AMAZON2018" == "1" ]]; then
  for seed in $AMAZON2018_SEEDS; do
    for fd in 2 3; do
      run_one \
        amazon2018_electronics \
        "$ROOT/cache/amazon2018_electronics_full5core64.npz" \
        funnel24 \
        "$fd" \
        "$seed" \
        "$AMAZON2018_TRAIN_LIMIT" \
        "$AMAZON2018_EVAL_LIMIT"
    done
  done
fi

summarize >>"$LOG" 2>&1
log "context-reranker queue done"

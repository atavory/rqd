#!/usr/bin/env bash
set -uo pipefail

ROOT=${ROOT:-/data/users/atavory/scratch/wsdm_experiments/rqd}
RESULTS_ROOT=${RESULTS_ROOT:-/data/users/atavory/scratch/wsdm_experiments/results}
PY=${PY:-/data/users/atavory/scratch/wsdm_experiments/venv/bin/python}
LOOP_ROOT=${LOOP_ROOT:-$RESULTS_ROOT/wsdm_paper_rescue_loop_20260817}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-300}
DEVICE_E2E=${DEVICE_E2E:-cuda:0}
DEVICE_CONTEXT=${DEVICE_CONTEXT:-cuda:1}

mkdir -p "$LOOP_ROOT/logs"
LOG="$LOOP_ROOT/logs/loop.log"
STATUS="$LOOP_ROOT/status.md"
PID_FILE="$LOOP_ROOT/logs/loop.pid"

exec 9>"$LOOP_ROOT/loop.lock"
if ! flock -n 9; then
  echo "paper rescue loop already running; lock=$LOOP_ROOT/loop.lock"
  exit 0
fi

echo "$$" >"$PID_FILE"

ts() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(ts)] $*" | tee -a "$LOG"
}

pid_alive() {
  local pid_file=$1
  [[ -s "$pid_file" ]] && ps -p "$(cat "$pid_file")" >/dev/null 2>&1
}

done_marker_present() {
  local log_file=$1
  local marker=$2
  [[ -s "$log_file" ]] && grep -qF "$marker" "$log_file"
}

e2e_lane_active() {
  pgrep -f 'run_fix1_target_split_queue_20260817.sh|run_stratified_retrained_generator_queue_20260817.sh' >/dev/null 2>&1
}

context_lane_active() {
  pgrep -f 'run_context_reranker_fix1_target_split_queue_20260817.sh' >/dev/null 2>&1
}

start_stage() {
  local label=$1
  local pid_file=$2
  local stage_log=$3
  shift 3

  mkdir -p "$(dirname "$pid_file")" "$(dirname "$stage_log")"
  log "start $label"
  (
    cd "$ROOT" || exit 1
    env "$@" >>"$stage_log" 2>&1 < /dev/null 9>&- &
    echo $! >"$pid_file"
  )
}

summarize_dir() {
  local out=$1
  if [[ -d "$out" ]]; then
    "$PY" "$ROOT/summarize_wsdm_results.py" --results-dir "$out" --output-dir "$out/csvs" >>"$LOG" 2>&1 || true
  fi
}

stage_state() {
  local log_file=$1
  local marker=$2
  local pid_file=$3
  local active_check=${4:-}
  if done_marker_present "$log_file" "$marker"; then
    echo done
  elif pid_alive "$pid_file"; then
    echo active
  elif [[ -n "$active_check" ]] && "$active_check"; then
    echo active
  else
    echo pending
  fi
}

write_status() {
  local tmp="$STATUS.tmp"
  {
    echo "# WSDM Paper Rescue Loop"
    echo
    echo "updated: $(date -Is)"
    echo "pid: $$"
    echo
    echo "## E2E Lane"
    echo "- fix1_core: $(stage_state "$RESULTS_ROOT/fix1_target_split_confirm_20260817/master.log" "FIX-1 target-split queue done" "$LOOP_ROOT/logs/e2e_fix1_core.pid" e2e_lane_active)"
    echo "- stratified_retrained_core: $(stage_state "$RESULTS_ROOT/stratified_retrained_generator_confirm_20260817/master.log" "stratified-retrained-generator queue done" "$LOOP_ROOT/logs/e2e_stratified_retrained_core.pid")"
    echo "- fix1_extra_catalogs: $(stage_state "$RESULTS_ROOT/fix1_target_split_extra_catalogs_20260817/master.log" "FIX-1 target-split queue done" "$LOOP_ROOT/logs/e2e_fix1_extra_catalogs.pid")"
    echo "- fix1_arch_sweep: $(stage_state "$RESULTS_ROOT/fix1_target_split_arch_sweep_20260817/master.log" "FIX-1 target-split queue done" "$LOOP_ROOT/logs/e2e_fix1_arch_sweep.pid")"
    echo
    echo "## Reranker Lane"
    echo "- context_fix1_core: $(stage_state "$RESULTS_ROOT/context_reranker_fix1_target_split_20260817/master.log" "context FIX-1 target-split queue done" "$LOOP_ROOT/logs/context_fix1_core.pid" context_lane_active)"
    echo "- context_fix1_finetuned: $(stage_state "$RESULTS_ROOT/context_reranker_fix1_target_split_finetuned_20260817/master.log" "context FIX-1 target-split queue done" "$LOOP_ROOT/logs/context_fix1_finetuned.pid")"
    echo "- context_fix1_extra_catalogs: $(stage_state "$RESULTS_ROOT/context_reranker_fix1_target_split_extra_catalogs_20260817/master.log" "context FIX-1 target-split queue done" "$LOOP_ROOT/logs/context_fix1_extra_catalogs.pid")"
    echo
    echo "## GPU"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true
    echo
    echo "## Active Experiment Processes"
    pgrep -af 'run_wsdm_web_recsys.py|run_context_reranker_recsys.py|run_fix1_target_split_queue_20260817.sh|run_context_reranker_fix1_target_split_queue_20260817.sh|run_stratified_retrained_generator_queue_20260817.sh' 2>/dev/null || true
  } >"$tmp"
  mv "$tmp" "$STATUS"
}

manage_e2e_lane() {
  local core_out="$RESULTS_ROOT/fix1_target_split_confirm_20260817"
  local strat_out="$RESULTS_ROOT/stratified_retrained_generator_confirm_20260817"
  local extra_out="$RESULTS_ROOT/fix1_target_split_extra_catalogs_20260817"
  local arch_out="$RESULTS_ROOT/fix1_target_split_arch_sweep_20260817"

  if ! done_marker_present "$core_out/master.log" "FIX-1 target-split queue done"; then
    if e2e_lane_active; then
      log "e2e lane busy before fix1_core is complete"
      return
    fi
    start_stage \
      "e2e fix1 core" \
      "$LOOP_ROOT/logs/e2e_fix1_core.pid" \
      "$LOOP_ROOT/logs/e2e_fix1_core.log" \
      OUT="$core_out" \
      DEVICE="$DEVICE_E2E" \
      ARCHES=funnel24 \
      FREEZE_DEPTHS="2 3" \
      SEEDS="0 1 2 3 4" \
      INCLUDE_AMAZON2018=1 \
      INCLUDE_AMAZON2023_ELECTRONICS=1 \
      INCLUDE_AMAZON2023_BOOKS=0 \
      INCLUDE_AMAZON2023_BEAUTY=0 \
      WAIT_FOR_EXISTING=1 \
      bash ./run_fix1_target_split_queue_20260817.sh
    return
  fi

  if ! done_marker_present "$strat_out/master.log" "stratified-retrained-generator queue done"; then
    if e2e_lane_active; then
      log "e2e lane busy before stratified_retrained_core is complete"
      return
    fi
    start_stage \
      "e2e stratified retrained generator core" \
      "$LOOP_ROOT/logs/e2e_stratified_retrained_core.pid" \
      "$LOOP_ROOT/logs/e2e_stratified_retrained_core.log" \
      OUT="$strat_out" \
      DEVICE="$DEVICE_E2E" \
      ARCHES=funnel24 \
      FREEZE_DEPTHS="2 3" \
      SEEDS="0 1 2 3 4" \
      INCLUDE_MOVIELENS=1 \
      INCLUDE_AMAZON2018=1 \
      INCLUDE_AMAZON2023_ELECTRONICS=1 \
      INCLUDE_AMAZON2023_BOOKS=0 \
      INCLUDE_AMAZON2023_BEAUTY=0 \
      WAIT_FOR_EXISTING=1 \
      bash ./run_stratified_retrained_generator_queue_20260817.sh
    return
  fi

  if ! done_marker_present "$extra_out/master.log" "FIX-1 target-split queue done"; then
    if e2e_lane_active; then
      log "e2e lane busy before fix1_extra_catalogs is complete"
      return
    fi
    start_stage \
      "e2e fix1 extra catalogs" \
      "$LOOP_ROOT/logs/e2e_fix1_extra_catalogs.pid" \
      "$LOOP_ROOT/logs/e2e_fix1_extra_catalogs.log" \
      OUT="$extra_out" \
      DEVICE="$DEVICE_E2E" \
      ARCHES=funnel24 \
      FREEZE_DEPTHS="2 3" \
      SEEDS="0 1 2 3 4" \
      INCLUDE_AMAZON2018=0 \
      INCLUDE_AMAZON2023_ELECTRONICS=0 \
      INCLUDE_AMAZON2023_BOOKS=1 \
      INCLUDE_AMAZON2023_BEAUTY=1 \
      WAIT_FOR_EXISTING=1 \
      bash ./run_fix1_target_split_queue_20260817.sh
    return
  fi

  if ! done_marker_present "$arch_out/master.log" "FIX-1 target-split queue done"; then
    if e2e_lane_active; then
      log "e2e lane busy before fix1_arch_sweep is complete"
      return
    fi
    start_stage \
      "e2e fix1 architecture sweep" \
      "$LOOP_ROOT/logs/e2e_fix1_arch_sweep.pid" \
      "$LOOP_ROOT/logs/e2e_fix1_arch_sweep.log" \
      OUT="$arch_out" \
      DEVICE="$DEVICE_E2E" \
      ARCHES="funnel16 funnel20 balanced24 funnel24" \
      FREEZE_DEPTHS="1 2 3" \
      SEEDS="0 1 2" \
      INCLUDE_AMAZON2018=1 \
      INCLUDE_AMAZON2023_ELECTRONICS=1 \
      INCLUDE_AMAZON2023_BOOKS=0 \
      INCLUDE_AMAZON2023_BEAUTY=0 \
      WAIT_FOR_EXISTING=1 \
      bash ./run_fix1_target_split_queue_20260817.sh
    return
  fi

  log "e2e lane all configured stages complete"
}

manage_context_lane() {
  local core_out="$RESULTS_ROOT/context_reranker_fix1_target_split_20260817"
  local finetuned_out="$RESULTS_ROOT/context_reranker_fix1_target_split_finetuned_20260817"
  local extra_out="$RESULTS_ROOT/context_reranker_fix1_target_split_extra_catalogs_20260817"

  if ! done_marker_present "$core_out/master.log" "context FIX-1 target-split queue done"; then
    if context_lane_active; then
      log "context lane busy before context_fix1_core is complete"
      return
    fi
    start_stage \
      "context fix1 core" \
      "$LOOP_ROOT/logs/context_fix1_core.pid" \
      "$LOOP_ROOT/logs/context_fix1_core.log" \
      OUT="$core_out" \
      DEVICE="$DEVICE_CONTEXT" \
      ARCHES=funnel24 \
      FREEZE_DEPTHS="2 3" \
      SEEDS="0 1 2 3 4" \
      INCLUDE_AMAZON2018=1 \
      INCLUDE_AMAZON2023_ELECTRONICS=1 \
      INCLUDE_AMAZON2023_BOOKS=0 \
      INCLUDE_AMAZON2023_BEAUTY=0 \
      SCORER_FINETUNE_MODEL=0 \
      WAIT_FOR_EXISTING=1 \
      bash ./run_context_reranker_fix1_target_split_queue_20260817.sh
    return
  fi

  if ! done_marker_present "$finetuned_out/master.log" "context FIX-1 target-split queue done"; then
    if context_lane_active; then
      log "context lane busy before context_fix1_finetuned is complete"
      return
    fi
    start_stage \
      "context fix1 finetuned scorer core" \
      "$LOOP_ROOT/logs/context_fix1_finetuned.pid" \
      "$LOOP_ROOT/logs/context_fix1_finetuned.log" \
      OUT="$finetuned_out" \
      DEVICE="$DEVICE_CONTEXT" \
      ARCHES=funnel24 \
      FREEZE_DEPTHS="2 3" \
      SEEDS="0 1 2" \
      INCLUDE_AMAZON2018=1 \
      INCLUDE_AMAZON2023_ELECTRONICS=1 \
      INCLUDE_AMAZON2023_BOOKS=0 \
      INCLUDE_AMAZON2023_BEAUTY=0 \
      SCORER_FINETUNE_MODEL=1 \
      WAIT_FOR_EXISTING=1 \
      bash ./run_context_reranker_fix1_target_split_queue_20260817.sh
    return
  fi

  if ! done_marker_present "$extra_out/master.log" "context FIX-1 target-split queue done"; then
    if context_lane_active; then
      log "context lane busy before context_fix1_extra_catalogs is complete"
      return
    fi
    start_stage \
      "context fix1 extra catalogs" \
      "$LOOP_ROOT/logs/context_fix1_extra_catalogs.pid" \
      "$LOOP_ROOT/logs/context_fix1_extra_catalogs.log" \
      OUT="$extra_out" \
      DEVICE="$DEVICE_CONTEXT" \
      ARCHES=funnel24 \
      FREEZE_DEPTHS="2 3" \
      SEEDS="0 1 2" \
      INCLUDE_AMAZON2018=0 \
      INCLUDE_AMAZON2023_ELECTRONICS=0 \
      INCLUDE_AMAZON2023_BOOKS=1 \
      INCLUDE_AMAZON2023_BEAUTY=1 \
      SCORER_FINETUNE_MODEL=0 \
      WAIT_FOR_EXISTING=1 \
      bash ./run_context_reranker_fix1_target_split_queue_20260817.sh
    return
  fi

  log "context lane all configured stages complete"
}

heartbeat() {
  summarize_dir "$RESULTS_ROOT/fix1_target_split_confirm_20260817"
  summarize_dir "$RESULTS_ROOT/context_reranker_fix1_target_split_20260817"
  summarize_dir "$RESULTS_ROOT/stratified_retrained_generator_confirm_20260817"
  summarize_dir "$RESULTS_ROOT/fix1_target_split_extra_catalogs_20260817"
  summarize_dir "$RESULTS_ROOT/context_reranker_fix1_target_split_finetuned_20260817"
  summarize_dir "$RESULTS_ROOT/context_reranker_fix1_target_split_extra_catalogs_20260817"
  summarize_dir "$RESULTS_ROOT/fix1_target_split_arch_sweep_20260817"
  summarize_dir "$RESULTS_ROOT/fix1_target_split_scout_20260817"
  "$PY" "$ROOT/summarize_fix1_wins_20260817.py" \
    --results-root "$RESULTS_ROOT" \
    --output "$LOOP_ROOT/wins.md" >>"$LOG" 2>&1 || true
  write_status
  log "heartbeat wrote $STATUS"
}

main() {
  log "paper rescue loop start interval=${INTERVAL_SECONDS}s"
  log "loop_root=$LOOP_ROOT"
  while true; do
    manage_e2e_lane
    manage_context_lane
    heartbeat
    sleep "$INTERVAL_SECONDS" 9>&-
  done
}

main "$@"

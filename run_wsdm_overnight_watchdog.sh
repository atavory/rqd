#!/usr/bin/env bash
set -uo pipefail

ROOT=${ROOT:-/data/users/atavory/scratch/wsdm_experiments/rqd}
RESULTS_ROOT=${RESULTS_ROOT:-/data/users/atavory/scratch/wsdm_experiments/results}
WATCH_ROOT=${WATCH_ROOT:-$RESULTS_ROOT/wsdm_overnight_watchdog_20260812}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-300}

AVAILABLE_ROOT=${AVAILABLE_ROOT:-$RESULTS_ROOT/wsdm_available_full_queue_20260812}
AMAZON2023_ROOT=${AMAZON2023_ROOT:-$RESULTS_ROOT/amazon2023_acquire_prepare_index_20260812}
TIER_C_ROOT=${TIER_C_ROOT:-$RESULTS_ROOT/tier_c_retrain_prediction_20260812}
AMAZON2023_DOWNSTREAM_ROOT=${AMAZON2023_DOWNSTREAM_ROOT:-$RESULTS_ROOT/amazon2023_downstream_rung_funnel24_20260814}

mkdir -p "$WATCH_ROOT/logs"
LOG="$WATCH_ROOT/logs/watchdog.log"
PID_FILE="$WATCH_ROOT/logs/watchdog.pid"

exec 9>"$WATCH_ROOT/watchdog.lock"
if ! flock -n 9; then
  echo "overnight watchdog already running; lock=$WATCH_ROOT/watchdog.lock"
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

log_has_done_marker() {
  local log_file=$1
  local marker=$2
  [[ -s "$log_file" ]] && grep -qF "$marker" "$log_file"
}

restart_queue() {
  local label=$1
  local script=$2
  local master_log=$3
  local pid_file=$4

  mkdir -p "$(dirname "$master_log")"
  log "restart $label with $script"
  (
    cd "$ROOT" || exit 1
    setsid bash "$script" >>"$master_log" 2>&1 < /dev/null &
    echo $! >"$pid_file"
  )
}

check_queue() {
  local label=$1
  local script=$2
  local pid_file=$3
  local master_log=$4
  local done_marker=$5

  if log_has_done_marker "$master_log" "$done_marker"; then
    log "$label done marker present; no restart"
    return
  fi

  if pid_alive "$pid_file"; then
    log "$label alive pid=$(cat "$pid_file")"
    return
  fi

  log "$label not alive and not marked done"
  restart_queue "$label" "$script" "$master_log" "$pid_file"
}

heartbeat() {
  local active_count
  active_count=$(pgrep -af 'run_wsdm_web_recsys.py|run_wsdm_index_sweep.py|run_tier_c_retrain_prediction.py' 2>/dev/null | wc -l || true)
  local gpu
  gpu=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ';' || true)
  log "heartbeat active_experiment_processes=$active_count gpu=$gpu"
}

main() {
  log "overnight watchdog start interval=${INTERVAL_SECONDS}s"
  log "watch_root=$WATCH_ROOT"

  while true; do
    heartbeat
    check_queue \
      "available-data WSDM queue" \
      "./run_wsdm_available_full_queue.sh" \
      "$AVAILABLE_ROOT/logs/master.pid" \
      "$AVAILABLE_ROOT/logs/master.log" \
      "queue done"
    check_queue \
      "Amazon2023 acquire/prepare/index queue" \
      "./run_amazon2023_acquire_prepare_index_queue.sh" \
      "$AMAZON2023_ROOT/logs/master.pid" \
      "$AMAZON2023_ROOT/logs/master.log" \
      "amazon2023 acquire/prepare/index queue done"
    check_queue \
      "Tier-C retrain-prediction queue" \
      "./run_tier_c_retrain_prediction_queue.sh" \
      "$TIER_C_ROOT/queue/logs/master.pid" \
      "$TIER_C_ROOT/queue/logs/master.log" \
      "Tier-C retrain-prediction queue done"
    check_queue \
      "Amazon2023 downstream follow-on queue" \
      "./run_local_gpu_amazon2023_downstream_full_followon_20260814.sh" \
      "$AMAZON2023_DOWNSTREAM_ROOT/followon_logs/master.pid" \
      "$AMAZON2023_DOWNSTREAM_ROOT/followon_logs/master.log" \
      "Amazon2023 downstream full follow-on done"
    sleep "$INTERVAL_SECONDS"
  done
}

main "$@"

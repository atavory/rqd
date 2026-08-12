#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/users/atavory/scratch/wsdm_experiments/rqd}
PY=${PY:-/data/users/atavory/scratch/wsdm_experiments/venv/bin/python}
CACHE=${CACHE:-/data/users/atavory/scratch/wsdm_experiments/cache/movielens64.npz}
OUT=${OUT:-/data/users/atavory/scratch/wsdm_experiments/results/phase0_ml1m_primary24_seeds34_20260812}
LOG="$OUT/logs"
CURRENT_LOG=${CURRENT_LOG:-/data/users/atavory/scratch/wsdm_experiments/results/phase0_ml1m_primary24_seeds12_20260812/logs}

mkdir -p "$LOG"

wait_for_pid_file() {
  local pid_file=$1
  while [[ -s "$pid_file" ]] && ps -p "$(cat "$pid_file")" >/dev/null 2>&1; do
    sleep 60
  done
}

launch_detached() {
  local name=$1
  local wait_pid_file=$2
  local command=$3
  local pid_file="$LOG/${name}.pid"
  local log_file="$LOG/${name}.log"

  if [[ -s "$pid_file" ]] && ps -p "$(cat "$pid_file")" >/dev/null 2>&1; then
    echo "$name already running as pid $(cat "$pid_file")"
    return
  fi

  : > "$log_file"
  (
    wait_for_pid_file "$wait_pid_file"
    cd "$ROOT"
    exec setsid bash -lc "$command"
  ) >> "$log_file" 2>&1 < /dev/null &
  echo $! > "$pid_file"
  echo "$name watcher launched as pid $(cat "$pid_file")"
}

run_one() {
  local arch=$1
  local seed=$2
  local gpu=$3
  local output="$OUT/downstream_movielens_${arch}_fd2_seed${seed}.json"
  printf "CUDA_VISIBLE_DEVICES=%s '%s' run_wsdm_web_recsys.py --dataset movielens --cache '%s' --arch %s --seed %s --freeze-depth 2 --epochs 50 --n-beams 10 --device cuda:0 --output '%s' --overwrite" \
    "$gpu" "$PY" "$CACHE" "$arch" "$seed" "$output"
}

launch_detached gpu0_funnel24_s3_s4_balanced24_s4 \
  "$CURRENT_LOG/gpu0_funnel24_s1_s2_balanced24_s2.pid" \
  "set -euo pipefail; $(run_one funnel24 3 0); $(run_one funnel24 4 0); $(run_one balanced24 4 0)"

launch_detached gpu1_balanced24_s3_uniform24_s3_s4 \
  "$CURRENT_LOG/gpu1_balanced24_s1_uniform24_s1_s2.pid" \
  "set -euo pipefail; $(run_one balanced24 3 1); $(run_one uniform24 3 1); $(run_one uniform24 4 1)"

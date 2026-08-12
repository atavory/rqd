#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/users/atavory/scratch/wsdm_experiments/rqd}
PY=${PY:-/data/users/atavory/scratch/wsdm_experiments/venv/bin/python}
CACHE=${CACHE:-/data/users/atavory/scratch/wsdm_experiments/cache/movielens64.npz}
OUT=${OUT:-/data/users/atavory/scratch/wsdm_experiments/results/phase0_ml1m_primary24_20260812}
LOG="$OUT/logs"

mkdir -p "$LOG"

launch_detached() {
  local name=$1
  local command=$2
  local pid_file="$LOG/${name}.pid"
  local log_file="$LOG/${name}.log"

  if [[ -s "$pid_file" ]] && ps -p "$(cat "$pid_file")" >/dev/null 2>&1; then
    echo "$name already running as pid $(cat "$pid_file")"
    return
  fi

  : > "$log_file"
  (
    cd "$ROOT"
    exec setsid bash -lc "$command" >> "$log_file" 2>&1 < /dev/null
  ) &
  echo $! > "$pid_file"
  echo "$name launched as pid $(cat "$pid_file")"
}

base_cmd() {
  local arch=$1
  local gpu=$2
  local output="$OUT/downstream_movielens_${arch}_fd2_seed0.json"
  printf "CUDA_VISIBLE_DEVICES=%s '%s' run_wsdm_web_recsys.py --dataset movielens --cache '%s' --arch %s --seed 0 --freeze-depth 2 --epochs 50 --n-beams 10 --device cuda:0 --output '%s' --overwrite" \
    "$gpu" "$PY" "$CACHE" "$arch" "$output"
}

launch_detached gpu0_funnel24_fd2_seed0 \
  "set -euo pipefail; $(base_cmd funnel24 0)"

launch_detached gpu1_balanced24_then_uniform24_fd2_seed0 \
  "set -euo pipefail; $(base_cmd balanced24 1); $(base_cmd uniform24 1)"

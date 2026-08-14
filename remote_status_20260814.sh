#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -z "${BUNDLE_ROOT:-}" ]]; then
  if [[ -d "$SCRIPT_DIR/rqd" && -d "$SCRIPT_DIR/cache" ]]; then
    BUNDLE_ROOT=$SCRIPT_DIR
  else
    BUNDLE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
  fi
fi
OUT_ROOT=${OUT_ROOT:-$BUNDLE_ROOT/results_remote_gpu_20260814}
PY=${PY:-python3}

echo "time=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "out_root=$OUT_ROOT"
echo

if [[ -d "$OUT_ROOT/logs" ]]; then
  echo "workers:"
  for pid_file in "$OUT_ROOT"/logs/*.pid; do
    [[ -e "$pid_file" ]] || continue
    pid=$(cat "$pid_file")
    if ps -p "$pid" >/dev/null 2>&1; then
      ps -p "$pid" -o pid,ppid,etime,stat,pcpu,pmem,cmd
    else
      echo "$(basename "$pid_file") dead pid=$pid"
    fi
  done
  echo
fi

"$PY" - "$OUT_ROOT" <<'PY'
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
pat = re.compile(r"downstream_(?P<cat>.+)_(?P<arch>[^_]+)_fd(?P<fd>\d+)_seed(?P<seed>\d+)\.json$")
complete = defaultdict(list)
partial = []
bad = []
for path in sorted(root.glob("downstream_*.json")):
    match = pat.match(path.name)
    key = match.group("cat"), match.group("arch"), match.group("fd") if match else ("unparsed", "", "")
    try:
        payload = json.loads(path.read_text())
        ok = isinstance(payload.get("strategies"), list) and len(payload["strategies"]) > 0
    except Exception as exc:
        bad.append((path.name, str(exc)))
        continue
    if ok:
        seed = match.group("seed") if match else "?"
        complete[key].append(seed)
    else:
        partial.append((path.name, path.stat().st_size))

print("complete downstream files:", sum(len(v) for v in complete.values()))
for key, seeds in sorted(complete.items()):
    print(key, sorted(seeds), "n=", len(seeds))
if partial:
    print("partial:", partial)
if bad:
    print("bad:", bad[:10])
PY

echo
if [[ -f "$OUT_ROOT/logs/gpu0.log" ]]; then
  echo "gpu0 tail:"
  tail -n 20 "$OUT_ROOT/logs/gpu0.log"
fi
if [[ -f "$OUT_ROOT/logs/gpu1.log" ]]; then
  echo
  echo "gpu1 tail:"
  tail -n 20 "$OUT_ROOT/logs/gpu1.log"
fi

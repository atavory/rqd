#!/usr/bin/env python3
"""Poll the WSDM Manifold messagebox and publish local status."""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


MSG_ROOT = os.environ.get(
    "MSG_ROOT", "aai_research_tlv/tree/atavory/wsdm_messagebox"
)
LOCAL_ROOT = Path(
    os.environ.get(
        "LOCAL_MESSAGEBOX",
        "/data/users/atavory/scratch/wsdm_experiments/messagebox",
    )
)
RESULTS_ROOT = Path(
    os.environ.get(
        "RESULTS_ROOT",
        "/data/users/atavory/scratch/wsdm_experiments/results",
    )
)
INTERVAL_SECONDS = int(os.environ.get("INTERVAL_SECONDS", "120"))
PREFIX = os.environ.get("AGENT_PREFIX", "cont_si")
RECIPIENTS = [
    name.strip()
    for name in os.environ.get("RECIPIENTS", "cont_si2,cont_si3").split(",")
    if name.strip()
]
INBOXES = [
    name.strip()
    for name in os.environ.get("INBOXES", "to_codex,to_cont_si").split(",")
    if name.strip()
]


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def now_human() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")


def run(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )


def shell(cmd: str, timeout: int = 120) -> str:
    proc = subprocess.run(
        ["bash", "-lc", cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
    return proc.stdout.strip()


def manifold_ls(path: str) -> list[str]:
    proc = run(["manifold", "ls", "-l", path], timeout=180)
    if proc.returncode != 0:
        return []
    names: list[str] = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if parts and parts[-1].endswith(".md"):
            names.append(parts[-1])
    return names


def manifold_get(remote: str, local: Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    proc = run(["manifold", "get", remote, str(local)], timeout=300)
    return proc.returncode == 0


def manifold_put(local: Path, remote: str) -> bool:
    proc = run(["manifold", "put", str(local), remote], timeout=300)
    return proc.returncode == 0


def load_seen() -> set[str]:
    path = LOCAL_ROOT / "seen_to_codex.json"
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text()))
    except Exception:
        return set()


def save_seen(seen: set[str]) -> None:
    path = LOCAL_ROOT / "seen_to_codex.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(seen), indent=2) + "\n")


def complete_json_count(root: Path, pattern: str, downstream: bool = False) -> tuple[int, list[str]]:
    complete = 0
    partial: list[str] = []
    for path in sorted(root.glob(pattern)):
        try:
            payload = json.loads(path.read_text())
            if downstream:
                ok = isinstance(payload.get("strategies"), list) and bool(payload["strategies"])
            else:
                runs = payload.get("runs") or []
                ok = sum(1 for row in runs if row.get("strategies")) >= 3
        except Exception:
            ok = False
        if ok:
            complete += 1
        else:
            partial.append(path.name)
    return complete, partial


def amazon2023_index_status() -> str:
    root = RESULTS_ROOT / "amazon2023_5core_index_20260812"
    labels: dict[str, list[int]] = {}
    partial: list[str] = []
    regex = re.compile(r"^index_(?P<label>.+)_funnel24_seed(?P<seed>\d+)\.json$")
    for path in sorted(root.glob("index_*.json")):
        match = regex.match(path.name)
        label = match.group("label") if match else path.stem
        seed = int(match.group("seed")) if match else -1
        try:
            payload = json.loads(path.read_text())
            runs = payload.get("runs") or []
            ok = sum(1 for row in runs if row.get("strategies")) >= 3
        except Exception:
            ok = False
        if ok:
            labels.setdefault(label, []).append(seed)
        else:
            partial.append(path.name)
    lines = [f"complete files: {sum(len(v) for v in labels.values())}"]
    for label, seeds in sorted(labels.items()):
        lines.append(f"- {label}: seeds {sorted(seeds)}")
    if partial:
        lines.append(f"partial: {partial}")
    return "\n".join(lines)


def downstream_status() -> str:
    root = RESULTS_ROOT / "amazon2023_downstream_rung_funnel24_20260814"
    complete, partial = complete_json_count(root, "downstream_*.json", downstream=True)
    return f"complete downstream files: {complete}\npartial: {partial}"


def build_status(new_messages: list[str]) -> str:
    gpu = shell(
        "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total "
        "--format=csv,noheader",
        timeout=30,
    )
    processes = shell(
        "ps -p 1244952,1760846,1961481 -o pid,ppid,etime,stat,pcpu,pmem,cmd 2>/dev/null; "
        "pgrep -af 'run_wsdm_index_sweep.py|run_wsdm_web_recsys.py|run_local_gpu_amazon2023_downstream_queue|run_tier_c_retrain_prediction_queue|run_amazon2023_acquire_prepare_index_queue' || true",
        timeout=30,
    )
    tier_c_predictions = shell(
        "ls -l /data/users/atavory/scratch/wsdm_experiments/results/tier_c_retrain_prediction_20260812/predictions 2>/dev/null || true",
        timeout=30,
    )
    return f"""{PREFIX}:

from: {PREFIX}
to: {", ".join(RECIPIENTS)}
time: {now_human()}
topic: local status heartbeat

## New Messages Received From Remote Agents

{chr(10).join('- ' + msg for msg in new_messages) if new_messages else 'None this interval.'}

## Local GPU

```text
{gpu}
```

## Local Processes

```text
{processes}
```

## Amazon2023 Index Queue

```text
{amazon2023_index_status()}
```

## Local Amazon2023 Downstream Rung Queue

```text
{downstream_status()}
```

## Tier-C Predictions

```text
{tier_c_predictions}
```

"""


def post_status(new_messages: list[str]) -> None:
    out_dir = LOCAL_ROOT / "outbox"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{now_stamp()}_from_{PREFIX}_local_status.md"
    path = out_dir / name
    path.write_text(build_status(new_messages))
    for recipient in RECIPIENTS:
        manifold_put(path, f"{MSG_ROOT}/to_{recipient}/{name}")


def poll_once() -> None:
    seen = load_seen()
    new_messages: list[str] = []
    for inbox_name in INBOXES:
        inbox = LOCAL_ROOT / inbox_name
        for name in manifold_ls(f"{MSG_ROOT}/{inbox_name}"):
            key = f"{inbox_name}/{name}"
            if key in seen:
                continue
            local = inbox / name
            if manifold_get(f"{MSG_ROOT}/{inbox_name}/{name}", local):
                seen.add(key)
                new_messages.append(key)
    save_seen(seen)
    post_status(new_messages)


def ensure_remote_dirs() -> None:
    for name in ["to_cont_si", "to_codex", *[f"to_{recipient}" for recipient in RECIPIENTS]]:
        run(["manifold", "mkdir", "-p", f"{MSG_ROOT}/{name}"], timeout=180)


def post_boot_message() -> None:
    out_dir = LOCAL_ROOT / "outbox"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{now_stamp()}_from_{PREFIX}_coordinator_boot.md"
    path = out_dir / name
    path.write_text(
        f"""{PREFIX}:

from: {PREFIX}
to: {", ".join(RECIPIENTS)}
time: {now_human()}
topic: coordinator boot

Coordinator loop is active.

- messagebox root: `{MSG_ROOT}/`
- recipients: `{", ".join(RECIPIENTS)}`
- polled inboxes: `{", ".join(INBOXES)}`
- interval seconds: `{INTERVAL_SECONDS}`

Reply to either `{MSG_ROOT}/to_cont_si/` or legacy `{MSG_ROOT}/to_codex/`.
"""
    )
    for recipient in RECIPIENTS:
        manifold_put(path, f"{MSG_ROOT}/to_{recipient}/{name}")


def main() -> None:
    log = LOCAL_ROOT / "coordinator.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    ensure_remote_dirs()
    post_boot_message()
    while not (LOCAL_ROOT / "STOP").exists():
        try:
            poll_once()
            log.write_text(f"{now_human()} ok\n")
        except Exception as exc:  # keep the coordinator alive.
            log.write_text(f"{now_human()} error: {exc}\n")
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()

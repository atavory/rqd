#!/usr/bin/env python3
"""Coordinate WSDM paper-result completion across local and remote workers."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path("/data/users/atavory/scratch/wsdm_experiments")
RQD = ROOT / "rqd"
RESULTS = ROOT / "results"
DOWNSTREAM = RESULTS / "amazon2023_downstream_rung_funnel24_20260814"
MESSAGEBOX = ROOT / "messagebox"
LOG_DIR = RESULTS / "paper_completion_loop_20260815"
ANALYSIS_DIR = ROOT / "overleaf_data" / "wsdm_analysis_latest"
DATA_OVERLEAF = Path("/home/atavory/fbsource/data_overleaf")
PAPER_OVERLEAF = Path("/home/atavory/fbsource/overleaf")
DATA_ANALYSIS_REL = Path("results/wsdm_2027_paper_analysis")
PAPER_GENERATED_REL = Path("generated")
MANIFOLD_ROOT = "aai_research_tlv/tree/atavory"
MSG_ROOT = f"{MANIFOLD_ROOT}/wsdm_messagebox"
ANALYSIS_INPUT_ROOT = f"{MANIFOLD_ROOT}/wsdm_analysis_inputs"
REMOTE_ARTIFACT_DIRS = [
    f"{MANIFOLD_ROOT}/wsdm_remote_results/cont_si2",
    f"{MANIFOLD_ROOT}/wsdm_remote_results/cont_si3",
    f"{ANALYSIS_INPUT_ROOT}/tier_c_synthetic_v2",
]

EXPECTED_STRATEGIES = {
    "frozen",
    "stratified",
    "warm_start_full_old_generator",
    "ema_streaming_vq_old_generator",
    "full_old_generator",
    "full_old_generator_centroid_relabel",
    "full_old_generator_assignment_relabel",
    "grm_only_retrained_generator",
    "full_retrained_generator",
}


def stamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def human() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{human()}] {message}"
    print(line, flush=True)
    with (LOG_DIR / "loop.log").open("a") as handle:
        handle.write(line + "\n")


def run(cmd: list[str], timeout: int = 300, cwd: Path = RQD) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )


def run_mrgitties(cmd: str, cwd: Path, timeout: int = 600) -> dict:
    sock = Path.home() / ".mrgitties" / "mrgitties.sock"
    if not sock.exists():
        return {
            "ok": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": f"mrgitties socket missing: {sock}",
        }
    request = {
        "cmd": cmd,
        "cwd": str(cwd),
        "timeout": timeout,
    }
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(str(sock))
    client.sendall(json.dumps(request).encode("utf-8") + b"\n")
    client.shutdown(socket.SHUT_WR)
    chunks = []
    while True:
        chunk = client.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
    client.close()
    return json.loads(b"".join(chunks).decode("utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def downstream_rows() -> list[dict]:
    rows = []
    for path in sorted(DOWNSTREAM.glob("downstream_*.json")):
        try:
            payload = read_json(path)
            strategies = payload.get("strategies") or []
            names = {row.get("strategy") for row in strategies}
            config = payload.get("configuration", {})
            dataset = payload.get("dataset", {})
            rows.append({
                "path": path,
                "artifact": path.name,
                "dataset": dataset.get("dataset_variant") or dataset.get("dataset", ""),
                "freeze_depth": config.get("freeze_depth", ""),
                "seed": config.get("seed", ""),
                "strategies": len(strategies),
                "complete": EXPECTED_STRATEGIES <= names,
                "missing": sorted(EXPECTED_STRATEGIES - names),
            })
        except Exception as exc:
            rows.append({
                "path": path,
                "artifact": path.name,
                "dataset": "",
                "freeze_depth": "",
                "seed": "",
                "strategies": 0,
                "complete": False,
                "missing": [f"unreadable:{exc}"],
            })
    return rows


def completion_signature() -> str:
    rows = downstream_rows()
    analysis_script = RQD / "make_wsdm_overleaf_analysis.py"
    payload = {
        "analysis_script_sha256": (
            sha256(analysis_script) if analysis_script.exists() else ""
        ),
        "complete": sorted(row["artifact"] for row in rows if row["complete"]),
        "partial": {
            row["artifact"]: row["strategies"]
            for row in rows
            if not row["complete"]
        },
        "remote_artifacts": remote_artifact_refs(),
    }
    return json.dumps(payload, sort_keys=True)


def write_status_json() -> dict:
    rows = downstream_rows()
    artifact_refs = remote_artifact_refs()
    status = {
        "time": human(),
        "complete_count": sum(1 for row in rows if row["complete"]),
        "partial_count": sum(1 for row in rows if not row["complete"]),
        "remote_artifacts": artifact_refs,
        "rows": [
            {
                "artifact": row["artifact"],
                "dataset": row["dataset"],
                "freeze_depth": row["freeze_depth"],
                "seed": row["seed"],
                "strategies": row["strategies"],
                "complete": row["complete"],
                "missing": row["missing"],
            }
            for row in rows
        ],
    }
    path = LOG_DIR / "downstream_status.json"
    path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    return status


def manifold_names(path: str) -> list[str]:
    proc = run(["manifold", "ls", path], timeout=180)
    if proc.returncode != 0:
        log(f"manifold ls failed rc={proc.returncode} path={path}: {proc.stdout.strip()}")
        return []
    names = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        name = parts[-1]
        if name.startswith("I") or name == "DIR":
            continue
        names.append(name)
    return names


def remote_artifact_refs() -> list[str]:
    refs = []
    for remote_dir in REMOTE_ARTIFACT_DIRS:
        for name in manifold_names(remote_dir):
            if name.endswith(".sha256"):
                continue
            if not (
                name.endswith(".tar.zst")
                or name.endswith(".csv")
                or name.endswith(".json")
            ):
                continue
            refs.append(f"manifold:{remote_dir}/{name}")
    return sorted(set(refs))


def send_work_orders() -> None:
    text = f"""cont_si:

from: cont_si
to: cont_si2, cont_si3
time: {human()} PDT
topic: paper completion loop work order

We need complete, script-consumable result artifacts. Prose-only reports are not enough.

cont_si2:
- Finish/upload the BTT fallback matrix you own.
- Prioritize complete 9-strategy downstream JSONs. Partial JSONs do not count.
- Upload fresh tarball + sha256 under:
  {MANIFOLD_ROOT}/wsdm_remote_results/cont_si2/
- Reply under {MSG_ROOT}/to_cont_si/ with complete rows, partial rows,
  active PIDs, tarball path, checksum path, and digest.

cont_si3:
- Continue LC-Rec/Reformer to final evaluator output.
- Use spare capacity on non-duplicating fallback rows: Tools/Toys before Beauty.
- Upload fresh tarball + sha256 under:
  {MANIFOLD_ROOT}/wsdm_remote_results/cont_si3/
- Reply under {MSG_ROOT}/to_cont_si/ with LC-Rec/Reformer status, fallback
  complete rows, active PIDs, tarball path, checksum path, and digest.

Both:
- A downstream JSON is complete only with these 9 strategies:
  {", ".join(sorted(EXPECTED_STRATEGIES))}
- Use scripts to summarize. No manual-only result claims.
"""
    local = LOG_DIR / f"{stamp()}_from_cont_si_paper_completion_work_order.md"
    local.write_text(text)
    for recipient in ("cont_si2", "cont_si3"):
        remote = f"{MSG_ROOT}/to_{recipient}/{local.name}"
        proc = run(["manifold", "put", str(local), remote], timeout=300)
        log(f"work order to {recipient}: rc={proc.returncode} remote={remote}")
        if proc.returncode != 0:
            log(proc.stdout.strip())


def send_required_status(reason: str) -> None:
    rows = downstream_rows()
    partials = [
        f"- {row['artifact']}: {row['strategies']}/9, missing={','.join(row['missing'])}"
        for row in rows
        if not row["complete"]
    ]
    if not partials:
        partials = ["- none"]
    text = f"""cont_si:

from: cont_si
to: cont_si2, cont_si3
time: {human()} PDT
topic: required remote status report
reason: {reason}

This is a direct mailbox request. Reply under:
  {MSG_ROOT}/to_cont_si/

Required in the reply:
- active PIDs and exact commands
- GPU utilization and memory
- newest completed result files, with strategy counts where applicable
- newest partial files and mtimes
- current artifact tarballs/checksums uploaded under wsdm_remote_results/
- exact blocker if there is no progress
- next row/job planned

Current cont_si local state:
- complete downstream rows: {sum(1 for row in rows if row["complete"])}
- partial downstream rows: {sum(1 for row in rows if not row["complete"])}
{chr(10).join(partials)}

cont_si2:
- Report current Beauty/Tools/Toys fallback state.
- Upload a fresh snapshot if any rows changed since the last cont_si2 tarball.

cont_si3:
- Report current DACT/LC-Rec/Reformer state.
- Report current BTT fallback queue state.
- Upload a fresh artifact snapshot if DACT logs/checkpoints, LC-Rec eval/checkpoints,
  or fallback results changed since the last cont_si3 tarballs.

Text-only claims are not enough for final table readiness; changed artifacts must be
uploaded and named with checksums.
"""
    local = LOG_DIR / f"{stamp()}_from_cont_si_required_remote_status.md"
    local.write_text(text)
    for recipient in ("cont_si2", "cont_si3"):
        remote = f"{MSG_ROOT}/to_{recipient}/{local.name}"
        proc = run(["manifold", "put", str(local), remote], timeout=300)
        log(f"required status to {recipient}: rc={proc.returncode} remote={remote}")
        if proc.returncode != 0:
            log(proc.stdout.strip())


def load_seen() -> set[str]:
    path = LOG_DIR / "seen_to_cont_si.json"
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text()))
    except Exception:
        return set()


def save_seen(seen: set[str]) -> None:
    (LOG_DIR / "seen_to_cont_si.json").write_text(
        json.dumps(sorted(seen), indent=2) + "\n"
    )


def poll_remote_replies() -> list[str]:
    proc = run(["manifold", "ls", f"{MSG_ROOT}/to_cont_si/"], timeout=180)
    if proc.returncode != 0:
        log(f"poll replies failed rc={proc.returncode}: {proc.stdout.strip()}")
        return []
    seen = load_seen()
    new_names = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        name = parts[-1]
        if not name.endswith(".md") or name in seen:
            continue
        local = MESSAGEBOX / "to_cont_si" / name
        local.parent.mkdir(parents=True, exist_ok=True)
        if local.exists():
            seen.add(name)
            continue
        get = run(
            ["manifold", "get", f"{MSG_ROOT}/to_cont_si/{name}", str(local)],
            timeout=300,
        )
        if get.returncode == 0:
            seen.add(name)
            new_names.append(name)
        else:
            log(f"download reply failed name={name} rc={get.returncode}: {get.stdout.strip()}")
    save_seen(seen)
    return new_names


def regenerate_analysis() -> bool:
    cmd = [
        "./make_wsdm_overleaf_analysis.py",
        "--experiment-root",
        str(ROOT),
        "--output-dir",
        "overleaf_data/wsdm_analysis_latest",
        "--hash-inputs",
    ]
    artifact_refs = remote_artifact_refs()
    artifact_refs.append(
        "manifold:aai_research_tlv/tree/atavory/wsdm_results_snapshot_20260814_210842.tar.zst#ed6eb768a97dfa6ecc60d5dd7d2cfcd12e2c2fff4aad7df18f079e926cd2c7e3"
    )
    for ref in sorted(set(artifact_refs)):
        cmd.extend(["--artifact-ref", ref])
    proc = run(cmd, timeout=900)
    log(f"regenerate analysis rc={proc.returncode}")
    if proc.stdout.strip():
        log(proc.stdout.strip())
    return proc.returncode == 0


def make_analysis_tarball() -> tuple[Path, str]:
    name = f"wsdm_overleaf_analysis_auto_{stamp()}.tar.zst"
    path = ROOT / name
    proc = run(
        ["tar", "--zstd", "-cf", str(path), "-C", str(ROOT / "overleaf_data"), "wsdm_analysis_latest"],
        timeout=300,
        cwd=ROOT,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout)
    digest = sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n")
    return path, digest


def upload_file(local: Path, remote: str) -> bool:
    proc = run(["manifold", "put", str(local), remote], timeout=300)
    log(f"upload rc={proc.returncode} remote={remote}")
    if proc.returncode != 0:
        log(proc.stdout.strip())
    return proc.returncode == 0


def upload_analysis_snapshot() -> None:
    path, digest = make_analysis_tarball()
    checksum = path.with_suffix(path.suffix + ".sha256")
    remote = f"{MANIFOLD_ROOT}/{path.name}"
    upload_file(path, remote)
    upload_file(checksum, f"{MANIFOLD_ROOT}/{checksum.name}")
    verify = run(["manifold", "ls", MANIFOLD_ROOT + "/"], timeout=180)
    ok = path.name in verify.stdout and checksum.name in verify.stdout
    log(f"analysis upload verified={ok} path={remote} sha256={digest}")


def push_repo(repo: Path) -> None:
    push = run(["git", "push", "origin", "master"], timeout=600, cwd=repo)
    if push.returncode == 0:
        log(f"overleaf push rc=0 repo={repo}")
        return
    log(f"direct overleaf push failed repo={repo} rc={push.returncode}: {push.stdout.strip()}")
    fallback = run_mrgitties("git push origin master", repo, timeout=600)
    log(
        f"mrgitties overleaf push repo={repo} ok={fallback.get('ok')} "
        f"rc={fallback.get('exit_code')}"
    )
    output = (fallback.get("stdout") or "") + (fallback.get("stderr") or "")
    if output.strip():
        log(output.strip())


def commit_and_push_repo(repo: Path, paths: list[str], message: str) -> None:
    add = run(["git", "add", *paths], timeout=120, cwd=repo)
    if add.returncode != 0:
        log(f"git add failed repo={repo} rc={add.returncode}: {add.stdout.strip()}")
        return
    diff = run(["git", "diff", "--cached", "--quiet"], timeout=120, cwd=repo)
    if diff.returncode == 0:
        log(f"no overleaf changes to commit repo={repo}")
        return
    if diff.returncode != 1:
        log(f"git diff --cached failed repo={repo} rc={diff.returncode}: {diff.stdout.strip()}")
        return
    commit = run(["git", "commit", "-m", message], timeout=180, cwd=repo)
    log(f"overleaf commit repo={repo} rc={commit.returncode}")
    if commit.stdout.strip():
        log(commit.stdout.strip())
    if commit.returncode == 0:
        push_repo(repo)


def sync_data_overleaf() -> None:
    if not DATA_OVERLEAF.exists():
        log(f"data overleaf missing: {DATA_OVERLEAF}")
        return
    target = DATA_OVERLEAF / DATA_ANALYSIS_REL
    target.parent.mkdir(parents=True, exist_ok=True)
    rsync = run(
        ["rsync", "-a", "--delete", f"{ANALYSIS_DIR}/", f"{target}/"],
        timeout=300,
        cwd=ROOT,
    )
    log(f"data overleaf rsync rc={rsync.returncode} target={target}")
    if rsync.returncode != 0:
        log(rsync.stdout.strip())
        return
    commit_and_push_repo(
        DATA_OVERLEAF,
        [str(DATA_ANALYSIS_REL)],
        "Update scripted WSDM paper analysis snapshot",
    )


def sync_paper_overleaf() -> None:
    if not PAPER_OVERLEAF.exists():
        log(f"paper overleaf missing: {PAPER_OVERLEAF}")
        return
    target = PAPER_OVERLEAF / PAPER_GENERATED_REL
    target.mkdir(parents=True, exist_ok=True)
    copies = {
        "README.md": "README.md",
        "tables/abstract_downstream_table.tex": "abstract_downstream_table.tex",
        "tables/abstract_readiness_table.tex": "abstract_readiness_table.tex",
        "tables/tier_c_summary_table.tex": "tier_c_summary_table.tex",
        "figures/downstream_ndcg10_pgfplots.tex": "downstream_ndcg10_pgfplots.tex",
        "figures/tier_c_real_actions_pgfplots.tex": "tier_c_real_actions_pgfplots.tex",
    }
    for source_rel, dest_name in copies.items():
        shutil.copy2(ANALYSIS_DIR / source_rel, target / dest_name)
    log(f"paper overleaf generated snippets synced target={target}")
    commit_and_push_repo(
        PAPER_OVERLEAF,
        [str(PAPER_GENERATED_REL)],
        "Update generated WSDM analysis snippets",
    )


def sync_overleaf_repos() -> None:
    sync_data_overleaf()
    sync_paper_overleaf()


def loop_once(force_upload: bool = False) -> str:
    status = write_status_json()
    log(
        f"downstream complete={status['complete_count']} partial={status['partial_count']} "
        + ", ".join(
            f"{row['artifact']}:{row['strategies']}/9"
            for row in status["rows"]
            if not row["complete"]
        )
    )
    new_replies = poll_remote_replies()
    if new_replies:
        log(f"new remote replies: {', '.join(new_replies)}")
    signature = completion_signature()
    last_path = LOG_DIR / "last_signature.txt"
    previous = last_path.read_text() if last_path.exists() else ""
    changed = signature != previous
    if changed or force_upload:
        if regenerate_analysis():
            upload_analysis_snapshot()
            sync_overleaf_repos()
        last_path.write_text(signature)
    else:
        log("no completion-signature change; analysis upload skipped")
    return signature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-seconds", type=int, default=600)
    parser.add_argument("--max-iterations", type=int, default=0)
    parser.add_argument("--send-work-orders", action="store_true")
    parser.add_argument("--work-order-every-iterations", type=int, default=0)
    parser.add_argument("--required-status-every-iterations", type=int, default=0)
    parser.add_argument("--force-upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if args.send_work_orders:
        send_work_orders()
    iteration = 0
    while True:
        iteration += 1
        log(f"iteration={iteration} start")
        if (
            args.work_order_every_iterations
            and iteration > 1
            and iteration % args.work_order_every_iterations == 0
        ):
            send_work_orders()
        if (
            args.required_status_every_iterations
            and iteration > 1
            and iteration % args.required_status_every_iterations == 0
        ):
            send_required_status(
                f"scheduled every {args.required_status_every_iterations} iterations"
            )
        try:
            loop_once(force_upload=args.force_upload and iteration == 1)
        except Exception as exc:
            log(f"iteration={iteration} error: {exc}")
        if args.max_iterations and iteration >= args.max_iterations:
            log("max iterations reached")
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()

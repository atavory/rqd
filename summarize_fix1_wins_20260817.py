#!/usr/bin/env python3
"""Write a compact leaderboard for FIX-1 target-split wins."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rows_from_payload(path: Path):
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return []
    config = payload.get("configuration", {})
    dataset = payload.get("dataset", {})
    dataset_name = dataset.get("dataset", path.parent.name)
    common = {
        "file": str(path),
        "stem": path.stem,
        "dataset": dataset_name,
        "arch": config.get("arch", ""),
        "freeze_depth": config.get("freeze_depth", ""),
        "seed": config.get("seed", ""),
    }
    rows = []
    for kind, key in (
        ("e2e", "target_item_split_rows"),
        ("reranker", "context_reranker_target_item_split_rows"),
    ):
        for row in payload.get(key, []):
            out = dict(common)
            out.update(row)
            out["kind"] = kind
            rows.append(out)
    return rows


def _fmt_float(value, digits=4):
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}g}"
    except Exception:
        return str(value)


def _pct(delta, base):
    if base is None or abs(base) < 1e-12:
        return None
    return 100.0 * delta / abs(base)


def _comparison(row, frozen, strategy=None):
    frozen_ndcg = float(frozen.get("ndcg_at_10", 0.0))
    frozen_recall = float(frozen.get("recall_at_10", 0.0))
    ndcg = float(row.get("ndcg_at_10", 0.0))
    recall = float(row.get("recall_at_10", 0.0))
    return {
        "kind": row.get("kind"),
        "stem": row.get("stem"),
        "strategy": strategy or row.get("strategy"),
        "split": row.get("target_item_split"),
        "n": row.get("target_item_split_n_eval"),
        "ndcg": ndcg,
        "recall": recall,
        "frozen_ndcg": frozen_ndcg,
        "frozen_recall": frozen_recall,
        "delta_ndcg": ndcg - frozen_ndcg,
        "delta_recall": recall - frozen_recall,
        "pct_ndcg": _pct(ndcg - frozen_ndcg, frozen_ndcg),
        "pct_recall": _pct(recall - frozen_recall, frozen_recall),
        "coverage": row.get("routing_coverage"),
        "churn": row.get("prefix_churn_headline"),
        "file": row.get("file"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.results_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for result_dir in [
        "fix1_target_split_20260817",
        "fix1_target_split_confirm_20260817",
        "fix1_target_split_extra_catalogs_20260817",
        "fix1_target_split_arch_sweep_20260817",
        "fix1_target_split_scout_20260817",
        "fix1_graft_fast_20260817",
        "context_reranker_fix1_target_split_20260817",
        "context_reranker_fix1_target_split_finetuned_20260817",
        "context_reranker_fix1_target_split_extra_catalogs_20260817",
    ]:
        path = root / result_dir
        if not path.exists():
            continue
        for json_path in sorted(path.glob("*.json")):
            all_rows.extend(_rows_from_payload(json_path))

    groups = {}
    for row in all_rows:
        key = (
            row.get("kind"),
            row.get("stem"),
            row.get("target_item_split"),
        )
        groups.setdefault(key, []).append(row)

    wins = []
    for key, rows in groups.items():
        frozen = next((row for row in rows if row.get("strategy") == "frozen"), None)
        if frozen is None:
            continue
        for row in rows:
            if row.get("strategy") == "frozen":
                continue
            wins.append(_comparison(row, frozen))

        stratified = next(
            (row for row in rows if row.get("strategy") == "stratified"),
            None,
        )
        assignment = next(
            (
                row for row in rows
                if str(row.get("strategy", "")).endswith("assignment_relabel")
            ),
            None,
        )
        for threshold, label in (
            (0.001, "policy_churn_gate_0p1pct"),
            (0.01, "policy_churn_gate_1pct"),
        ):
            chosen = stratified
            if assignment is not None:
                churn = float(assignment.get("prefix_churn_headline", 1.0))
                if churn <= threshold:
                    chosen = assignment
            if chosen is not None:
                strategy = f"{label}_choose_{chosen.get('strategy')}"
                wins.append(_comparison(chosen, frozen, strategy=strategy))

    positive = [
        row for row in wins
        if row["delta_ndcg"] > 0 or row["delta_recall"] > 0
    ]
    wins.sort(
        key=lambda row: (
            row["delta_ndcg"],
            row["delta_recall"],
            row["ndcg"],
            row["recall"],
        ),
        reverse=True,
    )

    lines = [
        "# FIX-1 Win Leaderboard",
        "",
        f"rows_scanned: {len(all_rows)}",
        f"comparisons: {len(wins)}",
        f"positive_comparisons: {len(positive)}",
        "",
        "| kind | config | strategy | split | n | ndcg@10 | vs frozen | recall@10 | vs frozen | coverage | churn |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in wins[:60]:
        pct_ndcg = (
            "" if row["pct_ndcg"] is None else f"{row['pct_ndcg']:+.1f}%"
        )
        pct_recall = (
            "" if row["pct_recall"] is None else f"{row['pct_recall']:+.1f}%"
        )
        lines.append(
            "| "
            + " | ".join([
                str(row["kind"]),
                str(row["stem"]),
                str(row["strategy"]),
                str(row["split"]),
                str(row["n"]),
                _fmt_float(row["ndcg"]),
                pct_ndcg,
                _fmt_float(row["recall"]),
                pct_recall,
                _fmt_float(row["coverage"]),
                _fmt_float(row["churn"]),
            ])
            + " |"
        )
    lines.append("")
    output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

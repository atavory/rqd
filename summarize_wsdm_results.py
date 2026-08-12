#!/usr/bin/env python3
"""Write audit-friendly CSV summaries from WSDM experiment JSON artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


DOWNSTREAM_FIELDS = [
    "artifact",
    "dataset",
    "arch",
    "codebook_sizes",
    "total_bits",
    "freeze_depth",
    "seed",
    "strategy",
    "consumer_retrained",
    "consumer_token_relabeling",
    "hit_rate_at_5",
    "hit_rate_at_10",
    "hit_rate_at_20",
    "hit_rate_at_50",
    "hit_rate_at_200",
    "ndcg_at_5",
    "ndcg_at_10",
    "ndcg_at_20",
    "ndcg_at_50",
    "ndcg_at_200",
    "recall_at_5",
    "recall_at_10",
    "recall_at_20",
    "recall_at_50",
    "recall_at_200",
    "routing_coverage",
    "candidate_count_mean",
    "query_milliseconds",
    "mse",
    "prefix_churn_headline",
    "prefix_churn_raw",
    "prefix_churn_centroid_aligned",
    "prefix_churn_assignment_aligned",
    "items_reindexed_headline",
    "items_reindexed",
    "items_reindexed_centroid_aligned",
    "items_reindexed_assignment_aligned",
    "codebook_update_bytes",
    "token_relabeling_bytes",
    "evaluation_seconds",
]

INDEX_RUN_FIELDS = [
    "artifact",
    "dataset",
    "arch",
    "codebook_sizes",
    "total_bits",
    "freeze_depth",
    "seed",
    "strategy",
    "mse",
    "normalized_mse",
    "prefix_churn_headline",
    "prefix_churn_raw",
    "prefix_churn_centroid_aligned",
    "prefix_churn_assignment_aligned",
    "items_reindexed_headline",
    "items_reindexed",
    "items_reindexed_centroid_aligned",
    "items_reindexed_assignment_aligned",
    "codebook_update_bytes",
    "occupied_prefixes",
    "possible_prefixes",
    "prefix_occupancy_fraction",
    "items_per_prefix_mean",
    "items_per_prefix_p50",
    "items_per_prefix_p95",
    "items_per_prefix_max",
    "prefix_entropy_bits",
    "effective_prefixes",
    "stratified_gap_recovery",
    "warm_start_full_gap_recovery",
    "ema_gap_recovery",
]

INDEX_SUMMARY_FIELDS = [
    "dataset",
    "arch",
    "codebook_sizes",
    "total_bits",
    "freeze_depth",
    "strategy",
    "n",
    "mse_mean",
    "mse_min",
    "mse_max",
    "prefix_churn_headline_mean",
    "prefix_churn_raw_mean",
    "prefix_churn_assignment_aligned_mean",
    "items_reindexed_headline_mean",
    "codebook_update_bytes_mean",
    "stratified_gap_recovery_mean",
    "warm_start_full_gap_recovery_mean",
    "ema_gap_recovery_mean",
]


def _read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _dataset_name(payload: dict) -> str:
    dataset = payload.get("dataset", {})
    if isinstance(dataset, dict):
        return dataset.get("dataset_variant") or dataset.get("dataset", "")
    return ""


def _codebook_sizes(configuration: dict) -> str:
    sizes = configuration.get("codebook_sizes") or configuration.get(
        "codes_per_stage", []
    )
    return "x".join(str(value) for value in sizes)


def _headline_churn(row: dict) -> float:
    return row.get(
        "prefix_churn_headline",
        row.get("prefix_churn_assignment_aligned", row.get("prefix_churn", 0.0)),
    )


def _headline_items(row: dict) -> int:
    return row.get(
        "items_reindexed_headline",
        row.get("items_reindexed_assignment_aligned", row.get("items_reindexed", 0)),
    )


def _base_fields(path: Path, payload: dict) -> dict:
    configuration = payload.get("configuration", {})
    return {
        "artifact": path.name,
        "dataset": _dataset_name(payload),
        "arch": configuration.get("arch", ""),
        "codebook_sizes": _codebook_sizes(configuration),
        "total_bits": configuration.get("total_bits", ""),
        "seed": configuration.get("seed", ""),
    }


def _downstream_rows(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("downstream_*.json")):
        payload = _read_json(path)
        base = _base_fields(path, payload)
        base["freeze_depth"] = payload.get("configuration", {}).get(
            "freeze_depth", ""
        )
        for strategy in payload.get("strategies", []):
            row = dict(base)
            row.update(strategy)
            row["prefix_churn_headline"] = _headline_churn(strategy)
            row["items_reindexed_headline"] = _headline_items(strategy)
            rows.append(row)
    return rows


def _index_rows(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("index_*.json")):
        payload = _read_json(path)
        base = _base_fields(path, payload)
        for run in payload.get("runs", []):
            run_base = dict(base)
            run_base.update({
                "freeze_depth": run.get("freeze_depth", ""),
                "stratified_gap_recovery": run.get(
                    "stratified_gap_recovery", ""
                ),
                "warm_start_full_gap_recovery": run.get(
                    "warm_start_full_gap_recovery", ""
                ),
                "ema_gap_recovery": run.get("ema_gap_recovery", ""),
            })
            for strategy in run.get("strategies", []):
                row = dict(run_base)
                row.update(strategy)
                row["prefix_churn_headline"] = _headline_churn(strategy)
                row["items_reindexed_headline"] = _headline_items(strategy)
                rows.append(row)
    return rows


def _mean(values: list[float]) -> float | str:
    values = [value for value in values if value != ""]
    if not values:
        return ""
    return sum(values) / len(values)


def _index_summary(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        key = (
            row.get("dataset", ""),
            row.get("arch", ""),
            row.get("codebook_sizes", ""),
            row.get("total_bits", ""),
            row.get("freeze_depth", ""),
            row.get("strategy", ""),
        )
        groups[key].append(row)

    summary = []
    for key, group in sorted(groups.items()):
        mses = [float(row["mse"]) for row in group if row.get("mse") != ""]
        out = {
            "dataset": key[0],
            "arch": key[1],
            "codebook_sizes": key[2],
            "total_bits": key[3],
            "freeze_depth": key[4],
            "strategy": key[5],
            "n": len(group),
            "mse_mean": _mean(mses),
            "mse_min": min(mses) if mses else "",
            "mse_max": max(mses) if mses else "",
            "prefix_churn_headline_mean": _mean([
                float(row["prefix_churn_headline"]) for row in group
                if row.get("prefix_churn_headline") != ""
            ]),
            "prefix_churn_raw_mean": _mean([
                float(row["prefix_churn_raw"]) for row in group
                if row.get("prefix_churn_raw") != ""
            ]),
            "prefix_churn_assignment_aligned_mean": _mean([
                float(row["prefix_churn_assignment_aligned"]) for row in group
                if row.get("prefix_churn_assignment_aligned") != ""
            ]),
            "items_reindexed_headline_mean": _mean([
                float(row["items_reindexed_headline"]) for row in group
                if row.get("items_reindexed_headline") != ""
            ]),
            "codebook_update_bytes_mean": _mean([
                float(row["codebook_update_bytes"]) for row in group
                if row.get("codebook_update_bytes") != ""
            ]),
            "stratified_gap_recovery_mean": _mean([
                float(row["stratified_gap_recovery"]) for row in group
                if row.get("stratified_gap_recovery") != ""
            ]),
            "warm_start_full_gap_recovery_mean": _mean([
                float(row["warm_start_full_gap_recovery"]) for row in group
                if row.get("warm_start_full_gap_recovery") != ""
            ]),
            "ema_gap_recovery_mean": _mean([
                float(row["ema_gap_recovery"]) for row in group
                if row.get("ema_gap_recovery") != ""
            ]),
        }
        summary.append(out)
    return summary


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    downstream = _downstream_rows(results_dir)
    index_rows = _index_rows(results_dir)
    index_summary = _index_summary(index_rows)

    _write_csv(output_dir / "downstream_rows.csv", DOWNSTREAM_FIELDS, downstream)
    _write_csv(output_dir / "index_runs.csv", INDEX_RUN_FIELDS, index_rows)
    _write_csv(
        output_dir / "index_summary.csv", INDEX_SUMMARY_FIELDS, index_summary,
    )
    print(
        f"wrote {len(downstream)} downstream rows, "
        f"{len(index_rows)} index rows, {len(index_summary)} index summaries "
        f"to {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()

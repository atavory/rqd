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
    "consumer_retrain_required",
    "consumer_token_relabeling",
    "id_migration_required",
    "serving_index_rebuild_required",
    "tokenizer_update_seconds",
    "consumer_retrain_seconds",
    "consumer_training_sequences",
    "update_wall_seconds",
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

BOUNDED_CANDIDATE_FIELDS = [
    "artifact",
    "dataset",
    "arch",
    "codebook_sizes",
    "total_bits",
    "freeze_depth",
    "seed",
    "strategy",
    "n_eval",
    "n_beams",
    "candidate_budget",
    "candidate_budget_mode",
    "candidate_budget_exact_scan_simulation",
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
    "uncapped_routing_coverage",
    "candidate_count_mean",
    "candidate_count_p50",
    "candidate_count_p95",
    "uncapped_candidate_count_mean",
    "uncapped_candidate_count_p50",
    "uncapped_candidate_count_p95",
    "items_returned_mean",
    "items_returned_p50",
    "items_returned_p95",
    "items_accessed_mean",
    "items_accessed_p50",
    "items_accessed_p95",
    "candidate_pool_truncated_fraction",
    "query_milliseconds",
    "evaluation_seconds",
]

CANDIDATE_GRID_FIELDS = BOUNDED_CANDIDATE_FIELDS + [
    "candidate_grid",
]

CONTEXT_RERANKER_FIELDS = [
    "artifact",
    "dataset",
    "arch",
    "codebook_sizes",
    "freeze_depth",
    "seed",
    "strategy",
    "scorer",
    "n_eval",
    "n_beams",
    "candidate_budget",
    "candidate_budget_mode",
    "candidate_budget_exact_scan_simulation",
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
    "uncapped_routing_coverage",
    "candidate_count_mean",
    "candidate_count_p50",
    "candidate_count_p95",
    "uncapped_candidate_count_mean",
    "uncapped_candidate_count_p50",
    "uncapped_candidate_count_p95",
    "items_returned_mean",
    "items_returned_p50",
    "items_returned_p95",
    "items_accessed_mean",
    "items_accessed_p50",
    "items_accessed_p95",
    "candidate_pool_truncated_fraction",
    "query_milliseconds",
    "evaluation_seconds",
    "mse",
    "prefix_churn_headline",
    "prefix_churn_raw",
    "prefix_churn_centroid_aligned",
    "prefix_churn_assignment_aligned",
    "items_reindexed_headline",
    "codebook_update_bytes",
    "tokenizer_update_seconds",
    "update_wall_seconds",
]

CONTEXT_RERANKER_GRID_FIELDS = CONTEXT_RERANKER_FIELDS + [
    "candidate_grid",
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
    "consumer_retrain_required",
    "id_migration_required",
    "serving_index_rebuild_required",
    "tokenizer_update_seconds",
    "consumer_retrain_seconds",
    "consumer_training_sequences",
    "update_wall_seconds",
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
    "tokenizer_update_seconds_mean",
    "consumer_retrain_seconds_mean",
    "update_wall_seconds_mean",
    "stratified_gap_recovery_mean",
    "warm_start_full_gap_recovery_mean",
    "ema_gap_recovery_mean",
]

COST_SUMMARY_FIELDS = [
    "dataset",
    "arch",
    "codebook_sizes",
    "total_bits",
    "freeze_depth",
    "strategy",
    "n",
    "ndcg_at_10_mean",
    "hit_rate_at_10_mean",
    "recall_at_200_mean",
    "mse_mean",
    "prefix_churn_headline_mean",
    "items_reindexed_headline_mean",
    "codebook_update_bytes_mean",
    "tokenizer_update_seconds_mean",
    "consumer_retrain_required_any",
    "consumer_retrain_seconds_mean",
    "consumer_training_sequences_mean",
    "id_migration_required_any",
    "serving_index_rebuild_required_any",
    "update_wall_seconds_mean",
]

DIAGNOSTIC_FIELDS = [
    "artifact",
    "artifact_type",
    "dataset",
    "arch",
    "codebook_sizes",
    "total_bits",
    "freeze_depth",
    "seed",
    "xi_s",
    "prefix_subspace_rank",
    "prefix_subspace_dim_bound",
    "drift_energy",
    "prefix_drift_energy",
    "orthogonal_drift_energy",
    "prefix_drift_rms",
    "drift_norm_mean",
    "drift_norm_p50",
    "drift_norm_p95",
    "epsilon_s_temporal",
    "stable_item_fraction",
    "prefix_margin_source_min_mean",
    "prefix_margin_source_min_p01",
    "prefix_margin_source_min_p05",
    "prefix_margin_source_min_p10",
    "prefix_margin_source_min_p50",
    "prefix_margin_source_min_p95",
    "fragile_margin_fraction_at_half_prefix_rms",
    "fragile_margin_fraction_at_prefix_rms",
    "fragile_margin_fraction_at_2x_prefix_rms",
    "suffix_repair_residual_mse",
    "suffix_repair_residual_ratio",
    "stratified_gap_recovery_diagnostic",
    "delta_task_tv_weighted",
    "delta_task_tv_common_contexts",
    "delta_task_context_overlap",
    "delta_task_contexts_source",
    "delta_task_contexts_target",
    "delta_task_contexts_common",
    "delta_task_source_transitions",
    "delta_task_target_transitions",
    "delta_task_stable_item_fraction",
    "candidate_coverage_frozen",
    "candidate_coverage_stratified",
    "candidate_coverage_grm_only",
    "ndcg_at_10_frozen",
    "ndcg_at_10_stratified",
    "ndcg_at_10_grm_only",
    "stratified_lift_vs_frozen_ndcg_at_10",
    "grm_lift_vs_frozen_ndcg_at_10",
    "grm_lift_vs_stratified_ndcg_at_10",
]

DOWNSTREAM_FIELDS.extend([
    field for field in DIAGNOSTIC_FIELDS
    if field not in DOWNSTREAM_FIELDS and field != "artifact_type"
])
INDEX_RUN_FIELDS.extend([
    field for field in DIAGNOSTIC_FIELDS
    if field not in INDEX_RUN_FIELDS and field != "artifact_type"
])


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


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def _tokenizer_update_seconds(strategy: str, timing: dict, run: dict | None):
    if strategy == "frozen" or strategy == "grm_only_retrained_generator":
        return 0.0
    if strategy == "stratified":
        if run is not None:
            return run.get("suffix_update_seconds", 0.0)
        return timing.get("suffix_update_seconds", 0.0)
    if strategy.startswith("warm_start_full"):
        return timing.get("warm_full_codebook_seconds", 0.0)
    if strategy.startswith("ema_streaming_vq"):
        return timing.get("ema_codebook_seconds", 0.0)
    if strategy.startswith("full_"):
        return timing.get("full_codebook_seconds", 0.0)
    return 0.0


def _consumer_retrain_seconds(strategy: str, timing: dict):
    if strategy == "grm_only_retrained_generator":
        return timing.get("grm_generator_seconds", 0.0)
    if strategy == "full_retrained_generator":
        return timing.get("target_generator_seconds", 0.0)
    return 0.0


def _consumer_training_sequences(strategy: str, timing: dict) -> int:
    if strategy == "grm_only_retrained_generator":
        return int(timing.get("grm_generator_training_sequences", 0))
    if strategy == "full_retrained_generator":
        return int(timing.get("target_generator_training_sequences", 0))
    return 0


def _fill_cost_fields(row: dict, payload: dict, run: dict | None = None) -> None:
    strategy = row.get("strategy", "")
    timing = payload.get("timing", {})
    headline_items = _headline_items(row)
    consumer_retrain_required = _truthy(row.get("consumer_retrained", False))
    tokenizer_seconds = row.get("tokenizer_update_seconds")
    if tokenizer_seconds in (None, ""):
        tokenizer_seconds = _tokenizer_update_seconds(strategy, timing, run)
    consumer_seconds = row.get("consumer_retrain_seconds")
    if consumer_seconds in (None, ""):
        consumer_seconds = _consumer_retrain_seconds(strategy, timing)
    consumer_sequences = row.get("consumer_training_sequences")
    if consumer_sequences in (None, ""):
        consumer_sequences = _consumer_training_sequences(strategy, timing)

    tokenizer_seconds = float(tokenizer_seconds)
    consumer_seconds = float(consumer_seconds)
    row["id_migration_required"] = row.get(
        "id_migration_required", headline_items > 0,
    )
    row["serving_index_rebuild_required"] = row.get(
        "serving_index_rebuild_required", headline_items > 0,
    )
    row["consumer_retrain_required"] = row.get(
        "consumer_retrain_required", consumer_retrain_required,
    )
    row["tokenizer_update_seconds"] = tokenizer_seconds
    row["consumer_retrain_seconds"] = consumer_seconds
    row["consumer_training_sequences"] = int(consumer_sequences)
    row["update_wall_seconds"] = row.get(
        "update_wall_seconds", tokenizer_seconds + consumer_seconds,
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
        diagnostics = payload.get("diagnostics", {})
        for strategy in payload.get("strategies", []):
            row = dict(base)
            row.update(strategy)
            row.update(diagnostics)
            row["prefix_churn_headline"] = _headline_churn(strategy)
            row["items_reindexed_headline"] = _headline_items(strategy)
            _fill_cost_fields(row, payload)
            rows.append(row)
    return rows


def _bounded_candidate_rows(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("downstream_*.json")):
        payload = _read_json(path)
        base = _base_fields(path, payload)
        base["freeze_depth"] = payload.get("configuration", {}).get(
            "freeze_depth", ""
        )
        for result in payload.get("candidate_budget_sweep", []):
            row = dict(base)
            row.update(result)
            rows.append(row)
    return rows


def _candidate_grid_rows(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("downstream_*.json")):
        payload = _read_json(path)
        base = _base_fields(path, payload)
        base["freeze_depth"] = payload.get("configuration", {}).get(
            "freeze_depth", ""
        )
        for result in payload.get("candidate_grid_sweep", []):
            row = dict(base)
            row.update(result)
            rows.append(row)
    return rows


def _context_reranker_rows(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("context_*.json")):
        payload = _read_json(path)
        base = _base_fields(path, payload)
        base["freeze_depth"] = payload.get("configuration", {}).get(
            "freeze_depth", ""
        )
        for result in payload.get("context_reranker_rows", []):
            row = dict(base)
            row.update(result)
            rows.append(row)
    return rows


def _context_reranker_grid_rows(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("context_*.json")):
        payload = _read_json(path)
        base = _base_fields(path, payload)
        base["freeze_depth"] = payload.get("configuration", {}).get(
            "freeze_depth", ""
        )
        for result in payload.get("context_reranker_grid", []):
            row = dict(base)
            row.update(result)
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
                row.update(run.get("diagnostics", {}))
                row["prefix_churn_headline"] = _headline_churn(strategy)
                row["items_reindexed_headline"] = _headline_items(strategy)
                _fill_cost_fields(row, payload, run)
                rows.append(row)
    return rows


def _strategy_by_name(payload: dict) -> dict:
    return {
        row.get("strategy", ""): row
        for row in payload.get("strategies", [])
    }


def _strategy_metric(strategies: dict, name: str, metric: str):
    value = strategies.get(name, {}).get(metric, "")
    return value if value is not None else ""


def _difference(left, right):
    if left == "" or right == "":
        return ""
    return float(left) - float(right)


def _diagnostic_rows(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("downstream_*.json")):
        payload = _read_json(path)
        diagnostics = payload.get("diagnostics", {})
        if not diagnostics:
            continue
        row = _base_fields(path, payload)
        row["artifact_type"] = "downstream"
        row["freeze_depth"] = payload.get("configuration", {}).get(
            "freeze_depth", ""
        )
        row.update(diagnostics)
        strategies = _strategy_by_name(payload)
        frozen_ndcg = _strategy_metric(strategies, "frozen", "ndcg_at_10")
        stratified_ndcg = _strategy_metric(strategies, "stratified", "ndcg_at_10")
        grm_ndcg = _strategy_metric(
            strategies, "grm_only_retrained_generator", "ndcg_at_10"
        )
        row.update({
            "candidate_coverage_frozen": _strategy_metric(
                strategies, "frozen", "routing_coverage",
            ),
            "candidate_coverage_stratified": _strategy_metric(
                strategies, "stratified", "routing_coverage",
            ),
            "candidate_coverage_grm_only": _strategy_metric(
                strategies, "grm_only_retrained_generator", "routing_coverage",
            ),
            "ndcg_at_10_frozen": frozen_ndcg,
            "ndcg_at_10_stratified": stratified_ndcg,
            "ndcg_at_10_grm_only": grm_ndcg,
            "stratified_lift_vs_frozen_ndcg_at_10": _difference(
                stratified_ndcg, frozen_ndcg,
            ),
            "grm_lift_vs_frozen_ndcg_at_10": _difference(
                grm_ndcg, frozen_ndcg,
            ),
            "grm_lift_vs_stratified_ndcg_at_10": _difference(
                grm_ndcg, stratified_ndcg,
            ),
        })
        rows.append(row)

    for path in sorted(results_dir.glob("index_*.json")):
        payload = _read_json(path)
        base = _base_fields(path, payload)
        for run in payload.get("runs", []):
            diagnostics = run.get("diagnostics", {})
            if not diagnostics:
                continue
            row = dict(base)
            row["artifact_type"] = "index"
            row["freeze_depth"] = run.get("freeze_depth", "")
            row.update(diagnostics)
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
            "tokenizer_update_seconds_mean": _mean([
                float(row["tokenizer_update_seconds"]) for row in group
                if row.get("tokenizer_update_seconds") != ""
            ]),
            "consumer_retrain_seconds_mean": _mean([
                float(row["consumer_retrain_seconds"]) for row in group
                if row.get("consumer_retrain_seconds") != ""
            ]),
            "update_wall_seconds_mean": _mean([
                float(row["update_wall_seconds"]) for row in group
                if row.get("update_wall_seconds") != ""
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


def _cost_summary(rows: list[dict]) -> list[dict]:
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
        summary.append({
            "dataset": key[0],
            "arch": key[1],
            "codebook_sizes": key[2],
            "total_bits": key[3],
            "freeze_depth": key[4],
            "strategy": key[5],
            "n": len(group),
            "ndcg_at_10_mean": _mean([
                float(row["ndcg_at_10"]) for row in group
                if row.get("ndcg_at_10") != ""
            ]),
            "hit_rate_at_10_mean": _mean([
                float(row["hit_rate_at_10"]) for row in group
                if row.get("hit_rate_at_10") != ""
            ]),
            "recall_at_200_mean": _mean([
                float(row["recall_at_200"]) for row in group
                if row.get("recall_at_200") != ""
            ]),
            "mse_mean": _mean([
                float(row["mse"]) for row in group
                if row.get("mse") != ""
            ]),
            "prefix_churn_headline_mean": _mean([
                float(row["prefix_churn_headline"]) for row in group
                if row.get("prefix_churn_headline") != ""
            ]),
            "items_reindexed_headline_mean": _mean([
                float(row["items_reindexed_headline"]) for row in group
                if row.get("items_reindexed_headline") != ""
            ]),
            "codebook_update_bytes_mean": _mean([
                float(row["codebook_update_bytes"]) for row in group
                if row.get("codebook_update_bytes") != ""
            ]),
            "tokenizer_update_seconds_mean": _mean([
                float(row["tokenizer_update_seconds"]) for row in group
                if row.get("tokenizer_update_seconds") != ""
            ]),
            "consumer_retrain_required_any": any(
                _truthy(row.get("consumer_retrain_required", False))
                for row in group
            ),
            "consumer_retrain_seconds_mean": _mean([
                float(row["consumer_retrain_seconds"]) for row in group
                if row.get("consumer_retrain_seconds") != ""
            ]),
            "consumer_training_sequences_mean": _mean([
                float(row["consumer_training_sequences"]) for row in group
                if row.get("consumer_training_sequences") != ""
            ]),
            "id_migration_required_any": any(
                _truthy(row.get("id_migration_required", False))
                for row in group
            ),
            "serving_index_rebuild_required_any": any(
                _truthy(row.get("serving_index_rebuild_required", False))
                for row in group
            ),
            "update_wall_seconds_mean": _mean([
                float(row["update_wall_seconds"]) for row in group
                if row.get("update_wall_seconds") != ""
            ]),
        })
    return summary


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
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
    bounded_candidate = _bounded_candidate_rows(results_dir)
    candidate_grid = _candidate_grid_rows(results_dir)
    context_reranker = _context_reranker_rows(results_dir)
    context_reranker_grid = _context_reranker_grid_rows(results_dir)
    index_rows = _index_rows(results_dir)
    diagnostic_rows = _diagnostic_rows(results_dir)
    index_summary = _index_summary(index_rows)
    cost_summary = _cost_summary(downstream)

    _write_csv(output_dir / "downstream_rows.csv", DOWNSTREAM_FIELDS, downstream)
    _write_csv(
        output_dir / "bounded_candidate_rows.csv",
        BOUNDED_CANDIDATE_FIELDS,
        bounded_candidate,
    )
    _write_csv(
        output_dir / "candidate_grid_rows.csv",
        CANDIDATE_GRID_FIELDS,
        candidate_grid,
    )
    _write_csv(
        output_dir / "context_reranker_rows.csv",
        CONTEXT_RERANKER_FIELDS,
        context_reranker,
    )
    _write_csv(
        output_dir / "context_reranker_grid_rows.csv",
        CONTEXT_RERANKER_GRID_FIELDS,
        context_reranker_grid,
    )
    _write_csv(output_dir / "index_runs.csv", INDEX_RUN_FIELDS, index_rows)
    _write_csv(
        output_dir / "diagnostic_rows.csv", DIAGNOSTIC_FIELDS, diagnostic_rows,
    )
    _write_csv(
        output_dir / "index_summary.csv", INDEX_SUMMARY_FIELDS, index_summary,
    )
    _write_csv(
        output_dir / "cost_summary.csv", COST_SUMMARY_FIELDS, cost_summary,
    )
    print(
        f"wrote {len(downstream)} downstream rows, "
        f"{len(bounded_candidate)} bounded-candidate rows, "
        f"{len(candidate_grid)} candidate-grid rows, "
        f"{len(context_reranker)} context-reranker rows, "
        f"{len(context_reranker_grid)} context-reranker grid rows, "
        f"{len(index_rows)} index rows, {len(diagnostic_rows)} diagnostics, "
        f"{len(index_summary)} index summaries, "
        f"{len(cost_summary)} cost summaries "
        f"to {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()

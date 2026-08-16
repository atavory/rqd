#!/usr/bin/env python3
"""Build Overleaf-ready analysis artifacts from raw WSDM result files.

This is the only paper-facing analysis layer for the current WSDM run. It
reads raw JSON/CSV outputs, excludes incomplete JSONs from headline tables,
writes coverage reports for partial work, and emits CSV/Tex/PGFPlots artifacts
that can be checked into the data Overleaf project.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


EXPECTED_DOWNSTREAM_STRATEGIES = [
    "frozen",
    "stratified",
    "warm_start_full_old_generator",
    "ema_streaming_vq_old_generator",
    "full_old_generator",
    "full_old_generator_centroid_relabel",
    "full_old_generator_assignment_relabel",
    "grm_only_retrained_generator",
    "full_retrained_generator",
]

PLOT_STRATEGIES = [
    "frozen",
    "stratified",
    "full_old_generator_assignment_relabel",
    "grm_only_retrained_generator",
    "full_retrained_generator",
]

STRATEGY_LABELS = {
    "frozen": "Frozen",
    "stratified": "Suffix-only",
    "warm_start_full_old_generator": "Warm full, old model",
    "ema_streaming_vq_old_generator": "EMA VQ, old model",
    "full_old_generator": "Full, old model",
    "full_old_generator_centroid_relabel": "Full + centroid relabel",
    "full_old_generator_assignment_relabel": "Full + assignment relabel",
    "grm_only_retrained_generator": "GRM-only retrain",
    "full_retrained_generator": "Full + new model",
}

DEFAULT_RESULTS = [
    "results/amazon2023_5core_index_20260812",
    "results/tier_c_retrain_prediction_20260812/real_index",
    "results/amazon2023_downstream_rung_funnel24_20260814",
]

DEFAULT_TIER_C_REAL = (
    "results/tier_c_retrain_prediction_20260812/predictions/"
    "tier_c_real_retrain_predictions.csv"
)
DEFAULT_TIER_C_SYNTHETIC = (
    "results/tier_c_retrain_prediction_20260812/synthetic/"
    "tier_c_synthetic_retrain_predictions_v2.csv"
)

LEGACY_LABEL_ALIASES = {
    "consumer_retrain": "model_retrain",
    "keep_consumer": "keep_model",
    "suffix_plus_consumer_retrain": "suffix_plus_model_retrain",
    "consumer_only": "model_only",
}


def _read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _paper_label(value: str) -> str:
    return LEGACY_LABEL_ALIASES.get(value, value)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, hash_files: bool) -> dict:
    stat = path.stat()
    record = {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(
            stat.st_mtime, timezone.utc
        ).isoformat(),
    }
    if hash_files:
        record["sha256"] = _sha256(path)
    return record


def _float(value, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def _int(value, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _fmt(value, digits: int = 4) -> str:
    number = _float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def _fmt_pct(value, digits: int = 1) -> str:
    number = _float(value)
    if number is None:
        return ""
    return f"{100.0 * number:.{digits}f}\\%"


def _latex_escape(value) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


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


def _strategy_map(strategies: list[dict]) -> dict[str, dict]:
    return {row.get("strategy", ""): row for row in strategies}


def _headline_churn(row: dict) -> float | None:
    return _float(
        row.get(
            "prefix_churn_headline",
            row.get("prefix_churn_assignment_aligned", row.get("prefix_churn")),
        )
    )


def _is_complete_downstream(payload: dict) -> bool:
    names = {row.get("strategy", "") for row in payload.get("strategies", [])}
    return all(name in names for name in EXPECTED_DOWNSTREAM_STRATEGIES)


def _resolve_paths(root: Path, values: list[str]) -> list[Path]:
    paths = []
    for value in values:
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        paths.append(path)
    return paths


def _collect_downstream(result_dirs: list[Path]) -> tuple[list[dict], list[dict]]:
    rows = []
    coverage = []
    for result_dir in result_dirs:
        for path in sorted(result_dir.glob("downstream_*.json")):
            payload = _read_json(path)
            config = payload.get("configuration", {})
            strategies = payload.get("strategies", [])
            names = [row.get("strategy", "") for row in strategies]
            complete = _is_complete_downstream(payload)
            coverage.append({
                "artifact": path.name,
                "result_dir": str(result_dir),
                "dataset": _dataset_name(payload),
                "arch": config.get("arch", ""),
                "freeze_depth": config.get("freeze_depth", ""),
                "seed": config.get("seed", ""),
                "strategies_written": len(strategies),
                "complete": complete,
                "missing_strategies": ",".join(
                    name for name in EXPECTED_DOWNSTREAM_STRATEGIES
                    if name not in set(names)
                ),
                "bytes": path.stat().st_size,
                "mtime_utc": datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc
                ).isoformat(),
            })
            diagnostics = payload.get("diagnostics", {})
            for strategy in strategies:
                row = {
                    "artifact": path.name,
                    "dataset": _dataset_name(payload),
                    "arch": config.get("arch", ""),
                    "codebook_sizes": _codebook_sizes(config),
                    "total_bits": config.get("total_bits", ""),
                    "freeze_depth": config.get("freeze_depth", ""),
                    "seed": config.get("seed", ""),
                    "strategy": strategy.get("strategy", ""),
                    "complete_artifact": complete,
                    "n_eval": strategy.get("n_eval", ""),
                    "ndcg_at_10": strategy.get("ndcg_at_10", ""),
                    "hit_rate_at_10": strategy.get("hit_rate_at_10", ""),
                    "recall_at_200": strategy.get("recall_at_200", ""),
                    "routing_coverage": strategy.get("routing_coverage", ""),
                    "mse": strategy.get("mse", ""),
                    "prefix_churn_headline": _headline_churn(strategy),
                    "items_reindexed_headline": strategy.get(
                        "items_reindexed_headline",
                        strategy.get(
                            "items_reindexed_assignment_aligned",
                            strategy.get("items_reindexed", ""),
                        ),
                    ),
                    "codebook_update_bytes": strategy.get(
                        "codebook_update_bytes", ""
                    ),
                    "model_retrain_seconds": strategy.get(
                        "consumer_retrain_seconds", ""
                    ),
                    "model_training_sequences": strategy.get(
                        "consumer_training_sequences", ""
                    ),
                    "update_wall_seconds": strategy.get("update_wall_seconds", ""),
                    "xi_s": diagnostics.get("xi_s", ""),
                    "epsilon_s_temporal": diagnostics.get(
                        "epsilon_s_temporal", ""
                    ),
                    "delta_task_tv_weighted": diagnostics.get(
                        "delta_task_tv_weighted", ""
                    ),
                    "suffix_repair_residual_ratio": diagnostics.get(
                        "suffix_repair_residual_ratio", ""
                    ),
                }
                rows.append(row)
    return rows, coverage


def _collect_index(result_dirs: list[Path]) -> tuple[list[dict], list[dict]]:
    rows = []
    coverage = []
    for result_dir in result_dirs:
        for path in sorted(result_dir.glob("index_*.json")):
            payload = _read_json(path)
            config = payload.get("configuration", {})
            runs = payload.get("runs", [])
            coverage.append({
                "artifact": path.name,
                "result_dir": str(result_dir),
                "dataset": _dataset_name(payload),
                "arch": config.get("arch", ""),
                "seed": config.get("seed", ""),
                "runs": len(runs),
                "bytes": path.stat().st_size,
                "mtime_utc": datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc
                ).isoformat(),
            })
            for run in runs:
                diagnostics = run.get("diagnostics", {})
                for strategy in run.get("strategies", []):
                    rows.append({
                        "artifact": path.name,
                        "dataset": _dataset_name(payload),
                        "arch": config.get("arch", ""),
                        "codebook_sizes": _codebook_sizes(config),
                        "total_bits": config.get("total_bits", ""),
                        "freeze_depth": run.get("freeze_depth", ""),
                        "seed": config.get("seed", ""),
                        "strategy": strategy.get("strategy", ""),
                        "mse": strategy.get("mse", ""),
                        "normalized_mse": strategy.get("normalized_mse", ""),
                        "prefix_churn_headline": _headline_churn(strategy),
                        "items_reindexed_headline": strategy.get(
                            "items_reindexed_headline",
                            strategy.get(
                                "items_reindexed_assignment_aligned",
                                strategy.get("items_reindexed", ""),
                            ),
                        ),
                        "codebook_update_bytes": strategy.get(
                            "codebook_update_bytes", ""
                        ),
                        "update_wall_seconds": strategy.get(
                            "update_wall_seconds", ""
                        ),
                        "stratified_gap_recovery": run.get(
                            "stratified_gap_recovery", ""
                        ),
                        "warm_start_full_gap_recovery": run.get(
                            "warm_start_full_gap_recovery", ""
                        ),
                        "ema_gap_recovery": run.get("ema_gap_recovery", ""),
                        "xi_s": diagnostics.get("xi_s", ""),
                        "epsilon_s_temporal": diagnostics.get(
                            "epsilon_s_temporal", ""
                        ),
                        "delta_task_tv_weighted": diagnostics.get(
                            "delta_task_tv_weighted", ""
                        ),
                        "suffix_repair_residual_ratio": diagnostics.get(
                            "suffix_repair_residual_ratio", ""
                        ),
                    })
    return rows, coverage


def _downstream_headlines(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if not row.get("complete_artifact"):
            continue
        key = (
            row.get("dataset", ""),
            row.get("arch", ""),
            row.get("freeze_depth", ""),
            row.get("seed", ""),
        )
        grouped[key][row.get("strategy", "")] = row

    headlines = []
    for key, strategies in sorted(grouped.items()):
        frozen = strategies.get("frozen", {})
        suffix = strategies.get("stratified", {})
        full_new = strategies.get("full_retrained_generator", {})
        assignment = strategies.get("full_old_generator_assignment_relabel", {})
        grm = strategies.get("grm_only_retrained_generator", {})
        frozen_ndcg = _float(frozen.get("ndcg_at_10"))
        suffix_ndcg = _float(suffix.get("ndcg_at_10"))
        full_ndcg = _float(full_new.get("ndcg_at_10"))
        suffix_lift_abs = (
            suffix_ndcg - frozen_ndcg
            if frozen_ndcg is not None and suffix_ndcg is not None
            else None
        )
        suffix_lift_rel = (
            suffix_lift_abs / frozen_ndcg
            if suffix_lift_abs is not None and frozen_ndcg
            else None
        )
        zero_migration = [
            row for row in strategies.values()
            if _float(row.get("prefix_churn_headline"), 1.0) == 0.0
        ]
        best_zero = max(
            zero_migration,
            key=lambda row: _float(row.get("ndcg_at_10"), -1.0) or -1.0,
        ) if zero_migration else {}
        headlines.append({
            "dataset": key[0],
            "arch": key[1],
            "freeze_depth": key[2],
            "seed": key[3],
            "n_eval": frozen.get("n_eval", ""),
            "frozen_ndcg_at_10": frozen_ndcg,
            "suffix_ndcg_at_10": suffix_ndcg,
            "suffix_lift_abs": suffix_lift_abs,
            "suffix_lift_rel": suffix_lift_rel,
            "suffix_churn": suffix.get("prefix_churn_headline", ""),
            "suffix_update_seconds": suffix.get("update_wall_seconds", ""),
            "best_zero_migration_strategy": best_zero.get("strategy", ""),
            "best_zero_migration_ndcg_at_10": best_zero.get("ndcg_at_10", ""),
            "assignment_relabel_ndcg_at_10": assignment.get("ndcg_at_10", ""),
            "grm_only_ndcg_at_10": grm.get("ndcg_at_10", ""),
            "full_new_model_ndcg_at_10": full_ndcg,
            "full_new_model_churn": full_new.get(
                "prefix_churn_headline", ""
            ),
            "full_new_model_update_seconds": full_new.get(
                "update_wall_seconds", ""
            ),
        })
    return headlines


def _aggregate_index(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if row.get("strategy") not in {
            "frozen",
            "stratified",
            "warm_start_full",
            "warm_start_full_old_generator",
            "full_retrained",
            "full_retrained_generator",
        }:
            continue
        key = (
            row.get("dataset", ""),
            row.get("arch", ""),
            row.get("freeze_depth", ""),
            row.get("strategy", ""),
        )
        grouped[key].append(row)
    out = []
    for key, group in sorted(grouped.items()):
        mse = [_float(row.get("mse")) for row in group]
        churn = [_float(row.get("prefix_churn_headline")) for row in group]
        gaps = [_float(row.get("stratified_gap_recovery")) for row in group]
        mse = [value for value in mse if value is not None]
        churn = [value for value in churn if value is not None]
        gaps = [value for value in gaps if value is not None]
        out.append({
            "dataset": key[0],
            "arch": key[1],
            "freeze_depth": key[2],
            "strategy": key[3],
            "n": len(group),
            "mse_mean": mean(mse) if mse else "",
            "prefix_churn_mean": mean(churn) if churn else "",
            "stratified_gap_recovery_mean": mean(gaps) if gaps else "",
        })
    return out


def _tier_c_summary(real_rows: list[dict], synthetic_rows: list[dict]) -> dict:
    real_actions = Counter(
        row.get("predicted_cheapest_rung", "") for row in real_rows
        if row.get("predicted_cheapest_rung", "")
    )
    synthetic_actions = Counter(
        row.get("predicted_cheapest_rung", "") for row in synthetic_rows
        if row.get("predicted_cheapest_rung", "")
    )
    matches = [
        row.get("matches_expected_regime", "").lower() == "true"
        for row in synthetic_rows
        if row.get("matches_expected_regime", "") != ""
    ]
    synthetic_by_scenario = defaultdict(list)
    for row in synthetic_rows:
        scenario = _paper_label(row.get("scenario", ""))
        if not scenario:
            continue
        match = row.get("matches_expected_regime", "").lower() == "true"
        synthetic_by_scenario[scenario].append(match)
    return {
        "real_rows": len(real_rows),
        "synthetic_rows": len(synthetic_rows),
        "real_actions": real_actions,
        "synthetic_actions": synthetic_actions,
        "synthetic_accuracy": (
            sum(matches) / len(matches) if matches else ""
        ),
        "synthetic_by_scenario": {
            scenario: {
                "n": len(values),
                "accuracy": sum(values) / len(values) if values else "",
            }
            for scenario, values in sorted(synthetic_by_scenario.items())
        },
    }


def _tier_c_predict(row: dict, thresholds: dict) -> dict:
    frozen_excess = _float(row.get("frozen_excess_mse"), 0.0) or 0.0
    residual_ratio = _float(row.get("suffix_repair_residual_ratio"), 1.0) or 1.0
    epsilon = _float(row.get("epsilon_s_temporal"), 0.0) or 0.0
    xi_s = _float(row.get("xi_s"), 0.0) or 0.0
    fragile = (
        _float(row.get("fragile_margin_fraction_at_prefix_rms"), 0.0) or 0.0
    )
    delta_task = _float(row.get("delta_task_tv_weighted"), 0.0) or 0.0

    if frozen_excess <= thresholds["frozen_excess_tol"]:
        reconstruction_action = "do_nothing"
    elif residual_ratio <= thresholds["suffix_residual_tol"]:
        reconstruction_action = "suffix_only"
    else:
        reconstruction_action = "widen_suffix_or_full_tokenizer"

    interface_action = (
        "selective_or_full_migration"
        if (
            epsilon >= thresholds["crossing_tol"]
            or xi_s >= thresholds["xi_tol"]
            or fragile >= thresholds["fragile_tol"]
        )
        else "stable_interface"
    )
    model_action = (
        "model_retrain"
        if delta_task >= thresholds["delta_task_tol"]
        else "keep_model"
    )

    if interface_action != "stable_interface":
        rung = "selective_or_full_migration"
    elif model_action == "model_retrain" and reconstruction_action == "suffix_only":
        rung = "suffix_plus_model_retrain"
    elif model_action == "model_retrain":
        rung = "grm_only"
    else:
        rung = reconstruction_action

    return {
        "predicted_reconstruction_action": reconstruction_action,
        "predicted_interface_action": interface_action,
        "predicted_model_action": model_action,
        "predicted_cheapest_rung": rung,
    }


def _tier_c_match(row: dict, prediction: dict) -> bool:
    expected = _paper_label(row.get("expected_regime", ""))
    if expected == "model_only":
        expected = "grm_only"
    if not expected:
        return False
    if expected == "widen_suffix_or_full_tokenizer":
        return prediction["predicted_reconstruction_action"] == expected
    return prediction["predicted_cheapest_rung"] == expected


def _tier_c_calibration(synthetic_rows: list[dict]) -> dict:
    if not synthetic_rows:
        return {
            "accuracy": "",
            "correct": 0,
            "n": 0,
            "thresholds": {},
        }

    defaults = {
        "frozen_excess_tol": 0.02,
        "suffix_residual_tol": 0.35,
        "crossing_tol": 0.05,
        "xi_tol": 0.35,
        "fragile_tol": 0.35,
        "delta_task_tol": 0.35,
    }
    grid = {
        "frozen_excess_tol": [0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0],
        "suffix_residual_tol": [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.1],
        "crossing_tol": [0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.1],
        "xi_tol": [0.35, 0.5, 0.8, 0.95, 1.1],
        "fragile_tol": [0.35, 0.5, 0.8, 0.95, 1.1],
        "delta_task_tol": [0.1, 0.35, 0.5, 0.8, 1.1],
    }

    best = None
    names = list(grid)
    for frozen_excess_tol in grid["frozen_excess_tol"]:
        for suffix_residual_tol in grid["suffix_residual_tol"]:
            for crossing_tol in grid["crossing_tol"]:
                for xi_tol in grid["xi_tol"]:
                    for fragile_tol in grid["fragile_tol"]:
                        for delta_task_tol in grid["delta_task_tol"]:
                            thresholds = {
                                "frozen_excess_tol": frozen_excess_tol,
                                "suffix_residual_tol": suffix_residual_tol,
                                "crossing_tol": crossing_tol,
                                "xi_tol": xi_tol,
                                "fragile_tol": fragile_tol,
                                "delta_task_tol": delta_task_tol,
                            }
                            correct = sum(
                                1 for row in synthetic_rows
                                if _tier_c_match(row, _tier_c_predict(row, thresholds))
                            )
                            distance = sum(
                                abs(thresholds[name] - defaults[name])
                                for name in names
                            )
                            candidate = (correct, -distance, thresholds)
                            if best is None or candidate > best:
                                best = candidate

    assert best is not None
    correct, _negative_distance, thresholds = best
    return {
        "accuracy": correct / len(synthetic_rows),
        "correct": correct,
        "n": len(synthetic_rows),
        "thresholds": thresholds,
    }


def _tier_c_action_counts(rows: list[dict], thresholds: dict | None) -> Counter:
    if thresholds is None:
        return Counter(
            row.get("predicted_cheapest_rung", "") for row in rows
            if row.get("predicted_cheapest_rung", "")
        )
    return Counter(
        _tier_c_predict(row, thresholds)["predicted_cheapest_rung"]
        for row in rows
    )


def _tier_c_identifiability(synthetic_rows: list[dict]) -> list[dict]:
    feature_fields = [
        "seed",
        "arch",
        "freeze_depth",
        "magnitude",
        "frozen_mse",
        "stratified_mse",
        "full_mse",
        "frozen_excess_mse",
        "suffix_repair_residual_ratio",
        "stratified_gap_recovery",
        "xi_s",
        "epsilon_s_temporal",
        "fragile_margin_fraction_at_prefix_rms",
        "delta_task_tv_weighted",
    ]
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in synthetic_rows:
        grouped[tuple(row.get(field, "") for field in feature_fields)].append(row)

    conflicts = []
    for group_id, (_, group) in enumerate(sorted(grouped.items())):
        labels = sorted({
            _paper_label(row.get("expected_regime", "")) for row in group
        })
        scenarios = sorted({
            _paper_label(row.get("scenario", "")) for row in group
        })
        if len(labels) <= 1:
            continue
        conflicts.append({
            "conflict_group": group_id,
            "rows": len(group),
            "scenarios": ",".join(scenarios),
            "expected_regimes": ",".join(labels),
        })
    return conflicts


def _analysis_status(
    downstream_coverage: list[dict],
    index_coverage: list[dict],
    tier_c: dict,
    tier_c_conflicts: list[dict],
    artifact_refs: list[str],
) -> list[dict]:
    complete_downstream = [
        row for row in downstream_coverage if row.get("complete")
    ]
    partial_downstream = [
        row for row in downstream_coverage if not row.get("complete")
    ]
    return [
        {
            "component": "downstream_complete_jsons",
            "status": "available" if complete_downstream else "missing",
            "count": len(complete_downstream),
            "note": "headline tables use only complete 9-strategy JSONs",
        },
        {
            "component": "downstream_partial_jsons",
            "status": "running" if partial_downstream else "none",
            "count": len(partial_downstream),
            "note": "listed in coverage, excluded from headline tables",
        },
        {
            "component": "index_jsons",
            "status": "available" if index_coverage else "missing",
            "count": len(index_coverage),
            "note": "full-catalog reconstruction/churn rows",
        },
        {
            "component": "tier_c_real_predictions",
            "status": "available" if tier_c["real_rows"] else "missing",
            "count": tier_c["real_rows"],
            "note": "real-dataset retrain-rung predictions",
        },
        {
            "component": "tier_c_synthetic_predictions",
            "status": "available" if tier_c["synthetic_rows"] else "missing",
            "count": tier_c["synthetic_rows"],
            "note": "synthetic-control calibration rows",
        },
        {
            "component": "tier_c_identifiability_conflicts",
            "status": "needs_fix" if tier_c_conflicts else "clean",
            "count": len(tier_c_conflicts),
            "note": (
                "feature-identical synthetic rows with conflicting expected labels"
            ),
        },
        {
            "component": "remote_artifact_refs",
            "status": "recorded" if artifact_refs else "not_recorded",
            "count": len(artifact_refs),
            "note": "Manifold tarball refs, not interpreted as final tables",
        },
    ]


def _abstract_readiness_rows(
    downstream_coverage: list[dict],
    downstream_headlines: list[dict],
    index_coverage: list[dict],
    index_rows: list[dict],
    tier_c: dict,
    tier_c_calibration: dict,
    tier_c_conflicts: list[dict],
    artifact_refs: list[str],
) -> list[dict]:
    complete_downstream = [
        row for row in downstream_coverage if row.get("complete")
    ]
    partial_downstream = [
        row for row in downstream_coverage if not row.get("complete")
    ]
    datasets = sorted({
        row.get("dataset", "") for row in downstream_headlines
        if row.get("dataset", "")
    })
    suffix_lifts = [
        _float(row.get("suffix_lift_rel"))
        for row in downstream_headlines
    ]
    suffix_lifts = [value for value in suffix_lifts if value is not None]
    positive_suffix = sum(1 for value in suffix_lifts if value > 0)
    suffix_zero_churn = sum(
        1 for row in downstream_headlines
        if (_float(row.get("suffix_churn")) or 0.0) == 0.0
    )
    best_zero_beats_frozen = sum(
        1 for row in downstream_headlines
        if (
            _float(row.get("best_zero_migration_ndcg_at_10"), -1.0)
            > _float(row.get("frozen_ndcg_at_10"), -1.0)
        )
    )
    full_new_beats_frozen = sum(
        1 for row in downstream_headlines
        if (
            _float(row.get("full_new_model_ndcg_at_10"), -1.0)
            > _float(row.get("frozen_ndcg_at_10"), -1.0)
        )
    )
    mean_suffix_lift = mean(suffix_lifts) if suffix_lifts else None

    downstream_ready = bool(complete_downstream)
    tier_c_ready = (
        bool(tier_c.get("real_rows"))
        and bool(tier_c.get("synthetic_rows"))
        and not tier_c_conflicts
    )
    index_ready = bool(index_coverage and index_rows)
    abstract_ready = downstream_ready and tier_c_ready and index_ready
    paper_ready = abstract_ready and not partial_downstream

    mean_suffix_lift_text = _fmt_pct(mean_suffix_lift).replace(r"\%", "%")
    tier_c_accuracy_text = _fmt_pct(
        tier_c_calibration.get("accuracy")
    ).replace(r"\%", "%")
    dataset_text = ", ".join(datasets) if datasets else "none"
    downstream_evidence = (
        f"{len(complete_downstream)} complete headline rows across {dataset_text}; "
        f"suffix-only lift positive in {positive_suffix}/"
        f"{len(downstream_headlines)}; best zero-migration beats frozen in "
        f"{best_zero_beats_frozen}/{len(downstream_headlines)}; suffix churn is "
        f"zero in {suffix_zero_churn}/{len(downstream_headlines)}; mean suffix lift "
        f"{mean_suffix_lift_text}"
    )
    tier_c_evidence = (
        f"{tier_c.get('real_rows', 0)} real prediction rows; "
        f"{tier_c.get('synthetic_rows', 0)} synthetic controls; calibrated "
        f"synthetic accuracy {tier_c_accuracy_text}; "
        f"identifiability conflicts {len(tier_c_conflicts)}"
    )
    index_evidence = (
        f"{len(index_coverage)} index JSONs; {len(index_rows)} strategy rows"
    )

    return [
        {
            "claim": "abstract_ready",
            "status": "ready_with_scope" if abstract_ready else "not_ready",
            "evidence": (
                "downstream, index, and Tier-C scripted analyses are present"
                if abstract_ready else
                "one or more required scripted analyses are missing"
            ),
            "caveat": (
                "Use scoped wording; final result tables are not frozen until "
                "remaining local/remote matrices finish."
                if abstract_ready else
                "Do not write abstract claims until required analyses are present."
            ),
        },
        {
            "claim": "paper_ready",
            "status": "ready" if paper_ready else "not_ready",
            "evidence": (
                f"partial downstream JSONs {len(partial_downstream)}; remote "
                f"artifact refs {len(artifact_refs)}"
            ),
            "caveat": (
                "Final tables require remaining downstream/remote jobs and a "
                "fresh final snapshot."
                if not paper_ready else
                "All locally tracked downstream JSONs are complete."
            ),
        },
        {
            "claim": "downstream_headline_signal",
            "status": "ready_with_scope" if downstream_ready else "missing",
            "evidence": downstream_evidence,
            "caveat": (
                "Headlines exclude partial JSONs and currently cover completed "
                "local Amazon2023 rows only."
            ),
        },
        {
            "claim": "full_model_ceiling",
            "status": "ready_with_scope" if downstream_ready else "missing",
            "evidence": (
                f"full new-model retraining beats frozen in "
                f"{full_new_beats_frozen}/{len(downstream_headlines)} headline rows"
            ),
            "caveat": (
                "Full retraining is an upper-cost comparator, not the zero-migration "
                "serving path."
            ),
        },
        {
            "claim": "tier_c_gating_analysis",
            "status": "ready" if tier_c_ready else "needs_fix",
            "evidence": tier_c_evidence,
            "caveat": (
                "Synthetic calibration is a control/diagnostic for the rung policy, "
                "not a replacement for task metrics."
            ),
        },
        {
            "claim": "index_reconstruction_analysis",
            "status": "ready" if index_ready else "missing",
            "evidence": index_evidence,
            "caveat": (
                "Index rows support reconstruction/churn analysis; downstream NDCG "
                "claims come only from complete downstream JSONs."
            ),
        },
        {
            "claim": "scripted_analysis_discipline",
            "status": "ready",
            "evidence": (
                "All generated tables, CSVs, PGFPlots snippets, coverage, and "
                "readiness notes are emitted by make_wsdm_overleaf_analysis.py"
            ),
            "caveat": "Do not edit generated outputs by hand.",
        },
    ]


def _downstream_tex(headlines: list[dict]) -> str:
    lines = [
        "% Generated by make_wsdm_overleaf_analysis.py. Do not edit by hand.",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        (
            "Dataset & FD & Frozen & Suffix-only & Lift & Churn & "
            r"Best zero-migration & Full+new model \\"
        ),
        r"\midrule",
    ]
    for row in headlines:
        lines.append(
            " & ".join([
                _latex_escape(row["dataset"]),
                _latex_escape(row["freeze_depth"]),
                _fmt(row["frozen_ndcg_at_10"]),
                _fmt(row["suffix_ndcg_at_10"]),
                _fmt_pct(row["suffix_lift_rel"]),
                _fmt_pct(row["suffix_churn"]),
                _fmt(row["best_zero_migration_ndcg_at_10"]),
                _fmt(row["full_new_model_ndcg_at_10"]),
            ]) + r" \\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ])
    return "\n".join(lines)


def _abstract_readiness_tex(rows: list[dict]) -> str:
    lines = [
        "% Generated by make_wsdm_overleaf_analysis.py. Do not edit by hand.",
        r"\begin{tabular}{lp{0.16\linewidth}p{0.36\linewidth}p{0.28\linewidth}}",
        r"\toprule",
        r"Claim & Status & Evidence & Caveat \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join([
                _latex_escape(row["claim"]),
                _latex_escape(row["status"]),
                _latex_escape(row["evidence"]),
                _latex_escape(row["caveat"]),
            ]) + r" \\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ])
    return "\n".join(lines)


def _tier_c_tex(summary: dict, calibration: dict) -> str:
    lines = [
        "% Generated by make_wsdm_overleaf_analysis.py. Do not edit by hand.",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Component & Rows & Accuracy \\",
        r"\midrule",
        (
            "Synthetic controls, default & "
            f"{summary['synthetic_rows']} & "
            f"{_fmt_pct(summary['synthetic_accuracy'])} " + r"\\"
        ),
        (
            "Synthetic controls, calibrated & "
            f"{calibration['n']} & "
            f"{_fmt_pct(calibration['accuracy'])} " + r"\\"
        ),
        f"Real-dataset predictions & {summary['real_rows']} &  " + r"\\",
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    return "\n".join(lines)


def _coverage_tex(status_rows: list[dict]) -> str:
    lines = [
        "% Generated by make_wsdm_overleaf_analysis.py. Do not edit by hand.",
        r"\begin{tabular}{llrl}",
        r"\toprule",
        r"Component & Status & Count & Note \\",
        r"\midrule",
    ]
    for row in status_rows:
        lines.append(
            " & ".join([
                _latex_escape(row["component"]),
                _latex_escape(row["status"]),
                str(row["count"]),
                _latex_escape(row["note"]),
            ]) + r" \\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ])
    return "\n".join(lines)


def _index_tex(index_summary: list[dict]) -> str:
    selected = [
        row for row in index_summary
        if row["strategy"] in {"frozen", "stratified", "full_retrained"}
    ]
    lines = [
        "% Generated by make_wsdm_overleaf_analysis.py. Do not edit by hand.",
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"Dataset & FD & Strategy & N & MSE & Churn \\",
        r"\midrule",
    ]
    for row in selected:
        lines.append(
            " & ".join([
                _latex_escape(row["dataset"]),
                _latex_escape(row["freeze_depth"]),
                _latex_escape(STRATEGY_LABELS.get(row["strategy"], row["strategy"])),
                str(row["n"]),
                _fmt(row["mse_mean"]),
                _fmt_pct(row["prefix_churn_mean"]),
            ]) + r" \\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ])
    return "\n".join(lines)


def _plot_data(
    output_dir: Path,
    downstream_rows: list[dict],
    tier_c: dict,
) -> tuple[list[int], list[int]]:
    complete = [
        row for row in downstream_rows
        if row.get("complete_artifact") and row.get("strategy") in PLOT_STRATEGIES
    ]
    plot_keys = sorted({
        (
            row.get("dataset", ""),
            str(row.get("freeze_depth", "")),
            str(row.get("seed", "")),
        )
        for row in complete
    })
    plot_index = {key: idx for idx, key in enumerate(plot_keys)}
    for row in complete:
        key = (
            row.get("dataset", ""),
            str(row.get("freeze_depth", "")),
            str(row.get("seed", "")),
        )
        row["x"] = plot_index[key]
        row["plot_label"] = f"{key[0]} fd{key[1]}"
        row["strategy_label"] = STRATEGY_LABELS.get(
            row.get("strategy", ""), row.get("strategy", "")
        )
    fields = [
        "x",
        "plot_label",
        "dataset",
        "freeze_depth",
        "seed",
        "strategy",
        "strategy_label",
        "ndcg_at_10",
        "prefix_churn_headline",
        "update_wall_seconds",
    ]
    _write_csv(output_dir / "plot_data/downstream_ndcg10.csv", complete, fields)
    for strategy in PLOT_STRATEGIES:
        strategy_rows = [
            row for row in complete if row.get("strategy") == strategy
        ]
        _write_csv(
            output_dir / f"plot_data/downstream_ndcg10_{strategy}.csv",
            strategy_rows,
            fields,
        )

    label_rows = [
        {
            "x": idx,
            "plot_label": f"{key[0]} fd{key[1]}",
        }
        for key, idx in sorted(plot_index.items(), key=lambda item: item[1])
    ]
    _write_csv(
        output_dir / "plot_data/downstream_plot_labels.csv",
        label_rows,
        ["x", "plot_label"],
    )

    action_rows = [
        {
            "x": idx,
            "action": action,
            "action_label": action.replace("_", r"\_"),
            "count": count,
        }
        for idx, (action, count) in enumerate(
            sorted(tier_c["real_actions"].items())
        )
    ]
    _write_csv(
        output_dir / "plot_data/tier_c_real_action_counts.csv",
        action_rows,
        ["x", "action", "action_label", "count"],
    )
    return list(range(len(label_rows))), list(range(len(action_rows)))


def _plot_tex(downstream_ticks: list[int], tier_c_ticks: list[int]) -> dict[str, str]:
    downstream_tick_text = ",".join(str(value) for value in downstream_ticks)
    tier_c_tick_text = ",".join(str(value) for value in tier_c_ticks)
    downstream = [
        "% Generated by make_wsdm_overleaf_analysis.py. Do not edit by hand.",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"  ybar,",
        r"  ymin=0,",
        r"  ylabel={NDCG@10},",
        f"  xtick={{{downstream_tick_text}}},",
        (
            r"  xticklabels from table="
            r"{plot_data/downstream_plot_labels.csv}{plot_label},"
        ),
        r"  x tick label style={rotate=35,anchor=east},",
        r"  legend style={at={(0.5,1.02)},anchor=south,legend columns=3},",
        r"]",
        (
            r"\addplot table[x=x,y=ndcg_at_10,col sep=comma] "
            r"{plot_data/downstream_ndcg10_frozen.csv};"
        ),
        (
            r"\addplot table[x=x,y=ndcg_at_10,col sep=comma] "
            r"{plot_data/downstream_ndcg10_stratified.csv};"
        ),
        (
            r"\addplot table[x=x,y=ndcg_at_10,col sep=comma] "
            r"{plot_data/downstream_ndcg10_full_retrained_generator.csv};"
        ),
        r"\legend{Frozen,Suffix-only,Full+new model}",
        r"\end{axis}",
        r"\end{tikzpicture}",
        "",
    ]
    tier_c = [
        "% Generated by make_wsdm_overleaf_analysis.py. Do not edit by hand.",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"  ybar,",
        r"  ymin=0,",
        r"  ylabel={Rows},",
        f"  xtick={{{tier_c_tick_text}}},",
        (
            r"  xticklabels from table="
            r"{plot_data/tier_c_real_action_counts.csv}{action_label},"
        ),
        r"  x tick label style={rotate=35,anchor=east},",
        r"]",
        (
            r"\addplot table[x=x,y=count,col sep=comma] "
            r"{plot_data/tier_c_real_action_counts.csv};"
        ),
        r"\end{axis}",
        r"\end{tikzpicture}",
        "",
    ]
    return {
        "figures/downstream_ndcg10_pgfplots.tex": "\n".join(downstream),
        "figures/tier_c_real_actions_pgfplots.tex": "\n".join(tier_c),
    }


def _markdown_table(rows: list[dict], fields: list[str]) -> str:
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| " + " | ".join(str(row.get(field, "")) for field in fields) + " |"
        )
    return "\n".join(lines)


def _readme(output_dir: Path, args) -> str:
    command = " ".join([
        "python3",
        "scripts/make_wsdm_overleaf_analysis.py",
        "--experiment-root",
        str(args.experiment_root),
        "--output-dir",
        str(output_dir),
    ])
    if args.hash_inputs:
        command += " --hash-inputs"
    for result_dir in args.results_dir or DEFAULT_RESULTS:
        command += f" --results-dir {result_dir}"
    for artifact_ref in args.artifact_ref:
        command += f" --artifact-ref {artifact_ref}"

    return "\n".join([
        "# WSDM Overleaf Analysis Package",
        "",
        "Generated artifacts in this directory are derived only from raw JSON/CSV",
        "result files by `make_wsdm_overleaf_analysis.py`. Do not edit generated",
        "tables, plot data, or PGFPlots snippets by hand.",
        "",
        "## Regenerate",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Contents",
        "",
        "- `manifest.json`: input files, timestamps, hashes, and remote artifact refs.",
        "- `scripts/`: the analysis generator that produced this package.",
        "- `coverage.md`: complete/partial result inventory.",
        "- `abstract_readiness.md`: generated claim scope and readiness notes.",
        "- `csv/`: normalized row-level and headline CSVs.",
        "- `tables/`: generated LaTeX tables.",
        "- `plot_data/`: CSV inputs for generated figures.",
        "- `figures/`: PGFPlots snippets that read `plot_data/`.",
        "",
        "Headline downstream tables exclude partial JSONs. Partial rows are present",
        "only in coverage reports and normalized CSVs.",
        "",
    ])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-root",
        default="/data/users/atavory/scratch/wsdm_experiments",
    )
    parser.add_argument(
        "--results-dir",
        action="append",
        default=[],
        help="Result directory to parse. Relative paths resolve under experiment-root.",
    )
    parser.add_argument(
        "--tier-c-real-csv",
        default=DEFAULT_TIER_C_REAL,
        help="Tier-C real prediction CSV. Relative paths resolve under experiment-root.",
    )
    parser.add_argument(
        "--tier-c-synthetic-csv",
        default=DEFAULT_TIER_C_SYNTHETIC,
        help="Tier-C synthetic prediction CSV. Relative paths resolve under experiment-root.",
    )
    parser.add_argument(
        "--output-dir",
        default="overleaf_data/wsdm_analysis_latest",
        help="Output directory. Relative paths resolve under experiment-root.",
    )
    parser.add_argument(
        "--artifact-ref",
        action="append",
        default=[],
        help="Remote artifact reference to record in manifest, e.g. manifold:path#sha256.",
    )
    parser.add_argument(
        "--hash-inputs",
        action="store_true",
        help="SHA256 all parsed JSON/CSV inputs for a stronger manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.experiment_root)
    result_args = args.results_dir or DEFAULT_RESULTS
    result_dirs = _resolve_paths(root, result_args)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tier_c_real = Path(args.tier_c_real_csv)
    tier_c_synthetic = Path(args.tier_c_synthetic_csv)
    if not tier_c_real.is_absolute():
        tier_c_real = root / tier_c_real
    if not tier_c_synthetic.is_absolute():
        tier_c_synthetic = root / tier_c_synthetic

    downstream_rows, downstream_coverage = _collect_downstream(result_dirs)
    index_rows, index_coverage = _collect_index(result_dirs)
    real_tier_c_rows = _read_csv(tier_c_real)
    synthetic_tier_c_rows = _read_csv(tier_c_synthetic)
    tier_c = _tier_c_summary(real_tier_c_rows, synthetic_tier_c_rows)
    tier_c_calibration = _tier_c_calibration(synthetic_tier_c_rows)
    calibrated_thresholds = tier_c_calibration.get("thresholds") or None
    tier_c_conflicts = _tier_c_identifiability(synthetic_tier_c_rows)
    headlines = _downstream_headlines(downstream_rows)
    index_summary = _aggregate_index(index_rows)
    status_rows = _analysis_status(
        downstream_coverage,
        index_coverage,
        tier_c,
        tier_c_conflicts,
        args.artifact_ref,
    )
    abstract_readiness = _abstract_readiness_rows(
        downstream_coverage,
        headlines,
        index_coverage,
        index_rows,
        tier_c,
        tier_c_calibration,
        tier_c_conflicts,
        args.artifact_ref,
    )

    _write_csv(
        output_dir / "csv/downstream_rows.csv",
        downstream_rows,
        [
            "artifact",
            "dataset",
            "arch",
            "codebook_sizes",
            "total_bits",
            "freeze_depth",
            "seed",
            "strategy",
            "complete_artifact",
            "n_eval",
            "ndcg_at_10",
            "hit_rate_at_10",
            "recall_at_200",
            "routing_coverage",
            "mse",
            "prefix_churn_headline",
            "items_reindexed_headline",
            "codebook_update_bytes",
            "model_retrain_seconds",
            "model_training_sequences",
            "update_wall_seconds",
            "xi_s",
            "epsilon_s_temporal",
            "delta_task_tv_weighted",
            "suffix_repair_residual_ratio",
        ],
    )
    _write_csv(
        output_dir / "csv/downstream_headlines.csv",
        headlines,
        [
            "dataset",
            "arch",
            "freeze_depth",
            "seed",
            "n_eval",
            "frozen_ndcg_at_10",
            "suffix_ndcg_at_10",
            "suffix_lift_abs",
            "suffix_lift_rel",
            "suffix_churn",
            "suffix_update_seconds",
            "best_zero_migration_strategy",
            "best_zero_migration_ndcg_at_10",
            "assignment_relabel_ndcg_at_10",
            "grm_only_ndcg_at_10",
            "full_new_model_ndcg_at_10",
            "full_new_model_churn",
            "full_new_model_update_seconds",
        ],
    )
    _write_csv(
        output_dir / "csv/index_rows.csv",
        index_rows,
        [
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
            "items_reindexed_headline",
            "codebook_update_bytes",
            "update_wall_seconds",
            "stratified_gap_recovery",
            "warm_start_full_gap_recovery",
            "ema_gap_recovery",
            "xi_s",
            "epsilon_s_temporal",
            "delta_task_tv_weighted",
            "suffix_repair_residual_ratio",
        ],
    )
    _write_csv(
        output_dir / "csv/index_summary.csv",
        index_summary,
        [
            "dataset",
            "arch",
            "freeze_depth",
            "strategy",
            "n",
            "mse_mean",
            "prefix_churn_mean",
            "stratified_gap_recovery_mean",
        ],
    )
    real_calibrated_actions = _tier_c_action_counts(
        real_tier_c_rows, calibrated_thresholds,
    )
    synthetic_calibrated_actions = _tier_c_action_counts(
        synthetic_tier_c_rows, calibrated_thresholds,
    )
    action_rows = []
    for source, counts in [
        ("real_default", tier_c["real_actions"]),
        ("real_calibrated", real_calibrated_actions),
        ("synthetic_default", tier_c["synthetic_actions"]),
        ("synthetic_calibrated", synthetic_calibrated_actions),
    ]:
        action_rows.extend(
            {"source": source, "action": action, "count": count}
            for action, count in sorted(counts.items())
        )
    _write_csv(
        output_dir / "csv/tier_c_action_counts.csv",
        action_rows,
        ["source", "action", "count"],
    )
    calibration_row = {
        "synthetic_accuracy": tier_c_calibration["accuracy"],
        "correct": tier_c_calibration["correct"],
        "n": tier_c_calibration["n"],
    }
    calibration_row.update(tier_c_calibration.get("thresholds") or {})
    _write_csv(
        output_dir / "csv/tier_c_calibration.csv",
        [calibration_row],
        [
            "synthetic_accuracy",
            "correct",
            "n",
            "frozen_excess_tol",
            "suffix_residual_tol",
            "crossing_tol",
            "xi_tol",
            "fragile_tol",
            "delta_task_tol",
        ],
    )
    scenario_rows = [
        {
            "scenario": scenario,
            "n": values["n"],
            "accuracy": values["accuracy"],
        }
        for scenario, values in tier_c["synthetic_by_scenario"].items()
    ]
    _write_csv(
        output_dir / "csv/tier_c_synthetic_accuracy_by_scenario.csv",
        scenario_rows,
        ["scenario", "n", "accuracy"],
    )
    _write_csv(
        output_dir / "csv/tier_c_identifiability_conflicts.csv",
        tier_c_conflicts,
        ["conflict_group", "rows", "scenarios", "expected_regimes"],
    )
    _write_csv(
        output_dir / "csv/downstream_coverage.csv",
        downstream_coverage,
        [
            "artifact",
            "result_dir",
            "dataset",
            "arch",
            "freeze_depth",
            "seed",
            "strategies_written",
            "complete",
            "missing_strategies",
            "bytes",
            "mtime_utc",
        ],
    )
    _write_csv(
        output_dir / "csv/index_coverage.csv",
        index_coverage,
        [
            "artifact",
            "result_dir",
            "dataset",
            "arch",
            "seed",
            "runs",
            "bytes",
            "mtime_utc",
        ],
    )
    _write_csv(
        output_dir / "csv/analysis_status.csv",
        status_rows,
        ["component", "status", "count", "note"],
    )
    _write_csv(
        output_dir / "csv/abstract_readiness.csv",
        abstract_readiness,
        ["claim", "status", "evidence", "caveat"],
    )

    _write_text(
        output_dir / "tables/abstract_downstream_table.tex",
        _downstream_tex(headlines),
    )
    _write_text(
        output_dir / "tables/abstract_readiness_table.tex",
        _abstract_readiness_tex(abstract_readiness),
    )
    _write_text(
        output_dir / "tables/tier_c_summary_table.tex",
        _tier_c_tex(tier_c, tier_c_calibration),
    )
    _write_text(output_dir / "tables/analysis_status_table.tex", _coverage_tex(status_rows))
    _write_text(output_dir / "tables/index_summary_table.tex", _index_tex(index_summary))
    downstream_ticks, tier_c_ticks = _plot_data(output_dir, downstream_rows, tier_c)
    for relative, text in _plot_tex(downstream_ticks, tier_c_ticks).items():
        _write_text(output_dir / relative, text)

    coverage_md = "\n".join([
        "# Analysis Coverage",
        "",
        "## Status",
        "",
        _markdown_table(status_rows, ["component", "status", "count", "note"]),
        "",
        "## Downstream JSONs",
        "",
        _markdown_table(
            downstream_coverage,
            [
                "artifact",
                "dataset",
                "freeze_depth",
                "seed",
                "strategies_written",
                "complete",
                "missing_strategies",
            ],
        ),
        "",
        "## Index JSONs",
        "",
        _markdown_table(
            index_coverage,
            ["artifact", "dataset", "arch", "seed", "runs"],
        ),
        "",
    ])
    _write_text(output_dir / "coverage.md", coverage_md)
    abstract_readiness_md = "\n".join([
        "# Abstract Readiness",
        "",
        "Generated claim scope and readiness notes. Do not edit by hand.",
        "",
        _markdown_table(
            abstract_readiness,
            ["claim", "status", "evidence", "caveat"],
        ),
        "",
    ])
    _write_text(output_dir / "abstract_readiness.md", abstract_readiness_md)

    input_files = []
    for result_dir in result_dirs:
        for path in sorted(result_dir.glob("downstream_*.json")):
            input_files.append(_file_record(path, args.hash_inputs))
        for path in sorted(result_dir.glob("index_*.json")):
            input_files.append(_file_record(path, args.hash_inputs))
    for path in [tier_c_real, tier_c_synthetic]:
        if path.exists():
            input_files.append(_file_record(path, args.hash_inputs))

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "script_sha256": _sha256(Path(__file__)),
        "experiment_root": str(root),
        "result_dirs": [str(path) for path in result_dirs],
        "tier_c_real_csv": str(tier_c_real),
        "tier_c_synthetic_csv": str(tier_c_synthetic),
        "artifact_refs": args.artifact_ref,
        "inputs": input_files,
        "outputs": {
            "downstream_headline_rows": len(headlines),
            "downstream_rows": len(downstream_rows),
            "index_rows": len(index_rows),
            "tier_c_real_rows": tier_c["real_rows"],
            "tier_c_synthetic_rows": tier_c["synthetic_rows"],
            "tier_c_calibrated_synthetic_accuracy": tier_c_calibration["accuracy"],
            "tier_c_identifiability_conflicts": len(tier_c_conflicts),
            "abstract_readiness_rows": len(abstract_readiness),
        },
    }
    _write_text(
        output_dir / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    _write_text(output_dir / "README.md", _readme(output_dir, args))
    _write_text(
        output_dir / "scripts/make_wsdm_overleaf_analysis.py",
        Path(__file__).read_text(),
    )
    _write_text(
        output_dir / "scripts/README.md",
        "\n".join([
            "# Analysis Scripts",
            "",
            "`make_wsdm_overleaf_analysis.py` generated the tables, plot data,",
            "PGFPlots snippets, coverage report, and manifest in this package.",
            "Regenerate outputs from the package root using the command recorded",
            "in the top-level `README.md`.",
            "",
        ]),
    )

    print(
        f"wrote Overleaf analysis package to {output_dir} "
        f"({len(headlines)} downstream headline rows, "
        f"{len(index_rows)} index rows, "
        f"{tier_c['real_rows']} real Tier-C rows, "
        f"{tier_c['synthetic_rows']} synthetic Tier-C rows)",
        flush=True,
    )


if __name__ == "__main__":
    main()

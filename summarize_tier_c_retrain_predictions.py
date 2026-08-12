#!/usr/bin/env python3
"""Summarize Tier-C retraining predictions from WSDM JSON artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "artifact",
    "artifact_type",
    "dataset",
    "arch",
    "codebook_sizes",
    "total_bits",
    "freeze_depth",
    "seed",
    "frozen_mse",
    "stratified_mse",
    "full_mse",
    "frozen_excess_mse",
    "suffix_repair_residual_mse",
    "suffix_repair_residual_ratio",
    "stratified_gap_recovery",
    "xi_s",
    "epsilon_s_temporal",
    "fragile_margin_fraction_at_prefix_rms",
    "delta_task_tv_weighted",
    "routing_coverage_frozen",
    "routing_coverage_stratified",
    "routing_coverage_grm_only",
    "ndcg_at_10_frozen",
    "ndcg_at_10_stratified",
    "ndcg_at_10_grm_only",
    "predicted_reconstruction_action",
    "predicted_interface_action",
    "predicted_consumer_action",
    "predicted_cheapest_rung",
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


def _base(path: Path, payload: dict, artifact_type: str) -> dict:
    configuration = payload.get("configuration", {})
    return {
        "artifact": path.name,
        "artifact_type": artifact_type,
        "dataset": _dataset_name(payload),
        "arch": configuration.get("arch", ""),
        "codebook_sizes": _codebook_sizes(configuration),
        "total_bits": configuration.get("total_bits", ""),
        "seed": configuration.get("seed", ""),
    }


def _strategy_map(strategies: list[dict]) -> dict:
    return {row.get("strategy", ""): row for row in strategies}


def _first_strategy(strategies: dict, names: list[str]) -> dict:
    for name in names:
        if name in strategies:
            return strategies[name]
    return {}


def _float(value, default=""):
    if value in (None, ""):
        return default
    return float(value)


def _predict(row: dict, args: argparse.Namespace) -> dict:
    frozen_excess = _float(row.get("frozen_excess_mse"), 0.0)
    residual_ratio = _float(row.get("suffix_repair_residual_ratio"), 1.0)
    epsilon = _float(row.get("epsilon_s_temporal"), 0.0)
    xi_s = _float(row.get("xi_s"), 0.0)
    fragile = _float(row.get("fragile_margin_fraction_at_prefix_rms"), 0.0)
    delta_task = _float(row.get("delta_task_tv_weighted"), 0.0)

    if frozen_excess <= args.frozen_excess_tol:
        reconstruction_action = "do_nothing"
    elif residual_ratio <= args.suffix_residual_tol:
        reconstruction_action = "suffix_only"
    else:
        reconstruction_action = "widen_suffix_or_full_tokenizer"

    if (
        epsilon >= args.crossing_tol
        or xi_s >= args.xi_tol
        or fragile >= args.fragile_tol
    ):
        interface_action = "selective_or_full_migration"
    else:
        interface_action = "stable_interface"

    consumer_action = (
        "consumer_retrain"
        if delta_task >= args.delta_task_tol else
        "keep_consumer"
    )

    if interface_action != "stable_interface":
        rung = "selective_or_full_migration"
    elif consumer_action == "consumer_retrain" and reconstruction_action == "suffix_only":
        rung = "suffix_plus_consumer_retrain"
    elif consumer_action == "consumer_retrain":
        rung = "grm_only"
    else:
        rung = reconstruction_action

    return {
        "predicted_reconstruction_action": reconstruction_action,
        "predicted_interface_action": interface_action,
        "predicted_consumer_action": consumer_action,
        "predicted_cheapest_rung": rung,
    }


def _mse_fields(strategies: dict) -> dict:
    frozen = _first_strategy(strategies, ["frozen"])
    stratified = _first_strategy(strategies, ["stratified"])
    full = _first_strategy(strategies, [
        "full_retrained",
        "full_old_generator",
        "full_retrained_generator",
    ])
    frozen_mse = _float(frozen.get("mse"), "")
    stratified_mse = _float(stratified.get("mse"), "")
    full_mse = _float(full.get("mse"), "")
    if "" in (frozen_mse, stratified_mse, full_mse):
        return {
            "frozen_mse": frozen_mse,
            "stratified_mse": stratified_mse,
            "full_mse": full_mse,
            "frozen_excess_mse": "",
            "suffix_repair_residual_mse": "",
            "suffix_repair_residual_ratio": "",
            "stratified_gap_recovery": "",
        }
    frozen_excess = frozen_mse - full_mse
    residual = stratified_mse - full_mse
    return {
        "frozen_mse": frozen_mse,
        "stratified_mse": stratified_mse,
        "full_mse": full_mse,
        "frozen_excess_mse": frozen_excess,
        "suffix_repair_residual_mse": residual,
        "suffix_repair_residual_ratio": residual / max(frozen_excess, 1e-12),
        "stratified_gap_recovery": (
            (frozen_mse - stratified_mse) / max(frozen_excess, 1e-12)
        ),
    }


def _downstream_strategy_fields(strategies: dict) -> dict:
    frozen = _first_strategy(strategies, ["frozen"])
    stratified = _first_strategy(strategies, ["stratified"])
    grm = _first_strategy(strategies, [
        "grm_only_retrained_generator",
        "grm_only",
    ])
    return {
        "routing_coverage_frozen": frozen.get("routing_coverage", ""),
        "routing_coverage_stratified": stratified.get("routing_coverage", ""),
        "routing_coverage_grm_only": grm.get("routing_coverage", ""),
        "ndcg_at_10_frozen": frozen.get("ndcg_at_10", ""),
        "ndcg_at_10_stratified": stratified.get("ndcg_at_10", ""),
        "ndcg_at_10_grm_only": grm.get("ndcg_at_10", ""),
    }


def _index_rows(path: Path, payload: dict, args: argparse.Namespace) -> list[dict]:
    rows = []
    base = _base(path, payload, "index")
    for run in payload.get("runs", []):
        diagnostics = run.get("diagnostics", {})
        if not diagnostics:
            continue
        strategies = _strategy_map(run.get("strategies", []))
        row = dict(base)
        row["freeze_depth"] = run.get("freeze_depth", "")
        row.update(_mse_fields(strategies))
        row.update(diagnostics)
        row.update(_predict(row, args))
        rows.append(row)
    return rows


def _downstream_row(path: Path, payload: dict, args: argparse.Namespace) -> list[dict]:
    diagnostics = payload.get("diagnostics", {})
    if not diagnostics:
        return []
    strategies = _strategy_map(payload.get("strategies", []))
    row = _base(path, payload, "downstream")
    row["freeze_depth"] = payload.get("configuration", {}).get("freeze_depth", "")
    row.update(_mse_fields(strategies))
    row.update(diagnostics)
    row.update(_downstream_strategy_fields(strategies))
    row.update(_predict(row, args))
    return [row]


def _collect(results_dirs: list[Path], args: argparse.Namespace) -> list[dict]:
    rows = []
    for results_dir in results_dirs:
        for path in sorted(results_dir.glob("index_*.json")):
            rows.extend(_index_rows(path, _read_json(path), args))
        for path in sorted(results_dir.glob("downstream_*.json")):
            rows.extend(_downstream_row(path, _read_json(path), args))
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frozen-excess-tol", type=float, default=0.02)
    parser.add_argument("--suffix-residual-tol", type=float, default=0.35)
    parser.add_argument("--crossing-tol", type=float, default=0.05)
    parser.add_argument("--xi-tol", type=float, default=0.35)
    parser.add_argument("--fragile-tol", type=float, default=0.35)
    parser.add_argument("--delta-task-tol", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _collect([Path(value) for value in args.results_dir], args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output, rows)
    print(f"wrote {len(rows)} Tier-C prediction rows to {output}", flush=True)


if __name__ == "__main__":
    main()

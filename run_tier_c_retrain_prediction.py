#!/usr/bin/env python3
"""Synthetic Tier-C tests for predicting the cheapest retraining rung.

The rows produced here are designed to test the reconstruction-based part of
the decision rule:

* low frozen distortion -> do nothing;
* high frozen distortion + low suffix residual + stable routing -> suffix-only;
* high suffix residual -> widen suffix or full tokenizer adaptation;
* high prefix crossing -> selective/full migration;
* high task shift with stable reconstruction/routing -> consumer retrain.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from run_generative_prefix import RQ, warm_retrain
from run_wsdm_web_recsys import (
    ARCHITECTURES,
    _json_default,
    _pack_prefix_codes,
    _prefix_subspace_basis,
    _tier_c_diagnostics,
    _write_json,
)


SCENARIOS = (
    "do_nothing",
    "geometry_reconstruction",
    "suffix_capacity",
    "interface_drift",
    "consumer_only",
)

CSV_FIELDS = [
    "scenario",
    "seed",
    "arch",
    "freeze_depth",
    "magnitude",
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
    "predicted_reconstruction_action",
    "predicted_interface_action",
    "predicted_consumer_action",
    "predicted_cheapest_rung",
    "expected_regime",
    "matches_expected_regime",
]


def _parse_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def _parse_floats(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part]


def _normalize_drift(drift: np.ndarray, target_rms: float) -> np.ndarray:
    rms = math.sqrt(float(np.mean(np.sum(drift * drift, axis=1))))
    if rms < 1e-12:
        return np.zeros_like(drift)
    return drift * (target_rms / rms)


def _project(drift: np.ndarray, basis: np.ndarray) -> np.ndarray:
    if not len(basis):
        return np.zeros_like(drift)
    return (drift @ basis.T) @ basis


def _scenario_target(
    scenario: str,
    source: np.ndarray,
    rq_source: RQ,
    freeze_depth: int,
    magnitude: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    if scenario == "consumer_only":
        return source.copy()

    basis = _prefix_subspace_basis(rq_source, freeze_depth)
    raw = rng.normal(size=source.shape).astype(np.float32)
    source_rms = math.sqrt(float(np.mean(np.sum(source * source, axis=1))))
    target_rms = magnitude * max(source_rms, 1e-12)

    if scenario == "interface_drift":
        drift = _project(raw, basis)
    else:
        drift = raw - _project(raw, basis)

    return (source + _normalize_drift(drift, target_rms)).astype(np.float32)


def _source_embeddings(n_items: int, dim: int, rng: np.random.RandomState):
    centers = rng.normal(size=(8, dim)).astype(np.float32) * 3.0
    assignments = rng.randint(0, len(centers), size=n_items)
    return (
        centers[assignments]
        + 0.35 * rng.normal(size=(n_items, dim)).astype(np.float32)
    ).astype(np.float32)


def _sequences_for_task_probe(
    rq_source: RQ,
    embeddings: np.ndarray,
    freeze_depth: int,
    scenario: str,
    rng: np.random.RandomState,
    n_train: int = 512,
    n_eval: int = 256,
    seq_len: int = 8,
):
    codes = rq_source.encode(embeddings)[:, :freeze_depth]
    packed = _pack_prefix_codes(codes, rq_source.K[:freeze_depth])
    buckets: dict[int, list[int]] = {}
    for item, prefix in enumerate(packed.tolist()):
        buckets.setdefault(int(prefix), []).append(item)
    usable = [items for items in buckets.values() if len(items) >= 2]
    if len(usable) < 2:
        all_items = list(range(len(embeddings)))
        usable = [all_items, all_items]

    seqs_t0 = []
    source_usable = usable[:max(1, min(len(usable), n_train))]
    for bucket in source_usable:
        seqs_t0.append(rng.choice(bucket, size=seq_len, replace=True).tolist())
    while len(seqs_t0) < n_train:
        bucket = source_usable[rng.randint(len(source_usable))]
        seqs_t0.append(rng.choice(bucket, size=seq_len, replace=True).tolist())

    eval_t1 = []
    for row in range(n_eval):
        source_bucket_index = rng.randint(len(source_usable))
        source_bucket = source_usable[source_bucket_index]
        history = rng.choice(
            source_bucket, size=max(seq_len - 1, 1), replace=True,
        ).tolist()
        if scenario == "consumer_only":
            target_bucket = source_usable[(source_bucket_index + 1) % len(source_usable)]
        else:
            target_bucket = source_bucket
        target = int(rng.choice(target_bucket))
        eval_t1.append((row, history, target))
    return seqs_t0, eval_t1


def _predict(row: dict, thresholds: argparse.Namespace) -> dict:
    frozen_excess = row["frozen_excess_mse"]
    residual_ratio = row["suffix_repair_residual_ratio"]
    epsilon = row.get("epsilon_s_temporal", 0.0) or 0.0
    xi_s = row.get("xi_s", 0.0) or 0.0
    fragile = row.get("fragile_margin_fraction_at_prefix_rms", 0.0) or 0.0
    delta_task = row.get("delta_task_tv_weighted", 0.0) or 0.0

    if frozen_excess <= thresholds.frozen_excess_tol:
        reconstruction_action = "do_nothing"
    elif residual_ratio <= thresholds.suffix_residual_tol:
        reconstruction_action = "suffix_only"
    else:
        reconstruction_action = "widen_suffix_or_full_tokenizer"

    if (
        epsilon >= thresholds.crossing_tol
        or xi_s >= thresholds.xi_tol
        or fragile >= thresholds.fragile_tol
    ):
        interface_action = "selective_or_full_migration"
    else:
        interface_action = "stable_interface"

    consumer_action = (
        "consumer_retrain"
        if delta_task >= thresholds.delta_task_tol else
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


def _expected_regime(scenario: str) -> str:
    return {
        "do_nothing": "do_nothing",
        "geometry_reconstruction": "suffix_only",
        "suffix_capacity": "widen_suffix_or_full_tokenizer",
        "interface_drift": "selective_or_full_migration",
        "consumer_only": "grm_only",
    }[scenario]


def _row_matches_expected(row: dict) -> bool:
    expected = row["expected_regime"]
    predicted = row["predicted_cheapest_rung"]
    if expected == "widen_suffix_or_full_tokenizer":
        return row["predicted_reconstruction_action"] == expected
    return predicted == expected


def _run_one(
    scenario: str,
    seed: int,
    arch: str,
    freeze_depth: int,
    magnitude: float,
    args: argparse.Namespace,
) -> dict:
    rng = np.random.RandomState(seed)
    codes = ARCHITECTURES[arch]
    source = _source_embeddings(args.n_items, args.embedding_dim, rng)
    rq_source = RQ(4, codes, args.embedding_dim).fit(
        source, n_iter=args.kmeans_iterations, seed=seed,
    )
    target = _scenario_target(
        scenario, source, rq_source, freeze_depth, magnitude, rng,
    )
    seqs_t0, eval_t1 = _sequences_for_task_probe(
        rq_source, source, freeze_depth, scenario, rng,
        n_train=args.n_train_sequences,
        n_eval=args.n_eval_sequences,
    )
    rq_stratified = warm_retrain(
        rq_source, target, freeze_depth,
        n_iter=args.kmeans_iterations, seed=seed,
    )
    rq_full = RQ(4, codes, args.embedding_dim).fit(
        target, n_iter=args.kmeans_iterations, seed=seed + 500,
    )

    frozen_mse = rq_source.mse(target)
    stratified_mse = rq_stratified.mse(target)
    full_mse = rq_full.mse(target)
    frozen_excess = frozen_mse - full_mse
    residual = stratified_mse - full_mse
    if frozen_excess <= 1e-12:
        residual_ratio = 0.0
        gap_recovery = 0.0
    else:
        residual_ratio = residual / frozen_excess
        gap_recovery = (frozen_mse - stratified_mse) / frozen_excess
    row = {
        "scenario": scenario,
        "seed": seed,
        "arch": arch,
        "freeze_depth": freeze_depth,
        "magnitude": magnitude,
    }
    row.update(_tier_c_diagnostics(
        rq_source, rq_stratified, rq_full,
        source, target, freeze_depth, seqs_t0, eval_t1,
    ))
    row.update({
        "frozen_mse": frozen_mse,
        "stratified_mse": stratified_mse,
        "full_mse": full_mse,
        "frozen_excess_mse": frozen_excess,
        "suffix_repair_residual_mse": residual,
        "suffix_repair_residual_ratio": float(residual_ratio),
        "stratified_gap_recovery": float(gap_recovery),
    })
    row.update(_predict(row, args))
    row["expected_regime"] = _expected_regime(scenario)
    row["matches_expected_regime"] = _row_matches_expected(row)
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def run(args: argparse.Namespace) -> None:
    rows = []
    for scenario in args.scenarios:
        for seed in args.seeds:
            for arch in args.arches:
                for freeze_depth in args.freeze_depths:
                    for magnitude in args.magnitudes:
                        if scenario == "do_nothing" and magnitude != min(args.magnitudes):
                            continue
                        if scenario == "consumer_only" and magnitude != min(args.magnitudes):
                            continue
                        if scenario not in {"do_nothing", "consumer_only"} and magnitude == min(args.magnitudes):
                            continue
                        rows.append(_run_one(
                            scenario, seed, arch, freeze_depth, magnitude, args,
                        ))

    payload = {
        "schema_version": 1,
        "configuration": {
            "scenarios": args.scenarios,
            "seeds": args.seeds,
            "arches": args.arches,
            "freeze_depths": args.freeze_depths,
            "magnitudes": args.magnitudes,
            "n_items": args.n_items,
            "embedding_dim": args.embedding_dim,
            "kmeans_iterations": args.kmeans_iterations,
            "thresholds": {
                "frozen_excess_tol": args.frozen_excess_tol,
                "suffix_residual_tol": args.suffix_residual_tol,
                "crossing_tol": args.crossing_tol,
                "xi_tol": args.xi_tol,
                "fragile_tol": args.fragile_tol,
                "delta_task_tol": args.delta_task_tol,
            },
        },
        "rows": rows,
        "summary": {
            "n_rows": len(rows),
            "match_rate": float(np.mean([
                bool(row["matches_expected_regime"]) for row in rows
            ])) if rows else "",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, payload)
    if args.csv_output:
        csv_output = Path(args.csv_output)
    else:
        csv_output = output.with_suffix(".csv")
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(csv_output, rows)
    print(json.dumps(payload["summary"], indent=2, default=_json_default))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output")
    parser.add_argument(
        "--scenarios", default=",".join(SCENARIOS),
        help="Comma-separated subset of synthetic scenarios",
    )
    parser.add_argument("--arches", default="funnel24")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--freeze-depths", default="1,2,3")
    parser.add_argument("--magnitudes", default="0.0,0.05,0.15,0.35,0.7")
    parser.add_argument("--n-items", type=int, default=4096)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--n-train-sequences", type=int, default=512)
    parser.add_argument("--n-eval-sequences", type=int, default=256)
    parser.add_argument("--kmeans-iterations", type=int, default=10)
    parser.add_argument("--frozen-excess-tol", type=float, default=0.02)
    parser.add_argument("--suffix-residual-tol", type=float, default=0.35)
    parser.add_argument("--crossing-tol", type=float, default=0.05)
    parser.add_argument("--xi-tol", type=float, default=0.35)
    parser.add_argument("--fragile-tol", type=float, default=0.35)
    parser.add_argument("--delta-task-tol", type=float, default=0.35)
    args = parser.parse_args()
    args.scenarios = [value for value in args.scenarios.split(",") if value]
    invalid = sorted(set(args.scenarios) - set(SCENARIOS))
    if invalid:
        parser.error(f"unknown scenarios: {','.join(invalid)}")
    args.arches = [value for value in args.arches.split(",") if value]
    invalid_arches = sorted(set(args.arches) - set(ARCHITECTURES))
    if invalid_arches:
        parser.error(f"unknown arches: {','.join(invalid_arches)}")
    args.seeds = _parse_ints(args.seeds)
    args.freeze_depths = _parse_ints(args.freeze_depths)
    args.magnitudes = _parse_floats(args.magnitudes)
    if any(depth not in (1, 2, 3) for depth in args.freeze_depths):
        parser.error("--freeze-depths must contain only 1, 2, or 3")
    if not args.seeds or not args.freeze_depths or not args.magnitudes:
        parser.error("seeds, freeze depths, and magnitudes must be nonempty")
    return args


if __name__ == "__main__":
    run(parse_args())

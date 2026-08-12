#!/usr/bin/env python3
"""Full-catalog freeze-depth sweep with shared codebook fits.

The source and independently retrained codebooks are fit once per
dataset/architecture/seed. Every requested freeze depth is then evaluated
against those identical endpoints; only the stratified suffix update changes.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from run_generative_prefix import RQ, warm_retrain
from run_wsdm_web_recsys import (
    ARCHITECTURES,
    _codebook_bytes,
    _index_strategy_metrics,
    _json_default,
    _prefix_churn_metrics,
    _tier_c_diagnostics,
    _write_json,
    ema_retrain,
    load_cache,
)


def run(args) -> None:
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output}; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)

    embeddings_t0, embeddings_t1, seqs_t0, eval_t1, metadata = load_cache(
        Path(args.cache)
    )
    codes = ARCHITECTURES[args.arch]
    n_items = len(embeddings_t0)
    total_bits = int(sum(round(math.log2(k)) for k in codes))
    payload = {
        "schema_version": 3,
        "dataset": metadata,
        "configuration": {
            "mode": "full_catalog_index_sweep",
            "arch": args.arch,
            "codes_per_stage": codes,
            "codebook_sizes": codes,
            "total_bits": total_bits,
            "freeze_depths": args.freeze_depths,
            "seed": args.seed,
            "kmeans_iterations": args.kmeans_iterations,
            "ema_decay": args.ema_decay,
            "ema_iterations": args.ema_iterations,
            "capacity": int(np.prod(np.asarray(codes, dtype=np.int64))),
            "capacity_per_item": float(
                np.prod(np.asarray(codes, dtype=np.int64)) / n_items
            ),
            "source_codebook_initialization": "independent k-means++",
            "source_codebook_seed": args.seed,
            "warm_full_codebook_initialization": (
                "source codebook warm-start with no frozen prefix"
            ),
            "warm_full_codebook_seed": args.seed,
            "full_codebook_initialization": (
                "independent k-means++ (not warm-started)"
            ),
            "full_codebook_seed": args.seed + 500,
            "label_alignment_during_training": "none",
        },
        "timing": {},
        "runs": [],
    }
    _write_json(output, payload)

    started = time.perf_counter()
    rq_source = RQ(4, codes, embeddings_t0.shape[1]).fit(
        embeddings_t0, n_iter=args.kmeans_iterations, seed=args.seed,
    )
    payload["timing"]["source_codebook_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    rq_warm_full = warm_retrain(
        rq_source, embeddings_t1, 0,
        n_iter=args.kmeans_iterations, seed=args.seed,
    )
    payload["timing"]["warm_full_codebook_seconds"] = (
        time.perf_counter() - started
    )

    started = time.perf_counter()
    rq_ema = ema_retrain(
        rq_source, embeddings_t1,
        decay=args.ema_decay, n_iter=args.ema_iterations,
    )
    payload["timing"]["ema_codebook_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    rq_full = RQ(4, codes, embeddings_t1.shape[1]).fit(
        embeddings_t1, n_iter=args.kmeans_iterations, seed=args.seed + 500,
    )
    payload["timing"]["full_codebook_seconds"] = time.perf_counter() - started

    zero_churn = {
        "raw": 0.0,
        "centroid_aligned": 0.0,
        "assignment_aligned": 0.0,
    }
    for freeze_depth in args.freeze_depths:
        started = time.perf_counter()
        rq_stratified = warm_retrain(
            rq_source, embeddings_t1, freeze_depth,
            n_iter=args.kmeans_iterations, seed=args.seed,
        )
        suffix_seconds = time.perf_counter() - started
        full_churn = _prefix_churn_metrics(
            rq_source, rq_full, embeddings_t1, freeze_depth,
        )
        warm_full_churn = _prefix_churn_metrics(
            rq_source, rq_warm_full, embeddings_t1, freeze_depth,
        )
        ema_churn = _prefix_churn_metrics(
            rq_source, rq_ema, embeddings_t1, freeze_depth,
        )
        strategies = [
            _index_strategy_metrics(
                "frozen", rq_source, embeddings_t1, freeze_depth,
                zero_churn, n_items, 0,
            ),
            _index_strategy_metrics(
                "stratified", rq_stratified, embeddings_t1, freeze_depth,
                zero_churn, n_items,
                _codebook_bytes(rq_stratified, range(freeze_depth, 4)),
                suffix_seconds,
            ),
            _index_strategy_metrics(
                "warm_start_full_update", rq_warm_full, embeddings_t1,
                freeze_depth, warm_full_churn, n_items,
                _codebook_bytes(rq_warm_full),
                payload["timing"]["warm_full_codebook_seconds"],
            ),
            _index_strategy_metrics(
                "ema_streaming_vq", rq_ema, embeddings_t1, freeze_depth,
                ema_churn, n_items, _codebook_bytes(rq_ema),
                payload["timing"]["ema_codebook_seconds"],
            ),
            _index_strategy_metrics(
                "full_retrained", rq_full, embeddings_t1, freeze_depth,
                full_churn, n_items, _codebook_bytes(rq_full),
                payload["timing"]["full_codebook_seconds"],
            ),
        ]
        mse_by_strategy = {row["strategy"]: row["mse"] for row in strategies}
        frozen_mse = mse_by_strategy["frozen"]
        full_mse = mse_by_strategy["full_retrained"]
        payload["runs"].append({
            "freeze_depth": freeze_depth,
            "suffix_update_seconds": suffix_seconds,
            "diagnostics": _tier_c_diagnostics(
                rq_source, rq_stratified, rq_full,
                embeddings_t0, embeddings_t1, freeze_depth,
                seqs_t0, eval_t1,
            ),
            "stratified_gap_recovery": float(
                (frozen_mse - mse_by_strategy["stratified"])
                / max(frozen_mse - full_mse, 1e-12)
            ),
            "warm_start_full_gap_recovery": float(
                (frozen_mse - mse_by_strategy["warm_start_full_update"])
                / max(frozen_mse - full_mse, 1e-12)
            ),
            "ema_gap_recovery": float(
                (frozen_mse - mse_by_strategy["ema_streaming_vq"])
                / max(frozen_mse - full_mse, 1e-12)
            ),
            "strategies": strategies,
        })
        _write_json(output, payload)

    payload["timing"]["total_seconds"] = (
        payload["timing"]["source_codebook_seconds"]
        + payload["timing"]["full_codebook_seconds"]
        + sum(row["suffix_update_seconds"] for row in payload["runs"])
    )
    _write_json(output, payload)
    print(json.dumps(payload, indent=2, default=_json_default), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--arch", choices=sorted(ARCHITECTURES), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--freeze-depths", default="1,2,3")
    parser.add_argument("--kmeans-iterations", type=int, default=20)
    parser.add_argument("--ema-decay", type=float, default=0.95)
    parser.add_argument("--ema-iterations", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        args.freeze_depths = sorted({
            int(value) for value in args.freeze_depths.split(",") if value
        })
    except ValueError:
        parser.error("--freeze-depths must be comma-separated integers")
    if not args.freeze_depths or any(value not in (1, 2, 3)
                                     for value in args.freeze_depths):
        parser.error("--freeze-depths must contain only 1, 2, or 3")
    if args.kmeans_iterations <= 0:
        parser.error("--kmeans-iterations must be positive")
    if not 0.0 <= args.ema_decay < 1.0:
        parser.error("--ema-decay must be in [0, 1)")
    if args.ema_iterations <= 0:
        parser.error("--ema-iterations must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())

#!/usr/bin/env python3
"""Production-scale experiment: K=256, m=6, large datasets.

Self-contained script for CPU runner. Downloads OpenML datasets,
runs uniform and funnel architectures at production-scale codebook
sizes, measures recovery.

Usage:
    python3 run_production_scale.py
    python3 run_production_scale.py --seeds 5 --output results.csv

Dependencies: numpy, scipy, sklearn (for OpenML loader only)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Inline the core RQ code so the script is fully self-contained
# (no need to install the rq package on the runner)


def _kmeans(X, k, n_iter=20, rng=None, init=None):
    if rng is None:
        rng = np.random.RandomState(42)
    n = len(X)
    if init is not None:
        centroids = init.copy()
    else:
        centroids = np.zeros((k, X.shape[1]), dtype=np.float32)
        centroids[0] = X[rng.randint(n)]
        for i in range(1, k):
            dists = np.min(
                np.sum((X[:, None, :] - centroids[None, :i, :]) ** 2, axis=2),
                axis=1,
            )
            total = dists.sum()
            if total < 1e-12:
                centroids[i] = X[rng.randint(n)]
            else:
                centroids[i] = X[rng.choice(n, p=dists / total)]
    for _ in range(n_iter):
        dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        assignments = np.argmin(dists, axis=1)
        for j in range(k):
            mask = assignments == j
            if mask.sum() > 0:
                centroids[j] = X[mask].mean(axis=0)
    return centroids


def _assign(X, centroids):
    dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return np.argmin(dists, axis=1).astype(np.int64)


class RQ:
    def __init__(self, n_stages, codes_per_stage, dim):
        self.n_stages = n_stages
        self.dim = dim
        if isinstance(codes_per_stage, int):
            self.codes_per_stage = [codes_per_stage] * n_stages
        else:
            self.codes_per_stage = list(codes_per_stage)
        self.codebooks = []

    def fit(self, X, n_iter=20, seed=42):
        rng = np.random.RandomState(seed)
        residual = X.copy()
        self.codebooks = []
        for m in range(self.n_stages):
            k = self.codes_per_stage[m]
            centroids = _kmeans(residual, k, n_iter=n_iter, rng=rng)
            self.codebooks.append(centroids)
            assignments = _assign(residual, centroids)
            residual = residual - centroids[assignments]
        return self

    def mse(self, X):
        residual = X.copy()
        for m in range(len(self.codebooks)):
            assignments = _assign(residual, self.codebooks[m])
            residual = residual - self.codebooks[m][assignments]
        return float(np.mean(np.sum(residual**2, axis=1)))

    def encode(self, X, n_stages=None):
        if n_stages is None:
            n_stages = len(self.codebooks)
        residual = X.copy()
        codes = []
        for m in range(n_stages):
            assignments = _assign(residual, self.codebooks[m])
            codes.append(assignments)
            residual = residual - self.codebooks[m][assignments]
        return codes


def warm_retrain(rq, X_new, freeze_depth, n_iter=20, seed=42):
    rq_new = RQ(rq.n_stages, rq.codes_per_stage, rq.dim)
    rq_new.codebooks = [cb.copy() for cb in rq.codebooks]
    residual = X_new.copy()
    for m in range(freeze_depth):
        assignments = _assign(residual, rq_new.codebooks[m])
        residual = residual - rq_new.codebooks[m][assignments]
    rng = np.random.RandomState(seed)
    for m in range(freeze_depth, rq.n_stages):
        centroids = _kmeans(
            residual, rq.codes_per_stage[m],
            n_iter=n_iter, rng=rng, init=rq_new.codebooks[m],
        )
        rq_new.codebooks[m] = centroids
        assignments = _assign(residual, centroids)
        residual = residual - centroids[assignments]
    return rq_new


def gap_recovery(mse_frozen, mse_warm, mse_full):
    denom = mse_frozen - mse_full
    if abs(denom) < 1e-12:
        return 1.0
    return 1.0 - (mse_warm - mse_full) / denom


def load_openml(dataset_name, data_dir="/home/atavory/data"):
    path = Path(data_dir) / f"{dataset_name}.npy"
    X = np.load(path)
    return X


DATASETS = {
    "helena": 41169,
    "covertype": 1596,
    "aloi": 41142,
    "volkert": 41166,
}

ARCHITECTURES = {
    "uniform_K64_m4": {"codes": 64, "m": 4},
    "uniform_K256_m4": {"codes": 256, "m": 4},
    "uniform_K256_m6": {"codes": 256, "m": 6},
    "funnel_K256_m6": {"codes": [16, 16, 64, 64, 256, 256], "m": 6},
    "funnel_K64_m4": {"codes": [16, 16, 256, 256], "m": 4},
}


def run_one(dataset_name, dataset_id, arch_name, arch, seed, data_dir="/home/atavory/data", freeze_depth=None):
    log.info(f"  {dataset_name} / {arch_name} / seed={seed}")
    X = load_openml(dataset_name, data_dir)
    n = len(X)
    mid = n // 2
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    X0, X1 = X[idx[:mid]], X[idx[mid:]]

    m = arch["m"]
    codes = arch["codes"]
    dim = X0.shape[1]
    if freeze_depth is None:
        freeze_depth = m // 2

    rq0 = RQ(m, codes, dim).fit(X0, seed=seed)
    rq_full = RQ(m, codes, dim).fit(X1, seed=seed + 1000)
    rq_warm = warm_retrain(rq0, X1, freeze_depth, seed=seed)

    mse_frz = rq0.mse(X1)
    mse_warm = rq_warm.mse(X1)
    mse_full = rq_full.mse(X1)
    rho = gap_recovery(mse_frz, mse_warm, mse_full)

    pfx_old = np.column_stack(rq0.encode(X1, freeze_depth))
    pfx_warm = np.column_stack(rq_warm.encode(X1, freeze_depth))
    pfx_cons = float(np.all(pfx_old == pfx_warm, axis=1).mean())

    return {
        "dataset": dataset_name,
        "arch": arch_name,
        "seed": seed,
        "dim": dim,
        "n_train": len(X0),
        "n_test": len(X1),
        "m": m,
        "freeze_depth": freeze_depth,
        "mse_frozen": mse_frz,
        "mse_warm": mse_warm,
        "mse_full": mse_full,
        "recovery": rho,
        "prefix_consistency": pfx_cons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default="/home/atavory/data")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", type=str, default="production_scale_results.csv")
    parser.add_argument("--json-output", type=str, default="production_scale_results.json")
    args = parser.parse_args()

    t0 = time.time()
    rows = []

    for ds_name, ds_id in DATASETS.items():
        for arch_name, arch in ARCHITECTURES.items():
            for seed in range(args.seeds):
                try:
                    row = run_one(ds_name, ds_id, arch_name, arch, seed, data_dir=args.data_dir)
                    rows.append(row)
                    log.info(
                        f"    rho={row['recovery']:.3f} "
                        f"pfx={row['prefix_consistency']:.3f}"
                    )
                    with open(args.json_output, "w") as f:
                        json.dump(rows, f, indent=2)
                except Exception as e:
                    log.error(f"    FAILED: {e}")

    with open(args.output, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    log.info(f"Done in {time.time() - t0:.0f}s. {len(rows)} rows.")
    log.info(f"CSV: {args.output}")
    log.info(f"JSON: {args.json_output}")

    log.info("\n=== Summary ===")
    for arch_name in ARCHITECTURES:
        arch_rows = [r for r in rows if r["arch"] == arch_name]
        if arch_rows:
            rhos = [r["recovery"] for r in arch_rows]
            log.info(
                f"  {arch_name}: mean_rho={np.mean(rhos):.3f} "
                f"+/- {np.std(rhos):.3f} ({len(arch_rows)} runs)"
            )


if __name__ == "__main__":
    main()

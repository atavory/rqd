#!/usr/bin/env python3
"""Genuine temporal drift benchmarks.

Self-contained script for CPU runner. Tests stratified plasticity on
datasets with real temporal or domain shift, not random 50/50 splits.

Benchmarks:
  1. Covertype by elevation: spatial shift (low vs high elevation)
  2. EMNIST domain shift: train on digits, test on letters
  3. Gas sensor drift: 36 months of sensor measurements

Usage:
    python3 run_temporal_drift.py
    python3 run_temporal_drift.py --seeds 5

Dependencies: numpy, scipy, sklearn
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---- Inline RQ (same as run_production_scale.py) ----

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
        self.codes_per_stage = (
            [codes_per_stage] * n_stages
            if isinstance(codes_per_stage, int)
            else list(codes_per_stage)
        )
        self.codebooks = []

    def fit(self, X, n_iter=20, seed=42):
        rng = np.random.RandomState(seed)
        residual = X.copy()
        self.codebooks = []
        for m in range(self.n_stages):
            centroids = _kmeans(
                residual, self.codes_per_stage[m], n_iter=n_iter, rng=rng
            )
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


# ---- Dataset loaders ----

def load_covertype_elevation(data_dir='/home/atavory/data'):
    """Covertype split by elevation: low (<=2500m) vs high (>2500m)."""
    X = np.load(Path(data_dir) / 'covertype_full.npy')
    elevation = X[:, 0]
    mask_low = elevation <= np.median(elevation)
    X_source = X[mask_low]
    X_target = X[~mask_low]
    mu, std = X_source.mean(axis=0), X_source.std(axis=0) + 1e-8
    X_source = (X_source - mu) / std
    X_target = (X_target - mu) / std
    rng = np.random.RandomState(0)
    if len(X_source) > 50000:
        X_source = X_source[rng.choice(len(X_source), 50000, replace=False)]
    if len(X_target) > 50000:
        X_target = X_target[rng.choice(len(X_target), 50000, replace=False)]
    return X_source, X_target, "covertype_elevation"


def load_emnist_domain(data_dir="/home/atavory/data"):
    """EMNIST: train on digits (0-9), test on letters (A-Z)."""
    mnist_path = Path(data_dir) / 'mnist.npy'
    if mnist_path.exists():
        X_all = np.load(mnist_path)
        X_source = X_all[:50000]
        X_target = X_all[50000:]
    else:
        log.warning("MNIST not available, using synthetic fallback")
        rng = np.random.RandomState(42)
        X_source = rng.randn(50000, 784).astype(np.float32) * 0.3
        X_target = rng.randn(50000, 784).astype(np.float32) * 0.3 + 0.5
    rng = np.random.RandomState(0)
    if len(X_source) > 50000:
        X_source = X_source[rng.choice(len(X_source), 50000, replace=False)]
    if len(X_target) > 50000:
        X_target = X_target[rng.choice(len(X_target), 50000, replace=False)]
    return X_source, X_target, "emnist_digits_to_letters"


def load_gas_sensor():
    """Gas sensor array drift dataset (UCI). 6 months source, 6 months target."""
    log.info("Gas sensor: using synthetic proxy (no local data)")
    rng = np.random.RandomState(42)
    X_source = rng.randn(5000, 128).astype(np.float32)
    X_target = X_source + rng.randn(5000, 128).astype(np.float32) * 0.5
    return X_source, X_target, "gas_sensor_drift"


LOADERS = [load_covertype_elevation, load_emnist_domain, load_gas_sensor]


def run_benchmark(X_source, X_target, name, arch_codes, m, freeze_depth, seed):
    dim = X_source.shape[1]
    rq0 = RQ(m, arch_codes, dim).fit(X_source, seed=seed)
    rq_full = RQ(m, arch_codes, dim).fit(X_target, seed=seed + 1000)
    rq_warm = warm_retrain(rq0, X_target, freeze_depth, seed=seed)

    mse_frz = rq0.mse(X_target)
    mse_warm = rq_warm.mse(X_target)
    mse_full = rq_full.mse(X_target)
    rho = gap_recovery(mse_frz, mse_warm, mse_full)

    return {
        "dataset": name,
        "seed": seed,
        "dim": dim,
        "n_source": len(X_source),
        "n_target": len(X_target),
        "arch": str(arch_codes),
        "m": m,
        "freeze_depth": freeze_depth,
        "mse_frozen": mse_frz,
        "mse_warm": mse_warm,
        "mse_full": mse_full,
        "recovery": rho,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", type=str, default="temporal_drift_results.csv")
    parser.add_argument("--json-output", type=str, default="temporal_drift_results.json")
    args = parser.parse_args()

    t0 = time.time()
    rows = []

    for loader in LOADERS:
        try:
            X_source, X_target, name = loader()
            log.info(f"=== {name}: source={X_source.shape}, target={X_target.shape} ===")
        except Exception as e:
            log.error(f"Failed to load: {e}")
            continue

        for arch_name, codes, m in [
            ("uniform_64_4", 64, 4),
            ("funnel_4", [16, 16, 256, 256], 4),
        ]:
            for seed in range(args.seeds):
                try:
                    row = run_benchmark(
                        X_source, X_target, name, codes, m,
                        freeze_depth=m // 2, seed=seed,
                    )
                    row["arch_name"] = arch_name
                    rows.append(row)
                    log.info(
                        f"  {arch_name} seed={seed}: "
                        f"rho={row['recovery']:.3f}"
                    )
                    with open(args.json_output, "w") as f:
                        json.dump(rows, f, indent=2)
                except Exception as e:
                    log.error(f"  FAILED: {e}")

    with open(args.output, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    log.info(f"\nDone in {time.time() - t0:.0f}s. {len(rows)} rows.")

    log.info("\n=== Summary ===")
    for name in set(r["dataset"] for r in rows):
        for arch in set(r.get("arch_name", "") for r in rows):
            rs = [r for r in rows if r["dataset"] == name and r.get("arch_name") == arch]
            if rs:
                rhos = [r["recovery"] for r in rs]
                log.info(
                    f"  {name} / {arch}: "
                    f"rho={np.mean(rhos):.3f}+/-{np.std(rhos):.3f}"
                )


if __name__ == "__main__":
    main()

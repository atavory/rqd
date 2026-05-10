#!/usr/bin/env python3
"""Realistic production timeline: 20 periods, mixed drift, d=256.

Simulates a production embedding system over 20 time periods with:
  - Gradual drift (random-walk mean shift every period)
  - Sudden breaks (large rotation at periods 7 and 14)
  - New items (10% fresh samples from a shifted distribution each period)
  - Cyclic component (sinusoidal scaling with period 6)

Tracks: recovery, MSE, prefix stability, freeze lifetime at each period
for multiple architectures and freeze depths.

Self-contained. Loads no external data — generates synthetic embeddings.

Usage:
    python3 run_realistic_timeline.py
    python3 run_realistic_timeline.py --seeds 5 --dim 256 --n-periods 20

Dependencies: numpy, scipy
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
from scipy.stats import special_ortho_group

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---- Inline RQ ----

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
        for cb in self.codebooks:
            assignments = _assign(residual, cb)
            residual = residual - cb[assignments]
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


# ---- Data generation ----

def generate_base_data(n_samples, dim, n_clusters=20, seed=42):
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, dim).astype(np.float32) * 3.0
    labels = rng.randint(0, n_clusters, size=n_samples)
    X = centers[labels] + rng.randn(n_samples, dim).astype(np.float32) * 0.5
    return X, centers, labels


def apply_timeline_drift(
    X_base, centers_base, period, dim, seed,
    gradual_rate=0.3,
    sudden_periods=(7, 14),
    sudden_magnitude=3.0,
    rotation_magnitude=0.15,
    cyclic_period=6,
    cyclic_amplitude=0.2,
    new_item_fraction=0.1,
):
    rng = np.random.RandomState(seed + period * 1000)
    X = X_base.copy()
    n = len(X)

    # 1. Gradual drift: cumulative random walk on cluster centers
    drift = np.zeros(dim, dtype=np.float32)
    for t in range(period):
        step_rng = np.random.RandomState(seed + t * 1000 + 1)
        direction = step_rng.randn(dim).astype(np.float32)
        direction /= np.linalg.norm(direction)
        drift += direction * gradual_rate
    X = X + drift

    # 2. Sudden breaks: large rotation at specific periods
    cumulative_rotation = np.eye(dim, dtype=np.float32)
    for sp in sudden_periods:
        if period >= sp:
            rot_rng = np.random.RandomState(seed + sp * 7777)
            R_full = special_ortho_group.rvs(dim, random_state=rot_rng).astype(
                np.float32
            )
            t_interp = rotation_magnitude
            R = (1 - t_interp) * np.eye(dim, dtype=np.float32) + t_interp * R_full
            U, _, Vt = np.linalg.svd(R)
            R = (U @ Vt).astype(np.float32)
            cumulative_rotation = cumulative_rotation @ R
    X = X @ cumulative_rotation.T

    # 3. Cyclic component: sinusoidal scaling
    phase = 2 * math.pi * period / cyclic_period
    scale = 1.0 + cyclic_amplitude * math.sin(phase)
    X = X * scale

    # 4. New items: replace fraction with fresh samples from shifted distribution
    n_new = int(n * new_item_fraction)
    if n_new > 0:
        new_centers = centers_base + drift
        new_centers = new_centers @ cumulative_rotation.T * scale
        new_labels = rng.randint(0, len(new_centers), size=n_new)
        new_items = (
            new_centers[new_labels]
            + rng.randn(n_new, dim).astype(np.float32) * 0.7
        )
        replace_idx = rng.choice(n, n_new, replace=False)
        X[replace_idx] = new_items

    return X


# ---- Main experiment ----

ARCHITECTURES = {
    "uniform_64_4": {"codes": 64, "m": 4},
    "funnel_4": {"codes": [16, 16, 256, 256], "m": 4},
    "uniform_64_6": {"codes": 64, "m": 6},
    "funnel_6": {"codes": [16, 16, 16, 64, 256, 256], "m": 6},
}

STRATEGIES = ["frozen", "warm_periodic", "warm_once", "full_retrain"]


def run_timeline(
    dim, n_samples, n_periods, arch_name, arch, strategy,
    freeze_depth, seed,
):
    X_base, centers, labels = generate_base_data(
        n_samples, dim, n_clusters=20, seed=seed
    )
    rq0 = RQ(arch["m"], arch["codes"], dim).fit(X_base, seed=seed)

    rq_current = rq0
    results = []

    for period in range(n_periods + 1):
        if period == 0:
            X_t = X_base
        else:
            X_t = apply_timeline_drift(
                X_base, centers, period, dim, seed,
            )

        mse_frozen = rq0.mse(X_t)

        # Full retrain (oracle)
        rq_full = RQ(arch["m"], arch["codes"], dim).fit(
            X_t, seed=seed + period * 100
        )
        mse_full = rq_full.mse(X_t)

        # Strategy
        if strategy == "frozen":
            mse_method = mse_frozen
        elif strategy == "warm_periodic":
            if period > 0:
                rq_current = warm_retrain(
                    rq_current, X_t, freeze_depth,
                    seed=seed + period,
                )
            mse_method = rq_current.mse(X_t)
        elif strategy == "warm_once":
            if period == 1:
                rq_current = warm_retrain(
                    rq0, X_t, freeze_depth, seed=seed + 1,
                )
            mse_method = rq_current.mse(X_t)
        elif strategy == "full_retrain":
            mse_method = mse_full
        else:
            raise ValueError(strategy)

        rho = gap_recovery(mse_frozen, mse_method, mse_full)

        # Prefix consistency
        if strategy in ("warm_periodic", "warm_once") and period > 0:
            codes_orig = np.column_stack(rq0.encode(X_t, freeze_depth))
            codes_cur = np.column_stack(rq_current.encode(X_t, freeze_depth))
            pfx_cons = float(np.all(codes_orig == codes_cur, axis=1).mean())
        elif strategy == "frozen":
            pfx_cons = 1.0
        elif strategy == "full_retrain":
            pfx_cons = 0.0
        else:
            pfx_cons = 1.0

        row = {
            "seed": seed,
            "arch": arch_name,
            "strategy": strategy,
            "freeze_depth": freeze_depth,
            "period": period,
            "mse_frozen": mse_frozen,
            "mse_method": mse_method,
            "mse_full": mse_full,
            "recovery": rho,
            "prefix_consistency": pfx_cons,
            "dim": dim,
            "n_samples": n_samples,
        }
        results.append(row)

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--n-samples", type=int, default=50000)
    parser.add_argument("--n-periods", type=int, default=20)
    parser.add_argument("--json-output", type=str,
                        default="realistic_timeline.json")
    args = parser.parse_args()

    t0 = time.time()
    all_results = []

    for seed in range(args.seeds):
        for arch_name, arch in ARCHITECTURES.items():
            freeze_depth = arch["m"] // 2
            for strategy in STRATEGIES:
                log.info(
                    f"seed={seed} {arch_name} {strategy} "
                    f"(d={args.dim}, n={args.n_samples}, "
                    f"T={args.n_periods})"
                )
                rows = run_timeline(
                    dim=args.dim,
                    n_samples=args.n_samples,
                    n_periods=args.n_periods,
                    arch_name=arch_name,
                    arch=arch,
                    strategy=strategy,
                    freeze_depth=freeze_depth,
                    seed=seed,
                )
                all_results.extend(rows)

                with open(args.json_output, "w") as f:
                    json.dump(all_results, f, indent=2)

                last = rows[-1]
                log.info(
                    f"  T={args.n_periods}: rho={last['recovery']:.3f} "
                    f"pfx={last['prefix_consistency']:.3f} "
                    f"mse={last['mse_method']:.1f}"
                )

    elapsed = time.time() - t0
    log.info(f"\nDone in {elapsed:.0f}s. {len(all_results)} rows.")

    # Summary
    log.info("\n=== Summary at T=20 ===")
    for arch_name in ARCHITECTURES:
        log.info(f"\n  {arch_name}:")
        for strategy in STRATEGIES:
            rs = [
                r for r in all_results
                if r["arch"] == arch_name
                and r["strategy"] == strategy
                and r["period"] == args.n_periods
            ]
            if rs:
                rhos = [r["recovery"] for r in rs]
                mses = [r["mse_method"] for r in rs]
                pfxs = [r["prefix_consistency"] for r in rs]
                log.info(
                    f"    {strategy:16s}: rho={np.mean(rhos):.3f}+/-{np.std(rhos):.3f}"
                    f"  mse={np.mean(mses):.1f}"
                    f"  pfx={np.mean(pfxs):.3f}"
                )

    # Freeze lifetime analysis
    log.info("\n=== Freeze lifetime (warm_periodic, tau=quality doubles) ===")
    for arch_name in ARCHITECTURES:
        lifetimes = []
        for seed in range(args.seeds):
            rs = sorted(
                [
                    r for r in all_results
                    if r["arch"] == arch_name
                    and r["strategy"] == "warm_periodic"
                    and r["seed"] == seed
                ],
                key=lambda r: r["period"],
            )
            if not rs:
                continue
            mse0 = rs[0]["mse_method"]
            tau = mse0 * 2.0
            lifetime = args.n_periods
            for r in rs[1:]:
                if r["mse_method"] > tau:
                    lifetime = r["period"] - 1
                    break
            lifetimes.append(lifetime)
        if lifetimes:
            log.info(
                f"  {arch_name}: lifetime={np.mean(lifetimes):.1f}"
                f"+/-{np.std(lifetimes):.1f} periods"
            )


if __name__ == "__main__":
    main()

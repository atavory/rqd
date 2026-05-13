#!/usr/bin/env python3
"""Shape selection experiment: joint (s, K_n) trajectories over time.

For each candidate architecture shape, tracks excess distortion R_{a,t}(s)
over multiple drift periods.  Answers two questions simultaneously:
  1. Which shape minimizes excess loss at each time horizon?
  2. Is the trajectory predictable (smooth or bursty)?

For each (shape, drift_type, dim, seed), records the full time series of
R_{a,t}, crossing fraction eps_t, and sample autocorrelation at lags 1..5.

Drift types:
  random_walk   — smooth accumulation
  sudden_break  — smooth then 5x jump at n_steps//2
  accelerating  — magnitude grows linearly
  cyclic        — direction reverses every 5 steps

Shapes at ~24 total bits (m=4 stages):
  Shallow+wide:  s=1, various K_n, 3 suffix stages
  Moderate:      s=2, various K_n, 2 suffix stages
  Deep+narrow:   s=3, various K_n, 1 suffix stage
  Fully frozen:  s=4 (baseline)
  Full retrain:  s=0 (baseline)

Usage:
    python3 run_autocorrelation.py --seeds 10
    python3 run_autocorrelation.py --seeds 2 --n-steps 10 --n-samples 2000
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ── Inline RQ (self-contained for runners) ───────────────────────


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
            d = np.min(
                np.sum((X[:, None, :] - centroids[None, :i, :]) ** 2, axis=2),
                axis=1,
            )
            t = d.sum()
            centroids[i] = (
                X[rng.choice(n, p=d / max(t, 1e-12))]
                if t > 1e-12
                else X[rng.randint(n)]
            )
    for _ in range(n_iter):
        a = np.argmin(
            np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2), axis=1
        )
        for j in range(k):
            m = a == j
            if m.sum() > 0:
                centroids[j] = X[m].mean(axis=0)
    return centroids


def _assign(X, c):
    return np.argmin(
        np.sum((X[:, None, :] - c[None, :, :]) ** 2, axis=2), axis=1
    ).astype(np.int64)


class RQ:
    def __init__(self, m, codes, dim):
        self.m = m
        self.dim = dim
        self.K = [codes] * m if isinstance(codes, int) else list(codes)
        self.cb: list[np.ndarray] = []

    def fit(self, X, n_iter=20, seed=42):
        rng = np.random.RandomState(seed)
        r = X.copy()
        self.cb = []
        for i in range(self.m):
            c = _kmeans(r, self.K[i], n_iter=n_iter, rng=rng)
            self.cb.append(c)
            a = _assign(r, c)
            r = r - c[a]
        return self

    def mse(self, X):
        r = X.copy()
        for c in self.cb:
            a = _assign(r, c)
            r = r - c[a]
        return float(np.mean(np.sum(r**2, axis=1)))


def warm_retrain(rq, X, fd, n_iter=20, seed=42):
    rq2 = RQ(rq.m, rq.K, rq.dim)
    rq2.cb = [c.copy() for c in rq.cb]
    r = X.copy()
    for i in range(fd):
        a = _assign(r, rq2.cb[i])
        r = r - rq2.cb[i][a]
    rng = np.random.RandomState(seed)
    for i in range(fd, rq.m):
        c = _kmeans(r, rq.K[i], n_iter=n_iter, rng=rng, init=rq2.cb[i])
        rq2.cb[i] = c
        a = _assign(r, c)
        r = r - c[a]
    return rq2


# ── Drift functions ──────────────────────────────────────────────


def _drift_random_walk(X0, step, drift_per_step, seed):
    rng = np.random.RandomState(seed)
    dim = X0.shape[1]
    td = np.zeros(dim, dtype=np.float32)
    for t in range(step):
        d = rng.randn(dim).astype(np.float32)
        d /= np.linalg.norm(d)
        td += d * drift_per_step
    return X0 + td


def _drift_sudden_break(X0, step, drift_per_step, seed, break_step):
    rng = np.random.RandomState(seed)
    dim = X0.shape[1]
    td = np.zeros(dim, dtype=np.float32)
    for t in range(step):
        d = rng.randn(dim).astype(np.float32)
        d /= np.linalg.norm(d)
        mag = drift_per_step
        if t == break_step:
            mag *= 5.0
        td += d * mag
    return X0 + td


def _drift_accelerating(X0, step, drift_per_step, seed):
    rng = np.random.RandomState(seed)
    dim = X0.shape[1]
    td = np.zeros(dim, dtype=np.float32)
    for t in range(step):
        d = rng.randn(dim).astype(np.float32)
        d /= np.linalg.norm(d)
        td += d * drift_per_step * (1.0 + t / 10.0)
    return X0 + td


def _drift_cyclic(X0, step, drift_per_step, seed):
    rng = np.random.RandomState(seed)
    dim = X0.shape[1]
    td = np.zeros(dim, dtype=np.float32)
    for t in range(step):
        d = rng.randn(dim).astype(np.float32)
        d /= np.linalg.norm(d)
        sign = 1.0 if (t // 5) % 2 == 0 else -1.0
        td += d * drift_per_step * sign
    return X0 + td


def apply_drift(X0, step, drift_per_step, seed, drift_type, n_steps):
    if drift_type == "random_walk":
        return _drift_random_walk(X0, step, drift_per_step, seed)
    elif drift_type == "sudden_break":
        return _drift_sudden_break(
            X0, step, drift_per_step, seed, n_steps // 2
        )
    elif drift_type == "accelerating":
        return _drift_accelerating(X0, step, drift_per_step, seed)
    elif drift_type == "cyclic":
        return _drift_cyclic(X0, step, drift_per_step, seed)
    else:
        raise ValueError(f"Unknown drift type: {drift_type}")


# ── Autocorrelation ──────────────────────────────────────────────


def sample_autocorrelation(series, max_lag=5):
    x = np.array(series, dtype=np.float64)
    n = len(x)
    if n < max_lag + 2:
        return [float("nan")] * max_lag
    mu = x.mean()
    var = np.mean((x - mu) ** 2)
    if var < 1e-15:
        return [1.0] * max_lag
    acf = []
    for lag in range(1, max_lag + 1):
        c = np.mean((x[: n - lag] - mu) * (x[lag:] - mu))
        acf.append(float(c / var))
    return acf


# ── Crossing fraction ────────────────────────────────────────────


def crossing_fraction(rq, X0, X1, fd):
    def encode_prefix(x, depth):
        codes = []
        r = x.copy()
        for i in range(depth):
            a = _assign(r, rq.cb[i])
            codes.append(a)
            r = r - rq.cb[i][a]
        return np.stack(codes, axis=1)

    c0 = encode_prefix(X0, fd)
    c1 = encode_prefix(X1, fd)
    return float(np.any(c0 != c1, axis=1).mean())


# ── Shape configs ────────────────────────────────────────────────

# (name, freeze_depth, arch as [K per stage])
# Target ~24 bits total.  Freeze depth 0 = full retrain (baseline).
SHAPES = [
    # Full retrain baseline (no freezing)
    ("full_retrain", 0, [64, 64, 64, 64]),
    # s=1: shallow freeze, most suffix capacity
    ("s1_Kn4", 1, [4, 128, 128, 128]),
    ("s1_Kn16", 1, [16, 128, 128, 128]),
    ("s1_Kn64", 1, [64, 64, 64, 64]),
    ("s1_Kn256", 1, [256, 32, 32, 32]),
    # s=2: moderate freeze (the paper's default)
    ("s2_Kn4", 2, [4, 4, 256, 256]),
    ("s2_Kn16", 2, [16, 16, 256, 256]),
    ("s2_Kn64", 2, [64, 64, 64, 64]),
    # s=3: deep freeze, little suffix
    ("s3_Kn4", 3, [4, 4, 4, 256]),
    ("s3_Kn16", 3, [16, 16, 16, 256]),
    # Fully frozen baseline
    ("frozen", 4, [64, 64, 64, 64]),
]


# ── Single job (for multiprocessing) ─────────────────────────────


def run_one_job(job):
    shape_name, fd, arch, drift_type, dim, seed, n_steps, n_samples, drift_per_step, max_lag = job
    os.environ["OMP_NUM_THREADS"] = "1"

    n_stages = len(arch)
    total_bits = sum(int(round(np.log2(k))) for k in arch)
    n_prefix_buckets = 1
    for k in arch[:fd]:
        n_prefix_buckets *= k

    rng = np.random.RandomState(seed)
    X0 = rng.randn(n_samples, dim).astype(np.float32) * 3.0
    rq0 = RQ(n_stages, arch, dim).fit(X0, seed=seed)

    mse_warm_series = []
    mse_full_series = []
    mse_frozen_series = []
    r_series = []
    eps_series = []

    rq_warm = rq0
    for step in range(1, n_steps + 1):
        X_t = apply_drift(
            X0, step, drift_per_step, seed + 1, drift_type, n_steps
        )

        mse_frozen = rq0.mse(X_t)

        if fd == 0:
            rq_step = RQ(n_stages, arch, dim).fit(
                X_t, seed=seed + 500 + step
            )
            mse_warm = rq_step.mse(X_t)
        elif fd >= n_stages:
            mse_warm = mse_frozen
        else:
            rq_warm = warm_retrain(rq_warm, X_t, fd, seed=seed + step)
            mse_warm = rq_warm.mse(X_t)

        rq_full = RQ(n_stages, arch, dim).fit(
            X_t, seed=seed + 500 + step
        )
        mse_full = rq_full.mse(X_t)

        r_excess = mse_warm - mse_full

        if 0 < fd < n_stages:
            eps = crossing_fraction(rq0, X0, X_t, fd)
        else:
            eps = 0.0 if fd >= n_stages else 1.0

        mse_warm_series.append(round(mse_warm, 4))
        mse_full_series.append(round(mse_full, 4))
        mse_frozen_series.append(round(mse_frozen, 4))
        r_series.append(round(r_excess, 4))
        eps_series.append(round(eps, 4))

    acf = sample_autocorrelation(r_series, max_lag=max_lag)

    return {
        "shape": shape_name,
        "freeze_depth": fd,
        "arch": arch,
        "total_bits": total_bits,
        "n_prefix_buckets": n_prefix_buckets,
        "drift_type": drift_type,
        "dim": dim,
        "seed": seed,
        "mse_warm": mse_warm_series,
        "mse_full": mse_full_series,
        "mse_frozen": mse_frozen_series,
        "r_excess": r_series,
        "crossing_frac": eps_series,
        "acf": [round(v, 4) for v in acf],
    }


# ── Main ─────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--n-steps", type=int, default=20)
    p.add_argument("--n-samples", type=int, default=10000)
    p.add_argument("--drift-per-step", type=float, default=1.0)
    p.add_argument("--max-lag", type=int, default=5)
    p.add_argument("--workers", type=int, default=0,
                   help="0 = all cores")
    p.add_argument("--json-output", type=str,
                   default="shape_trajectories.json")
    p.add_argument("--drift-types", type=str, default=None,
                   help="Comma-separated subset, e.g. random_walk,cyclic")
    p.add_argument("--dims", type=str, default=None,
                   help="Comma-separated subset, e.g. 27,64")
    p.add_argument("--shapes", type=str, default=None,
                   help="Comma-separated subset, e.g. s1_Kn16,s2_Kn16")
    args = p.parse_args()

    t0 = time.time()

    all_dims = [27, 64, 128]
    all_drift_types = ["random_walk", "sudden_break", "accelerating", "cyclic"]

    dims = [int(d) for d in args.dims.split(",")] if args.dims else all_dims
    drift_types = args.drift_types.split(",") if args.drift_types else all_drift_types
    shape_filter = set(args.shapes.split(",")) if args.shapes else None

    jobs = []
    for drift_type in drift_types:
        for dim in dims:
            for shape_name, fd, arch in SHAPES:
                if shape_filter and shape_name not in shape_filter:
                    continue
                for seed in range(args.seeds):
                    jobs.append((
                        shape_name, fd, arch, drift_type, dim, seed,
                        args.n_steps, args.n_samples, args.drift_per_step,
                        args.max_lag,
                    ))

    n_workers = args.workers if args.workers > 0 else mp.cpu_count()
    log.info(
        f"Running {len(jobs)} jobs on {n_workers} workers "
        f"(n_samples={args.n_samples}, n_steps={args.n_steps})"
    )

    results = []
    with mp.Pool(n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(run_one_job, jobs)):
            results.append(result)
            if (i + 1) % 20 == 0 or (i + 1) == len(jobs):
                with open(args.json_output, "w") as f:
                    json.dump(results, f)
                acf1 = result["acf"][0] if not np.isnan(result["acf"][0]) else 0.0
                log.info(
                    f"[{i+1}/{len(jobs)}] {result['shape']:12s} "
                    f"{result['drift_type']:15s} d={result['dim']:3d} "
                    f"seed={result['seed']} "
                    f"R_final={result['r_excess'][-1]:.3f} ACF1={acf1:.3f}"
                )

    with open(args.json_output, "w") as f:
        json.dump(results, f)

    elapsed = time.time() - t0
    log.info(f"Done in {elapsed:.0f}s, {len(results)} rows")

    log.info("=== Summary by shape (all dims, all drifts) ===")
    for shape_name, fd, arch in SHAPES:
        rows = [r for r in results if r["shape"] == shape_name]
        if rows:
            r_finals = [r["r_excess"][-1] for r in rows]
            acf1s = [
                r["acf"][0] for r in rows if not np.isnan(r["acf"][0])
            ]
            log.info(
                f"  {shape_name:12s} (fd={fd}): "
                f"R_final={np.mean(r_finals):7.3f} +/- {np.std(r_finals):.3f}  "
                f"ACF1={np.mean(acf1s):.3f} +/- {np.std(acf1s):.3f}"
            )

    log.info("=== Summary by drift type (s2_Kn16 only) ===")
    for drift_type in drift_types:
        rows = [
            r for r in results
            if r["shape"] == "s2_Kn16" and r["drift_type"] == drift_type
        ]
        if rows:
            r_finals = [r["r_excess"][-1] for r in rows]
            acf1s = [
                r["acf"][0] for r in rows if not np.isnan(r["acf"][0])
            ]
            log.info(
                f"  {drift_type:15s}: "
                f"R_final={np.mean(r_finals):7.3f} +/- {np.std(r_finals):.3f}  "
                f"ACF1={np.mean(acf1s):.3f} +/- {np.std(acf1s):.3f}"
            )


if __name__ == "__main__":
    main()

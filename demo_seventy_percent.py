#!/usr/bin/env python3
"""Demo: the 70% law and funnel architecture.

Reproduces the core result from the paper: warm-retraining the last
floor(M/2) stages of an RQ codebook recovers ~70% of the quality gap
to full retraining (uniform), and >90% with the funnel architecture.

Sweeps M, K, d, and drift magnitude to show scale invariance.

Usage:
    python3 demo_seventy_percent.py
    python3 demo_seventy_percent.py --dim 128 --n-samples 20000
"""

import argparse
import numpy as np
from rq import (
    RQCodebook,
    warm_retrain,
    gap_recovery,
    generate_data,
    apply_drift,
    prefix_consistency,
)


def run_scale_invariance(
    dim: int = 64,
    n_samples: int = 10000,
    drift_magnitude: float = 2.0,
    seed: int = 42,
) -> None:
    """Show that rho ~ 70% regardless of M, K, d."""

    print("=== Scale Invariance of Gap Recovery ===")
    print(f"dim={dim}, n={n_samples}, drift={drift_magnitude} (mean-shift)")
    print()

    X_t0 = generate_data(n_samples, dim, seed=seed)
    X_t1 = apply_drift(X_t0, "mean_shift", drift_magnitude, seed=seed + 1)

    configs = [
        (2, 32), (2, 64), (2, 128), (2, 256),
        (3, 32), (3, 64), (3, 128),
        (4, 32), (4, 64), (4, 128),
        (6, 64), (8, 64),
    ]

    print(f"{'M':>3} {'K':>5} {'s':>3} | {'Frozen':>8} {'Warm':>8} "
          f"{'Full':>8} | {'rho':>7} {'Pfx cons':>9}")
    print("-" * 65)

    for M, K in configs:
        s = M // 2

        rq = RQCodebook(M, K, dim)
        rq.fit(X_t0, seed=seed)

        rq_warm = warm_retrain(rq, X_t1, freeze_depth=s, seed=seed)

        rq_full = RQCodebook(M, K, dim)
        rq_full.fit(X_t1, seed=seed + 2)

        mse_frz = rq.mse(X_t1)
        mse_w = rq_warm.mse(X_t1)
        mse_f = rq_full.mse(X_t1)
        rho = gap_recovery(mse_frz, mse_w, mse_f)

        codes_t0 = rq.encode(X_t0, n_stages=s)
        codes_t1 = rq_warm.encode(X_t1, n_stages=s)
        pfx_cons = prefix_consistency(codes_t0, codes_t1, s)

        print(f"{M:>3} {K:>5} {s:>3} | {mse_frz:>8.4f} {mse_w:>8.4f} "
              f"{mse_f:>8.4f} | {rho:>6.1%} {pfx_cons:>8.1%}")

    print()
    print("rho should cluster around 65-70% regardless of M and K.")
    print("Prefix consistency should be 100% (frozen stages never change).")


def run_funnel_comparison(
    dim: int = 64,
    n_samples: int = 10000,
    drift_magnitude: float = 2.0,
    seed: int = 42,
) -> None:
    """Compare uniform vs funnel architectures."""

    print()
    print("=== Uniform vs Funnel Architecture ===")
    print(f"dim={dim}, n={n_samples}, drift={drift_magnitude} (mean-shift)")
    print()

    X_t0 = generate_data(n_samples, dim, seed=seed)
    X_t1 = apply_drift(X_t0, "mean_shift", drift_magnitude, seed=seed + 1)

    architectures = [
        ("Uniform [64,64,64,64]", [64, 64, 64, 64]),
        ("Funnel  [16,16,256,256]", [16, 16, 256, 256]),
        ("Funnel  [16,16,256,512]", [16, 16, 256, 512]),
        ("Inverted [256,256,16,16]", [256, 256, 16, 16]),
        ("Narrow pfx [16,16,64,64]", [16, 16, 64, 64]),
    ]

    print(f"{'Architecture':<28} {'Bits':>5} | "
          f"{'Frozen':>8} {'Warm-2':>8} {'Full':>8} | {'rho':>7}")
    print("-" * 78)

    for name, ks in architectures:
        M = len(ks)
        s = M // 2
        bits = sum(np.log2(k) for k in ks)

        rq = RQCodebook(M, ks, dim)
        rq.fit(X_t0, seed=seed)

        rq_warm = warm_retrain(rq, X_t1, freeze_depth=s, seed=seed)

        rq_full = RQCodebook(M, ks, dim)
        rq_full.fit(X_t1, seed=seed + 2)

        mse_frz = rq.mse(X_t1)
        mse_w = rq_warm.mse(X_t1)
        mse_f = rq_full.mse(X_t1)
        rho = gap_recovery(mse_frz, mse_w, mse_f)

        print(f"{name:<28} {bits:>5.0f} | "
              f"{mse_frz:>8.4f} {mse_w:>8.4f} {mse_f:>8.4f} | {rho:>6.1%}")

    print()
    print("Funnel should push rho above 90%. Inverted should be worst.")


def main():
    parser = argparse.ArgumentParser(
        description="Demo: 70%% law and funnel architecture"
    )
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--drift", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_scale_invariance(
        dim=args.dim,
        n_samples=args.n_samples,
        drift_magnitude=args.drift,
        seed=args.seed,
    )
    run_funnel_comparison(
        dim=args.dim,
        n_samples=args.n_samples,
        drift_magnitude=args.drift,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

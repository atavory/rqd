#!/usr/bin/env python3
"""Demo: low-data transfer — frozen prefix as structural regularizer.

Shows that warm-2 outperforms full retrain when target data is scarce,
because the frozen prefix constrains the solution space. Full retrain
with tiny data overfits catastrophically (rho < 0).

Usage:
    python3 demo_lowdata.py
"""

import argparse
import numpy as np
from rq import (
    RQCodebook,
    warm_retrain,
    gap_recovery,
    generate_data,
    apply_drift,
)


def run_experiment(
    dim: int = 128,
    n_source: int = 10000,
    n_target: int = 5000,
    drift_magnitude: float = 2.0,
    seed: int = 42,
) -> None:
    print("=== Low-Data Transfer: Frozen Prefix as Regularizer ===")
    print(f"dim={dim}, n_source={n_source}, n_target={n_target}, "
          f"drift={drift_magnitude}")
    print()

    M, K = 4, 64
    s = M // 2

    X_source = generate_data(n_source, dim, seed=seed)
    X_target = apply_drift(X_source, "mean_shift", drift_magnitude, seed=seed + 1)
    X_target = X_target[:n_target]

    # Train base RQ on source
    rq = RQCodebook(M, K, dim)
    rq.fit(X_source, seed=seed)

    # Reference: full retrain and frozen on ALL target data
    rq_full_all = RQCodebook(M, K, dim)
    rq_full_all.fit(X_target, seed=seed + 2)
    mse_full_all = rq_full_all.mse(X_target)
    mse_frozen = rq.mse(X_target)

    fractions = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]

    print(f"{'Frac':>6} {'n':>6} | "
          f"{'rho_warm2':>10} {'rho_full':>10} | {'Winner':>8}")
    print("-" * 55)

    for frac in fractions:
        n = max(10, int(n_target * frac))
        rng = np.random.RandomState(seed + 10)
        idx = rng.choice(n_target, n, replace=False)
        X_sub = X_target[idx]

        # Warm-2 on subset
        rq_warm = warm_retrain(rq, X_sub, freeze_depth=s, seed=seed)
        mse_warm = rq_warm.mse(X_target)
        rho_warm = gap_recovery(mse_frozen, mse_warm, mse_full_all)

        # Full retrain on subset
        rq_full_sub = RQCodebook(M, K, dim)
        rq_full_sub.fit(X_sub, seed=seed + 3)
        mse_full_sub = rq_full_sub.mse(X_target)
        rho_full = gap_recovery(mse_frozen, mse_full_sub, mse_full_all)

        winner = "warm-2" if rho_warm > rho_full else "full"
        print(f"{frac:>5.0%} {n:>6} | "
              f"{rho_warm:>9.1%} {rho_full:>9.1%} | {winner:>8}")

    print()
    print("At small fractions, full retrain overfits (rho < 0 = worse than frozen).")
    print("Warm-2 stays positive because the frozen prefix constrains the space.")
    print("The crossover marks where full retrain has enough data to overcome")
    print("its lack of structural prior.")


def main():
    parser = argparse.ArgumentParser(
        description="Demo: low-data transfer"
    )
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n-source", type=int, default=10000)
    parser.add_argument("--n-target", type=int, default=5000)
    parser.add_argument("--drift", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_experiment(
        dim=args.dim,
        n_source=args.n_source,
        n_target=args.n_target,
        drift_magnitude=args.drift,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

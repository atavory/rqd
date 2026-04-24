#!/usr/bin/env python3
"""Demo: streaming adaptation strategies.

Compares five adaptation strategies over a multi-step timeline
with cumulative random-walk drift:

1. Static (frozen T0 codebook)
2. Full retrain at each step
3. Warm-2 once at T1 only
4. Warm-2 periodic (every step)
5. Warm-2 triggered (retrain when MSE > 1.5x baseline)

Usage:
    python3 demo_streaming.py
    python3 demo_streaming.py --n-steps 10 --drift-per-step 0.5
"""

import argparse
import numpy as np
from rq import (
    RQCodebook,
    warm_retrain,
    gap_recovery,
    generate_data,
    codebook_entropy,
)


def random_walk_drift(
    X_base: np.ndarray,
    step: int,
    drift_per_step: float,
    seed: int = 42,
) -> np.ndarray:
    """Apply cumulative random-walk drift over `step` steps."""
    rng = np.random.RandomState(seed)
    dim = X_base.shape[1]
    total_drift = np.zeros(dim, dtype=np.float32)
    for t in range(step):
        direction = rng.randn(dim).astype(np.float32)
        direction /= np.linalg.norm(direction)
        total_drift += direction * drift_per_step
    return X_base + total_drift


def run_experiment(
    dim: int = 128,
    n_samples: int = 10000,
    n_codes: int = 64,
    n_stages: int = 4,
    n_steps: int = 10,
    drift_per_step: float = 0.5,
    seed: int = 42,
) -> None:
    """Run the streaming adaptation comparison."""

    print("=== Streaming Adaptation Strategies ===")
    print(f"dim={dim}, K={n_codes}, M={n_stages}, "
          f"steps={n_steps}, drift/step={drift_per_step}")
    print()

    X_base = generate_data(n_samples, dim, seed=seed)
    rq_t0 = RQCodebook(n_stages, n_codes, dim)
    rq_t0.fit(X_base, seed=seed)
    mse_baseline = rq_t0.mse(X_base)

    freeze_depth = n_stages // 2

    # State for each strategy
    rq_once = None  # warm-2 once at T1
    rq_periodic = rq_t0  # warm-2 every step (chained)
    rq_triggered = rq_t0  # warm-2 when MSE > threshold
    trigger_threshold = mse_baseline * 1.5
    n_triggers = 0

    print(f"{'Step':>4} | {'Static':>8} {'Full':>8} {'Once':>8} "
          f"{'Periodic':>8} {'Triggered':>8} | {'Trig?':>5}")
    print("-" * 72)

    for step in range(n_steps + 1):
        X_t = random_walk_drift(X_base, step, drift_per_step, seed=seed + 1)

        # 1. Static
        mse_static = rq_t0.mse(X_t)

        # 2. Full retrain
        rq_full = RQCodebook(n_stages, n_codes, dim)
        rq_full.fit(X_t, seed=seed + step + 100)
        mse_full = rq_full.mse(X_t)

        # 3. Warm-2 once at T1
        if step == 1:
            rq_once = warm_retrain(rq_t0, X_t, freeze_depth, seed=seed + step)
        mse_once = rq_once.mse(X_t) if rq_once else mse_static

        # 4. Warm-2 periodic
        if step > 0:
            rq_periodic = warm_retrain(
                rq_periodic, X_t, freeze_depth, seed=seed + step
            )
        mse_periodic = rq_periodic.mse(X_t)

        # 5. Warm-2 triggered
        triggered = False
        mse_trig_check = rq_triggered.mse(X_t)
        if step > 0 and mse_trig_check > trigger_threshold:
            rq_triggered = warm_retrain(
                rq_triggered, X_t, freeze_depth, seed=seed + step
            )
            triggered = True
            n_triggers += 1
        mse_triggered = rq_triggered.mse(X_t)

        trig_str = "YES" if triggered else ""
        print(f"{step:>4} | {mse_static:>8.4f} {mse_full:>8.4f} "
              f"{mse_once:>8.4f} {mse_periodic:>8.4f} "
              f"{mse_triggered:>8.4f} | {trig_str:>5}")

    print()
    print(f"Triggered adaptation fired {n_triggers}/{n_steps} times "
          f"(threshold: {trigger_threshold:.4f})")
    print()
    print("Periodic warm-2 should track full retrain closely.")
    print("Once should go stale after a few steps.")
    print("Triggered should fire ~3x, a practical middle ground.")


def main():
    parser = argparse.ArgumentParser(
        description="Demo: streaming adaptation strategies"
    )
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--n-codes", type=int, default=64)
    parser.add_argument("--n-stages", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=10)
    parser.add_argument("--drift-per-step", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_experiment(
        dim=args.dim,
        n_samples=args.n_samples,
        n_codes=args.n_codes,
        n_stages=args.n_stages,
        n_steps=args.n_steps,
        drift_per_step=args.drift_per_step,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

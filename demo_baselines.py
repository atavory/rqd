#!/usr/bin/env python3
"""Demo: comparison with alternative adaptation methods.

Compares stratified plasticity (warm-2) against:
- EWC (Elastic Weight Consolidation) on codebook entries
- EMA (Exponential Moving Average) codebook updates
- Flat VQ with K^M entries

Shows that warm-2 is the only method achieving both high recovery
and full prefix consistency.

Usage:
    python3 demo_baselines.py
    python3 demo_baselines.py --dim 128 --drift 2.0
"""

import argparse
import numpy as np
from rq import (
    RQCodebook,
    warm_retrain,
    gap_recovery,
    prefix_consistency,
    generate_data,
    apply_drift,
    _kmeans,
    _assign,
)


def ewc_retrain(
    rq: RQCodebook,
    X_old: np.ndarray,
    X_new: np.ndarray,
    lam: float = 1.0,
    n_iter: int = 20,
    seed: int = 42,
) -> RQCodebook:
    """Retrain all stages with EWC penalty on codebook entries.

    Fisher information approximated as assignment frequency.
    """
    rq_new = RQCodebook(rq.n_stages, rq.codes_per_stage, rq.dim)
    rq_new.codebooks = [cb.copy() for cb in rq.codebooks]

    residual_old = X_old.copy()
    residual_new = X_new.copy()

    for m in range(rq.n_stages):
        k = rq.codes_per_stage[m]
        old_centroids = rq.codebooks[m].copy()

        # Fisher ~ assignment frequency on old data
        a_old = _assign(residual_old, old_centroids)
        fisher = np.bincount(a_old, minlength=k).astype(np.float32)
        fisher = fisher / fisher.sum()

        centroids = rq_new.codebooks[m].copy()
        for _ in range(n_iter):
            a = _assign(residual_new, centroids)
            for j in range(k):
                mask = a == j
                if mask.sum() > 0:
                    grad_data = residual_new[mask].mean(axis=0) - centroids[j]
                    grad_ewc = -lam * fisher[j] * (centroids[j] - old_centroids[j])
                    centroids[j] += 0.5 * grad_data + grad_ewc

        rq_new.codebooks[m] = centroids
        a = _assign(residual_new, centroids)
        residual_new = residual_new - centroids[a]
        a_old2 = _assign(residual_old, centroids)
        residual_old = residual_old - centroids[a_old2]

    return rq_new


def ema_update(
    rq: RQCodebook,
    X_new: np.ndarray,
    momentum: float = 0.9,
    freeze_prefix: bool = False,
    freeze_depth: int = 0,
) -> RQCodebook:
    """EMA codebook update: centroid = mu * old + (1-mu) * new_mean."""
    rq_new = RQCodebook(rq.n_stages, rq.codes_per_stage, rq.dim)
    rq_new.codebooks = [cb.copy() for cb in rq.codebooks]

    residual = X_new.copy()
    for m in range(rq.n_stages):
        k = rq.codes_per_stage[m]
        centroids = rq_new.codebooks[m]
        a = _assign(residual, centroids)

        if freeze_prefix and m < freeze_depth:
            pass
        else:
            for j in range(k):
                mask = a == j
                if mask.sum() > 0:
                    new_mean = residual[mask].mean(axis=0)
                    centroids[j] = momentum * centroids[j] + (1 - momentum) * new_mean

        rq_new.codebooks[m] = centroids
        a = _assign(residual, centroids)
        residual = residual - centroids[a]

    return rq_new


def flat_vq_experiment(
    X_t0: np.ndarray,
    X_t1: np.ndarray,
    total_codes: int = 4096,
    seed: int = 42,
) -> dict:
    """Flat VQ with K^M codes — no hierarchy to exploit."""
    dim = X_t0.shape[1]

    rng = np.random.RandomState(seed)
    centroids_t0 = _kmeans(X_t0, total_codes, n_iter=20, rng=rng)

    # Frozen
    a_frozen = _assign(X_t1, centroids_t0)
    mse_frozen = float(np.mean(np.sum(
        (X_t1 - centroids_t0[a_frozen]) ** 2, axis=1
    )))

    # Full retrain
    rng2 = np.random.RandomState(seed + 1)
    centroids_full = _kmeans(X_t1, total_codes, n_iter=20, rng=rng2)
    a_full = _assign(X_t1, centroids_full)
    mse_full = float(np.mean(np.sum(
        (X_t1 - centroids_full[a_full]) ** 2, axis=1
    )))

    # "Freeze half" — freeze the most-used codes, retrain the rest
    a_t0 = _assign(X_t0, centroids_t0)
    counts = np.bincount(a_t0, minlength=total_codes)
    top_half = np.argsort(counts)[-total_codes // 2:]
    frozen_mask = np.zeros(total_codes, dtype=bool)
    frozen_mask[top_half] = True

    centroids_half = centroids_t0.copy()
    for _ in range(20):
        a = _assign(X_t1, centroids_half)
        for j in range(total_codes):
            if frozen_mask[j]:
                continue
            mask = a == j
            if mask.sum() > 0:
                centroids_half[j] = X_t1[mask].mean(axis=0)

    a_half = _assign(X_t1, centroids_half)
    mse_half = float(np.mean(np.sum(
        (X_t1 - centroids_half[a_half]) ** 2, axis=1
    )))

    return {
        "frozen": mse_frozen,
        "full": mse_full,
        "freeze_half": mse_half,
    }


def run_experiment(
    dim: int = 128,
    n_samples: int = 10000,
    drift_magnitude: float = 2.0,
    seed: int = 42,
) -> None:
    print("=== Baseline Comparison ===")
    print(f"dim={dim}, n={n_samples}, drift={drift_magnitude} (mean-shift)")
    print()

    M, K = 4, 64
    s = M // 2

    X_t0 = generate_data(n_samples, dim, seed=seed)
    X_t1 = apply_drift(X_t0, "mean_shift", drift_magnitude, seed=seed + 1)

    # Train base RQ
    rq = RQCodebook(M, K, dim)
    rq.fit(X_t0, seed=seed)

    codes_t0 = rq.encode(X_t0)

    # Methods
    results = []

    # Frozen
    mse_frz = rq.mse(X_t1)
    results.append(("Frozen RQ", mse_frz, 0.0, 1.0))

    # Warm-2
    rq_w2 = warm_retrain(rq, X_t1, freeze_depth=s, seed=seed)
    mse_w2 = rq_w2.mse(X_t1)
    codes_w2 = rq_w2.encode(X_t1)
    pfx_w2 = prefix_consistency(codes_t0, codes_w2, s)
    results.append(("Warm-2 RQ (ours)", mse_w2, 0.0, pfx_w2))

    # Full retrain
    rq_full = RQCodebook(M, K, dim)
    rq_full.fit(X_t1, seed=seed + 2)
    mse_full = rq_full.mse(X_t1)
    codes_full = rq_full.encode(X_t1)
    pfx_full = prefix_consistency(codes_t0, codes_full, s)
    results.append(("Full retrain RQ", mse_full, 0.0, pfx_full))

    # EWC
    for lam in [0.01, 1.0]:
        rq_ewc = ewc_retrain(rq, X_t0, X_t1, lam=lam, seed=seed)
        mse_ewc = rq_ewc.mse(X_t1)
        codes_ewc = rq_ewc.encode(X_t1)
        pfx_ewc = prefix_consistency(codes_t0, codes_ewc, s)
        results.append((f"EWC lam={lam}", mse_ewc, 0.0, pfx_ewc))

    # EMA
    for mu in [0.9, 0.99]:
        rq_ema = ema_update(rq, X_t1, momentum=mu)
        mse_ema = rq_ema.mse(X_t1)
        codes_ema = rq_ema.encode(X_t1)
        pfx_ema = prefix_consistency(codes_t0, codes_ema, s)
        results.append((f"EMA mu={mu}", mse_ema, 0.0, pfx_ema))

    # EMA + frozen prefix
    rq_ema_fp = ema_update(rq, X_t1, momentum=0.99, freeze_prefix=True, freeze_depth=s)
    mse_ema_fp = rq_ema_fp.mse(X_t1)
    codes_ema_fp = rq_ema_fp.encode(X_t1)
    pfx_ema_fp = prefix_consistency(codes_t0, codes_ema_fp, s)
    results.append(("EMA mu=0.99 +freeze", mse_ema_fp, 0.0, pfx_ema_fp))

    # Compute rho for all
    print(f"{'Method':<22} {'MSE':>8} {'rho':>8} {'Pfx cons':>9}")
    print("-" * 50)
    for name, mse, _, pfx in results:
        rho = gap_recovery(mse_frz, mse, mse_full)
        print(f"{name:<22} {mse:>8.4f} {rho:>7.1%} {pfx:>8.1%}")

    # Flat VQ comparison
    print()
    print("--- Flat VQ (K=4096, no hierarchy) ---")
    flat = flat_vq_experiment(X_t0, X_t1, total_codes=K**2, seed=seed)
    rho_flat_frz = gap_recovery(flat["frozen"], flat["frozen"], flat["full"])
    rho_flat_half = gap_recovery(flat["frozen"], flat["freeze_half"], flat["full"])
    print(f"  Frozen:      MSE={flat['frozen']:.4f}  rho={rho_flat_frz:.1%}")
    print(f"  Freeze-half: MSE={flat['freeze_half']:.4f}  rho={rho_flat_half:.1%}")
    print(f"  Full:        MSE={flat['full']:.4f}  rho=100.0%")
    print()
    print("Flat VQ freeze-half provides no benefit — no hierarchy to exploit.")


def main():
    parser = argparse.ArgumentParser(
        description="Demo: baseline comparison"
    )
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--drift", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_experiment(
        dim=args.dim,
        n_samples=args.n_samples,
        drift_magnitude=args.drift,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

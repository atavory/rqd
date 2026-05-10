#!/usr/bin/env python3
"""Support experiments for the revised freeze-depth analysis.

Runs four experiments:
  1. Freeze-depth monotonicity sweep (D_s vs s)
  2. Trigger-threshold sweep (threshold vs trigger count / avg MSE)
  3. MSE vs entropy trigger comparison
  4. Stage interaction / non-additivity test

All pure numpy/scipy — no torch, no external data.
Results saved to JSON for figure generation.

Usage:
    python3 exp_freeze_depth_support.py
    python3 exp_freeze_depth_support.py --seeds 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rq import (
    RQCodebook,
    warm_retrain,
    codebook_entropy,
    generate_data,
    code_consistency,
)


def exp1_monotonicity(
    dims: list[int],
    n_samples: int,
    n_codes: int,
    n_stages: int,
    seeds: int,
    drift_mag: float,
) -> dict:
    """Freeze-depth monotonicity: D_s should be nondecreasing in s."""
    print("=== Exp 1: Freeze-depth monotonicity ===")
    results = {}
    for dim in dims:
        print(f"  dim={dim}")
        mse_by_s = {s: [] for s in range(n_stages + 1)}
        churn_by_s = {s: [] for s in range(n_stages + 1)}
        for seed in range(seeds):
            X0 = generate_data(n_samples, dim, seed=seed)
            rng = np.random.RandomState(seed + 9999)
            direction = rng.randn(dim).astype(np.float32)
            direction /= np.linalg.norm(direction)
            X1 = X0 + direction * drift_mag

            rq0 = RQCodebook(n_stages, n_codes, dim).fit(X0, seed=seed)
            codes_frozen = rq0.encode(X1)

            for s in range(n_stages + 1):
                if s == 0:
                    rq_s = RQCodebook(n_stages, n_codes, dim).fit(
                        X1, seed=seed + 500
                    )
                else:
                    rq_s = warm_retrain(rq0, X1, freeze_depth=s, seed=seed + s)
                mse_by_s[s].append(rq_s.mse(X1))
                codes_s = rq_s.encode(X1)
                churn_by_s[s].append(1.0 - code_consistency(codes_frozen, codes_s))

        results[dim] = {
            "mse": {
                s: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for s, v in mse_by_s.items()
            },
            "churn": {
                s: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for s, v in churn_by_s.items()
            },
        }
        for s in range(n_stages + 1):
            m = results[dim]["mse"][s]
            c = results[dim]["churn"][s]
            print(
                f"    s={s}: MSE={m['mean']:.2f}±{m['std']:.2f}, "
                f"churn={c['mean']:.3f}±{c['std']:.3f}"
            )
    return results


def exp2_trigger_sweep(
    dim: int,
    n_samples: int,
    n_codes: int,
    n_stages: int,
    n_steps: int,
    drift_per_step: float,
    thresholds: list[float],
    seeds: int,
) -> dict:
    """Trigger-threshold sweep: vary threshold, record triggers and avg MSE."""
    print("\n=== Exp 2: Trigger-threshold sweep ===")
    results = {}
    for thresh_mult in thresholds:
        trigger_counts = []
        avg_mses = []
        for seed in range(seeds):
            X0 = generate_data(n_samples, dim, seed=seed)
            rq0 = RQCodebook(n_stages, n_codes, dim).fit(X0, seed=seed)
            mse_baseline = rq0.mse(X0)
            threshold = mse_baseline * thresh_mult
            freeze_depth = n_stages // 2

            rq_cur = rq0
            n_triggers = 0
            mses = []

            for step in range(1, n_steps + 1):
                rng = np.random.RandomState(seed + 1)
                total_drift = np.zeros(dim, dtype=np.float32)
                for t in range(step):
                    d = rng.randn(dim).astype(np.float32)
                    d /= np.linalg.norm(d)
                    total_drift += d * drift_per_step
                X_t = X0 + total_drift

                mse_check = rq_cur.mse(X_t)
                if mse_check > threshold:
                    rq_cur = warm_retrain(
                        rq_cur, X_t, freeze_depth=freeze_depth, seed=seed + step
                    )
                    n_triggers += 1
                mses.append(rq_cur.mse(X_t))

            trigger_counts.append(n_triggers)
            avg_mses.append(float(np.mean(mses)))

        results[f"{thresh_mult:.2f}x"] = {
            "threshold_mult": thresh_mult,
            "triggers_mean": float(np.mean(trigger_counts)),
            "triggers_std": float(np.std(trigger_counts)),
            "avg_mse_mean": float(np.mean(avg_mses)),
            "avg_mse_std": float(np.std(avg_mses)),
        }
        r = results[f"{thresh_mult:.2f}x"]
        print(
            f"  {thresh_mult:.2f}x: triggers={r['triggers_mean']:.1f}±{r['triggers_std']:.1f}, "
            f"avg_mse={r['avg_mse_mean']:.2f}±{r['avg_mse_std']:.2f}"
        )
    return results


def exp3_mse_vs_entropy_trigger(
    dim: int,
    n_samples: int,
    n_codes: int,
    n_stages: int,
    n_steps: int,
    drift_per_step: float,
    seeds: int,
) -> dict:
    """Compare MSE-threshold vs entropy-drop trigger."""
    print("\n=== Exp 3: MSE vs entropy trigger ===")
    results = {"mse_trigger": [], "entropy_trigger": []}
    freeze_depth = n_stages // 2

    for seed in range(seeds):
        X0 = generate_data(n_samples, dim, seed=seed)
        rq0 = RQCodebook(n_stages, n_codes, dim).fit(X0, seed=seed)
        mse_baseline = rq0.mse(X0)
        ent_baseline = sum(codebook_entropy(rq0, X0))
        mse_thresh = mse_baseline * 1.5
        ent_thresh = ent_baseline * 0.85

        for trigger_type, thresh_check in [
            ("mse_trigger", lambda rq, X: rq.mse(X) > mse_thresh),
            ("entropy_trigger", lambda rq, X: sum(codebook_entropy(rq, X)) < ent_thresh),
        ]:
            rq_cur = rq0
            n_triggers = 0
            mses = []
            for step in range(1, n_steps + 1):
                rng = np.random.RandomState(seed + 1)
                total_drift = np.zeros(dim, dtype=np.float32)
                for t in range(step):
                    d = rng.randn(dim).astype(np.float32)
                    d /= np.linalg.norm(d)
                    total_drift += d * drift_per_step
                X_t = X0 + total_drift

                if thresh_check(rq_cur, X_t):
                    rq_cur = warm_retrain(
                        rq_cur, X_t, freeze_depth=freeze_depth, seed=seed + step
                    )
                    n_triggers += 1
                mses.append(rq_cur.mse(X_t))

            results[trigger_type].append({
                "seed": seed,
                "triggers": n_triggers,
                "avg_mse": float(np.mean(mses)),
                "final_mse": float(mses[-1]),
            })

    for tt in ["mse_trigger", "entropy_trigger"]:
        trigs = [r["triggers"] for r in results[tt]]
        mses = [r["avg_mse"] for r in results[tt]]
        print(
            f"  {tt}: triggers={np.mean(trigs):.1f}±{np.std(trigs):.1f}, "
            f"avg_mse={np.mean(mses):.2f}±{np.std(mses):.2f}"
        )
    return results


def exp4_non_additivity(
    dim: int,
    n_samples: int,
    n_codes: int,
    n_stages: int,
    drift_mag: float,
    seeds: int,
) -> dict:
    """Test whether stage gains are additive (they shouldn't be)."""
    print("\n=== Exp 4: Stage interaction / non-additivity ===")
    results = []
    for seed in range(seeds):
        X0 = generate_data(n_samples, dim, seed=seed)
        rng = np.random.RandomState(seed + 9999)
        direction = rng.randn(dim).astype(np.float32)
        direction /= np.linalg.norm(direction)
        X1 = X0 + direction * drift_mag

        rq0 = RQCodebook(n_stages, n_codes, dim).fit(X0, seed=seed)
        mse_frozen = rq0.mse(X1)

        rq_full = RQCodebook(n_stages, n_codes, dim).fit(X1, seed=seed + 500)
        mse_full = rq_full.mse(X1)

        gains_individual = {}
        for s_adapt in range(n_stages):
            rq_one = RQCodebook(n_stages, n_codes, dim)
            rq_one.codebooks = [cb.copy() for cb in rq0.codebooks]
            residual = X1.copy()
            for m in range(n_stages):
                if m == s_adapt:
                    from rq import _kmeans, _assign
                    centroids = _kmeans(
                        residual, n_codes, n_iter=20,
                        rng=np.random.RandomState(seed + m),
                        init=rq_one.codebooks[m],
                    )
                    rq_one.codebooks[m] = centroids
                    assignments = _assign(residual, centroids)
                else:
                    assignments = _assign(residual, rq_one.codebooks[m])
                residual = residual - rq_one.codebooks[m][assignments]
            mse_one = rq_one.mse(X1)
            gains_individual[s_adapt] = mse_frozen - mse_one

        sum_individual = sum(gains_individual.values())

        rq_all = warm_retrain(rq0, X1, freeze_depth=0, seed=seed + 500)
        mse_all_warm = rq_all.mse(X1)
        gain_all = mse_frozen - mse_all_warm

        row = {
            "seed": seed,
            "mse_frozen": mse_frozen,
            "mse_full": mse_full,
            "individual_gains": {str(k): v for k, v in gains_individual.items()},
            "sum_individual_gains": sum_individual,
            "joint_gain": gain_all,
            "additivity_ratio": sum_individual / max(gain_all, 1e-12),
        }
        results.append(row)
        print(
            f"  seed={seed}: sum_individual={sum_individual:.2f}, "
            f"joint={gain_all:.2f}, ratio={row['additivity_ratio']:.3f}"
        )

    ratios = [r["additivity_ratio"] for r in results]
    print(
        f"  Additivity ratio: {np.mean(ratios):.3f}±{np.std(ratios):.3f} "
        f"(1.0 = perfectly additive)"
    )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Support experiments for revised freeze-depth analysis"
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--n-codes", type=int, default=64)
    parser.add_argument("--n-stages", type=int, default=4)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parents[1] / "results" / "freeze_depth_support"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    all_results["exp1_monotonicity"] = exp1_monotonicity(
        dims=[27, 64, 128],
        n_samples=args.n_samples,
        n_codes=args.n_codes,
        n_stages=args.n_stages,
        seeds=args.seeds,
        drift_mag=5.0,
    )

    all_results["exp2_trigger_sweep"] = exp2_trigger_sweep(
        dim=128,
        n_samples=args.n_samples,
        n_codes=args.n_codes,
        n_stages=args.n_stages,
        n_steps=10,
        drift_per_step=0.5,
        thresholds=[1.05, 1.1, 1.2, 1.5, 2.0, 3.0],
        seeds=args.seeds,
    )

    all_results["exp3_trigger_comparison"] = exp3_mse_vs_entropy_trigger(
        dim=128,
        n_samples=args.n_samples,
        n_codes=args.n_codes,
        n_stages=args.n_stages,
        n_steps=10,
        drift_per_step=0.5,
        seeds=args.seeds,
    )

    all_results["exp4_non_additivity"] = exp4_non_additivity(
        dim=64,
        n_samples=args.n_samples,
        n_codes=args.n_codes,
        n_stages=args.n_stages,
        drift_mag=5.0,
        seeds=args.seeds,
    )

    out_path = output_dir / "freeze_depth_support.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

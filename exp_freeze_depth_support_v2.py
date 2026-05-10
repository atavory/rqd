#!/usr/bin/env python3
"""Support experiments v2: stronger drift + freeze-depth sweep across consumers.

Experiments:
  1. Trigger-threshold sweep with stronger drift (2.0, 3.0 per step)
  2. MSE vs entropy trigger with stronger drift
  3. Freeze-depth sweep across consumer types (pair consumer at s=1,2,3)
  4. Non-additivity cross-check with different dims and drift magnitudes
  5. Freeze-depth MSE accumulation over time (s=1,2,3 streaming)

Usage:
    python3 exp_freeze_depth_support_v2.py
    python3 exp_freeze_depth_support_v2.py --seeds 5
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
    _kmeans,
    _assign,
)


def random_walk_drift(
    X_base: np.ndarray, step: int, drift_per_step: float, seed: int
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    dim = X_base.shape[1]
    total_drift = np.zeros(dim, dtype=np.float32)
    for t in range(step):
        d = rng.randn(dim).astype(np.float32)
        d /= np.linalg.norm(d)
        total_drift += d * drift_per_step
    return X_base + total_drift


def exp1_trigger_strong_drift(
    dims: list[int],
    n_samples: int,
    n_codes: int,
    n_stages: int,
    n_steps: int,
    drift_rates: list[float],
    thresholds: list[float],
    seeds: int,
) -> dict:
    """Trigger sweep with multiple drift rates."""
    print("=== Exp 1: Trigger sweep (strong drift) ===")
    results = {}
    for dim in dims:
        results[dim] = {}
        for drift in drift_rates:
            results[dim][drift] = {}
            print(f"  dim={dim}, drift={drift}")
            for thresh in thresholds:
                trigger_counts = []
                avg_mses = []
                freeze_depth = n_stages // 2
                for seed in range(seeds):
                    X0 = generate_data(n_samples, dim, seed=seed)
                    rq0 = RQCodebook(n_stages, n_codes, dim).fit(X0, seed=seed)
                    mse_baseline = rq0.mse(X0)
                    threshold = mse_baseline * thresh

                    rq_cur = rq0
                    n_triggers = 0
                    mses = []
                    for step in range(1, n_steps + 1):
                        X_t = random_walk_drift(X0, step, drift, seed + 1)
                        if rq_cur.mse(X_t) > threshold:
                            rq_cur = warm_retrain(
                                rq_cur, X_t, freeze_depth=freeze_depth,
                                seed=seed + step,
                            )
                            n_triggers += 1
                        mses.append(rq_cur.mse(X_t))
                    trigger_counts.append(n_triggers)
                    avg_mses.append(float(np.mean(mses)))

                results[dim][drift][f"{thresh:.2f}x"] = {
                    "triggers_mean": float(np.mean(trigger_counts)),
                    "triggers_std": float(np.std(trigger_counts)),
                    "avg_mse_mean": float(np.mean(avg_mses)),
                    "avg_mse_std": float(np.std(avg_mses)),
                }
                r = results[dim][drift][f"{thresh:.2f}x"]
                print(
                    f"    {thresh:.2f}x: triggers={r['triggers_mean']:.1f}±{r['triggers_std']:.1f}, "
                    f"avg_mse={r['avg_mse_mean']:.2f}"
                )
    return results


def exp2_mse_vs_entropy_strong(
    dim: int,
    n_samples: int,
    n_codes: int,
    n_stages: int,
    n_steps: int,
    drift_rates: list[float],
    seeds: int,
) -> dict:
    """MSE vs entropy trigger with strong drift."""
    print("\n=== Exp 2: MSE vs entropy trigger (strong drift) ===")
    results = {}
    freeze_depth = n_stages // 2
    for drift in drift_rates:
        results[drift] = {"mse_trigger": [], "entropy_trigger": []}
        for seed in range(seeds):
            X0 = generate_data(n_samples, dim, seed=seed)
            rq0 = RQCodebook(n_stages, n_codes, dim).fit(X0, seed=seed)
            mse_baseline = rq0.mse(X0)
            ent_baseline = sum(codebook_entropy(rq0, X0))
            mse_thresh = mse_baseline * 1.3
            ent_thresh = ent_baseline * 0.90

            for trigger_type, check_fn in [
                ("mse_trigger", lambda rq, X: rq.mse(X) > mse_thresh),
                ("entropy_trigger", lambda rq, X: sum(codebook_entropy(rq, X)) < ent_thresh),
            ]:
                rq_cur = rq0
                n_triggers = 0
                mses = []
                for step in range(1, n_steps + 1):
                    X_t = random_walk_drift(X0, step, drift, seed + 1)
                    if check_fn(rq_cur, X_t):
                        rq_cur = warm_retrain(
                            rq_cur, X_t, freeze_depth=freeze_depth,
                            seed=seed + step,
                        )
                        n_triggers += 1
                    mses.append(rq_cur.mse(X_t))
                results[drift][trigger_type].append({
                    "seed": seed,
                    "triggers": n_triggers,
                    "avg_mse": float(np.mean(mses)),
                    "final_mse": float(mses[-1]),
                })

        for tt in ["mse_trigger", "entropy_trigger"]:
            trigs = [r["triggers"] for r in results[drift][tt]]
            mses = [r["avg_mse"] for r in results[drift][tt]]
            print(
                f"  drift={drift}, {tt}: triggers={np.mean(trigs):.1f}±{np.std(trigs):.1f}, "
                f"avg_mse={np.mean(mses):.2f}"
            )
    return results


def exp3_freeze_depth_consumer_sweep(
    n_samples: int,
    n_codes: int,
    n_stages: int,
    drift_mag: float,
    seeds: int,
) -> dict:
    """Sweep freeze depth and measure downstream proxy breakage.

    Simulates a "pair consumer" that depends on the first s tokens being stable.
    For each freeze depth, measures what fraction of prefix-pair IDs changed.
    """
    print("\n=== Exp 3: Freeze-depth sweep — prefix stability by depth ===")
    dim = 64
    results = {}
    for s in range(n_stages + 1):
        pfx_changes = []
        mses = []
        for seed in range(seeds):
            X0 = generate_data(n_samples, dim, seed=seed)
            rng = np.random.RandomState(seed + 9999)
            direction = rng.randn(dim).astype(np.float32)
            direction /= np.linalg.norm(direction)
            X1 = X0 + direction * drift_mag

            rq0 = RQCodebook(n_stages, n_codes, dim).fit(X0, seed=seed)
            if s == 0:
                rq_s = RQCodebook(n_stages, n_codes, dim).fit(X1, seed=seed + 500)
            else:
                rq_s = warm_retrain(rq0, X1, freeze_depth=s, seed=seed + s)

            mses.append(rq_s.mse(X1))

            for check_depth in range(1, n_stages + 1):
                codes_old = rq0.encode(X1, n_stages=check_depth)
                codes_new = rq_s.encode(X1, n_stages=check_depth)
                old_ids = np.column_stack(codes_old)
                new_ids = np.column_stack(codes_new)
                change_frac = float(np.any(old_ids != new_ids, axis=1).mean())
                key = f"pfx_change_depth_{check_depth}"
                if key not in results.get(s, {}):
                    results.setdefault(s, {})[key] = []
                results[s][key].append(change_frac)

        results.setdefault(s, {})["mse"] = mses
        m = np.mean(mses)
        print(f"  freeze_depth={s}: MSE={m:.2f}")
        for check_depth in range(1, n_stages + 1):
            key = f"pfx_change_depth_{check_depth}"
            vals = results[s][key]
            print(f"    tokens 1:{check_depth} changed: {np.mean(vals):.3f}±{np.std(vals):.3f}")

    summary = {}
    for s in range(n_stages + 1):
        summary[s] = {
            "mse_mean": float(np.mean(results[s]["mse"])),
            "mse_std": float(np.std(results[s]["mse"])),
        }
        for check_depth in range(1, n_stages + 1):
            key = f"pfx_change_depth_{check_depth}"
            summary[s][f"change_depth_{check_depth}_mean"] = float(np.mean(results[s][key]))
            summary[s][f"change_depth_{check_depth}_std"] = float(np.std(results[s][key]))
    return summary


def exp4_nonadditivity_crosscheck(
    dims: list[int],
    n_samples: int,
    n_codes: int,
    n_stages: int,
    drift_mags: list[float],
    seeds: int,
) -> dict:
    """Cross-check non-additivity at multiple dims and drift magnitudes."""
    print("\n=== Exp 4: Non-additivity cross-check ===")
    results = {}
    for dim in dims:
        results[dim] = {}
        for drift_mag in drift_mags:
            ratios = []
            for seed in range(seeds):
                X0 = generate_data(n_samples, dim, seed=seed)
                rng = np.random.RandomState(seed + 9999)
                direction = rng.randn(dim).astype(np.float32)
                direction /= np.linalg.norm(direction)
                X1 = X0 + direction * drift_mag

                rq0 = RQCodebook(n_stages, n_codes, dim).fit(X0, seed=seed)
                mse_frozen = rq0.mse(X1)

                gains_ind = []
                for s_adapt in range(n_stages):
                    rq_one = RQCodebook(n_stages, n_codes, dim)
                    rq_one.codebooks = [cb.copy() for cb in rq0.codebooks]
                    residual = X1.copy()
                    for m in range(n_stages):
                        if m == s_adapt:
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
                    gains_ind.append(mse_frozen - rq_one.mse(X1))

                rq_all = warm_retrain(rq0, X1, freeze_depth=0, seed=seed + 500)
                gain_all = mse_frozen - rq_all.mse(X1)
                ratio = sum(gains_ind) / max(gain_all, 1e-12)
                ratios.append(ratio)

            results[dim][drift_mag] = {
                "ratio_mean": float(np.mean(ratios)),
                "ratio_std": float(np.std(ratios)),
            }
            r = results[dim][drift_mag]
            print(f"  dim={dim}, drift={drift_mag}: ratio={r['ratio_mean']:.3f}±{r['ratio_std']:.3f}")
    return results


def exp5_streaming_by_freeze_depth(
    dim: int,
    n_samples: int,
    n_codes: int,
    n_stages: int,
    n_steps: int,
    drift_per_step: float,
    freeze_depths: list[int],
    seeds: int,
) -> dict:
    """Streaming MSE accumulation at different freeze depths."""
    print("\n=== Exp 5: Streaming MSE by freeze depth ===")
    results = {}
    for s in freeze_depths:
        step_mses = {step: [] for step in range(n_steps + 1)}
        for seed in range(seeds):
            X0 = generate_data(n_samples, dim, seed=seed)
            rq0 = RQCodebook(n_stages, n_codes, dim).fit(X0, seed=seed)

            rq_cur = rq0
            for step in range(n_steps + 1):
                X_t = random_walk_drift(X0, step, drift_per_step, seed + 1)
                if step > 0:
                    rq_cur = warm_retrain(
                        rq_cur, X_t, freeze_depth=s, seed=seed + step
                    )
                step_mses[step].append(rq_cur.mse(X_t))

        results[s] = {}
        print(f"  s={s}:", end="")
        for step in range(n_steps + 1):
            m = np.mean(step_mses[step])
            results[s][step] = {
                "mean": float(m),
                "std": float(np.std(step_mses[step])),
            }
            if step % 3 == 0:
                print(f" T{step}={m:.1f}", end="")
        print()

    # Also run full retrain and frozen baselines
    for label, do_retrain in [("full", True), ("frozen", False)]:
        step_mses = {step: [] for step in range(n_steps + 1)}
        for seed in range(seeds):
            X0 = generate_data(n_samples, dim, seed=seed)
            rq0 = RQCodebook(n_stages, n_codes, dim).fit(X0, seed=seed)
            for step in range(n_steps + 1):
                X_t = random_walk_drift(X0, step, drift_per_step, seed + 1)
                if do_retrain and step > 0:
                    rq_t = RQCodebook(n_stages, n_codes, dim).fit(
                        X_t, seed=seed + step + 100
                    )
                    step_mses[step].append(rq_t.mse(X_t))
                else:
                    step_mses[step].append(rq0.mse(X_t))
        results[label] = {}
        print(f"  {label}:", end="")
        for step in range(n_steps + 1):
            m = np.mean(step_mses[step])
            results[label][step] = {
                "mean": float(m),
                "std": float(np.std(step_mses[step])),
            }
            if step % 3 == 0:
                print(f" T{step}={m:.1f}", end="")
        print()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--n-codes", type=int, default=64)
    parser.add_argument("--n-stages", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parents[1] / "results" / "freeze_depth_support"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    all_results["exp1_trigger_strong"] = exp1_trigger_strong_drift(
        dims=[64, 128],
        n_samples=args.n_samples,
        n_codes=args.n_codes,
        n_stages=args.n_stages,
        n_steps=15,
        drift_rates=[2.0, 3.0, 5.0],
        thresholds=[1.1, 1.2, 1.5, 2.0, 3.0],
        seeds=args.seeds,
    )

    all_results["exp2_trigger_mse_vs_entropy"] = exp2_mse_vs_entropy_strong(
        dim=64,
        n_samples=args.n_samples,
        n_codes=args.n_codes,
        n_stages=args.n_stages,
        n_steps=15,
        drift_rates=[2.0, 3.0, 5.0],
        seeds=args.seeds,
    )

    all_results["exp3_freeze_depth_consumer"] = exp3_freeze_depth_consumer_sweep(
        n_samples=args.n_samples,
        n_codes=args.n_codes,
        n_stages=args.n_stages,
        drift_mag=5.0,
        seeds=args.seeds,
    )

    all_results["exp4_nonadditivity_cross"] = exp4_nonadditivity_crosscheck(
        dims=[27, 64, 128],
        n_samples=args.n_samples,
        n_codes=args.n_codes,
        n_stages=args.n_stages,
        drift_mags=[2.0, 5.0, 10.0],
        seeds=args.seeds,
    )

    all_results["exp5_streaming_by_depth"] = exp5_streaming_by_freeze_depth(
        dim=64,
        n_samples=args.n_samples,
        n_codes=args.n_codes,
        n_stages=args.n_stages,
        n_steps=15,
        drift_per_step=2.0,
        freeze_depths=[1, 2, 3],
        seeds=args.seeds,
    )

    out_path = output_dir / "freeze_depth_support_v2.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Freeze-depth support experiments v3: Optuna + incremental JSON logging.

Experiments:
  1. Trigger-threshold optimization (Optuna finds best threshold per drift rate)
  2. MSE vs entropy trigger comparison
  3. Freeze-depth prefix stability sweep
  4. Non-additivity cross-check
  5. Streaming MSE by freeze depth

All results written incrementally to a JSON file after each trial/config.

Usage:
    python3 exp_freeze_depth_v3.py
    python3 exp_freeze_depth_v3.py --seeds 5
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import optuna

from rq import (
    RQCodebook,
    warm_retrain,
    codebook_entropy,
    generate_data,
    _kmeans,
    _assign,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "freeze_depth_support"


def save(results: dict, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved {path}")


def random_walk_drift(
    X: np.ndarray, step: int, drift_per_step: float, seed: int
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    dim = X.shape[1]
    td = np.zeros(dim, dtype=np.float32)
    for t in range(step):
        d = rng.randn(dim).astype(np.float32)
        d /= np.linalg.norm(d)
        td += d * drift_per_step
    return X + td


def run_streaming(
    X0, rq0, drift_rate, n_steps, freeze_depth, threshold, seed, trigger_type="mse"
):
    mse_baseline = rq0.mse(X0)
    ent_baseline = sum(codebook_entropy(rq0, X0))
    dim = X0.shape[1]
    rq_cur = rq0
    n_triggers = 0
    mses = []
    for step in range(1, n_steps + 1):
        X_t = random_walk_drift(X0, step, drift_rate, seed + 1)
        if trigger_type == "mse":
            fire = rq_cur.mse(X_t) > mse_baseline * threshold
        else:
            fire = sum(codebook_entropy(rq_cur, X_t)) < ent_baseline * threshold
        if fire:
            rq_cur = warm_retrain(
                rq_cur, X_t, freeze_depth=freeze_depth, seed=seed + step
            )
            n_triggers += 1
        mses.append(rq_cur.mse(X_t))
    return n_triggers, mses


def exp1_trigger_optuna(
    dim: int, n_samples: int, n_codes: int, n_stages: int,
    n_steps: int, drift_rates: list[float], seeds: int,
) -> dict:
    log.info("=== Exp 1: Trigger threshold (Optuna) ===")
    results = {}

    for drift in drift_rates:
        def objective(trial):
            thresh = trial.suggest_float("threshold", 1.05, 3.0)
            total_mse = 0.0
            total_triggers = 0
            for seed in range(seeds):
                X0 = generate_data(n_samples, dim, seed=seed)
                rq0 = RQCodebook(n_stages, n_codes, dim).fit(X0, seed=seed)
                nt, mses = run_streaming(
                    X0, rq0, drift, n_steps, n_stages // 2, thresh, seed
                )
                total_mse += np.mean(mses)
                total_triggers += nt
            avg_mse = total_mse / seeds
            avg_triggers = total_triggers / seeds
            trial.set_user_attr("avg_triggers", avg_triggers)
            return avg_mse

        def callback(study, trial):
            results[str(drift)] = {
                "best_threshold": study.best_trial.params["threshold"],
                "best_avg_mse": study.best_trial.value,
                "best_avg_triggers": study.best_trial.user_attrs["avg_triggers"],
                "all_trials": [
                    {
                        "threshold": t.params["threshold"],
                        "avg_mse": t.value,
                        "avg_triggers": t.user_attrs.get("avg_triggers", None),
                    }
                    for t in study.trials
                ],
            }
            save(results, "exp1_trigger_optuna")
            log.info(
                f"  drift={drift} trial {trial.number}: "
                f"thresh={trial.params['threshold']:.3f}, "
                f"mse={trial.value:.2f}, "
                f"triggers={trial.user_attrs['avg_triggers']:.1f}"
            )

        study = optuna.create_study(
            direction="minimize",
            study_name=f"trigger_drift_{drift}",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=30, callbacks=[callback])

    return results


def exp2_mse_vs_entropy(
    dim: int, n_samples: int, n_codes: int, n_stages: int,
    n_steps: int, drift_rates: list[float], seeds: int,
) -> dict:
    log.info("=== Exp 2: MSE vs entropy trigger ===")
    results = {}
    freeze_depth = n_stages // 2

    for drift in drift_rates:
        results[str(drift)] = {}
        for ttype, thresh_val in [("mse", 1.3), ("entropy", 0.90)]:
            rows = []
            for seed in range(seeds):
                X0 = generate_data(n_samples, dim, seed=seed)
                rq0 = RQCodebook(n_stages, n_codes, dim).fit(X0, seed=seed)
                nt, mses = run_streaming(
                    X0, rq0, drift, n_steps, freeze_depth, thresh_val, seed, ttype
                )
                rows.append({
                    "seed": seed, "triggers": nt,
                    "avg_mse": float(np.mean(mses)),
                    "final_mse": float(mses[-1]),
                })
            results[str(drift)][ttype] = rows
            trigs = [r["triggers"] for r in rows]
            avg = [r["avg_mse"] for r in rows]
            log.info(
                f"  drift={drift} {ttype}: triggers={np.mean(trigs):.1f}±{np.std(trigs):.1f}, "
                f"avg_mse={np.mean(avg):.1f}"
            )
        save(results, "exp2_mse_vs_entropy")

    return results


def exp3_prefix_stability(
    dim: int, n_samples: int, n_codes: int, n_stages: int,
    drift_mag: float, seeds: int,
) -> dict:
    log.info("=== Exp 3: Freeze-depth prefix stability ===")
    results = {}

    for s in range(n_stages + 1):
        rows = []
        for seed in range(seeds):
            X0 = generate_data(n_samples, dim, seed=seed)
            rng = np.random.RandomState(seed + 9999)
            dr = rng.randn(dim).astype(np.float32)
            dr /= np.linalg.norm(dr)
            X1 = X0 + dr * drift_mag

            rq0 = RQCodebook(n_stages, n_codes, dim).fit(X0, seed=seed)
            if s == 0:
                rq_s = RQCodebook(n_stages, n_codes, dim).fit(X1, seed=seed + 500)
            else:
                rq_s = warm_retrain(rq0, X1, freeze_depth=s, seed=seed + s)

            row = {"seed": seed, "mse": rq_s.mse(X1)}
            for cd in range(1, n_stages + 1):
                c_old = np.column_stack(rq0.encode(X1, n_stages=cd))
                c_new = np.column_stack(rq_s.encode(X1, n_stages=cd))
                row[f"tok1:{cd}_changed"] = float(
                    np.any(c_old != c_new, axis=1).mean()
                )
            rows.append(row)

        results[s] = rows
        m = np.mean([r["mse"] for r in rows])
        c1 = np.mean([r["tok1:1_changed"] for r in rows])
        c2 = np.mean([r["tok1:2_changed"] for r in rows])
        log.info(f"  s={s}: MSE={m:.1f}  tok1_chg={c1:.3f}  tok1:2_chg={c2:.3f}")
        save(results, "exp3_prefix_stability")

    return results


def exp4_nonadditivity(
    dims: list[int], n_samples: int, n_codes: int, n_stages: int,
    drift_mags: list[float], seeds: int,
) -> dict:
    log.info("=== Exp 4: Non-additivity cross-check ===")
    results = {}

    for dim in dims:
        results[dim] = {}
        for dm in drift_mags:
            ratios = []
            for seed in range(seeds):
                X0 = generate_data(n_samples, dim, seed=seed)
                rng = np.random.RandomState(seed + 9999)
                dr = rng.randn(dim).astype(np.float32)
                dr /= np.linalg.norm(dr)
                X1 = X0 + dr * dm

                rq0 = RQCodebook(n_stages, n_codes, dim).fit(X0, seed=seed)
                mf = rq0.mse(X1)

                gi = []
                for sa in range(n_stages):
                    rq1 = RQCodebook(n_stages, n_codes, dim)
                    rq1.codebooks = [cb.copy() for cb in rq0.codebooks]
                    res = X1.copy()
                    for m in range(n_stages):
                        if m == sa:
                            c = _kmeans(
                                res, n_codes, n_iter=20,
                                rng=np.random.RandomState(seed + m),
                                init=rq1.codebooks[m],
                            )
                            rq1.codebooks[m] = c
                            a = _assign(res, c)
                        else:
                            a = _assign(res, rq1.codebooks[m])
                        res = res - rq1.codebooks[m][a]
                    gi.append(mf - rq1.mse(X1))

                rqa = warm_retrain(rq0, X1, freeze_depth=0, seed=seed + 500)
                ga = mf - rqa.mse(X1)
                ratios.append(sum(gi) / max(ga, 1e-12))

            results[dim][dm] = {
                "ratio_mean": float(np.mean(ratios)),
                "ratio_std": float(np.std(ratios)),
            }
            log.info(
                f"  d={dim} drift={dm}: ratio={np.mean(ratios):.2f}±{np.std(ratios):.2f}"
            )
            save(results, "exp4_nonadditivity")

    return results


def exp5_streaming_by_depth(
    dim: int, n_samples: int, n_codes: int, n_stages: int,
    n_steps: int, drift_rate: float, seeds: int,
) -> dict:
    log.info("=== Exp 5: Streaming MSE by freeze depth ===")
    results = {}

    for label in [1, 2, 3, "full", "frozen"]:
        step_mses = {st: [] for st in range(n_steps + 1)}
        for seed in range(seeds):
            X0 = generate_data(n_samples, dim, seed=seed)
            rq0 = RQCodebook(n_stages, n_codes, dim).fit(X0, seed=seed)
            rq_cur = rq0
            for step in range(n_steps + 1):
                X_t = random_walk_drift(X0, step, drift_rate, seed + 1)
                if step > 0:
                    if label == "full":
                        rq_t = RQCodebook(n_stages, n_codes, dim).fit(
                            X_t, seed=seed + step + 100
                        )
                        step_mses[step].append(rq_t.mse(X_t))
                        continue
                    elif label == "frozen":
                        step_mses[step].append(rq0.mse(X_t))
                        continue
                    else:
                        rq_cur = warm_retrain(
                            rq_cur, X_t, freeze_depth=label, seed=seed + step
                        )
                step_mses[step].append(rq_cur.mse(X_t))

        results[str(label)] = {
            step: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for step, v in step_mses.items()
        }
        vals = " ".join(
            f"T{s}={np.mean(step_mses[s]):.1f}" for s in [0, 3, 5, 7, 10]
        )
        log.info(f"  {str(label):>6s}: {vals}")
        save(results, "exp5_streaming_by_depth")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--n-codes", type=int, default=64)
    parser.add_argument("--n-stages", type=int, default=4)
    args = parser.parse_args()

    t0 = time.time()

    exp1_trigger_optuna(
        dim=64, n_samples=args.n_samples, n_codes=args.n_codes,
        n_stages=args.n_stages, n_steps=10,
        drift_rates=[3.0, 5.0], seeds=args.seeds,
    )

    exp2_mse_vs_entropy(
        dim=64, n_samples=args.n_samples, n_codes=args.n_codes,
        n_stages=args.n_stages, n_steps=10,
        drift_rates=[3.0, 5.0], seeds=args.seeds,
    )

    exp3_prefix_stability(
        dim=64, n_samples=args.n_samples, n_codes=args.n_codes,
        n_stages=args.n_stages, drift_mag=5.0, seeds=args.seeds,
    )

    exp4_nonadditivity(
        dims=[27, 64, 128], n_samples=args.n_samples,
        n_codes=args.n_codes, n_stages=args.n_stages,
        drift_mags=[2.0, 5.0, 10.0], seeds=args.seeds,
    )

    exp5_streaming_by_depth(
        dim=64, n_samples=args.n_samples, n_codes=args.n_codes,
        n_stages=args.n_stages, n_steps=10, drift_rate=3.0,
        seeds=args.seeds,
    )

    log.info(f"All done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

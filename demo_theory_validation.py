#!/usr/bin/env python3
"""Theory validation experiments for the new analysis section.

Measures five quantities predicted by the theory:
1. Stagewise gain-cost ratios (Theorem 1: suffix optimality)
2. Drift spectrum on the code tree (Theorem 2: coarse-drift identity)
3. Cross-subtree transport mass epsilon_s (Theorem 3: transport sandwich)
4. Cumulative transport for restart threshold (Corollary: restart)
5. Phase diagram: optimal freeze depth vs stability price lambda

Usage:
    python3 demo_theory_validation.py
    python3 demo_theory_validation.py --dim 64 --drift 2.0 --seeds 10
    python3 demo_theory_validation.py --output results/theory_validation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from rq import (
    RQCodebook,
    _assign,
    apply_drift,
    gap_recovery,
    generate_data,
    warm_retrain,
)


def stagewise_gain_cost(
    X_source: np.ndarray,
    X_target: np.ndarray,
    m: int = 4,
    K: int = 64,
    n_iter: int = 20,
    seed: int = 42,
) -> dict:
    """Measure stagewise distortion gain g_i and churn kappa_i.

    g_i: MSE reduction from adapting stage i (holding earlier stages frozen).
    kappa_i: fraction of points whose stage-i code changes when adapted.
    """
    codes_per_stage = [K] * m
    rng = np.random.RandomState(seed)

    rq_source = RQCodebook(m, codes_per_stage, X_source.shape[1])
    rq_source.fit(X_source, n_iter=n_iter, seed=seed)

    rq_full = RQCodebook(m, codes_per_stage, X_source.shape[1])
    rq_full.fit(X_target, n_iter=n_iter, seed=seed + 1000)

    mse_frozen = rq_source.mse(X_target)
    mse_full = rq_full.mse(X_target)

    gains = []
    churns = []
    ratios = []

    for s in range(m):
        rq_warm_s = warm_retrain(rq_source, X_target, freeze_depth=s, n_iter=n_iter, seed=seed)
        rq_warm_s1 = warm_retrain(rq_source, X_target, freeze_depth=s + 1, n_iter=n_iter, seed=seed)

        mse_s = rq_warm_s.mse(X_target)
        mse_s1 = rq_warm_s1.mse(X_target)
        g_i = mse_s1 - mse_s

        residual = X_target.copy()
        for j in range(s):
            a = _assign(residual, rq_source.codebooks[j])
            residual -= rq_source.codebooks[j][a]

        a_old = _assign(residual, rq_source.codebooks[s])
        a_new = _assign(residual, rq_warm_s.codebooks[s])
        kappa_i = float(np.mean(a_old != a_new))

        gains.append(float(g_i))
        churns.append(float(kappa_i))
        ratios.append(float(g_i / max(kappa_i, 1e-12)))

    return {
        "gains": gains,
        "churns": churns,
        "ratios": ratios,
        "monotone": all(ratios[i] <= ratios[i + 1] + 1e-8 for i in range(len(ratios) - 1)),
    }


def _conditional_means_on_tree(
    X: np.ndarray,
    rq: RQCodebook,
    m: int,
) -> list[np.ndarray]:
    """Compute E[X | F_i] for each depth i using the source codebook's tree.

    Returns a list of arrays, each (n, d), giving the conditional mean of X
    at each filtration depth. F_0 = trivial (global mean), F_i = first i codes.
    """
    n, d = X.shape
    codes = rq.encode(X, n_stages=m)

    means = [np.full((n, d), X.mean(axis=0), dtype=np.float32)]

    for depth in range(1, m + 1):
        prefix_keys = np.zeros(n, dtype=np.int64)
        multiplier = 1
        for i in range(depth):
            prefix_keys += codes[i] * multiplier
            multiplier *= rq.codes_per_stage[i]

        unique_keys = np.unique(prefix_keys)
        conditional = np.zeros((n, d), dtype=np.float32)
        for key in unique_keys:
            mask = prefix_keys == key
            conditional[mask] = X[mask].mean(axis=0)
        means.append(conditional)

    return means


def drift_spectrum(
    X_source: np.ndarray,
    X_target: np.ndarray,
    m: int = 4,
    K: int = 64,
    n_iter: int = 20,
    seed: int = 42,
) -> dict:
    """Measure the drift spectrum: E[||Delta_i^(t) - Delta_i^(0)||^2] per stage.

    Theorem 2 defines Delta_i^(t) = E[X_t | F_i] - E[X_t | F_{i-1}], the
    martingale increment of the target data on the *source* codebook's tree.
    The coarse drift energy is sum_{i<=s} E[||Delta_i^(t) - Delta_i^(0)||^2].

    We use the source codebook's filtration for both source and target data,
    which is what the theorem requires (same tree, different distributions).
    """
    codes_per_stage = [K] * m

    rq_source = RQCodebook(m, codes_per_stage, X_source.shape[1])
    rq_source.fit(X_source, n_iter=n_iter, seed=seed)

    source_means = _conditional_means_on_tree(X_source, rq_source, m)
    target_means = _conditional_means_on_tree(X_target, rq_source, m)

    drift_energies = []
    for i in range(1, m + 1):
        source_increment = source_means[i] - source_means[i - 1]
        target_increment = target_means[i] - target_means[i - 1]
        diff = target_increment - source_increment
        energy = float(np.mean(np.sum(diff**2, axis=1)))
        drift_energies.append(energy)

    rq_full = RQCodebook(m, codes_per_stage, X_source.shape[1])
    rq_full.fit(X_target, n_iter=n_iter, seed=seed + 1000)
    source_on_target = rq_source.encode(X_target, n_stages=m // 2)
    full_codes = rq_full.encode(X_target, n_stages=m // 2)
    routed_mask = np.ones(len(X_target), dtype=bool)
    for i in range(m // 2):
        routed_mask &= source_on_target[i] == full_codes[i]

    total = sum(drift_energies)
    s = m // 2
    prefix_energy = sum(drift_energies[:s])
    suffix_energy = sum(drift_energies[s:])

    return {
        "per_stage": drift_energies,
        "prefix_fraction": float(prefix_energy / max(total, 1e-12)),
        "suffix_fraction": float(suffix_energy / max(total, 1e-12)),
        "routed_fraction": float(routed_mask.mean()),
    }


def cross_subtree_transport(
    X_source: np.ndarray,
    X_target: np.ndarray,
    m: int = 4,
    K: int = 64,
    freeze_depth: int = 2,
    n_iter: int = 20,
    seed: int = 42,
) -> dict:
    """Measure cross-subtree transport mass epsilon_s."""
    codes_per_stage = [K] * m

    rq_source = RQCodebook(m, codes_per_stage, X_source.shape[1])
    rq_source.fit(X_source, n_iter=n_iter, seed=seed)

    rq_full = RQCodebook(m, codes_per_stage, X_source.shape[1])
    rq_full.fit(X_target, n_iter=n_iter, seed=seed + 1000)

    frozen_codes = rq_source.encode(X_target)
    full_codes = rq_full.encode(X_target)

    cross_subtree = np.zeros(len(X_target), dtype=bool)
    for i in range(freeze_depth):
        cross_subtree |= frozen_codes[i] != full_codes[i]

    epsilon_s = float(cross_subtree.mean())

    mse_frozen = rq_source.mse(X_target)
    mse_full = rq_full.mse(X_target)
    rq_warm = warm_retrain(rq_source, X_target, freeze_depth=freeze_depth, n_iter=n_iter, seed=seed)
    mse_warm = rq_warm.mse(X_target)

    excess_frozen = mse_frozen - mse_full
    excess_warm = mse_warm - mse_full

    return {
        "epsilon_s": epsilon_s,
        "excess_frozen": float(excess_frozen),
        "excess_warm": float(excess_warm),
        "lower_bound_tight": float(excess_frozen / max(epsilon_s, 1e-12)),
        "upper_bound_tight": float(excess_warm / max(epsilon_s, 1e-12)),
    }


def streaming_transport(
    dim: int = 64,
    m: int = 4,
    K: int = 64,
    n_points: int = 10000,
    n_periods: int = 10,
    drift_per_period: float = 0.5,
    freeze_depth: int = 2,
    n_iter: int = 20,
    seed: int = 42,
) -> dict:
    """Track cumulative transport mass over streaming periods."""
    rng = np.random.RandomState(seed)
    X_source = generate_data(n_points, dim, n_clusters=5, seed=seed)

    codes_per_stage = [K] * m
    rq_source = RQCodebook(m, codes_per_stage, dim)
    rq_source.fit(X_source, n_iter=n_iter, seed=seed)

    cumulative_epsilon = []
    per_period_epsilon = []
    cumulative_excess = []

    drift_vec = rng.randn(dim).astype(np.float32)
    drift_vec /= np.linalg.norm(drift_vec)

    for t in range(1, n_periods + 1):
        X_t = X_source + drift_vec * drift_per_period * t

        rq_full_t = RQCodebook(m, codes_per_stage, dim)
        rq_full_t.fit(X_t, n_iter=n_iter, seed=seed + t * 100)

        frozen_codes = rq_source.encode(X_t)
        full_codes = rq_full_t.encode(X_t)

        cross = np.zeros(len(X_t), dtype=bool)
        for i in range(freeze_depth):
            cross |= frozen_codes[i] != full_codes[i]

        eps_t = float(cross.mean())
        per_period_epsilon.append(eps_t)
        cumulative_epsilon.append(sum(per_period_epsilon))

        mse_frozen = rq_source.mse(X_t)
        mse_full = rq_full_t.mse(X_t)
        cumulative_excess.append(float(mse_frozen - mse_full))

    return {
        "per_period_epsilon": per_period_epsilon,
        "cumulative_epsilon": cumulative_epsilon,
        "cumulative_excess": cumulative_excess,
    }


def phase_diagram(
    X_source: np.ndarray,
    X_target: np.ndarray,
    m: int = 4,
    K: int = 64,
    n_lambdas: int = 20,
    n_iter: int = 20,
    seed: int = 42,
) -> dict:
    """Compute optimal freeze depth as a function of stability price lambda.

    For each lambda, find the freeze depth s that maximizes
    sum_{i in S} g_i - lambda * sum_{i in S} c_i.
    """
    gc = stagewise_gain_cost(X_source, X_target, m=m, K=K, n_iter=n_iter, seed=seed)
    gains = gc["gains"]
    churns = gc["churns"]

    lambdas = np.logspace(-3, 2, n_lambdas).tolist()
    optimal_depths = []

    for lam in lambdas:
        best_val = 0.0
        best_s = m
        for s in range(m + 1):
            val = sum(gains[s:]) - lam * sum(churns[s:])
            if val > best_val:
                best_val = val
                best_s = s
        optimal_depths.append(best_s)

    return {
        "lambdas": lambdas,
        "optimal_freeze_depths": optimal_depths,
        "gain_cost_ratios": gc["ratios"],
    }


def run_all(
    dim: int = 64,
    m: int = 4,
    K: int = 64,
    n_points: int = 10000,
    drift_magnitude: float = 2.0,
    drift_type: str = "mean_shift",
    seeds: int = 10,
    n_iter: int = 20,
    output: str | None = None,
) -> dict:
    """Run all theory validation experiments."""
    results: dict = {
        "config": {
            "dim": dim,
            "m": m,
            "K": K,
            "n_points": n_points,
            "drift_magnitude": drift_magnitude,
            "drift_type": drift_type,
            "seeds": seeds,
        },
        "gain_cost": [],
        "drift_spectrum": [],
        "transport": [],
        "streaming": [],
        "phase_diagram": [],
    }

    for seed in range(seeds):
        print(f"Seed {seed + 1}/{seeds}...", file=sys.stderr)
        X_source = generate_data(n_points, dim, n_clusters=5, seed=seed)
        X_target = apply_drift(X_source, drift_type, drift_magnitude, seed=seed)

        gc = stagewise_gain_cost(X_source, X_target, m=m, K=K, n_iter=n_iter, seed=seed)
        results["gain_cost"].append(gc)

        ds = drift_spectrum(X_source, X_target, m=m, K=K, n_iter=n_iter, seed=seed)
        results["drift_spectrum"].append(ds)

        tr = cross_subtree_transport(X_source, X_target, m=m, K=K, n_iter=n_iter, seed=seed)
        results["transport"].append(tr)

        st = streaming_transport(dim=dim, m=m, K=K, n_points=n_points, seed=seed)
        results["streaming"].append(st)

        pd = phase_diagram(X_source, X_target, m=m, K=K, n_iter=n_iter, seed=seed)
        results["phase_diagram"].append(pd)

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {output}", file=sys.stderr)

    return results


def print_summary(results: dict) -> None:
    """Print a human-readable summary."""
    seeds = len(results["gain_cost"])

    print("\n=== Stagewise Gain-Cost Ratios (Theorem 1) ===")
    all_ratios = [r["ratios"] for r in results["gain_cost"]]
    mean_ratios = np.mean(all_ratios, axis=0)
    monotone_count = sum(r["monotone"] for r in results["gain_cost"])
    print(f"  Mean g_i/c_i by stage: {[f'{r:.4f}' for r in mean_ratios]}")
    print(f"  Monotone in {monotone_count}/{seeds} seeds")

    print("\n=== Drift Spectrum (Theorem 2) ===")
    prefix_fracs = [r["prefix_fraction"] for r in results["drift_spectrum"]]
    suffix_fracs = [r["suffix_fraction"] for r in results["drift_spectrum"]]
    print(f"  Prefix drift energy fraction: {np.mean(prefix_fracs):.3f} +/- {np.std(prefix_fracs):.3f}")
    print(f"  Suffix drift energy fraction: {np.mean(suffix_fracs):.3f} +/- {np.std(suffix_fracs):.3f}")

    print("\n=== Cross-Subtree Transport (Theorem 3) ===")
    epsilons = [r["epsilon_s"] for r in results["transport"]]
    print(f"  epsilon_s: {np.mean(epsilons):.3f} +/- {np.std(epsilons):.3f}")
    lowers = [r["lower_bound_tight"] for r in results["transport"]]
    uppers = [r["upper_bound_tight"] for r in results["transport"]]
    print(f"  excess_frozen / epsilon_s: {np.mean(lowers):.4f} (empirical mu_s^2)")
    print(f"  excess_warm / epsilon_s: {np.mean(uppers):.4f} (empirical M_s^2)")

    print("\n=== Streaming Transport (Restart Criterion) ===")
    final_cum = [r["cumulative_epsilon"][-1] for r in results["streaming"]]
    print(f"  Cumulative epsilon after 10 periods: {np.mean(final_cum):.3f} +/- {np.std(final_cum):.3f}")

    print("\n=== Phase Diagram (Theorem 1 Corollary) ===")
    pd = results["phase_diagram"][0]
    transitions = []
    for i in range(1, len(pd["optimal_freeze_depths"])):
        if pd["optimal_freeze_depths"][i] != pd["optimal_freeze_depths"][i - 1]:
            transitions.append(
                (pd["lambdas"][i], pd["optimal_freeze_depths"][i - 1], pd["optimal_freeze_depths"][i])
            )
    if transitions:
        print("  Phase transitions:")
        for lam, s_from, s_to in transitions:
            print(f"    lambda={lam:.4f}: freeze depth {s_from} -> {s_to}")
    else:
        print("  No phase transitions (constant freeze depth)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Theory validation experiments")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--n-points", type=int, default=10000)
    parser.add_argument("--drift", type=float, default=2.0)
    parser.add_argument("--drift-type", default="mean_shift")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results = run_all(
        dim=args.dim,
        m=args.m,
        K=args.K,
        n_points=args.n_points,
        drift_magnitude=args.drift,
        drift_type=args.drift_type,
        seeds=args.seeds,
        output=args.output,
    )
    print_summary(results)


if __name__ == "__main__":
    main()

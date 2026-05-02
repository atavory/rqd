#!/usr/bin/env python3
"""Generate figures for the theory validation section.

Reads JSON outputs from demo_theory_validation.py and produces:
1. phase_diagram.pdf — freeze depth vs lambda
2. drift_spectrum.pdf — per-stage drift energy bar chart
3. transport_vs_dim.pdf — epsilon_s vs dimension (uniform vs funnel)

Usage:
    python3 plot_theory.py --results-dir results/theory/ --output-dir ../papers/main/figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_phase_diagram(results: dict, output: Path) -> None:
    """Plot freeze depth vs stability price lambda."""
    pd = results["phase_diagram"][0]
    lambdas = pd["lambdas"]
    depths = pd["optimal_freeze_depths"]

    fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
    ax.step(lambdas, depths, where="post", linewidth=1.5, color="black")
    ax.set_xscale("log")
    ax.set_xlabel(r"Stability price $\lambda$")
    ax.set_ylabel(r"Optimal freeze depth $s^\star$")
    ax.set_yticks(range(max(depths) + 1))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output}")


def plot_drift_spectrum(results: dict, output: Path) -> None:
    """Plot per-stage drift energy as a bar chart."""
    all_spectra = [r["per_stage"] for r in results["drift_spectrum"]]
    mean = np.mean(all_spectra, axis=0)
    std = np.std(all_spectra, axis=0)
    m = len(mean)

    fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
    stages = np.arange(1, m + 1)
    colors = ["#999999"] * (m // 2) + ["#333333"] * (m - m // 2)
    ax.bar(stages, mean, yerr=std, capsize=3, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(m // 2 + 0.5, color="red", linestyle="--", linewidth=1, label="Freeze boundary")
    ax.set_xlabel("RQ stage $i$")
    ax.set_ylabel(r"$\mathbb{E}[\|\Delta_i^{(t)} - \Delta_i^{(0)}\|^2]$")
    ax.set_xticks(stages)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output}")


def plot_transport_vs_dim(
    uniform_results: dict[int, dict],
    funnel_results: dict[int, dict],
    output: Path,
) -> None:
    """Plot epsilon_s vs dimension for uniform and funnel."""
    dims = sorted(uniform_results.keys())

    u_eps = []
    u_err = []
    f_eps = []
    f_err = []
    for d in dims:
        u_vals = [r["epsilon_s"] for r in uniform_results[d]["transport"]]
        u_eps.append(np.mean(u_vals))
        u_err.append(np.std(u_vals))
        if d in funnel_results:
            f_vals = [r["epsilon_s"] for r in funnel_results[d]["transport"]]
            f_eps.append(np.mean(f_vals))
            f_err.append(np.std(f_vals))

    fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
    ax.errorbar(dims, u_eps, yerr=u_err, marker="o", label="Uniform", capsize=3, linewidth=1.5, color="black")
    if f_eps:
        ax.errorbar(dims[:len(f_eps)], f_eps, yerr=f_err, marker="s", label="Funnel", capsize=3, linewidth=1.5, color="#666666", linestyle="--")
    ax.set_xlabel("Dimension $d$")
    ax.set_ylabel(r"Cross-subtree transport $\varepsilon_s$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/theory")
    parser.add_argument("--output-dir", type=str, default="../papers/main/figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    uniform = {}
    funnel = {}
    for p in sorted(results_dir.glob("*.json")):
        data = load_json(p)
        d = data["config"]["dim"]
        if "funnel" in p.stem:
            funnel[d] = data
        elif "uniform" in p.stem or "depth" not in p.stem:
            uniform[d] = data

    if uniform:
        first = list(uniform.values())[0]
        if first.get("phase_diagram"):
            plot_phase_diagram(first, output_dir / "phase_diagram.pdf")
        if first.get("drift_spectrum"):
            plot_drift_spectrum(first, output_dir / "drift_spectrum.pdf")

    if uniform and funnel:
        plot_transport_vs_dim(uniform, funnel, output_dir / "transport_vs_dim.pdf")
    elif uniform:
        plot_transport_vs_dim(uniform, {}, output_dir / "transport_vs_dim.pdf")

    print("Done.")


if __name__ == "__main__":
    main()

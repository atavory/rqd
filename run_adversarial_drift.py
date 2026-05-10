#!/usr/bin/env python3
"""Adversarial drift: worst-case scenarios for stratified plasticity.

Tests robustness under drift patterns designed to break frozen prefixes:
1. Boundary-targeted drift: shift items toward nearest prefix boundary
2. Rotation drift: rotate embedding space (misaligns all Voronoi cells)
3. Expansion drift: scale data so items leave their prefix cells
4. Cluster death: kill some clusters, birth new ones

For each, measures recovery and prefix stability over 10 periods.

Usage:
    python3 run_adversarial_drift.py --seeds 5
"""

from __future__ import annotations
import argparse, json, logging, math, time
import numpy as np
from scipy.stats import special_ortho_group

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

def _kmeans(X, k, n_iter=20, rng=None, init=None):
    if rng is None: rng = np.random.RandomState(42)
    n = len(X)
    if init is not None:
        centroids = init.copy()
    else:
        centroids = np.zeros((k, X.shape[1]), dtype=np.float32)
        centroids[0] = X[rng.randint(n)]
        for i in range(1, k):
            dists = np.min(np.sum((X[:, None, :] - centroids[None, :i, :]) ** 2, axis=2), axis=1)
            total = dists.sum()
            centroids[i] = X[rng.choice(n, p=dists/max(total,1e-12))] if total > 1e-12 else X[rng.randint(n)]
    for _ in range(n_iter):
        assignments = np.argmin(np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2), axis=1)
        for j in range(k):
            mask = assignments == j
            if mask.sum() > 0: centroids[j] = X[mask].mean(axis=0)
    return centroids

def _assign(X, c):
    return np.argmin(np.sum((X[:, None, :] - c[None, :, :]) ** 2, axis=2), axis=1).astype(np.int64)

class RQ:
    def __init__(self, m, codes, dim):
        self.m, self.dim = m, dim
        self.K = [codes]*m if isinstance(codes, int) else list(codes)
        self.cb = []
    def fit(self, X, n_iter=20, seed=42):
        rng = np.random.RandomState(seed)
        r = X.copy(); self.cb = []
        for i in range(self.m):
            c = _kmeans(r, self.K[i], n_iter=n_iter, rng=rng)
            self.cb.append(c); a = _assign(r, c); r = r - c[a]
        return self
    def mse(self, X):
        r = X.copy()
        for c in self.cb: a = _assign(r, c); r = r - c[a]
        return float(np.mean(np.sum(r**2, axis=1)))
    def encode(self, X, ns=None):
        if ns is None: ns = len(self.cb)
        r = X.copy(); codes = []
        for i in range(ns):
            a = _assign(r, self.cb[i]); codes.append(a); r = r - self.cb[i][a]
        return codes

def warm_retrain(rq, X, fd, n_iter=20, seed=42):
    rq2 = RQ(rq.m, rq.K, rq.dim); rq2.cb = [c.copy() for c in rq.cb]
    r = X.copy()
    for i in range(fd): a = _assign(r, rq2.cb[i]); r = r - rq2.cb[i][a]
    rng = np.random.RandomState(seed)
    for i in range(fd, rq.m):
        c = _kmeans(r, rq.K[i], n_iter=n_iter, rng=rng, init=rq2.cb[i])
        rq2.cb[i] = c; a = _assign(r, c); r = r - c[a]
    return rq2

def generate_data(n, dim, nc=20, seed=42):
    rng = np.random.RandomState(seed)
    centers = rng.randn(nc, dim).astype(np.float32) * 3.0
    labels = rng.randint(0, nc, size=n)
    return centers[labels] + rng.randn(n, dim).astype(np.float32) * 0.5, centers

def boundary_targeted_drift(X, rq, period, strength=0.3):
    """Push each item toward its nearest non-assigned centroid."""
    stage1_cb = rq.cb[0]
    assigned = _assign(X, stage1_cb)
    dists = np.sum((X[:, None, :] - stage1_cb[None, :, :]) ** 2, axis=2)
    dists[np.arange(len(X)), assigned] = np.inf
    nearest_other = np.argmin(dists, axis=1)
    direction = stage1_cb[nearest_other] - X
    norms = np.linalg.norm(direction, axis=1, keepdims=True) + 1e-8
    direction = direction / norms
    return X + direction * strength * period

def rotation_drift(X, dim, period, seed, magnitude=0.1):
    """Cumulative rotation."""
    R = np.eye(dim, dtype=np.float32)
    for t in range(period):
        Rf = special_ortho_group.rvs(dim, random_state=np.random.RandomState(seed+t*777)).astype(np.float32)
        Ri = (1-magnitude)*np.eye(dim, dtype=np.float32) + magnitude*Rf
        U,_,Vt = np.linalg.svd(Ri); R = R @ (U@Vt).astype(np.float32)
    return X @ R.T

def expansion_drift(X, period, rate=0.1):
    """Isotropic expansion."""
    return X * (1.0 + rate * period)

def cluster_death_birth(X, centers, period, dim, seed, kill_frac=0.15):
    """Kill some clusters, birth new ones."""
    rng = np.random.RandomState(seed + period * 3333)
    nc = len(centers)
    n_kill = max(1, int(nc * kill_frac))
    kill_idx = rng.choice(nc, n_kill, replace=False)
    assigned = _assign(X, centers)
    X_new = X.copy()
    for ki in kill_idx:
        mask = assigned == ki
        if mask.sum() == 0: continue
        new_center = rng.randn(dim).astype(np.float32) * 5.0
        X_new[mask] = new_center + rng.randn(int(mask.sum()), dim).astype(np.float32) * 0.5
    return X_new

DRIFT_TYPES = {
    "boundary_targeted": lambda X, rq, centers, period, dim, seed: boundary_targeted_drift(X, rq, period),
    "rotation": lambda X, rq, centers, period, dim, seed: rotation_drift(X, dim, period, seed),
    "expansion": lambda X, rq, centers, period, dim, seed: expansion_drift(X, period),
    "cluster_death": lambda X, rq, centers, period, dim, seed: cluster_death_birth(X, centers, period, dim, seed),
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--n-samples", type=int, default=30000)
    p.add_argument("--n-periods", type=int, default=10)
    p.add_argument("--json-output", type=str, default="adversarial_drift.json")
    args = p.parse_args()

    t0 = time.time()
    results = []

    for drift_name, drift_fn in DRIFT_TYPES.items():
        for aname, codes, m in [("uniform_64_4", 64, 4), ("funnel_4", [16,16,256,256], 4)]:
            fd = m // 2
            for seed in range(args.seeds):
                log.info(f"{drift_name} / {aname} / seed={seed}")
                X0, centers = generate_data(args.n_samples, args.dim, seed=seed)
                rq0 = RQ(m, codes, args.dim).fit(X0, seed=seed)
                rq_cur = rq0

                for period in range(args.n_periods + 1):
                    if period == 0:
                        Xt = X0
                    else:
                        Xt = drift_fn(X0, rq0, centers, period, args.dim, seed)

                    mf = rq0.mse(Xt)
                    rq_full = RQ(m, codes, args.dim).fit(Xt, seed=seed+period*100)
                    mfull = rq_full.mse(Xt)
                    if period > 0:
                        rq_cur = warm_retrain(rq_cur, Xt, fd, seed=seed+period)
                    mw = rq_cur.mse(Xt)
                    denom = mf - mfull
                    rho = 1.0 - (mw - mfull)/denom if abs(denom)>1e-12 else 1.0

                    # Prefix stability
                    c0 = np.column_stack(rq0.encode(Xt, fd))
                    cw = np.column_stack(rq_cur.encode(Xt, fd))
                    pfx = float(np.all(c0 == cw, axis=1).mean())

                    results.append({
                        "drift": drift_name, "arch": aname, "seed": seed,
                        "period": period, "mse_frozen": mf, "mse_warm": mw,
                        "mse_full": mfull, "recovery": rho, "prefix_consistency": pfx,
                    })

                with open(args.json_output, "w") as f:
                    json.dump(results, f)
                log.info(f"  T={args.n_periods}: rho={results[-1]['recovery']:.3f} pfx={results[-1]['prefix_consistency']:.3f}")

    log.info(f"Done in {time.time()-t0:.0f}s. {len(results)} rows.")

if __name__ == "__main__":
    main()

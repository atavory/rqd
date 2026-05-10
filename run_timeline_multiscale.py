#!/usr/bin/env python3
"""Multi-scale realistic timeline: sweep d, n, architectures.

20 periods × {d=64,128,256,512} × {n=10K,50K} × 4 archs × 5 seeds.
Same drift model as run_realistic_timeline.py.

Usage:
    python3 run_timeline_multiscale.py --seeds 5
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

def apply_drift(X_base, centers, period, dim, seed,
                grad=0.3, sudden=(7,14), rot_mag=0.15, cyc_p=6, cyc_a=0.2, new_frac=0.1):
    rng = np.random.RandomState(seed + period*1000)
    X = X_base.copy()
    drift = np.zeros(dim, dtype=np.float32)
    for t in range(period):
        d = np.random.RandomState(seed+t*1000+1).randn(dim).astype(np.float32)
        drift += d / np.linalg.norm(d) * grad
    X = X + drift
    R = np.eye(dim, dtype=np.float32)
    for sp in sudden:
        if period >= sp:
            Rf = special_ortho_group.rvs(dim, random_state=np.random.RandomState(seed+sp*7777)).astype(np.float32)
            Ri = (1-rot_mag)*np.eye(dim, dtype=np.float32) + rot_mag*Rf
            U,_,Vt = np.linalg.svd(Ri); R = R @ (U@Vt).astype(np.float32)
    X = X @ R.T
    X = X * (1.0 + cyc_a * math.sin(2*math.pi*period/cyc_p))
    n_new = int(len(X)*new_frac)
    if n_new > 0:
        nc = centers + drift
        nc = nc @ R.T * (1.0 + cyc_a * math.sin(2*math.pi*period/cyc_p))
        nl = rng.randint(0, len(nc), size=n_new)
        X[rng.choice(len(X), n_new, replace=False)] = nc[nl] + rng.randn(n_new, dim).astype(np.float32)*0.7
    return X

ARCHS = {
    "uniform_64_4": (64, 4),
    "funnel_4": ([16,16,256,256], 4),
    "uniform_64_6": (64, 6),
    "funnel_6": ([16,16,16,64,256,256], 6),
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n-periods", type=int, default=20)
    p.add_argument("--json-output", type=str, default="timeline_multiscale.json")
    args = p.parse_args()
    t0 = time.time()
    results = []
    for dim in [64, 128, 256, 512]:
        for n in [10000, 50000]:
            for aname, (codes, m) in ARCHS.items():
                fd = m // 2
                for seed in range(args.seeds):
                    log.info(f"d={dim} n={n} {aname} seed={seed}")
                    X0, centers = generate_data(n, dim, seed=seed)
                    rq0 = RQ(m, codes, dim).fit(X0, seed=seed)
                    rq_cur = rq0
                    for period in range(args.n_periods+1):
                        Xt = X0 if period==0 else apply_drift(X0, centers, period, dim, seed)
                        mf = rq0.mse(Xt)
                        rq_full = RQ(m, codes, dim).fit(Xt, seed=seed+period*100)
                        mfull = rq_full.mse(Xt)
                        if period > 0:
                            rq_cur = warm_retrain(rq_cur, Xt, fd, seed=seed+period)
                        mw = rq_cur.mse(Xt)
                        denom = mf - mfull
                        rho = 1.0 - (mw - mfull)/denom if abs(denom)>1e-12 else 1.0
                        results.append({"dim":dim,"n":n,"arch":aname,"seed":seed,
                                       "period":period,"mse_frozen":mf,"mse_warm":mw,
                                       "mse_full":mfull,"recovery":rho})
                    with open(args.json_output, "w") as f:
                        json.dump(results, f)
                    log.info(f"  T={args.n_periods}: rho={results[-1]['recovery']:.3f}")
    log.info(f"Done in {time.time()-t0:.0f}s. {len(results)} rows.")

if __name__ == "__main__":
    main()

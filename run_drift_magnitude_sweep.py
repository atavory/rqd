#!/usr/bin/env python3
"""Drift magnitude sweep: find the breaking point.

Sweep α from 0.1 to 5.0. At each magnitude, measure recovery for
uniform and funnel at d=64,128,256. 5 seeds.

Usage:
    python3 run_drift_magnitude_sweep.py --seeds 5
"""
from __future__ import annotations
import argparse, json, logging, time
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

def _kmeans(X, k, n_iter=20, rng=None, init=None):
    if rng is None: rng = np.random.RandomState(42)
    n = len(X)
    if init is not None: centroids = init.copy()
    else:
        centroids = np.zeros((k, X.shape[1]), dtype=np.float32); centroids[0] = X[rng.randint(n)]
        for i in range(1, k):
            dists = np.min(np.sum((X[:,None,:]-centroids[None,:i,:])**2, axis=2), axis=1)
            t = dists.sum(); centroids[i] = X[rng.choice(n, p=dists/max(t,1e-12))] if t>1e-12 else X[rng.randint(n)]
    for _ in range(n_iter):
        a = np.argmin(np.sum((X[:,None,:]-centroids[None,:,:])**2, axis=2), axis=1)
        for j in range(k):
            m = a==j
            if m.sum()>0: centroids[j] = X[m].mean(axis=0)
    return centroids

def _assign(X, c): return np.argmin(np.sum((X[:,None,:]-c[None,:,:])**2, axis=2), axis=1).astype(np.int64)

class RQ:
    def __init__(s, m, codes, dim):
        s.m, s.dim = m, dim; s.K = [codes]*m if isinstance(codes,int) else list(codes); s.cb = []
    def fit(s, X, n_iter=20, seed=42):
        rng = np.random.RandomState(seed); r = X.copy(); s.cb = []
        for i in range(s.m):
            c = _kmeans(r, s.K[i], n_iter=n_iter, rng=rng); s.cb.append(c); a = _assign(r,c); r = r - c[a]
        return s
    def mse(s, X):
        r = X.copy()
        for c in s.cb: a = _assign(r,c); r = r - c[a]
        return float(np.mean(np.sum(r**2, axis=1)))

def warm_retrain(rq, X, fd, n_iter=20, seed=42):
    rq2 = RQ(rq.m, rq.K, rq.dim); rq2.cb = [c.copy() for c in rq.cb]
    r = X.copy()
    for i in range(fd): a = _assign(r, rq2.cb[i]); r = r - rq2.cb[i][a]
    rng = np.random.RandomState(seed)
    for i in range(fd, rq.m):
        c = _kmeans(r, rq.K[i], n_iter=n_iter, rng=rng, init=rq2.cb[i]); rq2.cb[i] = c; a = _assign(r,c); r = r - c[a]
    return rq2

def generate(n, dim, nc=20, seed=42):
    rng = np.random.RandomState(seed)
    centers = rng.randn(nc, dim).astype(np.float32)*3.0
    labels = rng.randint(0, nc, size=n)
    return centers[labels] + rng.randn(n, dim).astype(np.float32)*0.5

def apply_drift(X, alpha, dim, seed):
    rng = np.random.RandomState(seed+9999)
    d = rng.randn(dim).astype(np.float32); d /= np.linalg.norm(d)
    return X + d * alpha

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--json-output", type=str, default="drift_magnitude_sweep.json")
    args = p.parse_args()
    t0 = time.time(); results = []
    alphas = [0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0, 10.0]
    for dim in [64, 128, 256]:
        for aname, codes, m in [("uniform_64_4",64,4), ("funnel_4",[16,16,256,256],4)]:
            fd = m//2
            for alpha in alphas:
                for seed in range(args.seeds):
                    X0 = generate(20000, dim, seed=seed)
                    X1 = apply_drift(X0, alpha, dim, seed)
                    rq0 = RQ(m, codes, dim).fit(X0, seed=seed)
                    rq_full = RQ(m, codes, dim).fit(X1, seed=seed+500)
                    rq_warm = warm_retrain(rq0, X1, fd, seed=seed)
                    mf, mw, mfull = rq0.mse(X1), rq_warm.mse(X1), rq_full.mse(X1)
                    denom = mf - mfull
                    rho = 1.0 - (mw-mfull)/denom if abs(denom)>1e-12 else 1.0
                    results.append({"dim":dim,"arch":aname,"alpha":alpha,"seed":seed,
                                   "mse_frozen":mf,"mse_warm":mw,"mse_full":mfull,"recovery":rho})
                with open(args.json_output, "w") as f: json.dump(results, f)
                rhos = [r["recovery"] for r in results if r["dim"]==dim and r["arch"]==aname and r["alpha"]==alpha]
                log.info(f"d={dim} {aname} alpha={alpha}: rho={np.mean(rhos):.3f}+/-{np.std(rhos):.3f}")
    log.info(f"Done in {time.time()-t0:.0f}s. {len(results)} rows.")

if __name__ == "__main__": main()

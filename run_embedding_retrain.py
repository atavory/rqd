#!/usr/bin/env python3
"""Embedding model retraining experiment.

Simulates the scariest real-world scenario: the embedding model itself
is retrained, rotating the coordinate system. Tests whether stratified
plasticity survives Procrustes-aligned and unaligned embedding shifts.

Protocol:
- Train a simple MLP autoencoder on source data
- Retrain it on target data (different init → different coordinate system)
- Encode both through the source/target autoencoders
- Apply RQ to the source embeddings, then test on target embeddings
- Compare: raw rotation vs Procrustes alignment before RQ adaptation

20 periods, 5 seeds, d_embed=128.

Usage:
    python3 run_embedding_retrain.py --seeds 5
"""

from __future__ import annotations
import argparse, json, logging, math, time
import numpy as np

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

def warm_retrain(rq, X, fd, n_iter=20, seed=42):
    rq2 = RQ(rq.m, rq.K, rq.dim); rq2.cb = [c.copy() for c in rq.cb]
    r = X.copy()
    for i in range(fd): a = _assign(r, rq2.cb[i]); r = r - rq2.cb[i][a]
    rng = np.random.RandomState(seed)
    for i in range(fd, rq.m):
        c = _kmeans(r, rq.K[i], n_iter=n_iter, rng=rng, init=rq2.cb[i])
        rq2.cb[i] = c; a = _assign(r, c); r = r - c[a]
    return rq2

def procrustes_align(X_source, X_target):
    """Align X_target to X_source via Procrustes (orthogonal rotation)."""
    mu_s, mu_t = X_source.mean(0), X_target.mean(0)
    Xs, Xt = X_source - mu_s, X_target - mu_t
    U, _, Vt = np.linalg.svd(Xt.T @ Xs)
    R = (U @ Vt).astype(np.float32)
    return (Xt @ R + mu_s).astype(np.float32), R

def simple_encoder(X, d_embed, seed):
    """Random linear projection as embedding model surrogate."""
    rng = np.random.RandomState(seed)
    d_in = X.shape[1]
    W = rng.randn(d_in, d_embed).astype(np.float32) * 0.1
    b = rng.randn(d_embed).astype(np.float32) * 0.01
    Z = X @ W + b
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    return Z, W, b

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n-periods", type=int, default=20)
    p.add_argument("--n-samples", type=int, default=30000)
    p.add_argument("--d-raw", type=int, default=256)
    p.add_argument("--d-embed", type=int, default=128)
    p.add_argument("--dims", type=str, default=None,
                   help="Comma-separated d-embed values to sweep, e.g. 128,256,512")
    p.add_argument("--json-output", type=str, default="embedding_retrain.json")
    args = p.parse_args()

    embed_dims = [int(d) for d in args.dims.split(",")] if args.dims else [args.d_embed]

    t0 = time.time()
    results = []

    for d_embed in embed_dims:
      args.d_embed = d_embed
      log.info(f"=== d_embed={d_embed} ===")
      for seed in range(args.seeds):
        log.info(f"seed={seed}")
        rng = np.random.RandomState(seed)

        # Generate raw data with gradual drift
        X_raw_base = rng.randn(args.n_samples, args.d_raw).astype(np.float32)

        # Source embedding model
        Z_source, W0, b0 = simple_encoder(X_raw_base, args.d_embed, seed=seed)

        # Train RQ on source embeddings
        for aname, codes, m in [("uniform_64_4", 64, 4), ("funnel_4", [16,16,256,256], 4)]:
            fd = m // 2
            rq0 = RQ(m, codes, args.d_embed).fit(Z_source, seed=seed)
            rq_warm_raw = rq0  # no alignment
            rq_warm_proc = rq0  # with Procrustes

            for period in range(args.n_periods + 1):
                # Drift the raw data
                drift = np.zeros(args.d_raw, dtype=np.float32)
                for t in range(period):
                    d = np.random.RandomState(seed+t*100+1).randn(args.d_raw).astype(np.float32)
                    drift += d / np.linalg.norm(d) * 0.5
                X_raw_t = X_raw_base + drift

                # Re-embed with a DIFFERENT encoder (simulates model retraining)
                encoder_seed = seed + period * 7 + 999
                Z_target, Wt, bt = simple_encoder(X_raw_t, args.d_embed, seed=encoder_seed)

                # Procrustes-align target embeddings to source coordinate system
                Z_aligned, R = procrustes_align(Z_source, Z_target)

                mse_frozen = rq0.mse(Z_target)
                mse_frozen_aligned = rq0.mse(Z_aligned)

                rq_full = RQ(m, codes, args.d_embed).fit(Z_target, seed=seed+period*100)
                mse_full = rq_full.mse(Z_target)

                # Warm retrain on raw (unaligned) target
                if period > 0:
                    rq_warm_raw = warm_retrain(rq_warm_raw, Z_target, fd, seed=seed+period)
                mse_warm_raw = rq_warm_raw.mse(Z_target)

                # Warm retrain on Procrustes-aligned target
                if period > 0:
                    rq_warm_proc = warm_retrain(rq_warm_proc, Z_aligned, fd, seed=seed+period)
                mse_warm_proc = rq_warm_proc.mse(Z_aligned)

                def rho(mf, mw, mfull):
                    d = mf - mfull
                    return 1.0 - (mw - mfull)/d if abs(d)>1e-12 else 1.0

                results.append({
                    "d_embed": d_embed,
                    "seed": seed, "arch": aname, "period": period,
                    "mse_frozen_raw": mse_frozen,
                    "mse_frozen_aligned": mse_frozen_aligned,
                    "mse_warm_raw": mse_warm_raw,
                    "mse_warm_aligned": mse_warm_proc,
                    "mse_full": mse_full,
                    "rho_raw": rho(mse_frozen, mse_warm_raw, mse_full),
                    "rho_aligned": rho(mse_frozen_aligned, mse_warm_proc, mse_full),
                })

            with open(args.json_output, "w") as f:
                json.dump(results, f)
            last = results[-1]
            log.info(f"  {aname} T={args.n_periods}: rho_raw={last['rho_raw']:.3f} rho_aligned={last['rho_aligned']:.3f}")

    log.info(f"Done in {time.time()-t0:.0f}s. {len(results)} rows.")

if __name__ == "__main__":
    main()

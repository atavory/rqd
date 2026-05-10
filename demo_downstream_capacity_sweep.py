#!/usr/bin/env python3
"""Downstream ANN recall vs codebook capacity.

Sweeps K in {8, 16, 32, 64} on MovieLens and Amazon, reporting
ANN recall@10 for frozen / stratified / full at each capacity.
Logs to file for real-time monitoring.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

sys.path.insert(0, os.path.dirname(__file__))
from rq import RQCodebook, warm_retrain

log = logging.getLogger(__name__)


def load_movielens():
    data_path = Path(__file__).resolve().parent.parent / "data" / "movielens" / "ml-1m" / "ratings.dat"
    log.info("Loading MovieLens-1M...")
    ratings = []
    with open(data_path, encoding="latin-1") as f:
        for line in f:
            p = line.strip().split("::")
            ratings.append((int(p[0]), int(p[1]), float(p[2]), int(p[3])))
    all_items = sorted(set(r[1] for r in ratings))
    all_users = sorted(set(r[0] for r in ratings))
    user_map = {u: i for i, u in enumerate(all_users)}
    item_map = {m: i for i, m in enumerate(all_items)}
    n_u, n_i = len(all_users), len(all_items)
    ts = np.array([r[3] for r in ratings])
    med = np.median(ts)
    old = [r for r in ratings if r[3] < med]
    new = [r for r in ratings if r[3] >= med]

    def sp(rs):
        return csr_matrix(([r[2] for r in rs],
                           ([user_map[r[0]] for r in rs],
                            [item_map[r[1]] for r in rs])),
                          shape=(n_u, n_i), dtype=np.float32)

    mat_old, mat_new = sp(old), sp(new)
    d = 64
    k = min(d, min(mat_old.shape) - 1)
    _, So, Vo = svds(mat_old, k=k)
    _, Sn, Vn = svds(mat_new, k=k)
    X_old = (Vo.T * So).astype(np.float32)
    X_new_raw = (Vn.T * Sn).astype(np.float32)
    R, _ = orthogonal_procrustes(X_new_raw, X_old)
    X_new = (X_new_raw @ R).astype(np.float32)
    log.info(f"  {n_i} items, Procrustes residual: {np.mean((X_new_raw @ R - X_old)**2):.2f}")
    return X_old, X_new, n_i, "MovieLens"


def load_amazon(max_items=50000):
    data_path = Path(__file__).resolve().parent.parent / "data" / "amazon" / "electronics_5.json.gz"
    log.info("Loading Amazon Electronics...")
    reviews = []
    with gzip.open(data_path, "rt") as f:
        for line in f:
            d = json.loads(line)
            reviews.append((d["reviewerID"], d["asin"],
                           float(d["overall"]), int(d["unixReviewTime"])))
    item_counts = Counter(r[1] for r in reviews)
    valid = {k for k, v in item_counts.items() if v >= 5}
    reviews = [r for r in reviews if r[1] in valid]
    all_items = list(set(r[1] for r in reviews))
    if len(all_items) > max_items:
        rng = np.random.RandomState(42)
        keep = set(rng.choice(all_items, max_items, replace=False))
        reviews = [r for r in reviews if r[1] in keep]
    all_users = sorted(set(r[0] for r in reviews))
    all_items = sorted(set(r[1] for r in reviews))
    user_map = {u: i for i, u in enumerate(all_users)}
    item_map = {m_: i for i, m_ in enumerate(all_items)}
    n_u, n_i = len(all_users), len(all_items)
    ts = np.array([r[3] for r in reviews])
    med = np.median(ts)
    old = [r for r in reviews if r[3] < med]
    new = [r for r in reviews if r[3] >= med]

    def sp(rs):
        rows = [user_map[r[0]] for r in rs if r[0] in user_map and r[1] in item_map]
        cols = [item_map[r[1]] for r in rs if r[0] in user_map and r[1] in item_map]
        vals = [r[2] for r in rs if r[0] in user_map and r[1] in item_map]
        return csr_matrix((vals, (rows, cols)), shape=(n_u, n_i), dtype=np.float32)

    mat_old, mat_new = sp(old), sp(new)
    d = 64
    k = min(d, min(mat_old.shape) - 1)
    _, So, Vo = svds(mat_old, k=k)
    _, Sn, Vn = svds(mat_new, k=k)
    X_old = (Vo.T * So).astype(np.float32)
    X_new_raw = (Vn.T * Sn).astype(np.float32)
    R, _ = orthogonal_procrustes(X_new_raw, X_old)
    X_new = (X_new_raw @ R).astype(np.float32)
    log.info(f"  {n_i} items, Procrustes residual: {np.mean((X_new_raw @ R - X_old)**2):.2f}")
    return X_old, X_new, n_i, "Amazon"


def ann_recall(raw, decoded, k=10, n_queries=2000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(raw)
    nq = min(n_queries, n // 2)
    q_idx = rng.choice(n, nq, replace=False)
    db_mask = np.ones(n, dtype=bool)
    db_mask[q_idx] = False
    db_idx = np.where(db_mask)[0]
    raw_q, raw_db = raw[q_idx], raw[db_idx]
    dec_q, dec_db = decoded[q_idx], decoded[db_idx]
    hits = 0
    batch = 200
    for st in range(0, nq, batch):
        en = min(st + batch, nq)
        td = np.sum((raw_q[st:en, None, :] - raw_db[None, :, :]) ** 2, axis=2)
        ad = np.sum((dec_q[st:en, None, :] - dec_db[None, :, :]) ** 2, axis=2)
        for i in range(en - st):
            t = set(np.argpartition(td[i], k)[:k].tolist())
            a = set(np.argpartition(ad[i], k)[:k].tolist())
            hits += len(t & a)
    return hits / (nq * k)


def sweep(X_old, X_new, n_items, dataset_name, Ks, m=4, s=2, seeds=5):
    rows = []
    for K in Ks:
        capacity = K ** m
        ratio = capacity / n_items
        log.info(f"\n--- {dataset_name} K={K}, capacity={capacity}, ratio={ratio:.1f}x ---")
        for seed in range(seeds):
            rq0 = RQCodebook(m, [K]*m, X_old.shape[1])
            rq0.fit(X_old, n_iter=20, seed=seed)
            rq_full = RQCodebook(m, [K]*m, X_new.shape[1])
            rq_full.fit(X_new, n_iter=20, seed=seed + 1000)
            rq_warm = warm_retrain(rq0, X_new, freeze_depth=s,
                                   n_iter=20, seed=seed)

            rf = ann_recall(X_new, rq0.reconstruct(X_new), seed=seed)
            rw = ann_recall(X_new, rq_warm.reconstruct(X_new), seed=seed)
            rr = ann_recall(X_new, rq_full.reconstruct(X_new), seed=seed)

            mf = rq0.mse(X_new)
            mw = rq_warm.mse(X_new)
            mr = rq_full.mse(X_new)

            pfx_o = np.column_stack(rq0.encode(X_new, n_stages=s))
            pfx_n = np.column_stack(rq_full.encode(X_new, n_stages=s))
            chg = np.any(pfx_o != pfx_n, axis=1).mean()

            row = {"dataset": dataset_name, "K": K, "m": m, "s": s,
                   "capacity": capacity, "ratio": ratio, "seed": seed,
                   "ann_frozen": rf, "ann_strat": rw, "ann_full": rr,
                   "mse_frozen": float(mf), "mse_strat": float(mw),
                   "mse_full": float(mr), "pfx_chg": float(chg)}
            rows.append(row)
            log.info(f"  seed {seed}: frz={rf:.4f} strat={rw:.4f} full={rr:.4f} "
                     f"MSE {mf:.1f}/{mw:.1f}/{mr:.1f} pfx={chg:.0%}")

        vals_f = [r["ann_frozen"] for r in rows if r["K"] == K and r["dataset"] == dataset_name]
        vals_w = [r["ann_strat"] for r in rows if r["K"] == K and r["dataset"] == dataset_name]
        vals_r = [r["ann_full"] for r in rows if r["K"] == K and r["dataset"] == dataset_name]
        gap = np.mean(vals_r) - np.mean(vals_f)
        rec = np.mean(vals_w) - np.mean(vals_f)
        pct = rec / gap * 100 if gap > 0 else 0
        log.info(f"  K={K} avg: frz={np.mean(vals_f):.4f} strat={np.mean(vals_w):.4f} "
                 f"full={np.mean(vals_r):.4f}  recovery={pct:.0f}%")
    return rows


def main():
    log_path = "results/theory/downstream_capacity_sweep.log"
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"),
                  logging.StreamHandler()])

    Ks = [8, 16, 32, 64]
    all_rows = []

    X_old, X_new, n_i, name = load_movielens()
    all_rows.extend(sweep(X_old, X_new, n_i, name, Ks))

    X_old, X_new, n_i, name = load_amazon()
    all_rows.extend(sweep(X_old, X_new, n_i, name, Ks))

    out = "results/theory/downstream_capacity_sweep.json"
    with open(out, "w") as f:
        json.dump({"rows": all_rows}, f, indent=2)
    log.info(f"\nSaved to {out}")

    log.info(f"\n{'='*80}")
    log.info("FINAL SUMMARY")
    log.info(f"{'='*80}")
    log.info(f"{'Dataset':<12} {'K':>4} {'Cap':>8} {'Ratio':>7} "
             f"{'Frozen':>8} {'Strat':>8} {'Full':>8} {'Recov':>7}")
    log.info("-" * 75)
    for ds in ["MovieLens", "Amazon"]:
        for K in Ks:
            rs = [r for r in all_rows if r["dataset"] == ds and r["K"] == K]
            af = np.mean([r["ann_frozen"] for r in rs])
            aw = np.mean([r["ann_strat"] for r in rs])
            ar = np.mean([r["ann_full"] for r in rs])
            gap = ar - af
            pct = (aw - af) / gap * 100 if gap > 0 else 0
            cap = K ** 4
            n = 3706 if ds == "MovieLens" else 50000
            log.info(f"{ds:<12} {K:>4} {cap:>8} {cap/n:>6.1f}x "
                     f"{af:>8.4f} {aw:>8.4f} {ar:>8.4f} {pct:>6.0f}%")


if __name__ == "__main__":
    main()

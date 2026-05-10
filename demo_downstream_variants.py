#!/usr/bin/env python3
"""Downstream experiment variants for CIKM paper motivation.

Tests multiple downstream setups to find configurations where
stratified plasticity clearly outperforms alternatives.

Variants:
1. prefix_aware: Ranker uses prefix code embedding + decoded vector.
   Full retrain changes prefix codes -> embedding lookup is wrong -> breaks.
2. raw_ranker: Ranker trained on raw features, served decoded vectors.
   Better reconstruction -> better recall.
3. decoded_ranker: (current) Ranker trained on decoded T0 vectors.

Logs to file for real-time monitoring.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

sys.path.insert(0, os.path.dirname(__file__))
from rq import RQCodebook, warm_retrain

log = logging.getLogger(__name__)


def load_movielens():
    data_path = Path(__file__).parent.parent / "data" / "movielens" / "ml-1m" / "ratings.dat"
    log.info("Loading MovieLens-1M...")
    ratings = []
    with open(data_path, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            ratings.append((int(parts[0]), int(parts[1]),
                           float(parts[2]), int(parts[3])))

    all_users = sorted(set(r[0] for r in ratings))
    all_items = sorted(set(r[1] for r in ratings))
    user_map = {u: i for i, u in enumerate(all_users)}
    item_map = {m: i for i, m in enumerate(all_items)}
    n_users, n_items = len(all_users), len(all_items)

    timestamps = np.array([r[3] for r in ratings])
    median_ts = np.median(timestamps)
    old = [r for r in ratings if r[3] < median_ts]
    new = [r for r in ratings if r[3] >= median_ts]
    log.info(f"  {n_users} users, {n_items} items, Old: {len(old)}, New: {len(new)}")

    def build_sparse(rs):
        rows = [user_map[r[0]] for r in rs]
        cols = [item_map[r[1]] for r in rs]
        vals = [r[2] for r in rs]
        return csr_matrix((vals, (rows, cols)),
                         shape=(n_users, n_items), dtype=np.float32)

    def build_pairs(rs):
        return [(user_map[r[0]], item_map[r[1]]) for r in rs]

    mat_old = build_sparse(old)
    mat_new = build_sparse(new)

    d = 64
    k = min(d, min(mat_old.shape) - 1)
    _, S_old, Vt_old = svds(mat_old, k=k)
    _, S_new, Vt_new = svds(mat_new, k=k)

    X_old = (Vt_old.T * S_old).astype(np.float32)
    X_new_raw = (Vt_new.T * S_new).astype(np.float32)

    from scipy.linalg import orthogonal_procrustes
    R, _ = orthogonal_procrustes(X_new_raw, X_old)
    X_new = (X_new_raw @ R).astype(np.float32)
    log.info(f"  Procrustes residual: {np.mean((X_new_raw @ R - X_old)**2):.4f}")

    return X_old, X_new, build_pairs(old), build_pairs(new), n_users, n_items


def get_prefix_int(rq, X, s=2, K=64):
    codes = rq.encode(X, n_stages=s)
    out = codes[0].copy()
    for i in range(1, s):
        out = out * K + codes[i]
    return out.astype(np.int64)


class PrefixAwareRanker(nn.Module):
    def __init__(self, n_users, n_prefix_codes, item_dim, emb_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.prefix_emb = nn.Embedding(n_prefix_codes, emb_dim)
        self.item_proj = nn.Linear(item_dim, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.prefix_emb.weight, std=0.01)

    def forward(self, user_ids, prefix_codes, item_vecs):
        u = self.user_emb(user_ids)
        p = self.prefix_emb(prefix_codes)
        v = self.item_proj(item_vecs)
        return (u * (p + v)).sum(dim=-1)


class SimpleRanker(nn.Module):
    def __init__(self, n_users, item_dim, emb_dim=32):
        super().__init__()
        self.item_proj = nn.Linear(item_dim, emb_dim)
        self.user_emb = nn.Embedding(n_users, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)

    def forward(self, user_ids, item_vecs):
        u = self.user_emb(user_ids)
        v = self.item_proj(item_vecs)
        return (u * v).sum(dim=-1)


def _train_bpr(model, pairs, n_items, get_features,
               epochs=15, lr=0.005, batch_size=4096, n_neg=4, seed=42):
    rng = np.random.RandomState(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pairs_arr = np.array(pairs)
    for epoch in range(epochs):
        rng.shuffle(pairs_arr)
        total_loss, n_batches = 0, 0
        for start in range(0, len(pairs_arr), batch_size):
            batch = pairs_arr[start:start + batch_size]
            users = torch.from_numpy(batch[:, 0].astype(np.int64))
            pos_idx = batch[:, 1].astype(np.int64)
            neg_idx = rng.randint(0, n_items, size=(len(batch), n_neg))

            pos_scores = model(users, *get_features(pos_idx))
            neg_loss = 0
            for j in range(n_neg):
                neg_scores = model(users, *get_features(neg_idx[:, j]))
                neg_loss += -F.logsigmoid(pos_scores - neg_scores).mean()
            loss = neg_loss / n_neg

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1


@torch.no_grad()
def _eval_recall(model, test_pairs, train_pairs, get_all_scores,
                 k=10, max_eval=5000, eval_seed=99):
    train_set, test_set = {}, {}
    for u, i in train_pairs:
        train_set.setdefault(u, set()).add(i)
    for u, i in test_pairs:
        test_set.setdefault(u, set()).add(i)
    eval_users = [u for u in test_set if u in train_set]
    if len(eval_users) > max_eval:
        rng = np.random.RandomState(eval_seed)
        eval_users = list(rng.choice(eval_users, max_eval, replace=False))
    if not eval_users:
        return 0.0

    scores = get_all_scores(eval_users)  # (n_eval, n_items)
    hits, total = 0, 0
    for i, u in enumerate(eval_users):
        for ti in train_set[u]:
            scores[i, ti] = -np.inf
        top_k = set(np.argpartition(scores[i], -k)[-k:])
        hits += len(test_set[u] & top_k)
        total += len(test_set[u])
    return hits / max(total, 1)


def run_variants(X_old, X_new, pairs_old, pairs_new, n_users, n_items,
                 m=4, K=64, s=2, n_iter=20, seeds=3):
    all_results = []

    for seed in range(seeds):
        log.info(f"\n{'='*60}")
        log.info(f"SEED {seed}")
        log.info(f"{'='*60}")

        rq_t0 = RQCodebook(m, [K]*m, X_old.shape[1])
        rq_t0.fit(X_old, n_iter=n_iter, seed=seed)

        rq_full = RQCodebook(m, [K]*m, X_new.shape[1])
        rq_full.fit(X_new, n_iter=n_iter, seed=seed + 1000)

        rq_warm = warm_retrain(rq_t0, X_new, freeze_depth=s,
                               n_iter=n_iter, seed=seed)

        recon_t0 = rq_t0.reconstruct(X_old)
        recon_frz = rq_t0.reconstruct(X_new)
        recon_full = rq_full.reconstruct(X_new)
        recon_warm = rq_warm.reconstruct(X_new)

        n_pfx = K ** s
        pfx_t0 = get_prefix_int(rq_t0, X_old, s, K)
        pfx_frz = get_prefix_int(rq_t0, X_new, s, K)
        pfx_full = get_prefix_int(rq_full, X_new, s, K)
        pfx_warm = get_prefix_int(rq_warm, X_new, s, K)

        pfx_chg = float(np.mean(pfx_frz != pfx_full))
        mse_frz = float(rq_t0.mse(X_new))
        mse_full = float(rq_full.mse(X_new))
        mse_warm = float(rq_warm.mse(X_new))
        log.info(f"  Prefix change: {pfx_chg:.1%}")
        log.info(f"  MSE: frozen={mse_frz:.1f} warm={mse_warm:.1f} full={mse_full:.1f}")

        result = {"seed": seed, "mse_frozen": mse_frz, "mse_full": mse_full,
                  "mse_warm": mse_warm, "prefix_change": pfx_chg}

        # ==== Variant A: prefix-aware ranker ====
        log.info("\n  --- prefix_aware_ranker ---")
        pfx_t0_t = torch.from_numpy(pfx_t0)
        recon_t0_t = torch.from_numpy(recon_t0).float()

        def pa_features_train(idx):
            return pfx_t0_t[idx], recon_t0_t[idx]

        pa = PrefixAwareRanker(n_users, n_pfx, X_old.shape[1])
        _train_bpr(pa, pairs_old, n_items, pa_features_train,
                   epochs=15, seed=seed)
        pa.eval()

        def _pa_scores(users, pfx_arr, recon_arr):
            pfx_t = torch.from_numpy(pfx_arr)
            rec_t = torch.from_numpy(recon_arr).float()
            all_proj = (pa.item_proj(rec_t) + pa.prefix_emb(pfx_t)).numpy()
            u_embs = pa.user_emb(
                torch.tensor(users, dtype=torch.long)).numpy()
            return u_embs @ all_proj.T

        r_pa = {}
        for tag, pf, rc in [("frozen", pfx_frz, recon_frz),
                             ("full_no_ds", pfx_full, recon_full),
                             ("stratified", pfx_warm, recon_warm)]:
            r_pa[tag] = _eval_recall(
                pa, pairs_new, pairs_old,
                lambda users, _pf=pf, _rc=rc: _pa_scores(users, _pf, _rc))

        pa_ds = PrefixAwareRanker(n_users, n_pfx, X_new.shape[1])
        pfx_full_t = torch.from_numpy(pfx_full)
        recon_full_t = torch.from_numpy(recon_full).float()
        _train_bpr(pa_ds, pairs_new, n_items,
                   lambda idx: (pfx_full_t[idx], recon_full_t[idx]),
                   epochs=15, seed=seed)
        pa_ds.eval()

        def _pa_ds_scores(users):
            all_proj = (pa_ds.item_proj(recon_full_t) +
                       pa_ds.prefix_emb(pfx_full_t)).numpy()
            u_embs = pa_ds.user_emb(
                torch.tensor(users, dtype=torch.long)).numpy()
            return u_embs @ all_proj.T
        r_pa["full_ds"] = _eval_recall(
            pa_ds, pairs_new, pairs_old, _pa_ds_scores)

        log.info(f"  Frozen:     {r_pa['frozen']:.4f}")
        log.info(f"  Full-noDS:  {r_pa['full_no_ds']:.4f}")
        log.info(f"  Full+DS:    {r_pa['full_ds']:.4f}")
        log.info(f"  Stratified: {r_pa['stratified']:.4f}")
        for k2, v in r_pa.items():
            result[f"pa_{k2}"] = float(v)

        # ==== Variant B: raw-trained ranker ====
        log.info("\n  --- raw_ranker ---")
        X_old_t = torch.from_numpy(X_old).float()
        raw_m = SimpleRanker(n_users, X_old.shape[1])
        _train_bpr(raw_m, pairs_old, n_items,
                   lambda idx: (X_old_t[idx],), epochs=15, seed=seed)
        raw_m.eval()

        def _raw_scores(users, vecs):
            all_proj = raw_m.item_proj(
                torch.from_numpy(vecs).float()).numpy()
            u_embs = raw_m.user_emb(
                torch.tensor(users, dtype=torch.long)).numpy()
            return u_embs @ all_proj.T

        for tag, vecs in [("raw_ub", X_new), ("frozen", recon_frz),
                          ("full", recon_full), ("stratified", recon_warm)]:
            val = _eval_recall(
                raw_m, pairs_new, pairs_old,
                lambda users, _v=vecs: _raw_scores(users, _v))
            result[f"raw_{tag}"] = float(val)
            log.info(f"  {tag:15s} {val:.4f}")

        # ==== Variant C: decoded-trained ranker (current) ====
        log.info("\n  --- decoded_ranker ---")
        dec_m = SimpleRanker(n_users, X_old.shape[1])
        _train_bpr(dec_m, pairs_old, n_items,
                   lambda idx: (recon_t0_t[idx],), epochs=15, seed=seed)
        dec_m.eval()

        def _dec_scores(users, vecs):
            all_proj = dec_m.item_proj(
                torch.from_numpy(vecs).float()).numpy()
            u_embs = dec_m.user_emb(
                torch.tensor(users, dtype=torch.long)).numpy()
            return u_embs @ all_proj.T

        for tag, vecs in [("frozen", recon_frz),
                          ("full_no_ds", recon_full),
                          ("stratified", recon_warm)]:
            val = _eval_recall(
                dec_m, pairs_new, pairs_old,
                lambda users, _v=vecs: _dec_scores(users, _v))
            result[f"dec_{tag}"] = float(val)
            log.info(f"  {tag:15s} {val:.4f}")

        dec_ds = SimpleRanker(n_users, X_new.shape[1])
        _train_bpr(dec_ds, pairs_new, n_items,
                   lambda idx: (recon_full_t[idx],),
                   epochs=15, seed=seed)
        dec_ds.eval()
        val = _eval_recall(
            dec_ds, pairs_new, pairs_old,
            lambda users: _dec_scores(users, recon_full))
        result["dec_full_ds"] = float(val)
        log.info(f"  {'full_ds':15s} {val:.4f}")

        all_results.append(result)

    log.info(f"\n{'='*80}")
    log.info(f"SUMMARY ({len(all_results)} seeds)")
    log.info(f"{'='*80}")
    for variant, keys in [
        ("Prefix-aware", ["pa_frozen", "pa_full_no_ds", "pa_full_ds",
                          "pa_stratified"]),
        ("Raw-trained", ["raw_raw_ub", "raw_frozen", "raw_full",
                         "raw_stratified"]),
        ("Decoded-trained", ["dec_frozen", "dec_full_no_ds", "dec_full_ds",
                             "dec_stratified"]),
    ]:
        log.info(f"\n  {variant}:")
        for k2 in keys:
            vals = [r[k2] for r in all_results]
            log.info(f"    {k2:25s} {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output",
                       default="results/theory/downstream_variants.json")
    args = parser.parse_args()

    log_path = Path(args.output).with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    X_old, X_new, pairs_old, pairs_new, n_users, n_items = load_movielens()
    results = run_variants(X_old, X_new, pairs_old, pairs_new,
                           n_users, n_items, seeds=args.seeds)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"variants": results}, f, indent=2)
    log.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

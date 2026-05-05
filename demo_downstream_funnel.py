#!/usr/bin/env python3
"""Funnel vs uniform downstream ranker + timing table.

Usage:
    python3 demo_downstream_funnel.py --seeds 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

sys.path.insert(0, os.path.dirname(__file__))
from rq import RQCodebook, warm_retrain


def load_movielens():
    data_dir = Path(__file__).parent.parent / "data" / "movielens" / "ml-1m"
    ratings = []
    with open(data_dir / "ratings.dat", encoding="latin-1") as f:
        for line in f:
            u, m, r, t = line.strip().split("::")
            ratings.append((int(u), int(m), float(r), int(t)))
    ratings = np.array(ratings)
    timestamps = ratings[:, 3]
    median_ts = np.median(timestamps)
    old_mask = timestamps < median_ts
    new_mask = ~old_mask
    all_users = np.unique(ratings[:, 0].astype(int))
    all_items = np.unique(ratings[:, 1].astype(int))
    user_map = {u: i for i, u in enumerate(all_users)}
    item_map = {m: i for i, m in enumerate(all_items)}
    n_users, n_items = len(all_users), len(all_items)

    def build_sparse(mask):
        r = ratings[mask]
        rows = [user_map[int(x)] for x in r[:, 0]]
        cols = [item_map[int(x)] for x in r[:, 1]]
        return csr_matrix((r[:, 2], (rows, cols)),
                         shape=(n_users, n_items), dtype=np.float32)

    def build_pairs(mask):
        r = ratings[mask]
        return [(user_map[int(x[0])], item_map[int(x[1])]) for x in r]

    mat_old = build_sparse(old_mask)
    mat_new = build_sparse(new_mask)
    d = 64
    _, S_old, Vt_old = svds(mat_old, k=min(d, min(mat_old.shape) - 1))
    _, S_new, Vt_new = svds(mat_new, k=min(d, min(mat_new.shape) - 1))
    X_old = (Vt_old.T * S_old).astype(np.float32)
    X_new = (Vt_new.T * S_new).astype(np.float32)
    return X_old, X_new, build_pairs(old_mask), build_pairs(new_mask), n_users, n_items


class Reco(nn.Module):
    def __init__(self, n_users, item_dim, emb_dim=32):
        super().__init__()
        self.item_proj = nn.Linear(item_dim, emb_dim)
        self.user_emb = nn.Embedding(n_users, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)

    def forward(self, user_ids, item_vecs):
        u = self.user_emb(user_ids)
        v = self.item_proj(item_vecs)
        return (u * v).sum(dim=-1)


def train_reco(model, pairs, item_recon, n_items,
               epochs=15, lr=0.005, batch_size=4096, n_neg=4, seed=42):
    rng = np.random.RandomState(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    recon_t = torch.from_numpy(item_recon).float()
    pairs_arr = np.array(pairs)
    for epoch in range(epochs):
        rng.shuffle(pairs_arr)
        for start in range(0, len(pairs_arr), batch_size):
            batch = pairs_arr[start:start + batch_size]
            users = torch.from_numpy(batch[:, 0].astype(np.int64))
            pos_vecs = recon_t[batch[:, 1].astype(np.int64)]
            neg_idx = rng.randint(0, n_items, size=(len(batch), n_neg))
            pos_scores = model(users, pos_vecs)
            neg_loss = sum(
                -F.logsigmoid(pos_scores - model(users, recon_t[neg_idx[:, j]])).mean()
                for j in range(n_neg)) / n_neg
            opt.zero_grad()
            neg_loss.backward()
            opt.step()


@torch.no_grad()
def eval_recall(model, test_pairs, train_pairs, item_recon, k=10):
    recon_t = torch.from_numpy(item_recon).float()
    all_proj = model.item_proj(recon_t).numpy()
    train_set = {}
    for u, i in train_pairs:
        train_set.setdefault(u, set()).add(i)
    test_set = {}
    for u, i in test_pairs:
        test_set.setdefault(u, set()).add(i)
    hits = total = 0
    for u, ti in test_set.items():
        if u not in train_set:
            continue
        u_emb = model.user_emb(torch.tensor([u]).long()).numpy()[0]
        scores = all_proj @ u_emb
        for t in train_set[u]:
            scores[t] = -np.inf
        top_k = set(np.argsort(scores)[-k:])
        hits += len(ti & top_k)
        total += len(ti)
    return hits / max(total, 1)


def run(X_old, X_new, pairs_old, pairs_new, n_users, n_items,
        arch_name, codes_per_stage, m=4, s=2, n_iter=20, seeds=10):
    results = []
    for seed in range(seeds):
        print(f"  {arch_name} seed {seed}...")

        # Train RQ
        t0 = time.time()
        rq_t0 = RQCodebook(m, codes_per_stage, X_old.shape[1])
        rq_t0.fit(X_old, n_iter=n_iter, seed=seed)
        time_rq_full = time.time() - t0

        recon_t0 = rq_t0.reconstruct(X_old)

        # Train ranker
        t0 = time.time()
        model = Reco(n_users, X_old.shape[1], emb_dim=32)
        train_reco(model, pairs_old, recon_t0, n_items, epochs=15, seed=seed)
        model.eval()
        time_ranker = time.time() - t0

        # Frozen
        recon_frz = rq_t0.reconstruct(X_new)
        recall_frz = eval_recall(model, pairs_new, pairs_old, recon_frz)
        mse_frz = rq_t0.mse(X_new)

        # Stratified
        t0 = time.time()
        rq_warm = warm_retrain(rq_t0, X_new, freeze_depth=s, n_iter=n_iter, seed=seed)
        time_suffix = time.time() - t0

        recon_warm = rq_warm.reconstruct(X_new)
        recall_strat = eval_recall(model, pairs_new, pairs_old, recon_warm)
        mse_warm = rq_warm.mse(X_new)

        # Full retrain
        t0 = time.time()
        rq_full = RQCodebook(m, codes_per_stage, X_new.shape[1])
        rq_full.fit(X_new, n_iter=n_iter, seed=seed + 1000)
        time_full = time.time() - t0

        recon_full = rq_full.reconstruct(X_new)
        recall_full_no = eval_recall(model, pairs_new, pairs_old, recon_full)
        mse_full = rq_full.mse(X_new)

        pfx_old = np.column_stack(rq_t0.encode(X_new, n_stages=s))
        pfx_full_c = np.column_stack(rq_full.encode(X_new, n_stages=s))
        pfx_changed = np.any(pfx_old != pfx_full_c, axis=1).mean()

        results.append({
            "seed": seed, "arch": arch_name,
            "recall_frozen": float(recall_frz),
            "recall_stratified": float(recall_strat),
            "recall_full_no_ds": float(recall_full_no),
            "mse_frozen": float(mse_frz),
            "mse_warm": float(mse_warm),
            "mse_full": float(mse_full),
            "prefix_change_rate": float(pfx_changed),
            "time_suffix_retrain": float(time_suffix),
            "time_full_retrain": float(time_full),
            "time_ranker_train": float(time_ranker),
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", type=str,
                       default="results/theory/downstream_funnel.json")
    args = parser.parse_args()

    print("Loading MovieLens...")
    X_old, X_new, po, pn, nu, ni = load_movielens()

    all_results = {}
    for name, cps in [("uniform", [64, 64, 64, 64]),
                       ("funnel", [16, 16, 256, 512])]:
        print(f"\n=== {name} ===")
        results = run(X_old, X_new, po, pn, nu, ni,
                     name, cps, seeds=args.seeds)
        all_results[name] = results

        r = {k: np.mean([d[k] for d in results]) for k in results[0] if isinstance(results[0][k], float)}
        print(f"\n  {name}:")
        print(f"    Frozen:     R@10={r['recall_frozen']:.4f}  MSE={r['mse_frozen']:.1f}")
        print(f"    Stratified: R@10={r['recall_stratified']:.4f}  MSE={r['mse_warm']:.1f}")
        print(f"    Full-noDS:  R@10={r['recall_full_no_ds']:.4f}  MSE={r['mse_full']:.1f}")
        print(f"    Pfx changed: {r['prefix_change_rate']:.0%}")
        print(f"    Time: suffix={r['time_suffix_retrain']:.1f}s, full={r['time_full_retrain']:.1f}s, ranker={r['time_ranker_train']:.1f}s")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

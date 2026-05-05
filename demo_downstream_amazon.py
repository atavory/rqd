#!/usr/bin/env python3
"""Downstream ranker stability on Amazon Electronics.

Same experiment as MovieLens but on Amazon data with temporal split.

Usage:
    python3 demo_downstream_amazon.py --seeds 10
"""

from __future__ import annotations

import argparse
import gzip
import json
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


def load_amazon(max_items=10000, min_reviews=5):
    """Load Amazon Electronics, temporal split, SVD embeddings."""
    data_path = Path(__file__).parent.parent / "data" / "amazon" / "electronics_5.json.gz"

    print("Loading Amazon Electronics...")
    reviews = []
    with gzip.open(data_path, "rt") as f:
        for line in f:
            d = json.loads(line)
            reviews.append((d["reviewerID"], d["asin"],
                          float(d["overall"]), int(d["unixReviewTime"])))

    print(f"  {len(reviews)} total reviews")

    # Filter to items with >= min_reviews
    from collections import Counter
    item_counts = Counter(r[1] for r in reviews)
    valid_items = {k for k, v in item_counts.items() if v >= min_reviews}
    reviews = [r for r in reviews if r[1] in valid_items]

    # Subsample items if too many
    all_items = list(set(r[1] for r in reviews))
    if len(all_items) > max_items:
        rng = np.random.RandomState(42)
        keep_items = set(rng.choice(all_items, max_items, replace=False))
        reviews = [r for r in reviews if r[1] in keep_items]

    # Build maps
    all_users = sorted(set(r[0] for r in reviews))
    all_items = sorted(set(r[1] for r in reviews))
    user_map = {u: i for i, u in enumerate(all_users)}
    item_map = {m: i for i, m in enumerate(all_items)}
    n_users, n_items = len(all_users), len(all_items)

    # Temporal split at median timestamp
    timestamps = np.array([r[3] for r in reviews])
    median_ts = np.median(timestamps)
    old_reviews = [r for r in reviews if r[3] < median_ts]
    new_reviews = [r for r in reviews if r[3] >= median_ts]

    print(f"  {n_users} users, {n_items} items")
    print(f"  Old: {len(old_reviews)}, New: {len(new_reviews)}")

    def build_sparse(revs):
        rows = [user_map[r[0]] for r in revs if r[0] in user_map and r[1] in item_map]
        cols = [item_map[r[1]] for r in revs if r[0] in user_map and r[1] in item_map]
        vals = [r[2] for r in revs if r[0] in user_map and r[1] in item_map]
        return csr_matrix((vals, (rows, cols)),
                         shape=(n_users, n_items), dtype=np.float32)

    def build_pairs(revs):
        return [(user_map[r[0]], item_map[r[1]])
                for r in revs if r[0] in user_map and r[1] in item_map]

    mat_old = build_sparse(old_reviews)
    mat_new = build_sparse(new_reviews)

    d = 64
    print(f"  SVD embeddings (d={d})...")
    k = min(d, min(mat_old.shape) - 1)
    _, S_old, Vt_old = svds(mat_old, k=k)
    _, S_new, Vt_new = svds(mat_new, k=k)

    X_old = (Vt_old.T * S_old).astype(np.float32)
    X_new = (Vt_new.T * S_new).astype(np.float32)

    pairs_old = build_pairs(old_reviews)
    pairs_new = build_pairs(new_reviews)

    return X_old, X_new, pairs_old, pairs_new, n_users, n_items


class RecoFromReconstruction(nn.Module):
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
        total_loss = 0
        n_batches = 0
        for start in range(0, len(pairs_arr), batch_size):
            batch = pairs_arr[start:start + batch_size]
            users = torch.from_numpy(batch[:, 0].astype(np.int64))
            pos_idx = batch[:, 1].astype(np.int64)
            pos_vecs = recon_t[pos_idx]
            neg_idx = rng.randint(0, n_items, size=(len(batch), n_neg))
            pos_scores = model(users, pos_vecs)
            neg_loss = 0
            for j in range(n_neg):
                neg_vecs = recon_t[neg_idx[:, j]]
                neg_scores = model(users, neg_vecs)
                neg_loss += -F.logsigmoid(pos_scores - neg_scores).mean()
            loss = neg_loss / n_neg
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch+1}: loss={total_loss/n_batches:.4f}")


@torch.no_grad()
def eval_recall(model, test_pairs, train_pairs, item_recon, k=10):
    recon_t = torch.from_numpy(item_recon).float()
    all_item_proj = model.item_proj(recon_t).numpy()
    train_set = {}
    for u, i in train_pairs:
        train_set.setdefault(u, set()).add(i)
    test_set = {}
    for u, i in test_pairs:
        test_set.setdefault(u, set()).add(i)
    hits = 0
    total = 0
    for u, test_items in test_set.items():
        if u not in train_set:
            continue
        u_emb = model.user_emb(torch.tensor([u]).long()).numpy()[0]
        scores = all_item_proj @ u_emb
        for ti in train_set[u]:
            scores[ti] = -np.inf
        top_k = set(np.argsort(scores)[-k:])
        hits += len(test_items & top_k)
        total += len(test_items)
    return hits / max(total, 1)


def run_experiment(X_old, X_new, pairs_old, pairs_new,
                   n_users, n_items, m=4, K=64, s=2,
                   n_iter=20, seeds=10):
    results = []
    for seed in range(seeds):
        print(f"\n=== Seed {seed} ===")
        rq_t0 = RQCodebook(m, [K] * m, X_old.shape[1])
        rq_t0.fit(X_old, n_iter=n_iter, seed=seed)
        recon_t0 = rq_t0.reconstruct(X_old)

        print("  Training recommender...")
        model = RecoFromReconstruction(n_users, X_old.shape[1], emb_dim=32)
        train_reco(model, pairs_old, recon_t0, n_items, epochs=15, seed=seed)
        model.eval()

        # Frozen
        recon_frz = rq_t0.reconstruct(X_new)
        recall_frz = eval_recall(model, pairs_new, pairs_old, recon_frz)
        mse_frz = rq_t0.mse(X_new)

        # Full retrain, no DS
        rq_full = RQCodebook(m, [K] * m, X_new.shape[1])
        rq_full.fit(X_new, n_iter=n_iter, seed=seed + 1000)
        recon_full = rq_full.reconstruct(X_new)
        recall_full_no = eval_recall(model, pairs_new, pairs_old, recon_full)
        mse_full = rq_full.mse(X_new)

        # Full retrain + DS retrain
        model_new = RecoFromReconstruction(n_users, X_new.shape[1], emb_dim=32)
        train_reco(model_new, pairs_new, recon_full, n_items, epochs=15, seed=seed)
        model_new.eval()
        recall_full_ds = eval_recall(model_new, pairs_new, pairs_old, recon_full)

        # Stratified
        rq_warm = warm_retrain(rq_t0, X_new, freeze_depth=s, n_iter=n_iter, seed=seed)
        recon_warm = rq_warm.reconstruct(X_new)
        recall_strat = eval_recall(model, pairs_new, pairs_old, recon_warm)
        mse_warm = rq_warm.mse(X_new)

        pfx_old = np.column_stack(rq_t0.encode(X_new, n_stages=s))
        pfx_full = np.column_stack(rq_full.encode(X_new, n_stages=s))
        pfx_changed = np.any(pfx_old != pfx_full, axis=1).mean()

        print(f"  Frozen:     R@10={recall_frz:.4f}  MSE={mse_frz:.1f}")
        print(f"  Full-noDS:  R@10={recall_full_no:.4f}  MSE={mse_full:.1f}")
        print(f"  Full+DS:    R@10={recall_full_ds:.4f}  MSE={mse_full:.1f}")
        print(f"  Stratified: R@10={recall_strat:.4f}  MSE={mse_warm:.1f}")

        results.append({
            "seed": seed,
            "recall_frozen": float(recall_frz),
            "recall_full_no_ds": float(recall_full_no),
            "recall_full_ds": float(recall_full_ds),
            "recall_stratified": float(recall_strat),
            "mse_frozen": float(mse_frz),
            "mse_full": float(mse_full),
            "mse_warm": float(mse_warm),
            "prefix_change_rate": float(pfx_changed),
        })

    r = {k: np.mean([d[k] for d in results]) for k in results[0]}
    s_dev = {k: np.std([d[k] for d in results]) for k in results[0]}
    print(f"\n{'='*80}")
    print(f"  Amazon Electronics ({len(results)} seeds)")
    print(f"{'='*80}")
    print(f"{'Method':<35} {'Pfx chg':>8} {'DS retr':>8} {'R@10':>12} {'RQ MSE':>12}")
    print("-" * 80)
    print(f"{'Frozen':<35} {'0%':>8} {'no':>8} {r['recall_frozen']:.4f}±{s_dev['recall_frozen']:.4f} {r['mse_frozen']:>8.1f}")
    print(f"{'Full retrain, no DS retrain':<35} {r['prefix_change_rate']:>7.0%} {'no':>8} {r['recall_full_no_ds']:.4f}±{s_dev['recall_full_no_ds']:.4f} {r['mse_full']:>8.1f}")
    print(f"{'Full retrain + DS retrain':<35} {r['prefix_change_rate']:>7.0%} {'yes':>8} {r['recall_full_ds']:.4f}±{s_dev['recall_full_ds']:.4f} {r['mse_full']:>8.1f}")
    print(f"{'Stratified plasticity':<35} {'0%':>8} {'no':>8} {r['recall_stratified']:.4f}±{s_dev['recall_stratified']:.4f} {r['mse_warm']:>8.1f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", type=str,
                       default="results/theory/downstream_amazon.json")
    args = parser.parse_args()

    X_old, X_new, pairs_old, pairs_new, n_users, n_items = load_amazon()
    results = run_experiment(X_old, X_new, pairs_old, pairs_new,
                            n_users, n_items, seeds=args.seeds)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"amazon": results}, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

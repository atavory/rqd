#!/usr/bin/env python3
"""Downstream recommender stability experiment.

The recommender consumes RQ-reconstructed item vectors.
After drift, full retrain changes reconstructions → breaks recommender.
Stratified plasticity: prefix stable, suffix improves reconstruction
→ recommender improves without retraining.

Usage:
    python3 demo_downstream.py --seeds 10 --output results/theory/downstream.json
"""

from __future__ import annotations

import argparse
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


def load_movielens():
    """Load MovieLens-1M, temporal split, SVD embeddings."""
    data_dir = Path(__file__).parent.parent / "data" / "movielens" / "ml-1m"

    print("Loading MovieLens-1M...")
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
    print(f"  SVD embeddings (d={d})...")
    _, S_old, Vt_old = svds(mat_old, k=min(d, min(mat_old.shape) - 1))
    _, S_new, Vt_new = svds(mat_new, k=min(d, min(mat_new.shape) - 1))

    X_old = (Vt_old.T * S_old).astype(np.float32)
    X_new = (Vt_new.T * S_new).astype(np.float32)

    pairs_old = build_pairs(old_mask)
    pairs_new = build_pairs(new_mask)

    print(f"  {n_users} users, {n_items} items")
    return X_old, X_new, pairs_old, pairs_new, n_users, n_items


class RecoFromReconstruction(nn.Module):
    """Two-tower recommender that consumes RQ-reconstructed item vectors.

    Item tower: linear projection of the RQ reconstruction.
    User tower: learned user embedding.
    Score: dot product.
    """

    def __init__(self, n_users, item_dim, emb_dim=32):
        super().__init__()
        self.item_proj = nn.Linear(item_dim, emb_dim)
        self.user_emb = nn.Embedding(n_users, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)

    def forward(self, user_ids, item_vecs):
        """user_ids: (batch,) long, item_vecs: (batch, d) float."""
        u = self.user_emb(user_ids)
        v = self.item_proj(item_vecs)
        return (u * v).sum(dim=-1)


def train_reco(model, pairs, item_reconstructions, n_items,
               epochs=15, lr=0.005, batch_size=4096, n_neg=4, seed=42):
    """Train with BPR on RQ-reconstructed item vectors."""
    rng = np.random.RandomState(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    recon_t = torch.from_numpy(item_reconstructions).float()

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
def eval_recall(model, test_pairs, train_pairs, item_reconstructions,
                k=10):
    """Recall@K using RQ-reconstructed item vectors."""
    recon_t = torch.from_numpy(item_reconstructions).float()
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
                   n_iter=20, seeds=3):
    results = []

    for seed in range(seeds):
        print(f"\n=== Seed {seed} ===")

        # 1. Train RQ at T0
        rq_t0 = RQCodebook(m, [K] * m, X_old.shape[1])
        rq_t0.fit(X_old, n_iter=n_iter, seed=seed)
        recon_t0 = rq_t0.reconstruct(X_old)

        # 2. Train recommender on T0 RQ reconstructions
        print("  Training recommender on T0 reconstructions...")
        model = RecoFromReconstruction(n_users, X_old.shape[1], emb_dim=32)
        train_reco(model, pairs_old, recon_t0, n_items,
                  epochs=15, seed=seed)
        model.eval()

        # Baseline recall
        recall_t0 = eval_recall(model, pairs_old, pairs_old, recon_t0)
        print(f"  T0 R@10: {recall_t0:.4f}")

        # 3. FROZEN: old codebook on new embeddings
        recon_frz = rq_t0.reconstruct(X_new)
        recall_frozen = eval_recall(model, pairs_new, pairs_old, recon_frz)
        mse_frozen = rq_t0.mse(X_new)
        print(f"  Frozen:      R@10={recall_frozen:.4f}  MSE={mse_frozen:.1f}")

        # 4. FULL RETRAIN: new codebook, old recommender
        rq_full = RQCodebook(m, [K] * m, X_new.shape[1])
        rq_full.fit(X_new, n_iter=n_iter, seed=seed + 1000)
        recon_full = rq_full.reconstruct(X_new)
        recall_full_no_ds = eval_recall(model, pairs_new, pairs_old,
                                       recon_full)
        mse_full = rq_full.mse(X_new)
        print(f"  Full-noDS:   R@10={recall_full_no_ds:.4f}  MSE={mse_full:.1f}")

        # 5. FULL RETRAIN + DS RETRAIN
        model_new = RecoFromReconstruction(n_users, X_new.shape[1],
                                          emb_dim=32)
        train_reco(model_new, pairs_new, recon_full, n_items,
                  epochs=15, seed=seed)
        model_new.eval()
        recall_full_ds = eval_recall(model_new, pairs_new, pairs_old,
                                    recon_full)
        print(f"  Full+DS:     R@10={recall_full_ds:.4f}  MSE={mse_full:.1f}")

        # 6. STRATIFIED PLASTICITY: old recommender, improved reconstruction
        rq_warm = warm_retrain(rq_t0, X_new, freeze_depth=s,
                              n_iter=n_iter, seed=seed)
        recon_warm = rq_warm.reconstruct(X_new)
        recall_strat = eval_recall(model, pairs_new, pairs_old,
                                  recon_warm)
        mse_warm = rq_warm.mse(X_new)
        print(f"  Stratified:  R@10={recall_strat:.4f}  MSE={mse_warm:.1f}")

        pfx_old = np.column_stack(rq_t0.encode(X_new, n_stages=s))
        pfx_full = np.column_stack(rq_full.encode(X_new, n_stages=s))
        pfx_warm = np.column_stack(rq_warm.encode(X_new, n_stages=s))
        pfx_changed = np.any(pfx_old != pfx_full, axis=1).mean()
        pfx_stable = np.all(pfx_old == pfx_warm, axis=1).mean()
        print(f"  Pfx changed: {pfx_changed:.0%}, stable: {pfx_stable:.0%}")

        results.append({
            "seed": seed,
            "recall_t0": float(recall_t0),
            "recall_frozen": float(recall_frozen),
            "recall_full_no_ds": float(recall_full_no_ds),
            "recall_full_ds": float(recall_full_ds),
            "recall_stratified": float(recall_strat),
            "mse_frozen": float(mse_frozen),
            "mse_full": float(mse_full),
            "mse_warm": float(mse_warm),
            "prefix_change_rate": float(pfx_changed),
            "prefix_stability": float(pfx_stable),
        })

    r = {k: np.mean([d[k] for d in results]) for k in results[0]}
    s_dev = {k: np.std([d[k] for d in results]) for k in results[0]}

    print(f"\n{'='*80}")
    print(f"  MovieLens-1M ({len(results)} seeds)")
    print(f"{'='*80}")
    print(f"{'Method':<35} {'Pfx chg':>8} {'DS retr':>8} {'R@10':>12} {'RQ MSE':>12}")
    print("-" * 80)
    print(f"{'Frozen':<35} {'0%':>8} {'no':>8} {r['recall_frozen']:>.4f}±{s_dev['recall_frozen']:.4f} {r['mse_frozen']:>8.1f}")
    print(f"{'Full retrain, no DS retrain':<35} {r['prefix_change_rate']:>7.0%} {'no':>8} {r['recall_full_no_ds']:>.4f}±{s_dev['recall_full_no_ds']:.4f} {r['mse_full']:>8.1f}")
    print(f"{'Full retrain + DS retrain':<35} {r['prefix_change_rate']:>7.0%} {'yes':>8} {r['recall_full_ds']:>.4f}±{s_dev['recall_full_ds']:.4f} {r['mse_full']:>8.1f}")
    print(f"{'Stratified plasticity':<35} {'0%':>8} {'no':>8} {r['recall_stratified']:>.4f}±{s_dev['recall_stratified']:.4f} {r['mse_warm']:>8.1f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", type=str,
                       default="results/theory/downstream.json")
    args = parser.parse_args()

    X_old, X_new, pairs_old, pairs_new, n_users, n_items = load_movielens()
    results = run_experiment(X_old, X_new, pairs_old, pairs_new,
                            n_users, n_items, seeds=args.seeds)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"movielens": results}, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

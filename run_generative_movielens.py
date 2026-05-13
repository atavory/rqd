#!/usr/bin/env python3
"""Generative recommender on real MovieLens sequences with RQ semantic IDs.

Trains a causal transformer on user interaction sequences where each item
is represented by its 4-token RQ semantic ID. Evaluates next-item prediction
(HR@10, NDCG@10) after temporal drift under four codebook strategies.

Key result: full codebook retraining breaks the old model's predictions
because the token vocabulary meaning changes. Stratified plasticity
preserves the prefix vocabulary so the old model remains functional.

Usage:
    python3 run_generative_movielens.py --seeds 3
    python3 run_generative_movielens.py --seeds 3 --device cuda
"""
from __future__ import annotations
import argparse, json, logging, math, os, time, urllib.request, zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download_movielens(data_dir="/tmp/ml-1m"):
    if os.path.exists(os.path.join(data_dir, "ratings.dat")):
        return data_dir
    zip_path = "/tmp/ml-1m.zip"
    if not os.path.exists(zip_path):
        log.info("Downloading MovieLens-1M...")
        urllib.request.urlretrieve(ML1M_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("/tmp")
    return data_dir


def load_movielens(data_dir):
    ratings = []
    with open(os.path.join(data_dir, "ratings.dat"), "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            uid, iid, rating, ts = int(parts[0]), int(parts[1]), float(parts[2]), int(parts[3])
            ratings.append((uid, iid, rating, ts))
    return ratings


def build_embeddings_and_sequences(ratings, min_seq_len=5, max_seq_len=20, emb_dim=64):
    item_set = sorted(set(r[1] for r in ratings))
    item_to_idx = {iid: i for i, iid in enumerate(item_set)}
    n_items = len(item_set)
    n_users = max(r[0] for r in ratings) + 1

    R = np.zeros((n_users, n_items), dtype=np.float32)
    for uid, iid, rating, ts in ratings:
        R[uid, item_to_idx[iid]] = rating

    norms = np.linalg.norm(R, axis=1, keepdims=True) + 1e-8
    R_norm = R / norms

    U, S, Vt = np.linalg.svd(R_norm, full_matrices=False)
    item_embs = (Vt[:emb_dim].T * S[:emb_dim]).astype(np.float32)
    norms = np.linalg.norm(item_embs, axis=1, keepdims=True) + 1e-8
    item_embs = item_embs / norms

    user_seqs = {}
    for uid, iid, rating, ts in ratings:
        if uid not in user_seqs:
            user_seqs[uid] = []
        user_seqs[uid].append((ts, item_to_idx[iid]))

    sequences = []
    for uid, items in user_seqs.items():
        items.sort()
        item_ids = [iid for _, iid in items]
        if len(item_ids) >= min_seq_len:
            sequences.append(item_ids[-max_seq_len:])

    return item_embs, sequences, item_to_idx


def _kmeans(X, k, n_iter=20, rng=None, init=None):
    if rng is None:
        rng = np.random.RandomState(42)
    n = len(X)
    if init is not None:
        centroids = init.copy()
    else:
        centroids = np.zeros((k, X.shape[1]), dtype=np.float32)
        centroids[0] = X[rng.randint(n)]
        for i in range(1, k):
            d = np.min(np.sum((X[:, None, :] - centroids[None, :i, :]) ** 2, axis=2), axis=1)
            t = d.sum()
            centroids[i] = X[rng.choice(n, p=d / max(t, 1e-12))] if t > 1e-12 else X[rng.randint(n)]
    for _ in range(n_iter):
        a = np.argmin(np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2), axis=1)
        for j in range(k):
            m = a == j
            if m.sum() > 0:
                centroids[j] = X[m].mean(axis=0)
    return centroids


def _assign(X, c):
    return np.argmin(np.sum((X[:, None, :] - c[None, :, :]) ** 2, axis=2), axis=1).astype(np.int64)


class RQ:
    def __init__(self, m, codes, dim):
        self.m, self.dim = m, dim
        self.K = [codes] * m if isinstance(codes, int) else list(codes)
        self.cb = []

    def fit(self, X, n_iter=20, seed=42):
        rng = np.random.RandomState(seed)
        r = X.copy()
        self.cb = []
        for i in range(self.m):
            c = _kmeans(r, self.K[i], n_iter=n_iter, rng=rng)
            self.cb.append(c)
            a = _assign(r, c)
            r = r - c[a]
        return self

    def encode(self, X):
        r = X.copy()
        codes = []
        for i in range(self.m):
            a = _assign(r, self.cb[i])
            codes.append(a)
            r = r - self.cb[i][a]
        return np.stack(codes, axis=1)

    def mse(self, X):
        r = X.copy()
        for c in self.cb:
            a = _assign(r, c)
            r = r - c[a]
        return float(np.mean(np.sum(r ** 2, axis=1)))


def warm_retrain(rq, X, fd, n_iter=20, seed=42):
    rq2 = RQ(rq.m, rq.K, rq.dim)
    rq2.cb = [c.copy() for c in rq.cb]
    r = X.copy()
    for i in range(fd):
        a = _assign(r, rq2.cb[i])
        r = r - rq2.cb[i][a]
    rng = np.random.RandomState(seed)
    for i in range(fd, rq.m):
        c = _kmeans(r, rq.K[i], n_iter=n_iter, rng=rng, init=rq2.cb[i])
        rq2.cb[i] = c
        a = _assign(r, c)
        r = r - c[a]
    return rq2


class SeqTransformer(nn.Module):
    def __init__(self, n_items, vocab_sizes, d_model=128, n_heads=4,
                 n_layers=3, max_seq_len=20, dropout=0.1):
        super().__init__()
        self.n_items = n_items
        self.m = len(vocab_sizes)
        self.tokens_per_item = self.m
        self.max_tokens = max_seq_len * self.tokens_per_item
        self.d_model = d_model
        self.vocab_sizes = vocab_sizes
        max_vocab = max(vocab_sizes)
        self.tok_embs = nn.ModuleList([
            nn.Embedding(vs, d_model) for vs in vocab_sizes
        ])
        self.pos_emb = nn.Embedding(self.max_tokens + 1, d_model)
        self.stage_emb = nn.Embedding(self.m, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.heads = nn.ModuleList([nn.Linear(d_model, vs) for vs in vocab_sizes])

    def forward(self, token_ids, stage_ids):
        B, T = token_ids.shape
        embs = torch.zeros(B, T, self.d_model, device=token_ids.device)
        for s in range(self.m):
            mask = stage_ids == s
            if mask.any():
                embs[mask] = self.tok_embs[s](token_ids[mask])
        pos = torch.arange(T, device=token_ids.device).unsqueeze(0).expand(B, -1)
        embs = embs + self.pos_emb(pos) + self.stage_emb(stage_ids)
        causal_mask = torch.triu(
            torch.ones(T, T, device=token_ids.device), diagonal=1
        ).bool()
        out = self.transformer(embs, mask=causal_mask)
        return out

    def predict_next_item(self, out, stage_ids):
        logits = []
        for s in range(self.m):
            mask = stage_ids == s
            logits.append(self.heads[s](out[mask]))
        return logits


def seqs_to_tokens(sequences, item_codes, m, max_seq_len=20):
    all_token_ids = []
    all_stage_ids = []
    all_targets = []
    for seq in sequences:
        if len(seq) < 3:
            continue
        seq = seq[-max_seq_len:]
        tokens = []
        stages = []
        for item_id in seq:
            codes = item_codes[item_id]
            for s in range(m):
                tokens.append(codes[s])
                stages.append(s)
        all_token_ids.append(tokens)
        all_stage_ids.append(stages)
        all_targets.append(seq[-1])
    return all_token_ids, all_stage_ids, all_targets


def pad_batch(token_lists, stage_lists, max_len):
    B = len(token_lists)
    tokens = np.zeros((B, max_len), dtype=np.int64)
    stages = np.zeros((B, max_len), dtype=np.int64)
    lengths = []
    for i in range(B):
        L = min(len(token_lists[i]), max_len)
        tokens[i, :L] = token_lists[i][:L]
        stages[i, :L] = stage_lists[i][:L]
        lengths.append(L)
    return tokens, stages, lengths


def train_model(model, token_lists, stage_lists, m,
                epochs=30, lr=3e-4, batch_size=128, device="cpu"):
    model = model.to(device).train()
    max_len = max(len(t) for t in token_lists)
    tokens_np, stages_np, lengths = pad_batch(token_lists, stage_lists, max_len)
    tokens_t = torch.from_numpy(tokens_np).long().to(device)
    stages_t = torch.from_numpy(stages_np).long().to(device)
    n = len(token_lists)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            bt = tokens_t[idx]
            bs = stages_t[idx]
            inp = bt[:, :-1]
            tgt = bt[:, 1:]
            si = bs[:, :-1]
            st = bs[:, 1:]
            out = model(inp, si)
            loss = 0.0
            count = 0
            for s in range(m):
                mask = st == s
                if mask.any():
                    logits_s = model.heads[s](out[mask])
                    targets_s = tgt[mask]
                    loss = loss + F.cross_entropy(logits_s, targets_s)
                    count += 1
            if count > 0:
                loss = loss / count
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * len(idx)
        scheduler.step()
    model.eval()
    return model


@torch.no_grad()
def evaluate_nextitem(model, token_lists, stage_lists, targets,
                      item_codes, m, device="cpu", top_k=10):
    model.eval()
    max_len = max(len(t) for t in token_lists)
    tokens_np, stages_np, lengths = pad_batch(token_lists, stage_lists, max_len)

    n_items = len(item_codes)
    hits = 0
    ndcg = 0.0
    total = 0
    batch_size = 256

    for start in range(0, len(token_lists), batch_size):
        end = min(start + batch_size, len(token_lists))
        bt = torch.from_numpy(tokens_np[start:end]).long().to(device)
        bs = torch.from_numpy(stages_np[start:end]).long().to(device)
        batch_targets = targets[start:end]
        B = bt.shape[0]

        context = bt[:, :-m]
        context_s = bs[:, :-m]
        if context.shape[1] == 0:
            continue

        out = model(context, context_s)
        last_hidden = out[:, -1, :]

        item_codes_t = torch.from_numpy(item_codes).long().to(device)
        scores = torch.zeros(B, n_items, device=device)
        for s in range(m):
            logits_s = model.heads[s](last_hidden)
            log_probs = F.log_softmax(logits_s, dim=-1)
            scores += log_probs[:, item_codes_t[:, s]]

        _, topk_indices = scores.topk(top_k, dim=-1)

        for b in range(B):
            target = batch_targets[b]
            topk = topk_indices[b].cpu().numpy()
            if target in topk:
                hits += 1
                rank = np.where(topk == target)[0][0]
                ndcg += 1.0 / math.log2(rank + 2)
            total += 1

    hr = hits / max(total, 1)
    ndcg_val = ndcg / max(total, 1)
    return {"hr@10": hr, "ndcg@10": ndcg_val, "n_eval": total}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--json-output", type=str, default="generative_movielens.json")
    args = p.parse_args()

    t0 = time.time()
    results = []
    m = 4
    fd = 2
    codes_list = [16, 16, 256, 256]

    data_dir = download_movielens()
    ratings = load_movielens(data_dir)
    log.info(f"Loaded {len(ratings)} ratings")

    timestamps = [r[3] for r in ratings]
    median_ts = int(np.median(timestamps))
    ratings_t0 = [r for r in ratings if r[3] <= median_ts]
    ratings_t1 = [r for r in ratings if r[3] > median_ts]
    log.info(f"T0: {len(ratings_t0)} ratings, T1: {len(ratings_t1)} ratings")

    emb_dim = 64
    item_embs_t0, seqs_t0, item_to_idx = build_embeddings_and_sequences(
        ratings_t0, emb_dim=emb_dim
    )
    item_embs_t1, seqs_t1, _ = build_embeddings_and_sequences(
        ratings_t1, emb_dim=emb_dim
    )
    n_items = len(item_embs_t0)
    log.info(f"{n_items} items, {len(seqs_t0)} T0 seqs, {len(seqs_t1)} T1 seqs")

    for seed in range(args.seeds):
        log.info(f"=== seed={seed} ===")

        log.info("  fitting RQ codebooks...")
        rq_source = RQ(m, codes_list, emb_dim).fit(item_embs_t0, seed=seed)
        rq_full = RQ(m, codes_list, emb_dim).fit(item_embs_t1, seed=seed + 500)
        rq_strat = warm_retrain(rq_source, item_embs_t1, fd, seed=seed)

        codes_t0_source = rq_source.encode(item_embs_t0)
        codes_t1_source = rq_source.encode(item_embs_t1)
        codes_t1_full = rq_full.encode(item_embs_t1)
        codes_t1_strat = rq_strat.encode(item_embs_t1)

        token_lists_t0, stage_lists_t0, targets_t0 = seqs_to_tokens(
            seqs_t0, codes_t0_source, m
        )
        log.info(f"  {len(token_lists_t0)} training sequences")

        log.info("  training source model...")
        torch.manual_seed(seed)
        model = SeqTransformer(
            n_items, codes_list, d_model=128, n_heads=4, n_layers=3
        )
        model = train_model(
            model, token_lists_t0, stage_lists_t0, m,
            epochs=args.epochs, device=args.device
        )

        strategies = {
            "frozen": codes_t1_source,
            "stratified": codes_t1_strat,
            "full_old_model": codes_t1_full,
        }

        for sname, item_codes in strategies.items():
            token_lists, stage_lists, targets = seqs_to_tokens(
                seqs_t1, item_codes, m
            )
            metrics = evaluate_nextitem(
                model, token_lists, stage_lists, np.array(targets),
                item_codes, m, device=args.device
            )
            log.info(f"  {sname}: HR@10={metrics['hr@10']:.4f} "
                     f"NDCG@10={metrics['ndcg@10']:.4f}")
            results.append({
                "seed": seed, "strategy": sname, **metrics,
                "mse": rq_source.mse(item_embs_t1) if sname == "frozen"
                       else rq_strat.mse(item_embs_t1) if sname == "stratified"
                       else rq_full.mse(item_embs_t1),
            })

        log.info("  training full-retrain model...")
        torch.manual_seed(seed + 1000)
        model_full = SeqTransformer(
            n_items, codes_list, d_model=128, n_heads=4, n_layers=3
        )
        token_lists_full, stage_lists_full, targets_full = seqs_to_tokens(
            seqs_t1, codes_t1_full, m
        )
        model_full = train_model(
            model_full, token_lists_full, stage_lists_full, m,
            epochs=args.epochs, device=args.device
        )
        metrics = evaluate_nextitem(
            model_full, token_lists_full, stage_lists_full,
            np.array(targets_full), codes_t1_full, m, device=args.device
        )
        log.info(f"  full_new_model: HR@10={metrics['hr@10']:.4f} "
                 f"NDCG@10={metrics['ndcg@10']:.4f}")
        results.append({
            "seed": seed, "strategy": "full_new_model", **metrics,
            "mse": rq_full.mse(item_embs_t1),
        })

        with open(args.json_output, "w") as f:
            json.dump(results, f)

    log.info(f"Done in {time.time() - t0:.0f}s. {len(results)} rows.")


if __name__ == "__main__":
    main()

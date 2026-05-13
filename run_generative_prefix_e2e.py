#!/usr/bin/env python3
"""E2E: Prefix generator + trained ranker resolver on MovieLens.

End-to-end generative recommendation pipeline:
1. Prefix generator (frozen, trained on T0) routes via prefix tokens
2. Trained ranker (frozen, trained on T0 raw vectors) scores candidates
   using their current-codebook decoded vectors
3. No oracle — only user history and learned models at inference

This is the full story: old generator routes, old ranker scores,
stratified codebook improves item representations without any
downstream retraining.

Usage:
    python3 run_generative_prefix_e2e.py --seeds 5
    python3 run_generative_prefix_e2e.py --seeds 5 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "algs"))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


def load_movielens_shared_basis(emb_dim=64, min_seq_len=5, max_seq_len=15):
    """Load MovieLens with shared-basis SVD (no rotation ambiguity)."""
    data_dir = "/tmp/ml-1m"
    if not os.path.exists(os.path.join(data_dir, "ratings.dat")):
        import urllib.request
        import zipfile

        urllib.request.urlretrieve(
            "https://files.grouplens.org/datasets/movielens/ml-1m.zip", "/tmp/ml-1m.zip"
        )
        with zipfile.ZipFile("/tmp/ml-1m.zip") as z:
            z.extractall("/tmp")

    ratings = []
    with open(os.path.join(data_dir, "ratings.dat"), "r", encoding="latin-1") as f:
        for line in f:
            p = line.strip().split("::")
            ratings.append((int(p[0]), int(p[1]), float(p[2]), int(p[3])))

    all_users = sorted({r[0] for r in ratings})
    all_items = sorted({r[1] for r in ratings})
    user_map = {u: i for i, u in enumerate(all_users)}
    item_map = {it: i for i, it in enumerate(all_items)}
    n_users, n_items = len(all_users), len(all_items)

    median_ts = np.median([r[3] for r in ratings])

    rows_old, cols_old, vals_old = [], [], []
    rows_new, cols_new, vals_new = [], [], []
    seqs_t0_raw, seqs_t1_raw = {}, {}
    seen_t0 = {}

    for uid, iid, rating, ts in ratings:
        u, i = user_map[uid], item_map[iid]
        if ts < median_ts:
            rows_old.append(u)
            cols_old.append(i)
            vals_old.append(rating)
            seqs_t0_raw.setdefault(u, []).append((ts, i))
            seen_t0.setdefault(u, set()).add(i)
        else:
            rows_new.append(u)
            cols_new.append(i)
            vals_new.append(rating)
            seqs_t1_raw.setdefault(u, []).append((ts, i))

    M_old = csr_matrix(
        (vals_old, (rows_old, cols_old)), shape=(n_users, n_items), dtype=np.float32
    )
    M_new = csr_matrix(
        (vals_new, (rows_new, cols_new)), shape=(n_users, n_items), dtype=np.float32
    )

    k = min(emb_dim, min(M_old.shape) - 1)
    U_old, S_old, Vt_old = svds(M_old, k=k)
    embs_t0 = (Vt_old.T * S_old).astype(np.float32)
    embs_t1 = (M_new.T @ (U_old * (1.0 / (S_old + 1e-8)))).astype(np.float32)
    embs_t1 = (embs_t1 * S_old).astype(np.float32)

    def make_seqs(raw):
        out = []
        for _uid, items in raw.items():
            items.sort()
            ids = [i for _, i in items]
            if len(ids) >= min_seq_len:
                out.append(ids[-max_seq_len:])
        return out

    def make_eval(raw):
        out = []
        for uid, items in raw.items():
            items.sort()
            ids = [i for _, i in items]
            if len(ids) >= min_seq_len:
                seq = ids[-max_seq_len:]
                out.append((uid, seq[:-1], seq[-1]))
        return out

    return (
        embs_t0,
        embs_t1,
        make_seqs(seqs_t0_raw),
        make_eval(seqs_t1_raw),
        seen_t0,
        n_users,
        n_items,
    )


# === RQ ===
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
            d = np.min(
                np.sum((X[:, None, :] - centroids[None, :i, :]) ** 2, axis=2), axis=1
            )
            t = d.sum()
            centroids[i] = (
                X[rng.choice(n, p=d / max(t, 1e-12))]
                if t > 1e-12
                else X[rng.randint(n)]
            )
    for _ in range(n_iter):
        a = np.argmin(
            np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2), axis=1
        )
        for j in range(k):
            m = a == j
            if m.sum() > 0:
                centroids[j] = X[m].mean(axis=0)
    return centroids


def _assign(X, c):
    return np.argmin(
        np.sum((X[:, None, :] - c[None, :, :]) ** 2, axis=2), axis=1
    ).astype(np.int64)


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

    def decode_codes(self, codes):
        out = np.zeros((len(codes), self.dim), dtype=np.float32)
        for i in range(self.m):
            out += self.cb[i][codes[:, i]]
        return out

    def mse(self, X):
        r = X.copy()
        for c in self.cb:
            a = _assign(r, c)
            r = r - c[a]
        return float(np.mean(np.sum(r**2, axis=1)))


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


# === Prefix-only generator ===
class PrefixGenerator(nn.Module):
    """Predicts only prefix tokens (stages 0..fd-1) from user history."""

    def __init__(
        self,
        vocab_sizes_prefix,
        d_model=128,
        n_heads=4,
        n_layers=3,
        max_tokens=60,
        dropout=0.1,
    ):
        super().__init__()
        self.n_prefix = len(vocab_sizes_prefix)
        self.d_model = d_model
        self.tok_embs = nn.ModuleList(
            [nn.Embedding(vs, d_model) for vs in vocab_sizes_prefix]
        )
        self.pos_emb = nn.Embedding(max_tokens, d_model)
        self.stage_emb = nn.Embedding(self.n_prefix, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.heads = nn.ModuleList(
            [nn.Linear(d_model, vs) for vs in vocab_sizes_prefix]
        )

    def forward(self, token_ids, stage_ids):
        B, T = token_ids.shape
        embs = torch.zeros(B, T, self.d_model, device=token_ids.device)
        for s in range(self.n_prefix):
            mask = stage_ids == s
            if mask.any():
                embs[mask] = self.tok_embs[s](token_ids[mask])
        pos = torch.arange(T, device=token_ids.device).unsqueeze(0)
        embs = embs + self.pos_emb(pos) + self.stage_emb(stage_ids)
        causal = torch.triu(
            torch.ones(T, T, device=token_ids.device), diagonal=1
        ).bool()
        return self.transformer(embs, mask=causal)

    @torch.no_grad()
    def generate_prefixes(self, token_ids, stage_ids, n_beams=10):
        B = token_ids.shape[0]
        device = token_ids.device
        all_prefixes, all_scores = [], []
        for b in range(B):
            ctx_tok = token_ids[b : b + 1]
            ctx_stg = stage_ids[b : b + 1]
            beams = [([], 0.0)]
            for s in range(self.n_prefix):
                new_beams = []
                for prev, prev_s in beams:
                    if prev:
                        ext_tok = torch.cat(
                            [
                                ctx_tok,
                                torch.tensor([prev], dtype=torch.long, device=device),
                            ],
                            dim=1,
                        )
                        ext_stg = torch.cat(
                            [
                                ctx_stg,
                                torch.tensor(
                                    [list(range(len(prev)))],
                                    dtype=torch.long,
                                    device=device,
                                ),
                            ],
                            dim=1,
                        )
                    else:
                        ext_tok, ext_stg = ctx_tok, ctx_stg
                    out = self.forward(ext_tok, ext_stg)
                    logits = self.heads[s](out[:, -1, :])
                    log_probs = F.log_softmax(logits, dim=-1)[0]
                    topk_vals, topk_ids = log_probs.topk(min(n_beams, len(log_probs)))
                    for v, idx in zip(topk_vals.tolist(), topk_ids.tolist()):
                        new_beams.append((prev + [idx], prev_s + v))
                new_beams.sort(key=lambda x: -x[1])
                beams = new_beams[:n_beams]
            all_prefixes.append([b[0] for b in beams])
            all_scores.append([b[1] for b in beams])
        return all_prefixes, all_scores


def prefix_seqs(sequences, item_codes, fd, max_items=15):
    """Build sequences of prefix-only tokens."""
    token_ids, stage_ids = [], []
    for seq in sequences:
        seq = seq[-max_items:]
        toks, stgs = [], []
        for item_id in seq:
            for s in range(fd):
                toks.append(item_codes[item_id, s])
                stgs.append(s)
        token_ids.append(toks)
        stage_ids.append(stgs)
    return token_ids, stage_ids


def pad(lists, max_len):
    B = len(lists)
    arr = np.zeros((B, max_len), dtype=np.int64)
    for i, L in enumerate(lists):
        n = min(len(L), max_len)
        arr[i, :n] = L[:n]
    return arr


def train_prefix_gen(
    model, token_ids, stage_ids, fd, epochs=50, lr=3e-4, batch_size=128, device="cpu"
):
    max_len = max(len(t) for t in token_ids)
    tok = torch.from_numpy(pad(token_ids, max_len)).long().to(device)
    stg = torch.from_numpy(pad(stage_ids, max_len)).long().to(device)
    model = model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(token_ids)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            bt, bs = tok[idx, :-1], stg[idx, :-1]
            tt, ts = tok[idx, 1:], stg[idx, 1:]
            out = model(bt, bs)
            loss = (
                sum(
                    F.cross_entropy(model.heads[s](out[ts == s]), tt[ts == s])
                    for s in range(fd)
                    if (ts == s).any()
                )
                / fd
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    model.eval()
    return model


class Ranker(nn.Module):
    """Two-tower ranker: user embedding × item projection."""

    def __init__(self, n_users, item_dim, emb_dim=32):
        super().__init__()
        self.item_proj = nn.Linear(item_dim, emb_dim)
        self.user_emb = nn.Embedding(n_users, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)

    def forward(self, user_ids, item_vecs):
        u = self.user_emb(user_ids)
        v = self.item_proj(item_vecs)
        return (u * v).sum(dim=-1)


def train_ranker(
    model,
    pairs,
    item_vecs,
    n_items,
    epochs=15,
    lr=0.005,
    batch_size=4096,
    n_neg=4,
    seed=42,
):
    rng = np.random.RandomState(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    vecs_t = torch.from_numpy(item_vecs).float()
    pairs_arr = np.array(pairs)
    model.train()
    for _ in range(epochs):
        rng.shuffle(pairs_arr)
        for start in range(0, len(pairs_arr), batch_size):
            batch = pairs_arr[start : start + batch_size]
            users = torch.from_numpy(batch[:, 0].astype(np.int64))
            pos_items = batch[:, 1].astype(np.int64)
            pos_vecs = vecs_t[pos_items]
            pos_scores = model(users, pos_vecs)
            neg_idx = rng.randint(0, n_items, size=(len(batch), n_neg))
            neg_vecs = vecs_t[neg_idx.reshape(-1)].reshape(len(batch), n_neg, -1)
            u_emb = model.user_emb(users).unsqueeze(1)
            neg_proj = model.item_proj(neg_vecs)
            neg_scores = (u_emb * neg_proj).sum(dim=-1)
            loss = -F.logsigmoid(pos_scores.unsqueeze(1) - neg_scores).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--n-beams", type=int, default=10)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--json-output", type=str, default="generative_prefix.json")
    args = p.parse_args()

    t0 = time.time()
    results = []
    m = 4
    fd = 2
    codes_list = [16, 16, 256, 512]
    prefix_codes = codes_list[:fd]

    embs_t0, embs_t1, seqs_t0, eval_t1, seen_t0, n_users, n_items = (
        load_movielens_shared_basis()
    )
    emb_dim = embs_t0.shape[1]
    log.info(f"{n_items} items, {len(seqs_t0)} T0 seqs, {len(eval_t1)} T1 eval")

    for seed in range(args.seed_offset, args.seed_offset + args.seeds):
        log.info(f"=== seed={seed} ===")

        rq_src = RQ(m, codes_list, emb_dim).fit(embs_t0, seed=seed)
        rq_strat = warm_retrain(rq_src, embs_t1, fd, seed=seed)
        rq_full = RQ(m, codes_list, emb_dim).fit(embs_t1, seed=seed + 500)

        codes_t0_src = rq_src.encode(embs_t0)

        # Train prefix-only generator on T0
        log.info("  training prefix generator...")
        tok_t0, stg_t0 = prefix_seqs(seqs_t0, codes_t0_src, fd)
        torch.manual_seed(seed)
        gen = PrefixGenerator(prefix_codes, d_model=128, n_heads=4, n_layers=3)
        gen = train_prefix_gen(
            gen, tok_t0, stg_t0, fd, epochs=args.epochs, device=args.device
        )

        # Train ranker on T0 RAW vectors (not decoded — avoids codebook bias)
        log.info("  training ranker on T0 raw vectors...")
        pairs_t0 = []
        for uid, items in seen_t0.items():
            for item_id in items:
                pairs_t0.append((uid, item_id))
        ranker = Ranker(n_users, emb_dim, emb_dim=32)
        train_ranker(ranker, pairs_t0, embs_t0, n_items, epochs=15, seed=seed)

        # Precompute user embeddings
        all_user_embs = ranker.user_emb.weight.detach().numpy()

        codes_t1_src = rq_src.encode(embs_t1)

        for sname, rq in [
            ("frozen", rq_src),
            ("stratified", rq_strat),
            ("full_retrain", rq_full),
        ]:
            codes_t1 = rq.encode(embs_t1)
            item_decoded = rq.decode_codes(codes_t1)
            # Project all items through ranker
            item_proj = (
                ranker.item_proj(torch.from_numpy(item_decoded).float())
                .detach()
                .numpy()
            )

            pfx_to_items = {}
            for i in range(n_items):
                pfx = tuple(codes_t1[i, :fd])
                pfx_to_items.setdefault(pfx, []).append(i)

            histories = [h for _, h, _ in eval_t1]
            targets = [t for _, _, t in eval_t1]
            uids = [u for u, _, _ in eval_t1]
            tok_eval, stg_eval = prefix_seqs(histories, codes_t1_src, fd)
            max_len = max(len(t) for t in tok_eval)
            tok_t = torch.from_numpy(pad(tok_eval, max_len)).long().to(args.device)
            stg_t = torch.from_numpy(pad(stg_eval, max_len)).long().to(args.device)

            hits, ndcg, total = 0, 0.0, 0
            batch_size = 64
            for start in range(0, len(eval_t1), batch_size):
                end = min(start + batch_size, len(eval_t1))
                prefixes, _ = gen.generate_prefixes(
                    tok_t[start:end], stg_t[start:end], n_beams=args.n_beams
                )

                for b_idx in range(end - start):
                    cands = set()
                    for pfx in prefixes[b_idx]:
                        key = tuple(pfx)
                        if key in pfx_to_items:
                            cands.update(pfx_to_items[key])
                    if not cands:
                        total += 1
                        continue

                    cands = np.array(sorted(cands))
                    uid = uids[start + b_idx]
                    u_emb = all_user_embs[uid]
                    # Score candidates with ranker (no oracle)
                    scores = item_proj[cands] @ u_emb
                    # Block seen items
                    seen = seen_t0.get(uid, set()) | set(histories[start + b_idx])
                    for j, item_id in enumerate(cands):
                        if item_id in seen:
                            scores[j] = -np.inf
                    top_idx = np.argsort(scores)[-10:][::-1]
                    ranking = cands[top_idx]

                    target = targets[start + b_idx]
                    if target in ranking:
                        hits += 1
                        rank = np.where(ranking == target)[0][0]
                        ndcg += 1.0 / math.log2(rank + 2)
                    total += 1

            hr = hits / max(total, 1)
            nd = ndcg / max(total, 1)
            log.info(
                f"  {sname:20s} HR@10={hr:.4f} NDCG={nd:.4f} MSE={rq.mse(embs_t1):.1f}"
            )
            results.append(
                {
                    "seed": seed,
                    "strategy": sname,
                    "hr@10": hr,
                    "ndcg@10": nd,
                    "mse": rq.mse(embs_t1),
                }
            )

        # Ranker-only baselines (no generator, score all items)
        for bname, rq in [
            ("ranker_frozen", rq_src),
            ("ranker_stratified", rq_strat),
            ("ranker_full", rq_full),
        ]:
            item_dec = rq.decode_codes(rq.encode(embs_t1))
            ip = ranker.item_proj(torch.from_numpy(item_dec).float()).detach().numpy()
            hits, ndcg, total = 0, 0.0, 0
            for uid, hist, target in eval_t1:
                u_emb = all_user_embs[uid]
                scores = ip @ u_emb
                seen = seen_t0.get(uid, set()) | set(hist)
                for i in seen:
                    if i < len(scores):
                        scores[i] = -np.inf
                top10 = np.argsort(scores)[-10:][::-1]
                if target in top10:
                    hits += 1
                    rank = np.where(top10 == target)[0][0]
                    ndcg += 1.0 / math.log2(rank + 2)
                total += 1
            hr = hits / max(total, 1)
            nd = ndcg / max(total, 1)
            log.info(
                f"  {bname:20s} HR@10={hr:.4f} NDCG={nd:.4f} MSE={rq.mse(embs_t1):.1f}"
            )
            results.append(
                {
                    "seed": seed,
                    "strategy": bname,
                    "hr@10": hr,
                    "ndcg@10": nd,
                    "mse": rq.mse(embs_t1),
                }
            )

        with open(args.json_output, "w") as f:
            json.dump(results, f)

    log.info(f"Done in {time.time() - t0:.0f}s. {len(results)} rows.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generative recommender CE on real MovieLens sequences.

Trains a causal transformer on real user interaction sequences where
each item is its 4-token RQ semantic ID. Measures per-stage cross-entropy
after temporal drift — the metric that directly tests whether the old
model still understands the token vocabulary.

Usage:
    python3 run_generative_movielens_ce.py --seeds 5
    python3 run_generative_movielens_ce.py --seeds 5 --device cuda
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


def load_and_split(data_dir):
    ratings = []
    with open(os.path.join(data_dir, "ratings.dat"), "r", encoding="latin-1") as f:
        for line in f:
            p = line.strip().split("::")
            ratings.append((int(p[0]), int(p[1]), float(p[2]), int(p[3])))

    items = sorted(set(r[1] for r in ratings))
    item_to_idx = {iid: i for i, iid in enumerate(items)}
    n_items = len(items)
    n_users = max(r[0] for r in ratings) + 1

    timestamps = [r[3] for r in ratings]
    median_ts = int(np.median(timestamps))

    R_t0 = np.zeros((n_users, n_items), dtype=np.float32)
    R_t1 = np.zeros((n_users, n_items), dtype=np.float32)
    seqs_t0, seqs_t1 = {}, {}

    for uid, iid, rating, ts in ratings:
        idx = item_to_idx[iid]
        if ts <= median_ts:
            R_t0[uid, idx] = rating
            seqs_t0.setdefault(uid, []).append((ts, idx))
        else:
            R_t1[uid, idx] = rating
            seqs_t1.setdefault(uid, []).append((ts, idx))

    U0, S0, Vt0 = np.linalg.svd(R_t0, full_matrices=False)
    embs_t0 = (Vt0[:64].T * S0[:64]).astype(np.float32)
    norms = np.linalg.norm(embs_t0, axis=1, keepdims=True) + 1e-8
    embs_t0 = embs_t0 / norms

    U1, S1, Vt1 = np.linalg.svd(R_t1, full_matrices=False)
    embs_t1 = (Vt1[:64].T * S1[:64]).astype(np.float32)
    norms = np.linalg.norm(embs_t1, axis=1, keepdims=True) + 1e-8
    embs_t1 = embs_t1 / norms

    def make_seqs(raw, min_len=5, max_len=20):
        out = []
        for uid, items in raw.items():
            items.sort()
            ids = [i for _, i in items]
            if len(ids) >= min_len:
                out.append(ids[-max_len:])
        return out

    return embs_t0, embs_t1, make_seqs(seqs_t0), make_seqs(seqs_t1), n_items


def _kmeans(X, k, n_iter=20, rng=None, init=None):
    if rng is None: rng = np.random.RandomState(42)
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
            if m.sum() > 0: centroids[j] = X[m].mean(axis=0)
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
        r = X.copy(); self.cb = []
        for i in range(self.m):
            c = _kmeans(r, self.K[i], n_iter=n_iter, rng=rng)
            self.cb.append(c); a = _assign(r, c); r = r - c[a]
        return self

    def encode(self, X):
        r = X.copy(); codes = []
        for i in range(self.m):
            a = _assign(r, self.cb[i]); codes.append(a); r = r - self.cb[i][a]
        return np.stack(codes, axis=1)

    def mse(self, X):
        r = X.copy()
        for c in self.cb: a = _assign(r, c); r = r - c[a]
        return float(np.mean(np.sum(r ** 2, axis=1)))


def warm_retrain(rq, X, fd, n_iter=20, seed=42):
    rq2 = RQ(rq.m, rq.K, rq.dim); rq2.cb = [c.copy() for c in rq.cb]
    r = X.copy()
    for i in range(fd): a = _assign(r, rq2.cb[i]); r = r - rq2.cb[i][a]
    rng = np.random.RandomState(seed)
    for i in range(fd, rq.m):
        c = _kmeans(r, rq.K[i], n_iter=n_iter, rng=rng, init=rq2.cb[i])
        rq2.cb[i] = c; a = _assign(r, c); r = r - c[a]
    return rq2


class TokenTransformer(nn.Module):
    def __init__(self, vocab_sizes, d_model=128, n_heads=4, n_layers=3,
                 max_tokens=80, dropout=0.1):
        super().__init__()
        self.m = len(vocab_sizes)
        self.vocab_sizes = vocab_sizes
        self.d_model = d_model
        self.tok_embs = nn.ModuleList([nn.Embedding(vs, d_model) for vs in vocab_sizes])
        self.pos_emb = nn.Embedding(max_tokens, d_model)
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
        pos = torch.arange(T, device=token_ids.device).unsqueeze(0)
        embs = embs + self.pos_emb(pos) + self.stage_emb(stage_ids)
        causal = torch.triu(torch.ones(T, T, device=token_ids.device), diagonal=1).bool()
        return self.transformer(embs, mask=causal)


def seqs_to_tokens(sequences, item_codes, m, max_items=20):
    token_ids, stage_ids = [], []
    for seq in sequences:
        seq = seq[-max_items:]
        toks, stgs = [], []
        for item_id in seq:
            for s in range(m):
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


def train_model(model, token_ids, stage_ids, m,
                epochs=30, lr=3e-4, batch_size=128, device="cpu"):
    max_len = max(len(t) for t in token_ids)
    tok = torch.from_numpy(pad(token_ids, max_len)).long().to(device)
    stg = torch.from_numpy(pad(stage_ids, max_len)).long().to(device)
    model = model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(token_ids)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            bt, bs = tok[idx, :-1], stg[idx, :-1]
            tt, ts = tok[idx, 1:], stg[idx, 1:]
            out = model(bt, bs)
            loss = sum(
                F.cross_entropy(model.heads[s](out[ts == s]), tt[ts == s])
                for s in range(m) if (ts == s).any()
            ) / m
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    model.eval()
    return model


@torch.no_grad()
def eval_ce(model, token_ids, stage_ids, m, device="cpu"):
    max_len = max(len(t) for t in token_ids)
    tok = torch.from_numpy(pad(token_ids, max_len)).long().to(device)
    stg = torch.from_numpy(pad(stage_ids, max_len)).long().to(device)
    inp_t, inp_s = tok[:, :-1], stg[:, :-1]
    tgt_t, tgt_s = tok[:, 1:], stg[:, 1:]
    model.eval()
    out = model(inp_t, inp_s)
    ce_per_stage = []
    acc_per_stage = []
    for s in range(m):
        mask = tgt_s == s
        if mask.any():
            logits = model.heads[s](out[mask])
            targets = tgt_t[mask]
            ce_per_stage.append(F.cross_entropy(logits, targets).item())
            acc_per_stage.append((logits.argmax(-1) == targets).float().mean().item())
        else:
            ce_per_stage.append(float('nan'))
            acc_per_stage.append(float('nan'))
    return ce_per_stage, acc_per_stage


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--json-output", type=str, default="generative_movielens_ce.json")
    args = p.parse_args()

    t0 = time.time()
    results = []
    m = 4; fd = 2
    codes_list = [16, 16, 256, 256]

    data_dir = download_movielens()
    embs_t0, embs_t1, seqs_t0, seqs_t1, n_items = load_and_split(data_dir)
    log.info(f"{n_items} items, {len(seqs_t0)} T0 seqs, {len(seqs_t1)} T1 seqs")

    for seed in range(args.seeds):
        log.info(f"=== seed={seed} ===")

        rq_src = RQ(m, codes_list, 64).fit(embs_t0, seed=seed)
        rq_full = RQ(m, codes_list, 64).fit(embs_t1, seed=seed + 500)
        rq_strat = warm_retrain(rq_src, embs_t1, fd, seed=seed)

        codes_t0_src = rq_src.encode(embs_t0)
        codes_t1_src = rq_src.encode(embs_t1)
        codes_t1_full = rq_full.encode(embs_t1)
        codes_t1_strat = rq_strat.encode(embs_t1)

        tok_t0, stg_t0 = seqs_to_tokens(seqs_t0, codes_t0_src, m)
        log.info(f"  {len(tok_t0)} training seqs, training model...")

        torch.manual_seed(seed)
        model = TokenTransformer(codes_list, d_model=128, n_heads=4, n_layers=3)
        model = train_model(model, tok_t0, stg_t0, m, epochs=args.epochs,
                            device=args.device)

        ce_base, acc_base = eval_ce(model, tok_t0, stg_t0, m, device=args.device)
        log.info(f"  baseline CE={[f'{c:.2f}' for c in ce_base]}")

        for sname, codes in [("frozen", codes_t1_src),
                             ("stratified", codes_t1_strat),
                             ("full_old_model", codes_t1_full)]:
            tok_eval, stg_eval = seqs_to_tokens(seqs_t1, codes, m)
            ce, acc = eval_ce(model, tok_eval, stg_eval, m, device=args.device)
            log.info(f"  {sname}: CE={[f'{c:.2f}' for c in ce]} acc={[f'{a:.3f}' for a in acc]}")
            results.append({"seed": seed, "strategy": sname,
                            "ce_per_stage": ce, "acc_per_stage": acc,
                            "mean_ce": sum(ce) / m})

        results.append({"seed": seed, "strategy": "baseline",
                        "ce_per_stage": ce_base, "acc_per_stage": acc_base,
                        "mean_ce": sum(ce_base) / m})

        log.info("  training retrained model on full-retrain codes...")
        torch.manual_seed(seed + 1000)
        model_new = TokenTransformer(codes_list, d_model=128, n_heads=4, n_layers=3)
        tok_f, stg_f = seqs_to_tokens(seqs_t1, codes_t1_full, m)
        model_new = train_model(model_new, tok_f, stg_f, m, epochs=args.epochs,
                                device=args.device)
        ce, acc = eval_ce(model_new, tok_f, stg_f, m, device=args.device)
        log.info(f"  full_new_model: CE={[f'{c:.2f}' for c in ce]}")
        results.append({"seed": seed, "strategy": "full_new_model",
                        "ce_per_stage": ce, "acc_per_stage": acc,
                        "mean_ce": sum(ce) / m})

        with open(args.json_output, "w") as f:
            json.dump(results, f)

    log.info(f"Done in {time.time() - t0:.0f}s. {len(results)} rows.")


if __name__ == "__main__":
    main()

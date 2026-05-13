#!/usr/bin/env python3
"""Multi-period generative recommender with stratified RQ semantic IDs.

Trains a causal transformer on user sequences across multiple periods,
where prefix tokens are stable and suffix tokens vary by period. Tests
whether the model generalizes to a new unseen period.

The key insight: a production generative recommender trained on stratified
IDs learns that prefix vocabulary is stable and suffix varies by period.
At a new period, prefix predictions transfer; suffix predictions adapt
because the model already knows suffixes are period-dependent.

Usage:
    python3 run_generative_multiperiod.py --seeds 3
    python3 run_generative_multiperiod.py --seeds 3 --device cuda
"""
from __future__ import annotations
import argparse, json, logging, math, os, time, urllib.request, zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def download_movielens(data_dir="/tmp/ml-1m"):
    if os.path.exists(os.path.join(data_dir, "ratings.dat")):
        return data_dir
    zip_path = "/tmp/ml-1m.zip"
    if not os.path.exists(zip_path):
        log.info("Downloading MovieLens-1M...")
        urllib.request.urlretrieve(
            "https://files.grouplens.org/datasets/movielens/ml-1m.zip", zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("/tmp")
    return data_dir


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

    def decode(self, codes):
        out = np.zeros((len(codes), self.dim), dtype=np.float32)
        for i in range(self.m):
            out += self.cb[i][codes[:, i]]
        return out

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


class PeriodAwareTransformer(nn.Module):
    """Causal transformer that conditions on period for suffix prediction."""

    def __init__(self, vocab_sizes, n_periods, d_model=128, n_heads=4,
                 n_layers=3, max_tokens=80, dropout=0.1):
        super().__init__()
        self.m = len(vocab_sizes)
        self.vocab_sizes = vocab_sizes
        self.d_model = d_model
        self.n_periods = n_periods
        self.tok_embs = nn.ModuleList([nn.Embedding(vs, d_model) for vs in vocab_sizes])
        self.pos_emb = nn.Embedding(max_tokens, d_model)
        self.stage_emb = nn.Embedding(self.m, d_model)
        self.period_emb = nn.Embedding(n_periods + 1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.heads = nn.ModuleList([nn.Linear(d_model, vs) for vs in vocab_sizes])

    def forward(self, token_ids, stage_ids, period_ids):
        B, T = token_ids.shape
        embs = torch.zeros(B, T, self.d_model, device=token_ids.device)
        for s in range(self.m):
            mask = stage_ids == s
            if mask.any():
                embs[mask] = self.tok_embs[s](token_ids[mask])
        pos = torch.arange(T, device=token_ids.device).unsqueeze(0)
        embs = embs + self.pos_emb(pos) + self.stage_emb(stage_ids) + self.period_emb(period_ids)
        causal = torch.triu(torch.ones(T, T, device=token_ids.device), diagonal=1).bool()
        return self.transformer(embs, mask=causal)


def build_period_data(ratings, n_periods=6, emb_dim=64, min_seq_len=5, max_seq_len=15):
    """Split ratings into temporal periods, compute embeddings and RQ codes per period."""
    items = sorted(set(r[1] for r in ratings))
    item_to_idx = {iid: i for i, iid in enumerate(items)}
    n_items = len(items)
    n_users = max(r[0] for r in ratings) + 1

    timestamps = sorted(set(r[3] for r in ratings))
    period_boundaries = np.linspace(0, len(timestamps) - 1, n_periods + 1, dtype=int)
    ts_boundaries = [timestamps[i] for i in period_boundaries]

    def get_period(ts):
        for p in range(n_periods):
            if ts <= ts_boundaries[p + 1]:
                return p
        return n_periods - 1

    period_ratings = [[] for _ in range(n_periods)]
    for uid, iid, rating, ts in ratings:
        p = get_period(ts)
        period_ratings[p].append((uid, item_to_idx[iid], rating, ts))

    log.info(f"  {n_items} items, {n_periods} periods: {[len(pr) for pr in period_ratings]}")
    return period_ratings, n_items, n_users, item_to_idx


def compute_embeddings(period_ratings, n_users, n_items, emb_dim=64):
    R = np.zeros((n_users, n_items), dtype=np.float32)
    for uid, idx, rating, ts in period_ratings:
        R[uid, idx] = rating
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    embs = (Vt[:emb_dim].T * S[:emb_dim]).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    return embs / norms


def build_sequences(period_ratings, min_len=5, max_len=15):
    user_items = {}
    for uid, idx, rating, ts in period_ratings:
        user_items.setdefault(uid, []).append((ts, idx))
    seqs = []
    for uid, items in user_items.items():
        items.sort()
        ids = [i for _, i in items]
        if len(ids) >= min_len:
            seqs.append(ids[-max_len:])
    return seqs


def seqs_to_tokens(sequences, item_codes, m, period_id, max_items=15):
    token_ids, stage_ids, period_ids = [], [], []
    for seq in sequences:
        seq = seq[-max_items:]
        toks, stgs, pids = [], [], []
        for item_id in seq:
            for s in range(m):
                toks.append(item_codes[item_id, s])
                stgs.append(s)
                pids.append(period_id)
        token_ids.append(toks)
        stage_ids.append(stgs)
        period_ids.append(pids)
    return token_ids, stage_ids, period_ids


def pad(lists, max_len, fill=0):
    B = len(lists)
    arr = np.full((B, max_len), fill, dtype=np.int64)
    for i, L in enumerate(lists):
        n = min(len(L), max_len)
        arr[i, :n] = L[:n]
    return arr


def train_model(model, all_token_ids, all_stage_ids, all_period_ids, m,
                epochs=30, lr=3e-4, batch_size=128, device="cpu"):
    max_len = max(len(t) for t in all_token_ids)
    tok = torch.from_numpy(pad(all_token_ids, max_len)).long().to(device)
    stg = torch.from_numpy(pad(all_stage_ids, max_len)).long().to(device)
    per = torch.from_numpy(pad(all_period_ids, max_len)).long().to(device)
    model = model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(all_token_ids)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            bt, bs, bp = tok[idx, :-1], stg[idx, :-1], per[idx, :-1]
            tt, ts = tok[idx, 1:], stg[idx, 1:]
            out = model(bt, bs, bp)
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
def eval_model(model, token_ids, stage_ids, period_ids, m, device="cpu", top_k=10):
    max_len = max(len(t) for t in token_ids)
    tok = torch.from_numpy(pad(token_ids, max_len)).long().to(device)
    stg = torch.from_numpy(pad(stage_ids, max_len)).long().to(device)
    per = torch.from_numpy(pad(period_ids, max_len)).long().to(device)
    model.eval()

    inp_t, inp_s, inp_p = tok[:, :-m], stg[:, :-m], per[:, :-m]
    tgt_t, tgt_s = tok[:, -m:], stg[:, -m:]
    if inp_t.shape[1] == 0:
        return {"ce_per_stage": [float('nan')] * m, "acc_per_stage": [float('nan')] * m}

    out = model(tok[:, :-1], stg[:, :-1], per[:, :-1])
    tgt_tok = tok[:, 1:]
    tgt_stg = stg[:, 1:]

    ce_per_stage, acc_per_stage = [], []
    for s in range(m):
        mask = tgt_stg == s
        if mask.any():
            logits = model.heads[s](out[mask])
            targets = tgt_tok[mask]
            ce_per_stage.append(F.cross_entropy(logits, targets).item())
            acc_per_stage.append((logits.argmax(-1) == targets).float().mean().item())
        else:
            ce_per_stage.append(float('nan'))
            acc_per_stage.append(float('nan'))

    # Item-level: decode last item's tokens via greedy generation
    last_hidden = out[:, -1, :]
    item_codes_pred = torch.zeros(tok.shape[0], m, dtype=torch.long, device=device)
    item_codes_true = tok[:, -m:]
    for s in range(m):
        pos = -m + s
        if pos < 0:
            logits = model.heads[s](out[:, pos - 1, :])
        else:
            logits = model.heads[s](out[:, -1, :])
        item_codes_pred[:, s] = logits.argmax(-1)

    prefix_match = (item_codes_pred[:, :2] == item_codes_true[:, :2]).all(dim=1).float().mean().item()
    full_match = (item_codes_pred == item_codes_true).all(dim=1).float().mean().item()

    return {
        "ce_per_stage": ce_per_stage,
        "acc_per_stage": acc_per_stage,
        "prefix_match": prefix_match,
        "full_match": full_match,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n-periods", type=int, default=6)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--json-output", type=str, default="generative_multiperiod.json")
    args = p.parse_args()

    t0 = time.time()
    results = []
    m = 4; fd = 2
    codes_list = [16, 16, 256, 256]
    emb_dim = 64
    n_train_periods = args.n_periods - 1
    test_period = args.n_periods - 1

    data_dir = download_movielens()
    ratings = []
    with open(os.path.join(data_dir, "ratings.dat"), "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            ratings.append((int(parts[0]), int(parts[1]), float(parts[2]), int(parts[3])))
    log.info(f"Loaded {len(ratings)} ratings")

    period_ratings, n_items, n_users, item_to_idx = build_period_data(
        ratings, n_periods=args.n_periods, emb_dim=emb_dim
    )

    for seed in range(args.seeds):
        log.info(f"=== seed={seed} ===")

        # Compute embeddings per period
        period_embs = []
        for pid in range(args.n_periods):
            cumulative = []
            for pp in range(pid + 1):
                cumulative.extend(period_ratings[pp])
            embs = compute_embeddings(cumulative, n_users, n_items, emb_dim)
            period_embs.append(embs)

        # Train source RQ on period 0
        log.info("  fitting source RQ on period 0...")
        rq_source = RQ(m, codes_list, emb_dim).fit(period_embs[0], seed=seed)

        # Stratified retrain for each training period
        period_codes_strat = []
        rq_current = rq_source
        for pid in range(n_train_periods):
            if pid == 0:
                codes = rq_source.encode(period_embs[pid])
            else:
                rq_current = warm_retrain(rq_current, period_embs[pid], fd, seed=seed + pid)
                codes = rq_current.encode(period_embs[pid])
            period_codes_strat.append(codes)

        # Build training sequences from all training periods
        all_tok, all_stg, all_per = [], [], []
        for pid in range(n_train_periods):
            seqs = build_sequences(period_ratings[pid])
            tok, stg, per = seqs_to_tokens(seqs, period_codes_strat[pid], m, pid)
            all_tok.extend(tok)
            all_stg.extend(stg)
            all_per.extend(per)
        log.info(f"  {len(all_tok)} training sequences across {n_train_periods} periods")

        # Train period-aware model
        log.info("  training period-aware model...")
        torch.manual_seed(seed)
        model = PeriodAwareTransformer(
            codes_list, n_periods=args.n_periods,
            d_model=128, n_heads=4, n_layers=3
        )
        model = train_model(model, all_tok, all_stg, all_per, m,
                            epochs=args.epochs, device=args.device)

        # Evaluate on test period with different strategies
        test_embs = period_embs[test_period]
        test_seqs = build_sequences(period_ratings[test_period])
        log.info(f"  {len(test_seqs)} test sequences in period {test_period}")

        # Strategy 1: stratified (warm retrain suffix from last training period)
        rq_test_strat = warm_retrain(rq_current, test_embs, fd, seed=seed + test_period)
        codes_strat = rq_test_strat.encode(test_embs)
        tok_s, stg_s, per_s = seqs_to_tokens(test_seqs, codes_strat, m, test_period)
        metrics = eval_model(model, tok_s, stg_s, per_s, m, device=args.device)
        log.info(f"  stratified: CE={[f'{c:.2f}' for c in metrics['ce_per_stage']]} "
                 f"pfx={metrics['prefix_match']:.3f} full={metrics['full_match']:.3f}")
        results.append({"seed": seed, "strategy": "stratified", **metrics,
                        "mse": rq_test_strat.mse(test_embs)})

        # Strategy 2: frozen (source codebook, no adaptation)
        codes_frz = rq_source.encode(test_embs)
        tok_f, stg_f, per_f = seqs_to_tokens(test_seqs, codes_frz, m, test_period)
        metrics = eval_model(model, tok_f, stg_f, per_f, m, device=args.device)
        log.info(f"  frozen: CE={[f'{c:.2f}' for c in metrics['ce_per_stage']]} "
                 f"pfx={metrics['prefix_match']:.3f} full={metrics['full_match']:.3f}")
        results.append({"seed": seed, "strategy": "frozen", **metrics,
                        "mse": rq_source.mse(test_embs)})

        # Strategy 3: full retrain (new codebook, old model)
        rq_full = RQ(m, codes_list, emb_dim).fit(test_embs, seed=seed + 500)
        codes_full = rq_full.encode(test_embs)
        tok_fu, stg_fu, per_fu = seqs_to_tokens(test_seqs, codes_full, m, test_period)
        metrics = eval_model(model, tok_fu, stg_fu, per_fu, m, device=args.device)
        log.info(f"  full_old_model: CE={[f'{c:.2f}' for c in metrics['ce_per_stage']]} "
                 f"pfx={metrics['prefix_match']:.3f} full={metrics['full_match']:.3f}")
        results.append({"seed": seed, "strategy": "full_old_model", **metrics,
                        "mse": rq_full.mse(test_embs)})

        # Strategy 4: full retrain + new model
        log.info("  training full-retrain model...")
        all_tok_full, all_stg_full, all_per_full = [], [], []
        for pid in range(n_train_periods):
            rq_p = RQ(m, codes_list, emb_dim).fit(period_embs[pid], seed=seed + 500 + pid)
            codes_p = rq_p.encode(period_embs[pid])
            seqs = build_sequences(period_ratings[pid])
            tok, stg, per = seqs_to_tokens(seqs, codes_p, m, pid)
            all_tok_full.extend(tok)
            all_stg_full.extend(stg)
            all_per_full.extend(per)

        torch.manual_seed(seed + 1000)
        model_full = PeriodAwareTransformer(
            codes_list, n_periods=args.n_periods,
            d_model=128, n_heads=4, n_layers=3
        )
        model_full = train_model(model_full, all_tok_full, all_stg_full, all_per_full, m,
                                 epochs=args.epochs, device=args.device)
        metrics = eval_model(model_full, tok_fu, stg_fu, per_fu, m, device=args.device)
        log.info(f"  full_new_model: CE={[f'{c:.2f}' for c in metrics['ce_per_stage']]} "
                 f"pfx={metrics['prefix_match']:.3f} full={metrics['full_match']:.3f}")
        results.append({"seed": seed, "strategy": "full_new_model", **metrics,
                        "mse": rq_full.mse(test_embs)})

        with open(args.json_output, "w") as f:
            json.dump(results, f)

    log.info(f"Done in {time.time() - t0:.0f}s. {len(results)} rows.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Full-token generator → ANN candidate recall on MovieLens.

Business test: can an old generator still produce useful ANN queries
after the codebook changes? Measures candidate recall at multiple
budgets — the gating metric for any downstream ranker.

Pipeline:
1. Train full 4-token generator on T0 user sequences
2. Generator produces beam of next-item codes from T1 history
3. Decode beam → query vectors (old codebook, generator's language)
4. ANN: find K nearest items (decoded with STRATEGY codebook)
5. Evaluate: is the actual next item in the candidate set?

Metrics:
- Recall@K for K in {10, 20, 50, 100, 200}
- MRR (reciprocal rank of true next item)
- Distance diagnostics: min/mean dist of top-K to true next item

Usage:
    python3 run_generative_recall.py --seeds 5
    python3 run_generative_recall.py --seeds 5 --device cuda
"""
from __future__ import annotations
import argparse, json, logging, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.spatial.distance import cdist

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def load_movielens_shared_basis(emb_dim=64, min_seq_len=5, max_seq_len=15):
    data_dir = "/tmp/ml-1m"
    if not os.path.exists(os.path.join(data_dir, "ratings.dat")):
        import urllib.request, zipfile
        urllib.request.urlretrieve(
            "https://files.grouplens.org/datasets/movielens/ml-1m.zip", "/tmp/ml-1m.zip")
        with zipfile.ZipFile("/tmp/ml-1m.zip") as z: z.extractall("/tmp")

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
    seqs_t0_raw, seqs_t1_raw = {}, {}
    seen_t0 = {}

    for uid, iid, rating, ts in ratings:
        u, i = user_map[uid], item_map[iid]
        if ts < median_ts:
            rows_old.append(u); cols_old.append(i); vals_old.append(rating)
            seqs_t0_raw.setdefault(u, []).append((ts, i))
            seen_t0.setdefault(u, set()).add(i)
        else:
            seqs_t1_raw.setdefault(u, []).append((ts, i))

    M_old = csr_matrix((vals_old, (rows_old, cols_old)),
                       shape=(n_users, n_items), dtype=np.float32)
    rows_new, cols_new, vals_new = [], [], []
    for uid, iid, rating, ts in ratings:
        if ts >= median_ts:
            rows_new.append(user_map[uid]); cols_new.append(item_map[iid]); vals_new.append(rating)
    M_new = csr_matrix((vals_new, (rows_new, cols_new)),
                       shape=(n_users, n_items), dtype=np.float32)

    k = min(emb_dim, min(M_old.shape) - 1)
    U_old, S_old, Vt_old = svds(M_old, k=k)
    embs_t0 = (Vt_old.T * S_old).astype(np.float32)
    embs_t1 = (M_new.T @ (U_old * (1.0 / (S_old + 1e-8)))).astype(np.float32)
    embs_t1 = (embs_t1 * S_old).astype(np.float32)

    def make_seqs(raw):
        out = []
        for items in raw.values():
            items.sort()
            ids = [i for _, i in items]
            if len(ids) >= min_seq_len: out.append(ids[-max_seq_len:])
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

    return embs_t0, embs_t1, make_seqs(seqs_t0_raw), make_eval(seqs_t1_raw), \
           seen_t0, n_users, n_items


# === RQ ===
def _kmeans(X, k, n_iter=20, rng=None, init=None):
    if rng is None: rng = np.random.RandomState(42)
    n = len(X)
    if init is not None: centroids = init.copy()
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
        rng = np.random.RandomState(seed); r = X.copy(); self.cb = []
        for i in range(self.m):
            c = _kmeans(r, self.K[i], n_iter=n_iter, rng=rng)
            self.cb.append(c); a = _assign(r, c); r = r - c[a]
        return self
    def encode(self, X):
        r = X.copy(); codes = []
        for i in range(self.m):
            a = _assign(r, self.cb[i]); codes.append(a); r = r - self.cb[i][a]
        return np.stack(codes, axis=1)
    def decode_codes(self, codes):
        out = np.zeros((len(codes), self.dim), dtype=np.float32)
        for i in range(self.m): out += self.cb[i][codes[:, i]]
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

def decode_with_codebook(codes, rq):
    out = np.zeros((len(codes), rq.dim), dtype=np.float32)
    for i in range(rq.m):
        c = np.clip(codes[:, i], 0, rq.K[i] - 1)
        out += rq.cb[i][c]
    return out


# === Full-token generator ===
class SeqGenerator(nn.Module):
    def __init__(self, vocab_sizes, d_model=128, n_heads=4, n_layers=3,
                 max_tokens=80, dropout=0.1):
        super().__init__()
        self.m = len(vocab_sizes)
        self.vocab_sizes = vocab_sizes; self.d_model = d_model
        self.tok_embs = nn.ModuleList([nn.Embedding(vs, d_model) for vs in vocab_sizes])
        self.pos_emb = nn.Embedding(max_tokens, d_model)
        self.stage_emb = nn.Embedding(self.m, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.heads = nn.ModuleList([nn.Linear(d_model, vs) for vs in vocab_sizes])

    def forward(self, token_ids, stage_ids):
        B, T = token_ids.shape
        embs = torch.zeros(B, T, self.d_model, device=token_ids.device)
        for s in range(self.m):
            mask = stage_ids == s
            if mask.any(): embs[mask] = self.tok_embs[s](token_ids[mask])
        pos = torch.arange(T, device=token_ids.device).unsqueeze(0)
        embs = embs + self.pos_emb(pos) + self.stage_emb(stage_ids)
        causal = torch.triu(torch.ones(T, T, device=token_ids.device), diagonal=1).bool()
        return self.transformer(embs, mask=causal)

    @torch.no_grad()
    def generate_next_codes(self, token_ids, stage_ids, n_beams=10):
        B = token_ids.shape[0]; device = token_ids.device
        all_codes, all_scores = [], []
        for b in range(B):
            ctx_tok = token_ids[b:b+1]; ctx_stg = stage_ids[b:b+1]
            beams = [([], 0.0)]
            for s in range(self.m):
                new_beams = []
                for prev, prev_s in beams:
                    if prev:
                        ext_tok = torch.cat([ctx_tok, torch.tensor([prev], dtype=torch.long, device=device)], dim=1)
                        ext_stg = torch.cat([ctx_stg, torch.tensor([list(range(len(prev)))], dtype=torch.long, device=device)], dim=1)
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
            all_codes.append(np.array([b_[0] for b_ in beams]))
            all_scores.append(np.array([b_[1] for b_ in beams]))
        return all_codes, all_scores


def full_seqs(sequences, item_codes, m, max_items=15):
    token_ids, stage_ids = [], []
    for seq in sequences:
        seq = seq[-max_items:]
        toks, stgs = [], []
        for item_id in seq:
            for s in range(m):
                toks.append(item_codes[item_id, s])
                stgs.append(s)
        token_ids.append(toks); stage_ids.append(stgs)
    return token_ids, stage_ids

def pad(lists, max_len):
    B = len(lists)
    arr = np.zeros((B, max_len), dtype=np.int64)
    for i, L in enumerate(lists):
        n = min(len(L), max_len); arr[i, :n] = L[:n]
    return arr

def train_model(model, token_ids, stage_ids, m, epochs=30, lr=3e-4,
                batch_size=128, device="cpu"):
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
            idx = perm[i:i + batch_size]
            bt, bs = tok[idx, :-1], stg[idx, :-1]
            tt, ts = tok[idx, 1:], stg[idx, 1:]
            out = model(bt, bs)
            loss = sum(F.cross_entropy(model.heads[s](out[ts == s]), tt[ts == s])
                       for s in range(m) if (ts == s).any()) / m
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
    model.eval(); return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--n-beams", type=int, default=10)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-eval", type=int, default=2000)
    p.add_argument("--json-output", type=str, default="generative_recall.json")
    args = p.parse_args()

    t0 = time.time(); results = []
    m = 4; fd = 2; codes_list = [16, 16, 256, 512]
    Ks = [10, 20, 50, 100, 200]

    embs_t0, embs_t1, seqs_t0, eval_t1, seen_t0, n_users, n_items = \
        load_movielens_shared_basis()
    emb_dim = embs_t0.shape[1]
    log.info(f"{n_items} items, {len(seqs_t0)} T0 seqs, {len(eval_t1)} T1 eval")

    for seed in range(args.seeds):
        log.info(f"=== seed={seed} ===")

        rq_src = RQ(m, codes_list, emb_dim).fit(embs_t0, seed=seed)
        rq_strat = warm_retrain(rq_src, embs_t1, fd, seed=seed)
        rq_full = RQ(m, codes_list, emb_dim).fit(embs_t1, seed=seed + 500)

        codes_t0_src = rq_src.encode(embs_t0)
        codes_t1_src = rq_src.encode(embs_t1)

        log.info("  training full-token generator...")
        tok_t0, stg_t0 = full_seqs(seqs_t0, codes_t0_src, m)
        torch.manual_seed(seed)
        gen = SeqGenerator(codes_list, d_model=128, n_heads=4, n_layers=3)
        gen = train_model(gen, tok_t0, stg_t0, m,
                          epochs=args.epochs, device=args.device)

        for sname, rq in [("frozen", rq_src), ("stratified", rq_strat),
                          ("full_retrain", rq_full)]:
            item_decoded = decode_with_codebook(rq.encode(embs_t1), rq)

            recalls = {k: 0 for k in Ks}
            ranks = []
            min_dists, mean_dists = [], []
            total = 0

            histories = [h for _, h, _ in eval_t1[:args.n_eval]]
            targets = [t for _, _, t in eval_t1[:args.n_eval]]
            tok_eval, stg_eval = full_seqs(histories, codes_t1_src, m)
            max_len = max(len(t) for t in tok_eval)
            tok_t = torch.from_numpy(pad(tok_eval, max_len)).long().to(args.device)
            stg_t = torch.from_numpy(pad(stg_eval, max_len)).long().to(args.device)

            for start in range(0, len(histories), 32):
                end = min(start + 32, len(histories))
                beam_codes, beam_scores = gen.generate_next_codes(
                    tok_t[start:end], stg_t[start:end], n_beams=args.n_beams)

                for b_idx in range(end - start):
                    # Decode generated codes with OLD codebook
                    query_vecs = decode_with_codebook(beam_codes[b_idx], rq_src)
                    # ANN: distances from each beam to each item
                    dists = cdist(query_vecs, item_decoded, metric="sqeuclidean")
                    best_per_item = dists.min(axis=0)
                    ranking = np.argsort(best_per_item)

                    target = targets[start + b_idx]
                    target_raw = embs_t1[target]

                    # Recall@K
                    for k in Ks:
                        if target in ranking[:k]:
                            recalls[k] += 1

                    # Rank
                    rank_pos = np.where(ranking == target)[0]
                    if len(rank_pos) > 0:
                        ranks.append(rank_pos[0] + 1)
                    else:
                        ranks.append(n_items)

                    # Distance diagnostics for top-10
                    top10 = ranking[:10]
                    top10_raw = embs_t1[top10]
                    d_to_target = np.sqrt(np.sum((top10_raw - target_raw) ** 2, axis=1))
                    min_dists.append(d_to_target.min())
                    mean_dists.append(d_to_target.mean())
                    total += 1

            row = {
                "seed": seed, "strategy": sname,
                "mse": rq.mse(embs_t1),
            }
            for k in Ks:
                row[f"recall@{k}"] = recalls[k] / max(total, 1)
            row["mrr"] = float(np.mean(1.0 / np.array(ranks)))
            row["median_rank"] = float(np.median(ranks))
            row["min_dist_mean"] = float(np.mean(min_dists))
            row["min_dist_std"] = float(np.std(min_dists))
            row["mean_dist_mean"] = float(np.mean(mean_dists))

            log.info(f"  {sname:15s} R@10={row['recall@10']:.4f} R@50={row['recall@50']:.4f} "
                     f"R@200={row['recall@200']:.4f} MRR={row['mrr']:.4f} "
                     f"min_d={row['min_dist_mean']:.3f} MSE={row['mse']:.1f}")
            results.append(row)

        with open(args.json_output, "w") as f:
            json.dump(results, f)

    log.info(f"Done in {time.time() - t0:.0f}s. {len(results)} rows.")


if __name__ == "__main__":
    main()

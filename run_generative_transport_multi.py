#!/usr/bin/env python3
"""Generate→Transport→Resolve on multiple recommendation datasets.

Supports: ml-1m, ml-25m, amazon-electronics, amazon-books, yelp.
Each dataset: train generator on T0 user sequences with RQ semantic IDs,
generate old codes → decode with current codebook → ANN → HR@10/NDCG@10.

Usage:
    python3 run_generative_transport_multi.py --dataset ml-25m --seeds 3 --device cuda
    python3 run_generative_transport_multi.py --dataset amazon-electronics --seeds 3 --device cuda
"""
from __future__ import annotations
import argparse, json, logging, math, os, time, urllib.request, zipfile, gzip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATASET_URLS = {
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "ml-25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
}


def load_movielens(name, data_dir=None):
    if data_dir is None:
        data_dir = f"/tmp/{name}"
    if name == "ml-1m":
        ratings_file = os.path.join(data_dir, "ratings.dat")
        if not os.path.exists(ratings_file):
            zip_path = f"/tmp/{name}.zip"
            if not os.path.exists(zip_path):
                log.info(f"Downloading {name}...")
                urllib.request.urlretrieve(DATASET_URLS[name], zip_path)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall("/tmp")
        ratings = []
        with open(ratings_file, encoding="latin-1") as f:
            for line in f:
                p = line.strip().split("::")
                ratings.append((int(p[0]), int(p[1]), float(p[2]), int(p[3])))
    elif name == "ml-25m":
        ratings_file = os.path.join(data_dir, "ratings.csv")
        if not os.path.exists(ratings_file):
            zip_path = f"/tmp/{name}.zip"
            if not os.path.exists(zip_path):
                log.info(f"Downloading {name} (~250MB)...")
                urllib.request.urlretrieve(DATASET_URLS[name], zip_path)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall("/tmp")
        ratings = []
        with open(ratings_file) as f:
            next(f)  # skip header
            for line in f:
                p = line.strip().split(",")
                ratings.append((int(p[0]), int(p[1]), float(p[2]), int(p[3])))
    else:
        raise ValueError(f"Unknown dataset: {name}")

    log.info(f"Loaded {len(ratings)} ratings from {name}")
    return ratings


def load_amazon(category, data_dir="/tmp"):
    """Load Amazon review data. Expects pre-downloaded .jsonl.gz files."""
    fname = f"{data_dir}/amazon_{category}.jsonl.gz"
    if not os.path.exists(fname):
        fname_alt = f"{data_dir}/{category}.jsonl.gz"
        if os.path.exists(fname_alt):
            fname = fname_alt
        else:
            log.error(f"Amazon data not found at {fname}. Download from "
                      f"https://amazon-reviews-2023.github.io/ and place as {fname}")
            return []
    ratings = []
    user_map, item_map = {}, {}
    with gzip.open(fname, "rt") as f:
        for line in f:
            r = json.loads(line)
            uid = r.get("user_id", r.get("reviewerID", ""))
            iid = r.get("parent_asin", r.get("asin", ""))
            rating = float(r.get("rating", r.get("overall", 3.0)))
            ts = int(r.get("timestamp", r.get("unixReviewTime", 0)))
            if uid not in user_map:
                user_map[uid] = len(user_map)
            if iid not in item_map:
                item_map[iid] = len(item_map)
            ratings.append((user_map[uid], item_map[iid], rating, ts))
    log.info(f"Loaded {len(ratings)} Amazon {category} ratings, "
             f"{len(user_map)} users, {len(item_map)} items")
    return ratings


def prepare_data(ratings, emb_dim=64, min_seq_len=5, max_seq_len=15,
                 max_items=50000, min_interactions=20):
    """Filter, embed, split, build sequences."""
    items = sorted(set(r[1] for r in ratings))
    if len(items) > max_items:
        from collections import Counter
        item_counts = Counter(r[1] for r in ratings)
        top_items = set(i for i, _ in item_counts.most_common(max_items))
        ratings = [r for r in ratings if r[1] in top_items]
        items = sorted(set(r[1] for r in ratings))
        log.info(f"  Filtered to top {max_items} items, {len(ratings)} ratings")

    item_to_idx = {iid: i for i, iid in enumerate(items)}
    n_items = len(items)

    user_counts = {}
    for uid, iid, rating, ts in ratings:
        user_counts[uid] = user_counts.get(uid, 0) + 1
    active_users = {u for u, c in user_counts.items() if c >= min_interactions}
    ratings = [r for r in ratings if r[0] in active_users]
    n_users = max(r[0] for r in ratings) + 1

    timestamps = [r[3] for r in ratings]
    median_ts = int(np.median(timestamps))

    R_t0 = np.zeros((n_users, n_items), dtype=np.float32)
    R_t1 = np.zeros((n_users, n_items), dtype=np.float32)
    seqs_t0_raw, seqs_t1_raw = {}, {}

    for uid, iid, rating, ts in ratings:
        idx = item_to_idx[iid]
        if ts <= median_ts:
            R_t0[uid, idx] = rating
            seqs_t0_raw.setdefault(uid, []).append((ts, idx))
        else:
            R_t1[uid, idx] = rating
            seqs_t1_raw.setdefault(uid, []).append((ts, idx))

    k = min(emb_dim, min(R_t0.shape) - 1)
    U0, S0, Vt0 = np.linalg.svd(R_t0, full_matrices=False)
    embs_t0 = (Vt0[:k].T * S0[:k]).astype(np.float32)
    embs_t0 /= (np.linalg.norm(embs_t0, axis=1, keepdims=True) + 1e-8)

    U1, S1, Vt1 = np.linalg.svd(R_t1, full_matrices=False)
    embs_t1 = (Vt1[:k].T * S1[:k]).astype(np.float32)
    embs_t1 /= (np.linalg.norm(embs_t1, axis=1, keepdims=True) + 1e-8)

    def make_seqs(raw):
        out = []
        for uid, items_list in raw.items():
            items_list.sort()
            ids = [i for _, i in items_list]
            if len(ids) >= min_seq_len:
                out.append(ids[-max_seq_len:])
        return out

    seqs_t0 = make_seqs(seqs_t0_raw)
    seqs_t1 = make_seqs(seqs_t1_raw)
    log.info(f"  {n_items} items, {len(seqs_t0)} T0 seqs, {len(seqs_t1)} T1 seqs, emb_dim={k}")

    return embs_t0, embs_t1, seqs_t0, seqs_t1, n_items, k


# === RQ (same as before, compact) ===
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
    def mse(self, X):
        r = X.copy()
        for c in self.cb: a = _assign(r, c); r = r - c[a]
        return float(np.mean(np.sum(r ** 2, axis=1)))

def decode_with_codebook(codes, rq):
    out = np.zeros((len(codes), rq.dim), dtype=np.float32)
    for i in range(rq.m):
        c = np.clip(codes[:, i], 0, rq.K[i] - 1)
        out += rq.cb[i][c]
    return out

def warm_retrain(rq, X, fd, n_iter=20, seed=42):
    rq2 = RQ(rq.m, rq.K, rq.dim); rq2.cb = [c.copy() for c in rq.cb]
    r = X.copy()
    for i in range(fd): a = _assign(r, rq2.cb[i]); r = r - rq2.cb[i][a]
    rng = np.random.RandomState(seed)
    for i in range(fd, rq.m):
        c = _kmeans(r, rq.K[i], n_iter=n_iter, rng=rng, init=rq2.cb[i])
        rq2.cb[i] = c; a = _assign(r, c); r = r - c[a]
    return rq2


# === Transformer ===
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
        B = token_ids.shape[0]; out = self.forward(token_ids, stage_ids)
        all_codes, all_scores = [], []
        for b in range(B):
            h = out[b:b+1, -1:, :]
            beams = [([], 0.0)]
            for s in range(self.m):
                logits = self.heads[s](h[:, 0, :])
                log_probs = F.log_softmax(logits, dim=-1)[0]
                topk_vals, topk_ids = log_probs.topk(min(n_beams, len(log_probs)))
                new_beams = []
                for prev, prev_s in beams:
                    for v, idx in zip(topk_vals.tolist(), topk_ids.tolist()):
                        new_beams.append((prev + [idx], prev_s + v))
                new_beams.sort(key=lambda x: -x[1])
                beams = new_beams[:n_beams]
            all_codes.append(np.array([b_[0] for b_ in beams]))
            all_scores.append(np.array([b_[1] for b_ in beams]))
        return all_codes, all_scores


def seqs_to_tokens(sequences, item_codes, m, max_items=15):
    token_ids, stage_ids = [], []
    for seq in sequences:
        seq = seq[-max_items:]; toks, stgs = [], []
        for item_id in seq:
            for s in range(m): toks.append(item_codes[item_id, s]); stgs.append(s)
        token_ids.append(toks); stage_ids.append(stgs)
    return token_ids, stage_ids

def pad(lists, max_len, fill=0):
    B = len(lists); arr = np.full((B, max_len), fill, dtype=np.int64)
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
    for ep in range(epochs):
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


def evaluate_gen_ann(model, eval_seqs, item_codes_input, rq_query, rq_items,
                     target_embs, m, n_beams=20, top_k=10, device="cpu"):
    tok_ids, stg_ids = seqs_to_tokens([s[:-1] for s in eval_seqs], item_codes_input, m)
    targets = [s[-1] for s in eval_seqs]
    max_len = max(len(t) for t in tok_ids)
    tok_t = torch.from_numpy(pad(tok_ids, max_len)).long().to(device)
    stg_t = torch.from_numpy(pad(stg_ids, max_len)).long().to(device)
    item_codes_dec = rq_items.encode(target_embs)
    item_decoded = decode_with_codebook(item_codes_dec, rq_items)
    model.eval(); hits, ndcg, total = 0, 0.0, 0
    batch_size = 64
    for start in range(0, len(eval_seqs), batch_size):
        end = min(start + batch_size, len(eval_seqs))
        beam_codes, beam_scores = model.generate_next_codes(
            tok_t[start:end], stg_t[start:end], n_beams=n_beams)
        for b_idx in range(len(beam_codes)):
            query_vecs = decode_with_codebook(beam_codes[b_idx], rq_query)
            dists = cdist(query_vecs, item_decoded, metric="sqeuclidean")
            weighted = dists - beam_scores[b_idx][:, None] * 0.1
            ranking = np.argsort(weighted.min(axis=0))[:top_k]
            target = targets[start + b_idx]
            if target in ranking:
                hits += 1
                ndcg += 1.0 / math.log2(np.where(ranking == target)[0][0] + 2)
            total += 1
    return {"hr@10": hits / max(total, 1), "ndcg@10": ndcg / max(total, 1), "n_eval": total}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="ml-1m",
                   choices=["ml-1m", "ml-25m", "amazon-electronics", "amazon-books"])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--n-beams", type=int, default=20)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--json-output", type=str, default="generative_transport_multi.json")
    args = p.parse_args()

    t0 = time.time(); results = []
    m = 4; fd = 2; codes_list = [16, 16, 256, 256]

    if args.dataset.startswith("ml-"):
        ratings = load_movielens(args.dataset)
    elif args.dataset.startswith("amazon-"):
        category = args.dataset.replace("amazon-", "")
        ratings = load_amazon(category)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if not ratings:
        log.error("No data loaded"); return

    embs_t0, embs_t1, seqs_t0, seqs_t1, n_items, emb_dim = prepare_data(ratings)

    for seed in range(args.seeds):
        log.info(f"=== {args.dataset} seed={seed} ===")
        rq_src = RQ(m, codes_list, emb_dim).fit(embs_t0, seed=seed)
        rq_full = RQ(m, codes_list, emb_dim).fit(embs_t1, seed=seed + 500)
        rq_strat = warm_retrain(rq_src, embs_t1, fd, seed=seed)
        codes_t0_src = rq_src.encode(embs_t0)
        codes_t1_src = rq_src.encode(embs_t1)

        log.info("  training generator on T0...")
        tok_t0, stg_t0 = seqs_to_tokens(seqs_t0, codes_t0_src, m)
        torch.manual_seed(seed)
        model = SeqGenerator(codes_list, d_model=128, n_heads=4, n_layers=3)
        model = train_model(model, tok_t0, stg_t0, m, epochs=args.epochs, device=args.device)

        for sname, rq_items in [("frozen", rq_src), ("stratified", rq_strat),
                                ("full_retrain", rq_full)]:
            log.info(f"  {sname}...")
            metrics = evaluate_gen_ann(model, seqs_t1, codes_t1_src,
                rq_query=rq_src, rq_items=rq_items,
                target_embs=embs_t1, m=m, n_beams=args.n_beams, device=args.device)
            log.info(f"    HR@10={metrics['hr@10']:.4f}")
            results.append({"dataset": args.dataset, "seed": seed,
                            "strategy": sname, **metrics, "mse": rq_items.mse(embs_t1)})

        log.info("  retrained generator upper bound...")
        codes_t1_full = rq_full.encode(embs_t1)
        tok_f, stg_f = seqs_to_tokens(seqs_t1, codes_t1_full, m)
        torch.manual_seed(seed + 1000)
        model_new = SeqGenerator(codes_list, d_model=128, n_heads=4, n_layers=3)
        model_new = train_model(model_new, tok_f, stg_f, m, epochs=args.epochs, device=args.device)
        metrics = evaluate_gen_ann(model_new, seqs_t1, codes_t1_full,
            rq_query=rq_full, rq_items=rq_full,
            target_embs=embs_t1, m=m, n_beams=args.n_beams, device=args.device)
        log.info(f"    retrained: HR@10={metrics['hr@10']:.4f}")
        results.append({"dataset": args.dataset, "seed": seed,
                        "strategy": "retrained", **metrics, "mse": rq_full.mse(embs_t1)})

        with open(args.json_output, "w") as f:
            json.dump(results, f)

    log.info(f"Done in {time.time() - t0:.0f}s. {len(results)} rows.")


if __name__ == "__main__":
    main()

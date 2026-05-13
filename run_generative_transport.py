#!/usr/bin/env python3
"""Generate -> decode -> re-quantize -> ANN on MovieLens temporal drift.

Training:
- build the same MovieLens temporal split / embedding pipeline as the
  downstream funnel experiment
- train a generator on T0 semantic-ID sequences

Serving under drift:
- frozen generator emits old codes
- decode with the old codebook
- re-encode/decode through the current codebook
- ANN search in the current decoded space

This is the search-style generative handoff story: the generator
produces the coarse semantic query, and the current codebook plus ANN
resolve it to current items.

Usage:
    python3 run_generative_transport.py --seeds 5
    python3 run_generative_transport.py --seeds 5 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
import urllib.request
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.spatial.distance import cdist

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

ARCH_TO_CODES = {
    "uniform": [64, 64, 64, 64],
    "funnel": [16, 16, 256, 512],
}


def download_movielens(data_dir="/tmp/ml-1m"):
    if os.path.exists(os.path.join(data_dir, "ratings.dat")):
        return data_dir
    zip_path = "/tmp/ml-1m.zip"
    if not os.path.exists(zip_path):
        log.info("Downloading MovieLens-1M...")
        urllib.request.urlretrieve(
            "https://files.grouplens.org/datasets/movielens/ml-1m.zip", zip_path
        )
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("/tmp")
    return data_dir


def load_movielens_temporal_split(emb_dim=64, min_seq_len=5, max_seq_len=15):
    data_dir = download_movielens()
    ratings = []
    with open(os.path.join(data_dir, "ratings.dat"), "r", encoding="latin-1") as f:
        for line in f:
            user_id, item_id, rating, timestamp = line.strip().split("::")
            ratings.append((int(user_id), int(item_id), float(rating), int(timestamp)))

    ratings_np = np.array(ratings)
    timestamps = ratings_np[:, 3]
    median_ts = np.median(timestamps)
    old_mask = timestamps < median_ts
    new_mask = ~old_mask

    all_users = np.unique(ratings_np[:, 0].astype(int))
    all_items = np.unique(ratings_np[:, 1].astype(int))
    user_map = {user_id: i for i, user_id in enumerate(all_users)}
    item_map = {item_id: i for i, item_id in enumerate(all_items)}
    n_users = len(all_users)
    n_items = len(all_items)

    def build_sparse(mask):
        rows = [user_map[int(x)] for x in ratings_np[mask][:, 0]]
        cols = [item_map[int(x)] for x in ratings_np[mask][:, 1]]
        vals = ratings_np[mask][:, 2]
        return csr_matrix(
            (vals, (rows, cols)), shape=(n_users, n_items), dtype=np.float32
        )

    mat_old = build_sparse(old_mask)
    mat_new = build_sparse(new_mask)
    k = min(emb_dim, min(mat_old.shape) - 1, min(mat_new.shape) - 1)

    _, s_old, vt_old = svds(mat_old, k=k)
    _, s_new, vt_new = svds(mat_new, k=k)
    embs_t0 = (vt_old.T * s_old).astype(np.float32)
    embs_t1 = (vt_new.T * s_new).astype(np.float32)

    seqs_t0_raw = {}
    seqs_t1_raw = {}
    seen_t0 = {}
    items_seen_t0 = set()
    for user_id, item_id, _rating, timestamp in ratings:
        mapped_user = user_map[user_id]
        mapped_item = item_map[item_id]
        if timestamp < median_ts:
            seqs_t0_raw.setdefault(mapped_user, []).append((timestamp, mapped_item))
            seen_t0.setdefault(mapped_user, set()).add(mapped_item)
            items_seen_t0.add(mapped_item)
        else:
            seqs_t1_raw.setdefault(mapped_user, []).append((timestamp, mapped_item))

    def make_sequences(raw):
        out = []
        for items_list in raw.values():
            items_list.sort()
            ids = [item_id for _, item_id in items_list]
            if len(ids) >= min_seq_len:
                out.append(ids[-max_seq_len:])
        return out

    def make_eval_examples(raw):
        out = []
        for uid, items_list in raw.items():
            items_list.sort()
            ids = [item_id for _, item_id in items_list]
            if len(ids) >= min_seq_len:
                seq = ids[-max_seq_len:]
                out.append((uid, seq[:-1], seq[-1]))
        return out

    return {
        "embs_t0": embs_t0,
        "embs_t1": embs_t1,
        "seqs_t0": make_sequences(seqs_t0_raw),
        "eval_examples_t1": make_eval_examples(seqs_t1_raw),
        "seen_t0": seen_t0,
        "items_seen_t0": items_seen_t0,
        "n_users": n_users,
        "n_items": n_items,
    }


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


def decode_with_codebook(codes, rq):
    """Decode integer codes using the given RQ's codebooks."""
    out = np.zeros((len(codes), rq.dim), dtype=np.float32)
    for i in range(rq.m):
        c = np.clip(codes[:, i], 0, rq.K[i] - 1)
        out += rq.cb[i][c]
    return out


class SeqGenerator(nn.Module):
    def __init__(
        self,
        vocab_sizes,
        d_model=128,
        n_heads=4,
        n_layers=3,
        max_tokens=80,
        dropout=0.1,
    ):
        super().__init__()
        self.m = len(vocab_sizes)
        self.vocab_sizes = vocab_sizes
        self.d_model = d_model
        self.tok_embs = nn.ModuleList([nn.Embedding(vs, d_model) for vs in vocab_sizes])
        self.pos_emb = nn.Embedding(max_tokens, d_model)
        self.stage_emb = nn.Embedding(self.m, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
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
        causal = torch.triu(
            torch.ones(T, T, device=token_ids.device), diagonal=1
        ).bool()
        return self.transformer(embs, mask=causal)

    @torch.no_grad()
    def generate_next_codes(self, token_ids, stage_ids, n_beams=10):
        """Autoregressively generate top-n_beams candidate codes for the next item.
        Each stage conditions on the tokens chosen for previous stages."""
        B = token_ids.shape[0]
        device = token_ids.device
        all_codes = []
        all_scores = []
        for b in range(B):
            ctx_tok = token_ids[b : b + 1]
            ctx_stg = stage_ids[b : b + 1]
            beams = [([], 0.0)]
            for s in range(self.m):
                new_beams = []
                for prev_codes, prev_score in beams:
                    if prev_codes:
                        ext_tok = torch.cat(
                            [
                                ctx_tok,
                                torch.tensor(
                                    [prev_codes], dtype=torch.long, device=device
                                ),
                            ],
                            dim=1,
                        )
                        ext_stg = torch.cat(
                            [
                                ctx_stg,
                                torch.tensor(
                                    [list(range(len(prev_codes)))],
                                    dtype=torch.long,
                                    device=device,
                                ),
                            ],
                            dim=1,
                        )
                    else:
                        ext_tok, ext_stg = ctx_tok, ctx_stg
                    out = self.forward(ext_tok, ext_stg)
                    h = out[:, -1:, :]
                    logits = self.heads[s](h[:, 0, :])
                    log_probs = F.log_softmax(logits, dim=-1)[0]
                    topk_vals, topk_ids = log_probs.topk(min(n_beams, len(log_probs)))
                    for val, idx in zip(topk_vals.tolist(), topk_ids.tolist()):
                        new_beams.append((prev_codes + [idx], prev_score + val))
                new_beams.sort(key=lambda x: -x[1])
                beams = new_beams[:n_beams]
            all_codes.append(np.array([b_[0] for b_ in beams]))
            all_scores.append(np.array([b_[1] for b_ in beams]))
        return all_codes, all_scores


class RecoFromReconstruction(nn.Module):
    """Two-tower ranker from the decoded-vector ANN experiment."""

    def __init__(self, n_users, item_dim, emb_dim=32):
        super().__init__()
        self.item_proj = nn.Linear(item_dim, emb_dim)
        self.user_emb = nn.Embedding(n_users, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)

    def forward(self, user_ids, item_vecs):
        u = self.user_emb(user_ids)
        v = self.item_proj(item_vecs)
        return (u * v).sum(dim=-1)


def seqs_to_tokens(sequences, item_codes, m, max_items=15):
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


def pad(lists, max_len, fill=0):
    B = len(lists)
    arr = np.full((B, max_len), fill, dtype=np.int64)
    for i, L in enumerate(lists):
        n = min(len(L), max_len)
        arr[i, :n] = L[:n]
    return arr


def train_reco(
    model,
    pairs,
    item_reconstructions,
    n_items,
    epochs=15,
    lr=0.005,
    batch_size=4096,
    n_neg=4,
    seed=42,
):
    rng = np.random.RandomState(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    recon_t = torch.from_numpy(item_reconstructions).float()
    pairs_arr = np.array(pairs)
    for _ in range(epochs):
        rng.shuffle(pairs_arr)
        for start in range(0, len(pairs_arr), batch_size):
            batch = pairs_arr[start : start + batch_size]
            users = torch.from_numpy(batch[:, 0].astype(np.int64))
            pos_vecs = recon_t[batch[:, 1].astype(np.int64)]
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


def train_model(
    model, token_ids, stage_ids, m, epochs=30, lr=3e-4, batch_size=128, device="cpu"
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
                    for s in range(m)
                    if (ts == s).any()
                )
                / m
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    model.eval()
    return model


def evaluate_gen_then_ann(
    model,
    eval_examples,
    item_codes_input,
    query_item_codes,
    rq_source,
    rq_items,
    target_embs,
    seen_by_user,
    m,
    query_mode="generator",
    n_beams=20,
    top_k=10,
    beam_weight=0.1,
    device="cpu",
):
    """
    Generate → Decode → Re-quantize → ANN.

    1. Query proposal in the old code space
       - generator mode: autoregressively predict old codes from history
       - oracle mode: use the target item's old-code assignment directly
    2. Decode with OLD codebook → continuous vector
    3. Re-quantize through CURRENT codebook (encode → decode)
       - Stratified: same prefix, better suffix → better query
       - Full retrain: different prefix → query lands in wrong region
    4. ANN against items decoded with CURRENT codebook
       - Both query and items are now in the SAME decoded space

    This mirrors the paper's ANN experiment: the codebook quality
    determines reconstruction fidelity in continuous space.
    """
    histories = [history for _, history, _ in eval_examples]
    targets = [target for _, _, target in eval_examples]
    user_ids = [uid for uid, _, _ in eval_examples]
    if query_mode == "generator":
        tok_ids, stg_ids = seqs_to_tokens(histories, item_codes_input, m)
        max_len = max(len(t) for t in tok_ids)
        tok_t = torch.from_numpy(pad(tok_ids, max_len)).long().to(device)
        stg_t = torch.from_numpy(pad(stg_ids, max_len)).long().to(device)

    # Items: encode + decode with current codebook
    item_decoded = decode_with_codebook(rq_items.encode(target_embs), rq_items)

    if query_mode == "generator":
        if model is None:
            raise ValueError("generator mode requires a trained model")
        model.eval()
    hits, ndcg, total = 0, 0.0, 0
    batch_size = 64

    for start in range(0, len(eval_examples), batch_size):
        end = min(start + batch_size, len(eval_examples))
        batch_targets = targets[start:end]
        batch_histories = histories[start:end]
        batch_users = user_ids[start:end]

        if query_mode == "generator":
            bt = tok_t[start:end]
            bs = stg_t[start:end]
            beam_codes, beam_scores = model.generate_next_codes(bt, bs, n_beams=n_beams)
        elif query_mode == "oracle":
            if query_item_codes is None:
                raise ValueError("oracle mode requires query_item_codes")
            beam_codes = [
                query_item_codes[np.array([target], dtype=np.int64)]
                for target in batch_targets
            ]
            beam_scores = [np.zeros(1, dtype=np.float32) for _ in batch_targets]
        else:
            raise ValueError(f"Unknown query_mode: {query_mode}")

        for b_idx, codes in enumerate(beam_codes):
            scores = beam_scores[b_idx]
            # Decode generated codes with OLD codebook → continuous vectors
            query_continuous = decode_with_codebook(codes, rq_source)
            # Re-quantize through CURRENT codebook → same space as items
            query_requant = decode_with_codebook(
                rq_items.encode(query_continuous), rq_items
            )
            # ANN in the shared decoded space
            dists = cdist(query_requant, item_decoded, metric="sqeuclidean")
            weighted = dists - scores[:, None] * beam_weight
            best_per_item = weighted.min(axis=0)
            blocked = seen_by_user.get(batch_users[b_idx], set()) | set(
                batch_histories[b_idx]
            )
            if blocked:
                best_per_item[list(blocked)] = np.inf
            ranking = np.argsort(best_per_item)[:top_k]

            target = batch_targets[b_idx]
            if target in ranking:
                hits += 1
                rank = np.where(ranking == target)[0][0]
                ndcg += 1.0 / math.log2(rank + 2)
            total += 1

    return {
        "hr@10": hits / max(total, 1),
        "ndcg@10": ndcg / max(total, 1),
        "n_eval": total,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--n-beams", type=int, default=20)
    p.add_argument("--arch", choices=sorted(ARCH_TO_CODES), default="funnel")
    p.add_argument("--freeze-depth", type=int, default=2)
    p.add_argument("--query-mode", choices=["generator", "oracle"], default="generator")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--beam-weight", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--json-output", type=str, default="generative_transport.json")
    args = p.parse_args()

    t0 = time.time()
    results = []
    codes_list = ARCH_TO_CODES[args.arch]
    m = len(codes_list)
    fd = args.freeze_depth
    data = load_movielens_temporal_split(emb_dim=64)
    embs_t0 = data["embs_t0"]
    embs_t1 = data["embs_t1"]
    seqs_t0 = data["seqs_t0"]
    eval_examples_t1 = data["eval_examples_t1"]
    seen_t0 = data["seen_t0"]
    items_seen_t0 = data["items_seen_t0"]
    emb_dim = embs_t0.shape[1]
    if args.query_mode == "oracle":
        eval_examples_t1 = [ex for ex in eval_examples_t1 if ex[2] in items_seen_t0]
    log.info(
        f"arch={args.arch} codes={codes_list} "
        f"T0 seqs={len(seqs_t0)} T1 eval examples={len(eval_examples_t1)}"
    )

    for seed in range(args.seeds):
        log.info(f"=== seed={seed} ===")

        rq_src = RQ(m, codes_list, emb_dim).fit(embs_t0, seed=seed)
        rq_full = RQ(m, codes_list, emb_dim).fit(embs_t1, seed=seed + 500)
        rq_strat = warm_retrain(rq_src, embs_t1, fd, seed=seed)

        codes_t0_src = rq_src.encode(embs_t0)
        model = None
        if args.query_mode == "generator":
            log.info("  training generator on T0...")
            tok_t0, stg_t0 = seqs_to_tokens(seqs_t0, codes_t0_src, m)
            torch.manual_seed(seed)
            model = SeqGenerator(codes_list, d_model=128, n_heads=4, n_layers=3)
            model = train_model(
                model, tok_t0, stg_t0, m, epochs=args.epochs, device=args.device
            )

        codes_t1_src = rq_src.encode(embs_t1)

        strategies = [
            ("items_frozen", rq_src),
            ("items_stratified", rq_strat),
            ("items_full_retrain", rq_full),
        ]

        for sname, rq_items in strategies:
            log.info(f"  evaluating {sname}...")
            metrics = evaluate_gen_then_ann(
                model,
                eval_examples_t1,
                codes_t1_src,
                query_item_codes=codes_t0_src,
                rq_source=rq_src,
                rq_items=rq_items,
                target_embs=embs_t1,
                seen_by_user=seen_t0,
                m=m,
                query_mode=args.query_mode,
                n_beams=args.n_beams,
                top_k=args.top_k,
                beam_weight=args.beam_weight,
                device=args.device,
            )
            log.info(
                f"    HR@10={metrics['hr@10']:.4f} NDCG@10={metrics['ndcg@10']:.4f}"
            )
            results.append(
                {
                    "seed": seed,
                    "strategy": sname,
                    "query_mode": args.query_mode,
                    **metrics,
                    "mse": rq_items.mse(embs_t1),
                }
            )

        # Held-out T1 upper bound: train on each user's T1 history, predict the last item.
        log.info("  evaluating full_retrained...")
        codes_t1_full = rq_full.encode(embs_t1)
        seqs_t1_train = [
            history for _, history, _ in eval_examples_t1 if len(history) >= 4
        ]
        seen_by_user_new = {uid: set(history) for uid, history, _ in eval_examples_t1}
        model_new = None
        if args.query_mode == "generator":
            log.info("    training T1 generator...")
            tok_t1f, stg_t1f = seqs_to_tokens(seqs_t1_train, codes_t1_full, m)
            torch.manual_seed(seed + 1000)
            model_new = SeqGenerator(codes_list, d_model=128, n_heads=4, n_layers=3)
            model_new = train_model(
                model_new, tok_t1f, stg_t1f, m, epochs=args.epochs, device=args.device
            )
        metrics = evaluate_gen_then_ann(
            model_new,
            eval_examples_t1,
            codes_t1_full,
            query_item_codes=codes_t1_full,
            rq_source=rq_full,
            rq_items=rq_full,
            target_embs=embs_t1,
            seen_by_user=seen_by_user_new,
            m=m,
            query_mode=args.query_mode,
            n_beams=args.n_beams,
            top_k=args.top_k,
            beam_weight=args.beam_weight,
            device=args.device,
        )
        log.info(
            f"    full_retrained: HR@10={metrics['hr@10']:.4f} "
            f"NDCG@10={metrics['ndcg@10']:.4f}"
        )
        results.append(
            {
                "seed": seed,
                "strategy": "full_retrained",
                "query_mode": args.query_mode,
                **metrics,
                "mse": rq_full.mse(embs_t1),
            }
        )

        with open(args.json_output, "w") as f:
            json.dump(results, f)

    log.info(f"Done in {time.time() - t0:.0f}s. {len(results)} rows.")


if __name__ == "__main__":
    main()

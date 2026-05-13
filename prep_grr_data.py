#!/usr/bin/env python3
"""Prepare MovieLens data for GRR training with custom RQ semantic IDs.

Generates:
  1. semantic_ids_t0.json, semantic_ids_t1_{frozen,strat,full}.json
  2. GRR-format Parquet files for training (T0) and evaluation (T1)

The Parquet format matches GRR's expected columns:
  - description: "The user has purchased: <|sid_begin|><s_a_X>...<|sid_end|>; ..."
  - groundtruth: "<|sid_begin|><s_a_X><s_b_X><s_c_X><s_d_X><|sid_end|>"

Usage:
    python3 prep_grr_data.py --output-dir /tmp/grr_movielens
"""
from __future__ import annotations
import argparse, json, logging, os, urllib.request, zipfile
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


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


def warm_retrain(rq, X, fd, n_iter=20, seed=42):
    rq2 = RQ(rq.m, rq.K, rq.dim); rq2.cb = [c.copy() for c in rq.cb]
    r = X.copy()
    for i in range(fd): a = _assign(r, rq2.cb[i]); r = r - rq2.cb[i][a]
    rng = np.random.RandomState(seed)
    for i in range(fd, rq.m):
        c = _kmeans(r, rq.K[i], n_iter=n_iter, rng=rng, init=rq2.cb[i])
        rq2.cb[i] = c; a = _assign(r, c); r = r - c[a]
    return rq2


def codes_to_sid_string(codes):
    prefixes = ["s_a", "s_b", "s_c", "s_d"]
    tokens = "".join(f"<{prefixes[i]}_{codes[i]}>" for i in range(len(codes)))
    return f"<|sid_begin|>{tokens}<|sid_end|>"


def build_parquet(user_seqs, item_codes, output_path, max_hist=10):
    rows = []
    for seq in user_seqs:
        if len(seq) < 3:
            continue
        hist = seq[:-1][-max_hist:]
        target = seq[-1]
        hist_sids = "; ".join(codes_to_sid_string(item_codes[i]) for i in hist)
        desc = f"The user has purchased the following items: {hist_sids}"
        gt = codes_to_sid_string(item_codes[target])
        rows.append({"description": desc, "groundtruth": gt})

    if HAS_PANDAS:
        df = pd.DataFrame(rows)
        df.to_parquet(output_path, index=False)
        log.info(f"  Wrote {len(df)} rows to {output_path}")
    else:
        json_path = output_path.replace(".parquet", ".jsonl")
        with open(json_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        log.info(f"  Wrote {len(rows)} rows to {json_path} (no pandas, using JSONL)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, default="/tmp/grr_movielens")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data_dir = download_movielens()

    ratings = []
    with open(os.path.join(data_dir, "ratings.dat"), "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            ratings.append((int(parts[0]), int(parts[1]), float(parts[2]), int(parts[3])))
    log.info(f"Loaded {len(ratings)} ratings")

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

    emb_dim = 64
    U0, S0, Vt0 = np.linalg.svd(R_t0, full_matrices=False)
    embs_t0 = (Vt0[:emb_dim].T * S0[:emb_dim]).astype(np.float32)
    embs_t0 = embs_t0 / (np.linalg.norm(embs_t0, axis=1, keepdims=True) + 1e-8)

    U1, S1, Vt1 = np.linalg.svd(R_t1, full_matrices=False)
    embs_t1 = (Vt1[:emb_dim].T * S1[:emb_dim]).astype(np.float32)
    embs_t1 = embs_t1 / (np.linalg.norm(embs_t1, axis=1, keepdims=True) + 1e-8)

    # Use 256^4 uniform to match GRR's hardcoded vocab
    m = 4; codes = 256; fd = 2
    log.info(f"Training RQ (uniform {codes}^{m})...")
    rq_src = RQ(m, codes, emb_dim).fit(embs_t0, seed=args.seed)
    rq_full = RQ(m, codes, emb_dim).fit(embs_t1, seed=args.seed + 500)
    rq_strat = warm_retrain(rq_src, embs_t1, fd, seed=args.seed)

    for name, rq, embs in [("t0_source", rq_src, embs_t0),
                            ("t1_frozen", rq_src, embs_t1),
                            ("t1_strat", rq_strat, embs_t1),
                            ("t1_full", rq_full, embs_t1)]:
        codes_arr = rq.encode(embs)
        sid_dict = {str(i): codes_arr[i].tolist() for i in range(n_items)}
        path = os.path.join(args.output_dir, f"semantic_ids_{name}.json")
        with open(path, "w") as f:
            json.dump(sid_dict, f)
        log.info(f"  {name}: {path}")

    def make_seqs(raw, min_len=3, max_len=20):
        out = []
        for uid, items_list in raw.items():
            items_list.sort()
            ids = [i for _, i in items_list]
            if len(ids) >= min_len:
                out.append(ids[-max_len:])
        return out

    s_t0 = make_seqs(seqs_t0)
    s_t1 = make_seqs(seqs_t1)
    log.info(f"{len(s_t0)} T0 seqs, {len(s_t1)} T1 seqs")

    codes_t0_src = rq_src.encode(embs_t0)
    codes_t1_frozen = rq_src.encode(embs_t1)
    codes_t1_strat = rq_strat.encode(embs_t1)
    codes_t1_full = rq_full.encode(embs_t1)

    build_parquet(s_t0, codes_t0_src,
                  os.path.join(args.output_dir, "train_t0.parquet"))
    for name, codes_arr in [("frozen", codes_t1_frozen),
                            ("strat", codes_t1_strat),
                            ("full", codes_t1_full)]:
        build_parquet(s_t1, codes_arr,
                      os.path.join(args.output_dir, f"eval_t1_{name}.parquet"))

    log.info("Done. Upload output_dir to devvm for GRR training.")


if __name__ == "__main__":
    main()

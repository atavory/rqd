#!/usr/bin/env python3
"""End-to-end generative recommender with RQ semantic IDs.

Trains a tiny causal transformer on RQ token sequences (like TIGER/DSI),
then measures whether the old model can still predict tokens after drift.

Key result: full retrain breaks the old model (cross-entropy explodes on
prefix stages because the vocabulary meaning changed). Stratified preserves
prefix prediction quality.

Usage:
    python3 run_generative_e2e.py --seeds 5
    python3 run_generative_e2e.py --seeds 5 --device cuda
"""
from __future__ import annotations
import argparse, json, logging, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


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

    def decode(self, codes):
        out = np.zeros((len(codes), self.dim), dtype=np.float32)
        for i in range(self.m):
            out += self.cb[i][codes[:, i]]
        return out

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


class TokenTransformer(nn.Module):
    def __init__(self, vocab_sizes, d_model=128, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.m = len(vocab_sizes)
        self.vocab_sizes = vocab_sizes
        self.d_model = d_model
        self.embeddings = nn.ModuleList([
            nn.Embedding(vs, d_model) for vs in vocab_sizes
        ])
        self.bos_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_emb = nn.Embedding(self.m, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.heads = nn.ModuleList([nn.Linear(d_model, vs) for vs in vocab_sizes])

    def forward(self, codes):
        B = codes.shape[0]
        embs = [self.bos_emb.expand(B, -1, -1)]
        for i in range(self.m - 1):
            embs.append(self.embeddings[i](codes[:, i]).unsqueeze(1))
        x = torch.cat(embs, dim=1)  # (B, m, d_model)
        pos = torch.arange(self.m, device=codes.device)
        x = x + self.pos_emb(pos).unsqueeze(0)
        mask = torch.triu(torch.ones(self.m, self.m, device=codes.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)
        return [self.heads[i](x[:, i]) for i in range(self.m)]

    def loss_per_stage(self, codes):
        logits = self.forward(codes)
        losses = []
        for i in range(self.m):
            losses.append(F.cross_entropy(logits[i], codes[:, i]).item())
        return losses

    def acc_per_stage(self, codes):
        logits = self.forward(codes)
        accs = []
        for i in range(self.m):
            pred = logits[i].argmax(dim=-1)
            accs.append((pred == codes[:, i]).float().mean().item())
        return accs

    def generate(self, n, device="cpu"):
        codes = torch.zeros(n, self.m, dtype=torch.long, device=device)
        with torch.no_grad():
            for i in range(self.m):
                if i == 0:
                    x = self.bos_emb.expand(n, -1, -1)
                else:
                    embs = [self.bos_emb.expand(n, -1, -1)]
                    for j in range(i):
                        embs.append(self.embeddings[j](codes[:, j]).unsqueeze(1))
                    x = torch.cat(embs, dim=1)
                pos = torch.arange(i + 1, device=device)
                x = x + self.pos_emb(pos[:i + 1]).unsqueeze(0)
                mask = torch.triu(torch.ones(i + 1, i + 1, device=device), diagonal=1).bool()
                x = self.transformer(x, mask=mask)
                logits = self.heads[i](x[:, -1])
                codes[:, i] = logits.argmax(dim=-1)
        return codes


def train_model(model, codes_np, epochs=100, lr=3e-4, batch_size=256, device="cpu"):
    codes_t = torch.from_numpy(codes_np).long().to(device)
    model = model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(codes_t)
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            batch = codes_t[perm[i:i + batch_size]]
            logits = model(batch)
            loss = sum(F.cross_entropy(logits[j], batch[:, j]) for j in range(model.m))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()
    model.eval()
    return model


@torch.no_grad()
def eval_model(model, codes_np, device="cpu"):
    codes_t = torch.from_numpy(codes_np).long().to(device)
    model.eval()
    ce = model.loss_per_stage(codes_t)
    acc = model.acc_per_stage(codes_t)
    generated = model.generate(len(codes_t), device=device).cpu().numpy()
    seq_match = float(np.mean(np.all(generated == codes_np, axis=1)))
    pfx_match_1 = float(np.mean(generated[:, 0] == codes_np[:, 0]))
    pfx_match_2 = float(np.mean(np.all(generated[:, :2] == codes_np[:, :2], axis=1)))
    return {
        "ce_per_stage": ce,
        "acc_per_stage": acc,
        "seq_exact_match": seq_match,
        "prefix_1_match": pfx_match_1,
        "prefix_2_match": pfx_match_2,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--json-output", type=str, default="generative_e2e.json")
    args = p.parse_args()

    t0 = time.time()
    results = []
    m = 4
    archs = [
        ("uniform_64_4", [64, 64, 64, 64]),
        ("funnel_4", [16, 16, 256, 256]),
    ]
    alphas = [1.0, 3.0, 5.0]
    fd = 2

    for alpha in alphas:
        for aname, codes_list in archs:
            for seed in range(args.seeds):
                log.info(f"alpha={alpha} {aname} seed={seed}")
                rng = np.random.RandomState(seed)
                nc = 20
                centers = rng.randn(nc, args.dim).astype(np.float32) * 3.0
                labels = rng.randint(0, nc, size=args.n)
                X0 = centers[labels] + rng.randn(args.n, args.dim).astype(np.float32) * 0.5
                d_vec = rng.randn(args.dim).astype(np.float32)
                d_vec /= np.linalg.norm(d_vec)
                X1 = (X0 + d_vec * alpha).astype(np.float32)

                log.info("  fitting RQ variants...")
                rq_source = RQ(m, codes_list, args.dim).fit(X0, seed=seed)
                rq_full = RQ(m, codes_list, args.dim).fit(X1, seed=seed + 500)
                rq_strat = warm_retrain(rq_source, X1, fd, seed=seed)

                codes_source_on_X0 = rq_source.encode(X0)
                codes_source_on_X1 = rq_source.encode(X1)  # frozen
                codes_full_on_X1 = rq_full.encode(X1)
                codes_strat_on_X1 = rq_strat.encode(X1)

                log.info("  training source model...")
                torch.manual_seed(seed)
                model = TokenTransformer(codes_list, d_model=128, n_heads=4, n_layers=3)
                model = train_model(model, codes_source_on_X0, epochs=args.epochs,
                                    device=args.device)

                baseline = eval_model(model, codes_source_on_X0, device=args.device)
                log.info(f"  baseline acc={[f'{a:.3f}' for a in baseline['acc_per_stage']]}")

                strategies = {
                    "baseline_source": (codes_source_on_X0, "source model on source data"),
                    "frozen": (codes_source_on_X1, "source model, frozen codebook on target"),
                    "stratified_old_model": (codes_strat_on_X1, "source model, stratified codebook on target"),
                    "full_old_model": (codes_full_on_X1, "source model, full-retrain codebook on target"),
                }

                for sname, (eval_codes, desc) in strategies.items():
                    metrics = eval_model(model, eval_codes, device=args.device)
                    row = {
                        "alpha": alpha, "arch": aname, "seed": seed,
                        "strategy": sname,
                        "ce_per_stage": metrics["ce_per_stage"],
                        "acc_per_stage": metrics["acc_per_stage"],
                        "seq_exact_match": metrics["seq_exact_match"],
                        "prefix_1_match": metrics["prefix_1_match"],
                        "prefix_2_match": metrics["prefix_2_match"],
                        "mean_ce": sum(metrics["ce_per_stage"]) / m,
                        "mean_acc": sum(metrics["acc_per_stage"]) / m,
                    }
                    results.append(row)
                    log.info(f"    {sname}: acc={[f'{a:.3f}' for a in metrics['acc_per_stage']]} "
                             f"ce={[f'{c:.2f}' for c in metrics['ce_per_stage']]}")

                log.info("  training full-retrain model...")
                torch.manual_seed(seed + 1000)
                model_full = TokenTransformer(codes_list, d_model=128, n_heads=4, n_layers=3)
                model_full = train_model(model_full, codes_full_on_X1, epochs=args.epochs,
                                         device=args.device)
                metrics = eval_model(model_full, codes_full_on_X1, device=args.device)
                results.append({
                    "alpha": alpha, "arch": aname, "seed": seed,
                    "strategy": "full_new_model",
                    "ce_per_stage": metrics["ce_per_stage"],
                    "acc_per_stage": metrics["acc_per_stage"],
                    "seq_exact_match": metrics["seq_exact_match"],
                    "prefix_1_match": metrics["prefix_1_match"],
                    "prefix_2_match": metrics["prefix_2_match"],
                    "mean_ce": sum(metrics["ce_per_stage"]) / m,
                    "mean_acc": sum(metrics["acc_per_stage"]) / m,
                })
                log.info(f"    full_new_model: acc={[f'{a:.3f}' for a in metrics['acc_per_stage']]}")

                log.info("  training stratified-retrained model...")
                torch.manual_seed(seed + 2000)
                model_strat = TokenTransformer(codes_list, d_model=128, n_heads=4, n_layers=3)
                model_strat = train_model(model_strat, codes_strat_on_X1, epochs=args.epochs,
                                          device=args.device)
                metrics = eval_model(model_strat, codes_strat_on_X1, device=args.device)
                results.append({
                    "alpha": alpha, "arch": aname, "seed": seed,
                    "strategy": "stratified_new_model",
                    "ce_per_stage": metrics["ce_per_stage"],
                    "acc_per_stage": metrics["acc_per_stage"],
                    "seq_exact_match": metrics["seq_exact_match"],
                    "prefix_1_match": metrics["prefix_1_match"],
                    "prefix_2_match": metrics["prefix_2_match"],
                    "mean_ce": sum(metrics["ce_per_stage"]) / m,
                    "mean_acc": sum(metrics["acc_per_stage"]) / m,
                })
                log.info(f"    strat_new_model: acc={[f'{a:.3f}' for a in metrics['acc_per_stage']]}")

                with open(args.json_output, "w") as f:
                    json.dump(results, f)

    log.info(f"Done in {time.time() - t0:.0f}s. {len(results)} rows.")


if __name__ == "__main__":
    main()

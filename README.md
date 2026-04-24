# Stable Semantic IDs under Distribution Shift

Code for "Stable Semantic IDs under Distribution Shift" (CIKM 2026 submission).

## Setup

```bash
pip install -r requirements.txt
```

Requires only `numpy` and `scipy`. No GPU needed.

## Library

`rq.py` — core implementation:
- `RQCodebook`: Residual quantizer with greedy stage-wise k-means. Supports uniform and funnel (non-uniform K) architectures.
- `warm_retrain`: Warm-retrain suffix stages on shifted data, keeping the prefix frozen.
- `gap_recovery`: Compute the gap recovery ratio ρ.
- `codebook_entropy`: Shannon entropy of codebook usage per stage.
- `retrieval_recall_at_k`: Recall@K via asymmetric decode.
- `generate_data` / `apply_drift`: Synthetic Gaussian blobs with mean-shift, scale, or rotation drift.

## Reproducing Key Results

### The 70% law and funnel architecture (Theorem 1, Tables 1-3)

```bash
python3 demo_seventy_percent.py
```

Sweeps M, K, d, and drift magnitude. Shows ~70% recovery (uniform) and >90% (funnel), scale-invariant.

### Streaming adaptation (Table 6, Figure 5)

```bash
python3 demo_streaming.py
```

Compares static, one-shot, periodic, and triggered warm-retraining over 10 snapshots of increasing drift.

### Baselines: EWC, EMA, flat VQ (Table 5)

```bash
python3 demo_baselines.py
```

Shows that EWC, EMA, and flat VQ cannot simultaneously preserve prefix codes and adapt to drift.

### Low-data transfer (Table 4)

```bash
python3 demo_lowdata.py
```

Demonstrates frozen prefix as structural regularizer: warm-2 at 5% data beats full retrain at 25%.

## License

MIT

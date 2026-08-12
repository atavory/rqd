# Semantic IDs as Interfaces Under Distribution Shift

Public experiment code for maintaining semantic-ID interfaces as embedding
models and interaction logs evolve.

## Setup

```bash
pip install -r requirements.txt
```

The committed producer runs on CPU and additionally uses CPU PyTorch for the
downstream consumer. A batched CUDA RQ/k-means backend is planned for the
larger WSDM matrix; GPU results are not interchangeable with the committed CPU
path until parity tests pass.

## Corrected web-recommender suite

`run_wsdm_web_recsys.py` is the public runner for the paper's focused temporal
experiments on MovieLens-1M, Amazon Reviews 2018, and the Amazon Reviews 2023
five-core benchmark. It provides:

- deterministic temporal data preparation and immutable numeric caches;
- funnel and matched-bit uniform residual quantizers;
- freeze-depth and beam-count sweeps;
- frozen, suffix-adapted, full-retrain/old-consumer, and
  full-retrain/rebuilt-consumer strategies;
- centroid-Hungarian and assignment-optimal global relabeling baselines for
  the old consumer, testing whether token permutation alone explains failure;
- recall, routing coverage, candidate work, MSE, prefix churn, reindex counts,
  and codebook-update size in one JSON artifact per seed;
- raw token churn together with centroid-Hungarian and assignment-optimal
  label-aligned churn, so arbitrary k-means token permutations are not
  mistaken for genuine interface changes.
- an `--index-only` mode for full-catalog reconstruction, aligned churn, and
  prefix-bucket statistics without the much smaller sampled consumer workload.

Variable-length histories are trained in exact-length batches, so padding never
enters the model or loss. MovieLens uses shared-basis alignment plus global RMS
calibration; Amazon uses independent SVD plus orthogonal Procrustes alignment.

Prepare a cache:

```bash
python3 run_wsdm_web_recsys.py \
  --dataset movielens \
  --cache data/movielens_scaled.npz \
  --prepare-only --embedding-dim 64
```

For the full Amazon Electronics 2018 5-core catalog, use zero additional
degree-filtering passes. Sequence limits affect generator training/evaluation,
not the interaction data used to construct the temporal embeddings:

```bash
python3 run_wsdm_web_recsys.py \
  --dataset amazon \
  --amazon-data data/amazon/Electronics_5.json.gz \
  --amazon-core-passes 0 \
  --max-train-sequences 50000 --max-eval-sequences 10000 \
  --cache data/amazon_full5core.npz \
  --prepare-only --embedding-dim 64
```

Amazon Reviews 2023 `benchmark/5core/rating_only/*.csv` files use the same
Amazon loader. For example:

```bash
python3 run_wsdm_web_recsys.py \
  --dataset amazon \
  --amazon-data data/amazon2023/Books.csv \
  --amazon-core-passes 0 \
  --max-train-sequences 50000 --max-eval-sequences 2000 \
  --cache data/amazon2023_books.npz \
  --prepare-only --embedding-dim 64
```

Run the full catalog without training a downstream generator:

```bash
python3 run_wsdm_web_recsys.py \
  --dataset amazon --cache data/amazon2023_books.npz \
  --arch funnel24 --freeze-depth 2 --seed 0 --index-only \
  --output results/amazon2023_books_funnel24_fd2_seed0.json
```

The index-only artifact contains normalized and absolute MSE, gap recovery,
all three churn conventions, update bytes, occupied-prefix counts, bucket-size
quantiles, and effective prefix count. Consumer results remain a separate
sampled evaluation and must report their sequence counts.

For the primary multi-seed study, reuse the same source/full endpoints across
all freeze depths:

```bash
python3 run_wsdm_index_sweep.py \
  --cache data/amazon2023_books.npz --arch funnel24 --seed 0 \
  --freeze-depths 1,2,3 \
  --output results/amazon2023_books_funnel24_seed0.json
```

The WSDM-scale protocol evaluates six full catalogs with five seeds and all
three freeze depths, plus a separate multi-million-item Amazon Books stress
test and focused architecture/consumer subsets. The larger raw matrix is
reported compactly: one dataset-and-scale table, one aggregate full-catalog
table, and one aggregate downstream-consumer table. Detailed rows remain in
the companion artifact rather than becoming one paper table per dataset.

Run one independently reproducible seed:

```bash
python3 run_wsdm_web_recsys.py \
  --dataset movielens \
  --cache data/movielens_scaled.npz \
  --arch funnel24 --freeze-depth 2 --seed 0 \
  --epochs 50 --n-beams 10 --beam-values 1,5,10 \
  --output results/movielens_funnel24_fd2_seed0.json
```

The companion data artifact is the canonical source for dataset URLs and
hashes, filtering/alignment metadata, per-seed JSONs, committed CSV aggregates,
and table-to-artifact mappings. The paper does not consume uncommitted or
locally aggregated results.

### Churn interpretation

Full retraining is an independent k-means++ fit (`seed + 500`), not a warm
start from the source codebook. Its integer cluster labels are arbitrary.
Every full-retrain result therefore records three conventions:

- `prefix_churn_raw`: direct token inequality;
- `prefix_churn_centroid_aligned`: a deployable global token permutation from
  minimum-cost centroid matching;
- `prefix_churn_assignment_aligned`: the maximum-agreement global permutation
  on retained catalog items, providing a lower bound after relabeling.

Raw churn alone is diagnostic and must not be reported as evidence of a
vocabulary migration without the aligned controls.

The downstream runner also evaluates the unchanged old consumer against the
fully retrained catalog after applying each global mapping to the retrained
prefix tokens. If either baseline restores routing, the failure is attributable
to a removable label permutation rather than genuine interface reassignment.

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

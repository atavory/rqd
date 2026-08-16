# Semantic IDs as Interfaces Under Distribution Shift

Public experiment code for maintaining semantic-ID interfaces as embedding
models and interaction logs evolve.

## Paper-facing analysis contract

This repository is the canonical public source for analysis and run scripts.
The data Overleaf project stores generated CSV, Markdown, LaTeX-table, and
plot-data artifacts, plus manifests that record the exact script hash and input
hashes used to produce them. Paper numbers and plots must be regenerated from
committed scripts; do not copy session-local calculations into the paper or
edit generated data artifacts by hand.

Current WSDM paper-analysis generator:

```bash
python3 make_wsdm_overleaf_analysis.py \
  --experiment-root /data/users/atavory/scratch/wsdm_experiments \
  --output-dir /data/users/atavory/scratch/wsdm_experiments/overleaf_data/wsdm_analysis_latest \
  --hash-inputs
```

The generated package is mirrored to the data Overleaf directory
`results/wsdm_2027_paper_analysis/`. A hash-identical provenance copy of the
generator may appear inside that package, but the public script source is this
GitHub repository.

## Setup

```bash
pip install -r requirements.txt
```

The committed producer runs on CPU and additionally uses CPU PyTorch for the
downstream model. A batched CUDA RQ/k-means backend is planned for the
larger WSDM matrix; GPU results are not interchangeable with the committed CPU
path until parity tests pass.

## Corrected web-recommender suite

`run_wsdm_web_recsys.py` is the public runner for the paper's focused temporal
experiments on MovieLens-1M, Amazon Reviews 2018, and the Amazon Reviews 2023
five-core benchmark. It provides:

- deterministic temporal data preparation and immutable numeric caches;
- funnel and matched-bit uniform residual quantizers;
- freeze-depth and beam-count sweeps;
- frozen, suffix-adapted, full-retrain/old-model, and
  full-retrain/rebuilt-model strategies;
- warm-start-full, EMA/streaming-VQ, and GRM-only comparison baselines;
- centroid-Hungarian and assignment-optimal global relabeling baselines for
  the old model, testing whether token permutation alone explains failure;
- HR/NDCG/Recall at standard cutoffs, routing coverage, candidate work, MSE,
  prefix churn, reindex counts, and codebook-update size in one JSON artifact
  per seed;
- raw token churn together with centroid-Hungarian and assignment-optimal
  label-aligned churn, so arbitrary k-means token permutations are not
  mistaken for genuine interface changes.
- per-strategy cost-axis fields: tokenizer update seconds, downstream-model retrain
  seconds, training-sequence counts, ID-migration flags, and update wall time.
- an `--index-only` mode for full-catalog reconstruction, aligned churn, and
  prefix-bucket statistics without the much smaller sampled model workload.

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
test and focused architecture/downstream-model subsets. The larger raw matrix is
reported compactly: one dataset-and-scale table, one aggregate full-catalog
table, and one aggregate downstream-model table. Detailed rows remain in
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

The downstream runner also evaluates the unchanged old model against the
fully retrained catalog after applying each global mapping to the retrained
prefix tokens. If either baseline restores routing, the failure is attributable
to a removable label permutation rather than genuine interface reassignment.

### Cost-axis interpretation

The WSDM comparison is not a best-NDCG-at-any-cost bakeoff. JSON and CSV rows
therefore distinguish:

- `tokenizer_update_seconds`: codebook maintenance cost for the strategy;
- legacy raw keys `consumer_retrain_seconds` and `consumer_training_sequences`:
  downstream GRM retraining cost, nonzero for GRM-only and rebuilt-model
  baselines;
- `id_migration_required` / `serving_index_rebuild_required`: whether retained
  item IDs must move under assignment-aligned churn;
- `update_wall_seconds`: tokenizer update plus downstream retrain cost.

Paper-facing churn uses `prefix_churn_headline`, which is assignment-optimal
aligned churn. Bare `prefix_churn` remains a raw-churn compatibility alias.

### Tier-C diagnostics

The retraining-necessity theory is an empirical target, not just prose. Future
JSON artifacts include a `diagnostics` object and the summarizer emits
`diagnostic_rows.csv` with:

- `xi_s`: target-minus-source drift energy in the frozen prefix
  centroid-difference subspace;
- prefix-margin quantiles and fragile-margin fractions;
- `epsilon_s_temporal`: source-to-target crossing through the frozen source
  prefix map;
- suffix repair residual after stratified adaptation;
- candidate coverage and `Delta_task` for within-stable-prefix task drift.

`prefix_churn_headline` remains the assignment-aligned migration churn for
independently updated codebooks. It is not a substitute for
`epsilon_s_temporal`.

The diagnostic claim is gated on tests and synthetic controls: drift projected
into the prefix orthogonal complement must have `xi_s` near zero and no
temporal crossings; drift projected into the prefix subspace must cross; and
downstream-model-only drift must raise `Delta_task`/GRM-only lift while geometry and
interface probes stay low.

`summarize_wsdm_results.py` writes `downstream_rows.csv`, `index_runs.csv`,
`diagnostic_rows.csv`, `index_summary.csv`, and `cost_summary.csv` from a
result directory.

### Paper-facing analysis and Overleaf artifacts

All paper-facing analysis and plotting for the WSDM run must be regenerated
from scripts. Do not compute abstract numbers, paper tables, or figures in an
interactive session.

`make_wsdm_overleaf_analysis.py` is the reproducible paper-facing generator.
It reads raw JSON/CSV result artifacts, excludes incomplete downstream JSONs
from headline tables, records partial jobs in coverage reports, and writes an
Overleaf-ready package containing:

- `manifest.json` with input paths, timestamps, optional input hashes, and
  Manifold artifact references;
- normalized CSVs under `csv/`;
- generated LaTeX tables under `tables/`;
- PGFPlots snippets under `figures/` backed by CSVs under `plot_data/`;
- `coverage.md`, which states exactly which results are complete or partial.

Example for the current WSDM run:

```bash
python3 make_wsdm_overleaf_analysis.py \
  --experiment-root /data/users/atavory/scratch/wsdm_experiments \
  --output-dir overleaf_data/wsdm_analysis_latest \
  --artifact-ref manifold:aai_research_tlv/tree/atavory/wsdm_results_snapshot_20260814_210842.tar.zst#ed6eb768a97dfa6ecc60d5dd7d2cfcd12e2c2fff4aad7df18f079e926cd2c7e3 \
  --artifact-ref manifold:aai_research_tlv/tree/atavory/wsdm_remote_results/cont_si2/wsdm_remote_results_snapshot_20260814T143740_cont_si2.tar.zst \
  --artifact-ref manifold:aai_research_tlv/tree/atavory/wsdm_remote_results/cont_si3/cont_si3_dact_tools_artifact_20260814T215749.tar.zst \
  --artifact-ref manifold:aai_research_tlv/tree/atavory/wsdm_remote_results/cont_si3/cont_si3_lcrec_qwen_adapter_snapshot_20260814T220200.tar.zst
```

`run_wsdm_paper_completion_loop_20260815.py` is the long-running coordinator
for the same package. It polls local downstream JSONs and remote Manifold
artifact directories, regenerates the Overleaf-ready analysis package when the
completion signature changes, uploads a verified tarball to Manifold, and
syncs the generated data package/snippets into the two Overleaf Git remotes.
It uses `~/.mrgitties/mrgitties.sock` only as a fallback when direct Overleaf
HTTPS push is blocked by the local proxy.

### Tier-C retraining prediction scripts

`run_tier_c_retrain_prediction.py` is the public synthetic-control runner for
testing the decision rule. It creates known drift regimes and predicts the
cheapest sufficient update rung:

- `do_nothing`: no meaningful drift;
- `geometry_reconstruction`: suffix-only tokenizer repair should be enough;
- `suffix_capacity`: the suffix cannot repair the drift at the current rate;
- `interface_drift`: prefix routes cross, so selective/full migration is
  required;
- legacy synthetic key `consumer_only`: geometry and routing stay stable, but
  GRM/model retrain is required.

Example:

```bash
python3 run_tier_c_retrain_prediction.py \
  --output results/tier_c_synthetic_retrain_predictions.json \
  --csv-output results/tier_c_synthetic_retrain_predictions.csv \
  --arches funnel24 --seeds 0,1,2,3,4 \
  --freeze-depths 1,2,3 \
  --magnitudes 0.0,0.05,0.15,0.35,0.7
```

`summarize_tier_c_retrain_predictions.py` reads completed WSDM JSON artifacts
and emits one real-dataset prediction CSV:

```bash
python3 summarize_tier_c_retrain_predictions.py \
  --results-dir results/phase0_ml1m_primary24_20260812 \
  --results-dir results/amazon_tierb_20260812 \
  --output results/tier_c_real_retrain_predictions.csv
```

`run_tier_c_retrain_prediction_queue.sh` is the overnight queue wrapper. It
runs the synthetic matrix, reruns real index sweeps with the committed
diagnostic schema, waits for the existing WSDM queues, and then writes the
combined Tier-C prediction CSV. Defaults are intentionally absolute for the
two-GPU scratch box:

```bash
OUT_ROOT=/data/users/atavory/scratch/wsdm_experiments/results/tier_c_retrain_prediction_20260812 \
  ./run_tier_c_retrain_prediction_queue.sh
```

The queue is restart-safe: complete valid JSONs are skipped, and logs live
under `$OUT_ROOT/queue/logs/`.

`run_wsdm_overnight_watchdog.sh` is the scratch-box watchdog used for long
overnight runs. It checks the persistent WSDM queue, Amazon 2023 queue, and
Tier-C queue every five minutes, restarts a queue if its PID is dead and its
log has not reached the done marker, and writes heartbeat logs under
`/data/users/atavory/scratch/wsdm_experiments/results/wsdm_overnight_watchdog_20260812/`.

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

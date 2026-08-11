#!/usr/bin/env python3
"""Focused WSDM experiment: evolving prefix-routed web recommendation.

The runner answers four questions on MovieLens-1M and Amazon Electronics:

1. Does suffix-only adaptation improve a frozen prefix-routed consumer?
2. Does full codebook retraining break that consumer until it is retrained?
3. Does a funnel beat uniform RQ at the same total bitrate?
4. What index churn, update time, storage, and query cost does each strategy incur?

Data preparation is separate from seeded runs so every seed/configuration uses
the exact same temporal split and embeddings. Each seeded process writes one
JSON artifact and should be invoked with a unique --output path.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import time
from array import array
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from run_generative_prefix import (
    PrefixGenerator,
    RQ,
    load_movielens_shared_basis,
    pad,
    prefix_seqs,
    train_prefix_gen,
    warm_retrain,
)


ARCHITECTURES = {
    # Matched-bit uniform/funnel pairs. The funnel spends fewer bits on the
    # stable routing namespace and moves them to the adaptive suffix.
    "uniform16": [16, 16, 16, 16],
    "funnel16": [4, 4, 64, 64],
    "uniform20": [32, 32, 32, 32],
    "funnel20": [8, 8, 128, 128],
    "uniform24": [64, 64, 64, 64],
    # Intermediate allocation for the web-indexing tradeoff: a larger stable
    # namespace than funnel24, while retaining more adaptive suffix capacity
    # than the uniform layout.
    "balanced24": [32, 32, 128, 128],
    "funnel24": [16, 16, 256, 256],
}


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _svds_deterministic(matrix, k: int):
    """SciPy-version-compatible deterministic truncated SVD."""
    try:
        return svds(matrix, k=k, rng=np.random.default_rng(0))
    except TypeError:
        return svds(matrix, k=k, random_state=0)


def _make_sequences(user_ids, item_ids, timestamps, split_ts,
                    min_seq_len=5, max_seq_len=15):
    old, new = {}, {}
    for u, i, ts in zip(user_ids.tolist(), item_ids.tolist(), timestamps.tolist()):
        target = old if ts < split_ts else new
        target.setdefault(u, []).append((ts, i))

    def ordered(raw):
        out = []
        for interactions in raw.values():
            interactions.sort()
            ids = [i for _, i in interactions]
            if len(ids) >= min_seq_len:
                out.append(ids[-max_seq_len:])
        return out

    seqs_t0 = ordered(old)
    seqs_t1 = ordered(new)
    eval_t1 = [(idx, seq[:-1], seq[-1]) for idx, seq in enumerate(seqs_t1)]
    return seqs_t0, eval_t1


def _prepare_movielens(cache_path: Path, emb_dim: int) -> dict:
    embs_t0, embs_t1, seqs_t0, eval_t1, _, n_users, n_items = \
        load_movielens_shared_basis(emb_dim=emb_dim)
    # The temporal halves contain different interaction mass. Shared singular
    # vectors remove rotation ambiguity but do not preserve embedding scale:
    # without this unlabeled scalar calibration the target RMS norm is about
    # 8.5x smaller and nearly every item collapses into one frozen prefix.
    source_rms = float(np.sqrt(np.mean(np.sum(embs_t0 ** 2, axis=1))))
    target_rms_raw = float(np.sqrt(np.mean(np.sum(embs_t1 ** 2, axis=1))))
    target_scale = source_rms / max(target_rms_raw, 1e-12)
    embs_t1 = (embs_t1 * target_scale).astype(np.float32)
    metadata = {
        "dataset": "movielens",
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": 1_000_209,
        "n_train_sequences": len(seqs_t0),
        "n_eval_sequences": len(eval_t1),
        "embedding_dim": int(embs_t0.shape[1]),
        "split": "global median timestamp",
        "embedding_alignment": "shared old-user SVD basis plus global RMS-norm calibration",
        "source_rms_norm": source_rms,
        "target_rms_norm_before_calibration": target_rms_raw,
        "target_scale": target_scale,
    }
    np.savez_compressed(
        cache_path,
        embs_t0=embs_t0,
        embs_t1=embs_t1,
        seqs_t0=np.asarray(seqs_t0, dtype=object),
        eval_t1=np.asarray(eval_t1, dtype=object),
        metadata=np.asarray(json.dumps(metadata)),
    )
    return metadata


def _load_amazon_arrays(data_path: Path, core_passes: int):
    """Read Amazon 2018 JSONL.gz or Amazon 2023 five-core CSV."""
    user_map, item_map = {}, {}
    users, items = array("I"), array("I")
    ratings, timestamps = array("f"), array("q")

    if data_path.suffix == ".csv":
        source = data_path.open("r", newline="")
        rows = csv.DictReader(source)

        def fields(row):
            return (
                row["user_id"], row["parent_asin"],
                float(row["rating"]), int(row["timestamp"]),
            )
    else:
        source = gzip.open(data_path, "rt")
        rows = (json.loads(line) for line in source)

        def fields(row):
            return (
                row["reviewerID"], row["asin"],
                float(row["overall"]), int(row["unixReviewTime"]),
            )

    with source:
        for line_no, row in enumerate(rows, 1):
            user, item, rating, timestamp = fields(row)
            u = user_map.setdefault(user, len(user_map))
            i = item_map.setdefault(item, len(item_map))
            users.append(u)
            items.append(i)
            ratings.append(rating)
            timestamps.append(timestamp)
            if line_no % 1_000_000 == 0:
                print(f"  parsed {line_no:,} Amazon reviews", flush=True)

    u = np.frombuffer(users, dtype=np.uint32).copy()
    i = np.frombuffer(items, dtype=np.uint32).copy()
    r = np.frombuffer(ratings, dtype=np.float32).copy()
    ts = np.frombuffer(timestamps, dtype=np.int64).copy()
    del users, items, ratings, timestamps, user_map, item_map

    active = np.ones(len(u), dtype=bool)
    # The May experiment used ten synchronous peeling passes. Zero keeps the
    # complete public Electronics 5-core category file for the scale study.
    # Positive values are approximate 20-cores, not peeling to convergence.
    for iteration in range(core_passes):
        uc = np.bincount(u[active], minlength=int(u.max()) + 1)
        ic = np.bincount(i[active], minlength=int(i.max()) + 1)
        updated = active & (uc[u] >= 20) & (ic[i] >= 20)
        print(
            f"  20-core iteration {iteration + 1}: {int(updated.sum()):,} reviews",
            flush=True,
        )
        active = updated

    u, i, r, ts = u[active], i[active], r[active], ts[active]
    unique_u, u = np.unique(u, return_inverse=True)
    unique_i, i = np.unique(i, return_inverse=True)
    return (
        u.astype(np.int32),
        i.astype(np.int32),
        r.astype(np.float32),
        ts.astype(np.int64),
        len(unique_u),
        len(unique_i),
    )


def _sample_sequences(values, limit: int, seed: int):
    if not limit or len(values) <= limit:
        return values
    rng = np.random.RandomState(seed)
    selected = np.sort(rng.choice(len(values), size=limit, replace=False))
    return [values[index] for index in selected]


def _prepare_amazon(cache_path: Path, data_path: Path, emb_dim: int,
                    core_passes: int, max_train_sequences: int,
                    max_eval_sequences: int, sequence_sample_seed: int) -> dict:
    user_ids, item_ids, ratings, timestamps, n_users, n_items = \
        _load_amazon_arrays(data_path, core_passes)
    split_ts = int(np.median(timestamps))
    old_mask = timestamps < split_ts
    new_mask = ~old_mask

    old = csr_matrix(
        (ratings[old_mask], (user_ids[old_mask], item_ids[old_mask])),
        shape=(n_users, n_items),
        dtype=np.float32,
    )
    new = csr_matrix(
        (ratings[new_mask], (user_ids[new_mask], item_ids[new_mask])),
        shape=(n_users, n_items),
        dtype=np.float32,
    )
    k = min(emb_dim, min(old.shape) - 1)
    _, s0, vt0 = _svds_deterministic(old, k)
    _, s1, vt1 = _svds_deterministic(new, k)
    embs_t0 = (vt0.T * s0).astype(np.float32)
    embs_t1_raw = (vt1.T * s1).astype(np.float32)
    rotation, _ = orthogonal_procrustes(embs_t1_raw, embs_t0)
    embs_t1 = (embs_t1_raw @ rotation).astype(np.float32)
    alignment_mse = float(np.mean((embs_t1 - embs_t0) ** 2))

    seqs_t0, eval_t1 = _make_sequences(
        user_ids, item_ids, timestamps, split_ts,
    )
    n_train_sequences_total = len(seqs_t0)
    n_eval_sequences_total = len(eval_t1)
    seqs_t0 = _sample_sequences(
        seqs_t0, max_train_sequences, sequence_sample_seed,
    )
    eval_t1 = _sample_sequences(
        eval_t1, max_eval_sequences, sequence_sample_seed + 1,
    )
    if data_path.suffix == ".csv":
        benchmark_core = (
            "zero-core" if "0core" in data_path.stem.lower() else "five-core"
        )
        source_release = (
            f"Amazon Reviews 2023 {benchmark_core} rating-only benchmark"
        )
    else:
        source_release = "Amazon Reviews 2018 five-core category file"
    filtering = (
        f"public {source_release} with no additional filtering"
        if core_passes == 0 else
        f"{core_passes} synchronous user/item degree-20 filtering passes"
    )
    metadata = {
        "dataset": "amazon",
        "dataset_variant": data_path.stem,
        "source_release": source_release,
        "source": str(data_path),
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": int(len(ratings)),
        "n_old_interactions": int(old_mask.sum()),
        "n_new_interactions": int(new_mask.sum()),
        "n_train_sequences": len(seqs_t0),
        "n_eval_sequences": len(eval_t1),
        "n_train_sequences_before_sampling": n_train_sequences_total,
        "n_eval_sequences_before_sampling": n_eval_sequences_total,
        "sequence_sample_seed": sequence_sample_seed,
        "amazon_core_passes": core_passes,
        "embedding_dim": int(k),
        "split_timestamp": split_ts,
        "filtering": filtering,
        "split": f"global median timestamp after {filtering}",
        "embedding_alignment": "independent SVD plus orthogonal Procrustes",
        "alignment_mse": alignment_mse,
    }
    np.savez_compressed(
        cache_path,
        embs_t0=embs_t0,
        embs_t1=embs_t1,
        seqs_t0=np.asarray(seqs_t0, dtype=object),
        eval_t1=np.asarray(eval_t1, dtype=object),
        metadata=np.asarray(json.dumps(metadata)),
    )
    return metadata


def prepare_dataset(args) -> None:
    cache_path = Path(args.cache)
    if cache_path.exists() and not args.overwrite:
        raise FileExistsError(f"cache exists: {cache_path}; pass --overwrite")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.dataset == "movielens":
        metadata = _prepare_movielens(cache_path, args.embedding_dim)
    else:
        metadata = _prepare_amazon(
            cache_path, Path(args.amazon_data), args.embedding_dim,
            args.amazon_core_passes, args.max_train_sequences,
            args.max_eval_sequences, args.sequence_sample_seed,
        )
    metadata["preparation_seconds"] = time.perf_counter() - started
    # Store the final timing next to the immutable numeric cache.
    _write_json(cache_path.with_suffix(".metadata.json"), metadata)
    print(json.dumps(metadata, indent=2), flush=True)


def load_cache(path: Path):
    data = np.load(path, allow_pickle=True)
    embs_t0 = data["embs_t0"].astype(np.float32)
    embs_t1 = data["embs_t1"].astype(np.float32)
    seqs_t0 = [list(map(int, seq)) for seq in data["seqs_t0"]]
    eval_t1 = []
    for row in data["eval_t1"]:
        _, history, target = row
        eval_t1.append((0, list(map(int, history)), int(target)))
    metadata = json.loads(str(data["metadata"].item()))
    return embs_t0, embs_t1, seqs_t0, eval_t1, metadata


def _codebook_bytes(rq: RQ, stages=None) -> int:
    if stages is None:
        stages = range(rq.m)
    return int(sum(rq.cb[s].nbytes for s in stages))


def _prefix_bucket_metrics(rq: RQ, embeddings, freeze_depth: int) -> dict:
    """Summarize the full catalog partition induced by a prefix."""
    codes = rq.encode(embeddings)[:, :freeze_depth]
    packed = np.zeros(len(codes), dtype=np.uint64)
    for stage in range(freeze_depth):
        packed = packed * np.uint64(rq.K[stage]) + codes[:, stage].astype(
            np.uint64
        )
    _, counts = np.unique(packed, return_counts=True)
    probabilities = counts.astype(np.float64) / max(len(codes), 1)
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    return {
        "occupied_prefixes": int(len(counts)),
        "possible_prefixes": int(np.prod(rq.K[:freeze_depth])),
        "prefix_occupancy_fraction": float(
            len(counts) / max(int(np.prod(rq.K[:freeze_depth])), 1)
        ),
        "items_per_prefix_mean": float(counts.mean()),
        "items_per_prefix_p50": float(np.percentile(counts, 50)),
        "items_per_prefix_p95": float(np.percentile(counts, 95)),
        "items_per_prefix_max": int(counts.max()),
        "prefix_entropy_bits": entropy,
        "effective_prefixes": float(2.0 ** entropy),
    }


def _index_strategy_metrics(name, rq, embeddings, freeze_depth, churn,
                            n_items: int, codebook_update_bytes: int) -> dict:
    mse = rq.mse(embeddings)
    energy = float(np.mean(np.sum(embeddings ** 2, axis=1)))
    return {
        "strategy": name,
        "mse": mse,
        "normalized_mse": float(mse / max(energy, 1e-12)),
        "prefix_churn": churn["raw"],
        "prefix_churn_raw": churn["raw"],
        "prefix_churn_centroid_aligned": churn["centroid_aligned"],
        "prefix_churn_assignment_aligned": churn["assignment_aligned"],
        "items_reindexed": int(round(churn["raw"] * n_items)),
        "items_reindexed_centroid_aligned": int(round(
            churn["centroid_aligned"] * n_items
        )),
        "items_reindexed_assignment_aligned": int(round(
            churn["assignment_aligned"] * n_items
        )),
        "codebook_update_bytes": codebook_update_bytes,
        **_prefix_bucket_metrics(rq, embeddings, freeze_depth),
    }


def _prefix_churn_metrics(rq_src: RQ, rq_new: RQ, embeddings,
                          freeze_depth: int, return_mappings: bool = False):
    """Measure raw churn and the part not removable by token relabeling.

    Independent k-means fits assign arbitrary integer labels to otherwise
    equivalent clusters. Raw token inequality therefore mixes genuine item
    reassignment with a removable permutation of the vocabulary. We report
    two explicit controls:

    * centroid-aligned churn uses a deployable global permutation obtained by
      minimum-cost bipartite matching of source and target code vectors;
    * assignment-aligned churn uses the permutation that maximizes agreement
      on the retained catalog, and is a lower bound for any global relabeling.

    Matching is stage-wise because each RQ stage has its own token namespace.
    """
    source = rq_src.encode(embeddings)[:, :freeze_depth]
    current = rq_new.encode(embeddings)[:, :freeze_depth]
    centroid_aligned = current.copy()
    assignment_aligned = current.copy()
    centroid_mappings = []
    assignment_mappings = []

    for stage in range(freeze_depth):
        k = rq_src.K[stage]

        source_centroids = rq_src.cb[stage]
        current_centroids = rq_new.cb[stage]
        centroid_cost = np.sum(
            (source_centroids[:, None, :] - current_centroids[None, :, :]) ** 2,
            axis=2,
        )
        source_rows, current_cols = linear_sum_assignment(centroid_cost)
        current_to_source = np.empty(k, dtype=np.int64)
        current_to_source[current_cols] = source_rows
        centroid_mappings.append(current_to_source.copy())
        centroid_aligned[:, stage] = current_to_source[current[:, stage]]

        overlap = np.zeros((k, k), dtype=np.int64)
        np.add.at(overlap, (source[:, stage], current[:, stage]), 1)
        source_rows, current_cols = linear_sum_assignment(-overlap)
        current_to_source = np.empty(k, dtype=np.int64)
        current_to_source[current_cols] = source_rows
        assignment_mappings.append(current_to_source.copy())
        assignment_aligned[:, stage] = current_to_source[current[:, stage]]

    metrics = {
        "raw": float(np.any(source != current, axis=1).mean()),
        "centroid_aligned": float(
            np.any(source != centroid_aligned, axis=1).mean()
        ),
        "assignment_aligned": float(
            np.any(source != assignment_aligned, axis=1).mean()
        ),
    }
    if return_mappings:
        return metrics, {
            "centroid_hungarian": centroid_mappings,
            "assignment_optimal": assignment_mappings,
        }
    return metrics


def _apply_prefix_mapping(codes, mappings):
    mapped = codes.copy()
    if mappings is not None:
        for stage, current_to_source in enumerate(mappings):
            mapped[:, stage] = current_to_source[mapped[:, stage]]
    return mapped


@torch.no_grad()
def evaluate_prefix_routing(model, eval_t1, history_rq, item_rq, embeddings,
                            freeze_depth, n_beams, device,
                            item_prefix_mapping=None):
    histories = [history for _, history, _ in eval_t1]
    targets = [target for _, _, target in eval_t1]
    history_codes = history_rq.encode(embeddings)
    item_codes_raw = item_rq.encode(embeddings)
    item_decoded = item_rq.decode_codes(item_codes_raw)
    item_codes = _apply_prefix_mapping(item_codes_raw, item_prefix_mapping)

    prefix_to_items = {}
    for item_id, row in enumerate(item_codes[:, :freeze_depth]):
        prefix_to_items.setdefault(tuple(row.tolist()), []).append(item_id)

    tok_eval, stg_eval = prefix_seqs(histories, history_codes, freeze_depth)
    max_len = max(len(tokens) for tokens in tok_eval)
    tok = torch.from_numpy(pad(tok_eval, max_len)).long().to(device)
    stg = torch.from_numpy(pad(stg_eval, max_len)).long().to(device)
    lengths = torch.tensor([len(tokens) for tokens in tok_eval], device=device)
    padding_mask = torch.arange(max_len, device=device)[None, :] >= lengths[:, None]

    recalls = {10: 0, 50: 0, 200: 0}
    conditional = {10: 0, 50: 0, 200: 0}
    coverage = 0
    candidate_counts = []
    started = time.perf_counter()
    for start in range(0, len(eval_t1), 64):
        end = min(start + 64, len(eval_t1))
        prefixes, _ = model.generate_prefixes(
            tok[start:end], stg[start:end], n_beams=n_beams,
            padding_mask=padding_mask[start:end],
        )
        for local in range(end - start):
            candidate_set = set()
            for prefix in prefixes[local]:
                candidate_set.update(prefix_to_items.get(tuple(prefix), ()))
            candidate_counts.append(len(candidate_set))
            if not candidate_set:
                continue
            target = targets[start + local]
            covered = target in candidate_set
            coverage += int(covered)
            candidates = np.asarray(sorted(candidate_set), dtype=np.int64)
            history = histories[start + local]
            query = embeddings[history[-3:]].mean(axis=0)
            distances = np.sum((item_decoded[candidates] - query) ** 2, axis=1)
            top_count = min(200, len(candidates))
            top = np.argpartition(distances, top_count - 1)[:top_count]
            top = top[np.argsort(distances[top])]
            ranking = candidates[top]
            for cutoff in recalls:
                hit = target in ranking[:cutoff]
                recalls[cutoff] += int(hit)
                conditional[cutoff] += int(hit and covered)

    elapsed = time.perf_counter() - started
    total = len(eval_t1)
    candidate_counts = np.asarray(candidate_counts, dtype=np.float64)
    return {
        "n_eval": total,
        "routing_coverage": coverage / max(total, 1),
        "recall_at_10": recalls[10] / max(total, 1),
        "recall_at_50": recalls[50] / max(total, 1),
        "recall_at_200": recalls[200] / max(total, 1),
        "conditional_recall_at_10": conditional[10] / max(coverage, 1),
        "conditional_recall_at_50": conditional[50] / max(coverage, 1),
        "conditional_recall_at_200": conditional[200] / max(coverage, 1),
        "candidate_count_mean": float(candidate_counts.mean()),
        "candidate_count_p50": float(np.percentile(candidate_counts, 50)),
        "candidate_count_p95": float(np.percentile(candidate_counts, 95)),
        "evaluation_seconds": elapsed,
        "query_milliseconds": 1000.0 * elapsed / max(total, 1),
    }


def evaluate_beam_sweep(model, eval_t1, history_rq, item_rq, embeddings,
                        freeze_depth, beam_values, device, strategy,
                        item_prefix_mapping=None):
    rows = []
    for n_beams in beam_values:
        print(f"evaluating {strategy} with {n_beams} beams", flush=True)
        metrics = evaluate_prefix_routing(
            model, eval_t1, history_rq, item_rq, embeddings,
            freeze_depth, n_beams, device, item_prefix_mapping,
        )
        metrics.update({"strategy": strategy, "n_beams": n_beams})
        rows.append(metrics)
    return rows


def run_seed(args) -> None:
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output}; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    embeddings_t0, embeddings_t1, seqs_t0, eval_t1, metadata = \
        load_cache(Path(args.cache))
    codes = ARCHITECTURES[args.arch]
    freeze_depth = args.freeze_depth
    seed = args.seed
    total_bits = int(sum(round(math.log2(k)) for k in codes))
    n_items = len(embeddings_t0)

    payload = {
        "schema_version": 3,
        "dataset": metadata,
        "configuration": {
            "arch": args.arch,
            "codes_per_stage": codes,
            "total_bits": total_bits,
            "freeze_depth": freeze_depth,
            "seed": seed,
            "epochs": args.epochs,
            "n_beams": args.n_beams,
            "beam_values": args.beam_values,
            "device": args.device,
            "capacity": int(np.prod(np.asarray(codes, dtype=np.int64))),
            "capacity_per_item": float(np.prod(np.asarray(codes, dtype=np.int64)) / n_items),
            "source_codebook_initialization": "independent k-means++",
            "source_codebook_seed": seed,
            "full_codebook_initialization": "independent k-means++ (not warm-started)",
            "full_codebook_seed": seed + 500,
            "label_alignment_during_training": "none",
        },
        "timing": {},
        "strategies": [],
        "beam_sweep": [],
    }
    _write_json(output, payload)

    started = time.perf_counter()
    rq_source = RQ(4, codes, embeddings_t0.shape[1]).fit(
        embeddings_t0, seed=seed,
    )
    payload["timing"]["source_codebook_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    rq_stratified = warm_retrain(
        rq_source, embeddings_t1, freeze_depth, seed=seed,
    )
    payload["timing"]["suffix_update_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    rq_full = RQ(4, codes, embeddings_t1.shape[1]).fit(
        embeddings_t1, seed=seed + 500,
    )
    payload["timing"]["full_codebook_seconds"] = time.perf_counter() - started
    full_churn, full_mappings = _prefix_churn_metrics(
        rq_source, rq_full, embeddings_t1, freeze_depth,
        return_mappings=True,
    )

    zero_churn = {"raw": 0.0, "centroid_aligned": 0.0,
                  "assignment_aligned": 0.0}
    if args.index_only:
        payload["configuration"]["mode"] = "full_catalog_index_only"
        payload["strategies"] = [
            _index_strategy_metrics(
                "frozen", rq_source, embeddings_t1, freeze_depth,
                zero_churn, n_items, 0,
            ),
            _index_strategy_metrics(
                "stratified", rq_stratified, embeddings_t1, freeze_depth,
                zero_churn, n_items,
                _codebook_bytes(rq_stratified, range(freeze_depth, 4)),
            ),
            _index_strategy_metrics(
                "full_retrained", rq_full, embeddings_t1, freeze_depth,
                full_churn, n_items, _codebook_bytes(rq_full),
            ),
        ]
        frozen_mse, stratified_mse, full_mse = (
            row["mse"] for row in payload["strategies"]
        )
        payload["summary"] = {
            "stratified_gap_recovery": float(
                (frozen_mse - stratified_mse)
                / max(frozen_mse - full_mse, 1e-12)
            ),
        }
        payload["timing"]["total_seconds"] = sum(
            value for value in payload["timing"].values()
            if isinstance(value, float)
        )
        _write_json(output, payload)
        print(json.dumps(payload, indent=2, default=_json_default), flush=True)
        return

    source_codes = rq_source.encode(embeddings_t0)
    tok_t0, stg_t0 = prefix_seqs(seqs_t0, source_codes, freeze_depth)
    torch.manual_seed(seed)
    source_model = PrefixGenerator(
        codes[:freeze_depth], d_model=128, n_heads=4, n_layers=3,
    )
    started = time.perf_counter()
    source_model = train_prefix_gen(
        source_model, tok_t0, stg_t0, freeze_depth,
        epochs=args.epochs, device=args.device,
    )
    payload["timing"]["source_generator_seconds"] = time.perf_counter() - started

    strategies = [
        (
            "frozen", source_model, rq_source, rq_source, zero_churn, False,
            None, "none",
        ),
        (
            "stratified", source_model, rq_source, rq_stratified, zero_churn,
            False, None, "none",
        ),
        (
            "full_old_generator", source_model, rq_source, rq_full,
            full_churn, False, None, "none",
        ),
        (
            "full_old_generator_centroid_relabel", source_model, rq_source,
            rq_full, full_churn, False,
            full_mappings["centroid_hungarian"], "centroid_hungarian",
        ),
        (
            "full_old_generator_assignment_relabel", source_model, rq_source,
            rq_full, full_churn, False,
            full_mappings["assignment_optimal"], "assignment_optimal",
        ),
    ]
    for (name, model, history_rq, item_rq, churn, retrained,
         item_prefix_mapping, relabeling) in strategies:
        print(f"evaluating {name}", flush=True)
        metrics = evaluate_prefix_routing(
            model, eval_t1, history_rq, item_rq, embeddings_t1,
            freeze_depth, args.n_beams, args.device, item_prefix_mapping,
        )
        metrics.update({
            "strategy": name,
            "mse": item_rq.mse(embeddings_t1),
            # Keep prefix_churn/items_reindexed as raw aliases for artifact
            # compatibility; papers must identify the alignment convention.
            "prefix_churn": churn["raw"],
            "prefix_churn_raw": churn["raw"],
            "prefix_churn_centroid_aligned": churn["centroid_aligned"],
            "prefix_churn_assignment_aligned": churn["assignment_aligned"],
            "items_reindexed": int(round(churn["raw"] * n_items)),
            "items_reindexed_centroid_aligned": int(round(
                churn["centroid_aligned"] * n_items
            )),
            "items_reindexed_assignment_aligned": int(round(
                churn["assignment_aligned"] * n_items
            )),
            "consumer_retrained": retrained,
            "consumer_token_relabeling": relabeling,
            "token_relabeling_bytes": int(sum(
                mapping.nbytes for mapping in (item_prefix_mapping or [])
            )),
            "codebook_update_bytes": (
                0 if name == "frozen" else
                _codebook_bytes(item_rq, range(freeze_depth, 4))
                if name == "stratified" else _codebook_bytes(item_rq)
            ),
        })
        payload["strategies"].append(metrics)
        payload["beam_sweep"].extend(evaluate_beam_sweep(
            model, eval_t1, history_rq, item_rq, embeddings_t1,
            freeze_depth,
            [value for value in args.beam_values if value != args.n_beams],
            args.device, name, item_prefix_mapping,
        ))
        _write_json(output, payload)

    if args.skip_rebuilt_consumer:
        payload["timing"]["total_seconds"] = sum(
            value for value in payload["timing"].values()
            if isinstance(value, float)
        ) + sum(row["evaluation_seconds"] for row in payload["strategies"]) \
            + sum(row["evaluation_seconds"] for row in payload["beam_sweep"])
        _write_json(output, payload)
        print(json.dumps(payload, indent=2, default=_json_default), flush=True)
        return

    # Full retraining plus a new consumer measures the expensive recovery path.
    # A true rebuild can re-encode the retained history under the new
    # vocabulary. Train this expensive recovery baseline on both source-period
    # sequences and target-period histories (with each evaluation target held
    # out), rather than unfairly giving it only the smaller target history.
    target_histories = [history for _, history, _ in eval_t1]
    full_codes_t0 = rq_full.encode(embeddings_t0)
    full_codes_t1 = rq_full.encode(embeddings_t1)
    tok_old, stg_old = prefix_seqs(seqs_t0, full_codes_t0, freeze_depth)
    tok_new, stg_new = prefix_seqs(target_histories, full_codes_t1, freeze_depth)
    tok_t1, stg_t1 = tok_old + tok_new, stg_old + stg_new
    torch.manual_seed(seed + 1000)
    target_model = PrefixGenerator(
        codes[:freeze_depth], d_model=128, n_heads=4, n_layers=3,
    )
    started = time.perf_counter()
    target_model = train_prefix_gen(
        target_model, tok_t1, stg_t1, freeze_depth,
        epochs=args.epochs, device=args.device,
    )
    payload["timing"]["target_generator_seconds"] = time.perf_counter() - started
    payload["timing"]["target_generator_training_sequences"] = len(tok_t1)
    print("evaluating full_retrained_generator", flush=True)
    churn = full_churn
    metrics = evaluate_prefix_routing(
        target_model, eval_t1, rq_full, rq_full, embeddings_t1,
        freeze_depth, args.n_beams, args.device,
    )
    metrics.update({
        "strategy": "full_retrained_generator",
        "mse": rq_full.mse(embeddings_t1),
        "prefix_churn": churn["raw"],
        "prefix_churn_raw": churn["raw"],
        "prefix_churn_centroid_aligned": churn["centroid_aligned"],
        "prefix_churn_assignment_aligned": churn["assignment_aligned"],
        "items_reindexed": int(round(churn["raw"] * n_items)),
        "items_reindexed_centroid_aligned": int(round(
            churn["centroid_aligned"] * n_items
        )),
        "items_reindexed_assignment_aligned": int(round(
            churn["assignment_aligned"] * n_items
        )),
        "consumer_retrained": True,
        "consumer_token_relabeling": "none",
        "token_relabeling_bytes": 0,
        "codebook_update_bytes": _codebook_bytes(rq_full),
    })
    payload["strategies"].append(metrics)
    payload["beam_sweep"].extend(evaluate_beam_sweep(
        target_model, eval_t1, rq_full, rq_full, embeddings_t1,
        freeze_depth,
        [value for value in args.beam_values if value != args.n_beams],
        args.device, "full_retrained_generator",
    ))
    payload["timing"]["total_seconds"] = sum(
        value for value in payload["timing"].values() if isinstance(value, float)
    ) + sum(row["evaluation_seconds"] for row in payload["strategies"]) \
        + sum(row["evaluation_seconds"] for row in payload["beam_sweep"])
    _write_json(output, payload)
    print(json.dumps(payload, indent=2, default=_json_default), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["movielens", "amazon"], required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--amazon-data",
        default="data/amazon/Electronics_5.json.gz",
    )
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument(
        "--amazon-core-passes", type=int, default=10,
        help="Synchronous degree-20 filtering passes; 0 keeps the full public 5-core file",
    )
    parser.add_argument(
        "--max-train-sequences", type=int, default=0,
        help="Deterministic history sample after embedding preparation; 0 keeps all",
    )
    parser.add_argument(
        "--max-eval-sequences", type=int, default=0,
        help="Deterministic target-history sample; 0 keeps all",
    )
    parser.add_argument("--sequence-sample-seed", type=int, default=2026)
    parser.add_argument("--arch", choices=sorted(ARCHITECTURES))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--freeze-depth", type=int, default=2, choices=(1, 2, 3),
        help="Number of stable prefix stages; remaining stages are adapted",
    )
    parser.add_argument("--n-beams", type=int, default=10)
    parser.add_argument(
        "--beam-values",
        default="",
        help="Optional comma-separated additional beam counts evaluated using the same models",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--index-only", action="store_true",
        help=(
            "Run full-catalog quantization, aligned churn, and prefix-bucket "
            "statistics without training or evaluating a prefix consumer"
        ),
    )
    parser.add_argument(
        "--skip-rebuilt-consumer",
        action="store_true",
        help="Skip the costly full-retrain consumer rebuild (useful for beam sweeps)",
    )
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        args.beam_values = sorted({
            int(value) for value in args.beam_values.split(",") if value
        })
    except ValueError:
        parser.error("--beam-values must be comma-separated positive integers")
    if any(value <= 0 for value in args.beam_values):
        parser.error("--beam-values must be comma-separated positive integers")
    if args.amazon_core_passes < 0:
        parser.error("--amazon-core-passes must be nonnegative")
    if args.max_train_sequences < 0 or args.max_eval_sequences < 0:
        parser.error("sequence limits must be nonnegative")
    if not args.prepare_only and (args.arch is None or args.seed is None or args.output is None):
        parser.error("seeded runs require --arch, --seed, and --output")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.prepare_only:
        prepare_dataset(parsed)
    else:
        run_seed(parsed)

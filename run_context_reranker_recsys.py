#!/usr/bin/env python3
"""Prefix routing with a learned context-dot reranker.

This runner replaces the old fixed last-items geometry scorer. The generator
still predicts only stable prefix tokens. The ranker scores items inside the
generated prefix buckets with a learned query vector from the sequence-model
hidden state:

    score(i | X) = q_theta(X)^T v_Q(i)

where v_Q(i) is the decoded item vector under the tokenizer being evaluated.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_generative_prefix import (
    PrefixGenerator,
    RQ,
    pad,
    prefix_seqs,
    train_prefix_gen,
    warm_retrain,
)
from run_wsdm_web_recsys import (
    ARCHITECTURES,
    RANKING_CUTOFFS,
    _apply_prefix_mapping,
    _churn_payload,
    _codebook_bytes,
    _json_default,
    _new_item_full_graft_artifacts,
    _prefix_churn_metrics,
    _sample_sequences,
    _strategy_cost_payload,
    _target_item_split_eval_t1,
    _tier_c_diagnostics,
    _tokenizer_update_seconds_for_strategy,
    _write_json,
    ema_retrain,
    load_cache,
)


class ContextProjection(nn.Module):
    def __init__(self, d_model: int, embedding_dim: int):
        super().__init__()
        self.projection = nn.Linear(d_model, embedding_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden)


def _next_item_examples(sequences):
    histories = []
    targets = []
    for seq in sequences:
        if len(seq) < 2:
            continue
        histories.append(seq[:-1])
        targets.append(seq[-1])
    return histories, np.asarray(targets, dtype=np.int64)


def _prepare_history_tokens(histories, item_codes, freeze_depth: int):
    token_ids, stage_ids = prefix_seqs(histories, item_codes, freeze_depth)
    lengths = np.asarray([len(tokens) for tokens in token_ids], dtype=np.int64)
    max_len = int(lengths.max()) if len(lengths) else 1
    return (
        torch.from_numpy(pad(token_ids, max_len)).long(),
        torch.from_numpy(pad(stage_ids, max_len)).long(),
        torch.from_numpy(lengths).long(),
    )


def _last_hidden(model, token_ids, stage_ids, lengths):
    padding_mask = (
        torch.arange(token_ids.shape[1], device=token_ids.device)[None, :]
        >= lengths[:, None]
    )
    hidden = model(token_ids, stage_ids, padding_mask=padding_mask)
    rows = torch.arange(token_ids.shape[0], device=token_ids.device)
    return hidden[rows, lengths - 1]


def _normalized_vectors(vectors: np.ndarray, normalize: bool) -> torch.Tensor:
    tensor = torch.from_numpy(vectors.astype(np.float32))
    if normalize:
        tensor = F.normalize(tensor, dim=1)
    return tensor


def train_context_projection(
    model,
    projection,
    histories,
    targets,
    history_codes,
    item_vectors,
    freeze_depth: int,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    negatives: int,
    temperature: float,
    device: str,
    normalize: bool,
    finetune_model: bool,
):
    token_ids, stage_ids, lengths = _prepare_history_tokens(
        histories, history_codes, freeze_depth,
    )
    token_ids = token_ids.to(device)
    stage_ids = stage_ids.to(device)
    lengths = lengths.to(device)
    targets_t = torch.from_numpy(targets).long().to(device)
    item_vectors_t = _normalized_vectors(item_vectors, normalize).to(device)

    model = model.to(device)
    projection = projection.to(device)
    model.train(mode=finetune_model)
    projection.train()
    for param in model.parameters():
        param.requires_grad_(finetune_model)

    parameters = list(projection.parameters())
    if finetune_model:
        parameters.extend(model.parameters())
    opt = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    rows = torch.arange(len(targets_t), device=device)

    started = time.perf_counter()
    for _ in range(epochs):
        shuffled = rows[torch.randperm(len(rows), device=device)]
        for start in range(0, len(shuffled), batch_size):
            idx = shuffled[start:start + batch_size]
            hidden = _last_hidden(
                model, token_ids[idx], stage_ids[idx], lengths[idx],
            )
            query = projection(hidden)
            if normalize:
                query = F.normalize(query, dim=1)

            neg = torch.randint(
                0, item_vectors_t.shape[0],
                (max(negatives, 1),),
                device=device,
            )
            candidate_ids = torch.unique(torch.cat([targets_t[idx], neg]))
            candidate_vectors = item_vectors_t[candidate_ids]
            logits = (query @ candidate_vectors.T) / max(temperature, 1e-6)
            labels = (
                candidate_ids[None, :] == targets_t[idx, None]
            ).float().argmax(dim=1)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1.0)
            opt.step()
        sched.step()

    for param in model.parameters():
        param.requires_grad_(True)
    model.eval()
    projection.eval()
    return model, projection, time.perf_counter() - started


@torch.no_grad()
def _context_queries(model, projection, token_ids, stage_ids, lengths, normalize):
    hidden = _last_hidden(model, token_ids, stage_ids, lengths)
    query = projection(hidden)
    if normalize:
        query = F.normalize(query, dim=1)
    return query


@torch.no_grad()
def evaluate_context_routing(
    model,
    projection,
    eval_t1,
    history_rq,
    item_rq,
    embeddings,
    freeze_depth: int,
    n_beams: int,
    device: str,
    *,
    item_prefix_mapping=None,
    candidate_budget=None,
    normalize: bool = True,
    item_codes_override=None,
    item_vectors_override=None,
):
    histories = [history for _, history, _ in eval_t1]
    targets = [target for _, _, target in eval_t1]
    history_codes = history_rq.encode(embeddings)
    if item_codes_override is None:
        item_codes_raw = item_rq.encode(embeddings)
    else:
        item_codes_raw = item_codes_override
    if item_vectors_override is None:
        item_vectors = item_rq.decode_codes(item_codes_raw)
    else:
        item_vectors = item_vectors_override
    item_codes = _apply_prefix_mapping(item_codes_raw, item_prefix_mapping)
    item_vectors_t = _normalized_vectors(item_vectors, normalize).to(device)

    prefix_to_items = {}
    for item_id, row in enumerate(item_codes[:, :freeze_depth]):
        prefix_to_items.setdefault(tuple(row.tolist()), []).append(item_id)

    token_ids, stage_ids, lengths = _prepare_history_tokens(
        histories, history_codes, freeze_depth,
    )
    token_ids = token_ids.to(device)
    stage_ids = stage_ids.to(device)
    lengths = lengths.to(device)
    padding_mask = (
        torch.arange(token_ids.shape[1], device=device)[None, :]
        >= lengths[:, None]
    )

    hits = {cutoff: 0 for cutoff in RANKING_CUTOFFS}
    dcg = {cutoff: 0.0 for cutoff in RANKING_CUTOFFS}
    conditional_hits = {cutoff: 0 for cutoff in RANKING_CUTOFFS}
    conditional_dcg = {cutoff: 0.0 for cutoff in RANKING_CUTOFFS}
    coverage = 0
    uncapped_coverage = 0
    candidate_counts = []
    uncapped_candidate_counts = []
    truncated_queries = 0
    started = time.perf_counter()
    model = model.to(device).eval()
    projection = projection.to(device).eval()

    for start in range(0, len(eval_t1), 64):
        end = min(start + 64, len(eval_t1))
        prefixes, _ = model.generate_prefixes(
            token_ids[start:end], stage_ids[start:end], n_beams=n_beams,
            padding_mask=padding_mask[start:end],
        )
        queries = _context_queries(
            model, projection, token_ids[start:end], stage_ids[start:end],
            lengths[start:end], normalize,
        )
        for local in range(end - start):
            candidate_set = set()
            for prefix in prefixes[local]:
                candidate_set.update(prefix_to_items.get(tuple(prefix), ()))
            uncapped_candidate_counts.append(len(candidate_set))
            if not candidate_set:
                candidate_counts.append(0)
                continue

            target = targets[start + local]
            candidates = np.asarray(sorted(candidate_set), dtype=np.int64)
            uncapped_coverage += int(target in candidate_set)
            candidate_ids_t = torch.from_numpy(candidates).long().to(device)
            scores = (
                item_vectors_t[candidate_ids_t] @ queries[local]
            ).detach().cpu().numpy()
            order = np.argsort(-scores)
            if candidate_budget is not None and len(order) > candidate_budget:
                truncated_queries += 1
                order = order[:candidate_budget]
            candidates = candidates[order]
            candidate_counts.append(len(candidates))
            target_positions = np.flatnonzero(candidates == target)
            target_rank = (
                int(target_positions[0]) if len(target_positions) else None
            )
            covered = target_rank is not None
            coverage += int(covered)
            target_gain = (
                1.0 / math.log2(target_rank + 2)
                if target_rank is not None else 0.0
            )
            for cutoff in RANKING_CUTOFFS:
                hit = target_rank is not None and target_rank < cutoff
                hits[cutoff] += int(hit)
                dcg[cutoff] += target_gain if hit else 0.0
                conditional_hits[cutoff] += int(hit and covered)
                conditional_dcg[cutoff] += (
                    target_gain if hit and covered else 0.0
                )

    elapsed = time.perf_counter() - started
    total = len(eval_t1)
    candidate_counts = np.asarray(candidate_counts, dtype=np.float64)
    uncapped_counts = np.asarray(uncapped_candidate_counts, dtype=np.float64)
    metrics = {
        "scorer": "model_hidden_dot",
        "n_eval": total,
        "n_beams": n_beams,
        "candidate_budget": int(candidate_budget) if candidate_budget else 0,
        "candidate_budget_mode": (
            "model_hidden_dot_topk" if candidate_budget else "uncapped"
        ),
        "candidate_budget_exact_scan_simulation": bool(candidate_budget),
        "routing_coverage": coverage / max(total, 1),
        "uncapped_routing_coverage": uncapped_coverage / max(total, 1),
        "candidate_pool_truncated_fraction": truncated_queries / max(total, 1),
        "evaluation_seconds": elapsed,
        "query_milliseconds": 1000.0 * elapsed / max(total, 1),
    }
    for cutoff in RANKING_CUTOFFS:
        hit_rate = hits[cutoff] / max(total, 1)
        conditional_hit_rate = conditional_hits[cutoff] / max(coverage, 1)
        metrics[f"hit_rate_at_{cutoff}"] = hit_rate
        metrics[f"recall_at_{cutoff}"] = hit_rate
        metrics[f"ndcg_at_{cutoff}"] = dcg[cutoff] / max(total, 1)
        metrics[f"conditional_hit_rate_at_{cutoff}"] = conditional_hit_rate
        metrics[f"conditional_recall_at_{cutoff}"] = conditional_hit_rate
        metrics[f"conditional_ndcg_at_{cutoff}"] = (
            conditional_dcg[cutoff] / max(coverage, 1)
        )
    if len(candidate_counts):
        returned_mean = float(candidate_counts.mean())
        returned_p50 = float(np.percentile(candidate_counts, 50))
        returned_p95 = float(np.percentile(candidate_counts, 95))
        accessed_mean = float(uncapped_counts.mean())
        accessed_p50 = float(np.percentile(uncapped_counts, 50))
        accessed_p95 = float(np.percentile(uncapped_counts, 95))
    else:
        returned_mean = returned_p50 = returned_p95 = 0.0
        accessed_mean = accessed_p50 = accessed_p95 = 0.0
    metrics.update({
        "candidate_count_mean": returned_mean,
        "candidate_count_p50": returned_p50,
        "candidate_count_p95": returned_p95,
        "uncapped_candidate_count_mean": accessed_mean,
        "uncapped_candidate_count_p50": accessed_p50,
        "uncapped_candidate_count_p95": accessed_p95,
        "items_returned_mean": returned_mean,
        "items_returned_p50": returned_p50,
        "items_returned_p95": returned_p95,
        "items_accessed_mean": accessed_mean,
        "items_accessed_p50": accessed_p50,
        "items_accessed_p95": accessed_p95,
    })
    return metrics


def _evaluate_grid(
    model,
    projection,
    eval_t1,
    history_rq,
    item_rq,
    embeddings,
    freeze_depth,
    beam_values,
    budget_values,
    device,
    strategy,
    *,
    item_prefix_mapping=None,
    normalize=True,
):
    rows = []
    for n_beams in beam_values:
        for budget in budget_values:
            row = evaluate_context_routing(
                model, projection, eval_t1, history_rq, item_rq, embeddings,
                freeze_depth, n_beams, device,
                item_prefix_mapping=item_prefix_mapping,
                candidate_budget=budget,
                normalize=normalize,
            )
            row.update({"strategy": strategy, "candidate_grid": True})
            rows.append(row)
    return rows


def run_seed(args) -> None:
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output}; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)

    embeddings_t0, embeddings_t1, seqs_t0, eval_t1, metadata = load_cache(
        Path(args.cache),
    )
    if args.run_train_sequence_limit:
        seqs_t0 = _sample_sequences(
            seqs_t0, args.run_train_sequence_limit,
            args.sequence_sample_seed + args.seed,
        )
    if args.run_eval_sequence_limit:
        eval_t1 = _sample_sequences(
            eval_t1, args.run_eval_sequence_limit,
            args.sequence_sample_seed + 1000 + args.seed,
        )

    codes = ARCHITECTURES[args.arch]
    freeze_depth = args.freeze_depth
    n_items = len(embeddings_t0)
    payload = {
        "schema_version": 1,
        "runner": "run_context_reranker_recsys.py",
        "dataset": metadata,
        "configuration": {
            "arch": args.arch,
            "codebook_sizes": codes,
            "freeze_depth": freeze_depth,
            "seed": args.seed,
            "epochs": args.epochs,
            "scorer_epochs": args.scorer_epochs,
            "n_beams": args.n_beams,
            "candidate_budget_values": args.candidate_budget_values,
            "candidate_grid_beam_values": args.candidate_grid_beam_values,
            "run_train_sequence_limit": args.run_train_sequence_limit,
            "run_eval_sequence_limit": args.run_eval_sequence_limit,
            "device": args.device,
            "scorer": "model_hidden_dot",
            "scorer_negatives": args.scorer_negatives,
            "scorer_temperature": args.scorer_temperature,
            "scorer_normalize": args.scorer_normalize,
            "scorer_finetune_model": args.scorer_finetune_model,
        },
        "timing": {},
        "diagnostics": {},
        "context_reranker_rows": [],
        "context_reranker_grid": [],
        "context_reranker_target_item_split_rows": [],
    }
    _write_json(output, payload)

    started = time.perf_counter()
    rq_source = RQ(4, codes, embeddings_t0.shape[1]).fit(
        embeddings_t0, seed=args.seed,
    )
    payload["timing"]["source_codebook_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    rq_stratified = warm_retrain(
        rq_source, embeddings_t1, freeze_depth, seed=args.seed,
    )
    payload["timing"]["suffix_update_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    rq_warm_full = warm_retrain(rq_source, embeddings_t1, 0, seed=args.seed)
    payload["timing"]["warm_full_codebook_seconds"] = (
        time.perf_counter() - started
    )
    warm_full_churn = _prefix_churn_metrics(
        rq_source, rq_warm_full, embeddings_t1, freeze_depth,
    )

    started = time.perf_counter()
    rq_ema = ema_retrain(
        rq_source, embeddings_t1,
        decay=args.ema_decay, n_iter=args.ema_iterations,
    )
    payload["timing"]["ema_codebook_seconds"] = time.perf_counter() - started
    ema_churn = _prefix_churn_metrics(
        rq_source, rq_ema, embeddings_t1, freeze_depth,
    )

    started = time.perf_counter()
    rq_full = RQ(4, codes, embeddings_t1.shape[1]).fit(
        embeddings_t1, seed=args.seed + 500,
    )
    payload["timing"]["full_codebook_seconds"] = time.perf_counter() - started
    full_churn, full_mappings = _prefix_churn_metrics(
        rq_source, rq_full, embeddings_t1, freeze_depth,
        return_mappings=True,
    )
    zero_churn = {"raw": 0.0, "centroid_aligned": 0.0,
                  "assignment_aligned": 0.0}

    payload["diagnostics"] = _tier_c_diagnostics(
        rq_source, rq_stratified, rq_full,
        embeddings_t0, embeddings_t1, freeze_depth,
        seqs_t0, eval_t1,
    )
    _write_json(output, payload)

    source_codes_t0 = rq_source.encode(embeddings_t0)
    tok_t0, stg_t0 = prefix_seqs(seqs_t0, source_codes_t0, freeze_depth)
    torch.manual_seed(args.seed)
    model = PrefixGenerator(
        codes[:freeze_depth], d_model=128, n_heads=4, n_layers=3,
    )
    started = time.perf_counter()
    model = train_prefix_gen(
        model, tok_t0, stg_t0, freeze_depth,
        epochs=args.epochs, device=args.device,
    )
    payload["timing"]["source_generator_seconds"] = time.perf_counter() - started

    train_histories, train_targets = _next_item_examples(seqs_t0)
    source_item_vectors = rq_source.decode_codes(source_codes_t0)
    projection = ContextProjection(model.d_model, embeddings_t0.shape[1])
    torch.manual_seed(args.seed + 2027)
    model, projection, scorer_seconds = train_context_projection(
        model, projection,
        train_histories, train_targets,
        source_codes_t0, source_item_vectors, freeze_depth,
        epochs=args.scorer_epochs,
        lr=args.scorer_lr,
        batch_size=args.scorer_batch_size,
        negatives=args.scorer_negatives,
        temperature=args.scorer_temperature,
        device=args.device,
        normalize=args.scorer_normalize,
        finetune_model=args.scorer_finetune_model,
    )
    payload["timing"]["source_context_scorer_seconds"] = scorer_seconds
    payload["timing"]["source_context_scorer_training_sequences"] = len(
        train_targets,
    )
    _write_json(output, payload)

    strategies = [
        ("frozen", rq_source, zero_churn, None),
        ("stratified", rq_stratified, zero_churn, None),
        ("warm_start_full_old_model", rq_warm_full, warm_full_churn, None),
        ("ema_streaming_vq_old_model", rq_ema, ema_churn, None),
        ("full_old_model", rq_full, full_churn, None),
        (
            "full_old_model_centroid_relabel", rq_full, full_churn,
            full_mappings["centroid_hungarian"],
        ),
        (
            "full_old_model_assignment_relabel", rq_full, full_churn,
            full_mappings["assignment_optimal"],
        ),
    ]

    if args.fix1_target_split_only:
        payload["configuration"]["mode"] = "fix1_target_split_only"
        payload["configuration"]["fix1_target_splits"] = args.fix1_target_splits
        split_eval_t1 = _target_item_split_eval_t1(
            embeddings_t0, embeddings_t1, eval_t1,
        )
        if args.fix1_target_splits:
            requested_splits = {
                value for value in args.fix1_target_splits.split(",")
                if value
            }
            unknown_splits = requested_splits - set(split_eval_t1)
            if unknown_splits:
                raise ValueError(
                    "unknown target splits: "
                    + ", ".join(sorted(unknown_splits))
                )
            split_eval_t1 = {
                name: rows for name, rows in split_eval_t1.items()
                if name in requested_splits
            }
        new_item_graft = _new_item_full_graft_artifacts(
            rq_source, rq_full, embeddings_t0, embeddings_t1,
        )
        new_item_graft_refresh_existing = _new_item_full_graft_artifacts(
            rq_source, rq_full, embeddings_t0, embeddings_t1,
            refresh_existing=True,
        )

        fix1_strategies = [
            {
                "name": "frozen",
                "item_rq": rq_source,
                "churn": zero_churn,
            },
            {
                "name": "stratified",
                "item_rq": rq_stratified,
                "churn": zero_churn,
            },
            {
                "name": "new_item_full_graft_static_existing",
                "item_rq": rq_full,
                "churn": zero_churn,
                "item_codes_override": new_item_graft["codes"],
                "item_vectors_override": new_item_graft["decoded"],
                "mse": new_item_graft["mse"],
                "new_item_count": new_item_graft["new_item_count"],
                "new_item_fraction": new_item_graft["new_item_fraction"],
                "tokenizer_update_seconds": payload["timing"][
                    "full_codebook_seconds"
                ],
                "codebook_update_bytes": _codebook_bytes(rq_full),
            },
            {
                "name": "new_item_full_graft_frozen_existing",
                "item_rq": rq_full,
                "churn": zero_churn,
                "item_codes_override": new_item_graft_refresh_existing[
                    "codes"
                ],
                "item_vectors_override": new_item_graft_refresh_existing[
                    "decoded"
                ],
                "mse": new_item_graft_refresh_existing["mse"],
                "new_item_count": new_item_graft_refresh_existing[
                    "new_item_count"
                ],
                "new_item_fraction": new_item_graft_refresh_existing[
                    "new_item_fraction"
                ],
                "tokenizer_update_seconds": payload["timing"][
                    "full_codebook_seconds"
                ],
                "codebook_update_bytes": _codebook_bytes(rq_full),
            },
            {
                "name": "full_old_model",
                "item_rq": rq_full,
                "churn": full_churn,
            },
            {
                "name": "full_old_model_assignment_relabel",
                "item_rq": rq_full,
                "churn": full_churn,
                "item_prefix_mapping": full_mappings["assignment_optimal"],
            },
        ]
        for strategy in fix1_strategies:
            name = strategy["name"]
            item_rq = strategy["item_rq"]
            churn = strategy["churn"]
            item_prefix_mapping = strategy.get("item_prefix_mapping")
            for split_name, split_rows in split_eval_t1.items():
                if not split_rows:
                    continue
                print(
                    f"evaluating context reranker {name} "
                    f"target_split={split_name}",
                    flush=True,
                )
                metrics = evaluate_context_routing(
                    model, projection, split_rows, rq_source, item_rq,
                    embeddings_t1, freeze_depth, args.n_beams, args.device,
                    item_prefix_mapping=item_prefix_mapping,
                    normalize=args.scorer_normalize,
                    item_codes_override=strategy.get("item_codes_override"),
                    item_vectors_override=strategy.get(
                        "item_vectors_override"
                    ),
                )
                codebook_update_bytes = (
                    strategy["codebook_update_bytes"]
                    if "codebook_update_bytes" in strategy else
                    0 if name == "frozen" else
                    _codebook_bytes(item_rq, range(freeze_depth, 4))
                    if name == "stratified" else _codebook_bytes(item_rq)
                )
                metrics.update({
                    "strategy": name,
                    "target_item_split": split_name,
                    "target_item_split_n_eval": len(split_rows),
                    "target_item_split_fraction": (
                        len(split_rows) / max(len(eval_t1), 1)
                    ),
                    "mse": strategy.get("mse", item_rq.mse(embeddings_t1)),
                    **_churn_payload(churn, n_items),
                    **_strategy_cost_payload(
                        codebook_update_bytes=codebook_update_bytes,
                        tokenizer_update_seconds=(
                            strategy["tokenizer_update_seconds"]
                            if "tokenizer_update_seconds" in strategy else
                            _tokenizer_update_seconds_for_strategy(
                                name, payload["timing"],
                            )
                        ),
                    ),
                })
                if "new_item_count" in strategy:
                    metrics["new_item_graft_count"] = strategy["new_item_count"]
                    metrics["new_item_graft_fraction"] = strategy[
                        "new_item_fraction"
                    ]
                payload["context_reranker_target_item_split_rows"].append(
                    metrics,
                )
                _write_json(output, payload)

        payload["timing"]["total_seconds"] = sum(
            value for value in payload["timing"].values()
            if isinstance(value, float)
        ) + sum(
            row["evaluation_seconds"]
            for row in payload["context_reranker_target_item_split_rows"]
        )
        _write_json(output, payload)
        print(json.dumps(payload, indent=2, default=_json_default), flush=True)
        return

    for name, item_rq, churn, item_prefix_mapping in strategies:
        print(f"evaluating context reranker {name}", flush=True)
        metrics = evaluate_context_routing(
            model, projection, eval_t1, rq_source, item_rq, embeddings_t1,
            freeze_depth, args.n_beams, args.device,
            item_prefix_mapping=item_prefix_mapping,
            normalize=args.scorer_normalize,
        )
        codebook_update_bytes = (
            0 if name == "frozen" else
            _codebook_bytes(item_rq, range(freeze_depth, 4))
            if name == "stratified" else _codebook_bytes(item_rq)
        )
        metrics.update({
            "strategy": name,
            "mse": item_rq.mse(embeddings_t1),
            **_churn_payload(churn, n_items),
            **_strategy_cost_payload(
                codebook_update_bytes=codebook_update_bytes,
                tokenizer_update_seconds=_tokenizer_update_seconds_for_strategy(
                    name, payload["timing"],
                ),
            ),
        })
        payload["context_reranker_rows"].append(metrics)
        payload["context_reranker_grid"].extend(_evaluate_grid(
            model, projection, eval_t1, rq_source, item_rq, embeddings_t1,
            freeze_depth, args.candidate_grid_beam_values,
            args.candidate_budget_values, args.device, name,
            item_prefix_mapping=item_prefix_mapping,
            normalize=args.scorer_normalize,
        ))
        _write_json(output, payload)

    payload["timing"]["total_seconds"] = sum(
        value for value in payload["timing"].values()
        if isinstance(value, float)
    ) + sum(
        row["evaluation_seconds"] for row in payload["context_reranker_rows"]
    ) + sum(
        row["evaluation_seconds"] for row in payload["context_reranker_grid"]
    )
    _write_json(output, payload)
    print(json.dumps(payload, indent=2, default=_json_default), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--arch", choices=sorted(ARCHITECTURES), required=True)
    parser.add_argument("--freeze-depth", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--scorer-epochs", type=int, default=20)
    parser.add_argument("--scorer-lr", type=float, default=3e-4)
    parser.add_argument("--scorer-batch-size", type=int, default=128)
    parser.add_argument("--scorer-negatives", type=int, default=2048)
    parser.add_argument("--scorer-temperature", type=float, default=0.07)
    parser.add_argument("--scorer-normalize", action="store_true", default=True)
    parser.add_argument("--no-scorer-normalize", dest="scorer_normalize",
                        action="store_false")
    parser.add_argument("--scorer-finetune-model", action="store_true")
    parser.add_argument("--n-beams", type=int, default=10)
    parser.add_argument("--candidate-budget-values", default="")
    parser.add_argument("--candidate-grid-beam-values", default="")
    parser.add_argument("--run-train-sequence-limit", type=int, default=0)
    parser.add_argument("--run-eval-sequence-limit", type=int, default=0)
    parser.add_argument("--sequence-sample-seed", type=int, default=2026)
    parser.add_argument("--ema-decay", type=float, default=0.95)
    parser.add_argument("--ema-iterations", type=int, default=20)
    parser.add_argument(
        "--fix1-target-split-only",
        action="store_true",
        help=(
            "Train the source prefix generator/scorer, then evaluate context "
            "reranker rows separately for all, carried-source-nonzero, and "
            "new-source-zero target items."
        ),
    )
    parser.add_argument(
        "--fix1-target-splits",
        default="",
        help=(
            "Optional comma-separated subset of FIX-1 splits to evaluate. "
            "Valid values: all, carried_source_nonzero, "
            "new_source_zero_target_nonzero."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        args.candidate_budget_values = sorted({
            int(value) for value in args.candidate_budget_values.split(",")
            if value
        })
    except ValueError:
        parser.error("--candidate-budget-values must be comma-separated positive integers")
    try:
        args.candidate_grid_beam_values = sorted({
            int(value) for value in args.candidate_grid_beam_values.split(",")
            if value
        })
    except ValueError:
        parser.error("--candidate-grid-beam-values must be comma-separated positive integers")
    if any(value <= 0 for value in args.candidate_budget_values):
        parser.error("--candidate-budget-values must be positive")
    if any(value <= 0 for value in args.candidate_grid_beam_values):
        parser.error("--candidate-grid-beam-values must be positive")
    if args.scorer_epochs <= 0:
        parser.error("--scorer-epochs must be positive")
    if args.scorer_negatives <= 0:
        parser.error("--scorer-negatives must be positive")
    return args


if __name__ == "__main__":
    run_seed(parse_args())

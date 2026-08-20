"""Nested OOF evidence stacking for Task 1.

V13 trained its evidence reranker from in-sample first-stage predictions and
V16 supplied only one small (325-post) OOF slice.  V20 creates first-stage
predictions for every outer-training post with four inner, user-disjoint
folds.  The reranker is fitted on three OOF folds, calibrated on the fourth,
and evaluated exactly once on the untouched outer user holdout.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from baseline import _post_phrase_f1
from configs.config import config
from datasets.dataset import SuicideRiskDataset
from inference.task1_evidence_v4 import (
    apply_evidence_policy,
    correct_risk_only,
    decode_model_evidence,
    load_evidence_calibration,
)
from models.multitask_model import SuicideRiskMultiTaskModel, get_optimizer_parameters
from trainer.task1_evidence_reranker_v13 import (
    PairDataset,
    _dedupe_ranked,
    _make_model,
    _pair_rows,
    _pool_records,
    _post_score_map,
    _score,
)
from trainer.task1_seed_ensemble_v14 import _criterion
from trainer.train import _loader, _move
from utils.seed import seed_everything
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_oof_stack_v20"
RERANKER = OUTPUT / "reranker.pt"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "task1-nested-oof-stack-v20"
INNER_FOLDS = 4
FIRST_STAGE_EPOCHS = 3
RERANKER_EPOCHS = 4


def _fold_paths(fold: int) -> tuple[Path, Path]:
    return OUTPUT / f"inner_fold{fold}_model.pt", OUTPUT / f"inner_fold{fold}_raw.pt"


@torch.no_grad()
def _collect_first_stage(model, dataset, indices, device, description):
    rows = []
    loader = _loader(dataset, indices, False)
    cursor = 0
    model.eval()
    for batch in tqdm(loader, desc=description):
        output = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device)
        )
        probability = torch.softmax(output["risk_logits"], -1).cpu().numpy()
        for local in range(len(batch["row_id"])):
            source = dataset.data[int(indices[cursor])]
            risk = correct_risk_only(
                batch["texts"][local], int(np.argmax(probability[local]))
            )
            rows.append({
                "global_index": int(indices[cursor]),
                "row_id": str(batch["row_id"][local]),
                "user": str(source["anon_user_id"]),
                "text": batch["texts"][local],
                "offsets": batch["offset_mappings"][local],
                "truth": int(batch["risk_labels"][local]),
                "gold": list(batch["evidences"][local]),
                "risk": int(risk),
                "old_probability": probability[local],
                "start": output["start_logits"][local].cpu(),
                "end": output["end_logits"][local].cpu(),
            })
            cursor += 1
    return rows


def _train_inner_fold(fold, fit_idx, oof_idx, dataset, labels, device):
    checkpoint, raw_file = _fold_paths(fold)
    if raw_file.exists():
        saved = torch.load(raw_file, map_location="cpu", weights_only=False)
        if (saved.get("training_version") == TRAINING_VERSION
                and np.array_equal(np.asarray(saved["oof_idx"]), oof_idx)):
            print(f"V20 inner fold {fold}: resumed {len(oof_idx)} OOF posts", flush=True)
            return saved["records"]

    seed_everything(config.SEED + 2020 + fold)
    model = SuicideRiskMultiTaskModel().to(device)
    # The 8 GB RTX 5060 can hold this batch shape without activation
    # checkpointing.  Disabling it avoids recomputing all encoder layers in
    # backward and roughly halves wall-clock time for the four nested folds.
    model.backbone.encoder.gradient_checkpointing_disable()
    optimizer = AdamW(get_optimizer_parameters(model), weight_decay=config.WEIGHT_DECAY)
    criterion = _criterion(dataset, fit_idx, labels, device)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    loader = _loader(dataset, fit_idx, True)
    history = []
    print(
        f"V20 inner fold {fold}: train={len(fit_idx)} OOF={len(oof_idx)}",
        flush=True,
    )
    for epoch in range(1, FIRST_STAGE_EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        progress = tqdm(loader, desc=f"V20 fold {fold} epoch {epoch}/{FIRST_STAGE_EPOCHS}")
        for step, batch in enumerate(progress, 1):
            batch = _move(batch, device)
            with torch.autocast(device_type="cuda", enabled=True):
                loss = criterion(
                    model(batch["input_ids"], batch["attention_mask"]), batch
                )["loss"] / config.GRADIENT_ACCUMULATION
            scaler.scale(loss).backward()
            losses.append(float(loss.detach()) * config.GRADIENT_ACCUMULATION)
            if step % config.GRADIENT_ACCUMULATION == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        history.append(float(np.mean(losses)))
        print(
            f"V20 inner fold={fold} epoch={epoch} loss={history[-1]:.4f}",
            flush=True,
        )
    torch.save(model.state_dict(), checkpoint)
    records = _collect_first_stage(
        model, dataset, oof_idx, device, f"V20 fold {fold} OOF inference"
    )
    torch.save({
        "training_version": TRAINING_VERSION,
        "fit_idx": np.asarray(fit_idx), "oof_idx": np.asarray(oof_idx),
        "history": history, "records": records,
    }, raw_file)
    del model, optimizer, criterion, loader
    torch.cuda.empty_cache()
    return records


def _nested_oof(dataset, outer_train, labels, groups, device):
    local_labels = labels[outer_train]
    local_groups = groups[outer_train]
    splitter = StratifiedGroupKFold(
        n_splits=INNER_FOLDS, shuffle=True, random_state=config.SEED + 2020
    )
    records = []
    fold_membership = {}
    for fold, (fit_local, oof_local) in enumerate(
        splitter.split(np.zeros(len(outer_train)), local_labels, local_groups)
    ):
        fit_idx = outer_train[fit_local]
        oof_idx = outer_train[oof_local]
        fold_rows = _train_inner_fold(
            fold, fit_idx, oof_idx, dataset, labels, device
        )
        records.extend(fold_rows)
        fold_membership.update({int(index): fold for index in oof_idx})
    records.sort(key=lambda row: row["global_index"])
    if {row["global_index"] for row in records} != set(map(int, outer_train)):
        raise ValueError("V20 did not create exactly one OOF prediction per outer-training post")
    return records, np.asarray([
        fold_membership[row["global_index"]] for row in records
    ], dtype=np.int64)


def _baseline_evidence(record, calibration):
    phrases = decode_model_evidence(
        record["text"], record["offsets"], record["start"], record["end"],
        threshold=float(calibration["threshold"]),
        max_tokens=int(calibration["max_tokens"]),
        end_policy=str(calibration["end_policy"]), limit=5,
    )
    return apply_evidence_policy(
        record["text"], int(record["risk"]), phrases,
        policy=str(calibration["cue_policy"]), topk=int(calibration["topk"]),
    )


def _candidate_scores(rows, scores):
    return _post_score_map(rows, scores)


def _normalise(value):
    return " ".join(str(value).casefold().split())


def _hybrid_prediction(post, values, baseline, parameters):
    """Metric-aware conservative reranking selected only on inner OOF data."""
    if post["risk"] == config.RISK_LABELS["Indicator"]:
        return []
    reranked = _dedupe_ranked(
        post, values, parameters["topk"], parameters["threshold"]
    )
    if parameters["mode"] == "rerank":
        return reranked
    maximum = max(values, default=0.0)
    if parameters["mode"] == "gated_replace":
        return reranked if reranked and maximum >= parameters["gate"] else baseline
    if parameters["mode"] == "prune_baseline":
        scored = {
            _normalise(candidate["phrase"]): float(score)
            for score, candidate in zip(values, post["candidates"])
        }
        kept = []
        for phrase in baseline:
            normal = _normalise(phrase)
            related = [score for key, score in scored.items() if normal in key or key in normal]
            if related and max(related) >= parameters["threshold"]:
                kept.append(phrase)
        return kept or (reranked[:1] if maximum >= parameters["gate"] else baseline)
    raise ValueError(parameters["mode"])


def _parameter_grid(posts, rows, pair_scores, post_indices, evidence_calibration):
    grouped = _candidate_scores(rows, pair_scores)
    baseline = {
        int(index): list(posts[int(index)]["baseline_evidence"])
        for index in post_indices
    }
    candidates = []
    for mode in ("rerank", "gated_replace", "prune_baseline"):
        for topk in (1, 2, 3, 4):
            for threshold in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
                gates = (0.0,) if mode == "rerank" else (0.50, 0.60, 0.70, 0.80, 0.90)
                for gate in gates:
                    parameters = {
                        "mode": mode, "topk": topk,
                        "threshold": threshold, "gate": gate,
                    }
                    scores = []
                    for index in post_indices:
                        index = int(index); post = posts[index]
                        values = [
                            grouped[index].get(candidate_index, 0.0)
                            for candidate_index in range(len(post["candidates"]))
                        ]
                        evidence = _hybrid_prediction(
                            post, values, baseline[index], parameters
                        )
                        scores.append(_post_phrase_f1(evidence, post["gold"]))
                    candidates.append({
                        **parameters, "phrase_f1": float(np.mean(scores))
                    })
    return sorted(candidates, key=lambda row: row["phrase_f1"], reverse=True)


def _train_reranker(
    posts, fold_membership, tokenizer, device, evidence_calibration,
    row_builder=_pair_rows, checkpoint_path=RERANKER,
):
    calibration_fold = INNER_FOLDS - 1
    fit_posts = np.flatnonzero(fold_membership != calibration_fold)
    calibration_posts = np.flatnonzero(fold_membership == calibration_fold)
    fit_rows = row_builder(posts, fit_posts, training=True)
    calibration_rows = row_builder(posts, calibration_posts, training=False)
    fit_dataset = PairDataset(fit_rows, tokenizer)
    calibration_dataset = PairDataset(calibration_rows, tokenizer)
    model = _make_model().to(device)
    positives = float(sum(row["label"] for row in fit_rows))
    negatives = float(len(fit_rows) - positives)
    pos_weight = torch.tensor(
        min(6.0, max(1.0, negatives / max(positives, 1.0))), device=device
    )
    backbone = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith("deberta.") and parameter.requires_grad
    ]
    head = [
        parameter for name, parameter in model.named_parameters()
        if not name.startswith("deberta.") and parameter.requires_grad
    ]
    optimizer = AdamW([
        {"params": backbone, "lr": 3e-6},
        {"params": head, "lr": 2e-5},
    ], weight_decay=config.WEIGHT_DECAY)
    loader = DataLoader(fit_dataset, batch_size=4, shuffle=True, num_workers=0)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    best = None; history = []
    for epoch in range(1, RERANKER_EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(
            tqdm(loader, desc=f"V20 reranker epoch {epoch}/{RERANKER_EPOCHS}"), 1
        ):
            with torch.autocast(device_type="cuda", enabled=True):
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                ).logits.squeeze(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, batch["label"].to(device), pos_weight=pos_weight
                ) / 4.0
            scaler.scale(loss).backward(); losses.append(float(loss.detach()) * 4.0)
            if step % 4 == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        pair_scores = _score(
            model, calibration_dataset, device, "V20 inner calibration"
        )
        selected = _parameter_grid(
            posts, calibration_rows, pair_scores, calibration_posts,
            evidence_calibration,
        )[0]
        row = {
            "epoch": epoch, "loss": float(np.mean(losses)), **selected,
        }
        history.append(row)
        print(
            f"V20 reranker epoch={epoch} inner_phrase={row['phrase_f1']:.4f} "
            f"mode={row['mode']} topk={row['topk']} "
            f"threshold={row['threshold']:.2f} gate={row['gate']:.2f}",
            flush=True,
        )
        if best is None or row["phrase_f1"] > best["phrase_f1"]:
            best = row
            torch.save(model.state_dict(), checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model, best, history, len(fit_rows), len(calibration_rows)


def _outer_records(dataset, outer_valid, device):
    raw = torch.load(
        config.OUTPUT_DIR / "task1_evidence_v4" / "fold0_raw.pt",
        map_location="cpu", weights_only=False,
    )
    if not np.array_equal(np.asarray(raw["valid_idx"]), outer_valid):
        raise ValueError("V20 outer holdout differs from the stable V4 holdout")
    # Use the original first-stage risk only.  The evidence reranker's input
    # distribution must match the inner OOF models, which have no V2/TF-IDF
    # stacking attached to them.
    records = raw["records"]
    for record in records:
        record["risk"] = correct_risk_only(
            record["text"], int(np.argmax(record["old_probability"]))
        )
    return records


def _bootstrap(groups, truth, baseline_risk, baseline_phrase, new_phrase, seed):
    unique = np.unique(groups); rng = np.random.default_rng(seed); deltas = []
    risk_f1 = f1_score(truth, baseline_risk, average="weighted", zero_division=0)
    for _ in range(4000):
        sampled_users = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([
            np.flatnonzero(groups == user) for user in sampled_users
        ])
        sampled_risk = f1_score(
            truth[indices], baseline_risk[indices],
            average="weighted", zero_division=0,
        )
        deltas.append(
            task1_score(sampled_risk, float(new_phrase[indices].mean()))
            - task1_score(sampled_risk, float(baseline_phrase[indices].mean()))
        )
    values = np.asarray(deltas)
    return {
        "fixed_risk_f1": float(risk_f1),
        "mean_delta": float(values.mean()),
        "p05_delta": float(np.quantile(values, 0.05)),
        "p95_delta": float(np.quantile(values, 0.95)),
        "positive_fraction": float((values > 0).mean()),
    }


def train_task1_oof_stack_v20():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("V20 nested OOF stacking requires CUDA")
    seed_everything(config.SEED + 2020)
    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    labels = np.asarray([int(row["risk_label"]) for row in dataset.data])
    groups = np.asarray([str(row["anon_user_id"]) for row in dataset.data])
    outer_train, outer_valid = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(dataset)), labels, groups))
    device = torch.device("cuda")
    inner_records, fold_membership = _nested_oof(
        dataset, outer_train, labels, groups, device
    )
    outer_records = _outer_records(dataset, outer_valid, device)
    evidence_calibration = load_evidence_calibration()
    if evidence_calibration is None:
        raise FileNotFoundError("Stable V4 evidence calibration is required")
    inner_posts = _pool_records(inner_records, use_truth=False)
    outer_posts = _pool_records(outer_records, use_truth=False)
    # _pool_records intentionally keeps only reranker fields.  Cache the
    # stable decoder output before offsets/logits are discarded so the gated
    # policy can compare against the exact baseline without a second model.
    for post, record in zip(inner_posts, inner_records):
        post["baseline_evidence"] = _baseline_evidence(
            record, evidence_calibration
        )
    for post, record in zip(outer_posts, outer_records):
        post["baseline_evidence"] = _baseline_evidence(
            record, evidence_calibration
        )
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME, use_fast=True, local_files_only=True
    )
    model, selected, history, fit_pairs, calibration_pairs = _train_reranker(
        inner_posts, fold_membership, tokenizer, device, evidence_calibration
    )

    outer_rows = _pair_rows(
        outer_posts, np.arange(len(outer_posts)), training=False
    )
    outer_dataset = PairDataset(outer_rows, tokenizer)
    outer_pair_scores = _score(model, outer_dataset, device, "V20 outer strict")
    grouped = _candidate_scores(outer_rows, outer_pair_scores)
    baseline_phrase = []; candidate_phrase = []; predictions = []
    for index, post in enumerate(outer_posts):
        baseline = list(post["baseline_evidence"])
        values = [
            grouped[index].get(candidate_index, 0.0)
            for candidate_index in range(len(post["candidates"]))
        ]
        evidence = _hybrid_prediction(post, values, baseline, selected)
        predictions.append(evidence)
        baseline_phrase.append(_post_phrase_f1(baseline, post["gold"]))
        candidate_phrase.append(_post_phrase_f1(evidence, post["gold"]))
    baseline_phrase = np.asarray(baseline_phrase, dtype=np.float32)
    candidate_phrase = np.asarray(candidate_phrase, dtype=np.float32)
    truth = labels[outer_valid]
    risk = np.asarray([post["risk"] for post in outer_posts], dtype=np.int64)
    risk_f1 = float(f1_score(truth, risk, average="weighted", zero_division=0))
    baseline_task1 = task1_score(risk_f1, float(baseline_phrase.mean()))
    candidate_task1 = task1_score(risk_f1, float(candidate_phrase.mean()))
    bootstrap = _bootstrap(
        groups[outer_valid], truth, risk, baseline_phrase, candidate_phrase,
        config.SEED + 2021,
    )
    adopted = bool(
        candidate_task1 >= baseline_task1 + 0.005
        and bootstrap["positive_fraction"] >= 0.80
    )
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation_scope": (
            "outer user-disjoint fold; reranker trained on complete four-fold "
            "nested OOF candidates; parameters selected on one inner fold"
        ),
        "inner_oof_posts": len(inner_posts),
        "fit_pairs": fit_pairs, "calibration_pairs": calibration_pairs,
        "history": history, "selected": selected,
        "baseline": {
            "risk_f1": risk_f1,
            "phrase_f1": float(baseline_phrase.mean()),
            "task1": baseline_task1,
        },
        "candidate": {
            "risk_f1": risk_f1,
            "phrase_f1": float(candidate_phrase.mean()),
            "task1": candidate_task1,
            "improved_posts": int((candidate_phrase > baseline_phrase).sum()),
            "worsened_posts": int((candidate_phrase < baseline_phrase).sum()),
            "risk_confusion": confusion_matrix(
                truth, risk, labels=np.arange(config.NUM_RISK_CLASSES)
            ).tolist(),
        },
        "user_cluster_bootstrap": bootstrap,
        "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({
        "training_version": TRAINING_VERSION,
        "adopted": adopted,
        **{key: selected[key] for key in ("mode", "topk", "threshold", "gate")},
        "strict_task1": candidate_task1,
        "bootstrap_positive_fraction": bootstrap["positive_fraction"],
    }, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    train_task1_oof_stack_v20()

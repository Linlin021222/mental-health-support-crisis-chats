"""Contextual cross-encoder reranking for Task 1 evidence candidates.

The first-stage span model has high candidate recall but poor candidate
selection.  V13 trains a second-stage DeBERTa on candidate/context pairs and
keeps the final strict users completely unseen by both stages.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from analyze_task1_evidence_v4 import _collect_old, _cue_cache, _decoder_grid, _evaluate
from baseline import _post_phrase_f1
from configs.config import config
from datasets.dataset import SuicideRiskDataset
from inference.task1_evidence_v4 import (
    apply_evidence_policy, correct_risk_only, cue_phrases,
    decode_model_evidence, load_evidence_calibration,
)
from utils.seed import seed_everything
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_evidence_reranker_v13"
TRAIN_RAW = OUTPUT / "train_raw.pt"
CHECKPOINT = OUTPUT / "strict_model.pt"
CALIBRATION = OUTPUT / "calibration.json"
RESULTS = OUTPUT / "results.json"
TRAINING_VERSION = "task1-context-reranker-v13"
MAX_CANDIDATES = 16
MAX_LENGTH = 192


def _normalise(text):
    return " ".join(str(text).casefold().split())


def _candidate_pool(record, risk_id):
    """High-recall pool, ranked initially by cross-decoder agreement."""
    votes = Counter(); original = {}
    for threshold in (0.40, 0.50, 0.55, 0.60, 0.65):
        for max_tokens in (8, 12, 16):
            for end_policy in ("nearest", "best"):
                phrases = decode_model_evidence(
                    record["text"], record["offsets"], record["start"], record["end"],
                    threshold=threshold, max_tokens=max_tokens,
                    end_policy=end_policy, limit=8,
                )
                for phrase in phrases:
                    for part in str(phrase).split(";"):
                        part = part.strip(); key = _normalise(part)
                        if key:
                            votes[key] += 1; original.setdefault(key, part)
    # Add both severity-conditioned and hierarchy-independent cues.  A cue is
    # a candidate, never an automatically accepted prediction.
    all_cues = cue_phrases(record["text"], int(risk_id), "current_first")
    for label_id in (1, 2, 3):
        all_cues += cue_phrases(record["text"], label_id, "predicted_extended_first")
    for phrase in all_cues:
        for part in str(phrase).split(";"):
            part = part.strip(); key = _normalise(part)
            if key:
                votes[key] += 4; original.setdefault(key, part)
    ranked = sorted(
        votes, key=lambda key: (votes[key], len(key.split()), len(key)), reverse=True
    )
    # Keep non-identical containment variants: the scorer's 3x length rule can
    # prefer the shorter or longer member depending on the unseen annotation.
    return [{"phrase": original[key], "votes": int(votes[key])} for key in ranked[:MAX_CANDIDATES]]


def _context(text, phrase, radius=420):
    match = re.search(re.escape(phrase), text, flags=re.IGNORECASE)
    if match is None:
        return text[: 2 * radius]
    start = max(0, match.start() - radius); end = min(len(text), match.end() + radius)
    return text[start:end]


def _prompt(risk_id, phrase, votes):
    risk = config.ID2RISK.get(int(risk_id), "Unknown")
    vote_bucket = "high" if votes >= 15 else "medium" if votes >= 5 else "low"
    return f"Risk level: {risk}. Candidate evidence: {phrase}. Span confidence: {vote_bucket}."


def _pool_records(records, use_truth=False):
    pooled = []
    for post_index, record in enumerate(tqdm(records, desc="v13 candidate pools", leave=False)):
        if use_truth:
            risk = int(record["truth"])
        else:
            risk = int(record["risk"])
        candidates = _candidate_pool(record, risk)
        pooled.append({
            "post_index": post_index, "row_id": str(record["row_id"]),
            "user": str(record["user"]), "text": record["text"],
            "risk": risk, "gold": list(record["gold"]), "candidates": candidates,
        })
    return pooled


def _pair_rows(posts, post_indices, training=False):
    rows = []
    for post_index in post_indices:
        post = posts[int(post_index)]
        positives, negatives = [], []
        for candidate_index, candidate in enumerate(post["candidates"]):
            label = float(_post_phrase_f1([candidate["phrase"]], post["gold"]) > 0)
            item = {
                "post_index": int(post_index), "candidate_index": candidate_index,
                "prompt": _prompt(post["risk"], candidate["phrase"], candidate["votes"]),
                "context": _context(post["text"], candidate["phrase"]), "label": label,
            }
            (positives if label else negatives).append(item)
        if training:
            # Preserve every positive and the highest-agreement hard negatives.
            negatives = negatives[:max(5, min(10, 3 * max(1, len(positives))))]
        rows.extend(positives + negatives)
    return rows


class PairDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows
        print(f"v13: tokenizing {len(rows)} candidate/context pairs...", flush=True)
        encoded = tokenizer(
            [row["prompt"] for row in rows], [row["context"] for row in rows],
            padding="max_length", truncation="only_second", max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]
        self.labels = torch.tensor([row["label"] for row in rows], dtype=torch.float32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "label": self.labels[index], "pair_index": int(index),
        }


def _make_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME, num_labels=1, local_files_only=True,
        ignore_mismatched_sizes=True, torch_dtype=torch.float32,
    )
    source = torch.load(config.OUTPUT_DIR / "best_model.pt", map_location="cpu")
    backbone = {
        key.removeprefix("backbone.encoder."): value
        for key, value in source.items() if key.startswith("backbone.encoder.")
    }
    missing, unexpected = model.deberta.load_state_dict(backbone, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected transferred backbone keys: {unexpected[:5]}")
    print(
        f"v13: transferred suicide-domain DeBERTa backbone; missing={len(missing)}",
        flush=True,
    )
    model.gradient_checkpointing_disable()
    for parameter in model.deberta.parameters():
        parameter.requires_grad = False
    for layer in model.deberta.encoder.layer[-4:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    return model


@torch.no_grad()
def _score(model, dataset, device, desc):
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    model.eval(); scores = np.empty(len(dataset), dtype=np.float32)
    for batch in tqdm(loader, desc=desc, leave=False):
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ).logits.squeeze(-1)
        scores[batch["pair_index"].numpy()] = torch.sigmoid(logits).cpu().numpy()
    return scores


def _dedupe_ranked(post, candidate_scores, maximum, threshold):
    ranked = sorted(
        zip(candidate_scores, post["candidates"]), key=lambda item: item[0], reverse=True
    )
    selected = []
    for score, candidate in ranked:
        if float(score) < float(threshold):
            continue
        phrase = candidate["phrase"]; normal = _normalise(phrase)
        if not any(normal in old or old in normal for _, old in selected):
            selected.append((phrase, normal))
        if len(selected) >= int(maximum):
            break
    return [phrase for phrase, _ in selected]


def _post_score_map(rows, pair_scores):
    grouped = defaultdict(dict)
    for row, score in zip(rows, pair_scores):
        grouped[row["post_index"]][row["candidate_index"]] = float(score)
    return grouped


def _evaluate_grid(posts, rows, pair_scores, post_indices):
    grouped = _post_score_map(rows, pair_scores); results = []
    for topk in (1, 2, 3, 4):
        for threshold in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
            per_post = []
            for post_index in post_indices:
                post = posts[int(post_index)]
                scores = [
                    grouped[int(post_index)].get(i, 0.0)
                    for i in range(len(post["candidates"]))
                ]
                evidence = [] if post["risk"] == 0 else _dedupe_ranked(
                    post, scores, topk, threshold
                )
                per_post.append(_post_phrase_f1(evidence, post["gold"]))
            results.append({
                "topk": topk, "threshold": threshold,
                "phrase_f1": float(np.mean(per_post)), "per_post": per_post,
            })
    return sorted(results, key=lambda row: row["phrase_f1"], reverse=True)


def _strict_baseline(records, calibration):
    decoded = _decoder_grid(records); cues = _cue_cache(records)
    _, scores = _evaluate(
        records, decoded[(
            calibration["threshold"], calibration["max_tokens"],
            calibration["end_policy"],
        )], cues, np.arange(len(records)), calibration["cue_policy"],
        calibration["topk"],
    )
    return scores


def train_task1_evidence_reranker_v13():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    seed_everything(config.SEED + 1313)
    if not torch.cuda.is_available():
        raise RuntimeError("V13 contextual reranker requires CUDA")
    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    labels = np.asarray([int(row["risk_label"]) for row in dataset.data])
    groups = np.asarray([str(row["anon_user_id"]) for row in dataset.data])
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(dataset)), labels, groups))
    strict_raw = torch.load(
        config.OUTPUT_DIR / "task1_evidence_v4" / "fold0_raw.pt",
        map_location="cpu", weights_only=False,
    )
    if not np.array_equal(strict_raw["valid_idx"], valid_idx):
        raise ValueError("V13 and V4 strict folds differ")
    if TRAIN_RAW.exists():
        saved = torch.load(TRAIN_RAW, map_location="cpu", weights_only=False)
        train_records = saved["records"]
        print("v13: resumed cached first-stage training candidates", flush=True)
    else:
        print("v13: collecting first-stage candidates for 1305 training posts...", flush=True)
        train_records = _collect_old(dataset, train_idx, torch.device("cuda"))
        for row in train_records:
            row["risk"] = correct_risk_only(
                row["text"], int(np.argmax(row["old_probability"]))
            )
        torch.save({"train_idx": train_idx, "records": train_records}, TRAIN_RAW)
    train_posts = _pool_records(train_records, use_truth=False)
    strict_posts = _pool_records(strict_raw["records"], use_truth=False)

    train_users = np.asarray([post["user"] for post in train_posts])
    fit_posts, calibration_posts = next(GroupShuffleSplit(
        n_splits=1, test_size=0.18, random_state=config.SEED + 1313
    ).split(np.arange(len(train_posts)), groups=train_users))
    fit_rows = _pair_rows(train_posts, fit_posts, training=True)
    calibration_rows = _pair_rows(train_posts, calibration_posts, training=False)
    strict_rows = _pair_rows(strict_posts, np.arange(len(strict_posts)), training=False)
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME, use_fast=True, local_files_only=True
    )
    fit_dataset = PairDataset(fit_rows, tokenizer)
    calibration_dataset = PairDataset(calibration_rows, tokenizer)
    strict_dataset = PairDataset(strict_rows, tokenizer)
    device = torch.device("cuda"); model = _make_model().to(device)
    positive = float(sum(row["label"] for row in fit_rows))
    negative = float(len(fit_rows) - positive)
    pos_weight = torch.tensor(
        min(6.0, max(1.0, negative / max(positive, 1.0))), device=device
    )
    backbone = [p for n, p in model.named_parameters() if n.startswith("deberta.") and p.requires_grad]
    head = [p for n, p in model.named_parameters() if not n.startswith("deberta.") and p.requires_grad]
    optimizer = AdamW([
        {"params": backbone, "lr": 4e-6}, {"params": head, "lr": 2e-5},
    ], weight_decay=config.WEIGHT_DECAY)
    loader = DataLoader(fit_dataset, batch_size=4, shuffle=True, num_workers=0)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    epochs = 2; best = None; history = []
    for epoch in range(1, epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(tqdm(loader, desc=f"v13 reranker epoch {epoch}/{epochs}"), 1):
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
        scores = _score(model, calibration_dataset, device, "v13 internal calibration")
        grid = _evaluate_grid(train_posts, calibration_rows, scores, calibration_posts)
        epoch_row = {
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            **{k: grid[0][k] for k in ("topk", "threshold", "phrase_f1")},
        }
        history.append(epoch_row)
        print(
            f"v13 epoch={epoch} loss={epoch_row['train_loss']:.4f} "
            f"internal_phrase_f1={epoch_row['phrase_f1']:.4f} "
            f"topk={epoch_row['topk']} threshold={epoch_row['threshold']:.2f}",
            flush=True,
        )
        if best is None or epoch_row["phrase_f1"] > best["phrase_f1"]:
            best = epoch_row
            torch.save(model.state_dict(), CHECKPOINT)

    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    strict_pair_scores = _score(model, strict_dataset, device, "v13 strict evaluation")
    strict_grouped = _post_score_map(strict_rows, strict_pair_scores)
    strict_scores = []
    strict_predictions = []
    for post_index, post in enumerate(strict_posts):
        scores = [
            strict_grouped[post_index].get(i, 0.0)
            for i in range(len(post["candidates"]))
        ]
        evidence = [] if post["risk"] == 0 else _dedupe_ranked(
            post, scores, best["topk"], best["threshold"]
        )
        strict_predictions.append(evidence)
        strict_scores.append(_post_phrase_f1(evidence, post["gold"]))
    strict_scores = np.asarray(strict_scores, dtype=np.float32)
    evidence_calibration = load_evidence_calibration()
    baseline_scores = _strict_baseline(strict_raw["records"], evidence_calibration)
    risk_f1 = float(evidence_calibration["strict_risk_f1"])

    local_groups = np.asarray([post["user"] for post in strict_posts])
    unique = np.unique(local_groups); rng = np.random.default_rng(config.SEED + 1313)
    deltas = []
    for _ in range(3000):
        sampled_users = rng.choice(unique, size=len(unique), replace=True)
        sampled = np.concatenate([
            np.flatnonzero(local_groups == user) for user in sampled_users
        ])
        deltas.append(float(strict_scores[sampled].mean() - baseline_scores[sampled].mean()))
    bootstrap = {
        "mean_phrase_delta": float(np.mean(deltas)),
        "p05_phrase_delta": float(np.quantile(deltas, 0.05)),
        "p95_phrase_delta": float(np.quantile(deltas, 0.95)),
        "positive_fraction": float(np.mean(np.asarray(deltas) > 0)),
    }
    strict_phrase = float(strict_scores.mean()); baseline_phrase = float(baseline_scores.mean())
    adopted = bool(
        strict_phrase >= baseline_phrase + 0.005
        and bootstrap["positive_fraction"] >= 0.80
    )
    payload = {
        "training_version": TRAINING_VERSION,
        "fit_posts": int(len(fit_posts)), "calibration_posts": int(len(calibration_posts)),
        "fit_pairs": int(len(fit_rows)), "strict_pairs": int(len(strict_rows)),
        "history": history, "selected": best,
        "baseline": {
            "risk_f1": risk_f1, "phrase_f1": baseline_phrase,
            "task1": task1_score(risk_f1, baseline_phrase),
        },
        "strict": {
            "risk_f1": risk_f1, "phrase_f1": strict_phrase,
            "task1": task1_score(risk_f1, strict_phrase),
            "improved_posts": int((strict_scores > baseline_scores).sum()),
            "worsened_posts": int((strict_scores < baseline_scores).sum()),
        },
        "user_cluster_bootstrap": bootstrap, "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({
        "training_version": TRAINING_VERSION, "adopted": adopted,
        "topk": int(best["topk"]), "threshold": float(best["threshold"]),
        "strict_baseline_phrase_f1": baseline_phrase,
        "strict_phrase_f1": strict_phrase,
        "strict_task1": task1_score(risk_f1, strict_phrase),
        "bootstrap_positive_fraction": bootstrap["positive_fraction"],
    }, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    train_task1_evidence_reranker_v13()

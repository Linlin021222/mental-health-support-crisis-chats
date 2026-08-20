"""Nested user-level calibration of Task 1 evidence extraction."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Subset

from baseline import _post_phrase_f1, _vectorizer
from configs.config import config
from datasets.collator import SuicideRiskCollator
from datasets.dataset import SuicideRiskDataset
from inference.task1_evidence_v4 import (
    CALIBRATION_FILE, correct_risk_only, cue_phrases, decode_model_evidence,
)
from models.multitask_model import SuicideRiskMultiTaskModel
from models.multitask_model_v2 import SuicideRiskMultiTaskModelV2, ordinal_class_probabilities
from preprocess.preprocess import load_train_data
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_evidence_v4"
RAW_FILE = OUTPUT / "fold0_raw.pt"


def _softmax(values, temperature=0.5):
    values = values / float(temperature)
    values -= values.max(1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(1, keepdims=True)


@torch.no_grad()
def _collect_old(dataset, valid_idx, device):
    loader = DataLoader(
        Subset(dataset, valid_idx), batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=SuicideRiskCollator(), num_workers=0,
    )
    model = SuicideRiskMultiTaskModel().to(device)
    model.load_state_dict(torch.load(config.OUTPUT_DIR / "best_model.pt", map_location=device))
    model.eval(); records = []; cursor = 0
    for batch in loader:
        output = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        probability = torch.softmax(output["risk_logits"], -1).cpu().numpy()
        for i in range(len(batch["row_id"])):
            source = dataset.data[int(valid_idx[cursor])]
            records.append({
                "row_id": batch["row_id"][i], "user": str(source["anon_user_id"]),
                "text": batch["texts"][i], "offsets": batch["offset_mappings"][i],
                "truth": int(batch["risk_labels"][i]), "gold": batch["evidences"][i],
                "old_probability": probability[i],
                "start": output["start_logits"][i].cpu(),
                "end": output["end_logits"][i].cpu(),
            })
            cursor += 1
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return records


@torch.no_grad()
def _collect_v2(dataset, valid_idx, device):
    loader = DataLoader(
        Subset(dataset, valid_idx), batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=SuicideRiskCollator(), num_workers=0,
    )
    model = SuicideRiskMultiTaskModelV2().to(device)
    model.load_state_dict(torch.load(
        config.OUTPUT_DIR / "task1_v2_strict_model.pt", map_location=device
    ))
    model.eval(); probabilities = []
    for batch in loader:
        output = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        standard = torch.softmax(output["risk_logits"], -1)
        ordinal = ordinal_class_probabilities(output["ordinal_logits"])
        probabilities.extend((0.75 * standard + 0.25 * ordinal).cpu().numpy())
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return np.asarray(probabilities)


def _load_or_collect(dataset, train_idx, valid_idx):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if RAW_FILE.exists():
        cached = torch.load(RAW_FILE, map_location="cpu", weights_only=False)
        if np.array_equal(cached["valid_idx"], valid_idx):
            return cached["records"]
    device = torch.device(config.DEVICE)
    records = _collect_old(dataset, valid_idx, device)
    v2_probability = _collect_v2(dataset, valid_idx, device)
    frame = load_train_data().reset_index(drop=True)
    vectorizer = _vectorizer()
    train_matrix = vectorizer.fit_transform(frame.text.iloc[train_idx])
    valid_matrix = vectorizer.transform(frame.text.iloc[valid_idx])
    lexical = LinearSVC(C=1.0, class_weight="balanced").fit(
        train_matrix, np.asarray(frame.risk_label)[train_idx]
    )
    lexical_probability = _softmax(lexical.decision_function(valid_matrix), 0.5)
    for i, record in enumerate(records):
        transformer = 0.8 * record["old_probability"] + 0.2 * v2_probability[i]
        final_probability = 0.7 * transformer + 0.3 * lexical_probability[i]
        record["risk"] = correct_risk_only(record["text"], int(np.argmax(final_probability)))
    torch.save({"valid_idx": valid_idx, "records": records}, RAW_FILE)
    return records


def _decoder_grid(records):
    cache = {}
    for threshold in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
        for max_tokens in (4, 6, 8, 10, 12, 16):
            for end_policy in ("nearest", "best"):
                key = (threshold, max_tokens, end_policy)
                cache[key] = [
                    decode_model_evidence(
                        row["text"], row["offsets"], row["start"], row["end"],
                        threshold, max_tokens, end_policy, 5,
                    )
                    for row in records
                ]
    return cache


POLICIES = (
    "current_first", "current_model_first", "none",
    "predicted_first", "predicted_model_first",
    "predicted_extended_first", "predicted_extended_model_first",
    "hierarchical_first", "hierarchical_model_first",
    "hierarchical_extended_first", "hierarchical_extended_model_first",
)


def _cue_cache(records):
    return {
        policy: [cue_phrases(row["text"], row["risk"], policy) for row in records]
        for policy in POLICIES
    }


def _fuse(row, model_phrases, cues, policy, topk):
    if int(row["risk"]) == config.RISK_LABELS["Indicator"]:
        return []
    phrases = list(model_phrases) + cues if policy.endswith("model_first") else cues + list(model_phrases)
    phrases = [part.strip() for phrase in phrases for part in str(phrase).split(";") if part.strip()]
    selected = []
    normalized_selected = []
    for phrase in phrases:
        normalized = " ".join(str(phrase).casefold().split())
        if phrase and not any(normalized in old or old in normalized for old in normalized_selected):
            selected.append(str(phrase)); normalized_selected.append(normalized)
        if len(selected) == int(topk):
            break
    return selected


def _evaluate(records, decoded, cue_cache, indices, policy, topk):
    scores = np.zeros(len(indices), dtype=np.float32)
    for local, index in enumerate(indices):
        row = records[int(index)]
        evidence = _fuse(
            row, decoded[int(index)], cue_cache[policy][int(index)], policy, topk
        )
        scores[local] = _post_phrase_f1(evidence, row["gold"])
    return float(scores.mean()), scores


def _search(records, decoded_cache, cue_cache, indices):
    rows = []
    for decoder_key, decoded in decoded_cache.items():
        threshold, max_tokens, end_policy = decoder_key
        for policy in POLICIES:
            for topk in (1, 2, 3, 4, 5):
                phrase_f1, _ = _evaluate(records, decoded, cue_cache, indices, policy, topk)
                rows.append({
                    "threshold": threshold, "max_tokens": max_tokens,
                    "end_policy": end_policy, "cue_policy": policy,
                    "topk": topk, "phrase_f1": phrase_f1,
                })
    rows.sort(key=lambda row: row["phrase_f1"], reverse=True)
    return rows


def _key(parameters):
    return (parameters["threshold"], parameters["max_tokens"], parameters["end_policy"])


def _median_member(values):
    """Return an observed grid value nearest the median (never invent a key)."""
    median = float(np.median(values))
    return min(values, key=lambda value: (
        round(abs(float(value) - median), 12), float(value)
    ))


def main():
    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    labels = np.asarray([int(item["risk_label"]) for item in dataset.data])
    groups = np.asarray([str(item["anon_user_id"]) for item in dataset.data])
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(dataset)), labels, groups))
    records = _load_or_collect(dataset, train_idx, valid_idx)
    decoded_cache = _decoder_grid(records)
    cue_cache = _cue_cache(records)
    local_groups = np.asarray([row["user"] for row in records])
    all_indices = np.arange(len(records))

    current_decoded = decoded_cache[(0.55, 8, "nearest")]
    baseline_phrase, baseline_scores = _evaluate(
        records, current_decoded, cue_cache, all_indices, "current_first", 5
    )
    truth = np.asarray([row["truth"] for row in records])
    risk_prediction = np.asarray([row["risk"] for row in records])
    risk_f1 = float(f1_score(truth, risk_prediction, average="weighted", zero_division=0))

    nested_phrase = np.zeros(len(records), dtype=np.float32); selected = []
    splitter = GroupKFold(n_splits=4)
    for fold, (fit, held) in enumerate(splitter.split(all_indices, groups=local_groups)):
        best = _search(records, decoded_cache, cue_cache, fit)[0]
        score, per_post = _evaluate(
            records, decoded_cache[_key(best)], cue_cache, held,
            best["cue_policy"], best["topk"]
        )
        nested_phrase[held] = per_post
        selected.append({"fold": fold, "heldout_phrase_f1": score, **best})

    threshold = float(_median_member([row["threshold"] for row in selected]))
    max_tokens = int(_median_member([row["max_tokens"] for row in selected]))
    end_policy = Counter(row["end_policy"] for row in selected).most_common(1)[0][0]
    cue_policy = Counter(row["cue_policy"] for row in selected).most_common(1)[0][0]
    topk = Counter(row["topk"] for row in selected).most_common(1)[0][0]
    fixed = {
        "threshold": threshold, "max_tokens": max_tokens,
        "end_policy": end_policy, "cue_policy": cue_policy, "topk": int(topk),
    }
    fixed_phrase, fixed_scores = _evaluate(
        records, decoded_cache[_key(fixed)], cue_cache, all_indices, cue_policy, topk
    )
    optimistic = _search(records, decoded_cache, cue_cache, all_indices)[0]
    nested_phrase_f1 = float(nested_phrase.mean())
    phrase_by_risk = {}
    for risk_id in range(config.NUM_RISK_CLASSES):
        mask = truth == risk_id
        phrase_by_risk[config.ID2RISK[risk_id]] = {
            "posts": int(mask.sum()),
            "baseline_phrase_f1": float(baseline_scores[mask].mean()) if mask.any() else 0.0,
            "fixed_phrase_f1": float(fixed_scores[mask].mean()) if mask.any() else 0.0,
        }
    evidence_diagnostics = {
        "fixed_improved_posts": int((fixed_scores > baseline_scores).sum()),
        "fixed_worsened_posts": int((fixed_scores < baseline_scores).sum()),
        "fixed_unchanged_posts": int((fixed_scores == baseline_scores).sum()),
        "phrase_f1_by_gold_risk": phrase_by_risk,
    }
    target_phrase_f1 = (7.0 * 0.78 - 4.0 * risk_f1) / 3.0
    # Require both nested evidence generalisation and a non-regressing fixed
    # production setting.  The 0.003 Task-1 margin is larger than small search
    # noise on this 32-user fold.
    adopted = bool(
        nested_phrase_f1 >= baseline_phrase + 0.007
        and fixed_phrase >= baseline_phrase
        and task1_score(risk_f1, nested_phrase_f1)
            >= task1_score(risk_f1, baseline_phrase) + 0.003
    )
    payload = {
        "training_version": "task1-evidence-v4",
        "strict_users": int(len(np.unique(local_groups))),
        "strict_posts": int(len(records)), "risk_f1": risk_f1,
        "target_phrase_f1_for_task1_0.78": target_phrase_f1,
        "baseline": {
            "phrase_f1": baseline_phrase,
            "task1": task1_score(risk_f1, baseline_phrase),
            "parameters": {"threshold": 0.55, "max_tokens": 8,
                           "end_policy": "nearest", "cue_policy": "current_first", "topk": 5},
        },
        "nested_crossfit": {
            "phrase_f1": nested_phrase_f1,
            "task1": task1_score(risk_f1, nested_phrase_f1), "folds": selected,
        },
        "fixed_production": {
            "phrase_f1": fixed_phrase, "task1": task1_score(risk_f1, fixed_phrase),
            "parameters": fixed,
        },
        "optimistic_full_holdout": optimistic,
        "diagnostics": evidence_diagnostics,
        "adopted": adopted,
    }
    (OUTPUT / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    calibration = {
        "training_version": payload["training_version"], "adopted": adopted,
        **fixed, "strict_risk_f1": risk_f1,
        "strict_baseline_phrase_f1": baseline_phrase,
        "strict_nested_phrase_f1": nested_phrase_f1,
        "strict_fixed_phrase_f1": fixed_phrase,
        "strict_nested_task1": task1_score(risk_f1, nested_phrase_f1),
        "strict_fixed_task1": task1_score(risk_f1, fixed_phrase),
    }
    CALIBRATION_FILE.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2)); return payload


if __name__ == "__main__":
    main()

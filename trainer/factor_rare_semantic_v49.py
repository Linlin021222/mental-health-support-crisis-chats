"""Rare-factor semantic specialist with balanced folds and boundary negatives.

The accepted 24-label prototype model gives very rare concepts little gradient.
V49 trains one shared NLI cross-encoder only for the four labels whose ranking
audit showed weak semantic understanding.  Each label has multiple positive
descriptions plus an explicit exclusion boundary, and its negatives are mined
from confusable labels and definition-similar posts using outer-fit users only.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from configs.config import config
from inference.factor_nli import _entailment_index, _rank_decode
from preprocess.factor_semantic_bank_v15 import FACTOR_SEMANTIC_BANK
from preprocess.preprocess import load_train_data
from trainer.factor_sentence_evidence_cv_v27 import _user_bootstrap
from utils.multilabel_group_split import multilabel_group_folds, split_audit
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "factor_rare_semantic_v49"
RESULTS = OUTPUT / "cv_results.json"
OOF_FILE = OUTPUT / "oof_predictions.npz"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "rare-semantic-boundary-nli-v49"
TARGET_LABELS = (13, 18, 22, 23)
BLEND_WEIGHT = 0.20
EPOCHS = 2
CONFUSIONS = {
    13: (9, 17, 14),       # own suicidality, means, stressful event
    18: (8, 11, 12, 14),   # violence, relationship, family, general event
    22: (10, 19, 21, 23),  # poor/support, capital, meaning
    23: (3, 19, 20, 21, 22),
}


def _prompts(label: int) -> list[str]:
    entry = FACTOR_SEMANTIC_BANK[label]
    return [
        entry["formal"] + " " + entry["direct"],
        entry["implicit"] + " " + entry["train_summary"],
        ("This factor is present only when " + entry["formal"] + " "
         + entry["distinction"] + " Exclude the factor when " + entry["negative"]),
    ]


class PairDataset(Dataset):
    def __init__(self, texts, pairs):
        self.texts, self.pairs = texts, pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        row, label, prompt, target, weight = self.pairs[index]
        return self.texts[row], label, prompt, target, weight


def _pairs(texts, targets, seed):
    """Build balanced positives and fit-only semantic/confusion negatives."""
    rng = random.Random(seed)
    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", ngram_range=(1, 3),
        min_df=1, max_features=100_000, sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    rows = []
    audit = {}
    for label in TARGET_LABELS:
        positive = np.flatnonzero(targets[:, label] > 0)
        absent = np.flatnonzero(targets[:, label] == 0)
        prompts = _prompts(label)
        definition = vectorizer.transform([" ".join(prompts)])
        similarity = (matrix @ definition.T).toarray().ravel()
        confusion_mask = targets[:, list(CONFUSIONS[label])].sum(axis=1) > 0
        confusion = [int(row) for row in absent if confusion_mask[row]]
        confusion.sort(key=lambda row: float(similarity[row]), reverse=True)
        lexical = sorted(
            (int(row) for row in absent if row not in set(confusion)),
            key=lambda row: float(similarity[row]), reverse=True,
        )
        negative_target = min(len(absent), max(48, 8 * len(positive)))
        chosen = confusion[:negative_target // 2]
        chosen_set = set(chosen)
        for row in lexical:
            if row not in chosen_set:
                chosen.append(row); chosen_set.add(row)
            if len(chosen) >= negative_target:
                break
        if len(chosen) < negative_target:
            remainder = [int(row) for row in absent if row not in chosen_set]
            rng.shuffle(remainder); chosen.extend(remainder[:negative_target-len(chosen)])

        # Multiple hypotheses force concept learning instead of memorising one
        # wording. Positives are mildly upweighted but not blindly duplicated.
        for row in positive:
            for prompt in range(len(prompts)):
                rows.append((int(row), label, prompt, 1, 1.5))
        for rank, row in enumerate(chosen):
            prompt = rank % len(prompts)
            weight = 1.25 if row in set(confusion) else 1.0
            rows.append((int(row), label, prompt, 0, weight))
        audit[config.ID2FACTOR[label]] = {
            "positives": int(len(positive)),
            "positive_pairs": int(3 * len(positive)),
            "negative_pairs": int(len(chosen)),
            "confusion_negatives": int(sum(row in set(confusion) for row in chosen)),
        }
    rng.shuffle(rows)
    return rows, audit


def _collator(tokenizer):
    def collate(rows):
        texts, labels, prompt_ids, targets, weights = zip(*rows)
        hypotheses = [_prompts(int(label))[int(prompt)]
                      for label, prompt in zip(labels, prompt_ids)]
        encoded = tokenizer(
            list(texts), hypotheses, padding=True, truncation="only_first",
            max_length=config.FACTOR_NLI_MAX_LENGTH, stride=128,
            return_overflowing_tokens=True, return_tensors="pt",
        )
        mapping = encoded.pop("overflow_to_sample_mapping").cpu().numpy()
        selected = []
        for local, target in enumerate(targets):
            indices = np.flatnonzero(mapping == local)
            maximum = 3 if int(target) else 1
            if len(indices) > maximum:
                positions = np.linspace(0, len(indices)-1, maximum).round().astype(int)
                indices = indices[positions]
            selected.extend(indices.tolist())
        selected = np.asarray(selected, dtype=np.int64)
        chosen = torch.tensor(selected, dtype=torch.long)
        output = {key: value.index_select(0, chosen) for key, value in encoded.items()}
        output["mapping"] = torch.tensor(mapping[selected], dtype=torch.long)
        output["targets"] = torch.tensor(targets, dtype=torch.float)
        output["weights"] = torch.tensor(weights, dtype=torch.float)
        return output
    return collate


def _trainable_tail(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    encoder = getattr(getattr(model, "deberta", None), "encoder", None)
    layers = getattr(encoder, "layer", [])
    for layer in list(layers)[-3:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    for name in ("pooler", "classifier"):
        module = getattr(model, name, None)
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = True
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


@torch.inference_mode()
def _predict(model, tokenizer, texts, device, description):
    model.eval(); entailment = _entailment_index(model)
    result = np.zeros((len(texts), len(TARGET_LABELS)), dtype=np.float32)
    batch_size = max(1, int(config.FACTOR_CROSS_ENCODER_BATCH_SIZE))
    for local_label, label in enumerate(tqdm(TARGET_LABELS, desc=description)):
        prompt_scores = np.zeros((len(texts), 3), dtype=np.float32)
        for prompt_id, hypothesis in enumerate(_prompts(label)):
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start+batch_size]
                encoded = tokenizer(
                    batch, [hypothesis] * len(batch), padding=True,
                    truncation="only_first", max_length=config.FACTOR_NLI_MAX_LENGTH,
                    stride=128, return_overflowing_tokens=True, return_tensors="pt",
                )
                mapping = encoded.pop("overflow_to_sample_mapping").cpu().numpy()
                selected = []
                for row in range(len(batch)):
                    indices = np.flatnonzero(mapping == row)
                    if len(indices) > 3:
                        positions = np.linspace(0, len(indices)-1, 3).round().astype(int)
                        indices = indices[positions]
                    selected.extend(indices.tolist())
                selected = np.asarray(selected, dtype=np.int64)
                chosen = torch.tensor(selected, dtype=torch.long)
                mapping = mapping[selected]
                encoded = {key: value.index_select(0, chosen).to(device)
                           for key, value in encoded.items()}
                with torch.autocast(device_type=device.type, enabled=config.FP16):
                    logits = model(**encoded).logits.float()
                scores = torch.softmax(logits, dim=-1)[:, entailment].cpu().numpy()
                for row in range(len(batch)):
                    prompt_scores[start+row, prompt_id] = scores[mapping == row].max()
        # Median resists one over-broad positive prompt or brittle boundary prompt.
        result[:, local_label] = np.median(prompt_scores, axis=1)
    return result


def _rank(values):
    values = np.asarray(values); result = np.zeros_like(values, dtype=np.float32)
    for column in range(values.shape[1]):
        order = np.argsort(values[:, column], kind="stable")
        result[order, column] = np.arange(len(values), dtype=np.float32) / max(1, len(values)-1)
    return result


def _baseline_probability():
    base = np.load(config.OUTPUT_DIR / "factor_cv" / "factor_oof_predictions.npz")
    v48 = np.load(config.OUTPUT_DIR / "factor_balanced_sparse_v48" /
                  "oof_predictions.npz")["balanced_tfidf"]
    old = np.load(config.OUTPUT_DIR / "factor_cross_encoder" /
                  "oof_predictions.npz")["probabilities"]
    prototype = np.load(config.OUTPUT_DIR / "factor_cross_encoder_v2" /
                        "oof_predictions.npz")["probabilities"]
    calibration = json.loads((config.OUTPUT_DIR / "factor_cross_encoder_v2" /
                              "calibration.json").read_text(encoding="utf-8"))
    sparse_base = .70 * base["semantic"] + .30 * v48
    return (float(calibration["base_weight"]) * sparse_base
            + float(calibration["old_cross_weight"]) * old
            + float(calibration["new_cross_weight"]) * prototype).astype(np.float32)


def _fold_paths(fold):
    return OUTPUT / f"fold{fold}_model.pt", OUTPUT / f"fold{fold}_valid.npz"


def _train_fold(fold, fit, valid, frame, targets, tokenizer, device):
    checkpoint, prediction_file = _fold_paths(fold)
    if checkpoint.exists() and prediction_file.exists():
        saved = np.load(prediction_file)
        if np.array_equal(saved["valid_indices"], valid):
            print(f"V49 fold {fold}: resumed", flush=True)
            return saved["probabilities"].astype(np.float32), json.loads(str(saved["summary"]))
    seed_everything(config.SEED + 4900 + fold)
    texts = frame.text.iloc[fit].astype(str).tolist()
    pairs, pair_audit = _pairs(texts, targets[fit], config.SEED + 4900 + fold)
    loader = DataLoader(
        PairDataset(texts, pairs), batch_size=2, shuffle=True,
        collate_fn=_collator(tokenizer), num_workers=0,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.FACTOR_NLI_MODEL_NAME, dtype=torch.float32,
        local_files_only=True,
    ).to(device)
    parameters = _trainable_tail(model)
    optimizer = AdamW(parameters, lr=1e-5, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=config.FP16)
    entailment = _entailment_index(model); accumulation = 8
    model.gradient_checkpointing_enable(); model.config.use_cache = False
    history = []
    for epoch in range(1, EPOCHS+1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(tqdm(loader, desc=f"V49 fold {fold} epoch {epoch}"), 1):
            truth = batch.pop("targets").to(device)
            weights = batch.pop("weights").to(device)
            mapping = batch.pop("mapping").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, enabled=config.FP16):
                logits = model(**batch).logits
                other = 1 - entailment
                margin = logits[:, entailment] - logits[:, other]
                pair_margin = torch.stack([
                    margin[mapping == pair].max() for pair in range(len(truth))
                ])
                raw = torch.nn.functional.binary_cross_entropy_with_logits(
                    pair_margin, truth, reduction="none",
                )
                loss = (raw * weights).mean() / accumulation
            scaler.scale(loss).backward(); losses.append(float(loss.detach()) * accumulation)
            if step % accumulation == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, config.GRAD_CLIP)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        history.append(float(np.mean(losses)))
        print(f"V49 fold={fold} epoch={epoch} loss={history[-1]:.5f}", flush=True)
    probability = _predict(
        model, tokenizer, frame.text.iloc[valid].astype(str).tolist(),
        device, f"V49 fold {fold} inference",
    )
    summary = {
        "fold": fold, "train": int(len(fit)), "valid": int(len(valid)),
        "pair_count": int(len(pairs)), "pair_audit": pair_audit,
        "history": history, "training_version": TRAINING_VERSION,
    }
    torch.save(model.state_dict(), checkpoint)
    np.savez_compressed(
        prediction_file, probabilities=probability, valid_indices=valid,
        summary=json.dumps(summary),
    )
    del model, optimizer, loader
    torch.cuda.empty_cache()
    return probability, summary


def _decode(probability, targets, folds):
    prediction = np.zeros_like(targets, dtype=bool)
    for fit, valid in folds:
        prediction[valid] = _rank_decode(
            probability[valid], targets[fit].mean(0), 1.10,
        )
    return prediction


def cross_validate(only_fold0=False):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    groups = frame.anon_user_id.astype(str).to_numpy()
    risk = frame.risk_label.to_numpy(dtype=np.int64)
    folds = multilabel_group_folds(targets, groups, risk, 5, config.SEED + 47)
    tokenizer = AutoTokenizer.from_pretrained(
        config.FACTOR_NLI_MODEL_NAME, use_fast=True, local_files_only=True,
    )
    device = torch.device(config.DEVICE)
    oof = np.zeros((len(frame), len(TARGET_LABELS)), dtype=np.float32)
    summaries = []
    selected = folds[:1] if only_fold0 else folds
    for fold, (fit, valid) in enumerate(selected):
        probability, summary = _train_fold(
            fold, fit, valid, frame, targets, tokenizer, device,
        )
        oof[valid] = probability; summaries.append(summary)

    baseline_probability = _baseline_probability()
    eval_indices = np.concatenate([valid for _, valid in selected])
    eval_set = set(eval_indices.tolist())
    eval_folds = [(fit, np.asarray([row for row in valid if row in eval_set], dtype=int))
                  for fit, valid in selected]
    candidate_probability = baseline_probability.copy()
    for _, valid in eval_folds:
        current_rank = _rank(baseline_probability[valid][:, TARGET_LABELS])
        specialist_rank = _rank(oof[valid])
        candidate_probability[np.ix_(valid, TARGET_LABELS)] = (
            (1.0-BLEND_WEIGHT) * current_rank + BLEND_WEIGHT * specialist_rank
        )
    baseline_prediction = np.zeros_like(targets, dtype=bool)
    candidate_prediction = np.zeros_like(targets, dtype=bool)
    for fit, valid in eval_folds:
        baseline_prediction[valid] = _rank_decode(
            baseline_probability[valid], targets[fit].mean(0), 1.10,
        )
        candidate_prediction[valid] = _rank_decode(
            candidate_probability[valid], targets[fit].mean(0), 1.10,
        )
    truth = targets[eval_indices]
    baseline_score = float(f1_score(
        truth, baseline_prediction[eval_indices], average="macro", zero_division=0,
    ))
    candidate_score = float(f1_score(
        truth, candidate_prediction[eval_indices], average="macro", zero_division=0,
    ))
    per_label = []
    for local, label in enumerate(TARGET_LABELS):
        old = float(f1_score(truth[:, label], baseline_prediction[eval_indices, label], zero_division=0))
        new = float(f1_score(truth[:, label], candidate_prediction[eval_indices, label], zero_division=0))
        score = oof[eval_indices, local]
        per_label.append({
            "label": config.ID2FACTOR[label], "support": int(truth[:, label].sum()),
            "baseline_f1": old, "candidate_f1": new, "delta": new-old,
            "specialist_roc_auc": float(roc_auc_score(truth[:, label], score)),
            "specialist_pr_auc": float(average_precision_score(truth[:, label], score)),
        })
    target_old = float(np.mean([row["baseline_f1"] for row in per_label]))
    target_new = float(np.mean([row["candidate_f1"] for row in per_label]))
    bootstrap = _user_bootstrap(
        truth, baseline_prediction[eval_indices], candidate_prediction[eval_indices],
        groups[eval_indices], seed=494949, draws=3000,
    )
    promising = bool(candidate_score > baseline_score and target_new > target_old)
    adopted = bool(
        not only_fold0 and candidate_score >= baseline_score + .0025
        and bootstrap["positive_fraction"] >= .75
        and target_new > target_old
    )
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation": "factor-balanced user-disjoint fold0" if only_fold0 else "five-fold factor-balanced user-disjoint OOF",
        "target_labels": [config.ID2FACTOR[x] for x in TARGET_LABELS],
        "fixed_blend_weight": BLEND_WEIGHT,
        "split_audit": split_audit(selected, targets, groups),
        "baseline_macro_f1": baseline_score,
        "candidate_macro_f1": candidate_score,
        "delta": candidate_score-baseline_score,
        "target_mean_baseline_f1": target_old,
        "target_mean_candidate_f1": target_new,
        "target_mean_delta": target_new-target_old,
        "per_label": per_label, "folds": summaries,
        "user_cluster_bootstrap": bootstrap,
        "promising_for_full_oof": promising,
        "adopted": adopted,
    }
    if not only_fold0:
        np.savez_compressed(OOF_FILE, probabilities=oof, targets=targets,
                            target_labels=np.asarray(TARGET_LABELS))
        CALIBRATION.write_text(json.dumps({
            "training_version": TRAINING_VERSION, "adopted": adopted,
            "blend_weight": BLEND_WEIGHT,
            "target_labels": list(TARGET_LABELS),
        }, indent=2), encoding="utf-8")
    result_path = OUTPUT / ("fold0_results.json" if only_fold0 else "cv_results.json")
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items()
                      if key not in ("folds", "split_audit")}, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    cross_validate(False)

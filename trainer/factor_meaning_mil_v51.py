"""Meaning-in-life multi-prototype MIL specialist (V51).

The released annotations repeat a factor when it is supported more than once.
For meaning in life this is useful weak supervision, but the submitted target
is still binary.  V51 therefore uses occurrence counts only as a capped sample
weight.  Unlike V49, a positive post is not forced to entail every definition:
the strongest of five mutually complementary semantic prototypes supplies the
positive MIL score.  Hard negatives must fail all prototypes.
"""
from __future__ import annotations

import json
import random

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from configs.config import config
from inference.factor_boundary_lexicon_v50 import boundary_flags
from inference.factor_nli import _entailment_index, _rank_decode
from preprocess.preprocess import load_train_data
from trainer.factor_rare_semantic_v49 import _baseline_probability, _trainable_tail
from utils.multilabel_group_split import multilabel_group_folds, split_audit
from utils.seed import seed_everything


OUTPUT = config.OUTPUT_DIR / "factor_meaning_mil_v51"
OOF_FILE = OUTPUT / "oof_predictions.npz"
RESULTS = OUTPUT / "cv_results.json"
LABEL = 23
EPOCHS = 2
MAX_LENGTH = 384
ACCUMULATION = 8
TRAINING_VERSION = "meaning-multi-prototype-mil-v51"

# Each prototype covers a genuine subtype of the paper-aligned definition.
# The explicit exclusion clauses provide the requested positive/negative
# boundary rather than treating generic positive emotion as life meaning.
PROTOTYPES = (
    "The author explicitly discusses whether life has meaning, purpose, a point, or is worth living. Generic sadness or hopelessness without this existential question is not enough.",
    "The author identifies a person, pet, relationship, value, activity, or commitment as a reason to live or as what makes continuing life meaningful. Merely mentioning family or a pet is not enough.",
    "The author describes personally important future goals, dreams, education, work, family, travel, or achievements that give life direction. A routine task or vague wish to feel better is not enough.",
    "The author actively chooses or wants to stay alive, keep living, or keep fighting because life still matters to them. A wish to die or passive survival alone is not this factor.",
    "The author describes losing, searching for, or needing a reason, purpose, direction, or something worth living for. Hopelessness concerns improvement; this factor specifically concerns significance or direction in life.",
)
CONFUSIONS = (3, 5, 10, 19, 20, 21, 22)


class MeaningDataset(Dataset):
    def __init__(self, texts, rows):
        self.texts, self.rows = texts, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row, target, weight, prompts = self.rows[index]
        return self.texts[row], target, weight, prompts


def _training_rows(texts, targets, counts, seed):
    rng = random.Random(seed)
    positive = np.flatnonzero(targets[:, LABEL] > 0)
    absent = np.flatnonzero(targets[:, LABEL] == 0)
    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", ngram_range=(1, 3),
        min_df=1, max_features=100_000, sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    definition = vectorizer.transform([" ".join(PROTOTYPES)])
    similarity = (matrix @ definition.T).toarray().ravel()
    confusion = targets[:, list(CONFUSIONS)].sum(axis=1) > 0
    # Prioritise definition-similar and confusing negatives.  Random examples
    # remain in the tail so the classifier also sees ordinary Reddit language.
    ranked = sorted(
        (int(row) for row in absent),
        key=lambda row: (bool(confusion[row]), float(similarity[row])),
        reverse=True,
    )
    negative_target = min(len(absent), max(160, 10 * len(positive)))
    hard_count = min(len(ranked), int(round(negative_target * .8)))
    chosen = ranked[:hard_count]
    chosen_set = set(chosen)
    remainder = [int(row) for row in absent if row not in chosen_set]
    rng.shuffle(remainder)
    chosen.extend(remainder[:negative_target-len(chosen)])

    rows = []
    all_prompts = tuple(range(len(PROTOTYPES)))
    for row in positive:
        salience = 1.0 + .25 * np.log1p(max(1, int(counts[row])))
        rows.append((int(row), 1, float(1.5 * salience), all_prompts))
    for rank, row in enumerate(chosen):
        # Two rotating prototypes expose every boundary while keeping the
        # negative batches small enough for an 8 GB GPU.
        prompt_ids = (rank % len(PROTOTYPES), (rank + 2) % len(PROTOTYPES))
        weight = 1.3 if confusion[row] or rank < hard_count else 1.0
        rows.append((int(row), 0, float(weight), prompt_ids))
    rng.shuffle(rows)
    return rows, {
        "positive_posts": int(len(positive)),
        "negative_posts": int(len(chosen)),
        "hard_negative_posts": int(hard_count),
        "positive_occurrence_histogram": {
            str(value): int(np.sum(counts[positive] == value))
            for value in np.unique(counts[positive])
        },
    }


def _collator(tokenizer):
    def collate(batch):
        flat_text, flat_hypothesis, owners, targets, weights = [], [], [], [], []
        for owner, (text, target, weight, prompts) in enumerate(batch):
            targets.append(target); weights.append(weight)
            for prompt in prompts:
                flat_text.append(text); flat_hypothesis.append(PROTOTYPES[int(prompt)])
                owners.append(owner)
        encoded = tokenizer(
            flat_text, flat_hypothesis, padding=True, truncation="only_first",
            max_length=MAX_LENGTH, stride=96, return_overflowing_tokens=True,
            return_tensors="pt",
        )
        pair_mapping = encoded.pop("overflow_to_sample_mapping").cpu().numpy()
        doc_mapping = np.asarray(owners, dtype=np.int64)[pair_mapping]
        selected = []
        for pair in range(len(flat_text)):
            indices = np.flatnonzero(pair_mapping == pair)
            owner = owners[pair]
            limit = 2 if int(targets[owner]) else 1
            if len(indices) > limit:
                positions = np.linspace(0, len(indices)-1, limit).round().astype(int)
                indices = indices[positions]
            selected.extend(indices.tolist())
        selected = np.asarray(selected, dtype=np.int64)
        chosen = torch.tensor(selected, dtype=torch.long)
        result = {key: value.index_select(0, chosen) for key, value in encoded.items()}
        result["mapping"] = torch.tensor(doc_mapping[selected], dtype=torch.long)
        result["targets"] = torch.tensor(targets, dtype=torch.float)
        result["weights"] = torch.tensor(weights, dtype=torch.float)
        return result
    return collate


@torch.inference_mode()
def _predict(model, tokenizer, texts, device, description):
    model.eval(); entailment = _entailment_index(model)
    scores = np.zeros((len(texts), len(PROTOTYPES)), dtype=np.float32)
    batch_size = max(1, int(config.FACTOR_CROSS_ENCODER_BATCH_SIZE))
    for prompt_id, hypothesis in enumerate(tqdm(PROTOTYPES, desc=description)):
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start+batch_size]
            encoded = tokenizer(
                batch, [hypothesis] * len(batch), padding=True,
                truncation="only_first", max_length=MAX_LENGTH, stride=96,
                return_overflowing_tokens=True, return_tensors="pt",
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
            local_mapping = mapping[selected]
            encoded = {key: value.index_select(0, chosen).to(device)
                       for key, value in encoded.items()}
            with torch.autocast(device_type=device.type, enabled=config.FP16):
                logits = model(**encoded).logits.float()
            probability = torch.softmax(logits, dim=-1)[:, entailment].cpu().numpy()
            for row in range(len(batch)):
                scores[start+row, prompt_id] = probability[local_mapping == row].max()
    # One genuine semantic subtype is sufficient. Hard-negative MIL training
    # explicitly teaches the maximum not to fire on boundary cases.
    return scores.max(1), scores


def _fold_paths(fold):
    return OUTPUT / f"fold{fold}_model.pt", OUTPUT / f"fold{fold}_valid.npz"


def _train_fold(fold, fit, valid, frame, targets, counts, tokenizer, device):
    checkpoint, prediction_file = _fold_paths(fold)
    if checkpoint.exists() and prediction_file.exists():
        saved = np.load(prediction_file)
        if np.array_equal(saved["valid_indices"], valid):
            print(f"V51 fold {fold}: resumed", flush=True)
            return saved["probability"], saved["prototype_scores"], json.loads(str(saved["summary"]))
    seed_everything(config.SEED + 5100 + fold)
    train_texts = frame.text.iloc[fit].astype(str).tolist()
    rows, audit = _training_rows(train_texts, targets[fit], counts[fit], config.SEED + 5100 + fold)
    loader = DataLoader(
        MeaningDataset(train_texts, rows), batch_size=1, shuffle=True,
        collate_fn=_collator(tokenizer), num_workers=0,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.FACTOR_NLI_MODEL_NAME, dtype=torch.float32, local_files_only=True,
    ).to(device)
    parameters = _trainable_tail(model)
    optimizer = AdamW(parameters, lr=8e-6, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=config.FP16)
    entailment = _entailment_index(model)
    model.gradient_checkpointing_enable(); model.config.use_cache = False
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(tqdm(loader, desc=f"V51 fold {fold} epoch {epoch}"), 1):
            truth = batch.pop("targets").to(device)
            weights = batch.pop("weights").to(device)
            mapping = batch.pop("mapping").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, enabled=config.FP16):
                logits = model(**batch).logits
                other = 1 - entailment
                margin = logits[:, entailment] - logits[:, other]
                document_margin = torch.stack([
                    margin[mapping == row].max() for row in range(len(truth))
                ])
                raw = torch.nn.functional.binary_cross_entropy_with_logits(
                    document_margin, truth, reduction="none",
                )
                loss = (raw * weights).mean() / ACCUMULATION
            scaler.scale(loss).backward(); losses.append(float(loss.detach()) * ACCUMULATION)
            if step % ACCUMULATION == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, config.GRAD_CLIP)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        history.append(float(np.mean(losses)))
        print(f"V51 fold={fold} epoch={epoch} loss={history[-1]:.5f}", flush=True)
    probability, prototype_scores = _predict(
        model, tokenizer, frame.text.iloc[valid].astype(str).tolist(), device,
        f"V51 fold {fold} inference",
    )
    summary = {"fold": fold, "train": int(len(fit)), "valid": int(len(valid)),
               "row_audit": audit, "history": history, "training_version": TRAINING_VERSION}
    torch.save(model.state_dict(), checkpoint)
    np.savez_compressed(prediction_file, probability=probability,
                        prototype_scores=prototype_scores, valid_indices=valid,
                        summary=json.dumps(summary))
    del model, optimizer, loader
    if device.type == "cuda": torch.cuda.empty_cache()
    return probability, prototype_scores, summary


def _rank(values):
    order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
    return (order + .5) / len(values)


def _meaning_prediction(probability, targets, folds, ratio=1.10):
    prediction = np.zeros(len(targets), dtype=bool)
    for fit, valid in folds:
        expected = len(valid) * targets[fit, LABEL].mean() * ratio
        count = max(1, int(round(expected)))
        prediction[valid[np.argsort(probability[valid])[-count:]]] = True
    return prediction


def cross_validate(only_fold0=False):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    groups = frame.anon_user_id.astype(str).to_numpy()
    risk = frame.risk_label.to_numpy(dtype=np.int64)
    counts = np.load(config.OUTPUT_DIR / "factor_cross_encoder_v2" /
                     "oof_predictions.npz")["factor_counts"][:, LABEL].astype(np.int16)
    folds = multilabel_group_folds(targets, groups, risk, 5, config.SEED + 47)
    selected = folds[:1] if only_fold0 else folds
    tokenizer = AutoTokenizer.from_pretrained(
        config.FACTOR_NLI_MODEL_NAME, use_fast=True, local_files_only=True,
    )
    device = torch.device(config.DEVICE)
    oof = np.zeros(len(frame), dtype=np.float32)
    prototype_oof = np.zeros((len(frame), len(PROTOTYPES)), dtype=np.float32)
    summaries = []
    for fold, (fit, valid) in enumerate(selected):
        probability, scores, summary = _train_fold(
            fold, fit, valid, frame, targets, counts, tokenizer, device,
        )
        oof[valid] = probability; prototype_oof[valid] = scores; summaries.append(summary)

    indices = np.concatenate([valid for _, valid in selected])
    eval_folds = selected
    base = _baseline_probability()[:, LABEL]
    flags = boundary_flags(frame.text.astype(str).tolist())[:, LABEL]
    v50 = base + .20 * flags
    truth = targets[indices, LABEL]
    base_prediction = _meaning_prediction(v50, targets, eval_folds)
    specialist_prediction = _meaning_prediction(oof, targets, eval_folds)
    rows = []
    for name, probability, prediction in (
        ("v50", v50, base_prediction),
        ("specialist", oof, specialist_prediction),
    ):
        rows.append({
            "name": name,
            "f1": float(f1_score(truth, prediction[indices], zero_division=0)),
            "roc_auc": float(roc_auc_score(truth, probability[indices])),
            "pr_auc": float(average_precision_score(truth, probability[indices])),
        })
    # Fixed diagnostic blends are reported; adoption is deferred to V52's
    # nested router after all five folds exist.
    for weight in (.25, .50, .75):
        mixed = v50.copy()
        for _, valid in eval_folds:
            mixed[valid] = ((1-weight) * _rank(v50[valid]) + weight * _rank(oof[valid]))
        prediction = _meaning_prediction(mixed, targets, eval_folds)
        rows.append({"name": f"blend_{weight:.2f}",
                     "f1": float(f1_score(truth, prediction[indices], zero_division=0)),
                     "roc_auc": float(roc_auc_score(truth, mixed[indices])),
                     "pr_auc": float(average_precision_score(truth, mixed[indices]))})
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation": "factor-balanced user-disjoint fold0" if only_fold0 else "five-fold factor-balanced user-disjoint OOF",
        "label": config.ID2FACTOR[LABEL], "support": int(truth.sum()),
        "split_audit": split_audit(selected, targets, groups),
        "models": rows, "folds": summaries,
        "adopted": False,
        "note": "V52 nested routing is required before production adoption.",
    }
    if not only_fold0:
        np.savez_compressed(OOF_FILE, probability=oof,
                            prototype_scores=prototype_oof, targets=targets)
    path = OUTPUT / ("fold0_results.json" if only_fold0 else "cv_results.json")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items()
                      if key not in ("folds", "split_audit")}, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    cross_validate(False)

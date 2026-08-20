"""Risk-definition-conditioned MentalRoBERTa evidence reranker (V32).

V31 showed that lexical evidence identity is not the limiting factor.  V32
scores each high-recall first-stage candidate jointly with its local context,
the predicted risk definition, and lightweight clinical discourse flags.  A
single calibration fold selects the epoch and decoding policy.  Every reported
validation fold is then trained and evaluated independently by user.
"""
from __future__ import annotations

from collections import defaultdict
import json
import re

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from baseline import _post_phrase_f1
from configs.config import config
from inference.task1_evidence_v4 import load_evidence_calibration
from preprocess.preprocess import load_train_data
from trainer.task1_atomic_v25 import _bootstrap, _load_records
from trainer.task1_evidence_reranker_v13 import (
    PairDataset, _context, _pool_records, _post_score_map,
)
from trainer.task1_oof_stack_v20 import _baseline_evidence, _hybrid_prediction, _parameter_grid
from utils.seed import seed_everything
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_clinical_reranker_v32"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "task1-clinical-context-reranker-v32"
CALIBRATION_FOLD = 3
VALIDATION_FOLDS = (0, 1, 2)
EPOCHS = 2
BATCH_SIZE = 8

DEFINITIONS = {
    0: "Indicator: no explicit mention of the author's suicide.",
    1: "Ideation: the author explicitly expresses suicide or a wish to die, without a plan.",
    2: "Behavior: the author expresses suicide and a self-harm method, means, intention, or plan.",
    3: "Attempt: the author explicitly says they made a recent or past suicide attempt.",
}

FIRST_PERSON = re.compile(r"\b(?:i|i'm|i've|i'd|me|my|mine|myself)\b", re.I)
OTHER_PERSON = re.compile(r"\b(?:he|she|they|friend|boyfriend|girlfriend|brother|sister|mother|father)\b", re.I)
NEGATION = re.compile(r"\b(?:not|never|no longer|don't|didn't|won't|wouldn't)\b", re.I)
ATTEMPT = re.compile(r"\b(?:attempt(?:ed)?|tried to|survived|failed attempt|overdosed|woke up)\b", re.I)
PLAN = re.compile(r"\b(?:plan|planning|method|pills|rope|gun|knife|bridge|overdose|hang|slit|jump|tonight|tomorrow)\b", re.I)
QUOTE = re.compile(r"(?:^|\s)[>\"“]|(?:said|says|told me|asked me)\b", re.I)


def _flags(context: str, phrase: str) -> str:
    match = re.search(re.escape(str(phrase)), str(context), re.I)
    if match:
        neighbourhood = context[max(0, match.start() - 100): min(len(context), match.end() + 100)]
    else:
        neighbourhood = context[:200]
    values = []
    for name, pattern in (
        ("first-person", FIRST_PERSON), ("other-person", OTHER_PERSON),
        ("negation", NEGATION), ("attempt-history", ATTEMPT),
        ("plan-or-means", PLAN), ("quotation-or-report", QUOTE),
    ):
        if pattern.search(neighbourhood):
            values.append(name)
    return ", ".join(values) if values else "none"


def _clinical_rows(posts, post_indices, training=False):
    rows = []
    for post_index in map(int, post_indices):
        post = posts[post_index]; positives = []; negatives = []
        risk = int(post["risk"]); risk_name = config.ID2RISK[risk]
        for candidate_index, candidate in enumerate(post["candidates"]):
            phrase = str(candidate["phrase"])
            context = _context(post["text"], phrase, radius=520)
            label = float(_post_phrase_f1([phrase], post["gold"]) > 0)
            votes = int(candidate["votes"])
            confidence = "high" if votes >= 15 else "medium" if votes >= 5 else "low"
            prompt = (
                "Task: decide whether the verbatim candidate is direct evidence for the "
                "AUTHOR'S predicted suicide-risk level. Exclude quoted speech, another "
                "person's risk, generic discussion, negated statements, and unsupported "
                "severity. "
                f"Predicted label: {risk_name}. Definition: {DEFINITIONS[risk]} "
                f"Candidate: {phrase}. Decoder agreement: {confidence}. "
                f"Local discourse flags: {_flags(context, phrase)}."
            )
            item = {"post_index": post_index, "candidate_index": candidate_index,
                    "prompt": prompt, "context": context, "label": label}
            (positives if label else negatives).append(item)
        if training:
            # Keep all positives.  Highest decoder-agreement candidates occur
            # first and are therefore the most useful hard negatives.
            negatives = negatives[:max(6, min(12, 4 * max(1, len(positives))))]
        rows.extend(positives + negatives)
    return rows


def _make_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        config.FACTOR_MODEL_NAME, num_labels=1, local_files_only=True,
        ignore_mismatched_sizes=True, dtype=torch.float32,
    )
    model.gradient_checkpointing_disable()
    # MentalRoBERTa has 12 encoder layers.  Freezing the bottom eight reduces
    # variance on 1.6k posts while allowing the top semantic layers to adapt.
    for parameter in model.roberta.parameters():
        parameter.requires_grad = False
    for layer in model.roberta.encoder.layer[-4:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    return model


@torch.no_grad()
def _score(model, dataset, device, description):
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    values = np.empty(len(dataset), dtype=np.float32); model.eval()
    for batch in tqdm(loader, desc=description, leave=False):
        logits = model(input_ids=batch["input_ids"].to(device),
                       attention_mask=batch["attention_mask"].to(device)).logits.squeeze(-1)
        values[batch["pair_index"].numpy()] = torch.sigmoid(logits).cpu().numpy()
    return values


def _train_model(fold_name, train_rows, valid_rows, posts, valid_indices,
                 tokenizer, device, epochs, select_policy):
    checkpoint = OUTPUT / f"{fold_name}_model.pt"
    train_dataset = PairDataset(train_rows, tokenizer)
    valid_dataset = PairDataset(valid_rows, tokenizer)
    seed_everything(config.SEED + 3200 + sum(ord(char) for char in fold_name))
    model = _make_model().to(device)
    trainable_backbone = [p for name, p in model.named_parameters()
                          if p.requires_grad and not name.startswith("classifier.")]
    head = list(model.classifier.parameters())
    optimizer = AdamW([{"params": trainable_backbone, "lr": 1e-5},
                       {"params": head, "lr": 5e-5}], weight_decay=.01)
    positives = float(sum(row["label"] for row in train_rows)); negatives = len(train_rows) - positives
    pos_weight = torch.tensor(min(8., max(1., negatives / max(positives, 1.))), device=device)
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history = []; best = None
    for epoch in range(1, epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for step, batch in enumerate(tqdm(loader, desc=f"V32 {fold_name} epoch {epoch}/{epochs}"), 1):
            with torch.autocast(device_type="cuda", enabled=True):
                logits = model(input_ids=batch["input_ids"].to(device),
                               attention_mask=batch["attention_mask"].to(device)).logits.squeeze(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, batch["label"].to(device), pos_weight=pos_weight
                ) / 2.0
            scaler.scale(loss).backward(); losses.append(float(loss.detach()) * 2.0)
            if step % 2 == 0 or step == len(loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if select_policy:
            scores = _score(model, valid_dataset, device, f"V32 {fold_name} calibration")
            selected = _parameter_grid(posts, valid_rows, scores, valid_indices,
                                       load_evidence_calibration())[0]
            row.update(selected)
            if best is None or row["phrase_f1"] > best["phrase_f1"]:
                best = dict(row); torch.save(model.state_dict(), checkpoint)
            print(f"V32 {fold_name} epoch={epoch} phrase={row['phrase_f1']:.6f} "
                  f"policy={row['mode']}", flush=True)
        history.append(row)
    if not select_policy:
        torch.save(model.state_dict(), checkpoint); best = history[-1]
    else:
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    scores = _score(model, valid_dataset, device, f"V32 {fold_name} final")
    del train_dataset, valid_dataset, loader, optimizer
    return model, scores, best, history


def _predictions(posts, rows, scores, indices, policy):
    grouped = _post_score_map(rows, scores); predictions = []
    for index in map(int, indices):
        post = posts[index]; baseline = list(post["baseline_evidence"])
        values = [grouped[index].get(candidate_index, 0.)
                  for candidate_index in range(len(post["candidates"]))]
        evidence = _hybrid_prediction(post, values, baseline, policy)
        predictions.append([] if int(post["risk"]) == 0 else evidence)
    return predictions


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("V32 requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True); device = torch.device("cuda")
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(); groups = frame.anon_user_id.astype(str).to_numpy()
    outer_train, _ = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(frame)), labels, groups))
    raw_records, membership, _ = _load_records()
    posts = _pool_records(raw_records, use_truth=False)
    global_to_local = {int(record["global_index"]): local
                       for local, record in enumerate(raw_records)}
    calibration = load_evidence_calibration()
    for post, record in zip(posts, raw_records):
        post["baseline_evidence"] = _baseline_evidence(record, calibration)
    tokenizer = AutoTokenizer.from_pretrained(
        config.FACTOR_MODEL_NAME, use_fast=True, local_files_only=True
    )

    # Pair rows address posts by local index in ``posts``.  Membership remains
    # attached to the original global training index.
    cal_local = np.asarray([global_to_local[int(i)] for i in outer_train
                            if membership[int(i)] == CALIBRATION_FOLD])
    fit_local = np.asarray([global_to_local[int(i)] for i in outer_train
                            if membership[int(i)] != CALIBRATION_FOLD])
    train_rows = _clinical_rows(posts, fit_local, training=True)
    valid_rows = _clinical_rows(posts, cal_local, training=False)
    model, _, selected_epoch, calibration_history = _train_model(
        "calibration", train_rows, valid_rows, posts, cal_local,
        tokenizer, device, EPOCHS, select_policy=True,
    )
    policy = {key: selected_epoch[key] for key in ("mode", "topk", "threshold", "gate")}
    chosen_epochs = int(selected_epoch["epoch"])
    print(f"V32 fixed policy from fold3: epochs={chosen_epochs}, {policy}", flush=True)
    del model; torch.cuda.empty_cache()

    all_global = []; all_old = []; all_new = []; all_risk = []; fold_rows = []
    for fold in VALIDATION_FOLDS:
        valid_global = np.asarray([i for i in outer_train if membership[int(i)] == fold])
        fit_global = np.asarray([i for i in outer_train if membership[int(i)] != fold])
        if set(groups[valid_global]) & set(groups[fit_global]):
            raise ValueError(f"V32 fold {fold} user leakage")
        valid_local = np.asarray([global_to_local[int(i)] for i in valid_global])
        fit_local = np.asarray([global_to_local[int(i)] for i in fit_global])
        train_rows = _clinical_rows(posts, fit_local, training=True)
        valid_rows = _clinical_rows(posts, valid_local, training=False)
        model, scores, _, history = _train_model(
            f"fold{fold}", train_rows, valid_rows, posts, valid_local,
            tokenizer, device, chosen_epochs, select_policy=False,
        )
        predictions = _predictions(posts, valid_rows, scores, valid_local, policy)
        baseline_predictions = [list(posts[int(i)]["baseline_evidence"]) for i in valid_local]
        gold = [list(posts[int(i)]["gold"]) for i in valid_local]
        old = np.asarray([_post_phrase_f1(pred, target)
                          for pred, target in zip(baseline_predictions, gold)], dtype=np.float32)
        new = np.asarray([_post_phrase_f1(pred, target)
                          for pred, target in zip(predictions, gold)], dtype=np.float32)
        risks = np.asarray([int(posts[int(i)]["risk"]) for i in valid_local])
        fold_rows.append({"fold": fold, "fit_posts": int(len(fit_local)),
                          "valid_posts": int(len(valid_local)),
                          "baseline_phrase_f1": float(old.mean()),
                          "candidate_phrase_f1": float(new.mean()),
                          "phrase_delta": float(new.mean() - old.mean()),
                          "improved_posts": int((new > old).sum()),
                          "worsened_posts": int((new < old).sum()), "history": history})
        print(f"V32 fold={fold} phrase {old.mean():.6f} -> {new.mean():.6f}", flush=True)
        all_global.extend(map(int, valid_global)); all_old.extend(old); all_new.extend(new)
        all_risk.extend(risks)
        del model; torch.cuda.empty_cache()

    order = np.argsort(all_global); indices = np.asarray(all_global)[order]
    old = np.asarray(all_old, dtype=np.float32)[order]
    new = np.asarray(all_new, dtype=np.float32)[order]
    risks = np.asarray(all_risk, dtype=np.int64)[order]
    risk_f1 = float(f1_score(labels[indices], risks, average="weighted", zero_division=0))
    old_task = task1_score(risk_f1, float(old.mean())); new_task = task1_score(risk_f1, float(new.mean()))
    bootstrap = _bootstrap(groups[indices], old, new)
    adopted = bool(new_task >= old_task + .003 and bootstrap["positive_fraction"] >= .80)
    payload = {"training_version": TRAINING_VERSION,
        "evaluation_scope": "epoch/policy fold3; independently trained untouched user folds0-2",
        "selected_epoch": selected_epoch, "fixed_policy": policy,
        "calibration_history": calibration_history, "folds": fold_rows,
        "baseline": {"risk_f1": risk_f1, "phrase_f1": float(old.mean()), "task1": old_task},
        "candidate": {"risk_f1": risk_f1, "phrase_f1": float(new.mean()), "task1": new_task,
                      "improved_posts": int((new > old).sum()),
                      "worsened_posts": int((new < old).sum())},
        "user_cluster_bootstrap": bootstrap, "adopted": adopted}
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({"training_version": TRAINING_VERSION,
        "adopted": adopted, "selected_epoch": chosen_epochs, "policy": policy,
        "crossvalidated_task1": new_task}, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True); return payload


if __name__ == "__main__":
    main()

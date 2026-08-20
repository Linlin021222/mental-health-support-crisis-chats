"""Boundary-aware continuation training for Task 1 risk (V39).

The four first-stage V20 checkpoints are already trained on user-disjoint
folds.  This experiment continues each checkpoint with a risk-only objective
that combines nominal, ordinal-boundary and class-prototype supervision.  All
reported probabilities remain out-of-fold; production is gated on the full
four-fold result and a user-cluster bootstrap.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from analyze_task1_oof_risk_v36 import (
    CACHE as V36_CACHE, _evidence_matrix, _predict as _v36_predict,
)
from configs.config import config
from datasets.collator import SuicideRiskCollator
from datasets.dataset import SuicideRiskDataset
from inference.task1_evidence_v4 import correct_risk_only
from models.multitask_model import SuicideRiskMultiTaskModel
from preprocess.preprocess import load_train_data
from trainer.task1_atomic_v25 import _load_records
from utils.seed import seed_everything
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_boundary_model_v39"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "task1-boundary-prototype-continuation-v39"
V20 = config.OUTPUT_DIR / "task1_oof_stack_v20"
EPOCHS = 1
BATCH_SIZE = 2
ACCUMULATION = 4
ORDINAL_MIX = .25


class BoundaryRiskModel(nn.Module):
    """Existing risk model plus cumulative boundary and prototype heads."""

    def __init__(self):
        super().__init__()
        self.base = SuicideRiskMultiTaskModel()
        hidden = int(config.HIDDEN_SIZE)
        self.ordinal_head = nn.Linear(hidden, 3)
        # Direct document-space prototypes avoid a randomly initialised
        # projection bottleneck on this small dataset.
        self.prototypes = nn.Parameter(torch.empty(4, hidden))
        self._initialise_from_classifier()

    def _initialise_from_classifier(self):
        classifier = self.base.risk_head.classifier
        with torch.no_grad():
            self.prototypes.copy_(classifier.weight)
            for threshold in range(3):
                lower_w = classifier.weight[:threshold + 1].mean(0)
                upper_w = classifier.weight[threshold + 1:].mean(0)
                lower_b = classifier.bias[:threshold + 1].mean()
                upper_b = classifier.bias[threshold + 1:].mean()
                self.ordinal_head.weight[threshold].copy_(upper_w - lower_w)
                self.ordinal_head.bias[threshold].copy_(upper_b - lower_b)

    def forward(self, input_ids, attention_mask):
        hidden = self.base.backbone(input_ids, attention_mask).float()
        document = self.base.pooling(hidden, attention_mask)
        nominal = self.base.risk_head(document)
        ordinal = self.ordinal_head(document)
        prototype_logits = (
            F.normalize(document, dim=-1)
            @ F.normalize(self.prototypes, dim=-1).t()
        ) / .12
        return nominal, ordinal, prototype_logits


def _freeze_for_continuation(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    layers = model.base.backbone.encoder.encoder.layer
    # Four trainable top layers provide enough boundary adaptation while the
    # frozen lower language representation limits catastrophic forgetting.
    for layer in layers[-4:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    for module in (model.base.pooling, model.base.risk_head, model.ordinal_head):
        for parameter in module.parameters():
            parameter.requires_grad = True
    model.prototypes.requires_grad = True


def _loader(dataset, indices, shuffle):
    return DataLoader(
        Subset(dataset, list(map(int, indices))), batch_size=BATCH_SIZE,
        shuffle=shuffle, collate_fn=SuicideRiskCollator(), num_workers=0,
        pin_memory=True,
    )


def _ordinal_targets(labels):
    return torch.stack([(labels > boundary).float() for boundary in range(3)], 1)


def _probabilities(nominal, ordinal):
    nominal_probability = torch.softmax(nominal.float(), -1)
    cumulative = torch.sigmoid(ordinal.float())
    # Enforce P(y>0) >= P(y>1) >= P(y>2).
    cumulative = torch.cummin(cumulative, dim=1).values
    ordinal_probability = torch.stack((
        1. - cumulative[:, 0],
        cumulative[:, 0] - cumulative[:, 1],
        cumulative[:, 1] - cumulative[:, 2],
        cumulative[:, 2],
    ), 1).clamp_min(1e-7)
    ordinal_probability /= ordinal_probability.sum(1, keepdim=True)
    return ((1. - ORDINAL_MIX) * nominal_probability
            + ORDINAL_MIX * ordinal_probability)


def _train_fold(fold, dataset, labels, device):
    raw = torch.load(V20 / f"inner_fold{fold}_raw.pt", map_location="cpu",
                     weights_only=False)
    fit_idx = np.asarray(raw["fit_idx"], dtype=np.int64)
    oof_idx = np.asarray(raw["oof_idx"], dtype=np.int64)
    prediction_file = OUTPUT / f"fold{fold}_probabilities.npz"
    checkpoint_file = OUTPUT / f"fold{fold}_model.pt"
    if prediction_file.exists():
        saved = np.load(prediction_file)
        if (str(saved["training_version"].item()) == TRAINING_VERSION
                and np.array_equal(saved["global_indices"], oof_idx)):
            print(f"V39 fold {fold}: resumed {len(oof_idx)} OOF probabilities", flush=True)
            return oof_idx, saved["probabilities"]

    seed_everything(config.SEED + 3900 + fold)
    model = BoundaryRiskModel()
    state = torch.load(V20 / f"inner_fold{fold}_model.pt", map_location="cpu",
                       weights_only=True)
    model.base.load_state_dict(state)
    # Initialise the new heads only after loading the trained classifier.
    model._initialise_from_classifier()
    _freeze_for_continuation(model)
    model.base.backbone.encoder.gradient_checkpointing_disable()
    model.to(device)

    counts = np.bincount(labels[fit_idx], minlength=4)
    class_weight = np.sqrt(len(fit_idx) / np.maximum(counts, 1))
    class_weight = torch.tensor(class_weight / class_weight.mean(),
                                dtype=torch.float32, device=device)
    ordinal_count = np.stack([(labels[fit_idx] > boundary).sum()
                              for boundary in range(3)])
    ordinal_pos_weight = np.sqrt(
        (len(fit_idx) - ordinal_count) / np.maximum(ordinal_count, 1)
    )
    ordinal_pos_weight = torch.tensor(ordinal_pos_weight, dtype=torch.float32,
                                      device=device)

    backbone = [parameter for name, parameter in model.named_parameters()
                if "base.backbone.encoder.encoder.layer" in name
                and parameter.requires_grad]
    backbone_ids = {id(parameter) for parameter in backbone}
    heads = [parameter for parameter in model.parameters()
             if parameter.requires_grad and id(parameter) not in backbone_ids]
    optimizer = AdamW([
        {"params": backbone, "lr": 4e-6},
        {"params": heads, "lr": 2e-5},
    ], weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    train_loader = _loader(dataset, fit_idx, True)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, EPOCHS + 1):
        model.train(); losses = []
        progress = tqdm(train_loader, desc=f"V39 fold {fold} epoch {epoch}/{EPOCHS}")
        for step, batch in enumerate(progress, 1):
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            target = batch["risk_labels"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=True):
                nominal, ordinal, prototypes = model(ids, mask)
                nominal_loss = F.cross_entropy(nominal, target, weight=class_weight)
                ordinal_loss = F.binary_cross_entropy_with_logits(
                    ordinal, _ordinal_targets(target), pos_weight=ordinal_pos_weight
                )
                prototype_loss = F.cross_entropy(prototypes, target,
                                                 weight=class_weight)
                loss = (nominal_loss + .30 * ordinal_loss
                        + .12 * prototype_loss) / ACCUMULATION
            scaler.scale(loss).backward()
            losses.append(float(loss.detach()) * ACCUMULATION)
            if step % ACCUMULATION == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters()
                     if parameter.requires_grad), 1.0
                )
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
        print(f"V39 fold={fold} loss={np.mean(losses):.5f}", flush=True)

    model.eval(); collected = []
    with torch.no_grad():
        for batch in tqdm(_loader(dataset, oof_idx, False),
                          desc=f"V39 fold {fold} OOF inference"):
            with torch.autocast(device_type="cuda", enabled=True):
                nominal, ordinal, _ = model(
                    batch["input_ids"].to(device, non_blocking=True),
                    batch["attention_mask"].to(device, non_blocking=True),
                )
            collected.append(_probabilities(nominal, ordinal).cpu().numpy())
    probabilities = np.vstack(collected)
    torch.save({"training_version": TRAINING_VERSION,
                "model": model.state_dict()}, checkpoint_file)
    np.savez_compressed(prediction_file, training_version=TRAINING_VERSION,
                        global_indices=oof_idx, probabilities=probabilities)
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    return oof_idx, probabilities


def _metric(truth, prediction, evidence, indices):
    indices = np.asarray(indices, dtype=np.int64)
    risk = float(f1_score(truth[indices], prediction[indices], average="weighted",
                          zero_division=0))
    phrase_values = evidence[np.arange(len(prediction)), prediction]
    phrase = float(phrase_values[indices].mean())
    return risk, phrase, task1_score(risk, phrase), phrase_values


def _main_impl():
    if not torch.cuda.is_available():
        raise RuntimeError("V39 boundary continuation requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    records, membership_map, _ = _load_records()
    global_indices = np.asarray([int(row["global_index"]) for row in records])
    membership = np.asarray([membership_map[int(index)] for index in global_indices])
    position = {int(index): pos for pos, index in enumerate(global_indices)}

    candidate_probability = np.zeros((len(records), 4), dtype=np.float32)
    device = torch.device("cuda")
    for fold in range(4):
        indices, probability = _train_fold(fold, dataset, labels, device)
        candidate_probability[[position[int(index)] for index in indices]] = probability

    saved = np.load(V36_CACHE, allow_pickle=True)
    names = saved["names"].tolist(); decisions = saved["decisions"]
    v36 = json.loads((config.OUTPUT_DIR / "task1_oof_risk_v36" / "calibration.json")
                     .read_text(encoding="utf-8"))
    decision = decisions[names.index(v36["expert"])]
    old_probability = np.vstack([row["old_probability"] for row in records])
    truth = labels[global_indices]
    local_groups = groups[global_indices]
    corrections = np.asarray([
        [correct_risk_only(row["text"], risk) for risk in range(4)]
        for row in records
    ], dtype=np.int64)
    evidence = _evidence_matrix(records)

    baseline = _v36_predict(old_probability, decision, v36, corrections)
    base = _metric(truth, baseline, evidence, np.arange(len(truth)))
    weights = (0., .25, .50, .75, 1.)
    crossfit = np.zeros(len(truth), dtype=np.int64); selections = []; folds = []
    for fold in range(4):
        fit = np.flatnonzero(membership != fold)
        held = np.flatnonzero(membership == fold)
        candidates = []
        for weight in weights:
            blended = ((1. - weight) * old_probability
                       + weight * candidate_probability)
            prediction = _v36_predict(blended, decision, v36, corrections)
            metric = _metric(truth, prediction, evidence, fit)
            candidates.append((metric[2], metric[0], weight, prediction))
        _, _, selected_weight, prediction = max(candidates,
                                                 key=lambda row: (row[0], row[1]))
        crossfit[held] = prediction[held]
        old = _metric(truth, baseline, evidence, held)
        new = _metric(truth, crossfit, evidence, held)
        selections.append(selected_weight)
        folds.append({"fold": fold, "posts": int(len(held)),
                      "new_model_weight": selected_weight,
                      "baseline_risk_f1": old[0], "candidate_risk_f1": new[0],
                      "baseline_task1": old[2], "candidate_task1": new[2]})
        print(f"V39 fold={fold} weight={selected_weight:.2f} "
              f"risk {old[0]:.6f}->{new[0]:.6f} "
              f"task1 {old[2]:.6f}->{new[2]:.6f}", flush=True)

    cross = _metric(truth, crossfit, evidence, np.arange(len(truth)))
    production_weight = Counter(selections).most_common(1)[0][0]
    fixed_probability = ((1. - production_weight) * old_probability
                         + production_weight * candidate_probability)
    fixed_prediction = _v36_predict(fixed_probability, decision, v36, corrections)
    fixed = _metric(truth, fixed_prediction, evidence, np.arange(len(truth)))

    unique = np.unique(local_groups); rng = np.random.default_rng(config.SEED + 3939)
    deltas = []
    for _ in range(4000):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([np.flatnonzero(local_groups == user)
                                   for user in sampled])
        old_risk = f1_score(truth[selected], baseline[selected],
                            average="weighted", zero_division=0)
        new_risk = f1_score(truth[selected], crossfit[selected],
                            average="weighted", zero_division=0)
        old_phrase = float(base[3][selected].mean())
        new_phrase = float(cross[3][selected].mean())
        deltas.append(task1_score(new_risk, new_phrase)
                      - task1_score(old_risk, old_phrase))
    deltas = np.asarray(deltas)
    bootstrap = {"mean_task1_delta": float(deltas.mean()),
                 "p05_task1_delta": float(np.quantile(deltas, .05)),
                 "p95_task1_delta": float(np.quantile(deltas, .95)),
                 "positive_fraction": float((deltas > 0).mean())}
    adopted = bool(cross[2] >= base[2] + .003
                   and fixed[2] >= base[2] + .002
                   and bootstrap["positive_fraction"] >= .80
                   and production_weight > 0.)
    payload = {
        "training_version": TRAINING_VERSION,
        "method": {"epochs": EPOCHS, "trainable_top_layers": 4,
                   "ordinal_loss_weight": .30, "prototype_loss_weight": .12,
                   "ordinal_probability_mix": ORDINAL_MIX},
        "evaluation_scope": "all 1305 nested OOF posts; user-disjoint four-fold continuation",
        "baseline_v36": {"risk_f1": base[0], "phrase_f1": base[1],
                         "task1": base[2]},
        "crossfit_candidate": {"risk_f1": cross[0], "phrase_f1": cross[1],
                               "task1": cross[2], "folds": folds},
        "fixed_production_diagnostic": {"new_model_weight": production_weight,
                                        "risk_f1": fixed[0], "phrase_f1": fixed[1],
                                        "task1": fixed[2]},
        "user_cluster_bootstrap": bootstrap,
        "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({
        "training_version": TRAINING_VERSION, "adopted": adopted,
        "new_model_weight": production_weight,
        "crossfit_task1": cross[2], "baseline_task1": base[2],
    }, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main():
    return _main_impl()


if __name__ == "__main__":
    main()

"""Shared hierarchical Task-1 continuation with neural evidence hard negatives.

The four V20 checkpoints provide user-disjoint first-stage predictions for all
1,305 outer-training posts.  This experiment continues each checkpoint for one
epoch and adds three definition-aligned binary heads:

    explicit suicide mention -> attempt -> behavior versus ideation

The heads share the DeBERTa token representation with the span/token evidence
heads.  Suicide-looking sentences that are not annotated evidence and contain
third-person, negated, hypothetical, or historical scope are up-weighted as
token-level hard negatives.  Model/evidence selection is nested: a fold's
parameters are selected without looking at that fold's users.
"""
from __future__ import annotations

from collections import Counter
import json
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from analyze_task1_oof_risk_v36 import (
    CACHE as V36_CACHE,
    _evidence_matrix,
    _predict as _v36_predict,
)
from configs.config import config
from datasets.collator import SuicideRiskCollator
from datasets.dataset import SuicideRiskDataset
from inference.task1_evidence_v4 import correct_risk_only
from models.losses import EvidenceLoss
from models.multitask_model import SuicideRiskMultiTaskModel
from preprocess.preprocess import load_train_data
from trainer.task1_atomic_v25 import _load_records
from utils.seed import seed_everything
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_joint_hard_negative_v77"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
V20 = config.OUTPUT_DIR / "task1_oof_stack_v20"
TRAINING_VERSION = "task1-shared-hierarchy-hard-negative-v77b"
EPOCHS = 1
BATCH_SIZE = 2
ACCUMULATION = 4

SUICIDE_CUE = re.compile(
    r"\b(?:suicid\w*|kill(?:ing)?\s+(?:myself|ourselves|himself|herself|themselves)|"
    r"end(?:ing)?\s+(?:my|our|his|her|their)\s+life|want(?:ed|ing)?\s+to\s+die|"
    r"wanna\s+die|wish(?:ed|ing)?\s+(?:i\s+)?(?:was\s+)?dead|better\s+off\s+dead|"
    r"not\s+wake\s+up|overdos\w*|slit(?:ting)?\s+(?:my|his|her|their)|"
    r"blow\s+(?:my|his|her|their)\s+(?:head|brains)|hang(?:ing)?\s+(?:myself|himself|herself))\b",
    re.I,
)
NEGATION = re.compile(
    r"\b(?:not|never|no longer|don['’]?t|didn['’]?t|doesn['’]?t|won['’]?t|"
    r"wouldn['’]?t|can['’]?t|without|stop(?:ped)?|avoid(?:ed|ing)?)\b", re.I,
)
HYPOTHETICAL = re.compile(
    r"\b(?:if|unless|would|could|might|maybe|perhaps|imagin\w*|fantas\w*|"
    r"hypothetical(?:ly)?|what if|jok(?:e|ed|ing)|dream(?:ed|ing)?)\b", re.I,
)
THIRD_PERSON = re.compile(
    r"\b(?:he|she|they|him|her|them|his|hers|their|friend|boyfriend|girlfriend|"
    r"husband|wife|partner|brother|sister|mother|father|mom|mum|dad|ex)\b", re.I,
)
HISTORICAL = re.compile(
    r"\b(?:used to|in the past|years? ago|months? ago|weeks? ago|days? ago|"
    r"when i was|previously|back then|former(?:ly)?|once)\b", re.I,
)
SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")


class SharedHierarchyModel(nn.Module):
    """V20 model plus evidence-conditioned, definition-aligned boundaries."""

    def __init__(self):
        super().__init__()
        self.base = SuicideRiskMultiTaskModel()
        hidden = int(config.HIDDEN_SIZE)
        self.token_evidence_head = nn.Sequential(nn.Dropout(.1), nn.Linear(hidden, 1))
        self.evidence_norm = nn.LayerNorm(hidden)
        self.explicit_head = nn.Linear(hidden, 1)
        self.attempt_head = nn.Linear(hidden, 1)
        self.behavior_head = nn.Linear(hidden, 1)

    def initialise_new_heads(self):
        classifier = self.base.risk_head.classifier
        span = self.base.evidence_head.span_classifier
        with torch.no_grad():
            # Start/end evidence weights are already trained; their mean is a
            # much safer token-evidence initialisation than a random head.
            self.token_evidence_head[1].weight.copy_(span.weight.mean(0, keepdim=True))
            self.token_evidence_head[1].bias.copy_(span.bias.mean().reshape(1))
            positive_w = classifier.weight[1:].mean(0)
            positive_b = classifier.bias[1:].mean()
            self.explicit_head.weight.copy_((positive_w - classifier.weight[0])[None])
            self.explicit_head.bias.copy_((positive_b - classifier.bias[0]).reshape(1))
            middle_w = classifier.weight[1:3].mean(0)
            middle_b = classifier.bias[1:3].mean()
            self.attempt_head.weight.copy_((classifier.weight[3] - middle_w)[None])
            self.attempt_head.bias.copy_((classifier.bias[3] - middle_b).reshape(1))
            self.behavior_head.weight.copy_((classifier.weight[2] - classifier.weight[1])[None])
            self.behavior_head.bias.copy_((classifier.bias[2] - classifier.bias[1]).reshape(1))

    def forward(self, input_ids, attention_mask):
        hidden = self.base.backbone(input_ids, attention_mask).float()
        document = self.base.pooling(hidden, attention_mask)
        start, end = self.base.evidence_head(hidden)
        token = self.token_evidence_head(hidden).squeeze(-1)

        mask = attention_mask.bool()
        attention_logits = token.masked_fill(~mask, -1e4)
        attention = torch.softmax(attention_logits.flatten(1), dim=1)
        evidence = (attention.unsqueeze(-1) * hidden.flatten(1, 2)).sum(1)
        # Keep the pretrained document representation dominant.  Evidence is
        # a residual cue rather than a random replacement representation.
        evidence_gate = torch.sigmoid(token.masked_fill(~mask, -20.).amax((1, 2)))
        boundary = self.evidence_norm(document + .20 * evidence_gate[:, None] * evidence)
        return {
            "risk_logits": self.base.risk_head(document),
            "start_logits": start,
            "end_logits": end,
            "token_logits": token,
            "explicit_logits": self.explicit_head(boundary).squeeze(-1),
            "attempt_logits": self.attempt_head(boundary).squeeze(-1),
            "behavior_logits": self.behavior_head(boundary).squeeze(-1),
        }


def _freeze_for_continuation(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for layer in model.base.backbone.encoder.encoder.layer[-4:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    modules = (
        model.base.pooling, model.base.risk_head, model.base.evidence_head,
        model.token_evidence_head, model.evidence_norm, model.explicit_head,
        model.attempt_head, model.behavior_head,
    )
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True


def _loader(dataset, indices, shuffle):
    return DataLoader(
        Subset(dataset, list(map(int, indices))), batch_size=BATCH_SIZE,
        shuffle=shuffle, collate_fn=SuicideRiskCollator(), num_workers=0,
        pin_memory=True,
    )


def _hard_negative_mask(texts, offsets_batch, token_targets):
    """Mark scoped suicide-looking sentences that contain no gold evidence."""
    result = torch.zeros_like(token_targets, dtype=torch.float32)
    for row, (text, chunks) in enumerate(zip(texts, offsets_batch)):
        spans = []
        for match in SENTENCE.finditer(str(text)):
            sentence = match.group(0)
            if (SUICIDE_CUE.search(sentence)
                    and any(pattern.search(sentence) for pattern in (
                        NEGATION, HYPOTHETICAL, THIRD_PERSON, HISTORICAL))):
                spans.append((match.start(), match.end()))
        if not spans:
            continue
        for chunk, offsets in enumerate(chunks):
            # A sentence containing annotated evidence must remain a positive
            # training context (past attempts are valid Attempt evidence).
            for left, right in spans:
                positions = []
                for token, pair in enumerate(offsets):
                    start, end = map(int, pair)
                    if end > start and start < right and end > left:
                        positions.append(token)
                if positions and token_targets[row, chunk, positions].sum() == 0:
                    result[row, chunk, positions] = 1.
    return result


class HierarchyEvidenceLoss(nn.Module):
    def __init__(self, labels, fit_idx, dataset, device):
        super().__init__()
        counts = np.bincount(labels[fit_idx], minlength=4)
        class_weight = np.sqrt(len(fit_idx) / np.maximum(counts, 1))
        self.register_buffer("class_weight", torch.tensor(
            class_weight / class_weight.mean(), dtype=torch.float32, device=device
        ))
        explicit_positive = int((labels[fit_idx] > 0).sum())
        attempted = labels[fit_idx][labels[fit_idx] > 0]
        behavior = labels[fit_idx][np.isin(labels[fit_idx], [1, 2])]
        self.register_buffer("explicit_pos", torch.tensor(
            np.sqrt((len(fit_idx) - explicit_positive) / max(explicit_positive, 1)),
            dtype=torch.float32, device=device,
        ))
        self.register_buffer("attempt_pos", torch.tensor(
            np.sqrt(max((attempted != 3).sum(), 1) / max((attempted == 3).sum(), 1)),
            dtype=torch.float32, device=device,
        ))
        self.register_buffer("behavior_pos", torch.tensor(
            np.sqrt(max((behavior == 1).sum(), 1) / max((behavior == 2).sum(), 1)),
            dtype=torch.float32, device=device,
        ))
        token_positive = sum(float(dataset.data[int(i)]["token_labels"].sum()) for i in fit_idx)
        token_total = sum(float(dataset.data[int(i)]["attention_mask"].sum()) for i in fit_idx)
        self.register_buffer("token_pos", torch.tensor(
            min(15., max(3., np.sqrt((token_total - token_positive) / max(token_positive, 1.)))),
            dtype=torch.float32, device=device,
        ))
        self.span = EvidenceLoss()

    def forward(self, output, batch, hard_negative):
        labels = batch["risk_labels"]
        nominal = F.cross_entropy(output["risk_logits"], labels, weight=self.class_weight)
        explicit = F.binary_cross_entropy_with_logits(
            output["explicit_logits"], (labels > 0).float(), pos_weight=self.explicit_pos
        )
        suicidal = labels > 0
        if suicidal.any():
            attempt = F.binary_cross_entropy_with_logits(
                output["attempt_logits"][suicidal], (labels[suicidal] == 3).float(),
                pos_weight=self.attempt_pos,
            )
        else:
            attempt = output["attempt_logits"].sum() * 0.
        middle = (labels == 1) | (labels == 2)
        if middle.any():
            behavior = F.binary_cross_entropy_with_logits(
                output["behavior_logits"][middle], (labels[middle] == 2).float(),
                pos_weight=self.behavior_pos,
            )
        else:
            behavior = output["behavior_logits"].sum() * 0.
        span = self.span(
            output["start_logits"], output["end_logits"],
            batch["start_labels"], batch["end_labels"], batch["attention_mask"],
        )
        target = batch["token_labels"].float()
        mask = batch["attention_mask"].float()
        raw = F.binary_cross_entropy_with_logits(output["token_logits"], target, reduction="none")
        weights = torch.ones_like(raw)
        weights = torch.where(target > 0, self.token_pos.to(raw.dtype), weights)
        weights = torch.where((hard_negative > 0) & (target == 0), 2.5, weights)
        token_bce = (raw * weights * mask).sum() / (weights * mask).sum().clamp_min(1.)
        probability = torch.sigmoid(output["token_logits"]) * mask
        intersection = (probability * target).sum((1, 2))
        dice = (1. - (2. * intersection + 1.) /
                (probability.sum((1, 2)) + target.sum((1, 2)) + 1.)).mean()

        contrast_terms = []
        for row in range(len(labels)):
            positive = output["token_logits"][row][target[row] > 0]
            negative = output["token_logits"][row][hard_negative[row] > 0]
            if len(positive) and len(negative):
                contrast_terms.append(F.softplus(negative.max() - positive.max() + .5))
        contrast = (torch.stack(contrast_terms).mean() if contrast_terms
                    else output["token_logits"].sum() * 0.)
        evidence = .45 * span + .35 * token_bce + .20 * dice
        total = (nominal + .22 * explicit + .32 * attempt + .32 * behavior
                 + .45 * evidence + .10 * contrast)
        return total


def _hierarchy_probability(output):
    explicit = torch.sigmoid(output["explicit_logits"].float())
    attempt = torch.sigmoid(output["attempt_logits"].float())
    behavior = torch.sigmoid(output["behavior_logits"].float())
    probability = torch.stack((
        1. - explicit,
        explicit * (1. - attempt) * (1. - behavior),
        explicit * (1. - attempt) * behavior,
        explicit * attempt,
    ), 1).clamp_min(1e-7)
    return probability / probability.sum(1, keepdim=True)


def _train_fold(fold, dataset, labels, device):
    raw = torch.load(V20 / f"inner_fold{fold}_raw.pt", map_location="cpu", weights_only=False)
    fit_idx = np.asarray(raw["fit_idx"], dtype=np.int64)
    oof_idx = np.asarray(raw["oof_idx"], dtype=np.int64)
    output_file = OUTPUT / f"fold{fold}_raw.pt"
    checkpoint = OUTPUT / f"fold{fold}_model.pt"
    if output_file.exists():
        saved = torch.load(output_file, map_location="cpu", weights_only=False)
        if (saved.get("training_version") == TRAINING_VERSION
                and np.array_equal(np.asarray(saved["global_indices"]), oof_idx)):
            print(f"V77 fold {fold}: resumed {len(oof_idx)} OOF posts", flush=True)
            return saved

    seed_everything(config.SEED + 7700 + fold)
    model = SharedHierarchyModel()
    model.base.load_state_dict(torch.load(
        V20 / f"inner_fold{fold}_model.pt", map_location="cpu", weights_only=True
    ))
    model.initialise_new_heads()
    _freeze_for_continuation(model)
    model.base.backbone.encoder.gradient_checkpointing_disable()
    model.to(device)
    criterion = HierarchyEvidenceLoss(labels, fit_idx, dataset, device).to(device)
    backbone = [p for n, p in model.named_parameters()
                if "base.backbone.encoder.encoder.layer" in n and p.requires_grad]
    backbone_ids = {id(p) for p in backbone}
    heads = [p for p in model.parameters()
             if p.requires_grad and id(p) not in backbone_ids]
    optimizer = AdamW([
        {"params": backbone, "lr": 3e-6},
        {"params": heads, "lr": 1.5e-5},
    ], weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    train_loader = _loader(dataset, fit_idx, True)
    optimizer.zero_grad(set_to_none=True)
    hard_tokens = 0
    for epoch in range(1, EPOCHS + 1):
        model.train(); losses = []
        progress = tqdm(train_loader, desc=f"V77 fold {fold} epoch {epoch}/{EPOCHS}")
        for step, batch in enumerate(progress, 1):
            hard = _hard_negative_mask(
                batch["texts"], batch["offset_mappings"], batch["token_labels"]
            ).to(device, non_blocking=True)
            hard_tokens += int(hard.sum().item())
            for key in ("input_ids", "attention_mask", "risk_labels", "start_labels",
                        "end_labels", "token_labels"):
                batch[key] = batch[key].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=True):
                output = model(batch["input_ids"], batch["attention_mask"])
                loss = criterion(output, batch, hard) / ACCUMULATION
            scaler.scale(loss).backward()
            losses.append(float(loss.detach()) * ACCUMULATION)
            if step % ACCUMULATION == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.
                )
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        print(f"V77 fold={fold} loss={np.mean(losses):.5f} hard_negative_tokens={hard_tokens}",
              flush=True)

    nominal, hierarchy, starts, ends, tokens = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(_loader(dataset, oof_idx, False),
                          desc=f"V77 fold {fold} OOF inference"):
            with torch.autocast(device_type="cuda", enabled=True):
                output = model(
                    batch["input_ids"].to(device, non_blocking=True),
                    batch["attention_mask"].to(device, non_blocking=True),
                )
            nominal.append(torch.softmax(output["risk_logits"].float(), -1).cpu().numpy())
            hierarchy.append(_hierarchy_probability(output).cpu().numpy())
            starts.extend(output["start_logits"].float().cpu())
            ends.extend(output["end_logits"].float().cpu())
            tokens.extend(output["token_logits"].float().cpu())
    saved = {
        "training_version": TRAINING_VERSION,
        "global_indices": oof_idx,
        "nominal": np.vstack(nominal),
        "hierarchy": np.vstack(hierarchy),
        "start": starts, "end": ends, "token": tokens,
        "hard_negative_tokens": hard_tokens,
    }
    torch.save(saved, output_file)
    torch.save({"training_version": TRAINING_VERSION, "model": model.state_dict()}, checkpoint)
    del model, optimizer, criterion, scaler
    torch.cuda.empty_cache()
    return saved


def _metric(truth, prediction, evidence, indices):
    indices = np.asarray(indices, dtype=np.int64)
    risk = float(f1_score(truth[indices], prediction[indices],
                          average="weighted", zero_division=0))
    phrase_values = evidence[np.arange(len(prediction)), prediction]
    phrase = float(phrase_values[indices].mean())
    return risk, phrase, task1_score(risk, phrase), phrase_values


def _main_impl():
    if not torch.cuda.is_available():
        raise RuntimeError("V77 shared hierarchy experiment requires CUDA")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    records, membership_map, _ = _load_records()
    global_indices = np.asarray([int(row["global_index"]) for row in records])
    truth = labels[global_indices]
    local_groups = groups[global_indices]
    membership = np.asarray([membership_map[int(i)] for i in global_indices])
    position = {int(index): pos for pos, index in enumerate(global_indices)}

    nominal = np.zeros((len(records), 4), dtype=np.float32)
    hierarchy = np.zeros_like(nominal)
    candidate_records = [dict(row) for row in records]
    hard_negative_tokens = 0
    device = torch.device("cuda")
    for fold in range(4):
        saved = _train_fold(fold, dataset, labels, device)
        indices = np.asarray(saved["global_indices"], dtype=np.int64)
        locations = [position[int(i)] for i in indices]
        nominal[locations] = saved["nominal"]
        hierarchy[locations] = saved["hierarchy"]
        hard_negative_tokens += int(saved["hard_negative_tokens"])
        for local, location in enumerate(locations):
            candidate_records[location]["start"] = saved["start"][local]
            candidate_records[location]["end"] = saved["end"][local]

    v36_saved = np.load(V36_CACHE, allow_pickle=True)
    names = v36_saved["names"].tolist(); decisions = v36_saved["decisions"]
    v36 = json.loads((config.OUTPUT_DIR / "task1_oof_risk_v36" / "calibration.json")
                     .read_text(encoding="utf-8"))
    decision = decisions[names.index(v36["expert"])]
    old_probability = np.vstack([row["old_probability"] for row in records])
    corrections = np.asarray([
        [correct_risk_only(row["text"], risk) for risk in range(4)] for row in records
    ], dtype=np.int64)
    old_evidence = _evidence_matrix(records)
    new_evidence = _evidence_matrix(candidate_records)
    baseline_prediction = _v36_predict(old_probability, decision, v36, corrections)
    baseline = _metric(truth, baseline_prediction, old_evidence, np.arange(len(truth)))

    hierarchy_mixes = (0., .15, .30, .50)
    model_weights = (0., .10, .25, .40, .60)
    crossfit_prediction = np.zeros(len(truth), dtype=np.int64)
    crossfit_phrase = np.zeros(len(truth), dtype=np.float32)
    selections, fold_rows = [], []
    for fold in range(4):
        fit = np.flatnonzero(membership != fold)
        held = np.flatnonzero(membership == fold)
        candidates = []
        for hierarchy_mix in hierarchy_mixes:
            continued = (1. - hierarchy_mix) * nominal + hierarchy_mix * hierarchy
            for model_weight in model_weights:
                blended = (1. - model_weight) * old_probability + model_weight * continued
                prediction = _v36_predict(blended, decision, v36, corrections)
                for evidence_source, evidence_matrix in (("old", old_evidence),
                                                         ("continued", new_evidence)):
                    score = _metric(truth, prediction, evidence_matrix, fit)
                    candidates.append((score[2], score[0], score[1], hierarchy_mix,
                                       model_weight, evidence_source, prediction,
                                       evidence_matrix))
        selected = max(candidates, key=lambda row: (row[0], row[1], row[2],
                                                     -row[3], -row[4]))
        _, _, _, hierarchy_mix, model_weight, evidence_source, prediction, evidence_matrix = selected
        crossfit_prediction[held] = prediction[held]
        phrase_values = evidence_matrix[np.arange(len(truth)), prediction]
        crossfit_phrase[held] = phrase_values[held]
        selections.append((hierarchy_mix, model_weight, evidence_source))
        old = _metric(truth, baseline_prediction, old_evidence, held)
        new = _metric(truth, crossfit_prediction, np.tile(crossfit_phrase[:, None], (1, 4)), held)
        # The tiled matrix above makes metric select the already-decoded score.
        fold_rows.append({
            "fold": fold, "posts": int(len(held)),
            "hierarchy_mix": hierarchy_mix, "new_model_weight": model_weight,
            "evidence_source": evidence_source,
            "baseline_task1": old[2], "candidate_task1": new[2],
        })
        print(f"V77 fold={fold} hierarchy={hierarchy_mix:.2f} model={model_weight:.2f} "
              f"evidence={evidence_source} task1 {old[2]:.6f}->{new[2]:.6f}", flush=True)

    cross_risk = float(f1_score(truth, crossfit_prediction, average="weighted", zero_division=0))
    cross_phrase_f1 = float(crossfit_phrase.mean())
    cross_task1 = task1_score(cross_risk, cross_phrase_f1)
    production = Counter(selections).most_common(1)[0][0]
    hm, mw, es = production
    continued = (1. - hm) * nominal + hm * hierarchy
    fixed_probability = (1. - mw) * old_probability + mw * continued
    fixed_prediction = _v36_predict(fixed_probability, decision, v36, corrections)
    fixed_evidence = old_evidence if es == "old" else new_evidence
    fixed = _metric(truth, fixed_prediction, fixed_evidence, np.arange(len(truth)))

    unique = np.unique(local_groups); rng = np.random.default_rng(config.SEED + 7777)
    deltas = []
    base_phrase_values = baseline[3]
    for _ in range(4000):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([np.flatnonzero(local_groups == user) for user in sampled])
        old_risk = f1_score(truth[selected], baseline_prediction[selected],
                            average="weighted", zero_division=0)
        new_risk = f1_score(truth[selected], crossfit_prediction[selected],
                            average="weighted", zero_division=0)
        deltas.append(task1_score(new_risk, float(crossfit_phrase[selected].mean()))
                      - task1_score(old_risk, float(base_phrase_values[selected].mean())))
    deltas = np.asarray(deltas)
    bootstrap = {
        "mean_task1_delta": float(deltas.mean()),
        "p05_task1_delta": float(np.quantile(deltas, .05)),
        "p95_task1_delta": float(np.quantile(deltas, .95)),
        "positive_fraction": float((deltas > 0).mean()),
    }
    adopted = bool(
        cross_task1 >= baseline[2] + .003
        and fixed[2] >= baseline[2] + .002
        and bootstrap["positive_fraction"] >= .80
        and production != (0., 0., "old")
    )
    payload = {
        "training_version": TRAINING_VERSION,
        "method": {
            "shared_heads": ["four_class", "explicit", "attempt",
                             "behavior_vs_ideation", "token_evidence"],
            "hard_negative_scopes": ["third_person", "negation", "hypothetical", "historical"],
            "hard_negative_tokens_seen": hard_negative_tokens,
            "epochs": EPOCHS, "trainable_top_layers": 4,
        },
        "same_user_context_audit": {
            "task1_v12": {"baseline": .789098, "context": .785451},
            "task1_v61_causal_four_posts": {"baseline": .798925, "context": .798278},
            "task2_v64": {"baseline": .608499, "context": .591238},
            "conclusion": "do not adopt; context propagates non-current risk/factors",
        },
        "evaluation": "all 1305 nested user-disjoint OOF posts",
        "baseline_v36": {"risk_f1": baseline[0], "phrase_f1": baseline[1], "task1": baseline[2]},
        "crossfit_candidate": {"risk_f1": cross_risk, "phrase_f1": cross_phrase_f1,
                               "task1": cross_task1, "folds": fold_rows},
        "fixed_production_diagnostic": {
            "hierarchy_mix": hm, "new_model_weight": mw, "evidence_source": es,
            "risk_f1": fixed[0], "phrase_f1": fixed[1], "task1": fixed[2],
        },
        "user_cluster_bootstrap": bootstrap,
        "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({
        "training_version": TRAINING_VERSION, "adopted": adopted,
        "hierarchy_mix": hm, "new_model_weight": mw, "evidence_source": es,
        "baseline_task1": baseline[2], "crossfit_task1": cross_task1,
        "bootstrap": bootstrap,
    }, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main():
    return _main_impl()


if __name__ == "__main__":
    main()

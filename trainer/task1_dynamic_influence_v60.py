"""PFA-DI-inspired causal dynamic factor influence expert for Task 1.

V58 established that user factor trajectories transfer to the leaderboard.
V60 implements the two paper mechanisms V58 omitted: a four-post temporal
attention window and risk/protective factor-state alignment supervised by
observed risk transitions.  The model predicts the current post and never
reads a future post.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from baseline import _post_phrase_f1
from configs.config import config
from inference.task1_evidence_v4 import correct_risk_only
from models.multitask_model_v2 import ordinal_class_probabilities
from preprocess.preprocess import load_train_data
from trainer.task1_factor_trajectory_v58 import (
    FACTOR_PROBABILITIES, _current_baseline, _fold0_factor_probabilities,
    _outer_split, _sequences,
)
from trainer.task1_local_counterfactual_train_v56 import _decode
from utils.seed import seed_everything
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_dynamic_influence_v60"
RESULTS = OUTPUT / "results.json"
SEEDS = (6060, 16060, 26060)
WINDOW = 4
TEMPERATURE = .4
FIXED_WEIGHT = .05


class DynamicInfluenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.risk = nn.Sequential(nn.Linear(19, 40), nn.GELU(), nn.LayerNorm(40))
        self.protective = nn.Sequential(nn.Linear(5, 24), nn.GELU(), nn.LayerNorm(24))
        self.gru = nn.GRU(64, 48, batch_first=True)
        self.temporal_query = nn.Linear(48, 48, bias=False)
        self.risk_state = nn.Linear(40, 48, bias=False)
        self.protective_state = nn.Linear(24, 48, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(48 + 64 + 48, 64), nn.GELU(), nn.Dropout(.15), nn.Linear(64, 4),
        )
        self.ordinal = nn.Linear(48 + 64 + 48, 3)

    def forward(self, factors):
        risk = self.risk(factors[..., :19])
        protective = self.protective(factors[..., 19:])
        current = torch.cat((risk, protective), dim=-1)
        hidden, _ = self.gru(current)
        contexts = []; alignments = []; dynamics = []
        for step in range(hidden.size(1)):
            start = max(0, step - WINDOW + 1)
            history = hidden[:, start:step + 1]
            query = self.temporal_query(hidden[:, step:step + 1])
            score = (history * query).sum(-1) / np.sqrt(history.size(-1))
            attention = torch.softmax(score, dim=-1)
            context = torch.einsum("bt,bth->bh", attention, history)
            risk_summary = torch.einsum("bt,bth->bh", attention, risk[:, start:step + 1])
            protective_summary = torch.einsum(
                "bt,bth->bh", attention, protective[:, start:step + 1],
            )
            risk_vector = self.risk_state(risk_summary)
            protective_vector = self.protective_state(protective_summary)
            state = F.normalize(context, dim=-1)
            alignment = torch.stack((
                (state * F.normalize(risk_vector, dim=-1)).sum(-1),
                (state * F.normalize(protective_vector, dim=-1)).sum(-1),
            ), dim=-1)
            influence = torch.softmax(alignment / TEMPERATURE, dim=-1)
            dynamic = (influence[:, :1] * risk_vector
                       + influence[:, 1:] * protective_vector)
            contexts.append(context); alignments.append(alignment); dynamics.append(dynamic)
        context = torch.stack(contexts, dim=1)
        alignment = torch.stack(alignments, dim=1)
        dynamic = torch.stack(dynamics, dim=1)
        representation = torch.cat((context, current, dynamic), dim=-1)
        return self.classifier(representation), self.ordinal(representation), alignment


def _probability(categorical, ordinal):
    standard = torch.softmax(categorical, -1)
    ordered = ordinal_class_probabilities(ordinal)
    return .75 * standard + .25 * ordered


@torch.no_grad()
def _infer(model, sequences, size, device):
    model.eval(); probability = np.zeros((size, 4), dtype=np.float32)
    for indices, factors, _ in sequences:
        categorical, ordinal, _ = model(factors.unsqueeze(0).to(device))
        probability[indices] = _probability(categorical[0], ordinal[0]).cpu().numpy()
    return probability


def _train_one(frame, factors, labels, fit_idx, valid_idx, seed, epochs=None):
    seed_everything(seed); device = torch.device(config.DEVICE)
    model = DynamicInfluenceModel().to(device)
    counts = np.bincount(labels[fit_idx], minlength=4).astype(np.float32)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1)); weights /= weights.mean()
    class_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-3)
    train_sequences = _sequences(frame, fit_idx, factors, labels)
    valid_sequences = _sequences(frame, valid_idx, factors, labels)
    rng = np.random.default_rng(seed); maximum = int(epochs or 45)
    best = {"score": -1., "epoch": 1, "state": None}; stale = 0
    for epoch in range(1, maximum + 1):
        model.train(); losses = []
        for position in rng.permutation(len(train_sequences)):
            _, values, truth = train_sequences[int(position)]
            values = values.to(device); truth = truth.to(device)
            if epochs is None:
                values = values * (torch.rand_like(values) > .04).float()
            optimizer.zero_grad(set_to_none=True)
            categorical, ordinal, alignment = model(values.unsqueeze(0))
            categorical = categorical[0]; ordinal = ordinal[0]; alignment = alignment[0]
            classification = F.cross_entropy(categorical, truth, weight=class_weight)
            ordinal_target = torch.stack([truth > threshold for threshold in range(3)], -1).float()
            ordinal_loss = F.binary_cross_entropy_with_logits(ordinal, ordinal_target)
            if len(truth) > 1:
                delta = truth[1:] - truth[:-1]
                effectiveness = torch.stack((delta > 0, delta < 0), -1).float()
                alignment_loss = F.binary_cross_entropy_with_logits(
                    alignment[1:] / TEMPERATURE, effectiveness,
                )
            else:
                alignment_loss = classification.new_zeros(())
            loss = classification + .15 * ordinal_loss + .10 * alignment_loss
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); losses.append(float(loss.detach()))
        if epochs is not None:
            continue
        probability = _infer(model, valid_sequences, len(frame), device)
        score = float(f1_score(labels[valid_idx], probability[valid_idx].argmax(1),
                               average="weighted", zero_division=0))
        if score > best["score"] + 1e-5:
            best = {"score": score, "epoch": epoch, "loss": float(np.mean(losses)),
                    "state": {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
        if stale >= 10:
            break
    if epochs is None:
        model.load_state_dict(best.pop("state"))
    return model, best


def _metric(truth, risk, phrase):
    risk_f1 = float(f1_score(truth, risk, average="weighted", zero_division=0))
    phrase_f1 = float(phrase.mean())
    return {"risk_f1": risk_f1, "phrase_f1": phrase_f1,
            "task1": task1_score(risk_f1, phrase_f1)}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    train_idx, valid_idx = _outer_split(frame)
    factors = _fold0_factor_probabilities(frame, train_idx, valid_idx)
    inner_fit_pos, inner_valid_pos = next(StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=config.SEED + 6060,
    ).split(np.zeros(len(train_idx)), labels[train_idx], groups[train_idx]))
    inner_fit, inner_valid = train_idx[inner_fit_pos], train_idx[inner_valid_pos]
    inner = []
    for seed in SEEDS:
        model, selected = _train_one(
            frame, factors, labels, inner_fit, inner_valid, seed,
        )
        inner.append({"seed": seed, **selected}); del model
    epochs = max(1, int(round(np.median([row["epoch"] for row in inner]))))
    sequences = _sequences(frame, valid_idx, factors, labels)
    probabilities = []
    for seed in SEEDS:
        model, _ = _train_one(
            frame, factors, labels, train_idx, valid_idx, seed, epochs=epochs,
        )
        probabilities.append(_infer(model, sequences, len(frame),
                                    torch.device(config.DEVICE))[valid_idx])
        del model
    dynamic = np.mean(probabilities, axis=0)
    baseline_probability, records, starts, ends, parameters = _current_baseline(
        frame, train_idx, valid_idx,
    )
    texts = frame.text.iloc[valid_idx].astype(str).tolist(); truth = labels[valid_idx]
    gold = [list(frame.iloc[int(index)].evidence) for index in valid_idx]
    rows = []; stored = {}
    for weight in (0., .02, .05, .08, .10):
        probability = (1. - weight) * baseline_probability + weight * dynamic
        risk = np.asarray([correct_risk_only(text, int(label))
                           for text, label in zip(texts, probability.argmax(1))])
        evidence = _decode(records, risk, starts, ends, parameters)
        phrase = np.asarray([_post_phrase_f1(a, b) for a, b in zip(evidence, gold)])
        rows.append({"weight": weight, **_metric(truth, risk, phrase),
                     "changed_risk": int(np.sum(risk != stored.get(0., (risk,))[0]))})
        stored[weight] = (risk, phrase)
    baseline = rows[0]; fixed = next(row for row in rows if row["weight"] == FIXED_WEIGHT)
    old_risk, old_phrase = stored[0.]; new_risk, new_phrase = stored[FIXED_WEIGHT]
    unique = np.unique(groups[valid_idx]); rng = np.random.default_rng(606060); deltas = []
    for _ in range(4000):
        sampled = rng.choice(unique, len(unique), replace=True)
        positions = np.concatenate([np.flatnonzero(groups[valid_idx] == user)
                                    for user in sampled])
        old_f1 = f1_score(truth[positions], old_risk[positions],
                          average="weighted", zero_division=0)
        new_f1 = f1_score(truth[positions], new_risk[positions],
                          average="weighted", zero_division=0)
        deltas.append(task1_score(new_f1, new_phrase[positions].mean())
                      - task1_score(old_f1, old_phrase[positions].mean()))
    deltas = np.asarray(deltas)
    bootstrap = {"mean_delta": float(deltas.mean()),
                 "p05_delta": float(np.quantile(deltas, .05)),
                 "p95_delta": float(np.quantile(deltas, .95)),
                 "positive_fraction": float((deltas > 0).mean())}
    adopted = bool(fixed["task1"] >= baseline["task1"] + .003
                   and bootstrap["positive_fraction"] >= .8
                   and bootstrap["p05_delta"] >= 0.)
    payload = {
        "training_version": "task1-pfa-dynamic-influence-v60",
        "evaluation_scope": "outer user-disjoint fold0; causal four-post history",
        "method": {"window": WINDOW, "temperature": TEMPERATURE,
                   "seeds": list(SEEDS), "inner_selected_epochs": epochs,
                   "fixed_weight": FIXED_WEIGHT,
                   "transition_supervision": "risk-up/protective-down alignment"},
        "inner_selection": inner,
        "standalone_risk_f1": float(f1_score(
            truth, dynamic.argmax(1), average="weighted", zero_division=0)),
        "baseline": baseline,
        "fixed_candidate": {**fixed, "confusion": confusion_matrix(
            truth, new_risk, labels=np.arange(4)).tolist()},
        "weight_ablation_outer_diagnostic_only": rows,
        "user_cluster_bootstrap": bootstrap, "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True); return payload


if __name__ == "__main__":
    main()

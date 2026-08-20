"""Paper-inspired causal factor trajectory expert for Task 1 (V58).

Li et al. model risk and protective factors as separate temporal signals.  The
competition differs from that work: it asks for the risk of the *current* post
and evidence from that post.  This strict experiment therefore uses the
current factor representation plus only earlier posts from the same user.  It
never reads a later post and the outer validation users are excluded from the
factor backbone and risk trajectory training.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from analyze_task1_lexical_v11 import _lexical_experts, _transformer_probability
from baseline import _post_phrase_f1
from configs.config import config
from datasets.dataset import SuicideRiskDataset
from datasets.factor_cache_builder import build_factor_cache
from inference.factor_predictor import _checkpoint_probabilities
from inference.task1_evidence_v4 import correct_risk_only
from models.factor_model import MentalRobertaFactorModel
from preprocess.preprocess import load_train_data
from trainer.factor_train import FactorDataset, _collate
from trainer.task1_atomic_v25 import _load_records
from trainer.task1_local_counterfactual_train_v56 import _decode
from trainer.task1_qwen7b_verbalizer_v53 import _softmax
from utils.seed import seed_everything
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_factor_trajectory_v58"
RESULTS = OUTPUT / "results.json"
PREDICTIONS = OUTPUT / "strict_predictions.npz"
FACTOR_PROBABILITIES = OUTPUT / "fold0_factor_probabilities.npz"
FACTOR_CHECKPOINT = config.OUTPUT_DIR / "factor_cv" / "fold0_model.pt"
FIXED_WEIGHT = 0.05
SEEDS = (5858, 15858, 25858)
FULL_CHECKPOINTS = [OUTPUT / f"full_seed{seed}.pt" for seed in SEEDS]
FULL_MANIFEST = OUTPUT / "full_manifest.json"


class FactorTrajectory(nn.Module):
    """Separate risk/protective projections followed by a causal GRU."""

    def __init__(self):
        super().__init__()
        self.risk = nn.Sequential(nn.Linear(19, 32), nn.GELU(), nn.LayerNorm(32))
        self.protective = nn.Sequential(nn.Linear(5, 16), nn.GELU(), nn.LayerNorm(16))
        self.gru = nn.GRU(48, 40, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(40 + 48, 48), nn.GELU(), nn.Dropout(0.15), nn.Linear(48, 4),
        )

    def forward(self, factors):
        risk = self.risk(factors[..., :19])
        protective = self.protective(factors[..., 19:])
        current = torch.cat((risk, protective), dim=-1)
        history, _ = self.gru(current)
        return self.head(torch.cat((history, current), dim=-1))


def _outer_split(frame):
    labels = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), labels, groups))
    return np.asarray(train_idx), np.asarray(valid_idx)


@torch.no_grad()
def _fold0_factor_probabilities(frame, train_idx, valid_idx):
    """Score every row with the factor model trained without outer users."""
    if FACTOR_PROBABILITIES.exists():
        saved = np.load(FACTOR_PROBABILITIES)
        if (np.array_equal(saved["train_idx"], train_idx)
                and np.array_equal(saved["valid_idx"], valid_idx)):
            print("V58 resumed cached fold-0 factor probabilities", flush=True)
            return saved["probabilities"]
    if not FACTOR_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing strict factor checkpoint: {FACTOR_CHECKPOINT}")
    cache = build_factor_cache(train=True)
    dataset = FactorDataset(cache)
    loader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=_collate,
        num_workers=config.NUM_WORKERS,
    )
    device = torch.device(config.DEVICE)
    probability = _checkpoint_probabilities(FACTOR_CHECKPOINT, loader, device)
    np.savez_compressed(
        FACTOR_PROBABILITIES, probabilities=probability,
        train_idx=train_idx, valid_idx=valid_idx,
    )
    return probability.astype(np.float32)


def _sequences(frame, indices, factors, labels):
    allowed = set(map(int, indices))
    result = []
    for _, rows in frame.groupby("anon_user_id", sort=False):
        ordered = [int(index) for index in rows.sort_values("post_id").index
                   if int(index) in allowed]
        if ordered:
            result.append((
                np.asarray(ordered, dtype=np.int64),
                torch.tensor(factors[ordered], dtype=torch.float32),
                torch.tensor(labels[ordered], dtype=torch.long),
            ))
    return result


@torch.no_grad()
def _predict(model, sequences, size, device):
    model.eval()
    probability = np.zeros((size, 4), dtype=np.float32)
    for indices, factors, _ in sequences:
        logits = model(factors.unsqueeze(0).to(device))[0]
        probability[indices] = torch.softmax(logits, -1).cpu().numpy()
    return probability


def _train_one(frame, factors, labels, fit_idx, valid_idx, seed, epochs=None):
    seed_everything(seed)
    device = torch.device(config.DEVICE)
    model = FactorTrajectory().to(device)
    counts = np.bincount(labels[fit_idx], minlength=4).astype(np.float32)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    train_sequences = _sequences(frame, fit_idx, factors, labels)
    valid_sequences = _sequences(frame, valid_idx, factors, labels)
    rng = np.random.default_rng(seed)
    best = {"score": -1.0, "epoch": 1, "state": None}
    maximum = int(epochs or 45)
    stale = 0
    for epoch in range(1, maximum + 1):
        model.train(); order = rng.permutation(len(train_sequences)); losses = []
        for position in order:
            _, values, truth = train_sequences[int(position)]
            values = values.to(device); truth = truth.to(device)
            # Mild factor dropout makes the trajectory robust to Task 2 errors.
            if epochs is None:
                keep = (torch.rand_like(values) > .04).float()
                values = values * keep
            optimizer.zero_grad(set_to_none=True)
            logits = model(values.unsqueeze(0))[0]
            loss = criterion(logits, truth)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); losses.append(float(loss.detach()))
        if epochs is not None:
            continue
        probability = _predict(model, valid_sequences, len(frame), device)
        prediction = probability[valid_idx].argmax(1)
        score = float(f1_score(labels[valid_idx], prediction,
                               average="weighted", zero_division=0))
        if score > best["score"] + 1e-5:
            best = {"score": score, "epoch": epoch,
                    "loss": float(np.mean(losses)),
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


def _fit_trajectory(frame, factors, labels, train_idx, valid_idx):
    groups = frame.anon_user_id.astype(str).to_numpy()
    inner_fit_pos, inner_valid_pos = next(StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=config.SEED + 5858,
    ).split(np.zeros(len(train_idx)), labels[train_idx], groups[train_idx]))
    inner_fit = train_idx[inner_fit_pos]; inner_valid = train_idx[inner_valid_pos]
    inner = []
    for seed in SEEDS:
        model, selected = _train_one(
            frame, factors, labels, inner_fit, inner_valid, seed,
        )
        inner.append({"seed": seed, **selected})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    selected_epochs = int(round(np.median([row["epoch"] for row in inner])))
    selected_epochs = max(1, selected_epochs)
    probabilities = []
    for seed in SEEDS:
        model, _ = _train_one(
            frame, factors, labels, train_idx, valid_idx, seed,
            epochs=selected_epochs,
        )
        sequences = _sequences(frame, valid_idx, factors, labels)
        probabilities.append(_predict(
            model, sequences, len(frame), torch.device(config.DEVICE),
        )[valid_idx])
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.mean(probabilities, axis=0), inner, selected_epochs


def _current_baseline(frame, train_idx, valid_idx):
    dataset = SuicideRiskDataset(config.CACHE_DIR / "train_cache.pt")
    _, _, raw = _load_records(); records = raw["records"]
    transformer = _transformer_probability(dataset, valid_idx, records)
    v52 = torch.load(
        config.OUTPUT_DIR / "task1_rationale_augment_v52" / "strict_predictions.pt",
        map_location="cpu", weights_only=False,
    )
    v57 = torch.load(
        config.OUTPUT_DIR / "task1_local_diverse_cf_v57" / "strict_predictions.pt",
        map_location="cpu", weights_only=False,
    )
    for name, saved in (("V52", v52), ("V57", v57)):
        if not np.array_equal(np.asarray(saved["valid_idx"]), valid_idx):
            raise RuntimeError(f"{name} strict fold differs from V58")
    v52_probability = np.vstack([row["probability"] for row in v52["rows"]])
    v57_probability = np.vstack([row["probability"] for row in v57["rows"]])
    calibration = json.loads((
        config.OUTPUT_DIR / "task1_oof_risk_v36" / "calibration.json"
    ).read_text(encoding="utf-8"))
    lexical_decision = _lexical_experts(frame, train_idx, valid_idx)[calibration["expert"]]
    lexical = _softmax(lexical_decision, float(calibration["temperature"]))
    neural = .8 * transformer + .1 * v52_probability + .1 * v57_probability
    probability = ((1. - float(calibration["lexical_weight"])) * neural
                   + float(calibration["lexical_weight"]) * lexical)
    logits = np.log(np.clip(probability, 1e-8, 1.0))
    logits[:, 0] += float(calibration.get("indicator_bias", 0.0))
    logits[:, 2] += float(calibration.get("behavior_bias", 0.0))
    logits[:, 3] += float(calibration.get("attempt_bias", 0.0))
    logits -= logits.max(1, keepdims=True)
    probability = np.exp(logits); probability /= probability.sum(1, keepdims=True)

    v18 = json.loads((config.OUTPUT_DIR / "task1_candidate_v18" / "results.json")
                     .read_text(encoding="utf-8"))
    v35 = json.loads((config.OUTPUT_DIR / "task1_oof_decoder_v35" / "calibration.json")
                     .read_text(encoding="utf-8"))
    parameters = (v35["parameters_by_predicted_risk"] if v35.get("adopted", False)
                  else v18["evidence_parameters_by_predicted_risk"])
    seed2 = torch.load(
        config.OUTPUT_DIR / "task1_seed_ensemble_v14" / "seed2_valid.pt",
        map_location="cpu", weights_only=False,
    )["rows"]
    starts = [.7 * record["start"] + .2 * seed["start"] + .1 * new["start"]
              for record, seed, new in zip(records, seed2, v57["rows"])]
    ends = [.7 * record["end"] + .2 * seed["end"] + .1 * new["end"]
            for record, seed, new in zip(records, seed2, v57["rows"])]
    return probability, records, starts, ends, parameters


def _risk(texts, probability):
    return np.asarray([
        correct_risk_only(text, int(label))
        for text, label in zip(texts, probability.argmax(1))
    ], dtype=np.int64)


def _metric(truth, risk, phrase):
    risk_f1 = float(f1_score(truth, risk, average="weighted", zero_division=0))
    phrase_f1 = float(np.mean(phrase))
    return {"risk_f1": risk_f1, "phrase_f1": phrase_f1,
            "task1": task1_score(risk_f1, phrase_f1)}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    train_idx, valid_idx = _outer_split(frame)
    factor_probability = _fold0_factor_probabilities(frame, train_idx, valid_idx)
    trajectory, inner, epochs = _fit_trajectory(
        frame, factor_probability, labels, train_idx, valid_idx,
    )
    baseline_probability, records, starts, ends, parameters = _current_baseline(
        frame, train_idx, valid_idx,
    )
    texts = frame.text.iloc[valid_idx].astype(str).tolist()
    truth = labels[valid_idx]
    baseline_risk = _risk(texts, baseline_probability)
    gold = [list(frame.iloc[int(index)].evidence) for index in valid_idx]

    rows = []
    stored = {}
    for weight in (0.0, 0.02, 0.05, 0.10, 0.15):
        probability = (1. - weight) * baseline_probability + weight * trajectory
        risk = _risk(texts, probability)
        evidence = _decode(records, risk, starts, ends, parameters)
        phrase = np.asarray([_post_phrase_f1(a, b) for a, b in zip(evidence, gold)])
        metric = _metric(truth, risk, phrase)
        rows.append({"weight": weight, **metric,
                     "changed_risk": int(np.sum(risk != baseline_risk))})
        stored[weight] = (risk, phrase)
    baseline = rows[0]; fixed = next(row for row in rows if row["weight"] == FIXED_WEIGHT)
    fixed_risk, fixed_phrase = stored[FIXED_WEIGHT]
    base_risk, base_phrase = stored[0.0]

    unique = np.unique(groups[valid_idx]); rng = np.random.default_rng(585858)
    deltas = []
    for _ in range(4000):
        sampled = rng.choice(unique, len(unique), replace=True)
        positions = np.concatenate([
            np.flatnonzero(groups[valid_idx] == user) for user in sampled
        ])
        old_f1 = f1_score(truth[positions], base_risk[positions],
                          average="weighted", zero_division=0)
        new_f1 = f1_score(truth[positions], fixed_risk[positions],
                          average="weighted", zero_division=0)
        deltas.append(task1_score(new_f1, fixed_phrase[positions].mean())
                      - task1_score(old_f1, base_phrase[positions].mean()))
    deltas = np.asarray(deltas)
    bootstrap = {"mean_delta": float(deltas.mean()),
                 "p05_delta": float(np.quantile(deltas, .05)),
                 "p95_delta": float(np.quantile(deltas, .95)),
                 "positive_fraction": float((deltas > 0).mean())}
    adopted = bool(fixed["task1"] >= baseline["task1"] + .003
                   and bootstrap["positive_fraction"] >= .80
                   and bootstrap["p05_delta"] >= 0.)
    np.savez_compressed(
        PREDICTIONS, valid_idx=valid_idx,
        baseline_probability=baseline_probability,
        trajectory_probability=trajectory,
        baseline_risk=base_risk,
        baseline_phrase=base_phrase,
    )
    payload = {
        "training_version": "task1-paper-factor-trajectory-v58",
        "evaluation_scope": "outer user-disjoint fold0; causal history only",
        "method": {
            "factor_backbone": "fold0 MentalRoBERTa trained without validation users",
            "trajectory": "separate 19-risk/5-protective projections plus causal GRU",
            "seeds": list(SEEDS), "inner_selected_epochs": epochs,
            "pre_registered_weight": FIXED_WEIGHT,
        },
        "inner_selection": inner,
        "trajectory_standalone_risk_f1": float(f1_score(
            truth, trajectory.argmax(1), average="weighted", zero_division=0,
        )),
        "baseline": baseline,
        "fixed_candidate": {**fixed, "confusion": confusion_matrix(
            truth, fixed_risk, labels=np.arange(4)).tolist()},
        "weight_ablation_outer_diagnostic_only": rows,
        "user_cluster_bootstrap": bootstrap,
        "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def train_full(force=False):
    """Fit production trajectory heads on leak-free factor OOF probabilities."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (not force and FULL_MANIFEST.exists()
            and all(path.exists() for path in FULL_CHECKPOINTS)):
        payload = json.loads(FULL_MANIFEST.read_text(encoding="utf-8"))
        print(json.dumps(payload, indent=2), flush=True)
        return payload
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(dtype=np.int64)
    saved = np.load(config.OUTPUT_DIR / "factor_cv" / "factor_oof_predictions.npz")
    factors = np.asarray(saved["semantic"], dtype=np.float32)
    if factors.shape != (len(frame), config.NUM_FACTORS):
        raise RuntimeError(f"Unexpected factor OOF shape: {factors.shape}")
    # Eleven epochs were selected solely inside the outer training users in
    # V58, before observing the strict outer fold.
    epochs = 11
    all_indices = np.arange(len(frame), dtype=np.int64)
    histories = []
    for seed, checkpoint in zip(SEEDS, FULL_CHECKPOINTS):
        model, _ = _train_one(
            frame, factors, labels, all_indices, np.asarray([], dtype=np.int64),
            seed, epochs=epochs,
        )
        torch.save(model.state_dict(), checkpoint)
        histories.append({"seed": seed, "epochs": epochs, "checkpoint": str(checkpoint)})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"V58 full trajectory seed={seed} saved", flush=True)
    payload = {
        "training_version": "task1-paper-factor-trajectory-v58-full",
        "experimental_leaderboard_override": True,
        "training_factor_features": "five-fold user-disjoint MentalRoBERTa OOF probabilities",
        "test_factor_features": "five-fold MentalRoBERTa ensemble probabilities",
        "epochs": epochs, "seeds": list(SEEDS), "risk_weight": FIXED_WEIGHT,
        "models": histories,
    }
    FULL_MANIFEST.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()

"""Nested user-disjoint calibration for the Task 1 clinical hierarchy."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from baseline import _post_phrase_f1
from configs.config import config
from inference.task1_clinical_v6 import correct_clinical_risk, fuse_evidence
from inference.task1_evidence_v4 import (
    apply_evidence_policy, decode_model_evidence, load_evidence_calibration,
)
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_clinical_v6"
RAW = config.OUTPUT_DIR / "task1_evidence_v4" / "fold0_raw.pt"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "task1-clinical-v6"


def _deduplicate(values, limit):
    selected, normalized = [], []
    for value in values:
        for part in str(value).split(";"):
            part = part.strip(); norm = " ".join(part.casefold().split())
            if part and not any(norm in old or old in norm for old in normalized):
                selected.append(part); normalized.append(norm)
            if len(selected) >= int(limit):
                return selected
    return selected


def _predict(records, decoded, parameters):
    predictions, evidences, phrase_scores = [], [], []
    for record, model_phrases in zip(records, decoded):
        old_risk = int(np.argmax(record["old_probability"]))
        risk = correct_clinical_risk(
            record["text"], int(record["risk"]), old_risk,
            policy=parameters["risk_policy"],
        )
        if parameters["evidence_mode"] == "current":
            evidence = apply_evidence_policy(
                record["text"], risk, model_phrases,
                policy="predicted_extended_first", topk=parameters["topk"],
            )
        else:
            evidence = fuse_evidence(
                record["text"], risk, model_phrases,
                mode=parameters["evidence_mode"], topk=parameters["topk"],
            )
        evidence = _deduplicate(evidence, parameters["topk"])
        predictions.append(risk); evidences.append(evidence)
        phrase_scores.append(_post_phrase_f1(evidence, record["gold"]))
    return (
        np.asarray(predictions, dtype=np.int64), evidences,
        np.asarray(phrase_scores, dtype=np.float32),
    )


def _evaluate(records, truth, decoded, indices, parameters):
    prediction, evidence, phrase = _predict(records, decoded, parameters)
    indices = np.asarray(indices, dtype=int)
    risk_f1 = float(f1_score(
        truth[indices], prediction[indices], average="weighted", zero_division=0,
    ))
    phrase_f1 = float(phrase[indices].mean())
    return {
        "risk_f1": risk_f1, "phrase_f1": phrase_f1,
        "task1": task1_score(risk_f1, phrase_f1),
        "prediction": prediction, "evidence": evidence, "phrase_scores": phrase,
    }


def _candidate_cache(records, truth, decoder_cache):
    """Compute every candidate once; folds only rescore saved arrays."""
    rows = []
    total = len(decoder_cache) * 4 * 5 * 4
    with tqdm(total=total, desc="clinical-v6 candidates", unit="cfg") as progress:
        for decoder_key, decoded in decoder_cache.items():
            threshold, max_tokens = decoder_key
            for risk_policy in ("current", "floor", "undo_weak", "undo_weak_floor"):
                for evidence_mode in (
                    "current", "predicted_first", "predicted_model_first",
                    "hierarchical_first", "hierarchical_model_first",
                ):
                    for topk in (1, 2, 3, 4):
                        parameters = {
                            "threshold": threshold, "max_tokens": max_tokens,
                            "risk_policy": risk_policy,
                            "evidence_mode": evidence_mode, "topk": topk,
                        }
                        metric = _evaluate(
                            records, truth, decoded, np.arange(len(records)), parameters
                        )
                        rows.append({
                            **parameters,
                            "prediction": metric["prediction"],
                            "phrase_scores": metric["phrase_scores"],
                        })
                        progress.update()
    return rows


def _grid(candidates, truth, indices):
    indices = np.asarray(indices, dtype=int)
    rows = []
    for candidate in candidates:
        risk_f1 = float(f1_score(
            truth[indices], candidate["prediction"][indices],
            average="weighted", zero_division=0,
        ))
        phrase_f1 = float(candidate["phrase_scores"][indices].mean())
        rows.append({
            **{key: value for key, value in candidate.items()
               if key not in {"prediction", "phrase_scores"}},
            "risk_f1": risk_f1, "phrase_f1": phrase_f1,
            "task1": task1_score(risk_f1, phrase_f1),
        })
    rows.sort(key=lambda row: row["task1"], reverse=True)
    return rows


def _decoder(records):
    result = {}
    settings = [(threshold, max_tokens)
                for threshold in (0.50, 0.55, 0.60, 0.65)
                for max_tokens in (8, 12, 16)]
    for threshold, max_tokens in tqdm(
        settings, desc="clinical-v6 evidence decoding", unit="cfg"
    ):
        result[(threshold, max_tokens)] = [
            decode_model_evidence(
                row["text"], row["offsets"], row["start"], row["end"],
                threshold=threshold, max_tokens=max_tokens,
                end_policy="best", limit=5,
            )
            for row in records
        ]
    return result


def _mode(values):
    return Counter(values).most_common(1)[0][0]


def _bootstrap(records, baseline, candidate, rounds=2000):
    users = np.asarray([row["user"] for row in records])
    unique = np.unique(users); rng = np.random.default_rng(config.SEED + 606)
    deltas = []
    for _ in range(rounds):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(users == user) for user in sampled])
        base_risk = f1_score(
            baseline["truth"][indices], baseline["prediction"][indices],
            average="weighted", zero_division=0,
        )
        new_risk = f1_score(
            candidate["truth"][indices], candidate["prediction"][indices],
            average="weighted", zero_division=0,
        )
        base_score = task1_score(base_risk, baseline["phrase_scores"][indices].mean())
        new_score = task1_score(new_risk, candidate["phrase_scores"][indices].mean())
        deltas.append(float(new_score - base_score))
    return {
        "mean_delta": float(np.mean(deltas)),
        "p05_delta": float(np.quantile(deltas, 0.05)),
        "p95_delta": float(np.quantile(deltas, 0.95)),
        "positive_fraction": float(np.mean(np.asarray(deltas) > 0)),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print("clinical-v6: loading saved strict predictions...", flush=True)
    raw = torch.load(RAW, map_location="cpu", weights_only=False)
    records = raw["records"]
    truth = np.asarray([row["truth"] for row in records], dtype=np.int64)
    groups = np.asarray([row["user"] for row in records])
    indices = np.arange(len(records))
    decoder_cache = _decoder(records)
    print("clinical-v6: precomputing candidate predictions once...", flush=True)
    candidates = _candidate_cache(records, truth, decoder_cache)
    evidence_v4 = load_evidence_calibration()
    if evidence_v4 is None:
        raise FileNotFoundError("Task 1 evidence-v4 calibration is not adopted")
    baseline_parameters = {
        "threshold": float(evidence_v4["threshold"]),
        "max_tokens": int(evidence_v4["max_tokens"]),
        "risk_policy": "current", "evidence_mode": "current",
        "topk": int(evidence_v4["topk"]),
    }
    baseline_metric = _evaluate(
        records, truth,
        decoder_cache[(baseline_parameters["threshold"], baseline_parameters["max_tokens"])],
        indices, baseline_parameters,
    )

    crossfit_prediction = np.zeros(len(records), dtype=np.int64)
    crossfit_phrase = np.zeros(len(records), dtype=np.float32)
    selected = []
    for fold, (fit, held) in enumerate(GroupKFold(n_splits=4).split(indices, groups=groups)):
        print(f"clinical-v6: calibrating user fold {fold + 1}/4...", flush=True)
        best = _grid(candidates, truth, fit)[0]
        metric = _evaluate(
            records, truth, decoder_cache[(best["threshold"], best["max_tokens"])],
            held, best,
        )
        crossfit_prediction[held] = metric["prediction"][held]
        crossfit_phrase[held] = metric["phrase_scores"][held]
        selected.append({
            "fold": fold, **best,
            "heldout_risk_f1": float(f1_score(
                truth[held], metric["prediction"][held], average="weighted", zero_division=0,
            )),
            "heldout_phrase_f1": float(metric["phrase_scores"][held].mean()),
        })
    crossfit_risk = float(f1_score(
        truth, crossfit_prediction, average="weighted", zero_division=0,
    ))
    crossfit_phrase_f1 = float(crossfit_phrase.mean())
    fixed = {
        "threshold": float(_mode([row["threshold"] for row in selected])),
        "max_tokens": int(_mode([row["max_tokens"] for row in selected])),
        "risk_policy": _mode([row["risk_policy"] for row in selected]),
        "evidence_mode": _mode([row["evidence_mode"] for row in selected]),
        "topk": int(_mode([row["topk"] for row in selected])),
    }
    fixed_metric = _evaluate(
        records, truth, decoder_cache[(fixed["threshold"], fixed["max_tokens"])],
        indices, fixed,
    )
    baseline_boot = {
        "truth": truth, "prediction": baseline_metric["prediction"],
        "phrase_scores": baseline_metric["phrase_scores"],
    }
    candidate_boot = {
        "truth": truth, "prediction": fixed_metric["prediction"],
        "phrase_scores": fixed_metric["phrase_scores"],
    }
    bootstrap = _bootstrap(records, baseline_boot, candidate_boot)
    adopted = bool(
        fixed_metric["task1"] >= baseline_metric["task1"] + 0.003
        and task1_score(crossfit_risk, crossfit_phrase_f1)
            >= baseline_metric["task1"] + 0.002
        and bootstrap["positive_fraction"] >= 0.80
    )
    def compact(metric):
        return {key: float(metric[key]) for key in ("risk_f1", "phrase_f1", "task1")}
    payload = {
        "training_version": TRAINING_VERSION,
        "baseline": compact(baseline_metric),
        "nested_crossfit": {
            "risk_f1": crossfit_risk, "phrase_f1": crossfit_phrase_f1,
            "task1": task1_score(crossfit_risk, crossfit_phrase_f1), "folds": selected,
        },
        "fixed_production": {
            **fixed, **compact(fixed_metric),
            "confusion": confusion_matrix(truth, fixed_metric["prediction"]).tolist(),
        },
        "optimistic_full_holdout": _grid(candidates, truth, indices)[0],
        "user_cluster_bootstrap": bootstrap,
        "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({
        "training_version": TRAINING_VERSION, "adopted": adopted, **fixed,
        "strict_baseline_task1": baseline_metric["task1"],
        "strict_crossfit_task1": payload["nested_crossfit"]["task1"],
        "strict_fixed_task1": fixed_metric["task1"],
        "bootstrap_positive_fraction": bootstrap["positive_fraction"],
    }, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    main()

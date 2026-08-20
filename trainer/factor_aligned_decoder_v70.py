"""Fold-aligned, label-local decoder repair for Task 2.

Unlike V47/V69, this evaluation uses the exact StratifiedGroupKFold split that
created every accepted OOF probability.  It probes high-AUC labels and strict
definition anchors, but deploys only two fixed count corrections that remain
useful without changing probability models or any of the other 22 labels.
"""
from __future__ import annotations

import json
import re

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from configs.config import config
from inference.factor_meaning_boundary_v54 import strong_meaning_flags
from preprocess.preprocess import load_train_data
from trainer.factor_balanced_calibration_v47 import _components, _rank, _topk
from trainer.factor_sentence_evidence_cv_v27 import _user_bootstrap


OUTPUT = config.OUTPUT_DIR / "factor_aligned_decoder_v70"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "fold-aligned-label-local-decoder-v70"
BASE_RATIO = 1.10
REPAIRS = {
    "interpersonal difficulty": .85,
    "dysfunctional family": .85,
}
DEFINITION_GUARDS = {
    "interpersonal difficulty": re.compile(
        r"(?:\b(?:friend|partner|boyfriend|girlfriend|people|everyone|someone|relationship)\b"
        r".{0,100}\b(?:betray|abandon|reject|break ?up|blocked|left me|argument|fight|conflict)"
        r"|\b(?:betray|abandon|reject|break ?up|blocked|left me|argument|fight|conflict)"
        r".{0,100}\b(?:friend|partner|boyfriend|girlfriend|people|everyone|someone|relationship)\b"
        r"|\bmess(?:ed|ing)? (?:it|things?) up.{0,100}\b(?:people|friend|relationship)\b)",
        re.I | re.S,
    ),
    "dysfunctional family": re.compile(
        r"(?:\b(?:mom|mother|dad|father|parent|family|home)\b.{0,120}"
        r"\b(?:abus|neglect|never (?:gave|gives?) (?:me )?attention|hate[sd]? me"
        r"|kick(?:ed)? me|toxic|fight(?:ing)?|argu(?:e|ing)|controll?ing|unsafe)"
        r"|\b(?:abus|neglect|toxic|controll?ing|unsafe|never (?:got|had) attention).{0,120}"
        r"\b(?:mom|mother|dad|father|parent|family|home)\b)",
        re.I | re.S,
    ),
}
EXPOSURE = re.compile(
    r"(?:\b(?:my|our) (?:brother|sister|mother|mom|father|dad|friend|partner|classmate|relative|family member)\b"
    r"|\bsomeone i (?:know|knew)\b|\bwatched|\bwitnessed"
    r"|\bsaw (?:a |the )?(?:man|woman|person|someone)"
    r"|\bread(?:ing)? about (?:people|someone|a person)).{0,140}"
    r"\b(?:suicid(?:e|al)|attempt(?:ed)? suicide|killed (?:him|her|them)self"
    r"|taking their own life|trying to kill (?:him|her|them)self)\b",
    re.I | re.S,
)


def _folds(frame):
    groups = frame.anon_user_id.astype(str).to_numpy()
    risk = frame.risk_label.to_numpy(np.int64)
    return list(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), risk, groups))


def _decode(probability, targets, folds, ratios=None):
    ratios = ratios or {}
    result = np.zeros_like(targets, dtype=bool)
    for fit, valid in folds:
        ranked = _rank(probability[valid])
        for label in range(config.NUM_FACTORS):
            ratio = float(ratios.get(config.ID2FACTOR[label], BASE_RATIO))
            result[valid, label] = _topk(
                ranked[:, label], float(targets[fit, label].mean()), ratio,
            )
    return result


def _definition_probe(probability, targets, folds, flags, label, boost):
    baseline = np.zeros(len(targets), dtype=bool)
    candidate = np.zeros(len(targets), dtype=bool)
    fold_delta = []
    for fit, valid in folds:
        score = _rank(probability[valid, label][:, None])[:, 0]
        baseline[valid] = _topk(score, targets[fit, label].mean(), BASE_RATIO)
        candidate[valid] = _topk(
            score + float(boost)*flags[valid],
            targets[fit, label].mean(), BASE_RATIO,
        )
        fold_delta.append(float(
            f1_score(targets[valid, label], candidate[valid], zero_division=0)
            - f1_score(targets[valid, label], baseline[valid], zero_division=0)
        ))
    old = float(f1_score(targets[:, label], baseline, zero_division=0))
    new = float(f1_score(targets[:, label], candidate, zero_division=0))
    return {
        "label": config.ID2FACTOR[label], "matches": int(flags.sum()),
        "true_matches": int((flags * targets[:, label]).sum()),
        "baseline_f1": old, "candidate_f1": new, "delta": new-old,
        "fold_delta": fold_delta, "adopted": False,
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    components, targets = _components()
    probability = components["current"].astype(np.float32)
    folds = _folds(frame)
    groups = frame.anon_user_id.astype(str).to_numpy()

    baseline = _decode(probability, targets, folds)
    candidate = _decode(probability, targets, folds, REPAIRS)
    guard_audit = {}
    texts = frame.text.astype(str).tolist()
    for name, pattern in DEFINITION_GUARDS.items():
        label = config.FACTOR2ID[name]
        flags = np.asarray([bool(pattern.search(text)) for text in texts], dtype=bool)
        before = candidate[:, label].copy()
        candidate[:, label] |= baseline[:, label] & flags
        guard_audit[name] = {
            "matches": int(flags.sum()),
            "restored_predictions": int(np.sum(candidate[:, label] & ~before)),
        }
    old = float(f1_score(targets, baseline, average="macro", zero_division=0))
    new = float(f1_score(targets, candidate, average="macro", zero_division=0))
    fold_rows = []
    for fold, (_, valid) in enumerate(folds):
        a = float(f1_score(targets[valid], baseline[valid], average="macro", zero_division=0))
        b = float(f1_score(targets[valid], candidate[valid], average="macro", zero_division=0))
        fold_rows.append({"fold": fold, "baseline": a, "candidate": b, "delta": b-a})
    bootstrap = _user_bootstrap(
        targets, baseline, candidate, groups, seed=707070, draws=5000,
    )
    per_label = []
    changed_ids = {config.FACTOR2ID[name] for name in REPAIRS}
    for label in range(config.NUM_FACTORS):
        a = float(f1_score(targets[:, label], baseline[:, label], zero_division=0))
        b = float(f1_score(targets[:, label], candidate[:, label], zero_division=0))
        per_label.append({
            "label": config.ID2FACTOR[label], "support": int(targets[:, label].sum()),
            "baseline_f1": a, "candidate_f1": b, "delta": b-a,
            "allowed_to_change": bool(label in changed_ids),
        })

    meaning = strong_meaning_flags(texts)
    exposure = np.asarray([bool(EXPOSURE.search(text)) for text in texts], dtype=np.float32)
    definition_probes = [
        _definition_probe(probability, targets, folds, meaning, 23, .25),
        _definition_probe(probability, targets, folds, exposure, 13, .25),
    ]
    unchanged_ok = all(
        abs(row["delta"]) < 1e-12 for row in per_label
        if not row["allowed_to_change"]
    )
    adopted = bool(
        new >= old + .0025 and unchanged_ok
        and all(per_label[config.FACTOR2ID[name]]["delta"] > 0 for name in REPAIRS)
        and sum(row["delta"] >= -1e-12 for row in fold_rows) >= 4
        and bootstrap["p05_delta"] >= 0
        and bootstrap["positive_fraction"] >= .90
    )
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation": "exact OOF-generating user folds; fixed two-label decoder repair",
        "repairs": REPAIRS, "baseline_macro_f1": old,
        "candidate_macro_f1": new, "delta": new-old,
        "folds": fold_rows, "bootstrap": bootstrap, "per_label": per_label,
        "definition_guards": guard_audit,
        "definition_probes_rejected": definition_probes,
        "unchanged_untargeted_labels": unchanged_ok,
        "experimental_adopted": adopted, "production_adopted": False,
    }
    calibration = {
        "training_version": TRAINING_VERSION,
        "experimental_adopted": adopted, "production_adopted": False,
        "base_ratio": BASE_RATIO, "repairs": REPAIRS,
        "definition_guards": list(DEFINITION_GUARDS),
        "training_prevalence": targets.mean(0).tolist(),
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print(json.dumps({
        "training_version": TRAINING_VERSION,
        "baseline_macro_f1": old, "candidate_macro_f1": new,
        "delta": new-old, "target_results": [
            per_label[config.FACTOR2ID[name]] for name in REPAIRS
        ], "folds": fold_rows, "bootstrap": bootstrap,
        "definition_probes_rejected": definition_probes,
        "experimental_adopted": adopted,
    }, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()

"""Pure-NumPy audit of the accepted prototype-MIL V3 OOF predictions."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
V3 = OUT / "factor_cross_encoder_v2"


def _config_labels():
    tree = ast.parse((ROOT / "configs" / "config.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "FACTOR_LABELS"
                   for target in node.targets):
                return ast.literal_eval(node.value)
    raise ValueError("FACTOR_LABELS not found")


def _rank_decode(probability, prevalence, ratio):
    probability = np.asarray(probability)
    prevalence = np.asarray(prevalence)
    ratio = np.broadcast_to(np.asarray(ratio), prevalence.shape)
    prediction = np.zeros_like(probability, dtype=bool)
    n = len(probability)
    for label in range(probability.shape[1]):
        count = min(n, max(1, int(round(n * prevalence[label] * ratio[label]))))
        chosen = np.argpartition(probability[:, label], n - count)[n - count:]
        prediction[chosen, label] = True
    return prediction


def _f1(truth, prediction):
    truth = np.asarray(truth, dtype=bool); prediction = np.asarray(prediction, dtype=bool)
    tp = np.logical_and(truth, prediction).sum(0).astype(float)
    fp = np.logical_and(~truth, prediction).sum(0).astype(float)
    fn = np.logical_and(truth, ~prediction).sum(0).astype(float)
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)


def main():
    labels = _config_labels()
    base_saved = np.load(OUT / "factor_cv" / "factor_oof_predictions.npz")
    truth = base_saved["targets"].astype(bool)
    base = 0.70 * base_saved["semantic"] + 0.30 * base_saved["cpu"]
    old = np.load(OUT / "factor_cross_encoder" / "oof_predictions.npz")["probabilities"]
    new = np.load(V3 / "oof_predictions.npz")["probabilities"]
    result = json.loads((V3 / "cv_results.json").read_text(encoding="utf-8"))
    calibration = json.loads((V3 / "calibration.json").read_text(encoding="utf-8"))

    crossfit = np.zeros_like(truth)
    old_crossfit = np.zeros_like(truth)
    fold_of = np.full(len(truth), -1, dtype=int)
    all_indices = np.arange(len(truth))
    for parameter in result["crossfit_parameters"]:
        fold = int(parameter["fold"])
        valid = np.load(V3 / f"fold{fold}_valid.npz")["valid_indices"].astype(int)
        fit = np.setdiff1d(all_indices, valid)
        prevalence = truth[fit].mean(0)
        mixed = (parameter["base_weight"] * base[valid]
                 + parameter["old_cross_weight"] * old[valid]
                 + parameter["new_cross_weight"] * new[valid])
        crossfit[valid] = _rank_decode(mixed, prevalence, parameter["prevalence_ratio"])
        old_crossfit[valid] = _rank_decode(
            0.5 * base[valid] + 0.5 * old[valid], prevalence, 1.10
        )
        fold_of[valid] = fold

    production_probability = (
        calibration["base_weight"] * base
        + calibration["old_cross_weight"] * old
        + calibration["new_cross_weight"] * new
    )
    production = _rank_decode(
        production_probability, truth.mean(0), calibration["prevalence_ratio"]
    )
    old_f1 = _f1(truth, old_crossfit)
    crossfit_f1 = _f1(truth, crossfit)
    production_f1 = _f1(truth, production)
    rows = []
    for index, label in enumerate(labels):
        rows.append({
            "label": label,
            "support": int(truth[:, index].sum()),
            "old_crossfit_f1": float(old_f1[index]),
            "v3_crossfit_f1": float(crossfit_f1[index]),
            "crossfit_delta": float(crossfit_f1[index] - old_f1[index]),
            "production_oof_f1": float(production_f1[index]),
            "false_positive": int(np.logical_and(~truth[:, index], production[:, index]).sum()),
            "false_negative": int(np.logical_and(truth[:, index], ~production[:, index]).sum()),
        })
    payload = {
        "old_crossfit_macro_f1": float(old_f1.mean()),
        "v3_crossfit_macro_f1": float(crossfit_f1.mean()),
        "production_oof_macro_f1": float(production_f1.mean()),
        "per_label": rows,
        "worst_production": sorted(rows, key=lambda item: item["production_oof_f1"])[:10],
        "largest_crossfit_gains": sorted(rows, key=lambda item: item["crossfit_delta"], reverse=True)[:10],
        "largest_crossfit_losses": sorted(rows, key=lambda item: item["crossfit_delta"])[:10],
    }
    path = V3 / "error_analysis.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

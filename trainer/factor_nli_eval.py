"""Leak-aware evaluation/calibration of the paper-definition NLI expert."""
import json
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from configs.config import config
from inference.factor_nli import (
    TRAIN_NLI_FILE, NLI_CALIBRATION_FILE, _rank_decode,
    compute_nli_probabilities,
)
from preprocess.preprocess import load_train_data
from trainer.factor_cv import OOF_FILE


RESULT_FILE = config.OUTPUT_DIR / "factor_nli" / "evaluation.json"
WEIGHT_GRID = np.linspace(0.0, 1.0, 11)
RATIO_GRID = np.asarray([0.60, 0.75, 0.90, 1.00, 1.10, 1.25, 1.40, 1.60, 2.00])


def _best_label_parameters(base, nli, truth):
    prevalence = float(truth.mean())
    best = (-1.0, 0.0, 1.0)
    for weight in WEIGHT_GRID:
        probability = (1.0 - weight) * base + weight * nli
        for ratio in RATIO_GRID:
            prediction = _rank_decode(
                probability[:, None], np.asarray([prevalence]), np.asarray([ratio])
            )[:, 0]
            score = f1_score(truth, prediction, zero_division=0)
            candidate = (float(score), float(weight), float(ratio))
            # Prefer less reliance on the new expert and a ratio near one on ties.
            if (candidate[0] > best[0] + 1e-12 or
                    (abs(candidate[0] - best[0]) <= 1e-12 and
                     (candidate[1], -abs(candidate[2] - 1.0)) <
                     (best[1], -abs(best[2] - 1.0)))):
                best = candidate
    return best


def evaluate_factor_nli(force=False):
    if not OOF_FILE.exists():
        raise FileNotFoundError(f"Missing strict Task 2 OOF probabilities: {OOF_FILE}")
    frame = load_train_data().reset_index(drop=True)
    saved = np.load(OOF_FILE)
    targets = saved["targets"].astype(np.int8)
    base = (config.FACTOR_SEMANTIC_MODEL_WEIGHT * saved["semantic"]
            + config.FACTOR_CPU_ENSEMBLE_WEIGHT * saved["cpu"])
    nli = compute_nli_probabilities(
        frame.text.tolist(), frame.row_id.astype(str).tolist(), TRAIN_NLI_FILE, force=force
    )
    prevalence = targets.mean(0)
    current_prediction = _rank_decode(base, prevalence, config.FACTOR_TOPK_RATIO)
    current_score = f1_score(targets, current_prediction, average="macro", zero_division=0)

    # Cross-fit every hyperparameter.  A held user's label is never used to
    # select its NLI weight or prevalence multiplier.
    folds = list(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(frame)), frame.risk_label, groups=frame.anon_user_id))
    crossfit_prediction = np.zeros_like(targets, dtype=bool)
    fold_parameters = []
    for fold, (fit, valid) in enumerate(folds):
        weights, ratios = [], []
        for label in range(config.NUM_FACTORS):
            _, weight, ratio = _best_label_parameters(
                base[fit, label], nli[fit, label], targets[fit, label]
            )
            weights.append(weight); ratios.append(ratio)
            mixed = ((1.0 - weight) * base[valid, label]
                     + weight * nli[valid, label])
            prediction = _rank_decode(
                mixed[:, None], np.asarray([targets[fit, label].mean()]),
                np.asarray([ratio]),
            )[:, 0]
            crossfit_prediction[valid, label] = prediction
        fold_parameters.append({"fold": fold, "nli_weights": weights,
                                "prevalence_ratios": ratios})
    crossfit_score = f1_score(
        targets, crossfit_prediction, average="macro", zero_division=0
    )

    # Lower-variance alternative: select one blend weight and one prevalence
    # ratio on four folds, then apply them unchanged to the held-out users.
    # With only 8--30 positives for several labels this is substantially more
    # stable than estimating 48 independent hyperparameters.
    crossfit_global_prediction = np.zeros_like(targets, dtype=bool)
    crossfit_global_parameters = []
    for fold, (fit, valid) in enumerate(folds):
        candidates = []
        fit_prevalence = targets[fit].mean(0)
        for weight in WEIGHT_GRID:
            fit_mixed = (1.0 - weight) * base[fit] + weight * nli[fit]
            valid_mixed = (1.0 - weight) * base[valid] + weight * nli[valid]
            for ratio in RATIO_GRID:
                fit_prediction = _rank_decode(fit_mixed, fit_prevalence, ratio)
                score = f1_score(
                    targets[fit], fit_prediction, average="macro", zero_division=0
                )
                candidates.append((float(score), float(weight), float(ratio), valid_mixed))
        _, weight, ratio, valid_mixed = max(
            candidates, key=lambda x: (x[0], -x[1], -abs(x[2] - 1.0))
        )
        crossfit_global_prediction[valid] = _rank_decode(
            valid_mixed, fit_prevalence, ratio
        )
        crossfit_global_parameters.append({
            "fold": fold, "nli_weight": weight, "prevalence_ratio": ratio,
        })
    crossfit_global_score = f1_score(
        targets, crossfit_global_prediction, average="macro", zero_division=0
    )

    # Fit final per-label parameters on all cross-fitted base predictions for
    # production.  Report this value as optimistic; crossfit_score is the fair
    # estimate used to decide whether the expert is adopted.
    final_weights, final_ratios = [], []
    for label in range(config.NUM_FACTORS):
        _, weight, ratio = _best_label_parameters(
            base[:, label], nli[:, label], targets[:, label]
        )
        final_weights.append(weight); final_ratios.append(ratio)
    final_mixed = ((1.0 - np.asarray(final_weights)[None, :]) * base
                   + np.asarray(final_weights)[None, :] * nli)
    final_prediction = _rank_decode(final_mixed, prevalence, final_ratios)
    fitted_score = f1_score(targets, final_prediction, average="macro", zero_division=0)

    # A low-variance global blend is a useful diagnostic and fallback.
    global_grid = []
    for weight in WEIGHT_GRID:
        mixed = (1.0 - weight) * base + weight * nli
        for ratio in RATIO_GRID:
            prediction = _rank_decode(mixed, prevalence, ratio)
            global_grid.append({
                "nli_weight": float(weight), "prevalence_ratio": float(ratio),
                "macro_f1": float(f1_score(
                    targets, prediction, average="macro", zero_division=0
                )),
            })
    global_grid.sort(key=lambda x: x["macro_f1"], reverse=True)
    per_label = {}
    for label, name in enumerate(config.FACTOR_LABELS):
        per_label[name] = {
            "support": int(targets[:, label].sum()),
            "current_f1": float(f1_score(
                targets[:, label], current_prediction[:, label], zero_division=0
            )),
            "crossfit_nli_f1": float(f1_score(
                targets[:, label], crossfit_prediction[:, label], zero_division=0
            )),
            "final_nli_weight": final_weights[label],
            "final_prevalence_ratio": final_ratios[label],
        }
    payload = {
        "current_oof_macro_f1": float(current_score),
        "nli_crossfit_macro_f1": float(crossfit_score),
        "nli_global_crossfit_macro_f1": float(crossfit_global_score),
        "nli_global_crossfit_parameters": crossfit_global_parameters,
        "nli_fitted_oof_macro_f1_optimistic": float(fitted_score),
        "best_global_grid_optimistic": global_grid[0],
        "per_label": per_label,
        "fold_parameters": fold_parameters,
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    calibration = {
        "model_name": config.FACTOR_NLI_MODEL_NAME,
        "nli_weights": final_weights,
        "prevalence_ratios": final_ratios,
        "training_prevalence": prevalence.tolist(),
        "crossfit_macro_f1": float(crossfit_score),
        "baseline_macro_f1": float(current_score),
        "global_crossfit_macro_f1": float(crossfit_global_score),
        "best_global_nli_weight": global_grid[0]["nli_weight"],
        "best_global_prevalence_ratio": global_grid[0]["prevalence_ratio"],
        "hypotheses": list(config.FACTOR_NLI_HYPOTHESES),
    }
    NLI_CALIBRATION_FILE.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items()
                      if key not in {"per_label", "fold_parameters"}}, indent=2))
    for name, values in per_label.items():
        delta = values["crossfit_nli_f1"] - values["current_f1"]
        print(f"{name:48s} {values['current_f1']:.4f} -> "
              f"{values['crossfit_nli_f1']:.4f} ({delta:+.4f})")
    print(f"Saved NLI evaluation: {RESULT_FILE}")
    return payload


if __name__ == "__main__":
    evaluate_factor_nli()

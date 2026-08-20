"""Strict OOF study of test-available same-user post context for Task 2."""
import json
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from configs.config import config
from inference.factor_nli import TRAIN_NLI_FILE, _rank_decode
from preprocess.preprocess import load_train_data
from trainer.factor_cv import OOF_FILE


OUTPUT = config.OUTPUT_DIR / "factor_user_context_results.json"


def user_aggregate(probabilities, groups, method):
    """Return an other-post summary for every row; singleton users keep self."""
    probabilities = np.asarray(probabilities, dtype=np.float32)
    groups = np.asarray(groups)
    result = np.empty_like(probabilities)
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        values = probabilities[rows]
        if len(rows) == 1:
            result[rows] = values
            continue
        if method == "mean_other":
            result[rows] = (values.sum(0, keepdims=True) - values) / (len(rows) - 1)
        elif method == "max_other":
            for local, row in enumerate(rows):
                result[row] = np.delete(values, local, axis=0).max(0)
        elif method == "top2_other":
            for local, row in enumerate(rows):
                other = np.delete(values, local, axis=0)
                top = np.sort(other, axis=0)[-min(2, len(other)):]
                result[row] = top.mean(0)
        else:
            raise ValueError(method)
    return result


def _crossfit(probability, targets, frame, folds):
    methods = ("mean_other", "top2_other", "max_other")
    own_weights = np.asarray([0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
    ratios = np.asarray([0.80, 0.90, 1.00, 1.10, 1.25])
    aggregates = {method: user_aggregate(
        probability, frame.anon_user_id.to_numpy(), method
    ) for method in methods}
    prediction = np.zeros_like(targets, dtype=bool)
    parameters = []
    for fold, (fit, valid) in enumerate(folds):
        prevalence = targets[fit].mean(0)
        candidates = []
        for method in methods:
            for own_weight in own_weights:
                mixed = own_weight * probability + (1.0 - own_weight) * aggregates[method]
                for ratio in ratios:
                    fit_prediction = _rank_decode(mixed[fit], prevalence, ratio)
                    score = f1_score(
                        targets[fit], fit_prediction, average="macro", zero_division=0
                    )
                    candidates.append((float(score), method, float(own_weight),
                                       float(ratio), mixed[valid]))
        _, method, own_weight, ratio, valid_probability = max(
            candidates, key=lambda x: (x[0], x[2], -abs(x[3] - 1.0))
        )
        prediction[valid] = _rank_decode(valid_probability, prevalence, ratio)
        parameters.append({"fold": fold, "method": method,
                           "own_weight": own_weight, "prevalence_ratio": ratio})
    return prediction, parameters


def analyze_user_context():
    frame = load_train_data().reset_index(drop=True)
    saved = np.load(OOF_FILE)
    targets = saved["targets"].astype(np.int8)
    base = (config.FACTOR_SEMANTIC_MODEL_WEIGHT * saved["semantic"]
            + config.FACTOR_CPU_ENSEMBLE_WEIGHT * saved["cpu"])
    systems = {"base": base}
    if TRAIN_NLI_FILE.exists():
        nli = np.load(TRAIN_NLI_FILE)["probabilities"]
        systems["base_nli_fixed"] = 0.70 * base + 0.30 * nli
    folds = list(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(frame)), frame.risk_label, groups=frame.anon_user_id))
    results = {}
    for name, probability in systems.items():
        baseline = _rank_decode(
            probability, targets.mean(0), 1.0 if "nli" in name else config.FACTOR_TOPK_RATIO
        )
        contextual, parameters = _crossfit(probability, targets, frame, folds)
        results[name] = {
            "no_context_oof_macro_f1": float(f1_score(
                targets, baseline, average="macro", zero_division=0
            )),
            "context_crossfit_macro_f1": float(f1_score(
                targets, contextual, average="macro", zero_division=0
            )),
            "fold_parameters": parameters,
            "per_label": {
                config.ID2FACTOR[j]: {
                    "without_context": float(f1_score(
                        targets[:, j], baseline[:, j], zero_division=0
                    )),
                    "with_context": float(f1_score(
                        targets[:, j], contextual[:, j], zero_division=0
                    )),
                } for j in range(config.NUM_FACTORS)
            },
        }
    OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({name: {k: v for k, v in result.items() if k != "per_label"}
                      for name, result in results.items()}, indent=2))
    print(f"Saved: {OUTPUT}")
    return results


if __name__ == "__main__":
    analyze_user_context()

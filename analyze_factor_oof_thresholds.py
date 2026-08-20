"""Cross-fit Task 2 thresholds so each fold is decoded without its labels."""
import json
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from configs.config import config
from preprocess.preprocess import load_train_data
from utils.factor_calibration import calibrate_factor_thresholds, apply_calibrated_thresholds


def main():
    saved = np.load(config.OUTPUT_DIR / "factor_cv" / "factor_oof_predictions.npz")
    targets = saved["targets"]
    probability = 0.70 * saved["semantic"] + 0.30 * saved["cpu"]
    frame = load_train_data()
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    prediction = np.zeros_like(targets, dtype=bool)
    threshold_prediction = np.zeros_like(targets, dtype=bool)
    fold_scores = []
    for fold, (train_idx, valid_idx) in enumerate(folds.split(
        np.zeros(len(frame)), frame.risk_label.to_numpy(), frame.anon_user_id.to_numpy()
    )):
        thresholds = calibrate_factor_thresholds(targets[train_idx], probability[train_idx])
        threshold_prediction[valid_idx] = probability[valid_idx] >= thresholds[None, :]
        calibrated, _ = apply_calibrated_thresholds(
            probability[valid_idx], thresholds, targets[train_idx].mean(0),
            config.FACTOR_PREVALENCE_FLOOR_RATIO,
            float((targets[train_idx].sum(1) == 0).mean()),
        )
        prediction[valid_idx] = calibrated
        fold_scores.append({
            "fold": fold,
            "threshold_f1": float(f1_score(
                targets[valid_idx], threshold_prediction[valid_idx],
                average="macro", zero_division=0,
            )),
            "calibrated_f1": float(f1_score(
                targets[valid_idx], prediction[valid_idx], average="macro", zero_division=0,
            )),
        })
    result = {
        "folds": fold_scores,
        "crossfit_threshold_macro_f1": float(f1_score(
            targets, threshold_prediction, average="macro", zero_division=0
        )),
        "crossfit_threshold_with_floor_macro_f1": float(f1_score(
            targets, prediction, average="macro", zero_division=0
        )),
    }
    print(json.dumps(result, indent=2))
    (config.OUTPUT_DIR / "factor_cv" / "crossfit_threshold_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

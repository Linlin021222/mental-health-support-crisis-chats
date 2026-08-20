"""Factor-balanced sparse/NB-SVM specialist inspired by CLPsych baselines.

The accepted sparse component uses TF-IDF logistic regression, but its folds
were stratified by Task-1 risk.  V48 retrains it on factor-balanced user folds
and adds an NB-SVM expert: label-specific log-count ratios amplify phrases that
are disproportionately associated with one factor.  The production ensemble
weights and 1.10 prevalence decoder remain fixed, so the OOF comparison does
not select a favourable weight on validation labels.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion
from tqdm import tqdm

from baseline import _factor_probabilities, _fit_factor_models, _vectorizer
from configs.config import config
from inference.factor_nli import _rank_decode
from preprocess.preprocess import load_train_data
from trainer.factor_sentence_evidence_cv_v27 import _user_bootstrap
from utils.multilabel_group_split import multilabel_group_folds, split_audit


OUTPUT = config.OUTPUT_DIR / "factor_balanced_sparse_v48"
OOF_FILE = OUTPUT / "oof_predictions.npz"
RESULTS = OUTPUT / "results.json"
FULL_MODEL = OUTPUT / "full_model.joblib"
TRAINING_VERSION = "factor-balanced-nbsvm-v48"


def _nb_vectorizer():
    return FeatureUnion([
        ("word", TfidfVectorizer(
            lowercase=True, strip_accents="unicode", ngram_range=(1, 4),
            min_df=1, max_df=.995, max_features=140_000, sublinear_tf=True,
        )),
        ("char", TfidfVectorizer(
            lowercase=True, strip_accents="unicode", analyzer="char_wb",
            ngram_range=(3, 6), min_df=2, max_features=180_000,
            sublinear_tf=True,
        )),
    ])


def _log_count_ratio(matrix, target):
    target = np.asarray(target, dtype=bool)
    positive = np.asarray(matrix[target].sum(axis=0)).ravel() + 1.0
    negative = np.asarray(matrix[~target].sum(axis=0)).ravel() + 1.0
    positive /= positive.sum()
    negative /= negative.sum()
    return np.log(positive / negative).astype(np.float32)


def _fit_nb_models(matrix, targets):
    models, ratios = [], []
    for label in range(config.NUM_FACTORS):
        target = targets[:, label].astype(np.int8)
        ratio = _log_count_ratio(matrix, target)
        transformed = matrix.multiply(ratio)
        model = LogisticRegression(
            C=1.5, class_weight="balanced", max_iter=700,
            solver="liblinear", random_state=config.SEED + 4800 + label,
        )
        model.fit(transformed, target)
        models.append(model); ratios.append(ratio)
    return models, np.stack(ratios)


def _nb_probabilities(models, ratios, matrix):
    result = np.zeros((matrix.shape[0], config.NUM_FACTORS), dtype=np.float32)
    for label, (model, ratio) in enumerate(zip(models, ratios)):
        result[:, label] = model.predict_proba(matrix.multiply(ratio))[:, 1]
    return result


def _current_components():
    base_saved = np.load(config.OUTPUT_DIR / "factor_cv" / "factor_oof_predictions.npz")
    semantic = base_saved["semantic"].astype(np.float32)
    old_cpu = base_saved["cpu"].astype(np.float32)
    old_cross = np.load(config.OUTPUT_DIR / "factor_cross_encoder" /
                        "oof_predictions.npz")["probabilities"].astype(np.float32)
    prototype = np.load(config.OUTPUT_DIR / "factor_cross_encoder_v2" /
                        "oof_predictions.npz")["probabilities"].astype(np.float32)
    calibration = json.loads((config.OUTPUT_DIR / "factor_cross_encoder_v2" /
                              "calibration.json").read_text(encoding="utf-8"))
    return semantic, old_cpu, old_cross, prototype, calibration


def _ensemble(semantic, cpu, old_cross, prototype, calibration):
    base = .70 * semantic + .30 * cpu
    return (float(calibration["base_weight"]) * base
            + float(calibration["old_cross_weight"]) * old_cross
            + float(calibration["new_cross_weight"]) * prototype)


def _decode_folds(probability, targets, folds):
    prediction = np.zeros_like(targets, dtype=bool)
    for fit, valid in folds:
        prediction[valid] = _rank_decode(
            probability[valid], targets[fit].mean(0), 1.10,
        )
    return prediction


def _metric_rows(targets, baseline, standard, nb):
    rows = []
    for label in range(config.NUM_FACTORS):
        values = []
        for prediction in (baseline, standard, nb):
            values.append(float(f1_score(
                targets[:, label], prediction[:, label], zero_division=0,
            )))
        rows.append({
            "label": config.ID2FACTOR[label],
            "support": int(targets[:, label].sum()),
            "baseline_f1": values[0],
            "balanced_tfidf_f1": values[1],
            "balanced_nbsvm_f1": values[2],
            "nbsvm_delta": values[2] - values[0],
        })
    return rows


def cross_validate():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    groups = frame.anon_user_id.astype(str).to_numpy()
    risk = frame.risk_label.to_numpy(dtype=np.int64)
    folds = multilabel_group_folds(
        targets, groups, risk, n_splits=config.N_FOLDS, seed=config.SEED + 47,
    )
    standard_oof = np.zeros_like(targets, dtype=np.float32)
    nb_oof = np.zeros_like(targets, dtype=np.float32)
    texts = frame.text.astype(str).to_numpy()
    for fold, (fit, valid) in enumerate(folds):
        print(f"V48 sparse fold {fold}: train={len(fit)} valid={len(valid)}", flush=True)
        standard_vectorizer = _vectorizer()
        train_matrix = standard_vectorizer.fit_transform(texts[fit])
        valid_matrix = standard_vectorizer.transform(texts[valid])
        standard_models = _fit_factor_models(train_matrix, targets[fit])
        standard_oof[valid] = _factor_probabilities(standard_models, valid_matrix)
        # Production must ensemble these balanced folds. Training one final
        # all-data TF-IDF model would be equivalent to the former CPU expert
        # and would not transfer the validated split improvement.
        joblib.dump({
            "training_version": TRAINING_VERSION,
            "kind": "balanced_tfidf_fold",
            "fold": fold,
            "vectorizer": standard_vectorizer,
            "models": standard_models,
        }, OUTPUT / f"fold{fold}_tfidf.joblib")
        del standard_vectorizer, standard_models, train_matrix, valid_matrix

        vectorizer = _nb_vectorizer()
        train_matrix = vectorizer.fit_transform(texts[fit])
        valid_matrix = vectorizer.transform(texts[valid])
        models, ratios = _fit_nb_models(train_matrix, targets[fit])
        nb_oof[valid] = _nb_probabilities(models, ratios, valid_matrix)
        del vectorizer, train_matrix, valid_matrix, models, ratios

    semantic, old_cpu, old_cross, prototype, calibration = _current_components()
    baseline_probability = _ensemble(
        semantic, old_cpu, old_cross, prototype, calibration,
    )
    standard_probability = _ensemble(
        semantic, standard_oof, old_cross, prototype, calibration,
    )
    nb_probability = _ensemble(
        semantic, nb_oof, old_cross, prototype, calibration,
    )
    baseline = _decode_folds(baseline_probability, targets, folds)
    standard = _decode_folds(standard_probability, targets, folds)
    nb = _decode_folds(nb_probability, targets, folds)
    baseline_score = float(f1_score(targets, baseline, average="macro", zero_division=0))
    standard_score = float(f1_score(targets, standard, average="macro", zero_division=0))
    nb_score = float(f1_score(targets, nb, average="macro", zero_division=0))
    standard_bootstrap = _user_bootstrap(
        targets, baseline, standard, groups, seed=484800, draws=4000,
    )
    nb_bootstrap = _user_bootstrap(
        targets, baseline, nb, groups, seed=484801, draws=4000,
    )
    chosen_name, chosen_score, chosen_bootstrap = max(
        (("balanced_tfidf", standard_score, standard_bootstrap),
         ("balanced_nbsvm", nb_score, nb_bootstrap)),
        key=lambda item: item[1],
    )
    adopted = bool(
        chosen_score >= baseline_score + .004
        and chosen_bootstrap["positive_fraction"] >= .80
        and chosen_bootstrap["p05_delta"] >= 0
    )
    per_label = _metric_rows(targets, baseline, standard, nb)
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation": "fixed accepted ensemble weights on factor-balanced user-disjoint OOF",
        "split_audit": split_audit(folds, targets, groups),
        "baseline_macro_f1": baseline_score,
        "balanced_tfidf_macro_f1": standard_score,
        "balanced_tfidf_delta": standard_score - baseline_score,
        "balanced_nbsvm_macro_f1": nb_score,
        "balanced_nbsvm_delta": nb_score - baseline_score,
        "balanced_tfidf_bootstrap": standard_bootstrap,
        "balanced_nbsvm_bootstrap": nb_bootstrap,
        "selected": chosen_name,
        "per_label": per_label,
        "adopted": adopted,
    }
    np.savez_compressed(
        OOF_FILE, balanced_tfidf=standard_oof, balanced_nbsvm=nb_oof,
        targets=targets,
    )
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items()
                      if key not in ("per_label", "split_audit")}, indent=2), flush=True)
    return payload


def train_full(kind=None):
    """Train a production sparse model after the strict gate adopts V48."""
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    if not result.get("adopted"):
        raise RuntimeError("V48 was rejected by strict validation")
    selected = result["selected"] if kind is None else kind
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    texts = frame.text.astype(str).tolist()
    if selected == "balanced_tfidf":
        vectorizer = _vectorizer(); matrix = vectorizer.fit_transform(texts)
        artifact = {"kind": selected, "vectorizer": vectorizer,
                    "models": _fit_factor_models(matrix, targets)}
    else:
        vectorizer = _nb_vectorizer(); matrix = vectorizer.fit_transform(texts)
        models, ratios = _fit_nb_models(matrix, targets)
        artifact = {"kind": selected, "vectorizer": vectorizer,
                    "models": models, "ratios": ratios}
    artifact["training_version"] = TRAINING_VERSION
    joblib.dump(artifact, FULL_MODEL)
    print(f"Saved V48 full sparse model: {FULL_MODEL}", flush=True)
    return FULL_MODEL


if __name__ == "__main__":
    cross_validate()

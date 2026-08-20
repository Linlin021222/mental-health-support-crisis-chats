"""Strict case-preserving lexical/syntax auxiliary experts for both tasks."""
from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.multiclass import OneVsRestClassifier

from analyze_task1_oof_risk_v36 import CACHE as V36_CACHE, _evidence_matrix, _predict
from configs.config import config
from inference.task1_evidence_v4 import correct_risk_only
from preprocess.preprocess import load_train_data
from preprocess.style_syntax import fit_transform_cased, transform_cased
from trainer.factor_balanced_sparse_v48 import _current_components, _ensemble
from trainer.factor_sentence_evidence_cv_v27 import _user_bootstrap
from trainer.task1_atomic_v25 import _load_records
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "case_syntax_v66"
RESULTS = OUTPUT / "results.json"
TRAINING_VERSION = "case-sensitive-syntax-auxiliary-v66"


def _risk_model():
    return OneVsRestClassifier(LogisticRegression(
        C=.5, class_weight="balanced", max_iter=1000,
        solver="liblinear", random_state=6601,
    ))


def _factor_models(matrix, targets, seed):
    models = []
    for label in range(config.NUM_FACTORS):
        target = targets[:, label]
        if len(np.unique(target)) < 2:
            model = DummyClassifier(strategy="constant", constant=int(target[0]))
        else:
            model = LogisticRegression(
                C=1.5, class_weight="balanced", max_iter=700,
                solver="liblinear", random_state=seed+label,
            )
        model.fit(matrix, target); models.append(model)
    return models


def _factor_probability(models, matrix):
    values = []
    for model in models:
        values.append(model.predict_proba(matrix)[:, list(model.classes_).index(1)]
                      if 1 in model.classes_ else np.zeros(matrix.shape[0]))
    return np.column_stack(values).astype(np.float32)


def _task1(frame):
    labels = frame.risk_label.to_numpy(dtype=np.int64)
    records, membership_map, _ = _load_records()
    global_indices = np.asarray([int(row["global_index"]) for row in records])
    truth = labels[global_indices]
    groups = frame.anon_user_id.astype(str).to_numpy()[global_indices]
    membership = np.asarray([membership_map[int(index)] for index in global_indices])
    texts = frame.text.astype(str).to_numpy()[global_indices]
    case_probability = np.zeros((len(records), 4), dtype=np.float32)
    fold_models = []
    for fold in range(4):
        fit = np.flatnonzero(membership != fold); valid = np.flatnonzero(membership == fold)
        vectorizer, matrix = fit_transform_cased(texts[fit])
        model = _risk_model().fit(matrix, truth[fit])
        case_probability[valid] = model.predict_proba(
            transform_cased(vectorizer, texts[valid]))
        fold_models.append((vectorizer, model))
    saved = np.load(V36_CACHE, allow_pickle=True)
    names = saved["names"].tolist(); decisions = saved["decisions"]
    parameters = json.loads((config.OUTPUT_DIR / "task1_oof_risk_v36" / "calibration.json")
                            .read_text(encoding="utf-8"))
    decision = decisions[names.index(parameters["expert"])]
    old_probability = np.vstack([row["old_probability"] for row in records])
    corrections = np.asarray([[correct_risk_only(row["text"], risk) for risk in range(4)]
                              for row in records], dtype=np.int64)
    evidence = _evidence_matrix(records)
    baseline_prediction = _predict(old_probability, decision, parameters, corrections)

    def metric(prediction, indices):
        indices = np.asarray(indices); risk = float(f1_score(
            truth[indices], prediction[indices], average="weighted", zero_division=0))
        phrase = float(evidence[np.arange(len(prediction)), prediction][indices].mean())
        return {"risk_f1": risk, "phrase_f1": phrase, "task1": task1_score(risk, phrase)}

    weights = (0.0, .03, .05, .08, .10, .15, .20)
    crossfit = baseline_prediction.copy(); rows = []
    for outer in range(4):
        fit = np.flatnonzero(membership != outer); valid = np.flatnonzero(membership == outer)
        candidates = []
        for weight in weights:
            mixed = (1.0-weight)*old_probability + weight*case_probability
            prediction = _predict(mixed, decision, parameters, corrections)
            candidates.append((metric(prediction, fit)["task1"], weight, prediction))
        _, weight, prediction = max(candidates, key=lambda row: (row[0], -row[1]))
        crossfit[valid] = prediction[valid]
        old = metric(baseline_prediction, valid); new = metric(crossfit, valid)
        rows.append({"fold": outer, "weight": weight, "baseline": old,
                     "candidate": new, "changed": int(np.sum(crossfit[valid] != baseline_prediction[valid]))})
    baseline = metric(baseline_prediction, np.arange(len(truth)))
    candidate = metric(crossfit, np.arange(len(truth)))
    rng = np.random.default_rng(666601); unique = np.unique(groups); deltas=[]
    for _ in range(3000):
        sampled=rng.choice(unique,len(unique),replace=True)
        idx=np.concatenate([np.flatnonzero(groups==user) for user in sampled])
        deltas.append(metric(crossfit,idx)["task1"]-metric(baseline_prediction,idx)["task1"])
    deltas=np.asarray(deltas)
    return {
        "baseline": baseline, "candidate": candidate,
        "delta": candidate["task1"]-baseline["task1"], "folds": rows,
        "bootstrap": {"mean_delta":float(deltas.mean()),"p05_delta":float(np.quantile(deltas,.05)),
                      "p95_delta":float(np.quantile(deltas,.95)),"positive_fraction":float((deltas>0).mean())},
        "adopted": False,
    }


def _task2(frame):
    targets=np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    texts=frame.text.astype(str).to_numpy(); groups=frame.anon_user_id.astype(str).to_numpy()
    risk=frame.risk_label.to_numpy(dtype=np.int64)
    folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=config.SEED)
               .split(np.zeros(len(frame)),risk,groups))
    cased=np.zeros_like(targets,dtype=np.float32)
    for fold,(fit,valid) in enumerate(folds):
        print(f"V66 Task2 fold {fold}",flush=True)
        vectorizer,matrix=fit_transform_cased(texts[fit])
        models=_factor_models(matrix,targets[fit],666600+100*fold)
        cased[valid]=_factor_probability(models,transform_cased(vectorizer,texts[valid]))
        joblib.dump({"vectorizer":vectorizer,"models":models,"fold":fold,
                     "training_version":TRAINING_VERSION},OUTPUT/f"factor_fold{fold}.joblib",compress=3)
    semantic,cpu,old_cross,prototype,calibration=_current_components()
    baseline_probability=_ensemble(semantic,cpu,old_cross,prototype,calibration)
    candidate_probability=_ensemble(semantic,cased,old_cross,prototype,calibration)
    baseline=np.zeros_like(targets,dtype=bool); candidate=np.zeros_like(targets,dtype=bool)
    from inference.factor_nli import _rank_decode
    fold_rows=[]
    for fold,(fit,valid) in enumerate(folds):
        prevalence=targets[fit].mean(0)
        baseline[valid]=_rank_decode(baseline_probability[valid],prevalence,1.10)
        candidate[valid]=_rank_decode(candidate_probability[valid],prevalence,1.10)
        old=float(f1_score(targets[valid],baseline[valid],average="macro",zero_division=0))
        new=float(f1_score(targets[valid],candidate[valid],average="macro",zero_division=0))
        fold_rows.append({"fold":fold,"baseline":old,"candidate":new,"delta":new-old})
    old=float(f1_score(targets,baseline,average="macro",zero_division=0))
    new=float(f1_score(targets,candidate,average="macro",zero_division=0))
    bootstrap=_user_bootstrap(targets,baseline,candidate,groups,seed=666602,draws=3000)
    per_label=[]
    for label in range(config.NUM_FACTORS):
        a=float(f1_score(targets[:,label],baseline[:,label],zero_division=0))
        b=float(f1_score(targets[:,label],candidate[:,label],zero_division=0))
        per_label.append({"label":config.ID2FACTOR[label],"support":int(targets[:,label].sum()),
                          "baseline_f1":a,"candidate_f1":b,"delta":b-a})
    adopted=bool(new>=old+.004 and bootstrap["positive_fraction"]>=.80 and bootstrap["p05_delta"]>=0)
    return {"baseline_macro_f1":old,"candidate_macro_f1":new,"delta":new-old,
            "folds":fold_rows,"bootstrap":bootstrap,"per_label":per_label,"adopted":adopted}


def _audit(frame):
    texts=frame.text.astype(str).tolist(); ratios=[]
    for text in texts:
        words=[w for w in __import__('re').findall(r"[A-Za-z]+",text)]
        ratios.append(sum(w.isupper() and len(w)>=2 for w in words)/max(1,len(words)))
    ratios=np.asarray(ratios)
    return {"all_caps_majority_posts":int((ratios>=.5).sum()),"all_caps_full_posts":int((ratios>=.9).sum()),
            "uppercase_word_ratio_mean":float(ratios.mean()),"uppercase_word_ratio_p95":float(np.quantile(ratios,.95))}


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True); frame=load_train_data().reset_index(drop=True)
    task1=_task1(frame); task2=_task2(frame)
    payload={"training_version":TRAINING_VERSION,"features":"case-sensitive word/char TF-IDF + 34 style/syntax proxies",
             "audit":_audit(frame),"task1":task1,"task2":task2,
             "adopted":{"task1":task1["adopted"],"task2":task2["adopted"]}}
    RESULTS.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({"training_version":TRAINING_VERSION,"audit":payload["audit"],
                      "task1":{k:v for k,v in task1.items() if k not in ('folds',)},
                      "task2":{k:v for k,v in task2.items() if k not in ('folds','per_label')}},indent=2),flush=True)
    return payload


if __name__=="__main__": main()

"""Transductive semi-supervised ablation for Task 2.

Each outer validation fold is treated as unlabeled.  The cased vocabulary may
observe its text, and a low-weight student may use only the accepted teacher's
most confident positive/negative pseudo-labels.  Gold validation labels are
used solely for the final evaluation.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from configs.config import config
from inference.factor_nli import _rank_decode
from preprocess.preprocess import load_train_data
from preprocess.style_syntax import fit_transform_cased, transform_cased
from trainer.factor_balanced_sparse_v48 import _current_components, _ensemble
from trainer.factor_sentence_evidence_cv_v27 import _user_bootstrap


OUTPUT = config.OUTPUT_DIR / "semi_supervised_v67"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "confidence-pseudo-label-consistency-v67"
PSEUDO_WEIGHT = .15
POSITIVE_FRACTION = .50
NEGATIVE_FRACTION = .20


def _fit_one(train_matrix, train_target, unlabeled_matrix, teacher, prevalence,
             use_pseudo, seed):
    y = np.asarray(train_target, dtype=np.int8)
    x = train_matrix; weights = np.ones(len(y), dtype=np.float32)
    if use_pseudo:
        n = len(teacher)
        positive_count = max(1, int(round(n * prevalence * POSITIVE_FRACTION)))
        negative_count = max(12, int(round(n * NEGATIVE_FRACTION)))
        positive = np.argsort(teacher)[-positive_count:]
        positive_set = set(positive.tolist())
        negative = [int(row) for row in np.argsort(teacher)
                    if int(row) not in positive_set][:negative_count]
        pseudo_rows = np.concatenate([positive, np.asarray(negative, dtype=int)])
        pseudo_y = np.concatenate([
            np.ones(len(positive), dtype=np.int8),
            np.zeros(len(negative), dtype=np.int8),
        ])
        x = sparse.vstack([x, unlabeled_matrix[pseudo_rows]], format="csr")
        y = np.concatenate([y, pseudo_y])
        weights = np.concatenate([
            weights, np.full(len(pseudo_y), PSEUDO_WEIGHT, dtype=np.float32),
        ])
    model = LogisticRegression(
        C=1.5, class_weight="balanced", max_iter=700,
        solver="liblinear", random_state=seed,
    )
    model.fit(x, y, sample_weight=weights)
    return model


def _systems(frame):
    texts=frame.text.astype(str).to_numpy()
    targets=np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    groups=frame.anon_user_id.astype(str).to_numpy()
    risk=frame.risk_label.to_numpy(dtype=np.int64)
    folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=config.SEED)
               .split(np.zeros(len(frame)),risk,groups))
    semantic,cpu,old_cross,prototype,calibration=_current_components()
    teacher=_ensemble(semantic,cpu,old_cross,prototype,calibration)
    vocabulary=np.zeros_like(targets,dtype=np.float32)
    pseudo=np.zeros_like(targets,dtype=np.float32)
    for fold,(fit,valid) in enumerate(folds):
        print(f"V67 semi-supervised fold {fold}",flush=True)
        # Fitting vocabulary on valid text is unsupervised/transductive.  No
        # validation target enters vectorisation or model fitting.
        vectorizer,_=fit_transform_cased(np.concatenate([texts[fit],texts[valid]]))
        fit_matrix=transform_cased(vectorizer,texts[fit])
        valid_matrix=transform_cased(vectorizer,texts[valid])
        prevalence=targets[fit].mean(0)
        for label in range(config.NUM_FACTORS):
            plain=_fit_one(fit_matrix,targets[fit,label],valid_matrix,
                           teacher[valid,label],prevalence[label],False,
                           670000+100*fold+label)
            student=_fit_one(fit_matrix,targets[fit,label],valid_matrix,
                             teacher[valid,label],prevalence[label],True,
                             671000+100*fold+label)
            vocabulary[valid,label]=plain.predict_proba(valid_matrix)[:,1]
            pseudo[valid,label]=student.predict_proba(valid_matrix)[:,1]
    return targets,groups,folds,teacher,(semantic,old_cross,prototype,calibration),vocabulary,pseudo


def _evaluate(name,cpu_probability,targets,groups,folds,parts):
    semantic,old_cross,prototype,calibration=parts
    probability=_ensemble(semantic,cpu_probability,old_cross,prototype,calibration)
    prediction=np.zeros_like(targets,dtype=bool); fold_values=[]
    for fit,valid in folds:
        prediction[valid]=_rank_decode(probability[valid],targets[fit].mean(0),1.10)
        fold_values.append(float(f1_score(targets[valid],prediction[valid],average="macro",zero_division=0)))
    return {"name":name,"probability":probability,"prediction":prediction,
            "macro_f1":float(f1_score(targets,prediction,average="macro",zero_division=0)),
            "fold_macro_f1":fold_values}


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True); frame=load_train_data().reset_index(drop=True)
    targets,groups,folds,teacher,parts,vocabulary,pseudo=_systems(frame)
    old_cpu=_current_components()[1]
    systems=[_evaluate("supervised_baseline",old_cpu,targets,groups,folds,parts),
             _evaluate("unlabeled_vocabulary",vocabulary,targets,groups,folds,parts),
             _evaluate("confidence_pseudo_labels",pseudo,targets,groups,folds,parts)]
    baseline=systems[0]; candidates=[]
    for system in systems[1:]:
        boot=_user_bootstrap(targets,baseline["prediction"],system["prediction"],groups,
                             seed=670067+len(candidates),draws=3000)
        candidates.append({"name":system["name"],"macro_f1":system["macro_f1"],
                           "delta":system["macro_f1"]-baseline["macro_f1"],
                           "fold_macro_f1":system["fold_macro_f1"],"bootstrap":boot})
    selected=max(candidates,key=lambda row:row["macro_f1"])
    adopted=bool(selected["name"]=="confidence_pseudo_labels"
                 and selected["macro_f1"]>=baseline["macro_f1"]+.004
                 and selected["bootstrap"]["positive_fraction"]>=.80
                 and selected["bootstrap"]["p05_delta"]>=0)
    payload={"training_version":TRAINING_VERSION,
             "evaluation":"five outer user-disjoint folds; validation text and teacher probabilities only, never validation labels",
             "pseudo_policy":{"weight":PSEUDO_WEIGHT,"positive_expected_fraction":POSITIVE_FRACTION,
                              "negative_row_fraction":NEGATIVE_FRACTION},
             "baseline_macro_f1":baseline["macro_f1"],"candidates":candidates,
             "selected":selected["name"],"adopted":adopted}
    RESULTS.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    CALIBRATION.write_text(json.dumps({"training_version":TRAINING_VERSION,"adopted":adopted,
                                       "selected":selected["name"],"result":selected},indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2),flush=True); return payload


if __name__=="__main__": main()

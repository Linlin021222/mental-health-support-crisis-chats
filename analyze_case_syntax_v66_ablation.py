"""Disentangle casing from explicit style/syntax in the accepted V66 view."""
from __future__ import annotations

import json

import joblib
import numpy as np
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

from baseline import _vectorizer
from configs.config import config
from inference.factor_nli import _rank_decode
from preprocess.preprocess import load_train_data
from preprocess.style_syntax import (
    cased_vectorizer, numeric_style_syntax, transform_cased,
)
from trainer.case_syntax_v66 import _factor_models, _factor_probability, OUTPUT as V66
from trainer.factor_balanced_sparse_v48 import _current_components, _ensemble


OUTPUT = config.OUTPUT_DIR / "case_syntax_v66" / "ablation.json"


def _score(cpu, parts, targets, folds):
    semantic, old_cross, prototype, calibration = parts
    probability = _ensemble(semantic, cpu, old_cross, prototype, calibration)
    prediction = np.zeros_like(targets, dtype=bool); values=[]
    for fit, valid in folds:
        prediction[valid] = _rank_decode(probability[valid], targets[fit].mean(0), 1.10)
        values.append(float(f1_score(
            targets[valid], prediction[valid], average="macro", zero_division=0,
        )))
    return {"macro_f1":float(f1_score(targets,prediction,average="macro",zero_division=0)),
            "folds":values}


def main():
    frame=load_train_data().reset_index(drop=True); texts=frame.text.astype(str).to_numpy()
    targets=np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    groups=frame.anon_user_id.astype(str).to_numpy(); risk=frame.risk_label.to_numpy(np.int64)
    folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=config.SEED)
               .split(np.zeros(len(frame)),risk,groups))
    semantic,cpu,old_cross,prototype,calibration=_current_components()
    parts=(semantic,old_cross,prototype,calibration)
    cased_only=np.zeros_like(targets,dtype=np.float32)
    lower_syntax=np.zeros_like(targets,dtype=np.float32)
    combined=np.zeros_like(targets,dtype=np.float32)
    for fold,(fit,valid) in enumerate(folds):
        print(f"V66 ablation fold {fold}",flush=True)
        cased=cased_vectorizer(); train=cased.fit_transform(texts[fit]); test=cased.transform(texts[valid])
        cased_only[valid]=_factor_probability(_factor_models(train,targets[fit],680000+fold*100),test)
        lower=_vectorizer(); train=lower.fit_transform(texts[fit]); test=lower.transform(texts[valid])
        train=sparse.hstack([train,numeric_style_syntax(texts[fit])],format="csr")
        test=sparse.hstack([test,numeric_style_syntax(texts[valid])],format="csr")
        lower_syntax[valid]=_factor_probability(_factor_models(train,targets[fit],681000+fold*100),test)
        artifact=joblib.load(V66/f"factor_fold{fold}.joblib")
        combined[valid]=_factor_probability(artifact["models"],transform_cased(artifact["vectorizer"],texts[valid]))
    systems={"lowercase_text_baseline":_score(cpu,parts,targets,folds),
             "case_sensitive_text_only":_score(cased_only,parts,targets,folds),
             "lowercase_text_plus_syntax":_score(lower_syntax,parts,targets,folds),
             "case_sensitive_plus_syntax":_score(combined,parts,targets,folds)}
    base=systems["lowercase_text_baseline"]["macro_f1"]
    for value in systems.values(): value["delta"]=value["macro_f1"]-base
    payload={"evaluation":"five user-disjoint OOF folds; identical fixed production ensemble/decoder",
             "systems":systems}
    OUTPUT.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2),flush=True); return payload


if __name__=="__main__": main()

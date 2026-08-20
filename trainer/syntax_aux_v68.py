"""Production-gated lowercase lexical + explicit syntax Task-2 expert."""
from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from configs.config import config
from inference.factor_nli import _rank_decode
from preprocess.preprocess import load_train_data
from preprocess.style_syntax import fit_transform_lower_syntax, transform_lower_syntax
from trainer.case_syntax_v66 import _factor_models, _factor_probability
from trainer.factor_balanced_sparse_v48 import _current_components, _ensemble
from trainer.factor_sentence_evidence_cv_v27 import _user_bootstrap


OUTPUT=config.OUTPUT_DIR/"syntax_aux_v68"; RESULTS=OUTPUT/"results.json"
TRAINING_VERSION="lowercase-lexical-explicit-syntax-v68"


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True); frame=load_train_data().reset_index(drop=True)
    texts=frame.text.astype(str).to_numpy(); targets=np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    groups=frame.anon_user_id.astype(str).to_numpy(); risk=frame.risk_label.to_numpy(np.int64)
    folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=config.SEED)
               .split(np.zeros(len(frame)),risk,groups))
    syntax=np.zeros_like(targets,dtype=np.float32)
    for fold,(fit,valid) in enumerate(folds):
        print(f"V68 syntax fold {fold}",flush=True)
        vectorizer,matrix=fit_transform_lower_syntax(texts[fit])
        models=_factor_models(matrix,targets[fit],680000+fold*100)
        syntax[valid]=_factor_probability(models,transform_lower_syntax(vectorizer,texts[valid]))
        joblib.dump({"training_version":TRAINING_VERSION,"fold":fold,
                     "vectorizer":vectorizer,"models":models},OUTPUT/f"fold{fold}.joblib",compress=3)
    semantic,cpu,old_cross,prototype,calibration=_current_components()
    base_prob=_ensemble(semantic,cpu,old_cross,prototype,calibration)
    new_prob=_ensemble(semantic,syntax,old_cross,prototype,calibration)
    baseline=np.zeros_like(targets,dtype=bool); candidate=np.zeros_like(targets,dtype=bool); fold_rows=[]
    for fold,(fit,valid) in enumerate(folds):
        prevalence=targets[fit].mean(0)
        baseline[valid]=_rank_decode(base_prob[valid],prevalence,1.10)
        candidate[valid]=_rank_decode(new_prob[valid],prevalence,1.10)
        a=float(f1_score(targets[valid],baseline[valid],average="macro",zero_division=0))
        b=float(f1_score(targets[valid],candidate[valid],average="macro",zero_division=0))
        fold_rows.append({"fold":fold,"baseline":a,"candidate":b,"delta":b-a})
    old=float(f1_score(targets,baseline,average="macro",zero_division=0))
    new=float(f1_score(targets,candidate,average="macro",zero_division=0))
    bootstrap=_user_bootstrap(targets,baseline,candidate,groups,seed=686868,draws=4000)
    adopted=bool(new>=old+.004 and bootstrap["positive_fraction"]>=.80 and bootstrap["p05_delta"]>=0
                 and sum(row["delta"]>=0 for row in fold_rows)>=4)
    per_label=[]
    for label in range(config.NUM_FACTORS):
        a=float(f1_score(targets[:,label],baseline[:,label],zero_division=0)); b=float(f1_score(targets[:,label],candidate[:,label],zero_division=0))
        per_label.append({"label":config.ID2FACTOR[label],"support":int(targets[:,label].sum()),
                          "baseline_f1":a,"candidate_f1":b,"delta":b-a})
    payload={"training_version":TRAINING_VERSION,
             "features":"original lowercase word/char TF-IDF plus 34 bounded syntax/style features",
             "evaluation":"five user-disjoint OOF folds; fixed production ensemble and decoder",
             "baseline_macro_f1":old,"candidate_macro_f1":new,"delta":new-old,
             "folds":fold_rows,"bootstrap":bootstrap,"per_label":per_label,"adopted":adopted}
    RESULTS.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in payload.items() if k!='per_label'},indent=2),flush=True); return payload


if __name__=="__main__": main()

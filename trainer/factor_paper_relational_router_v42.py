"""Paper-constrained selective V41 routing and soft factor dependencies.

Li et al. analyse directed risk->protective co-occurrence, while explicitly
treating protective factors as a separate (not inverse) dimension.  V42 uses
that structure as a constraint and learns all numerical relations from each
outer training partition only.  Relations are soft ranking messages, never
hard implications or exclusions.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from configs.config import config
from inference.factor_nli import _rank_decode
from preprocess.preprocess import load_train_data
from trainer.factor_cross_encoder_v2 import CONFUSION_GROUPS
from trainer.factor_mhlat_v4 import _current_components
from trainer.factor_sentence_evidence_cv_v27 import _user_bootstrap


OUTPUT = config.OUTPUT_DIR / "factor_paper_relational_router_v42"
RESULTS = OUTPUT / "cv_results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "paper-constrained-relational-router-v42"
V41_OOF = config.OUTPUT_DIR / "factor_prototype_contrast_v41" / "oof_predictions.npz"
ACCEPTED_OOF = config.OUTPUT_DIR / "factor_cross_encoder_v2" / "oof_predictions.npz"
ROUTE_WEIGHTS = (0., .10, .20, .35, .50)
GRAPH_ALPHAS = (0., .05, .10, .20, .35)
RATIO = 1.10


def _label_prediction(score, prevalence, ratio=RATIO):
    count=max(1,min(len(score),int(round(len(score)*float(prevalence)*ratio))))
    result=np.zeros(len(score),dtype=bool)
    result[np.argpartition(score,len(score)-count)[len(score)-count:]]=True
    return result


def _blocks(folds, outer, fit_idx):
    allowed=np.zeros(max(max(x[1]) for x in folds)+1,dtype=bool); allowed[fit_idx]=True
    return [valid[allowed[valid]] for fold,(_,valid) in enumerate(folds)
            if fold!=outer and allowed[valid].any()]


def _stable_select(scores, targets, fit_idx, blocks, values, minimum_scale=1.):
    """Per-label selection requiring pooled and block-level improvement."""
    selected=np.zeros(config.NUM_FACTORS,dtype=np.float32); audit=[]
    prevalence=targets[fit_idx].mean(0)
    for label in range(config.NUM_FACTORS):
        truth=targets[fit_idx,label]
        baseline=f1_score(truth,_label_prediction(scores[0][fit_idx,label],prevalence[label]),zero_division=0)
        block_base=[]
        for block in blocks:
            reference=np.setdiff1d(fit_idx,block,assume_unique=False)
            block_base.append(f1_score(
                targets[block,label],_label_prediction(scores[0][block,label],targets[reference,label].mean()),zero_division=0))
        candidates=[]
        for position,value in enumerate(values):
            total=f1_score(truth,_label_prediction(scores[position][fit_idx,label],prevalence[label]),zero_division=0)
            block_values=[]
            for block in blocks:
                reference=np.setdiff1d(fit_idx,block,assume_unique=False)
                block_values.append(f1_score(
                    targets[block,label],_label_prediction(scores[position][block,label],targets[reference,label].mean()),zero_division=0))
            positive=int((np.asarray(block_values)>np.asarray(block_base)+1e-12).sum())
            negative=int((np.asarray(block_values)<np.asarray(block_base)-1e-12).sum())
            candidates.append((float(total),positive,-negative,float(value)))
        best,positive,negative_signed,value=max(candidates,key=lambda x:(x[0],x[1],x[2],-x[3]))
        support=int(truth.sum()); required=(.05 if support<20 else (.02 if support<80 else .01))*minimum_scale
        negative=-negative_signed
        if best<baseline+required or positive<2 or negative>1:
            value=0.; best=float(baseline)
        selected[label]=value
        audit.append({"label":config.ID2FACTOR[label],"support":support,
                      "baseline_f1":float(baseline),"selected_f1":float(best),
                      "positive_blocks":positive,"negative_blocks":negative,
                      "selected_value":float(value)})
    return selected,audit


def _logit(probability):
    p=np.clip(np.asarray(probability,dtype=np.float64),1e-5,1-1e-5)
    return np.log(p)-np.log1p(-p)


def _sigmoid(value): return 1/(1+np.exp(-np.clip(value,-20,20)))


def _confusion_mask():
    mask=np.zeros((config.NUM_FACTORS,config.NUM_FACTORS),dtype=bool)
    for group in CONFUSION_GROUPS:
        for source in group:
            for target in group:
                if source!=target: mask[source,target]=True
    return mask


def _paper_graph(targets):
    """Smoothed directed conditional log-odds under paper constraints."""
    y=np.asarray(targets,dtype=np.float64); n=len(y); smoothing=2.
    support=y.sum(0); prevalence=(support+smoothing)/(n+2*smoothing)
    joint=y.T@y
    p_present=(joint+smoothing)/(support[:,None]+2*smoothing)
    p_absent=(support[None,:]-joint+smoothing)/(n-support[:,None]+2*smoothing)
    edge=_logit(p_present)-_logit(p_absent)
    np.fill_diagonal(edge,0.)
    reliable=(support[:,None]>=15)&(support[None,:]>=8)
    edge*=reliable
    confusion=_confusion_mask()
    source=np.arange(config.NUM_FACTORS)[:,None]
    target=np.arange(config.NUM_FACTORS)[None,:]
    risk_to_protective=(source<19)&(target>=19)
    protective_to_risk=(source>=19)&(target<19)
    same_group=((source<19)&(target<19))|((source>=19)&(target>=19))
    # Paper evidence: risk->protective positive co-occurrence. Protective
    # presence must never suppress a risk label, because it is not its inverse.
    allowed_positive=same_group|risk_to_protective
    allowed_negative=same_group&confusion
    edge=np.where(edge>0,np.where(allowed_positive,edge,0.),
                  np.where(allowed_negative,edge,0.))
    edge[protective_to_risk]=0.
    edge=np.clip(edge,-1.5,1.5)
    norm=np.maximum(np.abs(edge).sum(0,keepdims=True),1.)
    return (edge/norm).astype(np.float32),prevalence.astype(np.float32)


def _graph_scores(probability,graph,prevalence,alpha):
    message=(probability-prevalence[None,:])@graph
    return _sigmoid(_logit(probability)+message*alpha[None,:]).astype(np.float32)


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if not V41_OOF.exists(): raise FileNotFoundError("Run V41 five-fold CV first")
    frame=load_train_data().reset_index(drop=True)
    targets=np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    groups=frame.anon_user_id.astype(str).to_numpy()
    folds=list(StratifiedGroupKFold(n_splits=config.N_FOLDS,shuffle=True,random_state=config.SEED)
               .split(np.zeros(len(frame)),frame.risk_label,groups))
    current,_,accepted_calibration=_current_components()
    accepted=np.load(ACCEPTED_OOF)["probabilities"].astype(np.float32)
    v41=np.load(V41_OOF)["probabilities"].astype(np.float32)
    prototype_weight=float(accepted_calibration["new_cross_weight"])
    routed_prediction=np.zeros_like(targets,dtype=bool)
    relational_prediction=np.zeros_like(targets,dtype=bool)
    baseline_prediction=np.zeros_like(targets,dtype=bool)
    route_parameters=[]; graph_parameters=[]; fold_rows=[]
    for outer,(fit_idx,valid_idx) in enumerate(folds):
        blocks=_blocks(folds,outer,fit_idx)
        route_scores=[current+prototype_weight*w*(v41-accepted) for w in ROUTE_WEIGHTS]
        route_weights,route_audit=_stable_select(
            route_scores,targets,fit_idx,blocks,ROUTE_WEIGHTS,
        )
        routed=current+prototype_weight*route_weights[None,:]*(v41-accepted)
        graph,graph_prevalence=_paper_graph(targets[fit_idx])
        graph_scores=[_graph_scores(routed,graph,graph_prevalence,
                                    np.full(config.NUM_FACTORS,a,dtype=np.float32))
                      for a in GRAPH_ALPHAS]
        graph_alphas,graph_audit=_stable_select(
            graph_scores,targets,fit_idx,blocks,GRAPH_ALPHAS,minimum_scale=.75,
        )
        relational=_graph_scores(routed,graph,graph_prevalence,graph_alphas)
        prevalence=targets[fit_idx].mean(0)
        baseline_prediction[valid_idx]=_rank_decode(current[valid_idx],prevalence,RATIO)
        routed_prediction[valid_idx]=_rank_decode(routed[valid_idx],prevalence,RATIO)
        relational_prediction[valid_idx]=_rank_decode(relational[valid_idx],prevalence,RATIO)
        baseline=f1_score(targets[valid_idx],baseline_prediction[valid_idx],average="macro",zero_division=0)
        route=f1_score(targets[valid_idx],routed_prediction[valid_idx],average="macro",zero_division=0)
        relation=f1_score(targets[valid_idx],relational_prediction[valid_idx],average="macro",zero_division=0)
        fold_rows.append({"fold":outer,"baseline":float(baseline),"route":float(route),
                          "relational":float(relation),"route_delta":float(route-baseline),
                          "relational_delta":float(relation-baseline),
                          "routed_labels":int((route_weights>0).sum()),
                          "graph_labels":int((graph_alphas>0).sum())})
        route_parameters.append(route_weights); graph_parameters.append(graph_alphas)

    baseline=float(f1_score(targets,baseline_prediction,average="macro",zero_division=0))
    routed=float(f1_score(targets,routed_prediction,average="macro",zero_division=0))
    relational=float(f1_score(targets,relational_prediction,average="macro",zero_division=0))
    bootstrap=_user_bootstrap(targets,baseline_prediction,relational_prediction,groups,seed=424242,draws=3000)
    production_route=np.median(np.stack(route_parameters),axis=0)
    production_alpha=np.median(np.stack(graph_parameters),axis=0)
    production_graph,production_prevalence=_paper_graph(targets)
    production_routed=current+prototype_weight*production_route[None,:]*(v41-accepted)
    production_relational=_graph_scores(production_routed,production_graph,production_prevalence,production_alpha)
    production_prediction=_rank_decode(production_relational,targets.mean(0),RATIO)
    production_baseline=_rank_decode(current,targets.mean(0),RATIO)
    production_score=float(f1_score(targets,production_prediction,average="macro",zero_division=0))
    production_base=float(f1_score(targets,production_baseline,average="macro",zero_division=0))
    adopted=bool(relational>=baseline+.003 and bootstrap["positive_fraction"]>=.80
                 and bootstrap["p05_delta"]>=0 and production_score>=production_base+.003)
    per_label=[]
    for label in range(config.NUM_FACTORS):
        per_label.append({"label":config.ID2FACTOR[label],"support":int(targets[:,label].sum()),
                          "route_weight":float(production_route[label]),
                          "graph_alpha":float(production_alpha[label]),
                          "baseline_f1":float(f1_score(targets[:,label],baseline_prediction[:,label],zero_division=0)),
                          "candidate_f1":float(f1_score(targets[:,label],relational_prediction[:,label],zero_division=0))})
    payload={"training_version":TRAINING_VERSION,
             "paper_constraints":{"risk_factors":19,"protective_factors":5,
                                  "risk_to_protective_positive_only":True,
                                  "protective_to_risk_messages":False,
                                  "negative_edges_only_within_confusion_groups":True},
             "nested_baseline_macro_f1":baseline,"nested_route_macro_f1":routed,
             "nested_relational_macro_f1":relational,"nested_delta":relational-baseline,
             "production_baseline_macro_f1":production_base,
             "production_relational_macro_f1":production_score,
             "production_delta":production_score-production_base,
             "production_routed_labels":int((production_route>0).sum()),
             "production_graph_labels":int((production_alpha>0).sum()),
             "user_cluster_bootstrap":bootstrap,"folds":fold_rows,"per_label":per_label,
             "adopted":adopted}
    np.savez_compressed(OUTPUT/"production_calibration.npz",route_weights=production_route,
                        graph_alphas=production_alpha,graph=production_graph,
                        prevalence=production_prevalence)
    RESULTS.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    CALIBRATION.write_text(json.dumps({"training_version":TRAINING_VERSION,"adopted":adopted,
        "route_weights":production_route.tolist(),"graph_alphas":production_alpha.tolist(),
        "nested_macro_f1":relational,"baseline_macro_f1":baseline},indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2),flush=True); return payload


if __name__=="__main__": main()

"""Full nested-OOF audit of the production evidence decoder (V35).

V18 accidentally promoted V7's optimistic outer-holdout parameters.  V35
evaluates those deployed parameters on 1,305 genuinely OOF posts, performs a
four-fold label-conditional cross-fit, and compares both with the stable global
decoder.  No model is retrained and no outer-holdout labels select a recipe.
"""
from __future__ import annotations

from collections import Counter
import json

import numpy as np

from analyze_task1_evidence_v4 import _cue_cache, _decoder_grid
from analyze_task1_evidence_v7 import _candidate_scores, _parameter_key
from configs.config import config
from inference.task1_evidence_v4 import load_evidence_calibration
from trainer.task1_atomic_v25 import _bootstrap, _load_records


OUTPUT = config.OUTPUT_DIR / "task1_oof_decoder_v35"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
CACHE = OUTPUT / "candidate_scores.npz"
TRAINING_VERSION = "task1-full-oof-decoder-calibration-v35"
SHRINKAGE = 24.0


def _compact(parameters):
    return {key: parameters[key] for key in
            ("threshold", "max_tokens", "end_policy", "cue_policy", "topk")}


def _aggregate(rows):
    def mode(values):
        return Counter(values).most_common(1)[0][0]
    def median_member(values):
        median = float(np.median(values))
        return min(values, key=lambda value: (abs(float(value) - median), float(value)))
    return {"threshold": float(median_member([row["threshold"] for row in rows])),
            "max_tokens": int(median_member([row["max_tokens"] for row in rows])),
            "end_policy": mode([row["end_policy"] for row in rows]),
            "cue_policy": mode([row["cue_policy"] for row in rows]),
            "topk": int(median_member([row["topk"] for row in rows]))}


def _load_matrix(records):
    if CACHE.exists():
        saved = np.load(CACHE, allow_pickle=True)
        parameters = saved["parameters"].tolist()
        scores = saved["scores"]
        if scores.shape[1] == len(records):
            print(f"V35 resumed {scores.shape} decoder matrix", flush=True)
            return parameters, scores
    print("V35 decoding complete OOF candidate matrix", flush=True)
    decoded = _decoder_grid(records); cues = _cue_cache(records)
    parameters, scores = _candidate_scores(records, decoded, cues)
    values = np.empty(len(parameters), dtype=object); values[:] = parameters
    np.savez_compressed(CACHE, parameters=values, scores=scores)
    return parameters, scores


def _select(scores, fit_label, fit_all, baseline_index):
    if len(fit_label) < 8:
        return baseline_index
    label_mean = scores[:, fit_label].mean(1)
    global_mean = scores[:, fit_all].mean(1)
    objective = ((len(fit_label) * label_mean + SHRINKAGE * global_mean)
                 / (len(fit_label) + SHRINKAGE))
    selected = int(objective.argmax())
    # A tiny fit-fold win is selection noise across hundreds of decoders.
    if objective[selected] < objective[baseline_index] + .002:
        return baseline_index
    return selected


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records, membership_map, _ = _load_records()
    membership = np.asarray([membership_map[int(row["global_index"])] for row in records])
    groups = np.asarray([str(row["user"]) for row in records])
    predicted_risk = np.asarray([int(row["risk"]) for row in records])
    parameters, scores = _load_matrix(records)
    lookup = {_parameter_key(row): index for index, row in enumerate(parameters)}
    stable = _compact(load_evidence_calibration())
    stable_index = lookup[_parameter_key(stable)]
    stable_scores = scores[stable_index]

    v18 = json.loads((config.OUTPUT_DIR / "task1_candidate_v18" / "calibration.json")
                     .read_text(encoding="utf-8"))
    deployed = np.zeros(len(records), dtype=np.float32)
    for risk in range(4):
        idx = np.flatnonzero(predicted_risk == risk)
        recipe = _compact(v18["evidence_parameters_by_predicted_risk"][config.ID2RISK[risk]])
        deployed[idx] = scores[lookup[_parameter_key(recipe)], idx]

    crossfit = np.zeros(len(records), dtype=np.float32)
    selections = {risk: [] for risk in range(4)}; fold_rows = []
    all_indices = np.arange(len(records))
    for fold in range(4):
        fit = np.flatnonzero(membership != fold); held = np.flatnonzero(membership == fold)
        label_rows = []
        for risk in range(4):
            fit_label = fit[predicted_risk[fit] == risk]
            held_label = held[predicted_risk[held] == risk]
            selected_index = (stable_index if risk == 0 else
                              _select(scores, fit_label, fit, stable_index))
            recipe = _compact(parameters[selected_index]); selections[risk].append(recipe)
            crossfit[held_label] = scores[selected_index, held_label]
            label_rows.append({"risk": config.ID2RISK[risk],
                               "fit_posts": int(len(fit_label)),
                               "held_posts": int(len(held_label)), **recipe,
                               "held_phrase_f1": (float(scores[selected_index, held_label].mean())
                                                   if len(held_label) else None)})
        fold_rows.append({"fold": fold, "posts": int(len(held)), "labels": label_rows,
                          "stable_phrase_f1": float(stable_scores[held].mean()),
                          "deployed_phrase_f1": float(deployed[held].mean()),
                          "crossfit_phrase_f1": float(crossfit[held].mean())})
        print(f"V35 fold={fold} stable={stable_scores[held].mean():.6f} "
              f"deployed={deployed[held].mean():.6f} crossfit={crossfit[held].mean():.6f}", flush=True)

    fixed = {}; fixed_scores = np.zeros(len(records), dtype=np.float32)
    for risk in range(4):
        recipe = stable if risk == 0 else _aggregate(selections[risk])
        idx = np.flatnonzero(predicted_risk == risk)
        fixed[config.ID2RISK[risk]] = recipe
        fixed_scores[idx] = scores[lookup[_parameter_key(recipe)], idx]
    deployed_bootstrap = _bootstrap(groups, deployed, crossfit)
    stable_bootstrap = _bootstrap(groups, stable_scores, crossfit)
    # Adoption is relative to what production actually deploys.  It may select
    # the stable global decoder if cross-fitted label conditioning is no better.
    crossfit_mean = float(crossfit.mean()); stable_mean = float(stable_scores.mean())
    if stable_mean >= crossfit_mean:
        production = {config.ID2RISK[risk]: stable for risk in range(4)}
        production_source = "stable_global"
        estimated = stable_mean
    else:
        production = fixed; production_source = "crossfit_aggregate"
        estimated = crossfit_mean
    adopted = bool(estimated >= float(deployed.mean()) + .003
                   and deployed_bootstrap["positive_fraction"] >= .80)
    payload = {"training_version": TRAINING_VERSION,
        "evaluation_scope": "all 1305 nested OOF posts; decoder selected without their own labels",
        "stable_global": {"parameters": stable, "phrase_f1": stable_mean},
        "currently_deployed_v18_optimistic": {"phrase_f1": float(deployed.mean()),
                                               "parameters": v18["evidence_parameters_by_predicted_risk"]},
        "crossfit_label_conditional": {"phrase_f1": crossfit_mean, "folds": fold_rows},
        "fixed_aggregate_diagnostic": {"phrase_f1": float(fixed_scores.mean()),
                                        "parameters": fixed},
        "recommended_production": {"source": production_source,
                                    "estimated_oof_phrase_f1": estimated,
                                    "parameters_by_predicted_risk": production},
        "bootstrap_crossfit_vs_deployed": deployed_bootstrap,
        "bootstrap_crossfit_vs_stable": stable_bootstrap,
        "adopted": adopted}
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({"training_version": TRAINING_VERSION,
        "adopted": adopted, "source": production_source,
        "parameters_by_predicted_risk": production,
        "oof_phrase_f1": estimated,
        "deployed_oof_phrase_f1": float(deployed.mean())}, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True); return payload


if __name__ == "__main__":
    main()

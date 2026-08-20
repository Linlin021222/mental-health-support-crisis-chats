"""Paper-inspired emoji, style and user-language-profile ablation (V44)."""
from __future__ import annotations

import json
import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze_task1_oof_risk_v36 import CACHE as V36_CACHE, _evidence_matrix, _predict
from configs.config import config
from inference.task1_evidence_v4 import correct_risk_only
from preprocess.preprocess import load_train_data
from trainer.task1_atomic_v25 import _load_records
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_style_personality_v44"
RESULTS = OUTPUT / "results.json"
TRAINING_VERSION = "task1-paper-inspired-style-user-profile-v44"

TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
EMOJI = re.compile(r"[\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF\u2600-\u27BF]")
POS_EMOJI = set("😀😃😄😁😊🙂😍🥰😘😌😇🤗😎🤞👍❤♥💕💖✨🎉")
NEG_EMOJI = set("😔😞😢😭😩😫😖😣😕🙁☹😡🤬😨😰😥😓💔🥀🔪💀")
LEXICONS = [
    {"i", "me", "my", "mine", "myself"},
    {"we", "us", "our", "ours", "ourselves"},
    {"you", "your", "yours", "yourself"},
    {"not", "no", "never", "nothing", "nobody", "neither", "cannot", "can't", "don't"},
    {"always", "never", "completely", "totally", "everything", "nothing", "forever", "impossible"},
    {"will", "going", "tomorrow", "soon", "plan", "planning", "future"},
    {"was", "were", "had", "did", "ago", "yesterday", "previously", "before", "past"},
    {"family", "mother", "father", "mom", "mum", "dad", "sister", "brother", "parent", "parents"},
    {"friend", "friends", "partner", "boyfriend", "girlfriend", "wife", "husband", "people", "someone"},
    {"help", "therapy", "therapist", "doctor", "hospital", "support", "talk", "care"},
    {"die", "dying", "dead", "death", "suicide", "suicidal", "kill", "killing", "life"},
    {"attempt", "attempted", "tried", "survived", "overdose", "overdosed", "woke"},
    {"pills", "rope", "gun", "knife", "razor", "bridge", "jump", "hang", "hanging", "method"},
]


def _post_features(text):
    text = str(text); words = TOKEN.findall(text); lower = [word.casefold() for word in words]
    counts = {word: lower.count(word) for word in set(lower)}
    n = max(len(lower), 1); emojis = EMOJI.findall(text)
    uppercase = sum(word.isupper() and len(word) >= 3 for word in words)
    elongated = len(re.findall(r"(?i)([a-z])\1{2,}", text))
    positive_faces = len(re.findall(r"(?::|=|;)-?[)D]+|<3", text))
    negative_faces = len(re.findall(r"(?::|=|;)-?[(\/]+", text))
    base = [
        np.log1p(len(text)), np.log1p(len(words)),
        len(set(lower)) / n, sum(map(len, lower)) / n,
        text.count("!"), text.count("?"), text.count("..."),
        len(re.findall(r"[!?.,]{3,}", text)), elongated, uppercase / n,
        text.count("\n"), sum(character.isdigit() for character in text) / max(len(text), 1),
        len(emojis), len(set(emojis)),
        sum(emoji in POS_EMOJI for emoji in emojis),
        sum(emoji in NEG_EMOJI for emoji in emojis),
        positive_faces, negative_faces,
    ]
    lexical = [sum(counts.get(word, 0) for word in lexicon) / n for lexicon in LEXICONS]
    return np.asarray(base + lexical, dtype=np.float32)


def build_features(frame):
    post = np.vstack([_post_features(text) for text in frame.text])
    user = np.zeros((len(frame), post.shape[1] * 3 + 1), dtype=np.float32)
    groups = frame.anon_user_id.astype(str).to_numpy()
    for value in np.unique(groups):
        indices = np.flatnonzero(groups == value); block = post[indices]
        summary = np.concatenate((block.mean(0), block.std(0), block.max(0),
                                  [np.log1p(len(indices))])).astype(np.float32)
        user[indices] = summary
    return np.hstack((post, user))


def _model(c_value, balanced):
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=float(c_value), class_weight="balanced" if balanced else None,
        max_iter=3000, solver="lbfgs"))


def _metric(truth, prediction, evidence, indices):
    indices = np.asarray(indices, dtype=np.int64)
    risk = float(f1_score(truth[indices], prediction[indices], average="weighted", zero_division=0))
    phrase_values = evidence[np.arange(len(prediction)), prediction]
    phrase = float(phrase_values[indices].mean())
    return risk, phrase, task1_score(risk, phrase), phrase_values


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(dtype=np.int64)
    records, membership_map, _ = _load_records()
    global_indices = np.asarray([int(row["global_index"]) for row in records])
    truth = labels[global_indices]
    groups = frame.anon_user_id.astype(str).to_numpy()[global_indices]
    membership = np.asarray([membership_map[int(index)] for index in global_indices])
    features = build_features(frame)[global_indices]
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
    baseline = _metric(truth, baseline_prediction, evidence, np.arange(len(truth)))

    c_values = (.001, .003, .01, .03, .1, .3)
    weights = (0., .05, .10, .20, .30)
    crossfit = np.zeros(len(truth), dtype=np.int64); fold_rows = []
    for outer_fold in range(4):
        fit = np.flatnonzero(membership != outer_fold); held = np.flatnonzero(membership == outer_fold)
        fit_folds = [fold for fold in range(4) if fold != outer_fold]
        candidates = []
        for c_value in c_values:
            for balanced in (False, True):
                inner_probability = np.zeros((len(truth), 4), dtype=np.float32)
                for inner_fold in fit_folds:
                    inner_train = np.flatnonzero((membership != outer_fold) & (membership != inner_fold))
                    inner_held = np.flatnonzero(membership == inner_fold)
                    fitted = _model(c_value, balanced).fit(features[inner_train], truth[inner_train])
                    inner_probability[inner_held] = fitted.predict_proba(features[inner_held])
                for weight in weights:
                    blended = (1. - weight) * old_probability + weight * inner_probability
                    prediction = _predict(blended, decision, parameters, corrections)
                    score = _metric(truth, prediction, evidence, fit)
                    candidates.append((score[2], score[0], c_value, balanced, weight))
        _, _, c_value, balanced, weight = max(candidates, key=lambda row: (row[0], row[1]))
        fitted = _model(c_value, balanced).fit(features[fit], truth[fit])
        style = np.zeros_like(old_probability); style[held] = fitted.predict_proba(features[held])
        blended = old_probability.copy(); blended[held] = ((1. - weight) * old_probability[held]
                                                           + weight * style[held])
        prediction = _predict(blended, decision, parameters, corrections)
        crossfit[held] = prediction[held]
        old = _metric(truth, baseline_prediction, evidence, held)
        new = _metric(truth, crossfit, evidence, held)
        fold_rows.append({"fold": outer_fold, "c": c_value, "balanced": balanced,
                          "style_weight": weight,
                          "changed_predictions": int((prediction[held] != baseline_prediction[held]).sum()),
                          "baseline_task1": old[2], "candidate_task1": new[2]})
        print(f"V44 fold={outer_fold} C={c_value} balanced={balanced} weight={weight} "
              f"task1 {old[2]:.6f}->{new[2]:.6f}", flush=True)

    candidate = _metric(truth, crossfit, evidence, np.arange(len(truth)))
    unique = np.unique(groups); rng = np.random.default_rng(config.SEED + 4444); deltas = []
    for _ in range(4000):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([np.flatnonzero(groups == user) for user in sampled])
        old_risk = f1_score(truth[selected], baseline_prediction[selected], average="weighted", zero_division=0)
        new_risk = f1_score(truth[selected], crossfit[selected], average="weighted", zero_division=0)
        deltas.append(task1_score(new_risk, float(candidate[3][selected].mean()))
                      - task1_score(old_risk, float(baseline[3][selected].mean())))
    deltas = np.asarray(deltas)
    bootstrap = {"mean_task1_delta": float(deltas.mean()),
                 "p05_task1_delta": float(np.quantile(deltas, .05)),
                 "p95_task1_delta": float(np.quantile(deltas, .95)),
                 "positive_fraction": float((deltas > 0).mean())}
    adopted = bool(candidate[2] >= baseline[2] + .003
                   and bootstrap["positive_fraction"] >= .80
                   and sum(row["candidate_task1"] >= row["baseline_task1"] for row in fold_rows) >= 3)
    payload = {"training_version": TRAINING_VERSION, "feature_dimension": int(features.shape[1]),
               "evaluation_scope": "four outer user folds; nested style hyperparameter selection",
               "dataset_style_statistics": {"train_users": int(frame.anon_user_id.nunique()),
                                            "mean_posts_per_user": float(len(frame) / frame.anon_user_id.nunique())},
               "baseline_v36": {"risk_f1": baseline[0], "phrase_f1": baseline[1], "task1": baseline[2]},
               "crossfit_candidate": {"risk_f1": candidate[0], "phrase_f1": candidate[1],
                                      "task1": candidate[2], "folds": fold_rows},
               "user_cluster_bootstrap": bootstrap, "adopted": adopted}
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()

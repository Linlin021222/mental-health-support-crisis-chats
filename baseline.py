"""Fast CPU baseline for the competition.

This is used automatically when the installed PyTorch build has no CUDA.  It
keeps the same output schema as the Transformer pipeline and is deliberately
small enough to run on an ordinary laptop.
"""
from collections import Counter, defaultdict
import json
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from configs.config import config
from preprocess.preprocess import load_test_data, load_train_data
from utils.task1_metric import task1_score as competition_task1_score
from utils.task1_metric import composite_score as competition_composite_score


MODEL_FILE = config.OUTPUT_DIR / "cpu_baseline.joblib"

# Definition-aligned cues used only to correct an impossible Indicator output:
# explicit past attempts -> Attempt, means/plans -> Behavior, death wishes ->
# Ideation.  Matches are returned verbatim from the post as evidence.
ATTEMPT_CUE = re.compile(
    r"(?i)(attempted|attempting|tried to (?:kill|end)|tried twice|survived an attempt)"
)
BEHAVIOR_CUE = re.compile(
    r"(?i)(sodium nitrite|bleed out|swallow (?:all|every)|rope|gun|knife|razor|pills|"
    r"shoot myself|hang myself|overdose|wrote (?:my |her )?goodbyes)"
)
IDEATION_CUE = re.compile(
    r"(?i)(suicid(?:e|al)|kill myself|killing myself|want to die|wanting to die|"
    r"wish i (?:was |were )?dead|wish i died|end my life|stop existing|sleep forever|"
    r"not wake up|won.t wake|being dead|if i die|life to be over|do i have to live|won.t be around)"
)


def _vectorizer():
    return FeatureUnion([
        ("word", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 3),
                                 min_df=1, max_df=0.995, max_features=100_000, sublinear_tf=True)),
        ("char", TfidfVectorizer(lowercase=True, strip_accents="unicode", analyzer="char_wb",
                                 ngram_range=(3, 5), min_df=2, max_features=150_000, sublinear_tf=True)),
    ])


def _make_binary_model(y):
    if len(np.unique(y)) < 2:
        return DummyClassifier(strategy="constant", constant=int(y[0]))
    return LogisticRegression(C=2.0, class_weight="balanced", max_iter=500, solver="liblinear")


def _fit_factor_models(matrix, labels):
    models = []
    for factor_index in range(config.NUM_FACTORS):
        model = _make_binary_model(labels[:, factor_index])
        model.fit(matrix, labels[:, factor_index])
        models.append(model)
    return models


def _factor_probabilities(models, matrix):
    return np.column_stack([
        model.predict_proba(matrix)[:, list(model.classes_).index(1)]
        if 1 in model.classes_ else np.zeros(matrix.shape[0])
        for model in models
    ])


def _calibrate_factor_thresholds(train_matrix, train_labels, risk_labels):
    """Choose one threshold per factor using only an inner training split."""
    indices = np.arange(train_labels.shape[0])
    inner_train, calibration = train_test_split(
        indices, test_size=0.25, random_state=config.SEED + 17, stratify=np.asarray(risk_labels)
    )
    models = _fit_factor_models(train_matrix[inner_train], train_labels[inner_train])
    probabilities = _factor_probabilities(models, train_matrix[calibration])
    thresholds = []
    grid = np.arange(0.05, 0.76, 0.025)
    for factor_index in range(config.NUM_FACTORS):
        target = train_labels[calibration, factor_index]
        if target.sum() == 0:
            thresholds.append(config.FACTOR_THRESHOLD)
            continue
        candidates = [(float(f1_score(target, probabilities[:, factor_index] >= threshold,
                                      zero_division=0)), float(threshold)) for threshold in grid]
        # On ties prefer the threshold closest to 0.5 to reduce calibration
        # overfitting for very rare labels.
        _, threshold = max(candidates, key=lambda item: (item[0], -abs(item[1] - 0.5)))
        thresholds.append(threshold)
    return np.asarray(thresholds, dtype=np.float32)


def _evidence_lexicon(frame):
    """Learn high-precision evidence cues solely from annotated train spans."""
    positive_docs = defaultdict(lambda: defaultdict(set))
    full_phrases = defaultdict(Counter)
    texts = [text.casefold() for text in frame.text]
    anchors = {"suicide", "suicidal", "die", "dying", "dead", "death", "kms", "overdose",
               "attempt", "attempted", "kill", "killing", "hang", "hanging", "jump", "shoot",
               "drown", "wake", "life", "wrist", "wrists"}
    for doc_index, row in enumerate(frame.itertuples(index=False)):
        for phrase in row.evidence:
            phrase = " ".join(phrase.split())
            if len(phrase) >= 3:
                risk = int(row.risk_label)
                normalized = phrase.casefold()
                full_phrases[risk][normalized] += 1
                tokens = re.findall(r"[a-z0-9']+", normalized)
                for size in range(1, min(5, len(tokens)) + 1):
                    for start in range(len(tokens) - size + 1):
                        cue_tokens = tokens[start:start + size]
                        if size == 1 and cue_tokens[0] not in anchors:
                            continue
                        cue = " ".join(cue_tokens)
                        if len(cue) >= 3:
                            positive_docs[risk][cue].add(doc_index)

    result = {}
    for risk in range(config.NUM_RISK_CLASSES):
        ranked = []
        for cue, docs in positive_docs[risk].items():
            positive = len(docs)
            containing = sum(cue in text for text in texts)
            precision = positive / max(1, containing)
            words = len(cue.split())
            # Single words are broad; multiword cues can be retained at a
            # slightly lower empirical precision when repeated.
            keep = (words == 1 and positive >= 2 and precision >= 0.55) or \
                   (words >= 2 and positive >= 2 and precision >= 0.35)
            if keep:
                score = precision + 0.03 * min(positive, 10) + 0.01 * min(words, 5)
                ranked.append((score, cue))
        # Exact phrases remain useful for rare, highly specific expressions.
        for phrase, count in full_phrases[risk].items():
            ranked.append((0.70 + 0.03 * min(count, 10), phrase))
        seen = set()
        result[risk] = [cue for _, cue in sorted(ranked, reverse=True)
                        if not (cue in seen or seen.add(cue))][:1000]
    return result


def train():
    frame = load_train_data()
    texts = frame.text.tolist()
    vectorizer = _vectorizer()
    matrix = vectorizer.fit_transform(texts)
    risk_model = LinearSVC(C=1.0, class_weight="balanced")
    risk_model.fit(matrix, frame.risk_label)
    factors = np.vstack(frame.factor_vector.to_numpy())
    factor_models = _fit_factor_models(matrix, factors)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "risk_model": risk_model,
                 "factor_models": factor_models, "lexicon": _evidence_lexicon(frame)}, MODEL_FILE)
    print(f"CPU baseline trained on {len(frame)} posts: {MODEL_FILE}")


def _extract_evidence(text, risk_id, lexicon):
    # By task definition, Indicator contains no explicit suicide mention and
    # therefore must have no suicide-risk evidence span.
    if int(risk_id) == config.RISK_LABELS["Indicator"]:
        return []
    lowered = text.casefold()
    # Prefer phrases annotated for the predicted risk, then consider all labels.
    phrases = list(lexicon.get(int(risk_id), []))
    phrases.extend(p for rid, entries in lexicon.items() if rid != risk_id for p in entries)
    found = []
    for phrase in phrases:
        start = lowered.find(phrase)
        if start < 0:
            continue
        original = text[start:start + len(phrase)].strip()
        norm = " ".join(original.casefold().split())
        if original and not any(norm in x.casefold() or x.casefold() in norm for x in found):
            found.append(original)
        if len(found) == 5:
            break
    return found


def _apply_task1_rules(text, predicted_risk, predicted_evidence):
    """Enforce the label definitions and prepend verbatim high-confidence cues."""
    predicted_risk = int(predicted_risk)
    attempt = ATTEMPT_CUE.search(text)
    behavior = BEHAVIOR_CUE.search(text)
    ideation = IDEATION_CUE.search(text)
    indicator_id = config.RISK_LABELS["Indicator"]
    if predicted_risk == indicator_id:
        if attempt:
            predicted_risk = config.RISK_LABELS["Attempt"]
        elif behavior:
            predicted_risk = config.RISK_LABELS["Behavior"]
        elif ideation:
            predicted_risk = config.RISK_LABELS["Ideation"]

    cue_phrases = [match.group(0) for pattern in (ATTEMPT_CUE, BEHAVIOR_CUE, IDEATION_CUE)
                   for match in pattern.finditer(text)]
    selected = []
    for phrase in cue_phrases + list(predicted_evidence):
        normalized = " ".join(phrase.casefold().split())
        if phrase and not any(normalized in old.casefold() or old.casefold() in normalized for old in selected):
            selected.append(phrase)
        if len(selected) == 5:
            break
    return predicted_risk, ([] if predicted_risk == indicator_id else selected)


def _phrase_match(prediction, target):
    prediction = " ".join(str(prediction).casefold().split())
    target = " ".join(str(target).casefold().split())
    if not prediction or not target:
        return False
    contained = prediction in target or target in prediction
    length_ok = len(prediction.split()) <= 3 * max(1, len(target.split()))
    return contained and length_ok


def _post_phrase_f1(predictions, targets):
    """Official containment/length rules with maximum one-to-one matching."""
    if not predictions and not targets:
        return 1.0
    edges = [[j for j, target in enumerate(targets) if _phrase_match(pred, target)]
             for pred in predictions]
    matched = {}

    def augment(pred_index, visited):
        for target_index in edges[pred_index]:
            if target_index in visited:
                continue
            visited.add(target_index)
            if target_index not in matched or augment(matched[target_index], visited):
                matched[target_index] = pred_index
                return True
        return False

    true_positive = sum(augment(i, set()) for i in range(len(predictions)))
    precision = true_positive / len(predictions) if predictions else 0.0
    recall = true_positive / len(targets) if targets else 0.0
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _evaluate_holdout(frame, train_indices, test_indices, split_name, file_stem):
    private_train = frame.iloc[train_indices].copy()
    private_test = frame.iloc[test_indices].copy()

    # Everything learned from text—including the vocabulary and evidence
    # lexicon—is fitted strictly on private_train.
    vectorizer = _vectorizer()
    train_matrix = vectorizer.fit_transform(private_train.text)
    test_matrix = vectorizer.transform(private_test.text)
    model = LinearSVC(C=1.0, class_weight="balanced")
    model.fit(train_matrix, private_train.risk_label)
    risk_predictions = model.predict(test_matrix)

    train_factors = np.vstack(private_train.factor_vector.to_numpy())
    test_factors = np.vstack(private_test.factor_vector.to_numpy())
    factor_thresholds = _calibrate_factor_thresholds(
        train_matrix, train_factors, private_train.risk_label.to_numpy()
    )
    factor_models = _fit_factor_models(train_matrix, train_factors)
    factor_probabilities = _factor_probabilities(factor_models, test_matrix)
    factor_predictions = (factor_probabilities >= factor_thresholds).astype(int)
    lexicon = _evidence_lexicon(private_train)
    evidence_predictions = []
    corrected_risks = []
    for i, row in enumerate(private_test.itertuples(index=False)):
        raw_evidence = _extract_evidence(row.text, risk_predictions[i], lexicon)
        corrected_risk, corrected_evidence = _apply_task1_rules(row.text, risk_predictions[i], raw_evidence)
        corrected_risks.append(corrected_risk)
        evidence_predictions.append(corrected_evidence)
    risk_predictions = np.asarray(corrected_risks)
    phrase_scores = [_post_phrase_f1(pred, gold)
                     for pred, gold in zip(evidence_predictions, private_test.evidence)]

    risk_score = float(f1_score(private_test.risk_label, risk_predictions, average="weighted"))
    phrase_score = float(np.mean(phrase_scores))
    task1_score = competition_task1_score(risk_score, phrase_score)
    factor_score = float(f1_score(test_factors, factor_predictions, average="macro", zero_division=0))
    composite_score = competition_composite_score(risk_score, phrase_score, factor_score)
    overlapping_users = len(set(private_train.anon_user_id) & set(private_test.anon_user_id))
    metrics = {
        "split": split_name,
        "seed": config.SEED,
        "train_posts": int(len(private_train)),
        "private_test_posts": int(len(private_test)),
        "train_users": int(private_train.anon_user_id.nunique()),
        "private_test_users": int(private_test.anon_user_id.nunique()),
        "overlapping_users": int(overlapping_users),
        "risk_weighted_f1": risk_score,
        "phrase_f1": phrase_score,
        "task1_score": task1_score,
        "task2_factor_macro_f1": factor_score,
        "composite_score": composite_score,
        "factor_thresholds": {config.ID2FACTOR[j]: float(value)
                              for j, value in enumerate(factor_thresholds)},
    }
    details = pd.DataFrame({
        "row_id": private_test.row_id,
        "anon_user_id": private_test.anon_user_id,
        "post": private_test.text,
        "gold_risk": [config.ID2RISK[int(x)] for x in private_test.risk_label],
        "pred_risk": [config.ID2RISK[int(x)] for x in risk_predictions],
        "gold_evidence": ["; ".join(x) for x in private_test.evidence],
        "pred_evidence": ["; ".join(x) for x in evidence_predictions],
        "phrase_f1": phrase_scores,
        "gold_factors": ["; ".join(config.ID2FACTOR[j] for j, value in enumerate(row) if value)
                         for row in test_factors],
        "pred_factors": ["; ".join(config.ID2FACTOR[j] for j, value in enumerate(row) if value)
                         for row in factor_predictions],
    })
    details.to_csv(config.OUTPUT_DIR / f"{file_stem}_predictions.csv", index=False, encoding="utf-8-sig")
    return metrics


def private_evaluate():
    """Run a comparable 20% post holdout and a stricter user-disjoint check."""
    frame = load_train_data().reset_index(drop=True)
    all_indices = np.arange(len(frame))
    train_idx, test_idx = train_test_split(
        all_indices, test_size=0.20, random_state=config.SEED, stratify=frame.risk_label
    )
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparable = _evaluate_holdout(
        frame, train_idx, test_idx,
        "stratified random 80/20 post holdout (primary comparison)",
        "private_post_holdout",
    )

    group_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    group_train, group_test = next(group_splitter.split(
        frame.text, frame.risk_label, groups=frame.anon_user_id
    ))
    strict = _evaluate_holdout(
        frame, group_train, group_test,
        "5-fold stratified user-group holdout (strict diagnostic)",
        "private_user_holdout",
    )
    report = {"primary_post_holdout": comparable, "strict_user_holdout": strict}
    report_path = config.OUTPUT_DIR / "private_task1_metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved metrics: {report_path}")
    return report


def private_group_cv():
    """Reproduce the fast five-fold user-disjoint CPU baseline report."""
    frame = load_train_data().reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    folds = []
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(frame.text, frame.risk_label, groups=frame.anon_user_id)
    ):
        metrics = _evaluate_holdout(
            frame, train_indices, test_indices,
            f"stratified user-group fold {fold}",
            f"private_group_cv_fold{fold}",
        )
        folds.append(metrics)
        print(f"Fold {fold}: {metrics['task1_score']:.4f} "
              f"(risk={metrics['risk_weighted_f1']:.4f}, phrase={metrics['phrase_f1']:.4f}, "
              f"task2={metrics['task2_factor_macro_f1']:.4f}, composite={metrics['composite_score']:.4f})")
    scores = np.asarray([fold["task1_score"] for fold in folds])
    summary = {
        "folds": folds,
        "mean_risk_weighted_f1": float(np.mean([fold["risk_weighted_f1"] for fold in folds])),
        "mean_phrase_f1": float(np.mean([fold["phrase_f1"] for fold in folds])),
        "mean_task1": float(scores.mean()),
        "std_task1": float(scores.std()),
        "mean_task2_factor_macro_f1": float(np.mean([fold["task2_factor_macro_f1"] for fold in folds])),
        "mean_composite_score": float(np.mean([fold["composite_score"] for fold in folds])),
    }
    output = config.OUTPUT_DIR / "private_group_cv_metrics.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Mean Task 1: {summary['mean_task1']:.4f}")
    print(f"Std Task 1: {summary['std_task1']:.4f}")
    print(f"Mean Task 2 Macro F1: {summary['mean_task2_factor_macro_f1']:.4f}")
    print(f"Mean composite: {summary['mean_composite_score']:.4f}")
    print(f"Saved metrics: {output}")
    return summary


def predict():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"CPU baseline not trained: {MODEL_FILE}. Run main.py first.")
    saved = joblib.load(MODEL_FILE)
    frame = load_test_data()
    matrix = saved["vectorizer"].transform(frame.text)
    risks = saved["risk_model"].predict(matrix)
    probs = _factor_probabilities(saved["factor_models"], matrix)
    rows = []
    for i, row in enumerate(frame.itertuples(index=False)):
        evidence = _extract_evidence(row.text, risks[i], saved["lexicon"])
        risk, evidence = _apply_task1_rules(row.text, risks[i], evidence)
        labels = [config.ID2FACTOR[j] for j, p in enumerate(probs[i]) if p >= config.FACTOR_THRESHOLD]
        rows.append({"row_id": row.row_id, "suicide risk": config.ID2RISK[int(risk)],
                     "evidence for suicide risk level": "; ".join(evidence),
                     "factors": str(labels)})
    output = config.OUTPUT_DIR / "submission.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved CPU-baseline submission: {output}")

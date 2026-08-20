"""Leak-free metric-aligned evidence lexicon for Task 1 (V31).

The official evidence metric accepts a prediction when it contains a gold
phrase or is contained by one.  V31 therefore learns short, high-precision
subphrases from gold evidence, but rebuilds the lexicon inside every user-
disjoint fold.  Fold 3 is the only fold used to choose hyperparameters; folds
0--2 are untouched validation folds.

This is deliberately a post-processor.  It leaves the first-stage risk model
and the stable neural evidence decoder unchanged, so any measured difference
is attributable to the lexicon policy alone.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
import re

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from baseline import _post_phrase_f1
from configs.config import config
from inference.task1_evidence_v4 import load_evidence_calibration
from preprocess.preprocess import load_train_data
from trainer.task1_atomic_v25 import _baseline_evidence, _bootstrap, _load_records
from utils.task1_metric import task1_score


OUTPUT = config.OUTPUT_DIR / "task1_evidence_lexicon_v31"
RESULTS = OUTPUT / "results.json"
CALIBRATION = OUTPUT / "calibration.json"
TRAINING_VERSION = "task1-metric-aligned-evidence-lexicon-v31"
CALIBRATION_FOLD = 3
VALIDATION_FOLDS = (0, 1, 2)

TOKEN = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)?", re.IGNORECASE)
SINGLETONS = {
    "suicide", "suicidal", "overdose", "overdosed", "kms",
}
ANCHORS = {
    "suicide", "suicidal", "die", "dying", "dead", "death", "kill",
    "killing", "killed", "myself", "overdose", "overdosed", "kms",
    "attempt", "attempted", "hang", "hanging", "hung", "slit", "slitting",
    "jump", "jumping", "shoot", "shooting", "shot", "drown", "drowning",
    "wake", "alive", "life", "wrist", "wrists", "pills", "rope", "bridge",
    "forever", "exist", "existing", "goodbye", "goodbyes",
}


def _tokens(text: str):
    return [(match.group(0).casefold().replace("’", "'"), match.start(), match.end())
            for match in TOKEN.finditer(str(text))]


def _normalise(value: str) -> str:
    return " ".join(token for token, _, _ in _tokens(value))


def _candidate_patterns(frame, indices, max_tokens: int):
    """Generate only definition-aligned n-grams observed inside gold spans."""
    patterns = set()
    for index in map(int, indices):
        for phrase in frame.iloc[index].evidence:
            words = [token for token, _, _ in _tokens(phrase)]
            for size in range(1, min(max_tokens, len(words)) + 1):
                for start in range(len(words) - size + 1):
                    pattern = tuple(words[start:start + size])
                    if size == 1:
                        if pattern[0] not in SINGLETONS:
                            continue
                    elif not (set(pattern) & ANCHORS):
                        continue
                    patterns.add(pattern)
    return patterns


def _post_pattern_occurrences(text: str, patterns, lengths):
    tokens = _tokens(text); words = [token for token, _, _ in tokens]
    occurrences = defaultdict(list)
    for size in lengths:
        if size > len(words):
            continue
        for start in range(len(words) - size + 1):
            pattern = tuple(words[start:start + size])
            if pattern in patterns:
                begin = tokens[start][1]; end = tokens[start + size - 1][2]
                occurrences[pattern].append((begin, end, text[begin:end]))
    return occurrences


def _wilson_lower(tp: int, total: int, z: float = 1.0) -> float:
    """One-sided-ish Wilson lower score; protects rare perfect patterns."""
    if total <= 0:
        return 0.0
    p = tp / total; z2 = z * z
    centre = p + z2 / (2 * total)
    radius = z * math.sqrt((p * (1 - p) + z2 / (4 * total)) / total)
    return (centre - radius) / (1 + z2 / total)


def _build_statistics(frame, fit_indices, max_tokens: int):
    patterns = _candidate_patterns(frame, fit_indices, max_tokens)
    lengths = sorted({len(pattern) for pattern in patterns})
    stats = {pattern: {"tp": 0, "total": 0, "risks": [0, 0, 0, 0]}
             for pattern in patterns}
    for index in map(int, fit_indices):
        row = frame.iloc[index]
        occurrences = _post_pattern_occurrences(str(row.text), patterns, lengths)
        for pattern, spans in occurrences.items():
            # Count at document level.  The submitted format contains phrases,
            # not offsets, so any verbatim occurrence with a metric match is a
            # valid occurrence for this post.
            hit = any(_post_phrase_f1([phrase], list(row.evidence)) > 0
                      for _, _, phrase in spans)
            stats[pattern]["total"] += 1
            if hit:
                stats[pattern]["tp"] += 1
                stats[pattern]["risks"][int(row.risk_label)] += 1
    rows = {}
    for pattern, value in stats.items():
        tp = int(value["tp"]); total = int(value["total"])
        rows[pattern] = {
            **value,
            "precision": tp / max(total, 1),
            "lower": _wilson_lower(tp, total),
        }
    return rows


def _approved(stats, parameters):
    return {
        pattern: values for pattern, values in stats.items()
        if values["tp"] >= parameters["min_tp"]
        and values["precision"] >= parameters["min_precision"]
        and values["lower"] >= parameters["min_lower"]
        and len(pattern) <= parameters["max_tokens"]
    }


def _lexicon_matches(text: str, lexicon, risk: int):
    if risk == config.RISK_LABELS["Indicator"] or not lexicon:
        return []
    occurrences = _post_pattern_occurrences(
        text, lexicon, sorted({len(pattern) for pattern in lexicon})
    )
    ranked = []
    for pattern, spans in occurrences.items():
        values = lexicon[pattern]
        risk_hits = values["risks"][risk]
        # Precision is primary.  Repeated support, compatibility with the
        # predicted risk, and a little extra context break ties.
        rank = (values["lower"], values["precision"], risk_hits,
                values["tp"], len(pattern))
        ranked.append((rank, spans[0][2]))
    ranked.sort(reverse=True)
    result = []
    for _, phrase in ranked:
        normal = _normalise(phrase)
        if not normal or any(normal in old or old in normal for old in
                             (_normalise(item) for item in result)):
            continue
        result.append(phrase)
    return result


def _shorten_baseline(baseline, matches):
    result = []
    for phrase in baseline:
        normal = _normalise(phrase)
        inside = [item for item in matches if _normalise(item) in normal]
        # Use the longest learned subphrase: it retains enough context while
        # still removing neural decoder boundary noise.
        selected = max(inside, key=lambda item: len(_tokens(item))) if inside else phrase
        selected_normal = _normalise(selected)
        if selected_normal and not any(
            selected_normal in _normalise(old) or _normalise(old) in selected_normal
            for old in result
        ):
            result.append(selected)
    return result


def _combine(baseline, matches, parameters):
    mode = parameters["mode"]
    if mode == "baseline":
        return list(baseline)
    shortened = _shorten_baseline(baseline, matches)
    if mode == "shorten":
        return shortened
    if mode == "replace_if_found":
        return matches[:parameters["topk"]] if matches else shortened
    if mode in {"append", "shorten_append"}:
        result = list(baseline if mode == "append" else shortened)
        maximum_additions = parameters["maximum_additions"]
        added = 0
        for phrase in matches:
            normal = _normalise(phrase)
            if any(normal in _normalise(old) or _normalise(old) in normal for old in result):
                continue
            result.append(phrase); added += 1
            if added >= maximum_additions:
                break
        return result[:parameters["topk"]]
    raise ValueError(mode)


def _evaluate(frame, indices, records, stats, parameters):
    lexicon = _approved(stats, parameters)
    old_predictions = []; new_predictions = []; risks = []
    for index in map(int, indices):
        record = records[index]; risk = int(record["risk"])
        baseline = list(record["baseline"])
        matches = _lexicon_matches(str(frame.iloc[index].text), lexicon, risk)
        candidate = [] if risk == 0 else _combine(baseline, matches, parameters)
        old_predictions.append(baseline); new_predictions.append(candidate); risks.append(risk)
    gold = [list(frame.iloc[int(index)].evidence) for index in indices]
    old = np.asarray([_post_phrase_f1(pred, target)
                      for pred, target in zip(old_predictions, gold)], dtype=np.float32)
    new = np.asarray([_post_phrase_f1(pred, target)
                      for pred, target in zip(new_predictions, gold)], dtype=np.float32)
    return old, new, np.asarray(risks, dtype=np.int64), len(lexicon), new_predictions


def _parameter_grid():
    # The baseline row guarantees that calibration cannot force a regression.
    yield {"mode": "baseline", "min_tp": 2, "min_precision": .8,
           "min_lower": .6, "max_tokens": 6, "topk": 5, "maximum_additions": 0}
    for min_tp in (2, 3, 5):
        for min_precision in (.70, .80, .90, 1.00):
            for min_lower in (.50, .60, .70, .80):
                for max_tokens in (3, 5, 8):
                    for mode in ("shorten", "replace_if_found", "shorten_append"):
                        for topk in ((2, 3, 4) if mode != "shorten" else (5,)):
                            additions = (1, 2) if mode == "shorten_append" else (0,)
                            for maximum_additions in additions:
                                yield {"mode": mode, "min_tp": min_tp,
                                       "min_precision": min_precision,
                                       "min_lower": min_lower,
                                       "max_tokens": max_tokens, "topk": topk,
                                       "maximum_additions": maximum_additions}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    labels = frame.risk_label.to_numpy(); groups = frame.anon_user_id.astype(str).to_numpy()
    outer_train, _ = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    ).split(np.zeros(len(frame)), labels, groups))
    raw_records, membership, _ = _load_records()
    calibration = load_evidence_calibration()
    records = {}
    for record in raw_records:
        index = int(record["global_index"])
        records[index] = {**record, "baseline": _baseline_evidence(record, calibration)}

    # Hyperparameters are selected on fold 3 and nowhere else.
    calibration_idx = np.asarray([i for i in outer_train if membership[int(i)] == CALIBRATION_FOLD])
    calibration_fit = np.asarray([i for i in outer_train if membership[int(i)] != CALIBRATION_FOLD])
    if set(groups[calibration_idx]) & set(groups[calibration_fit]):
        raise ValueError("V31 calibration user leakage")
    print("V31 building fold-3 calibration lexicon", flush=True)
    calibration_stats = _build_statistics(frame, calibration_fit, max_tokens=8)
    rows = []
    for parameters in _parameter_grid():
        old, new, _, lexicon_size, _ = _evaluate(
            frame, calibration_idx, records, calibration_stats, parameters
        )
        rows.append({**parameters, "lexicon_size": lexicon_size,
                     "phrase_f1": float(new.mean()),
                     "delta": float(new.mean() - old.mean())})
    rows.sort(key=lambda row: (row["phrase_f1"], -row["lexicon_size"]), reverse=True)
    selected = {key: value for key, value in rows[0].items()
                if key not in {"phrase_f1", "delta", "lexicon_size"}}
    print(f"V31 selected on fold3: {rows[0]}", flush=True)

    all_indices = []; all_old = []; all_new = []; all_risk = []; fold_rows = []
    for fold in VALIDATION_FOLDS:
        valid_idx = np.asarray([i for i in outer_train if membership[int(i)] == fold])
        fit_idx = np.asarray([i for i in outer_train if membership[int(i)] != fold])
        if set(groups[valid_idx]) & set(groups[fit_idx]):
            raise ValueError(f"V31 fold {fold} user leakage")
        print(f"V31 rebuilding lexicon for untouched fold {fold}", flush=True)
        stats = _build_statistics(frame, fit_idx, max_tokens=8)
        old, new, risks, lexicon_size, _ = _evaluate(frame, valid_idx, records, stats, selected)
        fold_rows.append({"fold": fold, "fit_posts": int(len(fit_idx)),
                          "valid_posts": int(len(valid_idx)), "lexicon_size": lexicon_size,
                          "baseline_phrase_f1": float(old.mean()),
                          "candidate_phrase_f1": float(new.mean()),
                          "phrase_delta": float(new.mean() - old.mean()),
                          "improved_posts": int((new > old).sum()),
                          "worsened_posts": int((new < old).sum())})
        print(f"V31 fold={fold} phrase {old.mean():.6f} -> {new.mean():.6f}", flush=True)
        all_indices.extend(map(int, valid_idx)); all_old.extend(old); all_new.extend(new)
        all_risk.extend(risks)

    order = np.argsort(all_indices); indices = np.asarray(all_indices)[order]
    old = np.asarray(all_old, dtype=np.float32)[order]
    new = np.asarray(all_new, dtype=np.float32)[order]
    risks = np.asarray(all_risk, dtype=np.int64)[order]
    risk_f1 = float(f1_score(labels[indices], risks, average="weighted", zero_division=0))
    old_task = task1_score(risk_f1, float(old.mean()))
    new_task = task1_score(risk_f1, float(new.mean()))
    bootstrap = _bootstrap(groups[indices], old, new)
    adopted = bool(new_task >= old_task + .003 and bootstrap["positive_fraction"] >= .80)
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation_scope": "parameters selected fold3; lexicon rebuilt per fold; untouched user folds0-2",
        "selected": selected, "calibration_top10": rows[:10], "folds": fold_rows,
        "baseline": {"risk_f1": risk_f1, "phrase_f1": float(old.mean()), "task1": old_task},
        "candidate": {"risk_f1": risk_f1, "phrase_f1": float(new.mean()), "task1": new_task,
                      "improved_posts": int((new > old).sum()),
                      "worsened_posts": int((new < old).sum())},
        "user_cluster_bootstrap": bootstrap, "adopted": adopted,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    CALIBRATION.write_text(json.dumps({"training_version": TRAINING_VERSION,
        "adopted": adopted, "selected": selected,
        "crossvalidated_task1": new_task}, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()

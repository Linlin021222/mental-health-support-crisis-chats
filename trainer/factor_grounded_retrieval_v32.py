"""Leak-free retrieval expert built only from dual-confirmed factor evidence.

The accepted V31 phrases are too few for another Transformer fine-tune.  This
experiment therefore freezes the current V3 ensemble and uses the phrases as
high-precision prototypes.  Word and character TF-IDF similarities are
computed against every sentence in an outer-fold validation post.  A small,
pre-registered rank blend is applied only to labels with at least two
independently accepted training-user prototypes.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from configs.config import config
from inference.factor_nli import _rank_decode
from preprocess.preprocess import load_train_data
from trainer.factor_llm_lexical_v6 import _current_v3_probability
from trainer.factor_sentence_evidence_cv_v27 import _user_bootstrap
from trainer.factor_sentence_evidence_v16 import _sentences


OUTPUT = config.OUTPUT_DIR / "factor_grounded_retrieval_v32"
RESULTS = OUTPUT / "fold0_results.json"
RATIONALES = config.OUTPUT_DIR / "factor_dual_rationale_v31" / "grounded_rationales.json"
TRAINING_VERSION = "dual-grounded-evidence-retrieval-v32"
BLEND_WEIGHT = 0.05
TOPK_RATIO = 1.10
MIN_PROTOTYPES = 2


def _rank(values):
    """Return deterministic [0, 1] ranks; only within-label order matters."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float32)
    result[order] = np.linspace(0., 1., len(values), dtype=np.float32)
    return result


def _load_prototypes(train_idx):
    payload = json.loads(RATIONALES.read_text(encoding="utf-8"))
    allowed = set(map(int, train_idx))
    prototypes = {label: [] for label in range(config.NUM_FACTORS)}
    rejected = []
    for row in payload["records"]:
        label, index = int(row["label_id"]), int(row["row_index"])
        if index not in allowed:
            rejected.append(index)
            continue
        text = " ".join(str(row["evidence"]).split()).strip()
        if text and text.casefold() not in {x.casefold() for x in prototypes[label]}:
            prototypes[label].append(text)
    if rejected:
        raise RuntimeError(f"V31 contains non-training rows for fold0: {rejected}")
    return prototypes


def _similarity(texts, train_texts, prototypes):
    sentence_lists = [_sentences(text) or [str(text)] for text in texts]
    flat_sentences = [sentence for rows in sentence_lists for sentence in rows]
    offsets = np.cumsum([0] + [len(rows) for rows in sentence_lists])
    evidence = [item for rows in prototypes.values() for item in rows]
    fit_corpus = list(train_texts) + evidence

    word = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", sublinear_tf=True,
        ngram_range=(1, 2), min_df=1, max_df=.995, max_features=90000,
    )
    char = TfidfVectorizer(
        analyzer="char_wb", lowercase=True, sublinear_tf=True,
        ngram_range=(3, 5), min_df=1, max_features=120000,
    )
    word.fit(fit_corpus); char.fit(fit_corpus)
    sentence_word = word.transform(flat_sentences)
    sentence_char = char.transform(flat_sentences)
    result = np.zeros((len(texts), config.NUM_FACTORS), dtype=np.float32)
    for label, phrases in prototypes.items():
        if len(phrases) < MIN_PROTOTYPES:
            continue
        prototype_word = word.transform(phrases)
        prototype_char = char.transform(phrases)
        word_score = sentence_word @ prototype_word.T
        char_score = sentence_char @ prototype_char.T
        combined = .55 * word_score + .45 * char_score
        if sparse.issparse(combined):
            combined = combined.toarray()
        sentence_score = np.asarray(combined).max(axis=1)
        for row in range(len(texts)):
            result[row, label] = float(sentence_score[offsets[row]:offsets[row + 1]].max())
    return result


def evaluate_fold0():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_train_data().reset_index(drop=True)
    targets = np.vstack(frame.factor_vector.to_numpy()).astype(np.int8)
    risk = frame.risk_label.to_numpy(dtype=np.int64)
    groups = frame.anon_user_id.astype(str).to_numpy()
    train_idx, valid_idx = next(StratifiedGroupKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED,
    ).split(np.zeros(len(frame)), risk, groups))
    prototypes = _load_prototypes(train_idx)
    trusted = np.asarray(
        [len(prototypes[label]) >= MIN_PROTOTYPES for label in range(config.NUM_FACTORS)]
    )
    retrieval = _similarity(
        frame.text.iloc[valid_idx].astype(str).tolist(),
        frame.text.iloc[train_idx].astype(str).tolist(), prototypes,
    )
    current, _ = _current_v3_probability()
    base = current[valid_idx]
    mixed = base.copy()
    for label in np.flatnonzero(trusted):
        mixed[:, label] = (
            (1. - BLEND_WEIGHT) * _rank(base[:, label])
            + BLEND_WEIGHT * _rank(retrieval[:, label])
        )
    prevalence = targets[train_idx].mean(0)
    baseline_prediction = _rank_decode(base, prevalence, TOPK_RATIO)
    candidate_prediction = _rank_decode(mixed, prevalence, TOPK_RATIO)
    truth = targets[valid_idx]
    baseline = float(f1_score(truth, baseline_prediction, average="macro", zero_division=0))
    candidate = float(f1_score(truth, candidate_prediction, average="macro", zero_division=0))
    bootstrap = _user_bootstrap(
        truth, baseline_prediction, candidate_prediction, groups[valid_idx],
        seed=323232, draws=3000,
    )
    per_label = []
    for label in range(config.NUM_FACTORS):
        per_label.append({
            "label": config.ID2FACTOR[label],
            "support": int(truth[:, label].sum()),
            "prototype_count": len(prototypes[label]),
            "used": bool(trusted[label]),
            "baseline_f1": float(f1_score(
                truth[:, label], baseline_prediction[:, label], zero_division=0
            )),
            "candidate_f1": float(f1_score(
                truth[:, label], candidate_prediction[:, label], zero_division=0
            )),
        })
    payload = {
        "training_version": TRAINING_VERSION,
        "evaluation": "strict outer fold0; rationales and TF-IDF fit use training users only",
        "fixed_policy": {
            "blend_weight": BLEND_WEIGHT, "topk_ratio": TOPK_RATIO,
            "minimum_independent_prototypes": MIN_PROTOTYPES,
        },
        "trusted_labels": [config.ID2FACTOR[x] for x in np.flatnonzero(trusted)],
        "baseline_macro_f1": baseline,
        "candidate_macro_f1": candidate,
        "delta": candidate - baseline,
        "user_cluster_bootstrap": bootstrap,
        "per_label": per_label,
        "promising_for_expansion": bool(
            candidate >= baseline + .003 and bootstrap["positive_fraction"] >= .70
        ),
        "adopted": False,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    evaluate_fold0()

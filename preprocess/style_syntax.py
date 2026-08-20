"""Case-preserving style and lightweight syntax features for Reddit text.

Transformers already model syntax implicitly.  These transparent features are
an auxiliary view focused on signals lost by the lower-cased TF-IDF branch:
all-caps emphasis, punctuation intensity, negation scope, tense/modality,
conditionals, contrast and first-person constructions.
"""
from __future__ import annotations

import re

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
SENTENCE = re.compile(r"[^.!?\n]+[.!?]*")
PROFANITY = re.compile(r"(?i)\b(?:fuck|fucking|shit|bitch|damn|hell|asshole)\b")
FIRST_PERSON = re.compile(r"(?i)\b(?:i|i['’]?m|i['’]?ve|i['’]?ll|me|my|mine|myself)\b")
NEGATION = re.compile(r"(?i)\b(?:no|not|never|nothing|nobody|neither|cannot|can['’]?t|don['’]?t|won['’]?t|isn['’]?t|aren['’]?t)\b")
FUTURE = re.compile(r"(?i)\b(?:will|going\s+to|gonna|tomorrow|soon|later|plan(?:ning|ned)?|intend)\b")
PAST = re.compile(r"(?i)\b(?:was|were|had|did|ago|yesterday|previously|before|past|used\s+to)\b")
CONDITIONAL = re.compile(r"(?i)\b(?:if|unless|would|could|should|might|maybe|perhaps)\b")
CONTRAST = re.compile(r"(?i)\b(?:but|however|although|though|yet|except|instead)\b")
ABSOLUTE = re.compile(r"(?i)\b(?:always|never|completely|totally|everyone|no\s+one|nothing|everything|forever|impossible|all)\b")
REQUEST = re.compile(r"(?i)\b(?:help\s+me|please\s+help|what\s+should\s+i|any\s+advice|need\s+help)\b")
SELF_EVAL = re.compile(r"(?i)\b(?:i\s+(?:am|feel)|i['’]?m)\s+(?:worthless|useless|ugly|stupid|alone|hopeless|a\s+burden|failure)\b")
SUICIDE = re.compile(r"(?i)\b(?:suicid(?:e|al)|die|dead|death|kill\s+myself|kms|end\s+my\s+life|overdose)\b")


def numeric_style_syntax(texts):
    rows = []
    for raw in texts:
        text = str(raw); words = WORD.findall(text); n = max(1, len(words))
        letters = [character for character in text if character.isalpha()]
        upper_letters = sum(character.isupper() for character in letters)
        upper_words = sum(word.isupper() and len(word) >= 2 for word in words)
        title_words = sum(word[:1].isupper() and not word.isupper() for word in words)
        sentences = [item.group(0).strip() for item in SENTENCE.finditer(text) if item.group(0).strip()]
        sentence_lengths = [len(WORD.findall(sentence)) for sentence in sentences] or [0]
        pattern_counts = [
            len(FIRST_PERSON.findall(text)), len(NEGATION.findall(text)),
            len(FUTURE.findall(text)), len(PAST.findall(text)),
            len(CONDITIONAL.findall(text)), len(CONTRAST.findall(text)),
            len(ABSOLUTE.findall(text)), len(REQUEST.findall(text)),
            len(SELF_EVAL.findall(text)), len(SUICIDE.findall(text)),
            len(PROFANITY.findall(text)),
        ]
        values = [
            min(len(text) / 2000.0, 1.0), min(n / 400.0, 1.0),
            len(set(word.casefold() for word in words)) / n,
            upper_letters / max(1, len(letters)), upper_words / n, title_words / n,
            min(text.count("!") / 10.0, 1.0), min(text.count("?") / 10.0, 1.0),
            min(len(re.findall(r"[!?]{2,}", text)) / 5.0, 1.0),
            min(len(re.findall(r"\.{3,}", text)) / 5.0, 1.0),
            min(len(re.findall(r"(?i)([a-z])\1{2,}", text)) / 5.0, 1.0),
            min(text.count("\n") / 10.0, 1.0),
            min(len(sentences) / 20.0, 1.0),
            min(float(np.mean(sentence_lengths)) / 50.0, 1.0),
            min(float(np.std(sentence_lengths)) / 30.0, 1.0),
        ]
        values.extend(min(value / max(1.0, n * .10), 1.0) for value in pattern_counts)
        # High-value interactions unavailable to an ordinary unigram model.
        values.extend([
            float(bool(NEGATION.search(text) and SUICIDE.search(text))),
            float(bool(FUTURE.search(text) and SUICIDE.search(text))),
            float(bool(PAST.search(text) and SUICIDE.search(text))),
            float(bool(CONDITIONAL.search(text) and SUICIDE.search(text))),
            float(bool(CONTRAST.search(text) and SUICIDE.search(text))),
            float(bool(upper_words / n >= .50)),
            float(bool(upper_words / n >= .50 and PROFANITY.search(text))),
            float(bool(FIRST_PERSON.search(text) and SELF_EVAL.search(text))),
        ])
        rows.append(values)
    return sparse.csr_matrix(np.asarray(rows, dtype=np.float32))


def cased_vectorizer():
    return FeatureUnion([
        ("word", TfidfVectorizer(
            lowercase=False, strip_accents="unicode", ngram_range=(1, 3),
            min_df=1, max_df=.995, max_features=110_000, sublinear_tf=True,
        )),
        ("char", TfidfVectorizer(
            lowercase=False, strip_accents="unicode", analyzer="char_wb",
            ngram_range=(3, 5), min_df=2, max_features=160_000,
            sublinear_tf=True,
        )),
    ])


def fit_transform_cased(texts, extra_corpus=()):
    texts = list(map(str, texts)); vectorizer = cased_vectorizer()
    vectorizer.fit(texts + list(map(str, extra_corpus)))
    matrix = sparse.hstack(
        [vectorizer.transform(texts), numeric_style_syntax(texts)], format="csr",
    )
    return vectorizer, matrix


def transform_cased(vectorizer, texts):
    texts = list(map(str, texts))
    return sparse.hstack(
        [vectorizer.transform(texts), numeric_style_syntax(texts)], format="csr",
    )


def fit_transform_lower_syntax(texts):
    # Import lazily to keep preprocessing independent from competition logic.
    from baseline import _vectorizer
    texts = list(map(str, texts)); vectorizer = _vectorizer()
    matrix = sparse.hstack(
        [vectorizer.fit_transform(texts), numeric_style_syntax(texts)], format="csr",
    )
    return vectorizer, matrix


def transform_lower_syntax(vectorizer, texts):
    texts = list(map(str, texts))
    return sparse.hstack(
        [vectorizer.transform(texts), numeric_style_syntax(texts)], format="csr",
    )

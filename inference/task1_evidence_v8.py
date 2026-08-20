"""Source-aware evidence fusion for Task 1.

The neural span head frequently finds a faithful phrase, while the legacy
regular expressions emit a shorter substring (for example ``pills``) first.
Since the scorer penalises every unmatched phrase and prediction slots are
limited, source-aware fusion keeps contextual model spans and uses rules only
to fill genuinely missing evidence.
"""
from __future__ import annotations

import re

from configs.config import config


_WEAK_SINGLE_TOKEN_CUES = {
    "attempted", "attempting", "suicide", "suicidal", "rope", "gun",
    "knife", "razor", "pill", "pills", "overdose",
}
_EVIDENCE_ANCHOR = re.compile(
    r"(?i)(suicid|kill|die|dead|death|attempt|overdos|hang|shoot|stab|"
    r"drown|bleed|swallow|pills?|rope|gun|knife|razor|wrist|neck|jump|"
    r"note|goodbye|end my life|not wake|stop existing|sleep forever|"
    r"life (?:is|was) over|everything ready|materials ready)"
)


def _normalise(phrase):
    return " ".join(str(phrase).casefold().split())


def _split(phrases):
    return [
        part.strip()
        for phrase in phrases
        for part in str(phrase).split(";")
        if part.strip()
    ]


def _is_weak_cue(phrase):
    normalised = _normalise(phrase)
    return len(normalised.split()) == 1 and normalised in _WEAK_SINGLE_TOKEN_CUES


def _append_nonredundant(selected, phrase):
    """Append unless an existing phrase already gives the same coverage."""
    normalised = _normalise(phrase)
    if not normalised:
        return
    if any(normalised in old or old in normalised for _, old in selected):
        return
    selected.append((str(phrase), normalised))


def smart_fuse_evidence(text, risk_id, model_phrases, cue_phrases, mode, topk):
    """Fuse evidence with explicit source and precision policies.

    ``model_context`` is the conservative fix: model spans are kept first and
    one-token rules are used only when the model found no containing phrase.
    Anchor variants prioritise suicide-related spans without deleting the
    remaining neural candidates, so euphemistic evidence can still survive.
    """
    if int(risk_id) == config.RISK_LABELS["Indicator"]:
        return []
    model = _split(model_phrases)
    cues = _split(cue_phrases)
    strong_cues = [cue for cue in cues if not _is_weak_cue(cue)]
    weak_cues = [cue for cue in cues if _is_weak_cue(cue)]
    anchored_model = [phrase for phrase in model if _EVIDENCE_ANCHOR.search(phrase)]
    other_model = [phrase for phrase in model if not _EVIDENCE_ANCHOR.search(phrase)]

    if mode == "model_only":
        ordered = model
    elif mode == "model_context":
        ordered = model + strong_cues + weak_cues
    elif mode == "model_strong_only":
        ordered = model + strong_cues
    elif mode == "anchor_model_context":
        ordered = anchored_model + strong_cues + other_model + weak_cues
    elif mode == "anchor_model_strong":
        ordered = anchored_model + strong_cues + other_model
    elif mode == "strong_anchor_model":
        ordered = strong_cues + anchored_model + other_model
    else:
        raise ValueError(f"Unknown evidence-v8 fusion mode: {mode}")

    selected = []
    for phrase in ordered:
        _append_nonredundant(selected, phrase)
        if len(selected) >= int(topk):
            break
    return [phrase for phrase, _ in selected]


"""High-precision clinical hierarchy for Task 1 risk and evidence.

The four labels differ primarily by explicitness, planning/means, and whether
an act already happened.  These patterns intentionally require first-person
suicide context; isolated words such as ``pills`` or ``knife`` are not enough.
All returned evidence is a verbatim substring of the source post.
"""
from __future__ import annotations

import re
from functools import lru_cache

from configs.config import config


_SELF = r"(?:myself|my\s+(?:life|wrists?))"
_METHOD = (
    r"(?:overdos(?:e|ed|ing)(?:\s+on)?|hang(?:ed|ing)?|shoot|shot|stab(?:bed|bing)?|"
    r"cut(?:ting)?\s+my\s+wrists?|jump(?:ed|ing)?\s+(?:off|in\s+front)|"
    r"drown(?:ed|ing)?|poison(?:ed|ing)?|bleed(?:ing)?\s+out|"
    r"take|taking|took|swallow(?:ed|ing)?)"
)

CLINICAL_ATTEMPT_CUE = re.compile(
    rf"(?ix)("
    rf"\bI\s+(?:have\s+|had\s+)?(?:attempted|tried)\s+(?:to\s+)?"
    rf"(?:kill\s+myself|commit\s+suicide|end\s+my\s+life|{_METHOD})\b|"
    rf"\bI\s+(?:previously\s+|already\s+|once\s+|twice\s+)?"
    rf"(?:overdosed|hung\s+myself|shot\s+myself|stabbed\s+myself|"
    rf"cut\s+my\s+wrists?|jumped\s+(?:off|in\s+front)|tried\s+to\s+drown\s+myself)\b|"
    rf"\b(?:my|a|the)\s+(?:(?:last|previous|first|second|recent|past)\s+)?"
    rf"suicide\s+attempt\b|"
    rf"\b(?:after|since|survived|failed)\s+(?:my|a|the)?\s*(?:suicide\s+)?attempt\b|"
    rf"\bI\s+(?:was|got)\s+(?:hospitali[sz]ed|taken\s+to\s+(?:the\s+)?(?:ER|hospital))"
    rf"\s+(?:after|because)\s+(?:I\s+)?(?:tried|attempted|overdosed)\b"
    rf")"
)

CLINICAL_BEHAVIOR_CUE = re.compile(
    rf"(?ix)("
    rf"\bI\s+(?:am|'m|was)\s+(?:going|planning|preparing)\s+to\s+"
    rf"(?:kill\s+myself|commit\s+suicide|end\s+my\s+life|{_METHOD})\b|"
    rf"\bI\s+(?:will|plan|intend|decided|have\s+decided|want)\s+to\s+"
    rf"(?:kill\s+myself|commit\s+suicide|end\s+my\s+life|{_METHOD})\b|"
    rf"\b(?:tonight|tomorrow|today|soon)\s+I(?:'m|\s+am|'ll|\s+will)\s+"
    rf"(?:kill\s+myself|commit\s+suicide|{_METHOD})\b|"
    rf"\bI\s+(?:have|got|bought|prepared|saved|stockpiled)\s+"
    rf"(?:enough\s+|all\s+(?:of\s+)?my\s+)?(?:pills?|tablets?|rope|gun|poison)\s+"
    rf"(?:to|so\s+(?:that\s+)?I\s+can)\s+(?:kill\s+myself|die|overdose|hang\s+myself)\b|"
    rf"\b(?:set|picked|chosen)\s+(?:a|the)\s+(?:date|time|place)\s+"
    rf"(?:to|for)\s+(?:kill(?:ing)?\s+myself|suicide|dying)\b|"
    rf"\b(?:wrote|written|writing)\s+(?:my\s+)?(?:suicide\s+note|goodbye(?:\s+letters?)?)\b|"
    rf"\b(?:take|taking|swallow|swallowing)\s+(?:all|every)\s+(?:of\s+)?my\s+"
    rf"(?:pills?|tablets?)\b|"
    rf"\b(?:shoot|hang|stab|drown|poison)\s+myself\b|"
    rf"\bjump\s+off\s+(?:a|the|this)\s+(?:bridge|building|cliff|roof)\b"
    rf")"
)

CLINICAL_IDEATION_CUE = re.compile(
    r"(?ix)("
    r"\bI(?:'m|\s+am)\s+suicidal\b|"
    r"\bI\s+(?:feel\s+|felt\s+)?suicidal\b|"
    r"\bmy\s+suicidal\s+(?:thoughts?|ideation)\b|"
    r"\bI\s+(?:want|wanna|wanted|wish|wished|hope|hoped|need)\s+"
    r"(?:to\s+)?(?:fucking\s+)?(?:die|be\s+dead|kill\s+myself|end\s+my\s+life)\b|"
    r"\bI\s+(?:do\s+not|don['’]?t)\s+want\s+to\s+(?:live|be\s+alive|exist)\b|"
    r"\bI(?:'d|\s+would)\s+rather\s+(?:be\s+)?dead\b|"
    r"\bI(?:'m|\s+am)\s+better\s+off\s+dead\b|"
    r"\b(?:wish|hope)\s+I\s+(?:was|were|would\s+be)?\s*dead\b|"
    r"\b(?:wish|hope)\s+I\s+(?:do\s+not|don['’]?t|won['’]?t)\s+wake\s+up\b|"
    r"\b(?:no\s+reason|nothing)\s+to\s+live\s+for\b|"
    r"\bI\s+(?:cannot|can['’]?t)\s+go\s+on\s+living\b|"
    r"\bend\s+it\s+all\b|\bkill\s+me(?:\s+off)?\b|\bsleep\s+forever\b|"
    r"\bstop\s+existing\b|\bpass\s+away\s+in\s+my\s+sleep\b"
    r")"
)

# The old correction treated any occurrence of these nouns as a suicide plan.
# They are useful means only with explicit self-harm context.
WEAK_METHOD_CUE = re.compile(
    r"(?i)\b(?:pills?|tablets?|rope|gun|knife|razor|poison)\b"
)


@lru_cache(maxsize=4096)
def clinical_matches(text):
    """Return (risk_id, verbatim phrase) pairs, most severe first."""
    result = []
    for risk_id, pattern in (
        (config.RISK_LABELS["Attempt"], CLINICAL_ATTEMPT_CUE),
        (config.RISK_LABELS["Behavior"], CLINICAL_BEHAVIOR_CUE),
        (config.RISK_LABELS["Ideation"], CLINICAL_IDEATION_CUE),
    ):
        result.extend((risk_id, match.group(0).strip()) for match in pattern.finditer(text))
    result.sort(key=lambda item: item[0], reverse=True)
    return result


def clinical_level(text):
    matches = clinical_matches(text)
    return max((risk for risk, _ in matches), default=None)


def correct_clinical_risk(text, predicted_risk, old_model_risk=None, policy="floor"):
    """Apply a conservative hierarchy correction without consulting labels."""
    predicted_risk = int(predicted_risk)
    explicit = clinical_level(text)
    if policy in {"undo_weak", "undo_weak_floor"}:
        # Undo only the known broad-rule failure: the old neural model said
        # Indicator, a lone method noun promoted it to Behavior, and no
        # first-person suicide expression supports that promotion.
        if (
            predicted_risk == config.RISK_LABELS["Behavior"]
            and old_model_risk == config.RISK_LABELS["Indicator"]
            and WEAK_METHOD_CUE.search(text)
            and explicit is None
        ):
            predicted_risk = config.RISK_LABELS["Indicator"]
    if policy in {"floor", "undo_weak_floor"} and explicit is not None:
        predicted_risk = max(predicted_risk, int(explicit))
    return predicted_risk


def clinical_evidence(text, risk_id, hierarchical=True):
    """Select short verbatim cues compatible with the predicted hierarchy."""
    risk_id = int(risk_id)
    if risk_id == config.RISK_LABELS["Indicator"]:
        return []
    pairs = clinical_matches(text)
    if hierarchical:
        return [phrase for level, phrase in pairs if level <= risk_id]
    return [phrase for level, phrase in pairs if level == risk_id]


def fuse_evidence(text, risk_id, model_phrases, mode="hierarchical_first", topk=3):
    if int(risk_id) == config.RISK_LABELS["Indicator"]:
        return []
    hierarchical = "hierarchical" in mode
    cues = clinical_evidence(text, risk_id, hierarchical=hierarchical)
    phrases = list(model_phrases) + cues if mode.endswith("model_first") else cues + list(model_phrases)
    selected = []
    normalized = []
    for phrase in phrases:
        for part in str(phrase).split(";"):
            part = part.strip()
            norm = " ".join(part.casefold().split())
            if part and not any(norm in old or old in norm for old in normalized):
                selected.append(part); normalized.append(norm)
            if len(selected) >= int(topk):
                return selected
    return selected

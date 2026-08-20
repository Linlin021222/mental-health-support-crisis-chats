"""Offset-safe semantic augmentation for Task 1 auxiliary models.

The original Reddit post is never rewritten: semantic markers are appended to
the text.  Consequently this module can help a risk classifier without
changing the character offsets used by the evidence extractor.
"""
from __future__ import annotations

import re
import unicodedata


# Frequent or clinically relevant emoji receive compact, stable descriptions.
# Unknown emoji still receive their official Unicode name below.
EMOJI_MEANINGS = {
    "🤡": ("clown", "self_mockery", "sarcasm"),
    "🙂": ("slight_smile", "possible_masked_distress"),
    "🙃": ("upside_down_smile", "sarcasm", "masked_distress"),
    "😩": ("weary", "distress", "overwhelmed"),
    "🥺": ("pleading", "vulnerable", "distress"),
    "😞": ("disappointed", "sadness"),
    "😔": ("pensive", "sadness"),
    "😭": ("crying", "intense_sadness"),
    "😢": ("crying", "sadness"),
    "😃": ("happy", "positive_emotion"),
    "💔": ("broken_heart", "grief", "relationship_pain"),
    "❤️": ("heart", "love", "support"),
    "♥": ("heart", "love", "support"),
    "🏘️": ("houses", "home"),
    "🏘": ("houses", "home"),
    "💊": ("pill", "medication", "possible_means"),
    "🔪": ("knife", "possible_means"),
    "🪒": ("razor", "possible_means"),
    "🔫": ("gun", "possible_means"),
    "🩸": ("blood", "injury"),
    "⚰️": ("coffin", "death"),
    "⚰": ("coffin", "death"),
}

EMOTICONS = (
    (re.compile(r"(?<!\w)(?::|=|;)-?\)+"), ("smile", "positive_emotion")),
    (re.compile(r"(?<!\w)(?::|=)-?\(+"), ("sad_face", "sadness")),
    (re.compile(r"(?<!\w)(?::|=)-?[/\\]"), ("uneasy_face", "uncertainty")),
    (re.compile(r"(?<!\w)-_-"), ("annoyed_face", "frustration")),
    (re.compile(r"(?<!\w)(?:<3|❤)"), ("heart", "love", "support")),
)

# These markers distinguish Attempt, Behavior and Ideation more directly than
# generic past/future word counts.  They are deliberately descriptive rather
# than deterministic labels; the supervised classifier decides their value.
TEMPORAL_PATTERNS = (
    ("past_attempt", re.compile(
        r"(?i)\b(?:tried|attempted|survived|overdosed|woke\s+up|"
        r"was\s+(?:found|saved|hospitali[sz]ed))\b.{0,35}\b"
        r"(?:suicid\w*|kill(?:ed|ing)?\s+myself|die|overdos\w*|jump\w*|"
        r"hang\w*|cut\w*|shot|drown\w*)\b|"
        r"\b(?:last\s+(?:night|week|year)|yesterday|ago|previously|before)\b"
        r".{0,45}\b(?:attempt\w*|overdos\w*|suicid\w*|kill\w*|die|died)\b"
    )),
    ("recent_attempt", re.compile(
        r"(?i)\b(?:just|recently|today|tonight|this\s+(?:morning|week))\b"
        r".{0,45}\b(?:attempt\w*|overdos\w*|tried\s+to\s+(?:die|kill)|"
        r"woke\s+up|hospitali[sz]ed)\b"
    )),
    ("current_intent", re.compile(
        r"(?i)\b(?:right\s+now|currently|tonight|today|at\s+this\s+moment)\b"
        r".{0,45}\b(?:suicid\w*|die|kill\s+myself|end\s+(?:it|my\s+life))\b|"
        r"\b(?:suicid\w*|die|kill\s+myself|end\s+(?:it|my\s+life))\b"
        r".{0,45}\b(?:right\s+now|currently|tonight|today)\b"
    )),
    ("future_plan", re.compile(
        r"(?i)\b(?:will|going\s+to|gonna|plan(?:ning)?\s+to|intend\s+to|"
        r"decided\s+to|ready\s+to)\b.{0,45}\b"
        r"(?:die|kill\s+myself|end\s+(?:it|my\s+life)|overdos\w*|jump|hang)\b|"
        r"\b(?:tomorrow|tonight|soon|after\s+\w+|in\s+\d+\s+(?:minutes?|hours?|days?))\b"
        r".{0,45}\b(?:die|suicid\w*|kill\s+myself|end\s+(?:it|my\s+life))\b"
    )),
    ("hypothetical", re.compile(
        r"(?i)\b(?:if\s+i|would|could|might|what\s+if)\b.{0,35}\b"
        r"(?:die|died|suicid\w*|kill\s+myself|end\s+my\s+life)\b"
    )),
    ("past_ideation", re.compile(
        r"(?i)\b(?:used\s+to|no\s+longer|formerly|in\s+the\s+past)\b"
        r".{0,45}\b(?:suicid\w*|want\w*\s+to\s+die|kill\s+myself|"
        r"self[- ]?harm\w*)\b"
    )),
    ("negated_intent", re.compile(
        r"(?i)\b(?:do\s+not|don['’]?t|never|not)\b.{0,20}\b"
        r"(?:want\s+to\s+die|wanna\s+die|kill\s+myself|end\s+my\s+life|"
        r"plan\s+to\s+(?:die|kill))\b"
    )),
)


def _is_emoji(character: str) -> bool:
    code = ord(character)
    return (
        0x1F1E6 <= code <= 0x1F1FF
        or 0x1F300 <= code <= 0x1FAFF
        or 0x2600 <= code <= 0x27BF
    )


def emoji_markers(text: str) -> list[str]:
    """Return semantic markers for Unicode emoji and ASCII emoticons."""
    text = str(text)
    markers: list[str] = []
    # Check explicit multi-codepoint mappings first.
    for emoji, meanings in EMOJI_MEANINGS.items():
        count = text.count(emoji)
        for _ in range(count):
            markers.extend(f"emoji_{meaning}" for meaning in meanings)
    explicitly_covered = {character for emoji in EMOJI_MEANINGS if emoji in text
                          for character in emoji}
    for character in text:
        if character in explicitly_covered or not _is_emoji(character):
            continue
        name = unicodedata.name(character, "").casefold()
        if not name or "variation selector" in name or "skin tone" in name:
            continue
        normalized = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
        if normalized:
            markers.append(f"emoji_{normalized}")
    for pattern, meanings in EMOTICONS:
        for _ in pattern.finditer(text):
            markers.extend(f"emoticon_{meaning}" for meaning in meanings)
    return list(dict.fromkeys(markers))


def temporal_markers(text: str) -> list[str]:
    """Return sentence-level suicide timeline/scope markers."""
    return [f"timeline_{name}" for name, pattern in TEMPORAL_PATTERNS
            if pattern.search(str(text))]


def augment_task1_text(text: str, *, emoji: bool = True,
                       temporal: bool = True) -> str:
    """Append semantic tokens while keeping the original post untouched."""
    markers: list[str] = []
    if emoji:
        markers.extend(emoji_markers(text))
    if temporal:
        markers.extend(temporal_markers(text))
    if not markers:
        return str(text)
    return f"{text}\n[semantic {' '.join(markers)}]"


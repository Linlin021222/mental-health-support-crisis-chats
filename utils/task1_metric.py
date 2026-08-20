"""Competition-aligned score helpers for Subtask 1.

The leaderboard gives 0.4 of the composite score to risk classification and
0.3 to evidence extraction.  Therefore the normalized Subtask 1 score shown
by the platform is a 4:3 weighted mean, not a 1:1 average.
"""


RISK_WEIGHT = 4.0 / 7.0
EVIDENCE_WEIGHT = 3.0 / 7.0


def task1_score(risk_f1, phrase_f1):
    """Return the normalized leaderboard Subtask 1 score."""
    return RISK_WEIGHT * float(risk_f1) + EVIDENCE_WEIGHT * float(phrase_f1)


def composite_score(risk_f1, phrase_f1, factor_f1):
    """Return the complete leaderboard score from its three raw metrics."""
    return 0.4 * float(risk_f1) + 0.3 * float(phrase_f1) + 0.3 * float(factor_f1)

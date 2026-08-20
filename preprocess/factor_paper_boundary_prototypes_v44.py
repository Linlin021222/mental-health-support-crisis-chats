"""Strict Table-III factor prototypes with explicit annotation boundaries."""
from configs.config import config
from preprocess.factor_paper_definitions_v36 import PAPER_FACTOR_DEFINITIONS


BOUNDARIES = [
    "A temporary sad mood alone is insufficient; the post must describe an existing mental disorder.",
    "This concerns bodily health or a physical characteristic, not poverty or a stressful event by itself.",
    "Ordinary or occasional use is insufficient; use must be uncontrolled or harmful.",
    "Distress alone is insufficient; the author must feel hopeless, trapped, stuck, or unable to see improvement.",
    "A named emotion alone is insufficient; the post must indicate difficulty controlling or regulating it.",
    "General sadness is insufficient; the author must negatively evaluate their own worth, adequacy, or burden.",
    "School stress alone is insufficient; the post must describe poor performance, failing, bad grades, or inability to study.",
    "A single expense or work stressor is insufficient unless material insecurity, poverty, unemployment, debt, or homelessness is present.",
    "This is violence or assault outside the family or home context; family or household abuse belongs to dysfunctional family.",
    "This requires past, resolved, or previous self-harm, suicidal thought, plan, or attempt; a purely current expression is insufficient.",
    "Difficulty socialising is not enough; the post must describe absent support, loneliness, isolation, rejection, neglect, or abandonment.",
    "This means difficulty making friends or socialising; a breakup, isolated conflict, or lack of support alone is insufficient.",
    "The harmful issue must involve the family or home environment; violence outside that context belongs to interpersonal violence.",
    "The suicidal person must be someone other than the author.",
    "The event is challenging but does not overwhelm the author's ability to cope; otherwise it may be traumatic experience.",
    "The event must overwhelm the author's ability to cope; an ordinary difficult event alone is a stressful life event.",
    "Rumination or distress alone is insufficient unless attention, memory, reasoning, decision-making, or another cognitive ability is impaired.",
    "A vague wish to die is insufficient; a potential method, instrument, location, preparation, or access to means must be described.",
    "The post must describe distress or conflict connected to sexual orientation, gender, or a same-sex relationship.",
    "Support must actually be received or available from another person or professional; wanting support is insufficient.",
    "The author must engage in an activity to deal with stress; substance misuse or merely naming a hobby without coping use is insufficient.",
    "A brief positive feeling is insufficient; the post must show hope, resilience, confidence, optimism, self-efficacy, or belief in recovery.",
    "The author must express responsibility for survival, health, another person, animal, role, promise, or duty; affection alone is insufficient.",
    "The post must express purpose, values, goals, motivation, or a reason for living; responsibility alone is a distinct factor.",
]

DIRECT = list(config.FACTOR_NLI_HYPOTHESES)
PAPER_BOUNDARY_PROTOTYPES = [
    [
        PAPER_FACTOR_DEFINITIONS[label],
        DIRECT[label],
        f"{PAPER_FACTOR_DEFINITIONS[label]} Annotation boundary: {BOUNDARIES[label]}",
    ]
    for label in range(config.NUM_FACTORS)
]

if len(PAPER_BOUNDARY_PROTOTYPES) != 24 or any(len(x) != 3 for x in PAPER_BOUNDARY_PROTOTYPES):
    raise ValueError("Paper boundary bank must contain exactly three prototypes per factor")

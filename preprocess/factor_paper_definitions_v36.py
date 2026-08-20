"""Verbatim-equivalent definitions transcribed from Table III of Li et al. (2025).

Punctuation is normalised for model input, but the semantic content follows
the published PFA taxonomy.  These short definitions deliberately remain
separate from the richer, training-derived semantic bank.
"""

PAPER_FACTOR_DEFINITIONS = [
    "Existing mental disorders, for example depression or personality disorders.",
    "Physical health issues or characteristics, for example COVID-19, obesity, or being underweight.",
    "The uncontrollable use of drugs, alcohol, or tobacco.",
    "Feelings of hopelessness, or feeling trapped or stuck.",
    "Emotion regulation difficulties, for example uncontrolled anxiety or anger.",
    "Negative feelings about the self, for example worthlessness, being a burden, or self-hate.",
    "Low school performance, for example failing tests or receiving bad grades.",
    "Unemployment, poverty, homelessness, or other low socioeconomic status.",
    "Violence or assault occurring outside home settings.",
    "Past, resolved, or previous self-harm and suicidal thought, plan, or attempt.",
    "Lack of friends, loneliness, isolation, rejection, neglect, or abandonment.",
    "Difficulty making new friends, socialising, or maintaining social relationships.",
    "Family-related issues that lead to negative impacts on the individual.",
    "Mentioning or describing another person's suicidal thoughts, attempt, or death.",
    "Events that pose challenges but do not lead to traumatic responses.",
    "Events that overwhelm an individual's ability to cope.",
    "Difficulty in cognitive abilities such as attention, memory, reasoning, or decision making.",
    "Description of a potential suicide means or access to that means.",
    "Issues involving gender or sexual orientation, or same-sex relationships.",
    "Support from family, partners, friends, or healthcare professionals.",
    "Activities an individual engages in to deal with stressful situations.",
    "An individual's positive psychological state of development.",
    "Awareness of responsibility to one's own health or survival and to others.",
    "Meaning in life involving cognitive, motivational, and affective components.",
]

if len(PAPER_FACTOR_DEFINITIONS) != 24:
    raise ValueError("The published PFA taxonomy must contain 24 definitions")


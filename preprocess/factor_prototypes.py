"""Human-auditable positive descriptions for the 24 factor labels.

Each entry expresses the same label from a different semantic angle.  The
supervised cross-encoder can therefore learn the concept rather than memorize
one fixed prompt.  All descriptions are positive entailment hypotheses; label
boundaries are handled through confusion-aware hard negatives in training.
"""

FACTOR_PROTOTYPES = [
    [
        "The author has an existing mental disorder, such as depression, anxiety, bipolar disorder, psychosis, or a personality disorder.",
        "The post describes clinically significant mental-health symptoms, a psychiatric diagnosis, or treatment for a mental-health condition.",
        "The author is struggling with an ongoing mental illness rather than only a temporary negative mood.",
    ],
    [
        "The author has a physical health problem or body characteristic, such as illness, chronic pain, disability, COVID-19, obesity, or being underweight.",
        "A bodily illness, disability, injury, chronic pain, or physical limitation is contributing to the author's distress.",
        "The post describes a health or body-related characteristic that affects daily life or wellbeing.",
    ],
    [
        "The author describes uncontrolled or harmful use of alcohol, drugs, or tobacco.",
        "Alcohol, recreational drugs, prescription misuse, intoxication, addiction, or withdrawal is present in the post.",
        "The author relies on or is harmed by a substance in a way that goes beyond ordinary occasional use.",
    ],
    [
        "The author feels hopeless, trapped, stuck, or believes that the future cannot improve.",
        "The post expresses that nothing will get better, there is no future, or the situation is impossible to escape.",
        "The author has lost hope that their life or circumstances can meaningfully improve.",
    ],
    [
        "The author has difficulty regulating emotions, such as uncontrolled anxiety, anger, sadness, or mood changes.",
        "The post describes overwhelming emotions, emotional outbursts, panic, extreme mood shifts, or being unable to calm down.",
        "The author's feelings are intense or unstable enough that emotional control is impaired.",
    ],
    [
        "The author expresses a negative view of themself, such as worthlessness, being a burden, inadequacy, shame, or self-hatred.",
        "The post contains self-hatred, shame, worthlessness, failure, ugliness, uselessness, or feeling inferior to others.",
        "The author evaluates themself negatively and believes they have little personal value.",
    ],
    [
        "The author describes poor school performance, such as failing tests or classes, bad grades, or being unable to study.",
        "The post reports declining grades, academic failure, inability to complete schoolwork, or serious difficulty studying.",
        "The author's academic performance is poor or has deteriorated, not merely that school feels stressful.",
    ],
    [
        "The author has low socioeconomic status, such as unemployment, poverty, debt, homelessness, or lack of basic material resources.",
        "The post describes serious financial hardship, inability to afford necessities, unstable housing, debt, or unemployment.",
        "A lack of money, housing, employment, or material security is an important part of the author's situation.",
    ],
    [
        "The author describes violence, assault, or victimization involving another person.",
        "The post reports bullying, physical attack, sexual assault, partner violence, threats, or interpersonal victimization.",
        "Another person has physically or sexually harmed, attacked, intimidated, or violently controlled the author.",
    ],
    [
        "The author describes past or recurrent self-harm, suicidal thoughts, a suicide plan, or a suicide attempt.",
        "The post refers to a history of wanting to die, self-injury, suicide planning, or having previously attempted suicide.",
        "Prior or repeated suicidal thinking or self-harming behaviour is part of the author's history.",
    ],
    [
        "The author lacks social support and feels lonely, isolated, rejected, neglected, or abandoned.",
        "The post says that nobody listens, cares, understands, stays, or provides help when the author needs it.",
        "The author experiences an absence of dependable supportive relationships.",
    ],
    [
        "The author has difficulty making friends, socializing, maintaining relationships, or interacting with other people.",
        "The post describes recurring conflict, breakup, friendship problems, rejection, or difficulty connecting with others.",
        "The author struggles to form or maintain interpersonal relationships and social interactions.",
    ],
    [
        "The author describes family conflict, neglect, abuse, separation, instability, or another dysfunctional family problem.",
        "The post reports harmful parenting, family abuse, neglect, constant arguments, abandonment, divorce, or an unsafe home.",
        "The author's family or home environment is unstable, conflictual, neglectful, or abusive.",
    ],
    [
        "The author mentions or describes another person's suicidal thoughts, suicide attempt, or death by suicide.",
        "The post discusses a friend, relative, partner, acquaintance, or public figure who attempted or died by suicide.",
        "Another person's suicidal behaviour or suicide death has been witnessed, learned about, or experienced by the author.",
    ],
    [
        "The author describes a challenging life event, such as a breakup, loss, work, legal, financial, or school problem.",
        "A recent change, failure, bereavement, conflict, deadline, job problem, move, or other major stressor is present.",
        "The author is reacting to a concrete stressful event or difficult change in life circumstances.",
    ],
    [
        "The author describes an overwhelming or disturbing traumatic experience, including severe abuse, violence, or loss.",
        "The post refers to lasting distress from trauma, assault, abuse, disaster, severe loss, or another deeply disturbing event.",
        "A past or recent experience exceeded the author's ability to cope and continues to affect them psychologically.",
    ],
    [
        "The author has difficulty with cognitive abilities, such as concentrating, remembering, thinking clearly, deciding, or solving problems.",
        "The post reports brain fog, poor concentration, memory problems, indecision, confusion, rumination, or inability to think clearly.",
        "The author's attention, memory, reasoning, decision making, or problem-solving ability is impaired.",
    ],
    [
        "The author describes a possible suicide method or access to that method, such as pills, a weapon, a rope, poison, or a height.",
        "The post names a suicide method, preparation, location, instrument, lethal substance, or available means.",
        "The author has considered, obtained, or can access something that could be used to attempt suicide.",
    ],
    [
        "The author describes a problem related to sexual orientation, same-sex relationships, gender identity, coming out, stigma, discrimination, or rejection.",
        "The post discusses distress caused by being lesbian, gay, bisexual, transgender, queer, questioning, or by others' reactions to that identity.",
        "Sexual-orientation or gender-identity stigma, disclosure, conflict, discrimination, or relationship concerns are present.",
    ],
    [
        "The author receives support, care, comfort, understanding, or practical help from other people.",
        "A friend, partner, relative, professional, community member, or other person actively listens to or helps the author.",
        "The post identifies a supportive relationship or source of emotional, practical, or professional assistance.",
    ],
    [
        "The author uses an activity or strategy to deal with stress, such as seeking help, therapy, distraction, exercise, hobbies, relaxation, or problem solving.",
        "The post describes something the author does to manage distress, regulate stress, solve the problem, or stay safe.",
        "The author actively copes through help seeking, treatment, planning, distraction, creativity, exercise, spirituality, or another strategy.",
    ],
    [
        "The author expresses a positive psychological state, such as hope, resilience, confidence, optimism, self-efficacy, or belief in recovery.",
        "The post shows inner strength, hope for improvement, confidence in coping, perseverance, or belief that recovery is possible.",
        "The author demonstrates psychological resources that help them recover, persist, or expect a better future.",
    ],
    [
        "The author expresses responsibility for their own survival or responsibility toward family, children, friends, pets, work, or other people.",
        "The post says the author must stay alive, act, or recover because another person, animal, role, promise, or duty depends on them.",
        "A sense of obligation, accountability, caregiving, duty, or responsibility influences the author's decisions.",
    ],
    [
        "The author expresses meaning or purpose in life through values, goals, reasons for living, motivation, or emotionally significant commitments.",
        "The post identifies a life purpose, future goal, personal value, dream, reason to live, or deeply meaningful activity or relationship.",
        "The author experiences something as giving life direction, significance, motivation, or a reason to continue living.",
    ],
]


if len(FACTOR_PROTOTYPES) != 24 or any(len(items) < 2 for items in FACTOR_PROTOTYPES):
    raise ValueError("Factor prototype bank must align with all 24 labels")

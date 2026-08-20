"""Boundary-aware risk/protective taxonomy for Task 2 reranking.

The first 19 labels are risk factors and the final five are protective
factors, following the competition order.  Each hypothesis contains an
inclusion description and a deliberately conservative exclusion boundary.
"""

FACTORS = [
    "mental health issues",
    "physical health/characteristic",
    "substance use",
    "hopelessness",
    "emotion dysregulation",
    "low self-esteem",
    "poor school performance",
    "low socio-economic status",
    "interpersonal violence",
    "prior self-harm or suicidal thought/attempt",
    "poor social support",
    "interpersonal difficulty",
    "dysfunctional family",
    "exposure to others' suicide",
    "stressful life event",
    "traumatic experience",
    "cognitive deficits",
    "suicide means (with access)",
    "sexual orientation related issues",
    "social support",
    "coping strategy",
    "psychological capital",
    "sense of responsibility",
    "meaning in life",
]

RISK_COUNT = 19
RISK_INDICES = tuple(range(RISK_COUNT))
PROTECTIVE_INDICES = tuple(range(RISK_COUNT, len(FACTORS)))


def _spec(formal, include, distinction, exclude, confusions=()):
    return {
        "formal": formal,
        "include": include,
        "distinction": distinction,
        "exclude": exclude,
        "confusions": tuple(confusions),
    }


SPECS = [
    _spec(
        "An existing psychiatric disorder, clinically significant mental-health condition, diagnosis, or treatment affects the author.",
        "Depression, anxiety disorder, bipolar disorder, OCD, psychosis, PTSD, eating disorder, psychiatric medication, therapy, or persistent disabling psychiatric symptoms.",
        "Temporary intense emotion belongs to emotion dysregulation unless an ongoing disorder or clinically significant condition is supported.",
        "Do not infer a diagnosis solely from sadness, anger, hopelessness, crying, or another person's illness.",
        ("emotion dysregulation", "hopelessness"),
    ),
    _spec(
        "A physical illness, disability, injury, chronic pain, body condition, or bodily characteristic affects the author.",
        "Disease, disability, chronic pain, injury, COVID-19, weight, being underweight or obese, appearance, sleep-related bodily problems, or physical limitation.",
        "Financial inability to obtain care is socioeconomic status; emotional effects of an event are not by themselves a physical-health factor.",
        "Do not assign ordinary tiredness, metaphorical pain, or bodily effects of a suicide method without a health condition.",
        ("low socio-economic status", "stressful life event"),
    ),
    _spec(
        "Harmful, uncontrolled, or dependent use of alcohol, recreational drugs, tobacco, or misused medication is present.",
        "Addiction, withdrawal, intoxication, repeated harmful use, getting high, uncontrolled drinking, or relying on a substance to function or cope.",
        "Correctly taking prescribed medication is treatment; pills discussed as a lethal method belong to suicide means with access.",
        "Do not assign from a single ordinary drink or a mere mention of drugs or medication without the author's use, misuse, dependence, intoxication, or harm.",
        ("coping strategy", "suicide means (with access)"),
    ),
    _spec(
        "The author believes the future cannot improve, feels trapped or stuck, sees no way out, or has lost hope.",
        "Hopeless, no future, no point, nothing will change, never get better, cannot go on, trapped, or suffering forever.",
        "Low self-esteem evaluates the self negatively; hopelessness evaluates future possibility and escape.",
        "Do not assign temporary disappointment or sadness when the author still expects improvement.",
        ("low self-esteem", "meaning in life", "psychological capital"),
    ),
    _spec(
        "The author has difficulty controlling or regulating intense sadness, anxiety, anger, panic, numbness, or rapidly changing emotions.",
        "Uncontrollable crying, panic, rage, agitation, emotional breakdown, impulsivity, mood swings, overwhelming feelings, or inability to calm down.",
        "Hopelessness concerns the future and low self-esteem concerns self-worth; this factor concerns intensity and control of affect.",
        "Do not assign every ordinary mention of sadness, anger, or anxiety when it remains proportionate and controlled.",
        ("mental health issues", "hopelessness", "low self-esteem"),
    ),
    _spec(
        "The author negatively evaluates their own worth, adequacy, appearance, value, or burden to others.",
        "Worthless, useless, ugly, pathetic, failure, burden, ashamed, hate myself, inadequate, or undeserving.",
        "Hopelessness concerns future improvement; low self-esteem is a judgment that the self is defective or inferior.",
        "Do not assign merely for regretting one action, quoting another person's insult, or temporary embarrassment without self-devaluation.",
        ("hopelessness", "poor school performance", "poor social support"),
    ),
    _spec(
        "The author's school or academic performance is poor or has materially deteriorated.",
        "Failing classes or tests, bad or falling grades, missed assignments, inability to study, or academic probation or dismissal.",
        "School pressure without poor performance is a stressful event; inability to focus is a cognitive deficit.",
        "Do not assign merely because the author attends or dislikes school, fears an exam, or feels pressure while performance remains adequate.",
        ("stressful life event", "cognitive deficits"),
    ),
    _spec(
        "Poverty, unemployment, debt, homelessness, or insufficient material resources significantly affects the author.",
        "Cannot afford necessities, no money, debt, poverty, homelessness, unemployment, lost job, food insecurity, or unstable housing.",
        "A job loss can also be stressful, but this factor requires material or socioeconomic insecurity.",
        "Do not assign ordinary budgeting, wanting more money, disliking a job, or choosing not to make one purchase.",
        ("stressful life event",),
    ),
    _spec(
        "The author experiences bullying, threats, assault, sexual violence, partner violence, coercion, or harmful interpersonal victimisation.",
        "Bullied, beaten, hit, raped, assaulted, abused, threatened, attacked, controlled, or repeatedly humiliated by another person.",
        "Family or household abuse may additionally be dysfunctional family; ordinary relationship conflict lacks violence or coercion.",
        "Do not assign arguments, rejection, rude comments, or self-directed harm without victimisation by another person.",
        ("dysfunctional family", "interpersonal difficulty", "traumatic experience"),
    ),
    _spec(
        "The post describes past, previous, resolved, or recurrent self-harm, suicidal thoughts, planning, behaviour, or an attempt by the author.",
        "Attempted suicide, overdosed before, cut myself, previous attempt, used to be suicidal, relapsing, scars, or repeated self-harm episodes.",
        "A currently available method is suicide means; another person's suicide is exposure to others' suicide.",
        "Do not assign solely from a current wish, current plan, or present crisis when no past, previous, repeated, or self-harm history is expressed.",
        ("suicide means (with access)", "exposure to others' suicide"),
    ),
    _spec(
        "The author lacks dependable emotional or practical support and experiences loneliness, isolation, rejection, neglect, or abandonment.",
        "No one cares, nobody understands, no one to talk to, alone, lonely, abandoned, ignored, unsupported, or help is unavailable or dismissive.",
        "Interpersonal difficulty concerns relationship functioning; social support requires help actually received or clearly available.",
        "Do not assign temporary physical solitude, chosen solitude, or a situation where dependable support is received and available.",
        ("social support", "interpersonal difficulty"),
    ),
    _spec(
        "The author has persistent difficulty making friends, socialising, resolving conflicts, or maintaining close relationships.",
        "Cannot make or keep friends, repeated rejection, recurring breakups or arguments, social connection feels unmanageable, or relationships repeatedly deteriorate.",
        "Poor social support is lack of help; this factor concerns creating and maintaining relationships.",
        "Do not assign one ordinary disagreement, chosen solitude, or merely having few people while relationships and support are adequate.",
        ("poor social support", "stressful life event", "interpersonal violence"),
    ),
    _spec(
        "The family or home environment is unstable, neglectful, abusive, rejecting, or persistently conflictual.",
        "Abusive or toxic parents, neglect, repeated family conflict, being kicked out, unsafe home, controlling caregivers, addiction in the family, or family rejection.",
        "Violence by family may also be interpersonal violence; problems outside the family are interpersonal difficulty.",
        "Do not assign a normal isolated disagreement, a supportive family member, or merely mentioning family without harmful functioning.",
        ("interpersonal violence", "interpersonal difficulty"),
    ),
    _spec(
        "The author is affected by another person's suicidal thoughts, suicide attempt, or death by suicide.",
        "A friend or relative attempted suicide, killed themselves, died by suicide, the author witnessed an attempt, or experiences suicide bereavement or fear.",
        "The suicidal person must be someone other than the author; non-suicide bereavement is a stressful or traumatic event.",
        "Do not assign when the only suicidal person is the author or another person's death is not described as suicide.",
        ("prior self-harm or suicidal thought/attempt", "stressful life event", "traumatic experience"),
    ),
    _spec(
        "A concrete recent or ongoing event creates substantial stress, such as breakup, bereavement, work, school, legal, financial, health, or relocation problems.",
        "Identifiable loss, failure, change, pressure, breakup, job loss, exam, eviction, illness crisis, death, divorce, deadline, or move worsens distress.",
        "Traumatic experience overwhelms coping and usually has severe or lasting effects; a stressful event need not be traumatic.",
        "Do not assign longstanding mood symptoms when no identifiable external event or change is present.",
        ("traumatic experience", "low socio-economic status", "poor school performance"),
    ),
    _spec(
        "An overwhelming event such as severe abuse, assault, violence, disaster, or profound loss exceeds coping ability and continues to affect the author.",
        "Traumatized, PTSD, flashbacks, nightmares, severe abuse, rape, witnessed violence, cannot forget, or lasting psychological alteration after an event.",
        "Interpersonal violence identifies victimisation; stressful life event does not require overwhelming severity or continuing traumatic impact.",
        "Do not label every breakup, exam, argument, or unpleasant memory as trauma without overwhelming severity or sustained traumatic effects.",
        ("stressful life event", "interpersonal violence"),
    ),
    _spec(
        "Attention, memory, concentration, reasoning, decision making, or clear thinking is impaired.",
        "Cannot focus or concentrate, brain fog, forgetfulness, confusion, impaired memory, inability to decide, thoughts stuck, or inability to organise tasks.",
        "Poor school performance is an outcome; intense emotion alone is not a cognitive deficit unless a cognitive ability is impaired.",
        "Do not assign ordinary indecision, changing one's mind, low motivation, or a single distraction.",
        ("poor school performance", "emotion dysregulation"),
    ),
    _spec(
        "A potential suicide method, instrument, location, or lethal substance is considered and the author has or plausibly can obtain access to it.",
        "Possessing, acquiring, storing, preparing, approaching, or being near pills, a firearm, rope, bridge, height, train, poison, vehicle, or another suicide method.",
        "Past use of a method belongs to prior attempt; vague ideation without a method or access is insufficient.",
        "Do not assign figurative phrases, general hypothetical discussion, ordinary medication possession, or an inaccessible method without suicidal-use context.",
        ("prior self-harm or suicidal thought/attempt",),
    ),
    _spec(
        "Distress is specifically related to sexual orientation, gender identity, coming out, same-sex relationships, stigma, discrimination, or identity-based rejection.",
        "Gay, lesbian, bisexual, trans, queer, questioning, coming out, homophobia, gender identity conflict, concealment, or LGBTQ+-related rejection or discrimination.",
        "General romantic rejection, bullying, or family conflict qualifies only when explicitly connected to orientation or gender identity.",
        "Do not infer orientation from gendered partners, sexual activity, assault, or the word partner without identity-related distress.",
        ("interpersonal difficulty", "dysfunctional family", "interpersonal violence"),
    ),
    _spec(
        "The author actually receives or clearly has available emotional, practical, professional, or community support from another person.",
        "A friend listened, family helped, a partner stayed, a therapist or doctor supports them, someone checked on them, or people provide reliable care or safety.",
        "Seeking help is coping; poor social support is the absence or failure of help. This label requires support received or clearly available.",
        "Do not assign when the author only asks strangers for help, says nobody cares, or merely wishes someone would provide support.",
        ("poor social support", "coping strategy"),
    ),
    _spec(
        "The author actively uses a behaviour or strategy to manage distress, solve problems, obtain help, or remain safe.",
        "Therapy, hotline, talking to someone, asking for advice, exercise, music, hobbies, journaling, distraction, breathing, medication, planning, or deliberate problem solving.",
        "Social support is help received; psychological capital is an inner positive resource; coping is an action taken by the author.",
        "Do not assign passive wishing, suicidal action, uncontrolled avoidance, or saying 'I need help' without help-seeking or another coping action.",
        ("social support", "psychological capital", "substance use"),
    ),
    _spec(
        "The author expresses hope, optimism, resilience, confidence, self-efficacy, perseverance, or belief that recovery is possible.",
        "I can get better, still have hope, things will improve, I am strong enough, I will keep trying, recovery is possible, gratitude, or retained agency despite distress.",
        "Meaning in life supplies purpose, responsibility supplies duty, and coping is an action; psychological capital is a positive internal belief or resource.",
        "Do not assign a polite thank-you, brief happiness, love mentioned only as loss, or a goal without hope, confidence, or resilience.",
        ("hopelessness", "meaning in life", "coping strategy"),
    ),
    _spec(
        "A duty, promise, caregiving role, or obligation toward self, family, children, friends, pets, work, or others influences the author's survival or actions.",
        "My children need me, I cannot hurt my family, I must care for my pet, I promised someone, people depend on me, or concern for others restrains suicide.",
        "Meaning in life is purpose and value; social support is help received; responsibility requires an obligation that guides behaviour.",
        "Do not assign merely because family or pets are mentioned, the author feels like a burden, or others care without an expressed duty or obligation.",
        ("meaning in life", "social support", "low self-esteem"),
    ),
    _spec(
        "The author identifies purpose, values, goals, dreams, commitments, or reasons for living that give life direction and significance.",
        "Reason to live, purpose, life goal, dream, future plan, something to live for, meaningful work or relationship, wanting to graduate, or valued future possibility.",
        "Responsibility is duty to others, psychological capital is confidence or hope, and coping is an action; meaning in life is enduring significance or direction.",
        "Do not assign the negative statement 'no reason to live', every short-term task, vague wish to feel better, enjoyable distraction, or obligation without meaningful purpose.",
        ("hopelessness", "sense of responsibility", "psychological capital"),
    ),
]

if len(SPECS) != len(FACTORS):
    raise ValueError("The factor taxonomy must contain exactly 24 specifications")

FACTOR_TO_ID = {name.casefold(): index for index, name in enumerate(FACTORS)}


def factor_group(index):
    return "risk" if index < RISK_COUNT else "protective"


def factor_query(index):
    spec = SPECS[index]
    group = factor_group(index).upper()
    return (
        f"Factor label: {FACTORS[index]}\n"
        f"Factor group: {group}\n"
        f"Formal meaning: {spec['formal']}\n"
        f"Count as positive when: {spec['include']}\n"
        f"Important distinction: {spec['distinction']}\n"
        f"Do not count when: {spec['exclude']}\n"
        "Decision target: Is this factor explicitly or implicitly present for "
        "the author of the current Reddit post? Risk and protective factors may "
        "coexist. Judge only this factor and do not infer missing facts."
    )


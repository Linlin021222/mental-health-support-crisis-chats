"""Project-wide settings for the suicide-risk competition."""
from pathlib import Path
import torch


class Config:
    ROOT_DIR = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT_DIR / "data"
    TRAIN_FILE = DATA_DIR / "train.xlsx"
    TEST_FILE = DATA_DIR / "leaderboard.xlsx"
    CACHE_DIR = DATA_DIR / "cache"
    OUTPUT_DIR = ROOT_DIR / "outputs"

    SEED = 42
    # Use the base model by default: the former ``large`` setting is needlessly
    # expensive for a local baseline and its hidden size was hard-coded wrongly.
    MODEL_NAME = "microsoft/deberta-v3-base"
    FACTOR_MODEL_NAME = "mental/mental-roberta-base"
    HIDDEN_SIZE = 768
    MAX_LENGTH = 384
    STRIDE = 128
    # RTX 5060 Laptop (8 GB): two windows retain long-post coverage without
    # flattening 12 transformer sequences into one batch.
    MAX_CHUNKS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 4
    # The strict holdout peaked at epoch 3 (Task 1 = 0.7461) and declined at
    # epoch 4.  With no validation fold in final training, use that selected
    # epoch count and then fit those three epochs on every labelled post.
    FULL_TRAIN_EPOCHS = 3
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 8
    NUM_WORKERS = 0                 # Windows-safe default
    BACKBONE_LR = 1e-5
    HEAD_LR = 3e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    FP16 = torch.cuda.is_available()
    GRAD_CLIP = 1.0
    N_FOLDS = 5
    RISK_LABELS = {"Indicator": 0, "Ideation": 1, "Behavior": 2, "Attempt": 3}
    ID2RISK = {value: key for key, value in RISK_LABELS.items()}
    NUM_RISK_CLASSES = len(RISK_LABELS)
    FACTOR_LABELS = [
        "mental health issues", "physical health/characteristic", "substance use", "hopelessness",
        "emotion dysregulation", "low self-esteem", "poor school performance", "low socio-economic status",
        "interpersonal violence", "prior self-harm or suicidal thought/attempt", "poor social support",
        "interpersonal difficulty", "dysfunctional family", "exposure to others' suicide", "stressful life event",
        "traumatic experience", "cognitive deficits", "suicide means (with access)",
        "sexual orientation related issues", "social support", "coping strategy", "psychological capital",
        "sense of responsibility", "meaning in life",
    ]
    FACTOR_DESCRIPTIONS = [
        "mental illness, depression, anxiety, bipolar disorder, psychosis or other mental health problem",
        "physical illness, disability, pain, body characteristic or health limitation",
        "alcohol, drugs, intoxication, addiction or substance misuse",
        "hopelessness, no future, no possibility of improvement or feeling that nothing can change",
        "overwhelming anger, sadness, anxiety, mood swings or inability to regulate emotions",
        "worthlessness, self-hatred, shame, inadequacy or negative self-evaluation",
        "bad grades, academic failure, school pressure or difficulty studying",
        "poverty, unemployment, homelessness, debt or lack of material resources",
        "abuse, assault, bullying, domestic violence or interpersonal victimisation",
        "previous self-harm, suicidal thoughts, suicide planning or suicide attempt",
        "loneliness, isolation, rejection or absence of supportive relationships",
        "conflict, breakup, friendship problems, relationship problems or difficulty with others",
        "family conflict, neglect, abuse, separation or an unstable family environment",
        "another person's suicide, suicide attempt or suicidal behaviour",
        "recent loss, breakup, work problem, legal problem or other stressful life event",
        "trauma, abuse, severe loss or a disturbing past experience",
        "rigid thinking, impaired decision making, rumination or difficulty solving problems",
        "access to pills, weapons, rope, poison or another suicide method",
        "sexual orientation or gender identity related stigma, rejection or conflict",
        "receiving care, help, comfort, understanding or practical support from other people",
        "healthy coping, distraction, therapy, exercise, hobbies, problem solving or help seeking",
        "hope, resilience, confidence, optimism, self-control or belief in recovery",
        "responsibility or obligation toward family, children, friends, pets, work or other people",
        "purpose, values, goals, reasons for living or a sense that life is meaningful",
    ]
    # Table III of Li et al. (2025) defines several boundaries more narrowly
    # than the earlier hand-written prompts (notably interpersonal violence vs
    # dysfunctional family, and stressful events vs traumatic experiences).
    # Keep these paper-aligned hypotheses separate: existing checkpoints were
    # trained with FACTOR_DESCRIPTIONS and must remain load-compatible, while
    # the zero-shot NLI expert below can use the corrected taxonomy verbatim.
    FACTOR_NLI_HYPOTHESES = [
        "The author has an existing mental disorder, such as depression, anxiety, bipolar disorder, psychosis, or a personality disorder.",
        "The author has a physical health problem or body characteristic, such as illness, chronic pain, disability, COVID-19, obesity, or being underweight.",
        "The author describes uncontrolled or harmful use of alcohol, drugs, or tobacco.",
        "The author feels hopeless, trapped, stuck, or believes that the future cannot improve.",
        "The author has difficulty regulating emotions, such as uncontrolled anxiety, anger, sadness, or mood changes.",
        "The author expresses a negative view of themself, such as worthlessness, being a burden, inadequacy, shame, or self-hatred.",
        "The author describes poor school performance, such as failing tests or classes, bad grades, or being unable to study.",
        "The author has low socioeconomic status, such as unemployment, poverty, debt, homelessness, or lack of basic material resources.",
        "The author describes violence, assault, or victimization involving people outside the family or home setting.",
        "The author describes past or previous self-harm, suicidal thoughts, a suicide plan, or a suicide attempt, including a current post that refers to this prior history.",
        "The author lacks social support and feels lonely, isolated, rejected, neglected, or abandoned.",
        "The author has difficulty making friends, socializing, maintaining relationships, or interacting with other people.",
        "The author describes family conflict, neglect, abuse, separation, instability, or another dysfunctional family problem that harms them.",
        "The author mentions or describes another person's suicidal thoughts, suicide attempt, or death by suicide.",
        "The author describes a challenging life event, such as a breakup, loss, work, legal, financial, or school problem, without necessarily describing a traumatic response.",
        "The author describes an experience so overwhelming or disturbing that it exceeds their ability to cope, including severe abuse, violence, or loss.",
        "The author has difficulty with cognitive abilities, such as concentrating, remembering, thinking clearly, deciding, or solving problems.",
        "The author describes a possible suicide method or access to that method, such as pills, a weapon, a rope, poison, a height, or another means.",
        "The author describes a problem related to sexual orientation, same-sex relationships, gender identity, coming out, stigma, discrimination, or rejection.",
        "The author receives support, care, comfort, understanding, or practical help from family, a partner, friends, healthcare professionals, or other people.",
        "The author uses an activity or strategy to deal with stress, such as seeking help, therapy, distraction, exercise, hobbies, relaxation, or problem solving.",
        "The author expresses a positive psychological state, such as hope, resilience, confidence, optimism, self-efficacy, or belief in recovery.",
        "The author expresses responsibility for their own health or survival, or responsibility toward family, children, friends, pets, work, or other people.",
        "The author expresses meaning or purpose in life through values, goals, reasons for living, motivation, or emotionally significant commitments.",
    ]
    FACTOR2ID = {name: i for i, name in enumerate(FACTOR_LABELS)}
    ID2FACTOR = {i: name for name, i in FACTOR2ID.items()}
    NUM_FACTORS = len(FACTOR_LABELS)
    # Strict user-holdout sweep: 0.55 with an 8-token span improved Task 1
    # from 0.7404 to 0.752741 by reducing low-confidence/overlong evidence.
    EVIDENCE_THRESHOLD = 0.55
    MAX_EVIDENCE_TOKENS = 8
    TASK1_V2_ENSEMBLE_WEIGHT = 0.20
    TASK1_V2_ORDINAL_WEIGHT = 0.25
    # Five-fold Task 1 checkpoints are used only after task1-cv has produced
    # all folds. Inference falls back to the validated full-data models while
    # those checkpoints are absent.
    TASK1_USE_CV_ENSEMBLE = True
    MAX_EVIDENCE_PHRASES = 5
    TASK1_CV_MAX_EVIDENCE_PHRASES = 3
    TASK1_CV_TEST_WEIGHT = 0.50
    # Task 1 rationale-v3: preserve local evidence continuity and its expected
    # length while using smaller learning rates in the lower transformer
    # layers.  These switches only affect the optional user-disjoint CV model;
    # the validated full-data submission remains the fallback until CV adopts
    # the new branch.
    TASK1_LAYERWISE_LR_DECAY = 0.90
    TASK1_EVIDENCE_TRANSITION_WEIGHT = 0.04
    TASK1_EVIDENCE_COUNT_WEIGHT = 0.04
    TASK1_HEAD_RDROP_WEIGHT = 0.15
    FACTOR_THRESHOLD = 0.50
    FACTOR_PREVALENCE_FLOOR_RATIO = 0.85
    # Strict user-holdout ablation: the semantic-alignment model, the former
    # weighted-ASL checkpoint, and the sparse lexical model are complementary.
    # Their 0.50/0.25/0.25 blend reached 0.451965 Macro F1, versus 0.372677
    # for the former standalone neural model.
    FACTOR_USE_CV_ENSEMBLE = True
    FACTOR_SEMANTIC_MODEL_WEIGHT = 0.70
    FACTOR_LEGACY_MODEL_WEIGHT = 0.00
    FACTOR_CPU_ENSEMBLE_WEIGHT = 0.30
    # V48 keeps the validated TF-IDF architecture but trains it on user folds
    # balanced for all 24 factors. Strict fixed-weight OOF improved by .00505
    # with a positive user-bootstrap lower bound; NB-SVM was rejected.
    # Leaderboard rollback (2026-08-13): V48 improved local factor-balanced
    # OOF but reduced official Task 2. Restore the earlier accepted sparse
    # component that produced the 0.6065 official score.
    FACTOR_USE_BALANCED_SPARSE_V48 = False
    FACTOR_TOPK_RATIO = 1.10
    FACTOR_EPOCHS = 5
    FACTOR_SEMANTIC_LOSS_WEIGHT = 0.20
    FACTOR_SEMANTIC_TEMPERATURE = 0.10
    FACTOR_CONTEXTUAL_LABEL_INIT = True
    # Both switches were isolated on strict fold 0 and reduced Macro F1 from
    # 0.4329 to 0.3514, so production retains the validated legacy initialiser.
    FACTOR_PAPER_DEFINITION_INIT = False
    FACTOR_SEMANTIC_CLASSIFIER_INIT = False
    # Repeated entries in the released factor lists are treated as annotation
    # salience, not as duplicate output labels.  Count 1 keeps unit weight;
    # repeated evidence receives a conservative log-scaled positive weight.
    FACTOR_OCCURRENCE_ALPHA = 0.0
    FACTOR_TAIL_SAMPLING_ALPHA = 0.50
    FACTOR_RANKING_LOSS_WEIGHT = 0.10
    FACTOR_NLI_MODEL_NAME = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
    FACTOR_NLI_BATCH_SIZE = 12
    FACTOR_NLI_MAX_LENGTH = 512
    FACTOR_NLI_MAX_CHUNKS = 3
    FACTOR_CROSS_ENCODER_BATCH_SIZE = 4
    FACTOR_CROSS_ENCODER_ACCUMULATION = 4
    FACTOR_CROSS_ENCODER_EPOCHS = 2
    FACTOR_CROSS_ENCODER_LR = 7e-6
    # Five-fold OOF: a 50/50 blend with the validated MentalRoBERTa/TF-IDF
    # base reaches 0.5910 Macro F1.  It is within 0.0011 of the more complex
    # three-component optimum and is the stable median cross-fit setting.
    FACTOR_USE_CROSS_ENCODER = True
    FACTOR_CROSS_BASE_WEIGHT = 0.50
    FACTOR_CROSS_WEIGHT = 0.50
    FACTOR_CROSS_TOPK_RATIO = 1.10
    # Prototype-bank refinement is gated by its calibration JSON. Merely
    # creating one fold cannot change a production submission.
    FACTOR_USE_CROSS_ENCODER_V2 = True
    # V21 improved nested OOF only when it also reduced the global prevalence
    # ratio from 1.10 to 1.00.  The official leaderboard remained 0.6065 and a
    # fixed-ratio audit left only +0.00216 OOF with 70.1% bootstrap support.
    # Keep its artifacts for analysis but do not alter production submissions.
    FACTOR_USE_SIGNED_GRAPH_V21 = False
    # Prototype-bank refinement: positive annotations repeated in the source
    # list are treated as a weak, capped salience signal (the submitted target
    # remains binary).  Multi-instance training lets a positive factor be
    # supported by any of the selected long-post windows.
    FACTOR_PROTOTYPE_OCCURRENCE_ALPHA = 0.20
    FACTOR_PROTOTYPE_MAX_REPEAT_BOOST = 1.50
    FACTOR_PROTOTYPE_MAX_POSITIVE_WEIGHT = 6.00
    FACTOR_PROTOTYPE_TRAIN_BATCH_SIZE = 1
    FACTOR_PROTOTYPE_ACCUMULATION = 16
    FACTOR_PROTOTYPE_TRAIN_MAX_CHUNKS = 3
    # MHLAT-v4 is an optional continuation of each already-trained
    # MentalRoBERTa fold.  A second label-specific reading hop and a
    # label-centre contrastive loss target ambiguous/tail labels.  Production
    # uses it only if the nested user-disjoint gate in its calibration passes.
    FACTOR_USE_MHLAT_V4 = True
    FACTOR_MHLAT_EPOCHS = 2
    FACTOR_MHLAT_BACKBONE_LR = 4e-6
    FACTOR_MHLAT_HEAD_LR = 2e-5
    FACTOR_MHLAT_CONTRASTIVE_WEIGHT = 0.06
    FACTOR_MHLAT_SEMANTIC_WEIGHT = 0.10
    FACTOR_MHLAT_TEMPERATURE = 0.12
    FACTOR_LOSS = "asymmetric"
    # DB Loss was tested on the identical strict split and reached 0.3135,
    # below weighted ASL (0.3727), so ASL remains the production default.
    FACTOR_STANDALONE_LOSS = "asymmetric"
    ASL_GAMMA_NEG = 4.0
    ASL_GAMMA_POS = 0.0
    ASL_CLIP = 0.05
    # We keep all three objectives active.  These names are consumed by the
    # existing MultiTaskLoss implementation.
    EVIDENCE_POS_WEIGHT = 25.0
    LOSS_WEIGHTS = {"risk": 1.0, "evidence": 1.0, "factor": 1.0}


config = Config()

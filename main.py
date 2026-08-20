# ============================================================
# Suicide Risk Detection Competition
# Main Entry
# ============================================================


import argparse
import torch


from utils.seed import (
    seed_everything
)





def main():



    parser=argparse.ArgumentParser()



    parser.add_argument(

        "--mode",

        type=str,

        default="full-run",

        choices=[

            "full-run",

            "train-full",

            "train-strict",

            "factor-strict",

            "factor-full",

            "factor-cv",

            "factor-cross-cv",

            "factor-cross-v2-fold0",

            "factor-cross-v2-cv",

            "factor-mhlat-v4-fold0",

            "factor-mhlat-v4-cv",
            "factor-llm-lexical-v6-generate",
            "factor-llm-lexical-v6-cv",
            "factor-llm-lexical-v6-full",
            "factor-qwen-direct-v7",
            "factor-heterogeneous-v8",
            "factor-count-aux-v9-fold0",
            "factor-qwen-qlora-v10-fold0",
            "factor-multiview-v11-fold0",
            "factor-label-graph-v12",
            "factor-external-transfer-v13-fold0",
            "factor-frozen-expert-v14-fold0",
            "factor-semantic-contrast-v15-fold0",
            "factor-sentence-evidence-v16-fold0",
            "factor-sentence-evidence-v17-fold0",
            "factor-sentence-evidence-v18-fold0",
            "factor-boundary-review-v19-generate",
            "factor-boundary-review-v19-translate",
            "factor-boundary-review-v19-repair-translation",
            "factor-boundary-review-v19-import",
            "factor-definition-mil-v20-fold0",
            "factor-signed-graph-stack-v21",
            "factor-tail-augmentation-v22-generate",
            "factor-tail-augmentation-v22-cv",
            "factor-definition-ranker-v23-fold0",
            "factor-definition-rank-fusion-v24-fold0",
            "factor-definition-oof-ranker-v25-fold0",
            "factor-definition-selective-v26-fold0",
            "factor-sentence-evidence-v27-fold0",
            "factor-sentence-evidence-v27-cv",
            "factor-nested-decoder-v28",
            "factor-pairwise-ranking-v29-fold0",
            "factor-stable-expert-routing-v30",
            "factor-dual-rationale-v31-generate",
            "factor-grounded-retrieval-v32-fold0",
            "factor-noise-aware-cross-v33-fold0",
            "factor-fewshot-boundary-v34-fold0",
            "factor-large-label-attention-v35-fold0",
            "factor-paper-dual-branch-v36-fold0",
            "factor-paper-dual-branch-v37-cv",
            "factor-definition-retrieval-v38-fold0",
            "factor-definition-retrieval-tail-v39-fold0",
            "factor-prototype-retrieval-v40-fold0",
            "factor-prototype-contrast-v41-fold0",
            "factor-prototype-contrast-v41-cv",
            "factor-paper-relational-router-v42",
            "factor-protective-branch-v43-fold0",
            "factor-joint-pfa-v45-fold0",
            "factor-deberta-expert-v46-fold0",
            "factor-balanced-calibration-v47",
            "factor-balanced-sparse-v48",
            "factor-balanced-sparse-v48-full",
            "factor-rare-semantic-v49-fold0",
            "factor-rare-semantic-v49-cv",
            "factor-boundary-lexicon-v50",
            "factor-meaning-mil-v51-fold0",
            "factor-meaning-mil-v51-cv",
            "factor-balanced-neural-v53-fold0",
            "factor-meaning-boundary-v54",
            "factor-context-gate-v64",
            "factor-dedup-occurrence-v65",
            "case-syntax-v66",
            "semi-supervised-v67",
            "syntax-aux-v68",
            "factor-targeted-repair-v69",
            "factor-aligned-decoder-v70",
            "factor-paper-boundary-cross-v44-fold0",

            "task1-cv-fold0",

            "task1-cv",

            "task1-mental-strict",

            "task1-mental-full",

            "task1-risk-v5",

            "task1-clinical-v6",

            "task1-evidence-v7",

            "task1-evidence-v8",

            "task1-evidence-refine-v9",

            "task1-risk-v10",

            "task1-lexical-v11",

            "task1-user-context-v12",

            "task1-evidence-reranker-v13",

            "task1-evidence-reranker-v13-hybrid",

            "task1-seed-ensemble-v14",

            "task1-seed-ensemble-v14-weights",

            "task1-nli-v15",

            "task1-oof-reranker-v16",

            "task1-dynamic-topk-v17",

            "task1-candidate-v18",

            "task1-v18-full",

            "task1-llm-v19",

            "task1-oof-stack-v20",

            "task1-lexical-reranker-v21",

            "task1-nested-ensemble-v22",

            "task1-position-reranker-v23",

            "task1-reranker-ensemble-v24",

            "task1-atomic-v25",

            "task1-atomic-refine-v26",

            "task1-risk-only-v27",

            "task1-seed-evidence-v28",

            "task1-boundary-crossval-v29",

            "task1-boundary-augment-v30",

            "task1-evidence-lexicon-v31",

            "task1-clinical-reranker-v32",

            "task1-boundary-clinical-v33",

            "task1-temporal-v34",

            "task1-oof-decoder-v35",

            "task1-oof-risk-v36",

            "task1-oof-meta-v37",

            "task1-ordinal-lexical-v38",

            "task1-boundary-model-v39",

            "task1-boundary-components-v40",

            "task1-frozen-rehead-v41",

            "task1-evidence-conditioned-v42",

            "task1-selective-gate-v43",

            "task1-style-personality-v44",

            "task1-pseudo-evidence-v45",

            "task1-large-v46",

            "task1-qwen-lora-v47",

            "task1-qwen-oof-v48",

            "task1-qwen-verbalizer-v49",

            "task1-alignment-v50",

            "task1-repaired-multiseed-v51",

            "task1-rationale-augment-v52",

            "task1-rationale-augment-v52-full",
            "task1-qwen7b-verbalizer-v53",
            "task1-factor-bridge-v54",
            "task1-evidence-count-prior-v55",
            "task1-local-cf-v56-generate",
            "task1-local-cf-v56-train",
            "task1-local-diverse-cf-v57-generate",
            "task1-local-diverse-cf-v57-train",
            "task1-local-diverse-cf-v57-full",
            "task1-factor-trajectory-v58",
            "task1-factor-trajectory-gate-v59",
            "task1-factor-trajectory-v58-full",
            "task1-dynamic-influence-v60",
            "task1-windowed-trajectory-v61",
            "task1-same-post-factor-v62",
            "task1-polarity-v63",

            "train",

            "predict",
            "predict-v38",
            "predict-v69",
            "predict-v70",
            "predict-qwen3-8b",
            "predict-qwen3-8b-blend",
            "predict-qwen3-8b-attempt-calibrated",

            "private-eval",

            "private-group-cv",

            "ensemble"

        ]

    )

    parser.add_argument("--build-cache", action="store_true", help="Rebuild token caches from the Excel files")



    args=parser.parse_args()

    if args.mode == "full-run":
        from trainer.factor_train import STRICT_CALIBRATION
        if not STRICT_CALIBRATION.exists():
            raise FileNotFoundError(
                "Run `python main.py --mode factor-strict` first. The final run requires "
                "leak-free MentalRoBERTa factor thresholds."
            )

    if args.mode == "private-eval":
        from baseline import private_evaluate
        private_evaluate()
        return

    if args.mode == "private-group-cv":
        from baseline import private_group_cv
        private_group_cv()
        return

    if args.mode == "factor-strict":
        from trainer.factor_train import train_factor_strict
        train_factor_strict()
        return

    if args.mode == "factor-cross-cv":
        from trainer.factor_cross_encoder_cv import train_cross_encoder_cv
        train_cross_encoder_cv()
        return

    if args.mode == "factor-cv":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 2 CV training requires CUDA.")
        from trainer.factor_cv import train_factor_cv
        train_factor_cv()
        return

    if args.mode == "factor-cross-v2-fold0":
        from trainer.factor_cross_encoder_v2 import train_factor_cross_encoder_v2
        train_factor_cross_encoder_v2(only_fold0=True)
        return

    if args.mode == "factor-cross-v2-cv":
        from trainer.factor_cross_encoder_v2 import train_factor_cross_encoder_v2
        train_factor_cross_encoder_v2(only_fold0=False)
        return

    if args.mode == "factor-mhlat-v4-fold0":
        if not torch.cuda.is_available():
            raise RuntimeError("MHLAT-v4 training requires CUDA.")
        from trainer.factor_mhlat_v4 import train_factor_mhlat_v4
        train_factor_mhlat_v4(only_fold0=True)
        return

    if args.mode == "factor-mhlat-v4-cv":
        if not torch.cuda.is_available():
            raise RuntimeError("MHLAT-v4 training requires CUDA.")
        from trainer.factor_mhlat_v4 import train_factor_mhlat_v4
        train_factor_mhlat_v4(only_fold0=False)
        return

    if args.mode == "factor-llm-lexical-v6-generate":
        from trainer.factor_llm_lexical_v6 import generate
        generate()
        return

    if args.mode == "factor-llm-lexical-v6-cv":
        from trainer.factor_llm_lexical_v6 import cross_validate
        cross_validate()
        return

    if args.mode == "factor-llm-lexical-v6-full":
        from trainer.factor_llm_lexical_v6 import train_full
        train_full()
        return

    if args.mode == "factor-qwen-direct-v7":
        from trainer.factor_qwen_direct_v7 import predict_strict, evaluate
        predict_strict()
        evaluate()
        return

    if args.mode == "factor-heterogeneous-v8":
        from trainer.factor_heterogeneous_stack_v8 import cross_validate
        cross_validate()
        return

    if args.mode == "factor-count-aux-v9-fold0":
        from trainer.factor_count_aux_v9 import train_fold0
        train_fold0()
        return

    if args.mode == "factor-qwen-qlora-v10-fold0":
        from trainer.factor_qwen_qlora_v10 import train_fold0
        train_fold0()
        return

    if args.mode == "factor-multiview-v11-fold0":
        from trainer.factor_multiview_prompt_v11 import train_fold0
        train_fold0()
        return

    if args.mode == "factor-label-graph-v12":
        from trainer.factor_label_graph_v12 import main as factor_label_graph_v12
        factor_label_graph_v12()
        return

    if args.mode == "factor-external-transfer-v13-fold0":
        from trainer.factor_external_transfer_v13 import train_fold0
        train_fold0()
        return

    if args.mode == "factor-frozen-expert-v14-fold0":
        from trainer.factor_frozen_expert_v14 import train_fold0
        train_fold0()
        return

    if args.mode == "factor-semantic-contrast-v15-fold0":
        from trainer.factor_semantic_contrast_v15 import train_fold0
        train_fold0()
        return

    if args.mode == "factor-sentence-evidence-v16-fold0":
        from trainer.factor_sentence_evidence_v16 import train_fold0
        train_fold0()
        return

    if args.mode == "factor-sentence-evidence-v17-fold0":
        from trainer.factor_sentence_evidence_v17 import train_fold0
        train_fold0()
        return

    if args.mode == "factor-sentence-evidence-v18-fold0":
        from trainer.factor_sentence_evidence_v18 import train_fold0
        train_fold0()
        return

    if args.mode == "factor-boundary-review-v19-generate":
        from trainer.factor_boundary_review_v19 import generate_reviews
        generate_reviews()
        return

    if args.mode == "factor-boundary-review-v19-translate":
        from trainer.factor_boundary_translate_v19 import translate
        translate()
        return

    if args.mode == "factor-boundary-review-v19-repair-translation":
        from trainer.factor_boundary_translate_v19 import repair_fallbacks
        repair_fallbacks()
        return

    if args.mode == "factor-boundary-review-v19-import":
        from trainer.factor_boundary_review_import_v19 import import_reviews
        import_reviews()
        return

    if args.mode == "factor-definition-mil-v20-fold0":
        from trainer.factor_definition_mil_v20 import train_fold0
        train_fold0()
        return

    if args.mode == "factor-signed-graph-stack-v21":
        from trainer.factor_signed_graph_stack_v21 import cross_validate
        cross_validate()
        return

    if args.mode == "factor-tail-augmentation-v22-generate":
        from trainer.factor_tail_augmentation_v22 import generate
        generate()
        return

    if args.mode == "factor-tail-augmentation-v22-cv":
        from trainer.factor_tail_augmentation_v22 import cross_validate
        cross_validate()
        return

    if args.mode == "factor-definition-ranker-v23-fold0":
        from trainer.factor_definition_ranker_v23 import train_fold0
        train_fold0()
        return

    if args.mode == "factor-definition-rank-fusion-v24-fold0":
        from trainer.factor_definition_rank_fusion_v24 import train_fold0
        train_fold0()
        return

    if args.mode == "factor-definition-oof-ranker-v25-fold0":
        from trainer.factor_definition_oof_ranker_v25 import train_fold0
        train_fold0()
        return

    if args.mode == "factor-definition-selective-v26-fold0":
        from trainer.factor_definition_selective_v26 import train_fold0
        train_fold0()
        return

    if args.mode == "factor-sentence-evidence-v27-fold0":
        from trainer.factor_sentence_evidence_cv_v27 import cross_validate
        cross_validate(only_fold0=True)
        return

    if args.mode == "factor-sentence-evidence-v27-cv":
        from trainer.factor_sentence_evidence_cv_v27 import cross_validate
        cross_validate(only_fold0=False)
        return

    if args.mode == "factor-nested-decoder-v28":
        from trainer.factor_nested_decoder_v28 import cross_validate
        cross_validate()
        return

    if args.mode == "factor-pairwise-ranking-v29-fold0":
        from trainer.factor_pairwise_ranking_v29 import train_fold0
        train_fold0()
        return

    if args.mode == "factor-stable-expert-routing-v30":
        from trainer.factor_stable_expert_routing_v30 import cross_validate
        cross_validate()
        return

    if args.mode == "factor-dual-rationale-v31-generate":
        from trainer.factor_dual_rationale_v31 import generate
        generate()
        return

    if args.mode == "factor-grounded-retrieval-v32-fold0":
        from trainer.factor_grounded_retrieval_v32 import evaluate_fold0
        evaluate_fold0()
        return

    if args.mode == "factor-noise-aware-cross-v33-fold0":
        from trainer.factor_noise_aware_cross_v33 import train_fold0
        train_fold0()
        return

    if args.mode == "factor-fewshot-boundary-v34-fold0":
        from trainer.factor_fewshot_boundary_v34 import run
        run()
        return

    if args.mode == "factor-large-label-attention-v35-fold0":
        from trainer.factor_large_label_attention_v35 import train_fold0
        train_fold0()
        return

    if args.mode == "factor-paper-dual-branch-v36-fold0":
        from trainer.factor_paper_dual_branch_v36 import train_fold0
        train_fold0()
        return

    if args.mode == "factor-paper-dual-branch-v37-cv":
        from trainer.factor_paper_dual_branch_cv_v37 import cross_validate
        cross_validate()
        return

    if args.mode == "factor-definition-retrieval-v38-fold0":
        from trainer.factor_definition_retrieval_v38 import main as factor_definition_v38
        factor_definition_v38()
        return

    if args.mode == "factor-definition-retrieval-tail-v39-fold0":
        from trainer.factor_definition_retrieval_tail_v39 import main as factor_definition_v39
        factor_definition_v39()
        return

    if args.mode == "factor-prototype-retrieval-v40-fold0":
        from trainer.factor_prototype_retrieval_v40 import main as factor_prototype_v40
        factor_prototype_v40()
        return

    if args.mode == "factor-prototype-contrast-v41-fold0":
        from trainer.factor_prototype_contrast_v41 import main as factor_prototype_v41
        factor_prototype_v41(only_fold0=True)
        return

    if args.mode == "factor-prototype-contrast-v41-cv":
        from trainer.factor_prototype_contrast_v41 import main as factor_prototype_v41
        factor_prototype_v41(only_fold0=False)
        return

    if args.mode == "factor-paper-relational-router-v42":
        from trainer.factor_paper_relational_router_v42 import main as factor_relational_v42
        factor_relational_v42()
        return

    if args.mode == "factor-protective-branch-v43-fold0":
        from trainer.factor_protective_branch_v43 import main as factor_protective_v43
        factor_protective_v43()
        return

    if args.mode == "factor-joint-pfa-v45-fold0":
        from trainer.factor_joint_pfa_v45 import main as factor_joint_pfa_v45
        factor_joint_pfa_v45()
        return

    if args.mode == "factor-deberta-expert-v46-fold0":
        from trainer.factor_deberta_expert_v46 import main as factor_deberta_v46
        factor_deberta_v46()
        return

    if args.mode == "factor-balanced-calibration-v47":
        from trainer.factor_balanced_calibration_v47 import main as factor_balanced_v47
        factor_balanced_v47()
        return


    if args.mode == "factor-balanced-sparse-v48":
        from trainer.factor_balanced_sparse_v48 import cross_validate
        cross_validate()
        return

    if args.mode == "factor-balanced-sparse-v48-full":
        from trainer.factor_balanced_sparse_v48 import train_full
        train_full()
        return

    if args.mode == "factor-rare-semantic-v49-fold0":
        from trainer.factor_rare_semantic_v49 import cross_validate
        cross_validate(only_fold0=True)
        return

    if args.mode == "factor-rare-semantic-v49-cv":
        from trainer.factor_rare_semantic_v49 import cross_validate
        cross_validate(only_fold0=False)
        return

    if args.mode == "factor-boundary-lexicon-v50":
        from trainer.factor_boundary_lexicon_v50 import main as factor_boundary_v50
        factor_boundary_v50()
        return

    if args.mode == "factor-meaning-mil-v51-fold0":
        from trainer.factor_meaning_mil_v51 import cross_validate
        cross_validate(only_fold0=True)
        return

    if args.mode == "factor-meaning-mil-v51-cv":
        from trainer.factor_meaning_mil_v51 import cross_validate
        cross_validate(only_fold0=False)
        return

    if args.mode == "factor-balanced-neural-v53-fold0":
        from trainer.factor_balanced_neural_v53 import train_fold0
        train_fold0()
        return

    if args.mode == "factor-meaning-boundary-v54":
        from trainer.factor_meaning_boundary_v54 import main as factor_meaning_v54
        factor_meaning_v54()
        return

    if args.mode == "factor-context-gate-v64":
        from trainer.factor_context_gate_v64 import main as factor_context_v64
        factor_context_v64()
        return

    if args.mode == "factor-dedup-occurrence-v65":
        from trainer.factor_dedup_occurrence_v65 import main as factor_v65
        factor_v65()
        return

    if args.mode == "case-syntax-v66":
        from trainer.case_syntax_v66 import main as case_syntax_v66
        case_syntax_v66()
        return

    if args.mode == "semi-supervised-v67":
        from trainer.semi_supervised_v67 import main as semi_supervised_v67
        semi_supervised_v67()
        return

    if args.mode == "syntax-aux-v68":
        from trainer.syntax_aux_v68 import main as syntax_aux_v68
        syntax_aux_v68()
        return

    if args.mode == "factor-targeted-repair-v69":
        from trainer.factor_targeted_repair_v69 import main as factor_v69
        factor_v69()
        return

    if args.mode == "factor-aligned-decoder-v70":
        from trainer.factor_aligned_decoder_v70 import main as factor_v70
        factor_v70()
        return

    if args.mode == "factor-paper-boundary-cross-v44-fold0":
        from trainer.factor_paper_boundary_cross_v44 import main as factor_boundary_v44
        factor_boundary_v44()
        return

    if args.mode == "task1-cv-fold0":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 CV training requires CUDA.")
        from trainer.task1_cv import train_task1_cv
        train_task1_cv(only_fold0=True)
        return

    if args.mode == "task1-cv":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 CV training requires CUDA.")
        from trainer.task1_cv import train_task1_cv
        train_task1_cv(only_fold0=False)
        return

    if args.mode == "task1-mental-strict":
        if not torch.cuda.is_available():
            raise RuntimeError("MentalRoBERTa Task 1 training requires CUDA.")
        from trainer.train_task1_mental import train_task1_mental_strict
        train_task1_mental_strict()
        return

    if args.mode == "task1-mental-full":
        if not torch.cuda.is_available():
            raise RuntimeError("MentalRoBERTa Task 1 training requires CUDA.")
        from trainer.train_task1_mental import train_task1_mental_full
        train_task1_mental_full()
        return

    if args.mode == "task1-risk-v5":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 risk-v5 calibration requires CUDA.")
        from analyze_task1_risk_v5 import main as analyze_task1_risk_v5
        analyze_task1_risk_v5()
        return

    if args.mode == "task1-clinical-v6":
        from analyze_task1_clinical_v6 import main as analyze_task1_clinical_v6
        analyze_task1_clinical_v6()
        return

    if args.mode == "task1-evidence-v7":
        from analyze_task1_evidence_v7 import main as analyze_task1_evidence_v7
        analyze_task1_evidence_v7()
        return

    if args.mode == "task1-evidence-v8":
        from analyze_task1_evidence_v8 import main as analyze_task1_evidence_v8
        analyze_task1_evidence_v8()
        return

    if args.mode == "task1-evidence-refine-v9":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 evidence refinement requires CUDA.")
        from trainer.task1_evidence_refine import train_task1_evidence_refine
        train_task1_evidence_refine()
        return

    if args.mode == "task1-risk-v10":
        from analyze_task1_risk_v10 import main as analyze_task1_risk_v10
        analyze_task1_risk_v10()
        return

    if args.mode == "task1-lexical-v11":
        from analyze_task1_lexical_v11 import main as analyze_task1_lexical_v11
        analyze_task1_lexical_v11()
        return

    if args.mode == "task1-user-context-v12":
        from analyze_task1_user_context_v12 import main as analyze_task1_user_context_v12
        analyze_task1_user_context_v12()
        return

    if args.mode == "task1-evidence-reranker-v13":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 evidence reranker requires CUDA.")
        from trainer.task1_evidence_reranker_v13 import train_task1_evidence_reranker_v13
        train_task1_evidence_reranker_v13()
        return

    if args.mode == "task1-evidence-reranker-v13-hybrid":
        from analyze_task1_reranker_v13_hybrid import main as analyze_v13_hybrid
        analyze_v13_hybrid()
        return

    if args.mode == "task1-seed-ensemble-v14":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 seed ensemble requires CUDA.")
        from trainer.task1_seed_ensemble_v14 import train_task1_seed_ensemble_v14
        train_task1_seed_ensemble_v14()
        return

    if args.mode == "task1-seed-ensemble-v14-weights":
        from analyze_task1_seed_ensemble_v14_weights import main as analyze_v14_weights
        analyze_v14_weights()
        return

    if args.mode == "task1-nli-v15":
        from analyze_task1_nli_v15 import main as analyze_task1_nli_v15
        analyze_task1_nli_v15()
        return

    if args.mode == "task1-oof-reranker-v16":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 OOF evidence reranker requires CUDA.")
        from trainer.task1_oof_reranker_v16 import train_task1_oof_reranker_v16
        train_task1_oof_reranker_v16()
        return

    if args.mode == "task1-dynamic-topk-v17":
        from analyze_task1_dynamic_topk_v17 import main as analyze_task1_dynamic_topk_v17
        analyze_task1_dynamic_topk_v17()
        return

    if args.mode == "task1-candidate-v18":
        from analyze_task1_candidate_v18 import main as analyze_task1_candidate_v18
        analyze_task1_candidate_v18()
        return

    if args.mode == "task1-v18-full":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 V18 full-data training requires CUDA.")
        from trainer.task1_v18_full import train_task1_v18_full
        train_task1_v18_full()
        return

    if args.mode == "task1-llm-v19":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 constrained-LLM V19 requires CUDA.")
        from analyze_task1_llm_v19 import main as analyze_task1_llm_v19
        analyze_task1_llm_v19()
        return

    if args.mode == "task1-oof-stack-v20":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 nested OOF V20 requires CUDA.")
        from trainer.task1_oof_stack_v20 import train_task1_oof_stack_v20
        train_task1_oof_stack_v20()
        return

    if args.mode == "task1-lexical-reranker-v21":
        from analyze_task1_lexical_reranker_v21 import main as analyze_task1_lexical_reranker_v21
        analyze_task1_lexical_reranker_v21()
        return

    if args.mode == "task1-nested-ensemble-v22":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 nested ensemble V22 requires CUDA.")
        from analyze_task1_nested_ensemble_v22 import main as analyze_task1_nested_ensemble_v22
        analyze_task1_nested_ensemble_v22()
        return

    if args.mode == "task1-position-reranker-v23":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 position reranker V23 requires CUDA.")
        from analyze_task1_position_reranker_v23 import main as analyze_task1_position_reranker_v23
        analyze_task1_position_reranker_v23()
        return

    if args.mode == "task1-reranker-ensemble-v24":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 reranker ensemble V24 requires CUDA.")
        from analyze_task1_reranker_ensemble_v24 import main as analyze_task1_reranker_ensemble_v24
        analyze_task1_reranker_ensemble_v24()
        return

    if args.mode == "task1-atomic-v25":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 atomic sentence/token V25 requires CUDA.")
        from trainer.task1_atomic_v25 import train_task1_atomic_v25
        train_task1_atomic_v25()
        return

    if args.mode == "task1-atomic-refine-v26":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 atomic boundary V26 requires CUDA.")
        from analyze_task1_atomic_refine_v26 import main as analyze_task1_atomic_refine_v26
        analyze_task1_atomic_refine_v26()
        return

    if args.mode == "task1-risk-only-v27":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 dedicated risk V27 requires CUDA.")
        from trainer.task1_risk_only_v27 import train_task1_risk_only_v27
        train_task1_risk_only_v27()
        return

    if args.mode == "task1-seed-evidence-v28":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 multi-seed evidence V28 requires CUDA.")
        from trainer.task1_seed_evidence_v28 import train_task1_seed_evidence_v28
        train_task1_seed_evidence_v28()
        return

    if args.mode == "task1-boundary-crossval-v29":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 boundary cross-validation V29 requires CUDA.")
        from trainer.task1_boundary_crossval_v29 import main as task1_boundary_crossval_v29
        task1_boundary_crossval_v29()
        return

    if args.mode == "task1-boundary-augment-v30":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 boundary augmentation V30 requires CUDA.")
        from analyze_task1_boundary_augment_v30 import main as task1_boundary_augment_v30
        task1_boundary_augment_v30()
        return

    if args.mode == "task1-evidence-lexicon-v31":
        from analyze_task1_evidence_lexicon_v31 import main as task1_evidence_lexicon_v31
        task1_evidence_lexicon_v31()
        return

    if args.mode == "task1-clinical-reranker-v32":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 clinical reranker V32 requires CUDA.")
        from trainer.task1_clinical_reranker_v32 import main as task1_clinical_reranker_v32
        task1_clinical_reranker_v32()
        return

    if args.mode == "task1-boundary-clinical-v33":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 boundary/clinical V33 requires CUDA.")
        from analyze_task1_boundary_clinical_v33 import main as task1_boundary_clinical_v33
        task1_boundary_clinical_v33()
        return

    if args.mode == "task1-temporal-v34":
        from analyze_task1_temporal_v34 import main as task1_temporal_v34
        task1_temporal_v34()
        return

    if args.mode == "task1-oof-decoder-v35":
        from analyze_task1_oof_decoder_v35 import main as task1_oof_decoder_v35
        task1_oof_decoder_v35()
        return

    if args.mode == "task1-oof-risk-v36":
        from analyze_task1_oof_risk_v36 import main as task1_oof_risk_v36
        task1_oof_risk_v36()
        return

    if args.mode == "task1-oof-meta-v37":
        from trainer.task1_oof_meta_v37 import main as task1_oof_meta_v37
        task1_oof_meta_v37()
        return

    if args.mode == "task1-ordinal-lexical-v38":
        from analyze_task1_ordinal_lexical_v38 import main as task1_ordinal_lexical_v38
        task1_ordinal_lexical_v38()
        return

    if args.mode == "task1-boundary-model-v39":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 boundary model V39 requires CUDA.")
        from trainer.task1_boundary_model_v39 import main as task1_boundary_model_v39
        task1_boundary_model_v39()
        return

    if args.mode == "task1-boundary-components-v40":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 boundary component V40 requires CUDA.")
        from analyze_task1_boundary_components_v40 import main as task1_boundary_components_v40
        task1_boundary_components_v40()
        return

    if args.mode == "task1-frozen-rehead-v41":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 frozen re-head V41 requires CUDA.")
        from trainer.task1_frozen_rehead_v41 import main as task1_frozen_rehead_v41
        task1_frozen_rehead_v41()
        return

    if args.mode == "task1-evidence-conditioned-v42":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 evidence-conditioned V42 requires CUDA.")
        from trainer.task1_evidence_conditioned_v42 import main as task1_evidence_conditioned_v42
        task1_evidence_conditioned_v42()
        return

    if args.mode == "task1-selective-gate-v43":
        from analyze_task1_selective_gate_v43 import main as task1_selective_gate_v43
        task1_selective_gate_v43()
        return

    if args.mode == "task1-style-personality-v44":
        from analyze_task1_style_personality_v44 import main as task1_style_personality_v44
        task1_style_personality_v44()
        return

    if args.mode == "task1-pseudo-evidence-v45":
        from analyze_task1_pseudo_evidence_v45 import main as task1_pseudo_evidence_v45
        task1_pseudo_evidence_v45()
        return

    if args.mode == "task1-large-v46":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 large V46 requires CUDA.")
        from trainer.task1_large_v46 import main as task1_large_v46
        task1_large_v46()
        return

    if args.mode == "task1-qwen-lora-v47":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 Qwen QLoRA V47 requires CUDA.")
        from trainer.task1_qwen_lora_v47 import main as task1_qwen_lora_v47
        task1_qwen_lora_v47()
        return

    if args.mode == "task1-qwen-oof-v48":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 Qwen OOF V48 requires CUDA.")
        from trainer.task1_qwen_oof_v48 import main as task1_qwen_oof_v48
        task1_qwen_oof_v48()
        return

    if args.mode == "task1-qwen-verbalizer-v49":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 Qwen verbalizer V49 requires CUDA.")
        from trainer.task1_qwen_verbalizer_v49 import main as task1_qwen_verbalizer_v49
        task1_qwen_verbalizer_v49()
        return

    if args.mode == "task1-alignment-v50":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 evidence-alignment V50 requires CUDA.")
        from trainer.task1_alignment_v50 import main as task1_alignment_v50
        task1_alignment_v50()
        return

    if args.mode == "task1-repaired-multiseed-v51":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 repaired multi-seed V51 requires CUDA.")
        from trainer.task1_repaired_multiseed_v51 import main as task1_repaired_multiseed_v51
        task1_repaired_multiseed_v51()
        return

    if args.mode == "task1-rationale-augment-v52":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 rationale augmentation V52 requires CUDA.")
        from trainer.task1_rationale_augment_v52 import main as task1_rationale_augment_v52
        task1_rationale_augment_v52()
        return

    if args.mode == "task1-rationale-augment-v52-full":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 rationale augmentation V52 full training requires CUDA.")
        from trainer.task1_rationale_augment_v52 import train_full as task1_rationale_v52_full
        task1_rationale_v52_full()
        return

    if args.mode == "task1-qwen7b-verbalizer-v53":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 Qwen2.5-7B V53 requires CUDA.")
        from trainer.task1_qwen7b_verbalizer_v53 import main as task1_qwen7b_v53
        task1_qwen7b_v53()
        return

    if args.mode == "task1-factor-bridge-v54":
        from trainer.task1_factor_bridge_v54 import main as task1_factor_bridge_v54
        task1_factor_bridge_v54()
        return

    if args.mode == "task1-evidence-count-prior-v55":
        from trainer.task1_evidence_count_prior_v55 import main as task1_evidence_v55
        task1_evidence_v55()
        return

    if args.mode == "task1-local-cf-v56-generate":
        from trainer.task1_local_counterfactual_v56 import generate_counterfactuals
        generate_counterfactuals()
        return

    if args.mode == "task1-local-cf-v56-train":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 local counterfactual V56 training requires CUDA.")
        from trainer.task1_local_counterfactual_train_v56 import main as task1_local_cf_v56
        task1_local_cf_v56()
        return

    if args.mode == "task1-local-diverse-cf-v57-generate":
        from trainer.task1_local_diverse_cf_v57 import generate
        generate()
        return

    if args.mode == "task1-local-diverse-cf-v57-train":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 diverse counterfactual V57 training requires CUDA.")
        from trainer.task1_local_diverse_cf_train_v57 import main as task1_local_cf_v57
        task1_local_cf_v57()
        return

    if args.mode == "task1-local-diverse-cf-v57-full":
        if not torch.cuda.is_available():
            raise RuntimeError("Task 1 diverse counterfactual V57 full training requires CUDA.")
        from trainer.task1_local_diverse_cf_train_v57 import train_full
        train_full()
        return

    if args.mode == "task1-factor-trajectory-v58":
        from trainer.task1_factor_trajectory_v58 import main as task1_factor_trajectory_v58
        task1_factor_trajectory_v58()
        return

    if args.mode == "task1-factor-trajectory-gate-v59":
        from analyze_task1_factor_trajectory_gate_v59 import main as task1_factor_gate_v59
        task1_factor_gate_v59()
        return

    if args.mode == "task1-factor-trajectory-v58-full":
        from trainer.task1_factor_trajectory_v58 import train_full as task1_factor_v58_full
        task1_factor_v58_full()
        return

    if args.mode == "task1-dynamic-influence-v60":
        from trainer.task1_dynamic_influence_v60 import main as task1_dynamic_v60
        task1_dynamic_v60()
        return

    if args.mode == "task1-windowed-trajectory-v61":
        from trainer.task1_windowed_trajectory_v61 import main as task1_windowed_v61
        task1_windowed_v61()
        return

    if args.mode == "task1-same-post-factor-v62":
        from trainer.task1_same_post_factor_v62 import main as task1_factor_v62
        task1_factor_v62()
        return

    if args.mode == "task1-polarity-v63":
        from trainer.task1_polarity_v63 import main as task1_polarity_v63
        task1_polarity_v63()
        return

    if args.build_cache:
        from datasets.cache_builder import build_cache
        build_cache(True)
        build_cache(False)
        return

    # DeBERTa backpropagation is impractically slow in a CPU-only PyTorch
    # installation.  Keep the command interface unchanged and use the fast,
    # complete baseline until a CUDA-enabled PyTorch build is installed.
    if not torch.cuda.is_available():
        from baseline import train as cpu_train, predict as cpu_predict
        if args.mode in ("full-run", "train-full", "factor-full"):
            cpu_train()
            if args.mode == "full-run":
                cpu_predict()
        elif args.mode in ("train", "train-strict"):
            raise RuntimeError(
                "Strict DeBERTa training requires CUDA. Use --mode private-group-cv "
                "for the CPU strict evaluation."
            )
        elif args.mode in ("predict", "predict-v38"):
            cpu_predict()
        else:
            raise RuntimeError("Ensembling is available only for CUDA Transformer checkpoints.")
        return





    # -------------------------
    # seed
    # -------------------------


    seed_everything(
        42
    )





    # -------------------------
    # Train
    # -------------------------


    if args.mode in ("train", "train-strict"):



        from trainer.train import train



        train()

    # -------------------------
    # Final full-data training
    # -------------------------

    elif args.mode in ("train-full", "full-run"):

        from trainer.train import train_full

        checkpoint = train_full()

        if args.mode == "full-run":

            from trainer.factor_train import train_factor_full

            train_factor_full()

            from inference.predict import predict

            predict(checkpoint)

    elif args.mode == "factor-full":

        from trainer.factor_train import train_factor_full

        train_factor_full()





    # -------------------------
    # Predict
    # -------------------------


    elif args.mode in ("predict", "predict-v38", "predict-v69", "predict-v70",
                       "predict-qwen3-8b", "predict-qwen3-8b-blend",
                       "predict-qwen3-8b-attempt-calibrated"):



        from inference.predict import predict



        experiment = {"predict-v38": "v38", "predict-v69": "v69",
                      "predict-v70": "v70"}.get(args.mode)
        task1_experiment = {
            "predict-qwen3-8b": "qwen3-8b",
            "predict-qwen3-8b-blend": "qwen3-8b-blend",
            "predict-qwen3-8b-attempt-calibrated": "qwen3-8b-attempt-calibrated",
        }.get(args.mode)
        predict(task2_experiment=experiment, task1_experiment=task1_experiment)





    # -------------------------
    # Ensemble
    # -------------------------


    elif args.mode=="ensemble":



        from inference.ensemble import (

            ensemble_predict

        )



        checkpoints=[


            "outputs/fold0/best_model.pt",


            "outputs/fold1/best_model.pt",


            "outputs/fold2/best_model.pt",


            "outputs/fold3/best_model.pt",


            "outputs/fold4/best_model.pt"

        ]



        ensemble_predict(

            checkpoints

        )






if __name__=="__main__":


    main()

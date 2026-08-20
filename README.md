# Mental Health Support in Crisis Chats Using LLMs and ML

This repository contains the source code used for an explainable suicide-risk
detection dissertation and the IEEE Big Data Cup task. It predicts four suicide
risk levels, extracts supporting text spans, and assigns 24 risk and protective
factor labels.

The repository contains code only. The UMD Reddit training data, leaderboard
data, model checkpoints, cached probabilities, and submission files are not
distributed.

## Reported system

The final system combines several complementary components:

- DeBERTa-based risk classification with ordinal severity modelling
- token- and boundary-based evidence extraction
- MentalRoBERTa with label-specific attention for factor prediction
- asymmetric loss and label-wise calibration for long-tailed factors
- sparse and text-label cross-encoder experts
- Qwen3-8B QLoRA risk classification on Kaggle
- Qwen3-Reranker-8B pairwise risk/protective factor scoring on Kaggle

The final hidden-test scores reported in the dissertation are 0.7829 for Task 1,
0.6291 for Task 2, and 0.7367 for the weighted composite score.

## Repository structure

| Path | Purpose |
| --- | --- |
| `main.py` | Main local entry point for training, evaluation and prediction |
| `configs/` | Project settings, labels and model configuration |
| `preprocess/` | Input cleaning, evidence spans, style features and factor definitions |
| `datasets/` | PyTorch datasets, collators and cache builders |
| `models/` | DeBERTa, MentalRoBERTa and multi-task model components |
| `trainer/` | Training, cross-validation, calibration and ablation experiments |
| `inference/` | Ensemble prediction and evidence/factor decoding |
| `utils/` | Metrics, grouped splits, thresholds and reproducibility utilities |
| `baselines/` | Classic BERT multi-task baseline used in the dissertation |
| `kaggle_llm_package/` | Qwen/Gemma/Llama Task 1 QLoRA experiment |
| `kaggle_factor_qwen_package/` | Qwen3-8B Task 2 multi-label experiment |
| `kaggle_factor_reranker_package/` | Final Qwen3-Reranker-8B Task 2 experiment |
| `thesis_tools/` | Scripts used to build and check dissertation documents |
| `docs/` | Appendix-ready code availability and reproducibility text |

The many versioned trainer and analysis files are retained because they record
the development path, including rejected experiments. Production settings are
defined in `configs/config.py`, while `main.py` lists the available modes.

## Local setup

Python 3.11 was used for local experiments.

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place authorised copies of the competition files at:

```text
data/train.xlsx
data/leaderboard.xlsx
```

The required schemas are described in `data/README.md`.

Example commands:

```powershell
python .\main.py --mode private-group-cv
python .\main.py --mode train-full
python .\main.py --mode predict
```

Some final prediction modes require checkpoints and cached OOF probabilities
produced by earlier stages. These artefacts are intentionally excluded from
GitHub. The program reports a missing path when a required artefact has not yet
been generated.

## Kaggle Qwen experiments

The three Kaggle folders are self-contained code packages. Upload the selected
folder or a ZIP of it as one Kaggle Dataset, then attach the competition data as
a separate **private** Kaggle Dataset. The scripts search `/kaggle/input`
recursively for `train.xlsx`, `leaderboard.xlsx`, and any OOF artefacts required
by the selected stage.

Recommended final experiments:

```bash
python -u kaggle_llm_experiment.py --stage fold --model qwen3-8b --fold 0 --epochs 2
python -u kaggle_factor_reranker_v3.py --stage preflight --model qwen3-reranker-8b
python -u kaggle_factor_reranker_v3.py --stage fold --model qwen3-reranker-8b --fold 0 --epochs 2
```

See the README inside each Kaggle folder for the complete OOF and full-data
workflow. Hugging Face tokens must be stored in Kaggle Secrets and must never be
written into notebooks or committed to this repository.

## Data and safety statement

This project is a research prototype. Its labels are benchmark annotations, not
clinical diagnoses. The output must not be used as the sole basis for emergency,
medical, policing, employment, or access decisions. See `data/README.md` for the
data access boundary.

## Reproducibility note

User-disjoint validation is required because one user can contribute several
posts. Random post-level splits may leak writing style or repeated events. Task 1
is reported with weighted risk F1 and phrase F1. Task 2 is reported with Macro
F1 across the 24 labels.

No licence is granted by default. The competition dataset remains subject to
its original access and use conditions.


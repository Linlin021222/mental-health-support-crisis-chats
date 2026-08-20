# Task 2 V3: risk/protective Qwen3 reranker

This package changes Task 2 from a single 24-output causal classifier into 24
independent, constrained post-factor judgements.  For every factor, the model
reads a formal definition, positive indications, an exclusion boundary and the
Reddit post, then scores only `yes` versus `no` at the final decision token.

The first 19 competition labels are treated as **risk factors** and the final
five as **protective factors**.  The distinction is used in the prompt,
sampling, diagnostics and nested calibration.  The two groups are not mutually
exclusive: a post may contain both risk and protective factors.

The default backbone is `Qwen/Qwen3-Reranker-8B`, not the generic
`Qwen/Qwen3-8B` used by V2.

## 1. Kaggle setup

1. Upload `kaggle_factor_reranker_v3.zip` as a private Kaggle Dataset.
2. Create a Notebook and attach that Dataset.
3. Enable **GPU T4 x2** and **Internet**.
4. Run this setup cell:

```python
!pip install -q -U "transformers>=4.51" "peft>=0.15" accelerate bitsandbytes openpyxl scikit-learn

from pathlib import Path
import zipfile

scripts = list(Path("/kaggle/input").rglob("kaggle_factor_reranker_v3.py"))
if not scripts:
    archives = list(Path("/kaggle/input").rglob("kaggle_factor_reranker_v3.zip"))
    assert len(archives) == 1, archives
    extracted = Path("/kaggle/working/task2_reranker_v3")
    with zipfile.ZipFile(archives[0]) as archive:
        archive.extractall(extracted)
    scripts = list(extracted.rglob("kaggle_factor_reranker_v3.py"))

assert len(scripts) == 1, scripts
script = scripts[0]
print(script)
```

Qwen3-Reranker is public. `HF_TOKEN` is normally unnecessary. If Hugging Face
rate-limits the download, add a Kaggle Secret called `HF_TOKEN` and export it
before running the script.

## 2. Preflight

```python
!python -u {script} --stage preflight --model qwen3-reranker-8b
```

Do not train unless the output contains:

- `preflight_passed: true`
- `targets_shape: [1635, 24]`
- `risk_labels: 19`
- `protective_labels: 5`
- zero user overlap in all five folds
- at least 384 post tokens in `minimum_post_budget`

## 3. Run Fold 0 first

The pairwise formulation performs 24 judgements per validation post, so
inference is substantially slower than Task 1. Keep the Notebook open until the
NPZ and JSON are written.

```python
!python -u {script} \
  --stage fold \
  --model qwen3-reranker-8b \
  --fold 0 \
  --epochs 2
```

Download immediately:

- `qwen3-reranker-8b-factor-v3_fold0_probabilities.npz`
- `qwen3-reranker-8b-factor-v3_fold0_results.json`

Send both files to Codex before running the remaining folds.  Fold 0 should be
judged by Macro F1, risk/protective Macro F1, per-label PR-AUC and complementarity
with the packaged MentalRoBERTa OOF baseline—not merely by training loss.

### Optional cheaper architecture check

If an 8B Fold 0 is too slow, the exact same code supports the 4B reranker:

```python
!python -u {script} --stage fold --model qwen3-reranker-4b --fold 0 --epochs 2
```

Do not mix 4B and 8B fold files in the same OOF summary.

## 4. Five-fold OOF evaluation

If Fold 0 is promising, run folds 1-4 one at a time:

```python
FOLD = 1  # change to 2, 3, and 4 afterwards
!python -u {script} \
  --stage fold \
  --model qwen3-reranker-8b \
  --fold {FOLD} \
  --epochs 2
```

Download each fold NPZ immediately. If a session resets, upload the existing
fold NPZ files to a private Kaggle Dataset and attach it; the script searches
`/kaggle/input` automatically.

After all five folds are available:

```python
!python -u {script} --stage summarize-oof --model qwen3-reranker-8b
```

Download:

- `qwen3-reranker-8b-factor-v3_oof_summary.json`
- `qwen3-reranker-8b-factor-v3_oof_probabilities.npz`

The summary reports the standalone reranker, current baseline and nested
label-wise candidate for overall, risk-factor and protective-factor Macro F1.
It also records which labels pass the conservative three-of-five-fold gate.

## 5. Full training and leaderboard probabilities

Only run full training after the five-fold summary shows a genuine nested
user-disjoint gain:

```python
!python -u {script} \
  --stage full \
  --model qwen3-reranker-8b \
  --epochs 2 \
  --save-adapter
```

Download:

- `qwen3-reranker-8b-factor-v3_test_probabilities.npz`
- `qwen3-reranker-8b-factor-v3_full_results.json`
- optional directory `qwen3-reranker-8b-factor-v3_full_adapter/`

The test NPZ is not a submission by itself. Send the OOF summary, OOF NPZ and
test NPZ to Codex. Only labels accepted by nested user-disjoint validation will
replace or blend with the established Task 2 system.

To archive the important full outputs:

```python
from pathlib import Path
import zipfile

work = Path("/kaggle/working")
archive_path = work / "qwen3-reranker-8b-factor-v3_full_prediction.zip"
with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for name in (
        "qwen3-reranker-8b-factor-v3_test_probabilities.npz",
        "qwen3-reranker-8b-factor-v3_full_results.json",
        "qwen3-reranker-8b-factor-v3_oof_probabilities.npz",
        "qwen3-reranker-8b-factor-v3_oof_summary.json",
    ):
        path = work / name
        if path.exists():
            archive.write(path, path.name)
    adapter = work / "qwen3-reranker-8b-factor-v3_full_adapter"
    if adapter.exists():
        for path in adapter.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(work))

print(archive_path, archive_path.stat().st_size / 1024**2, "MB")
```

## Default training design

- 1,920 post-factor examples per epoch
- matched positive/negative batches for an always-active ranking loss
- label-balanced risk sampling plus modest protective oversampling (30%)
- 70% preference for confusion-label hard negatives
- repeated factor annotations used only as a weak positive-sampling signal
- binary multi-label targets and outputs remain unchanged
- full formal definitions and negative annotation boundaries
- 1,280-token input with head, cue-context and tail preservation
- user-disjoint five-fold evaluation


# Qwen3-8B Task 2 Hybrid V2 Kaggle experiment

This package trains a separate Qwen3-8B expert for the 24 factor labels. V2
uses a contextual-definition global classifier as its main head and a gated
label-specific-attention residual, occurrence-aware ASL, and cross-post
per-label ranking. It does **not** overwrite the accepted MentalRoBERTa Task 2
system. Five user-disjoint OOF folds must first show label-level Macro-F1 gain.

## 1. Kaggle setup

1. Upload `kaggle_factor_qwen_v2.zip` as a private Kaggle Dataset and extract
   it when creating the Dataset.
2. Create a Kaggle Notebook from that Dataset.
3. Enable **GPU T4 x2** and Internet.
4. Run:

```python
!pip install -q -U "transformers>=4.51" "peft>=0.15" accelerate bitsandbytes openpyxl scikit-learn

from pathlib import Path
import zipfile
scripts = list(Path("/kaggle/input").rglob("kaggle_factor_qwen.py"))
if not scripts:
    archives = list(Path("/kaggle/input").rglob("kaggle_factor_qwen_v2.zip"))
    assert len(archives) == 1, archives
    extracted = Path("/kaggle/working/task2_qwen_package")
    with zipfile.ZipFile(archives[0]) as z:
        z.extractall(extracted)
    scripts = list(extracted.rglob("kaggle_factor_qwen.py"))
assert len(scripts) == 1, scripts
script = scripts[0]
print(script)
```

Run the fast package/data check before training:

```python
!python -u {script} --stage preflight
```

It must report `preflight_passed: true`, target shape `[1635, 24]`, and zero
user overlap in every fold.

Qwen3-8B is public; `HF_TOKEN` is normally unnecessary. If Hugging Face rate
limits the download, add a Kaggle Secret named `HF_TOKEN` and export it before
running.

## 2. Run fold 0 first

```python
FOLD = 0
!python -u {script} --stage fold --fold {FOLD} --epochs 2
```

Expected files in `/kaggle/working`:

- `qwen3-8b-factor-v2_fold0_probabilities.npz`
- `qwen3-8b-factor-v2_fold0_results.json`

Download both. Send the JSON result to Codex before spending time on the other
folds. Fold adapters are intentionally not saved because OOF probabilities are
all that calibration needs.

The real V1 Fold-0 score was `0.3007`. Do not run full training merely because
V2 finishes without an error. First inspect Macro F1 and the per-label AUC/PR
AUC. A useful V2 should either improve standalone Fold-0 substantially or show
clear complementary ranking on several labels; otherwise it is rejected.

## 3. Run folds 1-4

Change `FOLD` and run the same cell once for each fold:

```python
FOLD = 1  # then 2, 3, 4
!python -u {script} --stage fold --fold {FOLD} --epochs 2
```

Download every fold NPZ immediately. If the Kaggle session is reset, create a
small private Kaggle Dataset containing the five fold NPZ files and attach it
to the notebook. The script searches `/kaggle/input` automatically.

When all five files are present or attached:

```python
!python -u {script} --stage summarize-oof
```

Download:

- `qwen3-8b-factor-v2_oof_summary.json`
- `qwen3-8b-factor-v2_oof_probabilities.npz`

The JSON reports the current Task 2 baseline, standalone Qwen Macro-F1,
nested label-wise fusion Macro-F1, and exactly which labels passed the gate.

## 4. Full-data training and leaderboard probabilities

Only do this after the OOF result is promising:

```python
!python -u {script} --stage full --epochs 2 --save-adapter
```

Download these two files first:

- `qwen3-8b-factor-v2_test_probabilities.npz`
- `qwen3-8b-factor-v2_full_results.json`

The optional reusable model is the directory:

- `qwen3-8b-factor-v2_full_adapter/`

The base 8B model does not need to be saved. Future inference reloads
`Qwen/Qwen3-8B`, then loads this LoRA adapter and `factor_head.pt`.

Create one download archive:

```python
from pathlib import Path
import zipfile

work = Path("/kaggle/working")
archive = work / "qwen3-8b-factor-v2_full_prediction.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for name in (
        "qwen3-8b-factor-v2_test_probabilities.npz",
        "qwen3-8b-factor-v2_full_results.json",
        "qwen3-8b-factor-v2_oof_summary.json",
        "qwen3-8b-factor-v2_oof_probabilities.npz",
    ):
        path = work / name
        if path.exists():
            z.write(path, path.name)
    adapter = work / "qwen3-8b-factor-v2_full_adapter"
    if adapter.exists():
        for path in adapter.rglob("*"):
            if path.is_file():
                z.write(path, path.relative_to(work))
print(archive, archive.stat().st_size / 1024**2, "MB")
```

Do not submit raw Qwen factor predictions directly. Send the OOF summary and
test probabilities to Codex. Only labels that improve nested user-disjoint F1
will be fused into the official `panda.csv`; all other factor columns remain
bit-for-bit identical to the accepted Task 2 model.

# Kaggle 强 LLM 实验包

本包只测试 Task 1 风险分类。现有最强 evidence 和 Task 2 factors 保留在
`baseline_panda.csv` 中，不会被新模型覆盖。

## 推荐顺序

1. `qwen3-8b`：无需 Hugging Face token，先运行 fold 0。
2. `gemma2-9b`：论文中表现最好，需要先接受 Gemma 协议并添加 `HF_TOKEN`。
3. `llama31-8b`：异构集成候选，需要接受 Meta Llama 协议及 `HF_TOKEN`。
4. `qwen25-14b`：无需 token，但更慢；建议前面模型完成后再测试。

不要在普通 Kaggle T4 上运行 72B。论文作者报告 Qwen2-72B 推理约需
160GB VRAM。

## Kaggle 设置

1. 将整个压缩包作为 **Private Dataset** 上传。训练数据不得设为 Public。
2. 新建 Notebook，Add Input 选择该 Private Dataset。ZIP 可以原样上传，下面的
   定位代码会在 Kaggle 未自动展开时将它解压到 `/kaggle/working`。
3. Notebook Settings：Accelerator 选 `GPU T4 x2`（没有时选单 T4/P100），
   Internet 设为 On。
4. 第一格运行：

```python
!pip install -q "transformers>=4.51,<6" "accelerate>=1.4" \
  "bitsandbytes>=0.45" "peft>=0.15" openpyxl scikit-learn tqdm
```

5. 第二格自动定位脚本：

```python
from pathlib import Path
import zipfile
scripts = list(Path('/kaggle/input').rglob('kaggle_llm_experiment.py'))
if scripts:
    script = scripts[0]
else:
    archive = next(Path('/kaggle/input').rglob('kaggle_strong_llm_v*.zip'))
    package = Path('/kaggle/working/strong_llm_package')
    package.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(archive).extractall(package)
    script = package / 'kaggle_llm_experiment.py'
print(script)
```

6. 首先只做一个严格用户隔离 fold：

```python
!python -u {script} --stage fold --model qwen3-8b --fold 0 --epochs 2
```

完成后下载 `/kaggle/working/qwen3-8b_fold0_results.json` 并发给 Codex。
不要看到一个 fold 提高就立即提交 leaderboard。

## 完整四折验证

模型通过 fold 0 筛选后，分别运行 fold 1、2、3：

```python
!python -u {script} --stage fold --model qwen3-8b --fold 1 --epochs 2
!python -u {script} --stage fold --model qwen3-8b --fold 2 --epochs 2
!python -u {script} --stage fold --model qwen3-8b --fold 3 --epochs 2
!python -u {script} --stage summarize-oof --model qwen3-8b
```

如果分多次 Kaggle Session 运行，请在每次结束前下载对应
`*_probabilities.npz`；下次将这些文件作为另一个 Private Dataset 添加，脚本会自动寻找。

## 全量训练和 leaderboard 文件

只有完整 OOF 结果通过后才运行：

```python
!python -u {script} --stage full --model qwen3-8b --epochs 2
```

会在 `/kaggle/working` 生成：

- `panda_qwen3-8b_standalone.csv`
- `panda_qwen3-8b_conf70.csv`
- `panda_qwen3-8b_conf80.csv`
- `qwen3-8b_test_probabilities.npz`

先把 OOF summary 和 test probabilities 发给 Codex，由本地严格结果决定最终融合，
不要一次把三个 CSV 都提交到官方。

## Gemma/Llama 的 HF_TOKEN

先在对应 Hugging Face 模型页面接受协议。然后在 Kaggle Notebook 的
`Add-ons -> Secrets` 中新增名为 `HF_TOKEN` 的 secret，并在第一格执行：

```python
from kaggle_secrets import UserSecretsClient
import os
os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
```

随后把 `--model qwen3-8b` 换成 `--model gemma2-9b` 或
`--model llama31-8b`。

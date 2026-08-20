"""Full-data probabilities for the accepted V52 risk-only expert."""
from __future__ import annotations

import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.config import config
from datasets.collator import SuicideRiskCollator
from datasets.dataset import SuicideRiskDataset
from models.multitask_model import SuicideRiskMultiTaskModel


OUTPUT = config.OUTPUT_DIR / "task1_rationale_augment_v52"
CHECKPOINT = OUTPUT / "full_model.pt"
RESULTS = OUTPUT / "results.json"


@torch.no_grad()
def task1_v52_probabilities():
    if not CHECKPOINT.exists() or not RESULTS.exists():
        return None, None, 0.0
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    if not result.get("risk_only_candidate", {}).get("adopted", False):
        return None, None, 0.0
    weight = float(result["augmentation"]["risk_weight"])
    dataset = SuicideRiskDataset(config.CACHE_DIR / "test_cache.pt")
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                        collate_fn=SuicideRiskCollator(), num_workers=config.NUM_WORKERS)
    device = torch.device(config.DEVICE)
    model = SuicideRiskMultiTaskModel().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device)); model.eval()
    probabilities = []
    for batch in loader:
        output = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        probabilities.append(torch.softmax(output["risk_logits"], -1).cpu().numpy())
    row_ids = [row["row_id"] for row in dataset.data]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row_ids, np.vstack(probabilities), weight


__all__ = ["task1_v52_probabilities"]

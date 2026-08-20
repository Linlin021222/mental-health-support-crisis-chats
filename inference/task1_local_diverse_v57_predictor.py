"""Full-data Task 1 outputs for the strict-accepted V57 expert."""
from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.config import config
from datasets.collator import SuicideRiskCollator
from datasets.dataset import SuicideRiskDataset
from models.multitask_model import SuicideRiskMultiTaskModel


OUTPUT = config.OUTPUT_DIR / "task1_local_diverse_cf_v57"
CHECKPOINT = OUTPUT / "full_model.pt"
RESULTS = OUTPUT / "strict_results.json"


@torch.no_grad()
def task1_v57_outputs():
    if not CHECKPOINT.exists() or not RESULTS.exists():
        return None, 0., 0.
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    if (not result.get("adopted", False)
            or result.get("selected_branch") != "fixed_10pct_evidence"):
        return None, 0., 0.
    dataset = SuicideRiskDataset(config.CACHE_DIR / "test_cache.pt")
    loader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=SuicideRiskCollator(), num_workers=config.NUM_WORKERS,
    )
    device = torch.device(config.DEVICE)
    model = SuicideRiskMultiTaskModel().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device)); model.eval()
    rows = []
    for batch in loader:
        output = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        probability = torch.softmax(output["risk_logits"], -1).cpu().numpy()
        for index, row_id in enumerate(batch["row_id"]):
            rows.append({
                "row_id": str(row_id), "probability": probability[index],
                "start": output["start_logits"][index].float().cpu(),
                "end": output["end_logits"][index].float().cpu(),
            })
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, float(result["augmentation"]["fixed_risk_weight"]), .10


__all__ = ["task1_v57_outputs"]

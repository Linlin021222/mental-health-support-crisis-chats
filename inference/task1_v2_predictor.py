"""Risk probabilities from the validated ordinal Task 1 V2 model."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.config import config
from datasets.collator import SuicideRiskCollator
from datasets.dataset import SuicideRiskDataset
from models.multitask_model_v2 import SuicideRiskMultiTaskModelV2, ordinal_class_probabilities


FULL_CHECKPOINT = config.OUTPUT_DIR / "task1_v2_full_model.pt"


@torch.no_grad()
def task1_v2_probabilities():
    if not FULL_CHECKPOINT.exists():
        return None, None
    dataset = SuicideRiskDataset(config.CACHE_DIR / "test_cache.pt")
    loader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=SuicideRiskCollator(), num_workers=config.NUM_WORKERS,
    )
    device = torch.device(config.DEVICE)
    model = SuicideRiskMultiTaskModelV2().to(device)
    model.load_state_dict(torch.load(FULL_CHECKPOINT, map_location=device)); model.eval()
    result = []
    for batch in loader:
        output = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        standard = torch.softmax(output["risk_logits"], -1)
        ordinal = ordinal_class_probabilities(output["ordinal_logits"])
        weight = float(config.TASK1_V2_ORDINAL_WEIGHT)
        result.append(((1.0-weight)*standard + weight*ordinal).cpu().numpy())
    row_ids = [x["row_id"] for x in dataset.data]
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return row_ids, np.vstack(result)

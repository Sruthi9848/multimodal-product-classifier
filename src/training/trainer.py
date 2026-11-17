"""
Training loop for the LoRA multimodal classifier.

One epoch = for every batch:
    1. forward pass        logits = model(batch)
    2. loss                L = CrossEntropy(logits, labels)   (with label smoothing)
    3. backward pass       ∂L/∂θ for the trainable θ only (LoRA + fusion + head)
    4. gradient clipping   keeps updates stable
    5. optimizer step      AdamW
    6. scheduler step      linear warm-up then linear decay of the learning rate

After every epoch we evaluate on the validation set and keep the checkpoint
with the best macro-F1 (early stopping if it stops improving).
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from src.config import Config
from src.evaluation.metrics import compute_metrics
from src.models.multimodal_model import MultimodalClassifier, save_checkpoint
from src.utils import get_logger, save_json

logger = get_logger(__name__)


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def model_inputs(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Keep only the keys the model's forward() accepts (drop 'labels')."""
    return {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask", "pixel_values")}


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device, desc: str = "eval"):
    """Run the model over a DataLoader. Returns (true labels, predicted labels, probabilities)."""
    model.eval()
    all_y, all_p = [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        batch = move_to_device(batch, device)
        logits = model(**model_inputs(batch))["logits"]
        all_p.append(torch.softmax(logits.float(), dim=-1).cpu())
        all_y.append(batch["labels"].cpu())
    probs = torch.cat(all_p).numpy()
    y_true = torch.cat(all_y).numpy()
    return y_true, probs.argmax(axis=1), probs


class Trainer:
    def __init__(self, model: MultimodalClassifier, cfg: Config, loaders: Dict[str, DataLoader],
                 label_map: Dict, device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.loaders = loaders
        self.label_map = label_map
        self.device = device
        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

        # Two parameter groups: LoRA adapters vs. the freshly initialised head.
        # The new head can use a larger learning rate than the adapters.
        lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
        head_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
        self.optimizer = torch.optim.AdamW(
            [{"params": lora_params, "lr": cfg.lr_lora},
             {"params": head_params, "lr": cfg.lr_head}],
            weight_decay=cfg.weight_decay,
        )
        total_steps = len(loaders["train"]) * cfg.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, int(cfg.warmup_ratio * total_steps), total_steps)

        # Mixed precision speeds up training ~2x on NVIDIA GPUs.
        self.use_amp = cfg.use_amp and device.type == "cuda"
        try:   # PyTorch >= 2.3
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except (AttributeError, TypeError):  # older PyTorch
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.float16) if self.use_amp else nullcontext()

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        losses, correct, seen = [], 0, 0
        bar = tqdm(self.loaders["train"], desc=f"epoch {epoch}/{self.cfg.epochs}")
        for batch in bar:
            batch = move_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with self._autocast():
                logits = self.model(**model_inputs(batch))["logits"]
                loss = self.criterion(logits, batch["labels"])

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)          # so clipping sees real gradient values
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            losses.append(loss.item())
            correct += (logits.argmax(-1) == batch["labels"]).sum().item()
            seen += batch["labels"].size(0)
            bar.set_postfix(loss=f"{np.mean(losses[-50:]):.4f}", acc=f"{correct / seen:.3f}")
        return {"train_loss": float(np.mean(losses)), "train_acc": correct / max(seen, 1)}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        losses, y_true, y_pred = [], [], []
        for batch in tqdm(self.loaders["val"], desc="val", leave=False):
            batch = move_to_device(batch, self.device)
            logits = self.model(**model_inputs(batch))["logits"].float()
            losses.append(self.criterion(logits, batch["labels"]).item())
            y_true.append(batch["labels"].cpu())
            y_pred.append(logits.argmax(-1).cpu())
        m = compute_metrics(torch.cat(y_true).numpy(), torch.cat(y_pred).numpy())
        return {"val_loss": float(np.mean(losses)), "val_acc": m["accuracy"], "val_f1_macro": m["f1_macro"]}

    def fit(self) -> List[Dict]:
        ckpt_path = self.cfg.checkpoint_file()
        best_f1, bad_epochs, history = -1.0, 0, []

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            stats = {"epoch": epoch, **self.train_one_epoch(epoch), **self.validate(),
                     "seconds": round(time.time() - t0, 1)}
            history.append(stats)
            logger.info(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in stats.items()))

            if stats["val_f1_macro"] > best_f1:
                best_f1, bad_epochs = stats["val_f1_macro"], 0
                save_checkpoint(self.model, self.cfg, self.label_map, ckpt_path,
                                extra={"best_epoch": epoch, "best_val_f1_macro": best_f1})
                logger.info(f"  ✓ new best model saved to {ckpt_path}")
            else:
                bad_epochs += 1
                if bad_epochs >= self.cfg.early_stopping_patience:
                    logger.info(f"Early stopping: no improvement for {bad_epochs} epochs.")
                    break

        save_json(history, self.cfg.metrics_dir / f"history_{self.cfg.run_tag}.json")
        return history

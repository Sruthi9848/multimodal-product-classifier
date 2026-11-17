"""Evaluate a saved LoRA checkpoint on the held-out TEST split and save all artefacts."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from transformers import AutoTokenizer

from src.data.dataset import build_dataloaders
from src.data.dataset_loader import get_split_datasets
from src.evaluation.metrics import full_evaluation, plot_confusion_matrix
from src.models.multimodal_model import load_model_from_checkpoint
from src.training.trainer import predict_loader
from src.utils import count_parameters, get_device, get_logger, save_json

logger = get_logger(__name__)


def describe_run(family: str, mode: str, masked: bool = False, seed: int = 42) -> str:
    """Human-readable model name, e.g. 'LoRA multimodal (masked, seed 7)'."""
    extras = (["masked"] if masked else []) + ([f"seed {seed}"] if seed != 42 else [])
    return f"{family} {mode}" + (f" ({', '.join(extras)})" if extras else "")


def save_predictions(cfg, name: str, y_true, y_pred, probs=None) -> Path:
    """Per-product predictions on the test set, needed for McNemar tests and bootstrap CIs."""
    path = cfg.predictions_dir / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, y_true=np.asarray(y_true), y_pred=np.asarray(y_pred),
                        **({"probs": np.asarray(probs, dtype=np.float32)} if probs is not None else {}))
    return path


def evaluate_lora_checkpoint(checkpoint_path: Path, num_workers: Optional[int] = None,
                             batch_size: Optional[int] = None) -> Dict:
    device = get_device()
    model, cfg, id2label, extra = load_model_from_checkpoint(Path(checkpoint_path), device)
    if num_workers is not None:
        cfg.num_workers = num_workers
    if batch_size is not None:
        cfg.batch_size = batch_size

    splits = get_split_datasets(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)
    test_loader = build_dataloaders(cfg, splits, tokenizer, names=("test",))["test"]

    t0 = time.time()
    y_true, y_pred, probs = predict_loader(model, test_loader, device, desc=f"test [{cfg.mode}]")
    elapsed = time.time() - t0

    class_names = [id2label[i] for i in sorted(id2label)]
    result = full_evaluation(y_true, y_pred, class_names, probs)
    params = count_parameters(model)
    tag = cfg.run_tag
    result.update({
        "model": describe_run("LoRA", cfg.mode, cfg.mask_keywords, cfg.seed),
        "run_name": tag,
        "seed": cfg.seed,
        "mask_keywords": cfg.mask_keywords,
        "family": "lora",
        "mode": cfg.mode,
        "fusion_type": cfg.fusion_type if cfg.mode == "multimodal" else None,
        "text_model": cfg.text_model_name if cfg.mode != "image" else None,
        "image_model": cfg.image_model_name if cfg.mode != "text" else None,
        "lora_r": cfg.lora_r,
        "total_params": params["total"],
        "trainable_params": params["trainable"],
        "trainable_pct": params["trainable_pct"],
        "checkpoint_mb": round(Path(checkpoint_path).stat().st_size / 1e6, 2),
        "ms_per_sample": round(1000 * elapsed / max(len(y_true), 1), 2),
        "device": str(device),
        **extra,
    })

    cfg.make_dirs()
    save_json(result, cfg.metrics_dir / f"lora_{tag}.json")
    save_predictions(cfg, f"lora_{tag}", y_true, y_pred, probs)
    plot_confusion_matrix(np.array(result["confusion_matrix"]), class_names,
                          cfg.figures_dir / f"cm_lora_{tag}.png",
                          title=f"{result['model']} — normalised confusion matrix (test set)")
    logger.info(f"[{result['model']}] test accuracy={result['accuracy']:.4f} | "
                f"macro-F1={result['f1_macro']:.4f} | weighted-F1={result['f1_weighted']:.4f}")
    return result

"""
Download the public dataset and create reproducible train / val / test splits.

Dataset
-------
`ashraq/fashion-product-images-small` on the Hugging Face Hub — a mirror of the
Kaggle "Fashion Product Images (Small)" dataset. Each row is a real product
from an Indian e-commerce catalogue with:

    productDisplayName  -> e.g. "Puma Men Black Running Shoes"
    image               -> product photo (PIL image)
    subCategory         -> e.g. "Shoes", "Topwear", "Watches", ...

What this module does
---------------------
1. Downloads the dataset once (cached in `data/raw/`).
2. Drops rare categories and caps very large ones (see `Config`).
3. Makes a *stratified* split, so every class appears in train/val/test
   in roughly the same proportion.
4. Saves the split indices and the label map to `data/processed/`, so that
   training, evaluation, the baseline and the app all use the SAME split.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict

import numpy as np
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split

from src.config import Config
from src.utils import get_logger, load_json, save_json

logger = get_logger(__name__)


def load_raw_dataset(cfg: Config) -> Dataset:
    """Download (first run only) and return the full Hugging Face dataset."""
    logger.info(f"Loading dataset '{cfg.dataset_name}' (first run downloads ~270 MB)...")
    ds = load_dataset(cfg.dataset_name, split="train", cache_dir=str(cfg.data_path / "raw"))
    logger.info(f"Loaded {len(ds):,} rows with columns: {ds.column_names}")
    return ds


def _select_indices(cfg: Config, labels: list, texts: list) -> np.ndarray:
    """Return the row indices we keep after filtering rare classes and capping big ones."""
    rng = np.random.default_rng(cfg.split_seed)

    # Keep only rows that have both a label and a non-empty product name.
    valid = [i for i, (y, t) in enumerate(zip(labels, texts)) if y and t and str(t).strip()]
    counts = Counter(labels[i] for i in valid)
    kept_classes = {c for c, n in counts.items() if n >= cfg.min_samples_per_class}

    dropped = sorted(c for c in counts if c not in kept_classes)
    logger.info(f"Keeping {len(kept_classes)} classes with >= {cfg.min_samples_per_class} samples; "
                f"dropping {len(dropped)} rare classes.")

    selected = []
    for cls in sorted(kept_classes):
        cls_idx = np.array([i for i in valid if labels[i] == cls])
        rng.shuffle(cls_idx)
        selected.extend(cls_idx[: cfg.max_samples_per_class].tolist())
    return np.array(sorted(selected))


def create_splits(cfg: Config, ds: Dataset | None = None) -> Dict:
    """Filter the data, split it, and save the split + label map to disk."""
    cfg.make_dirs()
    ds = ds if ds is not None else load_raw_dataset(cfg)

    # Reading single columns does NOT decode images, so this is fast.
    labels = ds[cfg.label_column]
    texts = ds[cfg.text_column]

    indices = _select_indices(cfg, labels, texts)
    y = [labels[i] for i in indices]

    # Two-step stratified split: (train+val) vs test, then train vs val.
    trainval_idx, test_idx, trainval_y, _ = train_test_split(
        indices, y, test_size=cfg.test_size, stratify=y, random_state=cfg.split_seed)
    relative_val = cfg.val_size / (1.0 - cfg.test_size)
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=relative_val, stratify=trainval_y, random_state=cfg.split_seed)

    classes = sorted(set(y))
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}

    splits = {"train": sorted(map(int, train_idx)),
              "val": sorted(map(int, val_idx)),
              "test": sorted(map(int, test_idx))}
    save_json(splits, cfg.splits_file)
    save_json({"label2id": label2id,
               "id2label": {str(k): v for k, v in id2label.items()},
               "class_counts": dict(Counter(y)),
               "label_column": cfg.label_column}, cfg.label_map_file)

    logger.info(f"Split sizes -> train {len(splits['train']):,} | val {len(splits['val']):,} | "
                f"test {len(splits['test']):,} | classes {len(classes)}")
    return splits


def load_label_map(cfg: Config) -> Dict:
    """Return {'label2id': {...}, 'id2label': {int: str}}."""
    raw = load_json(cfg.label_map_file)
    raw["id2label"] = {int(k): v for k, v in raw["id2label"].items()}
    return raw


def get_split_datasets(cfg: Config) -> Dict:
    """
    Main entry point used by the scripts.

    Returns a dict with:
        "train", "val", "test"  -> Hugging Face Dataset subsets
        "label2id", "id2label"  -> label maps
    Splits are created on the first call and re-used afterwards.
    """
    ds = load_raw_dataset(cfg)
    if not (cfg.splits_file.exists() and cfg.label_map_file.exists()):
        create_splits(cfg, ds)
    else:
        logger.info(f"Re-using saved split from {cfg.splits_file}")

    if not cfg.keywords_file.exists():
        from src.data.keywords import build_category_words
        build_category_words(cfg, ds, list(load_label_map(cfg)["label2id"]))

    splits = load_json(cfg.splits_file)
    label_map = load_label_map(cfg)
    keep_cols = [cfg.text_column, cfg.image_column, cfg.label_column]
    out = {name: ds.select(idx).select_columns(keep_cols) for name, idx in splits.items()}
    out.update(label_map)
    return out

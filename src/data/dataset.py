"""
PyTorch `Dataset` + `DataLoader` for the multimodal model.

How a batch is built
--------------------
  Dataset.__getitem__(i)  ->  one example: {"text": str, "pixel_values": Tensor[3,224,224], "label": int}
  MultimodalCollator      ->  stacks images and tokenizes all texts of the batch at once
  DataLoader              ->  yields batches:
        input_ids       [B, T]      token ids (padded to the longest name in the batch)
        attention_mask  [B, T]      1 for real tokens, 0 for padding
        pixel_values    [B, 3, H, W]
        labels          [B]
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

from src.config import Config
from src.data.preprocessing import build_image_transforms, clean_text, get_image_norm_stats


class ProductDataset(Dataset):
    """Wraps a Hugging Face dataset split and applies our preprocessing."""

    def __init__(self, hf_split, label2id: Dict[str, int], cfg: Config, image_transform=None,
                 text_transform=None):
        """`text_transform` (optional) changes the cleaned name, e.g. KeywordMasker."""
        self.cfg = cfg
        self.text_transform = text_transform
        self.label2id = label2id
        self.use_image = cfg.mode in ("multimodal", "image")

        # For a text-only model we never touch the images -> much faster loading.
        if not self.use_image and cfg.image_column in hf_split.column_names:
            hf_split = hf_split.remove_columns(cfg.image_column)
        self.data = hf_split
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        row = self.data[int(idx)]
        text = clean_text(row[self.cfg.text_column])
        if self.text_transform is not None:
            text = self.text_transform(text)
        item = {
            "text": text,
            "label": self.label2id[row[self.cfg.label_column]],
        }
        if self.use_image:
            item["pixel_values"] = self.image_transform(row[self.cfg.image_column])
        return item


class MultimodalCollator:
    """Turns a list of examples into one padded batch (used as `collate_fn`)."""

    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        out = {"labels": torch.tensor([b["label"] for b in batch], dtype=torch.long)}

        if self.tokenizer is not None:
            enc = self.tokenizer(
                [b["text"] for b in batch],
                padding=True,              # pad to the longest text in THIS batch (dynamic padding)
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            out["input_ids"] = enc["input_ids"]
            out["attention_mask"] = enc["attention_mask"]

        if "pixel_values" in batch[0]:
            out["pixel_values"] = torch.stack([b["pixel_values"] for b in batch])
        return out


def default_text_transform(cfg: Config):
    """The text transform used for training AND evaluation of this run (E1 masking or none)."""
    if not cfg.mask_keywords:
        return None
    from src.data.keywords import KeywordMasker, load_category_words
    return KeywordMasker(load_category_words(cfg))


def build_dataloaders(cfg: Config, splits: Dict, tokenizer,
                      names=("train", "val", "test"), text_transform="default") -> Dict[str, DataLoader]:
    """Create DataLoaders from the output of `get_split_datasets`.

    text_transform: "default" -> follow cfg.mask_keywords; or pass any callable / None
                    (the robustness experiments use this to corrupt test names).
    """
    if text_transform == "default":
        text_transform = default_text_transform(cfg)
    mean, std = get_image_norm_stats(cfg.image_model_name)
    train_tf = build_image_transforms(cfg.image_size, mean, std, train=True)
    eval_tf = build_image_transforms(cfg.image_size, mean, std, train=False)

    collator = MultimodalCollator(tokenizer if cfg.mode in ("multimodal", "text") else None,
                                  cfg.max_text_length)
    pin = torch.cuda.is_available()

    loaders = {}
    for name in names:
        ds = ProductDataset(splits[name], splits["label2id"], cfg,
                            image_transform=train_tf if name == "train" else eval_tf,
                            text_transform=text_transform)
        loaders[name] = DataLoader(
            ds,
            batch_size=cfg.batch_size if name == "train" else cfg.batch_size * 2,
            shuffle=(name == "train"),
            num_workers=cfg.num_workers,
            collate_fn=collator,
            pin_memory=pin,
            persistent_workers=cfg.num_workers > 0,
        )
    return loaders

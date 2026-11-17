"""
Text and image preprocessing.

Text:  light cleaning only — the tokenizer (WordPiece) handles the rest.
Image: resize -> (augment during training) -> tensor -> normalize with the
       exact mean/std the pretrained ViT was trained with.
"""

from __future__ import annotations

import re
from typing import Tuple

from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------------------- #
# Text
# ---------------------------------------------------------------------- #
def clean_text(text: str | None) -> str:
    """Lower-case, remove odd characters and collapse whitespace.

    >>> clean_text("  Puma Men  Black   Shoes!! ")
    'puma men black shoes'
    """
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9&\-\s]", " ", text)   # keep letters, digits, & and -
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------- #
# Image
# ---------------------------------------------------------------------- #
def to_rgb(image: Image.Image) -> Image.Image:
    """Some product photos are grayscale or RGBA; ViT expects 3-channel RGB."""
    if image.mode == "RGBA":
        # Paste on a white background so transparent areas don't turn black.
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    return image.convert("RGB") if image.mode != "RGB" else image


def get_image_norm_stats(image_model_name: str) -> Tuple[list, list]:
    """Read the normalization mean/std the pretrained ViT expects.

    Falls back to (0.5, 0.5, 0.5) — the values used by the ViT-in21k checkpoint —
    if the processor cannot be loaded (e.g. offline unit tests).
    """
    try:
        from transformers import AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained(image_model_name)
        return list(proc.image_mean), list(proc.image_std)
    except Exception:  # noqa: BLE001 - any failure -> safe defaults
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]


def build_image_transforms(image_size: int, mean: list, std: list, train: bool) -> transforms.Compose:
    """Return the torchvision pipeline for training (with augmentation) or evaluation."""
    if train:
        # Mild augmentation: product photos are centred on white backgrounds,
        # so we avoid aggressive crops that could cut the product out.
        return transforms.Compose([
            transforms.Lambda(to_rgb),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.05), fill=255),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    return transforms.Compose([
        transforms.Lambda(to_rgb),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

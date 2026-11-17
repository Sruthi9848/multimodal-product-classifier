"""
Text and image encoders.

Each encoder turns raw input into ONE fixed-size vector per product:

    TextEncoder : token ids  [B, T]       -> text vector  [B, 768]
    ImageEncoder: pixels     [B, 3, H, W] -> image vector [B, 768]

Both wrap a pretrained transformer whose weights are frozen; only the LoRA
adapters (added in `lora.py`) are trained.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


# ---------------------------------------------------------------------- #
# Builders for the pretrained backbones (downloaded on first use)
# ---------------------------------------------------------------------- #
def build_text_backbone(model_name: str) -> nn.Module:
    """e.g. 'distilbert-base-uncased' (66M params) or 'bert-base-uncased' (110M)."""
    return AutoModel.from_pretrained(model_name)


def build_image_backbone(model_name: str) -> nn.Module:
    """e.g. 'google/vit-base-patch16-224-in21k' (86M params).

    `add_pooling_layer=False` because we read the [CLS] token directly.
    """
    return AutoModel.from_pretrained(model_name, add_pooling_layer=False)


def _hidden_size(backbone: nn.Module) -> int:
    """Works for plain and PEFT-wrapped models (both expose `.config`)."""
    return int(backbone.config.hidden_size)


# ---------------------------------------------------------------------- #
# Encoders
# ---------------------------------------------------------------------- #
class TextEncoder(nn.Module):
    """DistilBERT/BERT + mean pooling over real (non-padding) tokens.

    Mean pooling:   v = Σ_t m_t · h_t  /  Σ_t m_t
    where h_t is the hidden state of token t and m_t is the attention mask (1/0).
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.output_dim = _hidden_size(backbone)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B, T, H]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)                                          # [B, T, 1]
        summed = (hidden * mask).sum(dim=1)                                                           # [B, H]
        counts = mask.sum(dim=1).clamp(min=1e-6)                                                      # [B, 1]
        return summed / counts


class ImageEncoder(nn.Module):
    """Vision Transformer: the image is cut into 16×16 patches, each patch becomes
    a token, and the special [CLS] token (position 0) summarises the whole image."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.output_dim = _hidden_size(backbone)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(pixel_values=pixel_values).last_hidden_state  # [B, 1 + num_patches, H]
        return hidden[:, 0]                                                    # [CLS] token -> [B, H]

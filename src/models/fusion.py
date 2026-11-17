"""
Multimodal feature fusion: combine the text vector and the image vector.

Two strategies are provided (choose with `Config.fusion_type`):

1. ConcatFusion  (simple baseline)
       t' = proj_t(t),  v' = proj_v(v)
       z  = MLP([t' ; v'])

2. GatedFusion   (default)
       t' = LN(proj_t(t)),  v' = LN(proj_v(v))
       g  = σ(W_g [t' ; v'] + b_g)           # one gate value in (0,1) per dimension
       z  = g ⊙ t' + (1 − g) ⊙ v'
   The model *learns* how much to trust text vs. image for each product.
   The average of g is a handy "text reliance" score we show in the web app.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def _projection(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    """Map an encoder output into the shared fusion space."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )


class ConcatFusion(nn.Module):
    def __init__(self, text_dim: int, image_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.proj_text = _projection(text_dim, out_dim, dropout)
        self.proj_image = _projection(image_dim, out_dim, dropout)
        self.mix = nn.Sequential(nn.Linear(2 * out_dim, out_dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, text_feat: torch.Tensor, image_feat: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        t, v = self.proj_text(text_feat), self.proj_image(image_feat)
        return self.mix(torch.cat([t, v], dim=-1)), None


class GatedFusion(nn.Module):
    def __init__(self, text_dim: int, image_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.proj_text = _projection(text_dim, out_dim, dropout)
        self.proj_image = _projection(image_dim, out_dim, dropout)
        self.gate = nn.Linear(2 * out_dim, out_dim)

    def forward(self, text_feat: torch.Tensor, image_feat: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        t, v = self.proj_text(text_feat), self.proj_image(image_feat)
        g = torch.sigmoid(self.gate(torch.cat([t, v], dim=-1)))  # [B, D]
        fused = g * t + (1.0 - g) * v
        return fused, g


def build_fusion(fusion_type: str, text_dim: int, image_dim: int, out_dim: int, dropout: float) -> nn.Module:
    if fusion_type == "gated":
        return GatedFusion(text_dim, image_dim, out_dim, dropout)
    if fusion_type == "concat":
        return ConcatFusion(text_dim, image_dim, out_dim, dropout)
    raise ValueError(f"Unknown fusion_type '{fusion_type}'. Use 'gated' or 'concat'.")

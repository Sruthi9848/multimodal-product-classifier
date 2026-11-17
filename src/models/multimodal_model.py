"""
The complete model:

    product name ──► DistilBERT + LoRA ──► text vector  ─┐
                                                          ├─► Fusion ─► Classifier ─► logits ─► softmax
    product photo ─► ViT + LoRA ────────► image vector ──┘

The same class also supports text-only and image-only models (`Config.mode`),
which we use as ablations to show what each modality contributes.

Checkpoints only store the TRAINABLE weights (LoRA adapters + fusion +
classifier, a few MB). When loading, the frozen pretrained backbones are
re-downloaded from the Hub (or read from the local cache) and the small
trained weights are placed on top.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.config import Config
from src.models.encoders import ImageEncoder, TextEncoder, build_image_backbone, build_text_backbone
from src.models.fusion import build_fusion
from src.models.lora import apply_lora


class MultimodalClassifier(nn.Module):
    def __init__(self, cfg: Config, num_classes: int,
                 text_backbone: Optional[nn.Module] = None,
                 image_backbone: Optional[nn.Module] = None):
        """
        Args:
            cfg: project configuration (mode, fusion type, dims, ...).
            num_classes: number of product categories.
            text_backbone / image_backbone: already LoRA-wrapped backbones.
                Use `MultimodalClassifier.build(cfg, num_classes)` to create them
                automatically from pretrained checkpoints.
        """
        super().__init__()
        self.mode = cfg.mode
        self.modality_dropout = cfg.modality_dropout
        d = cfg.fusion_dim

        self.text_encoder = TextEncoder(text_backbone) if text_backbone is not None else None
        self.image_encoder = ImageEncoder(image_backbone) if image_backbone is not None else None

        if self.mode == "multimodal":
            assert self.text_encoder is not None and self.image_encoder is not None
            self.fusion = build_fusion(cfg.fusion_type, self.text_encoder.output_dim,
                                       self.image_encoder.output_dim, d, cfg.dropout)
            # Learnable "this modality is missing" placeholders (used for modality dropout
            # during training and when the user gives only one input at prediction time).
            self.missing_text = nn.Parameter(torch.zeros(self.text_encoder.output_dim))
            self.missing_image = nn.Parameter(torch.zeros(self.image_encoder.output_dim))
        else:
            encoder = self.text_encoder if self.mode == "text" else self.image_encoder
            assert encoder is not None, f"mode='{self.mode}' needs the matching backbone"
            self.single_proj = nn.Sequential(nn.Linear(encoder.output_dim, d), nn.LayerNorm(d),
                                             nn.GELU(), nn.Dropout(cfg.dropout))

        self.classifier = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d // 2, num_classes),
        )

    # ------------------------------------------------------------------ #
    @classmethod
    def build(cls, cfg: Config, num_classes: int) -> "MultimodalClassifier":
        """Download pretrained backbones, add LoRA adapters and assemble the model."""
        text_bb = image_bb = None
        if cfg.mode in ("multimodal", "text"):
            text_bb = apply_lora(build_text_backbone(cfg.text_model_name), cfg.text_lora_targets,
                                 cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout)
        if cfg.mode in ("multimodal", "image"):
            image_bb = apply_lora(build_image_backbone(cfg.image_model_name), cfg.image_lora_targets,
                                  cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout)
        return cls(cfg, num_classes, text_bb, image_bb)

    # ------------------------------------------------------------------ #
    def _apply_modality_dropout(self, t: torch.Tensor, v: torch.Tensor):
        """During training, randomly replace one modality with its 'missing' placeholder."""
        if not (self.training and self.modality_dropout > 0):
            return t, v
        B = t.size(0)
        drop = torch.rand(B, device=t.device) < self.modality_dropout
        drop_text = (drop & (torch.rand(B, device=t.device) < 0.5)).unsqueeze(-1)
        drop_image = (drop & ~drop_text.squeeze(-1)).unsqueeze(-1)
        t = torch.where(drop_text, self.missing_text.expand_as(t), t)
        v = torch.where(drop_image, self.missing_image.expand_as(v), v)
        return t, v

    def forward(self, input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.Tensor] = None) -> Dict[str, Optional[torch.Tensor]]:
        """
        Returns:
            {"logits": [B, C], "text_weight": [B] or None}
            `text_weight` = mean gate value (gated fusion only): ~1 means the
            decision leaned on the text, ~0 means it leaned on the image.
        """
        text_weight = None

        if self.mode == "multimodal":
            has_text, has_image = input_ids is not None, pixel_values is not None
            if not (has_text or has_image):
                raise ValueError("Provide a product name, an image, or both.")
            B = (input_ids if has_text else pixel_values).size(0)

            t = (self.text_encoder(input_ids, attention_mask) if has_text
                 else self.missing_text.expand(B, -1))
            v = (self.image_encoder(pixel_values) if has_image
                 else self.missing_image.expand(B, -1))
            t, v = self._apply_modality_dropout(t, v)
            z, gate = self.fusion(t, v)
            if gate is not None:
                text_weight = gate.mean(dim=-1)
        elif self.mode == "text":
            z = self.single_proj(self.text_encoder(input_ids, attention_mask))
        else:  # image
            z = self.single_proj(self.image_encoder(pixel_values))

        return {"logits": self.classifier(z), "text_weight": text_weight}

    # ------------------------------------------------------------------ #
    # Saving / loading only the small trainable part
    # ------------------------------------------------------------------ #
    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(f"Checkpoint has unexpected keys: {result.unexpected_keys[:5]} ...")
        expected = {n for n, p in self.named_parameters() if p.requires_grad}
        missing = expected - set(state)
        if missing:
            raise RuntimeError(f"Checkpoint is missing trainable weights: {sorted(missing)[:5]} ...")


# ---------------------------------------------------------------------- #
def save_checkpoint(model: MultimodalClassifier, cfg: Config, label_map: Dict, path: Path,
                    extra: Optional[Dict] = None) -> None:
    """Save trainable weights + everything needed to rebuild the model later."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.trainable_state_dict(),
        "config": cfg.to_dict(),
        "label2id": label_map["label2id"],
        "id2label": {int(k): v for k, v in label_map["id2label"].items()},
        "extra": extra or {},
    }, path)


def load_model_from_checkpoint(path: Path, device: torch.device):
    """Rebuild the model (pretrained backbones + LoRA) and load trained weights.

    Returns: (model in eval mode, Config, id2label dict, extra dict)
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    id2label = {int(k): v for k, v in ckpt["id2label"].items()}
    model = MultimodalClassifier.build(cfg, num_classes=len(id2label))
    model.load_trainable_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, cfg, id2label, ckpt.get("extra", {})

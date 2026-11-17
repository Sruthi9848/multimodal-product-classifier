"""
LoRA (Low-Rank Adaptation) helpers built on the Hugging Face PEFT library.

The idea in one equation
------------------------
A pretrained linear layer computes  y = W0 · x  with a big frozen matrix W0 (d × k).
LoRA keeps W0 frozen and learns a *low-rank* correction:

        y = W0 · x  +  (alpha / r) · B · A · x

        A : r × k   (initialised randomly)
        B : d × r   (initialised to zeros, so training starts exactly at the
                     pretrained model)
        r : the rank, e.g. 8  — much smaller than d and k (768 for BERT / ViT)

Trainable parameters per layer drop from d·k (589,824) to r·(d+k) (12,288)
— about 2% — which is why LoRA trains fast, needs little GPU memory and saves
tiny checkpoints.

We only add LoRA to the attention *query* and *value* projections, which is
the setting used in the original LoRA paper (Hu et al., 2021).
"""

from __future__ import annotations

from typing import List

import torch.nn as nn
from peft import LoraConfig, get_peft_model


def apply_lora(backbone: nn.Module, target_modules: List[str], r: int, alpha: int, dropout: float) -> nn.Module:
    """Freeze `backbone` and inject trainable LoRA adapters into `target_modules`.

    Args:
        backbone: a pretrained Hugging Face model (DistilBERT, ViT, ...).
        target_modules: names of the nn.Linear layers to adapt, e.g. ["q_lin", "v_lin"].
        r: LoRA rank.
        alpha: LoRA scaling numerator (effective scale = alpha / r).
        dropout: dropout applied to the LoRA branch input.

    Returns:
        A `PeftModel` that behaves exactly like the original backbone
        (same forward arguments), but only the LoRA weights are trainable.
    """
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",   # do not train any bias terms of the backbone
    )
    return get_peft_model(backbone, lora_config)

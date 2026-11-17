"""
Central configuration for the whole project.

Every script (training, evaluation, baseline, prediction, Streamlit app)
reads its settings from the `Config` dataclass below, so there is exactly
ONE place to change a hyperparameter.

Command-line flags in `scripts/*.py` can override a few common values
(epochs, batch size, mode, ...), but the defaults all live here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

# Project root = the folder that contains `src/`, `scripts/`, `app/` ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    # Public dataset on the Hugging Face Hub (≈44k real product photos,
    # originally the Kaggle "Fashion Product Images (Small)" dataset).
    dataset_name: str = "ashraq/fashion-product-images-small"
    text_column: str = "productDisplayName"   # product name / short description
    image_column: str = "image"               # PIL image
    label_column: str = "subCategory"         # what we predict (Topwear, Shoes, Watches, ...)

    # Long-tail categories with very few examples are dropped, and very large
    # categories are capped so that one class does not dominate training.
    min_samples_per_class: int = 300
    max_samples_per_class: int = 1500

    val_size: float = 0.10                    # fraction of data for validation
    test_size: float = 0.15                   # fraction of data for the final test
    split_seed: int = 42                      # fixes the train/val/test split (never change for a paper)
    seed: int = 42                            # training seed: weight init, shuffling, dropout

    # ------------------------------------------------------------------ #
    # Paper experiments
    # ------------------------------------------------------------------ #
    # E1: remove category words ("watch", "shoes", ...) from every product name,
    # so the model can no longer read the answer straight from the text.
    mask_keywords: bool = False
    # Name used for checkpoints / metrics files. Empty -> built automatically
    # from mode, masking and seed (see `run_tag`).
    run_name: str = ""

    # ------------------------------------------------------------------ #
    # Pretrained backbones
    # ------------------------------------------------------------------ #
    text_model_name: str = "distilbert-base-uncased"
    image_model_name: str = "google/vit-base-patch16-224-in21k"
    max_text_length: int = 48                 # product names are short
    image_size: int = 224

    # ------------------------------------------------------------------ #
    # Model architecture
    # ------------------------------------------------------------------ #
    # "multimodal" = text + image, "text" = text only, "image" = image only
    mode: str = "multimodal"
    fusion_type: str = "gated"                # "gated" or "concat"
    fusion_dim: int = 512                     # size of the shared embedding space
    dropout: float = 0.2
    # Probability of hiding one modality during training (multimodal only).
    # This teaches the model to still work when the user gives only text or
    # only an image in the web app.
    modality_dropout: float = 0.15

    # ------------------------------------------------------------------ #
    # LoRA (Low-Rank Adaptation)
    # ------------------------------------------------------------------ #
    lora_r: int = 8                           # rank of the low-rank update
    lora_alpha: int = 16                      # scaling factor (update is scaled by alpha / r)
    lora_dropout: float = 0.1
    # Names of the attention projection layers that receive LoRA adapters.
    text_lora_targets: List[str] = field(default_factory=lambda: ["q_lin", "v_lin"])   # DistilBERT
    image_lora_targets: List[str] = field(default_factory=lambda: ["query", "value"])  # ViT

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    epochs: int = 5
    batch_size: int = 32
    lr_lora: float = 3e-4                     # learning rate for LoRA adapters
    lr_head: float = 1e-3                     # learning rate for fusion + classifier
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    early_stopping_patience: int = 2
    num_workers: int = 2                      # set to 0 on Windows if you hit DataLoader errors
    use_amp: bool = True                      # mixed precision (only used on CUDA GPUs)

    # ------------------------------------------------------------------ #
    # Paths (all relative to PROJECT_ROOT)
    # ------------------------------------------------------------------ #
    data_dir: str = "data"
    output_dir: str = "outputs"

    # ---------- convenience helpers ----------
    @property
    def data_path(self) -> Path:
        return PROJECT_ROOT / self.data_dir

    @property
    def output_path(self) -> Path:
        return PROJECT_ROOT / self.output_dir

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_path / "checkpoints"

    @property
    def metrics_dir(self) -> Path:
        return self.output_path / "metrics"

    @property
    def figures_dir(self) -> Path:
        return self.output_path / "figures"

    @property
    def splits_file(self) -> Path:
        return self.data_path / "processed" / "splits.json"

    @property
    def label_map_file(self) -> Path:
        return self.data_path / "processed" / "label_map.json"

    @property
    def keywords_file(self) -> Path:
        return self.data_path / "processed" / "category_words.json"

    @property
    def predictions_dir(self) -> Path:
        return self.output_path / "predictions"

    @property
    def run_tag(self) -> str:
        """e.g. 'multimodal', 'text_masked', 'image_s7', 'multimodal_masked_s123'.

        The default run (seed 42, no masking) keeps the plain mode name, so the
        files from earlier runs and the Streamlit app keep working.
        """
        if self.run_name:
            return self.run_name
        tag = self.mode + ("_masked" if self.mask_keywords else "")
        return tag + (f"_s{self.seed}" if self.seed != 42 else "")

    def checkpoint_file(self, name: str | None = None) -> Path:
        """Path of a checkpoint: `name` is a mode or run tag (default: this run)."""
        return self.checkpoint_dir / f"lora_{name or self.run_tag}.pt"

    def make_dirs(self) -> None:
        for d in [self.data_path / "processed", self.checkpoint_dir,
                  self.metrics_dir, self.figures_dir, self.predictions_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Rebuild a Config, ignoring unknown keys (useful for old checkpoints)."""
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

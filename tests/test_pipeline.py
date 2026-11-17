"""
Offline smoke tests: check that every component connects and runs.

They use TINY randomly-initialised transformers (no internet needed, runs in
seconds on CPU) — they test the plumbing, not accuracy.
Real results come only from training on the real dataset.

    python -m pytest tests -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import DistilBertConfig, DistilBertModel, ViTConfig, ViTModel  # noqa: E402

from src.baseline.traditional_ml import BaselineClassifier, image_to_features  # noqa: E402
from src.config import Config  # noqa: E402
from src.data.dataset import MultimodalCollator, ProductDataset  # noqa: E402
from src.data.preprocessing import build_image_transforms, clean_text  # noqa: E402
from src.evaluation.metrics import compute_metrics, full_evaluation  # noqa: E402
from src.models.lora import apply_lora  # noqa: E402
from src.models.multimodal_model import MultimodalClassifier  # noqa: E402

NUM_CLASSES = 4
IMG = 32


def tiny_cfg(mode="multimodal", fusion="gated") -> Config:
    return Config(mode=mode, fusion_type=fusion, fusion_dim=32, image_size=IMG, modality_dropout=0.5)


def tiny_model(cfg: Config) -> MultimodalClassifier:
    text_bb = img_bb = None
    if cfg.mode in ("multimodal", "text"):
        text_bb = apply_lora(DistilBertModel(DistilBertConfig(vocab_size=100, dim=32, n_layers=1, n_heads=2,
                                                              hidden_dim=64)),
                             cfg.text_lora_targets, r=4, alpha=8, dropout=0.0)
    if cfg.mode in ("multimodal", "image"):
        img_bb = apply_lora(ViTModel(ViTConfig(image_size=IMG, patch_size=8, hidden_size=32, num_hidden_layers=1,
                                               num_attention_heads=2, intermediate_size=64),
                                     add_pooling_layer=False),
                            cfg.image_lora_targets, r=4, alpha=8, dropout=0.0)
    return MultimodalClassifier(cfg, NUM_CLASSES, text_bb, img_bb)


def fake_batch(B=6, T=7):
    return {"input_ids": torch.randint(1, 100, (B, T)), "attention_mask": torch.ones(B, T, dtype=torch.long),
            "pixel_values": torch.randn(B, 3, IMG, IMG)}


# ---------------------------------------------------------------------- #
def test_clean_text():
    assert clean_text("  Puma Men  Black   Shoes!! ") == "puma men black shoes"
    assert clean_text(None) == ""


@pytest.mark.parametrize("mode", ["multimodal", "text", "image"])
@pytest.mark.parametrize("fusion", ["gated", "concat"])
def test_forward_shapes(mode, fusion):
    model = tiny_model(tiny_cfg(mode, fusion)).eval()
    b = fake_batch()
    if mode == "text":
        b.pop("pixel_values")
    if mode == "image":
        b.pop("input_ids"), b.pop("attention_mask")
    out = model(**b)
    assert out["logits"].shape == (6, NUM_CLASSES)


def test_missing_modality_and_gate():
    model = tiny_model(tiny_cfg()).eval()
    b = fake_batch()
    assert model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])["logits"].shape == (6, NUM_CLASSES)
    assert model(pixel_values=b["pixel_values"])["logits"].shape == (6, NUM_CLASSES)
    w = model(**b)["text_weight"]
    assert w.shape == (6,) and torch.all((w > 0) & (w < 1))


def test_only_lora_and_head_are_trainable():
    model = tiny_model(tiny_cfg())
    for name, p in model.named_parameters():
        if "backbone" in name and "lora_" not in name:
            assert not p.requires_grad, f"{name} should be frozen"
    assert any("lora_" in n for n, p in model.named_parameters() if p.requires_grad)


def test_training_step_reduces_loss():
    torch.manual_seed(0)
    model = tiny_model(tiny_cfg())
    model.modality_dropout = 0.0
    b, y = fake_batch(B=8), torch.arange(8) % NUM_CLASSES
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    first = None
    for _ in range(30):
        opt.zero_grad()
        loss = loss_fn(model(**b)["logits"], y)
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first


def test_trainable_checkpoint_roundtrip():
    torch.manual_seed(0)
    m1 = tiny_model(tiny_cfg()).eval()
    with torch.no_grad():  # make LoRA B non-zero so the test is meaningful
        for n, p in m1.named_parameters():
            if "lora_B" in n:
                p.normal_()
    state = m1.trainable_state_dict()
    torch.manual_seed(0)  # same frozen backbone init
    m2 = tiny_model(tiny_cfg()).eval()
    m2.load_trainable_state_dict(state)
    b = fake_batch()
    assert torch.allclose(m1(**b)["logits"], m2(**b)["logits"], atol=1e-5)


def test_dataset_and_collator():
    datasets = pytest.importorskip("datasets")
    cfg = tiny_cfg()
    rows = {"productDisplayName": ["Red Shoe", "Blue Watch"], "subCategory": ["Shoes", "Watches"],
            "image": [Image.new("RGB", (60, 80), (200, 20, 20)), Image.new("L", (60, 80), 90)]}
    hf = datasets.Dataset.from_dict(rows).cast_column("image", datasets.Image())
    tf = build_image_transforms(IMG, [0.5] * 3, [0.5] * 3, train=True)
    ds = ProductDataset(hf, {"Shoes": 0, "Watches": 1}, cfg, tf)
    item = ds[1]
    assert item["pixel_values"].shape == (3, IMG, IMG) and item["label"] == 1

    class DummyTok:  # mimics the tokenizer call signature
        def __call__(self, texts, **kw):
            T = max(len(t.split()) for t in texts)
            return {"input_ids": torch.ones(len(texts), T, dtype=torch.long),
                    "attention_mask": torch.ones(len(texts), T, dtype=torch.long)}

    batch = MultimodalCollator(DummyTok(), 16)([ds[0], ds[1]])
    assert batch["pixel_values"].shape == (2, 3, IMG, IMG) and batch["labels"].tolist() == [0, 1]


def test_metrics_and_baseline():
    y = np.array([0, 1, 2, 3, 0, 1])
    m = compute_metrics(y, y)
    assert m["accuracy"] == 1.0 and m["f1_macro"] == 1.0
    probs = np.eye(4)[y]
    full = full_evaluation(y, y, ["a", "b", "c", "d"], probs)
    assert full["top3_accuracy"] == 1.0 and len(full["confusion_matrix"]) == 4

    feats = image_to_features(Image.new("RGB", (60, 80), (10, 200, 30)))
    assert feats.shape == (48 + 32 + 576,)
    texts = ["red shoe", "blue watch", "green bag", "black belt"] * 3
    img = np.random.rand(12, feats.shape[0]).astype(np.float32)
    labels = np.array([0, 1, 2, 3] * 3)
    clf = BaselineClassifier("multimodal").fit(texts, img, labels)
    assert clf.predict_proba(texts, img).shape == (12, 4)

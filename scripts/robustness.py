"""
E2 — Robustness: how do the trained models behave when the input is damaged?

Conditions (applied to the TEST set only, the models are not retrained):
    clean               name + photo as they are
    no_photo            photo removed            (multimodal models)
    no_name             name removed             (multimodal models; text models cannot run)
    words_dropped_50    each word of the name dropped with p = 0.5
    masked_at_test      category words removed from names at test time only
                        (shortcut test for models trained on normal names)

Models evaluated (whichever exist): LoRA multimodal / text / image, their
"_masked" versions from E1, and the classical text / multimodal baselines.

Writes:
    outputs/metrics/robustness.csv  and  robustness.md
    outputs/figures/robustness.png

    python scripts/robustness.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

import argparse  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from src.baseline.traditional_ml import BaselineClassifier, extract_image_features  # noqa: E402
from src.config import Config  # noqa: E402
from src.data.dataset import build_dataloaders, default_text_transform  # noqa: E402
from src.data.dataset_loader import get_split_datasets  # noqa: E402
from src.data.keywords import Compose, KeywordMasker, WordDropper, load_category_words  # noqa: E402
from src.data.preprocessing import clean_text  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.models.multimodal_model import load_model_from_checkpoint  # noqa: E402
from src.training.trainer import move_to_device  # noqa: E402
from src.utils import get_device, get_logger, save_json  # noqa: E402

logger = get_logger("robustness")

CONDITIONS = ["clean", "no_photo", "no_name", "words_dropped_50", "masked_at_test"]
LORA_RUNS = ["multimodal", "text", "image", "multimodal_masked", "text_masked"]
BASELINE_RUNS = ["baseline_text", "baseline_multimodal", "baseline_text_masked", "baseline_multimodal_masked"]


def applicable(mode: str, masked: bool, condition: str) -> bool:
    """Which (model, condition) pairs make sense."""
    if condition in ("no_photo", "no_name"):
        return mode == "multimodal"
    if condition in ("words_dropped_50", "masked_at_test"):
        if mode == "image":
            return False
        return not (masked and condition == "masked_at_test")  # already masked: identical to clean
    return True


def condition_transform(condition: str, words):
    if condition == "words_dropped_50":
        return WordDropper(p=0.5)
    if condition == "masked_at_test":
        return KeywordMasker(words)
    return None


# ---------------------------------------------------------------------- #
# LoRA models
# ---------------------------------------------------------------------- #
@torch.no_grad()
def eval_lora(path: Path, splits, words, device, num_workers: int):
    model, cfg, _, _ = load_model_from_checkpoint(path, device)
    cfg.num_workers = num_workers
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)
    rows = []
    for cond in CONDITIONS:
        if not applicable(cfg.mode, cfg.mask_keywords, cond):
            continue
        tf = Compose(default_text_transform(cfg), condition_transform(cond, words))
        loader = build_dataloaders(cfg, splits, tokenizer, names=("test",), text_transform=tf)["test"]
        y_true, y_pred = [], []
        for batch in tqdm(loader, desc=f"{path.stem} | {cond}", leave=False):
            batch = move_to_device(batch, device)
            inputs = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask", "pixel_values")}
            if cond == "no_photo":
                inputs.pop("pixel_values", None)
            if cond == "no_name":
                inputs.pop("input_ids", None)
                inputs.pop("attention_mask", None)
            y_pred.append(model(**inputs)["logits"].argmax(-1).cpu())
            y_true.append(batch["labels"].cpu())
        m = compute_metrics(torch.cat(y_true).numpy(), torch.cat(y_pred).numpy())
        rows.append({"model": path.stem, "condition": cond, **m})
        logger.info(f"{path.stem:<24} {cond:<18} acc={m['accuracy']:.4f}  macro-F1={m['f1_macro']:.4f}")
    del model
    return rows


# ---------------------------------------------------------------------- #
# Classical baselines
# ---------------------------------------------------------------------- #
def eval_baseline(path: Path, cfg: Config, test, label2id, words):
    model = BaselineClassifier.load(path)
    masked = path.stem.endswith("_masked")
    base_mask = KeywordMasker(words) if masked else None
    texts = [clean_text(t) for t in test[cfg.text_column]]
    y = np.array([label2id[c] for c in test[cfg.label_column]])

    img = None
    if model.mode != "text":
        cache = cfg.data_path / "processed" / "baseline_image_features_test.npy"
        img = np.load(cache) if cache.exists() else extract_image_features(
            (r[cfg.image_column] for r in test), total=len(test))

    rows = []
    for cond in CONDITIONS:
        if not applicable(model.mode, masked, cond):
            continue
        t = [base_mask(x) for x in texts] if base_mask else list(texts)
        tf = condition_transform(cond, words)
        if tf is not None:
            t = [tf(x) for x in t]
        im = img
        if cond == "no_name":
            t = [""] * len(t)
        if cond == "no_photo":
            im = np.tile(model.img_scaler.mean_, (len(t), 1))  # "average image" = no information
        pred = model.predict(t, im)
        m = compute_metrics(y, pred)
        rows.append({"model": path.stem, "condition": cond, **m})
        logger.info(f"{path.stem:<24} {cond:<18} acc={m['accuracy']:.4f}  macro-F1={m['f1_macro']:.4f}")
    return rows


def plot(df: pd.DataFrame, path: Path) -> None:
    pivot = df.pivot(index="model", columns="condition", values="accuracy")
    pivot = pivot[[c for c in CONDITIONS if c in pivot.columns]]
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(pivot) + 2))
    colors = ["#1E2B4A", "#7A9CC6", "#B8C4D6", "#E0A526", "#C0504D"]
    h = 0.8 / len(pivot.columns)
    y = np.arange(len(pivot))
    for i, (cond, c) in enumerate(zip(pivot.columns, colors)):
        ax.barh(y + (i - len(pivot.columns) / 2 + 0.5) * h, pivot[cond].fillna(0), height=h,
                color=c, label=cond.replace("_", " "))
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.set_xlim(0, 1)
    ax.set_xlabel("test accuracy (missing bar = condition not applicable)")
    ax.set_title("E2 — Robustness to missing and degraded inputs")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="E2 robustness evaluation.")
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()

    cfg = Config()
    cfg.make_dirs()
    splits = get_split_datasets(cfg)
    words = load_category_words(cfg)
    device = get_device()

    rows = []
    for run in LORA_RUNS:
        path = cfg.checkpoint_file(run)
        if path.exists():
            rows += eval_lora(path, splits, words, device, args.num_workers)
        else:
            logger.info(f"(skipping {path.name}: not trained yet)")
    for run in BASELINE_RUNS:
        path = cfg.checkpoint_dir / f"{run}.joblib"
        if path.exists():
            rows += eval_baseline(path, cfg, splits["test"], splits["label2id"], words)

    if not rows:
        sys.exit("No trained models found. Train them first (see notebooks/paper_experiments_colab.ipynb).")
    df = pd.DataFrame(rows)
    df.to_csv(cfg.metrics_dir / "robustness.csv", index=False)
    table = df.pivot(index="model", columns="condition", values="accuracy")
    table = table[[c for c in CONDITIONS if c in table.columns]]
    (cfg.metrics_dir / "robustness.md").write_text(
        "Test accuracy under each condition (— = not applicable)\n\n"
        + table.to_markdown(floatfmt=".4f", missingval="—"))
    save_json(rows, cfg.metrics_dir / "robustness.json")
    plot(df, cfg.figures_dir / "robustness.png")
    print("\n" + table.round(4).to_string(na_rep="—"))


if __name__ == "__main__":
    main()

"""
STEP 2 — Fine-tune the LoRA model and evaluate it on the test set.

Examples (run from the project root):
    python scripts/train.py                          # multimodal (text + image), default settings
    python scripts/train.py --mode text              # text-only ablation
    python scripts/train.py --mode image             # image-only ablation
    python scripts/train.py --fusion concat          # try the simpler fusion
    python scripts/train.py --max-train-samples 2000 --epochs 2   # quick CPU smoke run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

import argparse  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402

from src.config import Config  # noqa: E402
from src.data.dataset import build_dataloaders  # noqa: E402
from src.data.dataset_loader import get_split_datasets  # noqa: E402
from src.evaluation.evaluator import evaluate_lora_checkpoint  # noqa: E402
from src.evaluation.metrics import plot_training_history  # noqa: E402
from src.models.multimodal_model import MultimodalClassifier  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402
from src.utils import count_parameters, get_device, get_logger, set_seed  # noqa: E402

logger = get_logger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the LoRA multimodal product classifier.")
    p.add_argument("--mode", choices=["multimodal", "text", "image"], default=None)
    p.add_argument("--fusion", choices=["gated", "concat"], default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--text-model", default=None, help="e.g. bert-base-uncased")
    p.add_argument("--seed", type=int, default=None, help="training seed (the data split never changes)")
    p.add_argument("--mask-keywords", action="store_true",
                   help="E1: remove category words from every product name")
    p.add_argument("--run-name", default=None, help="custom name for checkpoint/metrics files")
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="use only N training examples (quick experiments on CPU)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    # Apply command-line overrides on top of the defaults in src/config.py
    for arg_name, cfg_name in [("mode", "mode"), ("fusion", "fusion_type"), ("epochs", "epochs"),
                               ("batch_size", "batch_size"), ("lora_r", "lora_r"),
                               ("num_workers", "num_workers"), ("text_model", "text_model_name"),
                               ("seed", "seed"), ("run_name", "run_name")]:
        value = getattr(args, arg_name)
        if value is not None:
            setattr(cfg, cfg_name, value)
    if args.mask_keywords:
        cfg.mask_keywords = True
    if args.lora_r is not None:
        cfg.lora_alpha = 2 * args.lora_r  # keep alpha / r = 2

    cfg.make_dirs()
    set_seed(cfg.seed)
    device = get_device()
    logger.info(f"Device: {device} | run: {cfg.run_tag} | mode: {cfg.mode} | fusion: {cfg.fusion_type} | "
                f"seed: {cfg.seed} | masked names: {cfg.mask_keywords}")

    # ---- data ----
    splits = get_split_datasets(cfg)
    if args.max_train_samples and args.max_train_samples < len(splits["train"]):
        splits["train"] = splits["train"].shuffle(seed=cfg.seed).select(range(args.max_train_samples))
        logger.info(f"Using a subset of {args.max_train_samples} training examples.")
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)
    loaders = build_dataloaders(cfg, splits, tokenizer)

    # ---- model ----
    num_classes = len(splits["label2id"])
    model = MultimodalClassifier.build(cfg, num_classes)
    params = count_parameters(model)
    logger.info(f"Parameters: total {params['total']:,} | trainable {params['trainable']:,} "
                f"({params['trainable_pct']:.2f}%)  <- LoRA keeps this tiny")

    # ---- train ----
    label_map = {"label2id": splits["label2id"], "id2label": splits["id2label"]}
    history = Trainer(model, cfg, loaders, label_map, device).fit()
    plot_training_history(history, cfg.figures_dir / f"history_{cfg.run_tag}.png",
                          title=f"LoRA {cfg.run_tag} training history")

    # ---- final test evaluation of the BEST checkpoint ----
    del model
    evaluate_lora_checkpoint(cfg.checkpoint_file(), num_workers=cfg.num_workers)
    logger.info("Done. Next: python scripts/train_baseline.py  then  python scripts/compare_models.py")


if __name__ == "__main__":
    main()

"""
STEP 1 — Download the dataset, create the train/val/test split and save EDA plots.

Run from the project root:
    python scripts/prepare_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

import argparse  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import Config  # noqa: E402
from src.data.dataset_loader import create_splits, get_split_datasets, load_raw_dataset  # noqa: E402
from src.utils import get_logger, load_json, set_seed  # noqa: E402

logger = get_logger("prepare_data")


def plot_class_distribution(cfg: Config) -> None:
    counts = load_json(cfg.label_map_file)["class_counts"]
    items = sorted(counts.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(items) + 1.5))
    ax.barh([k for k, _ in items], [v for _, v in items], color="#2E4A7D")
    ax.set_xlabel("number of products (after capping)")
    ax.set_title(f"Class distribution — {cfg.label_column}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.figures_dir / "class_distribution.png", dpi=150)
    plt.close(fig)


def plot_sample_grid(cfg: Config, train_split, n: int = 12) -> None:
    fig, axes = plt.subplots(2, n // 2, figsize=(2.1 * n // 2, 5.2))
    for ax, i in zip(axes.ravel(), range(0, len(train_split), max(1, len(train_split) // n))):
        row = train_split[i]
        ax.imshow(row[cfg.image_column])
        name = row[cfg.text_column]
        ax.set_title(f"{row[cfg.label_column]}\n{name[:22]}{'…' if len(name) > 22 else ''}", fontsize=7)
        ax.axis("off")
    fig.suptitle("Random training examples (label / product name)")
    fig.tight_layout()
    fig.savefig(cfg.figures_dir / "sample_products.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download data and build the split.")
    parser.add_argument("--force", action="store_true", help="re-create the split even if it exists")
    args = parser.parse_args()

    cfg = Config()
    cfg.make_dirs()
    set_seed(cfg.seed)

    if args.force or not cfg.splits_file.exists():
        create_splits(cfg, load_raw_dataset(cfg))
    splits = get_split_datasets(cfg)

    plot_class_distribution(cfg)
    plot_sample_grid(cfg, splits["train"])
    logger.info(f"Classes ({len(splits['label2id'])}): {', '.join(splits['label2id'])}")
    logger.info(f"Saved EDA figures to {cfg.figures_dir}")


if __name__ == "__main__":
    main()

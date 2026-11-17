"""
STEP 3 — Train and evaluate the traditional ML baselines (TF-IDF / handcrafted
image features + Logistic Regression) on exactly the same split as the LoRA model.

    python scripts/train_baseline.py
    python scripts/train_baseline.py --modes text multimodal
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from src.baseline.traditional_ml import BaselineClassifier, extract_image_features  # noqa: E402
from src.config import Config  # noqa: E402
from src.data.dataset_loader import get_split_datasets  # noqa: E402
from src.data.keywords import KeywordMasker, load_category_words, masking_stats  # noqa: E402
from src.data.preprocessing import clean_text  # noqa: E402
from src.evaluation.evaluator import describe_run, save_predictions  # noqa: E402
from src.evaluation.metrics import full_evaluation, plot_confusion_matrix  # noqa: E402
from src.utils import get_logger, save_json, set_seed  # noqa: E402

logger = get_logger("baseline")


def cached_image_features(cfg: Config, split, name: str) -> np.ndarray:
    """Handcrafted image features are slow-ish to compute, so cache them to disk."""
    path = cfg.data_path / "processed" / f"baseline_image_features_{name}.npy"
    if path.exists():
        feats = np.load(path)
        if len(feats) == len(split):
            return feats
    images = (row[cfg.image_column] for row in split)
    feats = extract_image_features(images, total=len(split))
    np.save(path, feats)
    return feats


def main() -> None:
    p = argparse.ArgumentParser(description="Traditional ML baselines.")
    p.add_argument("--modes", nargs="+", default=["text", "image", "multimodal"],
                   choices=["text", "image", "multimodal"])
    p.add_argument("--mask-keywords", action="store_true",
                   help="E1: remove category words from all product names (train and test)")
    args = p.parse_args()
    if args.mask_keywords and "image" in args.modes:
        args.modes = [m for m in args.modes if m != "image"]   # images are unaffected by masking
        logger.info("Masked run: skipping the image-only baseline (identical to the unmasked one).")

    cfg = Config()
    cfg.make_dirs()
    set_seed(cfg.seed)
    splits = get_split_datasets(cfg)
    l2i = splits["label2id"]
    class_names = [splits["id2label"][i] for i in sorted(splits["id2label"])]

    masker = KeywordMasker(load_category_words(cfg)) if args.mask_keywords else None
    data = {}
    for name in ("train", "test"):
        ds = splits[name]
        texts = [clean_text(t) for t in ds[cfg.text_column]]
        if masker is not None:
            logger.info(f"Masking stats [{name}]: " + ", ".join(
                f"{k}={v:.1f}" for k, v in masking_stats(texts, masker).items()))
            texts = [masker(t) for t in texts]
        data[name] = {
            "text": texts,
            "y": np.array([l2i[c] for c in ds[cfg.label_column]]),
            "img": (cached_image_features(cfg, ds, name)
                    if any(m != "text" for m in args.modes) else None),
        }

    for mode in args.modes:
        tag = f"baseline_{mode}" + ("_masked" if args.mask_keywords else "")
        logger.info(f"Training baseline [{tag}] ...")
        model = BaselineClassifier(mode=mode, seed=cfg.seed)
        t0 = time.time()
        model.fit(data["train"]["text"], data["train"]["img"], data["train"]["y"])
        fit_s = time.time() - t0

        t0 = time.time()
        probs = model.predict_proba(data["test"]["text"], data["test"]["img"])
        pred_ms = 1000 * (time.time() - t0) / len(probs)

        result = full_evaluation(data["test"]["y"], probs.argmax(1), class_names, probs)
        result.update({"model": describe_run("Baseline", mode, args.mask_keywords),
                       "run_name": tag, "seed": cfg.seed, "mask_keywords": args.mask_keywords,
                       "family": "baseline", "mode": mode,
                       "classifier": "LogisticRegression",
                       "fit_seconds": round(fit_s, 1), "ms_per_sample": round(pred_ms, 3)})
        save_json(result, cfg.metrics_dir / f"{tag}.json")
        save_predictions(cfg, tag, data["test"]["y"], probs.argmax(1), probs)
        model.save(cfg.checkpoint_dir / f"{tag}.joblib")
        plot_confusion_matrix(np.array(result["confusion_matrix"]), class_names,
                              cfg.figures_dir / f"cm_{tag}.png",
                              title=f"{result['model']} — normalised confusion matrix (test set)")
        logger.info(f"[{result['model']}] accuracy={result['accuracy']:.4f} | "
                    f"macro-F1={result['f1_macro']:.4f} | fit {fit_s:.1f}s")


if __name__ == "__main__":
    main()

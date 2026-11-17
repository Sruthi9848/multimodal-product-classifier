"""
Classify a single product from the command line.

    python scripts/predict.py --text "Fastrack Men Black Analog Watch" --image path/to/watch.jpg
    python scripts/predict.py --text "Lakme Absolute Lip Gloss"          # text only
    python scripts/predict.py --image path/to/bag.jpg                   # image only
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

import argparse  # noqa: E402

from PIL import Image  # noqa: E402

from src.config import Config  # noqa: E402
from src.inference.predictor import ProductClassifier  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Predict the category of one product.")
    p.add_argument("--text", default=None, help="product name / description")
    p.add_argument("--image", default=None, help="path to a product photo")
    p.add_argument("--mode", choices=["multimodal", "text", "image"], default="multimodal")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()

    ckpt = Path(args.checkpoint) if args.checkpoint else Config().checkpoint_file(args.mode)
    if not ckpt.exists():
        sys.exit(f"Checkpoint not found: {ckpt}. Train a model first with scripts/train.py")

    image = Image.open(args.image) if args.image else None
    result = ProductClassifier(ckpt).predict(text=args.text, image=image, top_k=args.top_k)

    print(f"\nPredicted category : {result['label']}")
    print(f"Confidence         : {result['confidence']:.2%}")
    print(f"Inputs used        : {', '.join(result['inputs_used'])}")
    if result["text_weight"] is not None:
        print(f"Reliance on text   : {result['text_weight']:.0%}  (image: {1 - result['text_weight']:.0%})")
    print("\nTop predictions:")
    for label, prob in result["top_k"]:
        print(f"  {label:<20} {prob:6.2%}  {'█' * int(prob * 30)}")


if __name__ == "__main__":
    main()

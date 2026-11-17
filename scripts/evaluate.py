"""
Re-evaluate a trained LoRA checkpoint on the test set
(train.py already does this automatically at the end of training).

    python scripts/evaluate.py                        # multimodal checkpoint
    python scripts/evaluate.py --mode text
    python scripts/evaluate.py --checkpoint outputs/checkpoints/lora_image.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

import argparse  # noqa: E402

from src.config import Config  # noqa: E402
from src.evaluation.evaluator import evaluate_lora_checkpoint  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a LoRA checkpoint on the test split.")
    p.add_argument("--mode", default="multimodal",
                   help="mode or run tag, e.g. multimodal, text_masked, image_s7")
    p.add_argument("--checkpoint", default=None, help="path to a .pt checkpoint (overrides --mode)")
    p.add_argument("--num-workers", type=int, default=None)
    args = p.parse_args()

    ckpt = Path(args.checkpoint) if args.checkpoint else Config().checkpoint_file(args.mode)
    if not ckpt.exists():
        sys.exit(f"Checkpoint not found: {ckpt}\nTrain it first:  python scripts/train.py --mode {args.mode}")

    result = evaluate_lora_checkpoint(ckpt, num_workers=args.num_workers)
    print("\n=== Test-set results ===")
    for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "top3_accuracy"]:
        if key in result:
            print(f"{key:>18}: {result[key]:.4f}")
    print(f"\nPer-class report and confusion matrix saved under {Config().output_path}")


if __name__ == "__main__":
    main()

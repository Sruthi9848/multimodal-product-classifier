"""
STEP 4 — Compare every model that has been evaluated (LoRA + baselines).

Reads outputs/metrics/*.json (written by train.py / evaluate.py / train_baseline.py),
then writes:
    outputs/metrics/model_comparison.csv
    outputs/metrics/model_comparison.md
    outputs/figures/model_comparison.png

    python scripts/compare_models.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import Config  # noqa: E402
from src.utils import load_json  # noqa: E402

METRIC_COLS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]


def load_results(cfg: Config) -> pd.DataFrame:
    rows = []
    for path in sorted(cfg.metrics_dir.glob("*.json")):
        if not (path.name.startswith("lora_") or path.name.startswith("baseline_")):
            continue
        r = load_json(path)
        rows.append({"model": r["model"], "family": r["family"], "mode": r["mode"],
                     **{m: r.get(m) for m in METRIC_COLS},
                     "top3_accuracy": r.get("top3_accuracy"),
                     "trainable_params": r.get("trainable_params"),
                     "ms_per_sample": r.get("ms_per_sample"),
                     "test_samples": r.get("num_test_samples")})
    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame, path: Path) -> None:
    df = df.sort_values("f1_macro")
    metrics = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    y = np.arange(len(df))
    h = 0.2
    fig, ax = plt.subplots(figsize=(9, 0.9 * len(df) + 1.8))
    colors = ["#2E4A7D", "#E0A526", "#7A9CC6", "#B8C4D6"]
    for i, (m, c) in enumerate(zip(metrics, colors)):
        ax.barh(y + (i - 1.5) * h, df[m], height=h, label=m.replace("_", " "), color=c)
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"])
    lo = max(0.0, float(df[metrics].min().min()) - 0.05)
    ax.set_xlim(lo, 1.0)
    ax.set_xlabel("score on the test set")
    ax.set_title("Model comparison (higher is better)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = Config()
    cfg.make_dirs()
    df = load_results(cfg)
    if df.empty:
        sys.exit("No metrics found. Run scripts/train.py and scripts/train_baseline.py first.")
    df = df.sort_values("f1_macro", ascending=False).reset_index(drop=True)

    df.to_csv(cfg.metrics_dir / "model_comparison.csv", index=False)
    (cfg.metrics_dir / "model_comparison.md").write_text(df.to_markdown(index=False, floatfmt=".4f"))
    plot_comparison(df, cfg.figures_dir / "model_comparison.png")

    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(df[["model"] + METRIC_COLS + ["top3_accuracy"]].round(4).to_string(index=False))
    print(f"\nSaved table + chart to {cfg.metrics_dir} and {cfg.figures_dir}")


if __name__ == "__main__":
    main()

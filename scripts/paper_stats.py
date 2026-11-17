"""
E3 — Turn all finished runs into paper-ready tables and figures.

Reads  outputs/metrics/*.json  and  outputs/predictions/*.npz
Writes outputs/paper/
    main_results.{md,csv,tex}     mean ± std over seeds, bootstrap 95% CI (seed 42)
    significance.{md,csv,tex}     McNemar exact tests + Holm correction
    per_class_image_gain.md       which categories gain most from the photo (masked names)
    masking_stats.json            how much E1 masking changed the product names
    shortcut_gap.png              the key figure: original vs. masked names
    summary.md                    everything above in one file

    python scripts/paper_stats.py
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
from src.evaluation.evaluator import describe_run  # noqa: E402
from src.evaluation.statistics import accuracy, bootstrap_ci, holm_correction, macro_f1, mcnemar_exact  # noqa: E402
from src.utils import load_json, save_json  # noqa: E402

# Pairs compared with McNemar's test (files in outputs/predictions/, seed 42 runs).
PAIRS = [
    ("lora_multimodal", "lora_text"),
    ("lora_multimodal", "baseline_text"),
    ("lora_text", "baseline_text"),
    ("lora_multimodal", "baseline_multimodal"),
    ("lora_image", "baseline_image"),
    ("lora_multimodal_masked", "lora_text_masked"),
    ("lora_multimodal_masked", "baseline_text_masked"),
    ("lora_text_masked", "baseline_text_masked"),
    ("lora_multimodal_masked", "lora_image"),
]


def load_runs(cfg: Config) -> pd.DataFrame:
    rows = []
    for path in sorted(cfg.metrics_dir.glob("*.json")):
        if not path.name.startswith(("lora_", "baseline_")):
            continue
        r = load_json(path)
        if "accuracy" not in r:
            continue
        rows.append({"file": path.stem, "family": r["family"], "mode": r["mode"],
                     "masked": bool(r.get("mask_keywords", False)), "seed": int(r.get("seed", 42)),
                     "accuracy": r["accuracy"], "f1_macro": r["f1_macro"],
                     "trainable_params": r.get("trainable_params")})
    return pd.DataFrame(rows)


def main_table(runs: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = []
    for (family, mode, masked), g in runs.groupby(["family", "mode", "masked"]):
        name = describe_run("LoRA" if family == "lora" else "Baseline", mode, masked)
        row = {"Model": name, "Names": "masked" if masked else "original", "Seeds": len(g),
               "Accuracy": f"{100 * g.accuracy.mean():.2f} ± {100 * g.accuracy.std(ddof=1):.2f}"
               if len(g) > 1 else f"{100 * g.accuracy.mean():.2f}",
               "Macro-F1": f"{100 * g.f1_macro.mean():.2f} ± {100 * g.f1_macro.std(ddof=1):.2f}"
               if len(g) > 1 else f"{100 * g.f1_macro.mean():.2f}",
               "_acc": g.accuracy.mean()}
        # Bootstrap CI on the seed-42 run's per-product predictions
        f42 = g[g.seed == 42].file
        npz = cfg.predictions_dir / f"{f42.iloc[0]}.npz" if len(f42) else None
        if npz is not None and npz.exists():
            d = np.load(npz)
            lo, hi = bootstrap_ci(d["y_true"], d["y_pred"], accuracy, n_boot=2000)
            row["Acc. 95% CI (seed 42)"] = f"[{100 * lo:.2f}, {100 * hi:.2f}]"
        tp = g.trainable_params.dropna()
        row["Trainable params"] = f"{int(tp.iloc[0]):,}" if len(tp) else "—"
        out.append(row)
    df = pd.DataFrame(out).sort_values(["Names", "_acc"], ascending=[False, False]).drop(columns="_acc")
    return df.fillna("—")


def significance_table(cfg: Config) -> pd.DataFrame:
    rows = []
    for a, b in PAIRS:
        pa, pb = cfg.predictions_dir / f"{a}.npz", cfg.predictions_dir / f"{b}.npz"
        if not (pa.exists() and pb.exists()):
            continue
        da, db = np.load(pa), np.load(pb)
        if not np.array_equal(da["y_true"], db["y_true"]):
            print(f"  ! {a} vs {b}: different test sets, skipped")
            continue
        t = mcnemar_exact(da["y_true"], da["y_pred"], db["y_pred"])
        rows.append({"Model A": a, "Model B": b,
                     "Acc A": 100 * accuracy(da["y_true"], da["y_pred"]),
                     "Acc B": 100 * accuracy(db["y_true"], db["y_pred"]),
                     "A right, B wrong": t["a_right_b_wrong"], "A wrong, B right": t["a_wrong_b_right"],
                     "p": t["p_value"]})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["p (Holm)"] = holm_correction(df["p"].tolist())
        df["Significant (α=0.05)"] = np.where(df["p (Holm)"] < 0.05, "yes", "no")
    return df


def per_class_gain(cfg: Config) -> pd.DataFrame:
    a, b = cfg.metrics_dir / "lora_multimodal_masked.json", cfg.metrics_dir / "lora_text_masked.json"
    if not (a.exists() and b.exists()):
        return pd.DataFrame()
    ma, mb = load_json(a), load_json(b)
    rows = [{"Category": c, "F1 text only": mb["per_class"][c]["f1-score"],
             "F1 text + photo": ma["per_class"][c]["f1-score"],
             "Gain from photo": ma["per_class"][c]["f1-score"] - mb["per_class"][c]["f1-score"],
             "Test items": int(ma["per_class"][c]["support"])} for c in ma["class_names"]]
    return pd.DataFrame(rows).sort_values("Gain from photo", ascending=False)


def masking_report(cfg: Config) -> dict:
    try:
        from src.data.dataset_loader import get_split_datasets
        from src.data.keywords import KeywordMasker, load_category_words, masking_stats
        from src.data.preprocessing import clean_text
        test = get_split_datasets(cfg)["test"]
        masker = KeywordMasker(load_category_words(cfg))
        texts = [clean_text(t) for t in test[cfg.text_column]]
        stats = masking_stats(texts, masker)
        stats["examples"] = [{"original": t, "masked": masker(t)} for t in texts[:: max(1, len(texts) // 8)][:8]]
        stats["num_category_words"] = len(load_category_words(cfg))
        return stats
    except Exception as err:  # noqa: BLE001 - optional extra
        return {"error": str(err)}


def shortcut_figure(runs: pd.DataFrame, path: Path) -> None:
    groups = [("baseline", "text", "Baseline text\n(TF-IDF)"), ("lora", "text", "LoRA text\n(DistilBERT)"),
              ("baseline", "multimodal", "Baseline\ntext + image"), ("lora", "multimodal", "LoRA\nmultimodal")]
    labels, orig, orig_sd, mask, mask_sd = [], [], [], [], []
    for fam, mode, label in groups:
        g = runs[(runs.family == fam) & (runs["mode"] == mode)]
        if g.empty:
            continue
        o, m = g[~g.masked].accuracy, g[g.masked].accuracy
        labels.append(label)
        orig.append(100 * o.mean() if len(o) else np.nan)
        orig_sd.append(100 * o.std(ddof=1) if len(o) > 1 else 0)
        mask.append(100 * m.mean() if len(m) else np.nan)
        mask_sd.append(100 * m.std(ddof=1) if len(m) > 1 else 0)
    if not labels:
        return
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 5))
    b1 = ax.bar(x - 0.19, orig, 0.38, yerr=orig_sd, capsize=3, color="#1E2B4A", label="original names")
    b2 = ax.bar(x + 0.19, mask, 0.38, yerr=mask_sd, capsize=3, color="#E0A526", label="category words removed")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    values = [v for v in orig + mask if np.isfinite(v)]
    img = runs[(runs.family == "lora") & (runs["mode"] == "image")].accuracy
    if len(img):
        img_acc = 100 * img.mean()
        values.append(img_acc)
        ax.axhline(img_acc, ls="--", color="#5B6B86", lw=1.2, label=f"LoRA image only ({img_acc:.1f})")
    ax.set_ylim(max(0, min(values) - 8), 106)
    ax.set_yticks([t for t in ax.get_yticks() if t <= 100])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("test accuracy (%), mean ± std over seeds")
    ax.set_title("Text shortcut: accuracy with and without category words in the name", pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write(df: pd.DataFrame, out: Path, name: str, floatfmt: str = ".4f") -> str:
    if df.empty:
        return f"_{name}: no data yet_\n"
    df.to_csv(out / f"{name}.csv", index=False)
    md = df.to_markdown(index=False, floatfmt=floatfmt)
    (out / f"{name}.md").write_text(md)
    try:
        (out / f"{name}.tex").write_text(df.to_latex(index=False, float_format="%.4f", escape=True))
    except Exception:  # noqa: BLE001 - LaTeX export is a bonus
        pass
    return md


def main() -> None:
    cfg = Config()
    out = cfg.output_path / "paper"
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(cfg)
    if runs.empty:
        sys.exit("No results found. Run the experiments first.")

    main_md = write(main_table(runs, cfg), out, "main_results")
    sig_md = write(significance_table(cfg), out, "significance", floatfmt=".4g")
    gain_md = write(per_class_gain(cfg), out, "per_class_image_gain", floatfmt=".3f")
    mask_stats = masking_report(cfg)
    save_json(mask_stats, out / "masking_stats.json")
    shortcut_figure(runs, out / "shortcut_gap.png")

    robust = cfg.metrics_dir / "robustness.md"
    summary = ["# Paper results summary\n",
               "## Main results (test set)\n", main_md,
               "\n\n## Significance (McNemar exact, Holm-corrected)\n", sig_md,
               "\n\n## E1 masking statistics (test names)\n",
               "\n".join(f"- {k}: {v:.2f}" for k, v in mask_stats.items() if isinstance(v, float)),
               "\n\n## Per-class gain from the photo (masked names, seed 42)\n", gain_md,
               "\n\n## E2 robustness\n", robust.read_text() if robust.exists() else "_run scripts/robustness.py_"]
    (out / "summary.md").write_text("\n".join(summary))
    print(main_md)
    print("\n" + sig_md)
    print(f"\nAll paper tables and figures saved to {out}")


if __name__ == "__main__":
    main()

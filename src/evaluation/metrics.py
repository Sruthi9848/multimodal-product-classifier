"""
Evaluation metrics and plots.

For each class c (with TP, FP, FN counted on the test set):
    precision_c = TP / (TP + FP)      "when we say c, how often are we right?"
    recall_c    = TP / (TP + FN)      "of all real c products, how many did we find?"
    F1_c        = 2 · P · R / (P + R) harmonic mean of the two

    macro avg    = plain mean over classes   (every class counts equally -> fair to small classes)
    weighted avg = mean weighted by class size
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # render to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,  # noqa: E402
                             precision_recall_fscore_support, top_k_accuracy_score)


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    """Headline metrics used during training and for model comparison."""
    out = {"accuracy": float(accuracy_score(y_true, y_pred))}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        out[f"precision_{avg}"] = float(p)
        out[f"recall_{avg}"] = float(r)
        out[f"f1_{avg}"] = float(f)
    return out


def full_evaluation(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str],
                    probs: Optional[np.ndarray] = None) -> Dict:
    """Headline metrics + per-class report + confusion matrix (+ top-3 accuracy if probs given)."""
    labels = list(range(len(class_names)))
    result = compute_metrics(y_true, y_pred)
    if probs is not None and probs.shape[1] >= 3:
        result["top3_accuracy"] = float(top_k_accuracy_score(y_true, probs, k=3, labels=labels))
    result["per_class"] = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0)
    result["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    result["class_names"] = class_names
    result["num_test_samples"] = int(len(y_true))
    return result


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], path: Path,
                          title: str = "Confusion matrix", normalize: bool = True) -> None:
    """Row-normalised heat-map: each row shows where products of that TRUE class went."""
    cm = np.asarray(cm, dtype=float)
    if normalize:
        cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    n = len(class_names)
    size = max(6, 0.45 * n + 3)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(cm, annot=n <= 25, fmt=".2f" if normalize else "g", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, cbar=True,
                annot_kws={"size": 7}, ax=ax, vmin=0, vmax=1 if normalize else None)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_history(history: List[Dict], path: Path, title: str = "Training history") -> None:
    """Loss and accuracy/F1 curves per epoch."""
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], "o-", label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], "o-", label="validation")
    axes[0].set_title("Loss")
    axes[1].plot(epochs, [h["train_acc"] for h in history], "o-", label="train accuracy")
    axes[1].plot(epochs, [h["val_acc"] for h in history], "o-", label="val accuracy")
    axes[1].plot(epochs, [h["val_f1_macro"] for h in history], "s--", label="val macro-F1")
    axes[1].set_title("Accuracy / F1")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)

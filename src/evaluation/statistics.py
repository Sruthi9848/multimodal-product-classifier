"""
Statistics for comparing models honestly (E3).

1. Mean ± standard deviation over training seeds
   Retraining with a different seed changes weight init, data order and dropout.
   If model A beats B by less than the seed-to-seed spread, the "win" is noise.

2. McNemar's exact test (paired: both models see the SAME test products)
   For every test product, compare the two models:

                         B correct     B wrong
        A correct           n11           b
        A wrong              c           n00

   Only the disagreements b and c carry information. Under H0 ("equally good"),
   each disagreement is a fair coin flip:  b ~ Binomial(b + c, 1/2).
   p = two-sided exact binomial p-value.

3. Holm–Bonferroni correction
   Testing many pairs inflates false positives. Sort p-values p_(1) ≤ … ≤ p_(m);
   adjusted p_(i) = max_{j ≤ i} min(1, (m − j + 1) · p_(j)).

4. Bootstrap 95 % confidence interval
   Resample the n test products with replacement B times, recompute the metric,
   and take the 2.5th and 97.5th percentiles.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score


def mcnemar_exact(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, float]:
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    b = int(np.sum(a_ok & ~b_ok))   # A right, B wrong
    c = int(np.sum(~a_ok & b_ok))   # A wrong, B right
    p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "p_value": p}


def holm_correction(p_values: List[float]) -> List[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, metric: Callable, n_boot: int = 1000,
                 seed: int = 0, alpha: float = 0.05) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = [metric(y_true[idx], y_pred[idx]) for idx in (rng.integers(0, n, n) for _ in range(n_boot))]
    return float(np.percentile(stats, 100 * alpha / 2)), float(np.percentile(stats, 100 * (1 - alpha / 2)))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

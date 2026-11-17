"""Unit tests for the paper-experiment helpers (offline, fast)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.config import Config  # noqa: E402
from src.data.keywords import (Compose, KeywordMasker, WordDropper, category_words_from_names,  # noqa: E402
                               masking_stats)
from src.evaluation.statistics import bootstrap_ci, accuracy, holm_correction, mcnemar_exact  # noqa: E402


def test_category_words_and_masking():
    words = category_words_from_names(["Watches", "Casual Shoes", "Tshirts", "Flip Flops", "Lip Gloss",
                                       "Loungewear and Nightwear"])
    assert {"watch", "watches", "shoe", "shoes", "tshirt", "flip", "flops", "gloss"} <= set(words)
    assert "and" not in words
    m = KeywordMasker(words)
    assert m("titan women silver analog watch") == "titan women silver analog"
    assert m("puma men black casual shoes") == "puma men black"
    assert m("roadster men t-shirt") == "roadster men"          # hyphen variant is caught
    stats = masking_stats(["titan watch", "nike bag"], m)
    assert stats["names_changed_pct"] == 50.0


def test_word_dropper_is_deterministic():
    d = WordDropper(p=0.5)
    text = "one two three four five six seven eight"
    assert d(text) == d(text)
    assert len(d(text).split()) < 8
    assert Compose(None, lambda t: t.upper())("ab") == "AB"


def test_run_tags():
    assert Config().run_tag == "multimodal"
    assert Config(mode="text", mask_keywords=True).run_tag == "text_masked"
    assert Config(mode="image", seed=7).run_tag == "image_s7"
    assert Config(mode="text", seed=7).checkpoint_file().name == "lora_text_s7.pt"


def test_mcnemar_holm_bootstrap():
    y = np.zeros(100, dtype=int)
    a = np.zeros(100, dtype=int)            # always right
    b = np.zeros(100, dtype=int); b[:20] = 1  # wrong on 20
    t = mcnemar_exact(y, a, b)
    assert t["a_right_b_wrong"] == 20 and t["a_wrong_b_right"] == 0 and t["p_value"] < 1e-5
    assert mcnemar_exact(y, a, a)["p_value"] == 1.0
    adj = holm_correction([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])
    lo, hi = bootstrap_ci(y, b, accuracy, n_boot=300)
    assert lo < 0.8 < hi

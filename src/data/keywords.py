"""
Text transforms for the paper experiments.

E1 — Category-word masking
--------------------------
Product names often contain the answer: "Titan Women Silver Analog **Watch**".
To test whether a model really understands the product (instead of spotting one
word), we delete every word that names a category from ALL product names.

The list of category words V is built automatically from the dataset itself:
the words of every kept `subCategory` name (e.g. "Flip Flops") and of every
`articleType` inside it (e.g. "Casual Shoes", "Tshirts", "Lip Gloss"), plus
simple singular/plural variants. A name x = (w_1, ..., w_n) becomes

        mask(x) = ( w_i  for i in 1..n  if  norm(w_i) ∉ V )

IMPORTANT: V is the union over ALL classes and is applied to every product the
same way. Masking only the TRUE class's words would leak the label through the
pattern of removed words.

E2 — Word drop
--------------
Simulates short / incomplete seller titles: each word is kept with probability
1 − p. The random choice depends only on the text itself (CRC32 hash), so every
model sees exactly the same corrupted names.
"""

from __future__ import annotations

import random
import re
import zlib
from typing import Dict, Iterable, List, Set

from src.config import Config
from src.utils import get_logger, load_json, save_json

logger = get_logger(__name__)

# Words that appear in category names but describe *who* or are filler, not *what*.
_STOPWORDS = {"and", "or", "of", "the", "for", "with", "men", "women", "boys", "girls",
              "kids", "unisex", "set", "other", "others"}


def _norm(word: str) -> str:
    """Normalise a token for matching: lower-case, no hyphens ('t-shirt' -> 'tshirt')."""
    return word.lower().replace("-", "")


def _variants(word: str) -> Set[str]:
    """Singular and plural forms using simple English rules.

    'shoes' -> {shoes, shoe}   'watches' -> {watches, watch}   'watch' -> {watch, watches}
    'accessories' -> {accessories, accessory}   'dress' -> {dress, dresses}
    """
    w = _norm(word)
    out = {w}
    sibilant = ("s", "x", "z", "ch", "sh")
    # plural -> singular
    if w.endswith("ies") and len(w) > 4:
        out.add(w[:-3] + "y")
    elif w.endswith("es") and w[:-2].endswith(sibilant):
        out.add(w[:-2])
    elif w.endswith("s") and not w.endswith("ss"):
        out.add(w[:-1])
    # singular -> plural (applied to every singular form found so far)
    for base in list(out):
        if base.endswith("s") and not base.endswith("ss"):
            continue                                   # already plural
        if base.endswith(sibilant):
            out.add(base + "es")
        elif base.endswith("y") and len(base) > 3 and base[-2] not in "aeiou":
            out.add(base[:-1] + "ies")
        else:
            out.add(base + "s")
    return {v for v in out if len(v) >= 3}


def category_words_from_names(names: Iterable[str]) -> List[str]:
    vocab: Set[str] = set()
    for name in names:
        for word in re.split(r"[^A-Za-z\-]+", str(name)):
            if len(word) >= 3 and word.lower() not in _STOPWORDS:
                vocab |= _variants(word)
    return sorted(vocab)


def build_category_words(cfg: Config, ds, classes: List[str]) -> List[str]:
    """Collect category words from subCategory + articleType names and save them."""
    keep = set(classes)
    pairs = zip(ds[cfg.label_column], ds["articleType"]) if "articleType" in ds.column_names \
        else ((c, c) for c in ds[cfg.label_column])
    names = set(classes)
    for sub, art in pairs:
        if sub in keep and art:
            names.add(art)
    vocab = category_words_from_names(names)
    save_json({"source_names": sorted(names), "words": vocab}, cfg.keywords_file)
    logger.info(f"Built {len(vocab)} category words from {len(names)} category names "
                f"-> {cfg.keywords_file}")
    return vocab


def load_category_words(cfg: Config) -> List[str]:
    return load_json(cfg.keywords_file)["words"]


# ---------------------------------------------------------------------- #
# Transforms (classes, not lambdas, so DataLoader workers can pickle them)
# ---------------------------------------------------------------------- #
class KeywordMasker:
    """Remove every category word from an (already cleaned) product name."""

    def __init__(self, words: Iterable[str]):
        self.words = set(words)

    def __call__(self, text: str) -> str:
        return " ".join(t for t in text.split() if _norm(t) not in self.words)


class WordDropper:
    """Drop each word with probability p (deterministic per text)."""

    def __init__(self, p: float = 0.5, seed: int = 0):
        self.p, self.seed = p, seed

    def __call__(self, text: str) -> str:
        rng = random.Random(zlib.crc32(text.encode("utf-8")) + self.seed)
        return " ".join(t for t in text.split() if rng.random() >= self.p)


class Compose:
    """Apply several text transforms in order."""

    def __init__(self, *fns):
        self.fns = [f for f in fns if f is not None]

    def __call__(self, text: str) -> str:
        for f in self.fns:
            text = f(text)
        return text


def masking_stats(texts: List[str], masker: KeywordMasker) -> Dict[str, float]:
    """How strongly does masking change the names? (Report this in the paper.)"""
    changed = removed = empty = total_words = 0
    for t in texts:
        before = t.split()
        after = masker(t).split()
        total_words += len(before)
        removed += len(before) - len(after)
        changed += len(after) < len(before)
        empty += len(after) == 0
    n = max(len(texts), 1)
    return {"names_changed_pct": 100 * changed / n,
            "words_removed_pct": 100 * removed / max(total_words, 1),
            "names_empty_pct": 100 * empty / n,
            "avg_words_before": total_words / n,
            "avg_words_after": (total_words - removed) / n}

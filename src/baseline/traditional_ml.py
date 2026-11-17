"""
Traditional machine-learning baseline (no deep learning).

Text features : TF-IDF on word 1-2-grams + character 3-5-grams of the product name.
Image features: handcrafted —
                  • 16-bin colour histogram per RGB channel        (48 values)
                  • 16-bin hue / saturation histograms in HSV      (32 values)
                  • 24×24 grayscale thumbnail (rough shape info)   (576 values)
Classifier    : multinomial Logistic Regression.

Why a baseline? A fancy model is only worth it if it clearly beats a cheap,
fast, explainable one. Comparing against this keeps the project honest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import joblib
import numpy as np
from PIL import Image
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from src.data.preprocessing import clean_text, to_rgb


# ---------------------------------------------------------------------- #
# Handcrafted image features
# ---------------------------------------------------------------------- #
def image_to_features(image: Image.Image) -> np.ndarray:
    img = to_rgb(image).resize((64, 64))
    rgb = np.asarray(img, dtype=np.float32) / 255.0                       # [64, 64, 3]

    color_hist = [np.histogram(rgb[..., c], bins=16, range=(0, 1), density=True)[0] for c in range(3)]
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    hue_hist = np.histogram(hsv[..., 0], bins=16, range=(0, 1), density=True)[0]
    sat_hist = np.histogram(hsv[..., 1], bins=16, range=(0, 1), density=True)[0]
    thumb = np.asarray(img.convert("L").resize((24, 24)), dtype=np.float32).ravel() / 255.0

    return np.concatenate(color_hist + [hue_hist, sat_hist, thumb]).astype(np.float32)


def extract_image_features(images: Iterable[Image.Image], total: Optional[int] = None) -> np.ndarray:
    return np.stack([image_to_features(im) for im in tqdm(images, total=total, desc="image features")])


# ---------------------------------------------------------------------- #
# Baseline model
# ---------------------------------------------------------------------- #
class BaselineClassifier:
    """mode: 'text', 'image' or 'multimodal' (text + image features side by side)."""

    def __init__(self, mode: str = "multimodal", C: float = 5.0, seed: int = 42):
        assert mode in ("text", "image", "multimodal")
        self.mode = mode
        self.text_vec = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
        ])
        self.img_scaler = StandardScaler()
        self.clf = LogisticRegression(C=C, max_iter=3000, random_state=seed)

    def _features(self, texts: Optional[List[str]], img_feats: Optional[np.ndarray], fit: bool):
        blocks = []
        if self.mode in ("text", "multimodal"):
            texts = [clean_text(t) for t in texts]
            blocks.append(self.text_vec.fit_transform(texts) if fit else self.text_vec.transform(texts))
        if self.mode in ("image", "multimodal"):
            scaled = self.img_scaler.fit_transform(img_feats) if fit else self.img_scaler.transform(img_feats)
            # Down-weight the dense image block so it doesn't swamp the sparse text block.
            weight = 0.3 if self.mode == "multimodal" else 1.0
            blocks.append(sparse.csr_matrix(scaled * weight))
        return sparse.hstack(blocks).tocsr()

    def fit(self, texts, img_feats, y) -> "BaselineClassifier":
        self.clf.fit(self._features(texts, img_feats, fit=True), y)
        return self

    def predict_proba(self, texts, img_feats) -> np.ndarray:
        return self.clf.predict_proba(self._features(texts, img_feats, fit=False))

    def predict(self, texts, img_feats) -> np.ndarray:
        return self.predict_proba(texts, img_feats).argmax(axis=1)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "BaselineClassifier":
        return joblib.load(path)

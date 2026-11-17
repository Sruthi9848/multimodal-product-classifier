"""
Prediction on new products — used by `scripts/predict.py` and the Streamlit app.

Example
-------
    from PIL import Image
    from src.inference.predictor import ProductClassifier

    clf = ProductClassifier("outputs/checkpoints/lora_multimodal.pt")
    result = clf.predict(text="Nike Men Blue Running Shoes", image=Image.open("shoe.jpg"))
    print(result["label"], result["confidence"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoTokenizer

from src.data.preprocessing import build_image_transforms, clean_text, get_image_norm_stats
from src.models.multimodal_model import load_model_from_checkpoint
from src.utils import get_device


class ProductClassifier:
    def __init__(self, checkpoint_path: str | Path, device: Optional[torch.device] = None):
        self.device = device or get_device()
        self.model, self.cfg, self.id2label, self.meta = load_model_from_checkpoint(
            Path(checkpoint_path), self.device)
        self.mode = self.cfg.mode
        self.tokenizer = (AutoTokenizer.from_pretrained(self.cfg.text_model_name)
                          if self.mode in ("multimodal", "text") else None)
        # A model trained on masked names (E1) must also see masked names at prediction time.
        self.text_transform = None
        if self.cfg.mask_keywords and self.cfg.keywords_file.exists():
            from src.data.keywords import KeywordMasker, load_category_words
            self.text_transform = KeywordMasker(load_category_words(self.cfg))
        mean, std = get_image_norm_stats(self.cfg.image_model_name)
        self.image_tf = build_image_transforms(self.cfg.image_size, mean, std, train=False)

    @property
    def class_names(self) -> List[str]:
        return [self.id2label[i] for i in sorted(self.id2label)]

    @torch.no_grad()
    def predict(self, text: Optional[str] = None, image: Optional[Image.Image] = None,
                top_k: int = 5) -> Dict:
        """
        Returns:
            {
              "label": "Shoes",
              "confidence": 0.97,                       # softmax probability of the top class
              "top_k": [("Shoes", 0.97), ("Sandal", 0.02), ...],
              "text_weight": 0.63 or None,              # gated fusion only: reliance on text
              "inputs_used": ["text", "image"],
            }
        """
        text = clean_text(text) if text else ""
        if text and self.text_transform is not None:
            text = self.text_transform(text)
        use_text = bool(text) and self.mode in ("multimodal", "text")
        use_image = image is not None and self.mode in ("multimodal", "image")

        if self.mode == "text" and not use_text:
            raise ValueError("This model is text-only: please enter a product name.")
        if self.mode == "image" and not use_image:
            raise ValueError("This model is image-only: please upload a product image.")
        if not (use_text or use_image):
            raise ValueError("Please enter a product name, upload an image, or both.")

        inputs = {}
        if use_text:
            enc = self.tokenizer([text], truncation=True, max_length=self.cfg.max_text_length,
                                 return_tensors="pt")
            inputs["input_ids"] = enc["input_ids"].to(self.device)
            inputs["attention_mask"] = enc["attention_mask"].to(self.device)
        if use_image:
            inputs["pixel_values"] = self.image_tf(image).unsqueeze(0).to(self.device)

        out = self.model(**inputs)
        probs = torch.softmax(out["logits"].float(), dim=-1)[0].cpu()
        k = min(top_k, probs.numel())
        top_p, top_i = probs.topk(k)
        top = [(self.id2label[int(i)], float(p)) for p, i in zip(top_p, top_i)]

        text_weight = None
        if out["text_weight"] is not None and use_text and use_image:
            text_weight = float(out["text_weight"][0])

        return {
            "label": top[0][0],
            "confidence": top[0][1],
            "top_k": top,
            "text_weight": text_weight,
            "inputs_used": [m for m, used in (("text", use_text), ("image", use_image)) if used],
        }

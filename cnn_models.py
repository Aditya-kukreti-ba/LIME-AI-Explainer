"""
cnn_models.py
Pre-trained torchvision CNN models for LIME image explanation.

Supports: ResNet50, MobileNetV2, VGG16
All run on CPU – no GPU needed.
"""

from __future__ import annotations
import json, urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ── ImageNet class labels ─────────────────────────────────────────────────────
# Fetched once and cached locally as imagenet_classes.json
_LABELS_URL  = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
_LABELS_FILE = Path(__file__).parent / "imagenet_classes.json"
_labels_cache: list[str] | None = None

def get_imagenet_labels() -> list[str]:
    global _labels_cache
    if _labels_cache is not None:
        return _labels_cache

    if _LABELS_FILE.exists():
        with open(_LABELS_FILE) as f:
            _labels_cache = json.load(f)
    else:
        print("[cnn_models] Downloading ImageNet class labels…")
        try:
            with urllib.request.urlopen(_LABELS_URL, timeout=10) as r:
                _labels_cache = json.loads(r.read())
            with open(_LABELS_FILE, "w") as f:
                json.dump(_labels_cache, f)
        except Exception:
            # Fallback: numeric labels
            _labels_cache = [f"class_{i}" for i in range(1000)]

    return _labels_cache


# ── Standard ImageNet pre-processing ─────────────────────────────────────────
_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ── Model registry ────────────────────────────────────────────────────────────
CNN_MODEL_INFO = [
    {
        "id":          "resnet50",
        "name":        "ResNet-50  (ImageNet)",
        "description": "Classic deep residual network. Great all-rounder, ~25M params.",
    },
    {
        "id":          "mobilenet_v2",
        "name":        "MobileNetV2  (ImageNet)",
        "description": "Lightweight model optimised for speed. ~3.4M params.",
    },
    {
        "id":          "vgg16",
        "name":        "VGG-16  (ImageNet)",
        "description": "Classic very deep network. Slower but highly interpretable.",
    },
]

_model_cache: dict = {}


def _load_model(model_id: str):
    """Return (model_eval, preprocess_fn)."""
    if model_id in _model_cache:
        return _model_cache[model_id]

    print(f"[cnn_models] Loading {model_id}…")
    if model_id == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    elif model_id == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    elif model_id == "vgg16":
        m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unknown CNN model: {model_id!r}")

    m.eval()
    _model_cache[model_id] = m
    return m


# ── Public helpers ────────────────────────────────────────────────────────────

def predict_single(model_id: str, img_array: np.ndarray) -> tuple[str, float, list[tuple[str, float]]]:
    """
    Predict the top ImageNet class for a (H, W, 3) uint8 numpy image.
    Returns (top_class_name, top_prob, top5_list)
    """
    labels = get_imagenet_labels()
    model  = _load_model(model_id)

    pil_img = Image.fromarray(img_array)
    tensor  = _preprocess(pil_img).unsqueeze(0)   # (1, 3, 224, 224)

    with torch.no_grad():
        logits = model(tensor)
    probs = torch.softmax(logits[0], dim=0).numpy()

    top5_idx   = probs.argsort()[::-1][:5]
    top5       = [(labels[i], float(probs[i])) for i in top5_idx]
    top_name   = top5[0][0]
    top_prob   = top5[0][1]
    return top_name, top_prob, top5


def make_lime_predict_fn(model_id: str):
    """
    Return a predict_fn(images: np.ndarray) -> np.ndarray
    where images is (N, H, W, 3) uint8 and output is (N, 1000).
    This is what LimeImageExplainer calls.
    """
    model = _load_model(model_id)

    def predict_fn(images: np.ndarray) -> np.ndarray:
        tensors = []
        for img in images:
            pil = Image.fromarray(img.astype(np.uint8))
            tensors.append(_preprocess(pil))
        batch = torch.stack(tensors)          # (N, 3, 224, 224)
        with torch.no_grad():
            logits = model(batch)
        probs = torch.softmax(logits, dim=1).numpy()
        return probs

    return predict_fn

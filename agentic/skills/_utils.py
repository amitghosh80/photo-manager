from __future__ import annotations

import base64
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_cv2(image_path: Path):
    """Load image via OpenCV, falling back to Pillow for HEIC/HEIF."""
    img = cv2.imread(str(image_path))
    if img is not None:
        return img
    try:
        pil = Image.open(image_path).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def encode_image(image_path: Path, max_side: int = 1024) -> tuple[str, str]:
    """Return (base64_data, media_type) resized to max_side for the Anthropic API."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode(), "image/jpeg"

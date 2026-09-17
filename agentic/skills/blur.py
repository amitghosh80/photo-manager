from __future__ import annotations

import cv2
from pathlib import Path

from .base import Skill
from ._utils import load_cv2


class BlurSkill(Skill):
    name = "blur_detection"
    version = "1"
    description = "Measure image sharpness via Laplacian variance."

    def run(self, image_path: Path, **_) -> dict:
        img = load_cv2(image_path)
        if img is None:
            return {"sharpness_score": 0.0, "is_blurry": True, "laplacian_variance": 0.0}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        score = min(10.0, lap_var / 300.0)
        return {
            "sharpness_score": round(score, 3),
            "is_blurry": score < 3.0,
            "laplacian_variance": round(lap_var, 1),
        }

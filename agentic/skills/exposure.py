from __future__ import annotations

import cv2
from pathlib import Path

from .base import Skill
from ._utils import load_cv2


class ExposureSkill(Skill):
    name = "exposure_analysis"
    version = "1"
    description = "Analyse exposure quality via pixel histogram clipping."

    def run(self, image_path: Path, **_) -> dict:
        img = load_cv2(image_path)
        if img is None:
            return {"exposure_score": 5.0, "is_overexposed": False,
                    "is_underexposed": False, "shadow_clipping": 0.0, "highlight_clipping": 0.0}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist /= hist.sum()
        shadow = float(hist[:15].sum())
        highlight = float(hist[240:].sum())
        score = max(0.0, 10.0 - (shadow + highlight) * 20.0)
        return {
            "exposure_score": round(score, 3),
            "is_underexposed": shadow > 0.1,
            "is_overexposed": highlight > 0.1,
            "shadow_clipping": round(shadow, 4),
            "highlight_clipping": round(highlight, 4),
        }

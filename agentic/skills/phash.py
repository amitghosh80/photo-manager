from __future__ import annotations

from pathlib import Path

import imagehash
from PIL import Image

from .base import Skill


class PhashSkill(Skill):
    name = "phash"
    version = "1"
    description = "Compute perceptual hash for near-duplicate detection."

    def run(self, image_path: Path, **_) -> dict:
        try:
            with Image.open(image_path) as img:
                h = imagehash.phash(img)
            return {"hash": str(h), "ok": True}
        except Exception as exc:
            return {"hash": None, "ok": False, "error": str(exc)}

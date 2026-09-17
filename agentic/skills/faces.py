from __future__ import annotations

import cv2
from pathlib import Path

from .base import Skill
from ._utils import load_cv2


class FaceSkill(Skill):
    name = "face_detection"
    version = "1"
    description = "Detect faces using OpenCV Haar cascade."

    def __init__(self) -> None:
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def run(self, image_path: Path, **_) -> dict:
        img = load_cv2(image_path)
        if img is None:
            return {"face_count": 0, "has_faces": False, "largest_face_fraction": 0.0, "faces": []}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        detected = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if not len(detected):
            return {"face_count": 0, "has_faces": False, "largest_face_fraction": 0.0, "faces": []}
        img_area = h * w
        faces = sorted(
            [{"x": int(x), "y": int(y), "w": int(fw), "h": int(fh),
              "area_fraction": round(fw * fh / img_area, 4)}
             for x, y, fw, fh in detected],
            key=lambda f: f["area_fraction"],
            reverse=True,
        )
        return {
            "face_count": len(faces),
            "has_faces": True,
            "largest_face_fraction": faces[0]["area_fraction"],
            "faces": faces,
        }

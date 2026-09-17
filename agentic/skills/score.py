from __future__ import annotations

import json
from pathlib import Path

from .base import Skill
from ._utils import encode_image

_SYSTEM = """\
You are an expert photo curator with deep knowledge of photography composition, technical quality, and artistic merit.

Evaluate the supplied photo on exactly three dimensions (each scored 0.0–10.0):

• composition     – Rule of thirds, leading lines, framing, balance, visual flow
• subject_clarity – Sharpness on main subject, subject prominence, separation from background
• uniqueness      – Creative perspective, unusual angle, memorable moment, distinctiveness

For portrait photos (people as main subject), also assess portrait_quality:
  expression:       0.0–10.0  (naturalness and positivity of facial expression)
  eyes_open:        0.0–10.0  (10 = both eyes fully open and sharply in focus)
  subject_complete: 0.0–10.0  (10 = full subject in frame, nothing cut off)
Set portrait_quality to null for non-portrait photos.

Return ONLY a JSON object — no markdown fences, no extra text:
{
  "composition": <float>,
  "subject_clarity": <float>,
  "uniqueness": <float>,
  "portrait_quality": {"expression": <float>, "eyes_open": <float>, "subject_complete": <float>} | null,
  "brief_reason": "<one concise sentence>"
}"""

_DEFAULT = {
    "composition": 5.0,
    "subject_clarity": 5.0,
    "uniqueness": 5.0,
    "portrait_quality": None,
    "brief_reason": "Scoring unavailable — using defaults.",
}


class AestheticScoreSkill(Skill):
    name = "aesthetic_score"
    version = "1"
    description = "Score photo aesthetics: composition, clarity, uniqueness via Claude Sonnet."

    def run(self, image_path: Path, client=None, **_) -> dict:
        if client is None:
            return dict(_DEFAULT)
        try:
            b64, mime = encode_image(image_path)
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=384,
                system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}],
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                        {"type": "text", "text": "Score this photo."},
                    ],
                }],
            )
            text = resp.content[0].text.strip()
            if text.startswith("```"):
                text = "\n".join(l for l in text.splitlines()[1:] if not l.strip().startswith("```"))
            return json.loads(text)
        except Exception:
            return dict(_DEFAULT)

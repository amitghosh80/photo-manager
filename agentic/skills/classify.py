from __future__ import annotations

import json
from pathlib import Path

from .base import Skill
from ._utils import encode_image

_SYSTEM = """\
You are an expert travel photographer and photo editor.

Classify the supplied photo across these dimensions:

• scene_id           – A 1-3 word snake_case label for the dominant scene/location \
(e.g. "hotel_lobby", "mountain_trail"). Use the SAME label for photos that share a visually similar location.
• subject_type       – Exactly one of: landscape, portrait, architecture, wildlife, street, food, abstract, other
• narrative_category – Exactly one of: arrival, exploration, activity, meal, portrait, landmark, candid, atmosphere, departure, other
• emotional_tone     – Exactly one of: joyful, serene, dramatic, contemplative, energetic, intimate, playful
• setting            – Exactly one of: outdoor, indoor, urban, nature, restaurant, transit, accommodation

Return ONLY a JSON object — no markdown fences, no extra text:
{
  "scene_id": "<string>",
  "subject_type": "<string>",
  "narrative_category": "<string>",
  "emotional_tone": "<string>",
  "setting": "<string>"
}"""

_DEFAULT = {
    "scene_id": "unknown",
    "subject_type": "other",
    "narrative_category": "other",
    "emotional_tone": "serene",
    "setting": "outdoor",
}


class ClassifySkill(Skill):
    name = "narrative_classify"
    version = "1"
    description = "Classify photo: scene, subject type, narrative category, emotional tone, and setting via Claude Haiku."

    def run(self, image_path: Path, client=None, **_) -> dict:
        if client is None:
            return dict(_DEFAULT)
        try:
            b64, mime = encode_image(image_path)
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}],
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                        {"type": "text", "text": "Classify this photo."},
                    ],
                }],
            )
            text = resp.content[0].text.strip()
            if text.startswith("```"):
                text = "\n".join(l for l in text.splitlines()[1:] if not l.strip().startswith("```"))
            return json.loads(text)
        except Exception:
            return dict(_DEFAULT)

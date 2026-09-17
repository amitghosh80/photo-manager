"""ScannerAgent — runs all skills on every photo and builds a manifest."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..ingest import load_images, get_photo_datetime
from ..cache import SkillCache
from ..skills import BlurSkill, ExposureSkill, FaceSkill, PhashSkill, ClassifySkill, AestheticScoreSkill

_LOCAL_SKILLS = [BlurSkill(), ExposureSkill(), FaceSkill(), PhashSkill()]
_CLAUDE_SKILLS = [ClassifySkill(), AestheticScoreSkill()]

_AI_WEIGHTS = {"composition": 0.35, "subject_clarity": 0.35, "uniqueness": 0.30}


def _aggregate(photo: dict) -> float:
    blur = photo.get("blur_detection", {}).get("sharpness_score", 5.0)
    exp  = photo.get("exposure_analysis", {}).get("exposure_score", 5.0)
    tech = blur * 0.6 + exp * 0.4

    score = photo.get("aesthetic_score", {})
    ai = sum(score.get(k, 5.0) * w for k, w in _AI_WEIGHTS.items())
    base = tech * 0.25 + ai * 0.75

    pq = score.get("portrait_quality")
    if isinstance(pq, dict):
        avg = (pq.get("expression", 5.0) + pq.get("eyes_open", 5.0) + pq.get("subject_complete", 5.0)) / 3.0
        base = min(10.0, base + (avg - 5.0) * 0.03)

    return round(base, 4)


class ScannerAgent:
    """Runs every skill on every photo and returns a manifest list."""

    def scan(
        self,
        trip_dir: Path,
        client,
        cache: SkillCache,
        on_progress: Callable[[int, int, str, str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[dict]:
        images = load_images(trip_dir)
        manifest: list[dict] = []

        for i, path in enumerate(images, start=1):
            if should_stop and should_stop():
                raise StopIteration

            photo: dict = {
                "path": str(path),
                "filename": path.name,
                "taken_at": _iso(get_photo_datetime(path)),
            }

            for skill in _LOCAL_SKILLS:
                cached = cache.get(path.name, skill.name, skill.version)
                photo[skill.name] = cached if cached is not None else _run_cache(skill, path, cache)

            for skill in _CLAUDE_SKILLS:
                cached = cache.get(path.name, skill.name, skill.version)
                if cached is not None:
                    photo[skill.name] = cached
                else:
                    if on_progress:
                        on_progress(i, len(images), path.name, skill.name)
                    photo[skill.name] = _run_cache(skill, path, cache, client=client)

            photo["aggregate_score"] = _aggregate(photo)
            manifest.append(photo)

            if on_progress:
                on_progress(i, len(images), path.name, "done")

        cache.flush()
        return manifest


def _run_cache(skill, path: Path, cache: SkillCache, **kwargs) -> dict:
    result = skill.run(path, **kwargs)
    cache.set(path.name, skill.name, skill.version, result)
    return result


def _iso(dt) -> str | None:
    return dt.isoformat() if dt else None

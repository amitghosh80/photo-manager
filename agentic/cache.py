from __future__ import annotations

import json
from pathlib import Path


class SkillCache:
    """Disk-backed cache keyed by (filename, skill_name, skill_version)."""

    def __init__(self, trip_dir: Path) -> None:
        cache_dir = trip_dir / ".skill_cache"
        cache_dir.mkdir(exist_ok=True)
        self._path = cache_dir / "results.json"
        self._dirty = False
        try:
            self._data: dict[str, dict] = (
                json.loads(self._path.read_text(encoding="utf-8"))
                if self._path.exists() else {}
            )
        except Exception:
            self._data = {}

    @staticmethod
    def _key(filename: str, skill: str, version: str) -> str:
        return f"{filename}\x00{skill}\x00{version}"

    def get(self, filename: str, skill: str, version: str) -> dict | None:
        return self._data.get(self._key(filename, skill, version))

    def set(self, filename: str, skill: str, version: str, result: dict) -> None:
        self._data[self._key(filename, skill, version)] = result
        self._dirty = True

    def flush(self) -> None:
        if self._dirty:
            self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
            self._dirty = False

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    _HEIF = True
except ImportError:
    _HEIF = False

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif"}
if _HEIF:
    SUPPORTED_EXT |= {".heic", ".heif"}

MYTRIPS_DIR = Path(__file__).parent.parent / "mytrips"


def load_images(input_dir: Path) -> list[Path]:
    found: set[Path] = set()
    for ext in SUPPORTED_EXT:
        found.update(input_dir.glob(f"*{ext}"))
        found.update(input_dir.glob(f"*{ext.upper()}"))
    return sorted(found)


def get_photo_datetime(path: Path) -> datetime | None:
    from PIL import Image
    try:
        with Image.open(path) as img:
            exif = img._getexif() if hasattr(img, "_getexif") else None
            if exif:
                dt_str = exif.get(36867) or exif.get(306)
                if dt_str:
                    return datetime.strptime(str(dt_str)[:19], "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    name = path.stem
    for pattern, fmt in [
        (r"(\d{8})[_\-](\d{6})", "%Y%m%d%H%M%S"),
        (r"(\d{4})[_\-](\d{2})[_\-](\d{2})", "%Y%m%d"),
    ]:
        m = re.search(pattern, name)
        if m:
            try:
                return datetime.strptime("".join(m.groups()), fmt)
            except Exception:
                pass
    return None

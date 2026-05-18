#!/usr/bin/env python3
"""
Photo Curation App

Expects photos organised as:
  <script dir>/mytrips/<trip-name>/photo1.jpg ...

Pipeline:
  1. Ingest photos from mytrips/<trip-name>/
  2. Technical scoring  — sharpness (Laplacian variance) + exposure (histogram)
  3. AI scoring         — composition, subject clarity, uniqueness via Claude vision
  4. Aggregate scores   — 25% technical + 75% AI
  5. Pre-filter         — keep top N*3 pool before deduplication
  6. Deduplicate        — perceptual hashing; keep best-scored from each cluster
  7. Output             — ranked shortlist to console + JSON

Usage:
  python curator.py japan-2024
  python curator.py japan-2024 --top-n 15 --similarity-threshold 8
  python curator.py --list-trips
"""

import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import anthropic
import cv2
import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_SUPPORTED = True
except ImportError:
    HEIF_SUPPORTED = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif"}
if HEIF_SUPPORTED:
    SUPPORTED_EXTENSIONS |= {".heic", ".heif"}

# Weights for aggregating the three AI sub-scores
AI_WEIGHTS = {
    "composition":     0.35,
    "subject_clarity": 0.35,
    "uniqueness":      0.30,
}

# Fraction of final score that comes from technical analysis
TECHNICAL_WEIGHT = 0.25

# System prompt — cached across all API calls to save cost
SYSTEM_PROMPT = """\
You are an expert photo curator with deep knowledge of photography composition, \
technical quality, and artistic merit.

Evaluate the supplied photo on exactly three dimensions (each scored 0.0–10.0):

• composition     – Rule of thirds, leading lines, framing, balance, visual flow
• subject_clarity – Sharpness on main subject, subject prominence, separation from background
• uniqueness      – Creative perspective, unusual angle, memorable moment, distinctiveness

Also classify the photo:

• scene_id       – A 1–3 word snake_case label for the dominant scene/location \
(e.g. "beach_sunset", "city_street", "mountain_trail"). Use the SAME label for photos \
that share a visually similar location or scene.
• subject_type   – Exactly one of: "landscape", "portrait", "architecture", "wildlife", \
"street", "food", "abstract", "other"
• portrait_quality – ONLY if subject_type is "portrait"; otherwise set to null. \
Object with three floats:
    - expression: 0.0–10.0 (naturalness and positivity of facial expression)
    - eyes_open:  0.0–10.0 (10 = both eyes fully open and sharply in focus)
    - subject_complete: 0.0–10.0 (10 = full subject in frame, nothing cut off)

Return ONLY a JSON object — no markdown fences, no extra text — in this exact shape:
{
  "composition": <float>,
  "subject_clarity": <float>,
  "uniqueness": <float>,
  "scene_id": "<string>",
  "subject_type": "<string>",
  "portrait_quality": {"expression": <float>, "eyes_open": <float>, "subject_complete": <float>} | null,
  "brief_reason": "<one concise sentence explaining the overall quality>"
}"""

# Now that SYSTEM_PROMPT is defined, compute the real scoring version hash.
SCORING_VERSION: str = hashlib.sha256(
    json.dumps(
        {"prompt": SYSTEM_PROMPT, "weights": AI_WEIGHTS, "tech_weight": TECHNICAL_WEIGHT},
        sort_keys=True,
    ).encode()
).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Per-photo scores cache  (mytrips/<trip>/<trip>_scores.json)
# ---------------------------------------------------------------------------

def scores_cache_path(trip_dir: Path) -> Path:
    return trip_dir / f"{trip_dir.name}_scores.json"


def load_scores_cache(trip_dir: Path) -> dict[str, dict]:
    """
    Load cached AI scores keyed by filename.
    Returns {} if the file is missing, corrupt, or was written with a
    different SCORING_VERSION (i.e. the scoring logic has since changed).
    """
    path = scores_cache_path(trip_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("scoring_version") != SCORING_VERSION:
            return {}          # stale — logic changed, must re-score
        return data.get("scores", {})
    except Exception:
        return {}


def save_scores_cache(trip_dir: Path, scores: dict[str, dict]) -> None:
    """Persist AI scores (keyed by filename) alongside the scoring version."""
    scores_cache_path(trip_dir).write_text(
        json.dumps(
            {"scoring_version": SCORING_VERSION, "scores": scores},
            indent=2,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 1. Ingest
# ---------------------------------------------------------------------------

def load_images(input_dir: Path) -> list[Path]:
    """Return all supported image files found in *input_dir* (non-recursive)."""
    found: set[Path] = set()
    for ext in SUPPORTED_EXTENSIONS:
        found.update(input_dir.glob(f"*{ext}"))
        found.update(input_dir.glob(f"*{ext.upper()}"))
    return sorted(found)


# ---------------------------------------------------------------------------
# 2. Technical scoring
# ---------------------------------------------------------------------------

def compute_technical_score(image_path: Path) -> dict:
    """
    Returns sharpness, exposure, and a combined technical score (all 0–10).

    sharpness  — Laplacian variance of the greyscale image; higher = sharper.
    exposure   — penalises images with heavy clipping in shadows or highlights.
    technical  — 0.6 * sharpness + 0.4 * exposure
    """
    img = cv2.imread(str(image_path))
    if img is None:
        # Fallback for formats OpenCV can't open (e.g. HEIC) — open via Pillow
        try:
            pil = Image.open(image_path).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            return {"sharpness": 0.0, "exposure": 0.0, "technical": 0.0}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sharpness: Laplacian variance — typical range for a sharp photo: 300–5000+
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness = min(10.0, lap_var / 300.0)

    # Exposure: fraction of pixels that are heavily clipped
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist /= hist.sum()
    clipped = float(hist[:15].sum() + hist[240:].sum())   # black + white clipping
    exposure = max(0.0, 10.0 - clipped * 20.0)

    technical = round(sharpness * 0.6 + exposure * 0.4, 3)
    return {
        "sharpness": round(sharpness, 3),
        "exposure":  round(exposure, 3),
        "technical": technical,
    }


# ---------------------------------------------------------------------------
# 3. AI scoring via Claude vision
# ---------------------------------------------------------------------------

def _encode_image(image_path: Path, max_side: int = 1024) -> tuple[str, str]:
    """
    Open *image_path*, resize so the longest side ≤ *max_side*, and return
    (base64_data, media_type) suitable for the Anthropic messages API.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode(), "image/jpeg"


def _parse_ai_response(text: str) -> dict:
    """Parse Claude's JSON response, stripping accidental markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop opening fence (and optional language tag) + closing fence
        text = "\n".join(
            line for line in lines[1:]
            if not line.strip().startswith("```")
        ).strip()
    return json.loads(text)


def score_with_claude(
    client: anthropic.Anthropic,
    image_paths: list[Path],
    on_progress=None,
    should_stop=None,
) -> dict[str, dict]:
    """
    Call Claude once per image.  The system prompt is sent with
    cache_control="ephemeral" so it is cached across all calls,
    substantially reducing token cost for large batches.

    on_progress:  optional callable(done, total, filename) called after each image.
    should_stop:  optional callable() → bool; if it returns True before an image
                  is sent to the API the loop raises StopIteration to abort cleanly.
    """
    scores: dict[str, dict] = {}
    default = {
        "composition": 5.0,
        "subject_clarity": 5.0,
        "uniqueness": 5.0,
        "brief_reason": "Scoring unavailable — using defaults.",
    }
    total = len(image_paths)

    for idx, path in enumerate(tqdm(image_paths, desc="AI scoring (Claude)", unit="photo"), start=1):
        if should_stop and should_stop():
            raise StopIteration("Curation cancelled by user")
        try:
            b64_data, media_type = _encode_image(path)
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=384,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64_data,
                                },
                            },
                            {"type": "text", "text": "Score this photo."},
                        ],
                    }
                ],
            )
            scores[str(path)] = _parse_ai_response(response.content[0].text)
        except Exception as exc:
            tqdm.write(f"  Warning: could not score {path.name}: {exc}")
            scores[str(path)] = dict(default)
        if on_progress:
            on_progress(idx, total, path.name)

    return scores


# ---------------------------------------------------------------------------
# 4. Aggregate
# ---------------------------------------------------------------------------

def aggregate(technical: dict, ai: dict) -> float:
    """Combine technical (25%) and AI sub-scores (75%) into one 0–10 value.

    Portrait photos get a small bonus/penalty based on portrait_quality so that
    a well-executed portrait ranks above a mediocre one with the same base scores.
    """
    ai_score = sum(
        ai.get(dim, 5.0) * weight
        for dim, weight in AI_WEIGHTS.items()
    )
    base = TECHNICAL_WEIGHT * technical["technical"] + (1 - TECHNICAL_WEIGHT) * ai_score

    # Portrait quality bonus: ±0.15 swing around neutral (portrait_avg == 5.0)
    pq = ai.get("portrait_quality")
    if isinstance(pq, dict):
        portrait_avg = (
            pq.get("expression", 5.0) +
            pq.get("eyes_open",  5.0) +
            pq.get("subject_complete", 5.0)
        ) / 3.0
        base = min(10.0, base + (portrait_avg - 5.0) * 0.03)

    return round(base, 4)


# ---------------------------------------------------------------------------
# 6. Deduplication via perceptual hashing
# ---------------------------------------------------------------------------

def _phash(path: Path):
    """Return perceptual hash or None on failure."""
    try:
        with Image.open(path) as img:
            return imagehash.phash(img)
    except Exception:
        return None


def deduplicate(ranked: list[dict], threshold: int) -> list[dict]:
    """
    Walk the pre-sorted list (best first).  When a photo is within *threshold*
    Hamming distance of an already-kept photo, drop it.

    threshold=0  → only exact pixel-level duplicates removed
    threshold=10 → near-identical shots (typical burst/bracket pairs)
    threshold=24 → visually very similar compositions / same-scene reframes (web UI default)
    """
    print(f"\nDeduplicating (perceptual hash threshold = {threshold})…")
    hashes = {
        item["path"]: _phash(Path(item["path"]))
        for item in tqdm(ranked, desc="Hashing", unit="photo")
    }

    kept: list[dict] = []
    kept_hashes: list = []

    for item in ranked:
        h = hashes.get(item["path"])
        if h is None:
            kept.append(item)
            kept_hashes.append(None)
            continue

        # Check against every already-kept image
        is_duplicate = any(
            kh is not None and (h - kh) <= threshold
            for kh in kept_hashes
        )
        if not is_duplicate:
            kept.append(item)
            kept_hashes.append(h)

    removed = len(ranked) - len(kept)
    print(f"  Removed {removed} near-duplicate(s), {len(kept)} unique photos remain.")
    return kept


# ---------------------------------------------------------------------------
# 7. Chronological ordering helper
# ---------------------------------------------------------------------------

def get_photo_datetime(path: Path) -> datetime | None:
    """
    Return the capture datetime for *path*.

    Priority:
      1. EXIF DateTimeOriginal (tag 36867) or DateTime (tag 306)
      2. Parse YYYYMMDD_HHMMSS or YYYY-MM-DD_HH-MM-SS pattern from filename
      3. None — caller should fall back to filename alphabetical order
    """
    # 1. EXIF
    try:
        with Image.open(path) as img:
            exif = img._getexif() if hasattr(img, "_getexif") else None
            if exif:
                dt_str = exif.get(36867) or exif.get(306)
                if dt_str:
                    return datetime.strptime(str(dt_str)[:19], "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass

    # 2. Filename patterns
    name = path.stem
    for pattern, fmt in [
        (r"(\d{8})[_\-](\d{6})", "%Y%m%d%H%M%S"),           # 20230715_143022
        (r"(\d{4})[_\-](\d{2})[_\-](\d{2})", "%Y%m%d"),     # 2023-07-15
    ]:
        m = re.search(pattern, name)
        if m:
            try:
                return datetime.strptime("".join(m.groups()), fmt)
            except Exception:
                pass

    return None


# ---------------------------------------------------------------------------
# 7b. Diversity-aware selection
# ---------------------------------------------------------------------------

def diversity_aware_select(
    ranked: list[dict],
    top_n: int,
    max_per_scene: int = 2,
) -> tuple[list[dict], dict]:
    """
    Walk the pre-sorted pool (best first) and pick up to *top_n* photos,
    allowing at most *max_per_scene* photos per scene_id.

    If the constrained pass yields fewer than top_n, a second pass backfills
    from the remainder without scene limits.

    Returns (shortlist, diversity_info).
    """
    scene_counts: dict[str, int] = {}
    subject_counts: dict[str, int] = {}
    shortlist: list[dict] = []
    skipped: list[dict] = []

    for item in ranked:
        if len(shortlist) >= top_n:
            break
        scene_id     = item.get("scene_id") or "unknown"
        subject_type = item.get("subject_type") or "other"
        if scene_counts.get(scene_id, 0) < max_per_scene:
            shortlist.append(item)
            scene_counts[scene_id]     = scene_counts.get(scene_id, 0) + 1
            subject_counts[subject_type] = subject_counts.get(subject_type, 0) + 1
        else:
            skipped.append(item)

    # Backfill if pool was too small after diversity constraints
    for item in skipped:
        if len(shortlist) >= top_n:
            break
        subject_type = item.get("subject_type") or "other"
        shortlist.append(item)
        subject_counts[subject_type] = subject_counts.get(subject_type, 0) + 1

    diversity = {
        "scene_count":    len(scene_counts),
        "subject_counts": subject_counts,
    }
    return shortlist, diversity


# ---------------------------------------------------------------------------
# 8. Output helpers
# ---------------------------------------------------------------------------

CONSOLE_WIDTH = 74

def print_shortlist(shortlist: list[dict], total: int) -> None:
    print(f"\n{'=' * CONSOLE_WIDTH}")
    print(f"  RANKED PHOTO SHORTLIST  —  top {len(shortlist)} of {total} photos scanned")
    print(f"{'=' * CONSOLE_WIDTH}")
    header = f"{'#':<4} {'Score':<7} {'Comp':>6} {'Clarity':>8} {'Unique':>7}  Filename"
    print(header)
    print(f"{'-' * CONSOLE_WIDTH}")
    for item in shortlist:
        ai = item["ai_scores"]
        print(
            f"{item['rank']:<4} {item['aggregate_score']:<7.3f}"
            f" {ai['composition']:>6.1f} {ai['subject_clarity']:>8.1f}"
            f" {ai['uniqueness']:>7.1f}  {item['filename']}"
        )
        if item.get("reason"):
            print(f"      ↳ {item['reason']}")
    print(f"{'=' * CONSOLE_WIDTH}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MYTRIPS_DIR = Path(__file__).parent / "mytrips"


def list_trips() -> None:
    """Print available trip folders and exit."""
    if not MYTRIPS_DIR.is_dir():
        sys.exit(f"No 'mytrips' folder found at '{MYTRIPS_DIR}'. Create it and add trip sub-folders.")
    trips = sorted(p.name for p in MYTRIPS_DIR.iterdir() if p.is_dir())
    if not trips:
        sys.exit(f"'{MYTRIPS_DIR}' exists but contains no sub-folders yet.")
    print("Available trips:")
    for t in trips:
        count = len(load_images(MYTRIPS_DIR / t))
        print(f"  {t}  ({count} photo(s))")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curate a trip photo collection: score, filter, deduplicate, rank.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "trip", nargs="?",
        help="Trip name (sub-folder under mytrips/). Omit to see --list-trips.",
    )
    parser.add_argument(
        "--list-trips", action="store_true",
        help="List available trip folders and exit.",
    )
    parser.add_argument(
        "--top-n", type=int, default=20,
        help="Final shortlist size (default: 20)",
    )
    parser.add_argument(
        "--similarity-threshold", type=int, default=10,
        metavar="0-64",
        help=(
            "Perceptual hash Hamming distance threshold for deduplication "
            "(default: 10 — removes burst/bracket duplicates; "
            "lower = stricter, higher = more aggressive)"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Path for the JSON output file (default: mytrips/<trip>/<trip>_shortlist.json)",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Anthropic API key (falls back to ANTHROPIC_API_KEY env var)",
    )
    args = parser.parse_args()

    if args.list_trips or not args.trip:
        list_trips()
        return

    # Resolve trip directory
    trip_dir = MYTRIPS_DIR / args.trip
    if not trip_dir.is_dir():
        sys.exit(
            f"Error: trip folder '{trip_dir}' not found.\n"
            f"  Run with --list-trips to see available trips."
        )
    if not 0 <= args.similarity_threshold <= 64:
        sys.exit("Error: --similarity-threshold must be between 0 and 64.")

    output_path = args.output or trip_dir / f"{args.trip}_shortlist.json"

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit(
            "Error: Anthropic API key required.\n"
            "  Set ANTHROPIC_API_KEY environment variable or use --api-key."
        )

    # ── 1. Ingest ────────────────────────────────────────────────────────────
    print(f"Trip: {args.trip}")
    print(f"Scanning '{trip_dir}' for photos…")
    if not HEIF_SUPPORTED:
        print("  Note: pillow-heif not installed — HEIC/HEIF files will be skipped.")
    images = load_images(trip_dir)
    if not images:
        sys.exit("No supported images found. Supported: " + ", ".join(sorted(SUPPORTED_EXTENSIONS)))
    print(f"  Found {len(images)} photo(s).\n")

    # ── 2. Technical scoring ─────────────────────────────────────────────────
    print("Computing technical scores (sharpness + exposure)…")
    tech_scores: dict[str, dict] = {}
    for p in tqdm(images, desc="Technical", unit="photo"):
        tech_scores[str(p)] = compute_technical_score(p)

    # ── 3. AI scoring ────────────────────────────────────────────────────────
    client = anthropic.Anthropic(api_key=api_key)
    ai_scores = score_with_claude(client, images)

    # ── 4. Aggregate & sort ───────────────────────────────────────────────────
    results: list[dict] = []
    for p in images:
        key  = str(p)
        tech = tech_scores[key]
        ai   = ai_scores.get(key, {})
        dt   = get_photo_datetime(p)
        results.append({
            "path":            key,
            "filename":        p.name,
            "aggregate_score": aggregate(tech, ai),
            "technical":       tech,
            "ai_scores": {
                "composition":     ai.get("composition",     0.0),
                "subject_clarity": ai.get("subject_clarity", 0.0),
                "uniqueness":      ai.get("uniqueness",      0.0),
            },
            "scene_id":        ai.get("scene_id", ""),
            "subject_type":    ai.get("subject_type", "other"),
            "portrait_quality": ai.get("portrait_quality"),
            "taken_at":        dt.isoformat() if dt else None,
            "reason":          ai.get("brief_reason", ""),
        })
    results.sort(key=lambda x: x["aggregate_score"], reverse=True)

    # ── 5. Pre-filter — feed top N×3 into dedup ───────────────────────────────
    pool_size = min(len(results), args.top_n * 3)
    pool = results[:pool_size]
    print(f"\nTop {pool_size} photos selected for deduplication.")

    # ── 6. Deduplicate ────────────────────────────────────────────────────────
    deduped = deduplicate(pool, threshold=args.similarity_threshold)

    # ── 7. Diversity-aware final shortlist ────────────────────────────────────
    selected, diversity = diversity_aware_select(deduped, args.top_n)
    shortlist = [{"rank": i + 1, **item} for i, item in enumerate(selected)]

    # ── Output ────────────────────────────────────────────────────────────────
    all_photos = [{"rank": i + 1, **item} for i, item in enumerate(deduped)]
    output_data = {
        "trip":                  args.trip,
        "total_photos_scanned":  len(images),
        "shortlist_count":       len(shortlist),
        "all_photos_count":      len(all_photos),
        "similarity_threshold":  args.similarity_threshold,
        "top_n_requested":       args.top_n,
        "diversity":             diversity,
        "shortlist":             shortlist,
        "all_photos":            all_photos,
    }
    output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    print(f"JSON shortlist written to '{output_path}'")

    print_shortlist(shortlist, total=len(images))


if __name__ == "__main__":
    main()

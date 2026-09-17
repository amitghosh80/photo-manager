# Photo Curator

Automatically rank and shortlist your trip photos using Claude AI vision and computer vision.

Drop a folder of photos in, get back a ranked shortlist of the best shots — with near-duplicates removed and scene diversity enforced.

## How it works

The pipeline runs in 7 steps:

1. **Ingest** — scan `mytrips/<trip>/` for supported image files
2. **Technical scoring** — sharpness (Laplacian variance) + exposure (histogram clipping)
3. **AI scoring** — Claude vision scores each photo on composition, subject clarity, and uniqueness
4. **Aggregate** — combine scores: 25% technical + 75% AI
5. **Pre-filter** — keep top N×3 candidates before deduplication
6. **Deduplicate** — perceptual hashing (pHash) drops near-identical burst shots
7. **Output** — diversity-aware final shortlist written to console + JSON

AI scores are cached per trip so re-runs (e.g. after adjusting `--top-n`) skip the API entirely.

## Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)

## Setup

```bash
git clone https://github.com/amitghosh80/photo-manager
cd photo-manager
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

On Windows (PowerShell):
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

## Organise your photos

```
photo-manager/
└── mytrips/
    └── japan-2024/
        ├── IMG_001.jpg
        ├── IMG_002.jpg
        └── ...
```

## CLI usage

```bash
# List available trips
python curator.py --list-trips

# Curate a trip (default: top 20 photos)
python curator.py japan-2024

# Custom shortlist size and deduplication aggressiveness
python curator.py japan-2024 --top-n 15 --similarity-threshold 8

# Custom output path
python curator.py japan-2024 --output ~/Desktop/japan_picks.json
```

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--top-n` | 20 | Final shortlist size |
| `--similarity-threshold` | 10 | pHash Hamming distance for dedup (0 = exact only, 24 = same-scene reframes) |
| `--output` | `mytrips/<trip>/<trip>_shortlist.json` | JSON output path |
| `--api-key` | `$ANTHROPIC_API_KEY` | Anthropic API key |

## Web UI

```bash
python app.py
# Open http://localhost:5000
```

The web UI streams live progress via Server-Sent Events and lets you:

- Browse all trips and their photo counts
- Run curation with adjustable `top-n` and similarity threshold
- View the ranked gallery with AI scores and reasons
- Discover nearby photos (taken within 90 seconds of a selected shot)
- Save your final selection to `mytrips/<trip>/final shortlist/`
- Cancel a running job mid-flight

## Scoring breakdown

**Technical score** (25% weight)

| Metric | Method |
|--------|--------|
| Sharpness | Laplacian variance of greyscale image |
| Exposure | Penalty for clipped shadows/highlights |

**AI score** (75% weight, via Claude claude-sonnet-4-6)

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| Composition | 35% | Rule of thirds, leading lines, balance |
| Subject clarity | 35% | Sharpness on main subject, separation from background |
| Uniqueness | 30% | Creative angle, memorable moment, distinctiveness |

Portrait photos get an additional ±bonus based on expression, eyes-open, and subject completeness.

## Output

The shortlist JSON includes each photo's rank, aggregate score, per-dimension AI scores, scene label, subject type, capture timestamp, and a one-sentence reason from Claude.

```json
{
  "trip": "japan-2024",
  "shortlist_count": 20,
  "shortlist": [
    {
      "rank": 1,
      "filename": "IMG_3241.jpg",
      "aggregate_score": 8.412,
      "ai_scores": { "composition": 9.1, "subject_clarity": 8.5, "uniqueness": 7.8 },
      "scene_id": "temple_garden",
      "subject_type": "architecture",
      "taken_at": "2024-03-15T09:22:11",
      "reason": "Strong diagonal leading lines guide the eye through dappled light to the pagoda."
    }
  ]
}
```

## Supported formats

JPG, PNG, WEBP, TIFF — plus HEIC/HEIF when `pillow-heif` is installed (included in `requirements.txt`).

## Cost

The system prompt is prompt-cached across all API calls in a batch, so you only pay for the image tokens per photo. Typical cost for 100 photos is under $0.10.

Scores are cached locally keyed by a hash of the scoring logic. Changing the prompt or weights invalidates the cache and triggers a full re-score.

## License

MIT

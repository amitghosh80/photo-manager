# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```powershell
# Install dependencies
pip install -r requirements.txt

# Set API key (required for AI scoring)
$env:ANTHROPIC_API_KEY = "sk-ant-..."

# CLI: list available trips
python curator.py --list-trips

# CLI: curate a trip
python curator.py Vegas2026
python curator.py Vegas2026 --top-n 15 --similarity-threshold 8

# Web UI (original, port 5000)
python app.py

# Web UI (agentic rewrite, port 5001)
python -m agentic.app
```

There is no test suite or linter configured.

## Architecture

The repo has **two parallel implementations** of the same curation pipeline:

### Original (`curator.py` + `app.py`)
A single-file pipeline (`curator.py`) with a Flask web UI (`app.py`) that imports from it. The pipeline is a sequential 7-step function: ingest → technical scoring → Claude vision scoring → aggregate → pre-filter → pHash dedup → diversity-aware shortlist.

- AI scores are cached per trip in `mytrips/<trip>/<trip>_scores.json`, keyed by a SHA256 of the system prompt + weights. Changing the prompt or weights invalidates the entire cache.
- The Anthropic system prompt is sent with `cache_control: ephemeral` on every call so it is prompt-cached across the batch.

### Agentic rewrite (`agentic/`)
A Skills + Agents architecture that runs as `python -m agentic.app` on port 5001. The pipeline is orchestrated by `agentic/pipeline.py` through three agents:

1. **ScannerAgent** (`agents/scanner.py`) — runs all Skills on every photo, builds a manifest list. Local skills (blur, exposure, face, phash) run first; Claude skills (classify, score) follow with cache checks.
2. **SelectionAgent** (`agents/selector.py`) — a Claude tool-use agentic loop that calls tools (`list_photos`, `find_similar_photos`, `get_scene_breakdown`, `get_photo_details`, `finalize_shortlist`) to reason over the manifest and pick the best photos.
3. **SequencingAgent** (`agents/sequencer.py`) — a second Claude tool-use loop that arranges the shortlist into a narrative arc with a title, 2–3 sentence summary, and optional chapter breaks.

**Skills** (`agentic/skills/`) are stateless, versioned units (`name`, `version`, `run(path) → dict`). The skill version string controls cache invalidation — bump it when changing a skill's logic.

**SkillCache** (`agentic/cache.py`) is a per-trip disk cache at `mytrips/<trip>/.skill_cache/results.json`, keyed by `(filename, skill_name, skill_version)`. Call `cache.flush()` after a batch to persist.

### Scoring formula (both implementations)
`aggregate = 0.25 × technical + 0.75 × ai_weighted`
- Technical: `0.6 × sharpness + 0.4 × exposure` (both 0–10)
- AI sub-scores: `0.35 × composition + 0.35 × subject_clarity + 0.30 × uniqueness`
- Portrait bonus: `±(portrait_avg − 5.0) × 0.03` capped at 10.0

### Photo storage
Photos live in `mytrips/<trip-name>/` (non-recursive scan). Output JSON shortlists are written to `mytrips/<trip>/<trip>_shortlist.json`. The "Save shortlist" action in the web UI copies selected files to `mytrips/<trip>/final shortlist/`.

### SSE progress streaming
Both web UIs stream real-time progress via Server-Sent Events. The Flask backend spawns a daemon thread per job, emits events through a `queue.Queue`, and the `/api/progress/<job_id>` endpoint drains it. A `threading.Event` per job handles cancellation.

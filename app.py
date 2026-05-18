#!/usr/bin/env python3
"""
Flask web UI for the Photo Curator pipeline.

Run:  python app.py
Then open http://localhost:5000
"""

import json
import os
import queue
import shutil
import threading
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_from_directory

# Reuse all pipeline logic from curator.py
from curator import (
    MYTRIPS_DIR,
    SCORING_VERSION,
    aggregate,
    compute_technical_score,
    deduplicate,
    diversity_aware_select,
    get_photo_datetime,
    load_images,
    load_scores_cache,
    save_scores_cache,
    score_with_claude,
)
import anthropic

app = Flask(__name__)

# job_id -> Queue of SSE event dicts (None sentinel = done)
_jobs: dict[str, queue.Queue] = {}
# job_id -> Event; set() to request cancellation
_cancel_events: dict[str, threading.Event] = {}


# ---------------------------------------------------------------------------
# Background curation worker
# ---------------------------------------------------------------------------

def _run_curation(job_id: str, trip: str, top_n: int, threshold: int, api_key: str) -> None:
    q             = _jobs[job_id]
    cancel_event  = _cancel_events[job_id]

    def emit(event: str, **data):
        q.put({"event": event, "data": data})

    def should_stop() -> bool:
        return cancel_event.is_set()

    try:
        trip_dir = MYTRIPS_DIR / trip

        # ── 1. Scan + load per-photo scores cache ────────────────────────────
        images = load_images(trip_dir)
        if not images:
            emit("error", message="No supported images found in this trip folder.")
            return

        # Load previously computed AI scores (keyed by filename).
        # Returns {} if missing or scoring logic has changed since last run.
        cached_ai = load_scores_cache(trip_dir)

        # Split: photos whose AI scores are already cached vs need scoring
        to_score   = [p for p in images if p.name not in cached_ai]
        from_cache = len(images) - len(to_score)

        emit("scan", count=len(images), cached=from_cache, to_score=len(to_score))

        # ── 2. Technical scoring (fast — always recompute) ───────────────────
        tech_scores: dict[str, dict] = {}
        for i, p in enumerate(images, start=1):
            if should_stop():
                emit("cancelled"); return
            tech_scores[str(p)] = compute_technical_score(p)
            emit("technical", done=i, total=len(images))

        # ── 3. AI scoring — only photos not already in the scores cache ──────
        # Seed ai_scores with everything we have from cache
        ai_scores: dict[str, dict] = {
            str(p): cached_ai[p.name]
            for p in images if p.name in cached_ai
        }

        if to_score:
            client = anthropic.Anthropic(api_key=api_key)

            def on_ai_progress(done: int, total: int, filename: str):
                emit("ai", done=done, total=total, filename=filename,
                     cached=from_cache)

            new_scores = score_with_claude(
                client, to_score,
                on_progress=on_ai_progress,
                should_stop=should_stop,
            )
            ai_scores.update(new_scores)

            # Persist updated scores cache (merge old + new, keyed by filename)
            updated_cache = dict(cached_ai)
            for p in to_score:
                if str(p) in new_scores:
                    updated_cache[p.name] = new_scores[str(p)]
            save_scores_cache(trip_dir, updated_cache)
        else:
            # All photos were cached — emit a single ai event so the UI advances
            emit("ai", done=0, total=0, filename="", cached=from_cache)

        # ── 4. Aggregate & sort ──────────────────────────────────────────────
        results = []
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

        # ── 5. Pre-filter ────────────────────────────────────────────────────
        pool = results[: min(len(results), max(top_n * 5, 150))]
        emit("dedup", stage="start", pool=len(pool))

        # ── 6. Deduplicate ───────────────────────────────────────────────────
        deduped = deduplicate(pool, threshold=threshold)

        # ── 7. Diversity-aware shortlist + full deduped pool ─────────────────
        selected, diversity = diversity_aware_select(deduped, top_n)
        shortlist = [
            {"rank": i + 1, **item}
            for i, item in enumerate(selected)
        ]
        all_photos = [
            {"rank": i + 1, **item}
            for i, item in enumerate(deduped)
        ]

        # Persist shortlist (derived, can always be rebuilt from scores cache)
        out_path = MYTRIPS_DIR / trip / f"{trip}_shortlist.json"
        out_path.write_text(
            json.dumps({
                "trip": trip,
                "scoring_version": SCORING_VERSION,
                "total_photos_scanned": len(images),
                "shortlist_count": len(shortlist),
                "all_photos_count": len(all_photos),
                "similarity_threshold": threshold,
                "top_n_requested": top_n,
                "diversity": diversity,
                "shortlist": shortlist,
                "all_photos": all_photos,
            }, indent=2),
            encoding="utf-8",
        )

        emit("complete", shortlist=shortlist, all_photos=all_photos,
             total=len(images), diversity=diversity,
             scoring_quality=_scoring_quality(shortlist))

    except StopIteration:
        emit("cancelled")
    except Exception as exc:
        emit("error", message=str(exc))
    finally:
        _cancel_events.pop(job_id, None)
        q.put(None)  # sentinel — SSE stream can close


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/trips")
def api_trips():
    if not MYTRIPS_DIR.is_dir():
        return jsonify([])
    trips = sorted(
        {"name": p.name, "count": len(load_images(p))}
        for p in MYTRIPS_DIR.iterdir()
        if p.is_dir()
    )
    return jsonify(trips)


@app.route("/api/curate", methods=["POST"])
def api_curate():
    body      = request.get_json(force=True)
    trip      = body.get("trip", "").strip()
    top_n     = int(body.get("top_n", 20))
    threshold = int(body.get("threshold", 24))
    api_key   = body.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
    force     = bool(body.get("force", False))

    if not trip:
        return jsonify({"error": "trip name required"}), 400
    if not (MYTRIPS_DIR / trip).is_dir():
        return jsonify({"error": f"Trip '{trip}' not found in mytrips/"}), 404

    # Return cached results if available, not forcing a re-run, and scoring
    # logic hasn't changed since the cache was written.
    cache_path = MYTRIPS_DIR / trip / f"{trip}_shortlist.json"
    if not force and cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if data.get("scoring_version") != SCORING_VERSION:
                raise ValueError("stale scoring version — fall through to re-run")
            all_photos = data.get("all_photos", data["shortlist"])

            # Back-fill taken_at for any photo that is missing it (cache pre-dates
            # the feature, or EXIF was unavailable on a previous run).
            needs_backfill = any(p.get("taken_at") is None for p in all_photos)
            if needs_backfill:
                trip_dir = MYTRIPS_DIR / trip
                dt_cache: dict[str, str | None] = {}
                for photo in all_photos + data["shortlist"]:
                    fn = photo["filename"]
                    if fn not in dt_cache:
                        dt = get_photo_datetime(trip_dir / fn)
                        dt_cache[fn] = dt.isoformat() if dt else None
                    if photo.get("taken_at") is None:
                        photo["taken_at"] = dt_cache[fn]
                # Persist the back-filled cache so next load is instant
                cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            quality = _scoring_quality(data["shortlist"])
            if quality == "defaults":
                # Purge so the next non-forced load triggers a fresh run
                cache_path.unlink(missing_ok=True)

            return jsonify({
                "cached":          True,
                "shortlist":       data["shortlist"],
                "all_photos":      all_photos,
                "total":           data["total_photos_scanned"],
                "diversity":       data.get("diversity"),
                "scoring_quality": quality,
            })
        except Exception:
            pass  # corrupt cache — fall through to re-run

    if not api_key:
        return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 400

    # Delete stale cache so a fresh one is written when the run finishes
    cache_path.unlink(missing_ok=True)

    job_id = str(uuid.uuid4())
    _jobs[job_id]          = queue.Queue()
    _cancel_events[job_id] = threading.Event()
    threading.Thread(
        target=_run_curation,
        args=(job_id, trip, top_n, threshold, api_key),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id})


@app.route("/api/cancel/<job_id>", methods=["POST"])
def api_cancel(job_id: str):
    event = _cancel_events.get(job_id)
    if event:
        event.set()
        return jsonify({"ok": True})
    return jsonify({"error": "job not found"}), 404


@app.route("/api/progress/<job_id>")
def api_progress(job_id: str):
    if job_id not in _jobs:
        return jsonify({"error": "unknown job"}), 404

    def generate():
        q = _jobs[job_id]
        while True:
            msg = q.get()
            if msg is None:
                # Clean up and close the stream
                _jobs.pop(job_id, None)
                yield "event: close\ndata: {}\n\n"
                break
            yield f"event: {msg['event']}\ndata: {json.dumps(msg['data'])}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


NEARBY_WINDOW_SECS = 90   # photos within this many seconds count as "nearby"


def _scoring_quality(photos: list[dict]) -> str:
    """
    Return 'defaults' when every photo has the all-5.0 fallback scores —
    a reliable sign that every Claude API call failed during the scoring run.
    Return 'ok' otherwise.
    """
    if not photos:
        return "ok"
    return "defaults" if all(
        p["ai_scores"]["composition"]     == 5.0 and
        p["ai_scores"]["subject_clarity"] == 5.0 and
        p["ai_scores"]["uniqueness"]      == 5.0
        for p in photos
    ) else "ok"


@app.route("/api/nearby/<trip>/<path:filename>")
def api_nearby(trip: str, filename: str):
    """Return all photos taken within NEARBY_WINDOW_SECS of the given photo."""
    trip_dir = MYTRIPS_DIR / trip
    if not trip_dir.is_dir():
        return jsonify({"error": "trip not found"}), 404

    ref_path = trip_dir / filename
    if not ref_path.exists():
        return jsonify({"error": "photo not found"}), 404

    ref_dt = get_photo_datetime(ref_path)
    if ref_dt is None:
        return jsonify({"error": "no timestamp available for this photo"}), 400

    # Load AI scores cache so we can include scores for nearby photos
    ai_cache = load_scores_cache(trip_dir)

    nearby = []
    for p in load_images(trip_dir):
        if p.name == filename:
            continue
        dt = get_photo_datetime(p)
        if dt is None:
            continue
        diff = abs((dt - ref_dt).total_seconds())
        if diff > NEARBY_WINDOW_SECS:
            continue
        ai = ai_cache.get(p.name, {})
        nearby.append({
            "filename":        p.name,
            "taken_at":        dt.isoformat(),
            "seconds_apart":   int(diff),
            "ai_scores":       {
                "composition":     ai.get("composition"),
                "subject_clarity": ai.get("subject_clarity"),
                "uniqueness":      ai.get("uniqueness"),
            } if ai else None,
            "aggregate_score": None if not ai else round(
                (ai.get("composition", 5) * 0.35 +
                 ai.get("subject_clarity", 5) * 0.35 +
                 ai.get("uniqueness", 5) * 0.30) * 0.75 + 5 * 0.25, 2
            ),
            "scene_id":        ai.get("scene_id", ""),
            "subject_type":    ai.get("subject_type", "other"),
            "portrait_quality": ai.get("portrait_quality"),
            "reason":          ai.get("brief_reason", ""),
        })

    nearby.sort(key=lambda x: x["seconds_apart"])
    return jsonify({
        "reference":   {"filename": filename, "taken_at": ref_dt.isoformat()},
        "nearby":      nearby,
        "window_secs": NEARBY_WINDOW_SECS,
    })


FINAL_SHORTLIST_FOLDER = "final shortlist"


@app.route("/api/save-shortlist/<trip>", methods=["POST"])
def api_save_shortlist(trip: str):
    """Copy the current gallery selection into mytrips/<trip>/final shortlist/."""
    trip_dir = MYTRIPS_DIR / trip
    if not trip_dir.is_dir():
        return jsonify({"error": f"Trip '{trip}' not found"}), 404

    body     = request.get_json(force=True)
    filenames = body.get("filenames", [])
    if not filenames:
        return jsonify({"error": "No filenames provided"}), 400

    dest_dir = trip_dir / FINAL_SHORTLIST_FOLDER
    # Clear any previous save so the folder exactly matches the current selection
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir()

    saved, missing = [], []
    for fn in filenames:
        src = trip_dir / fn
        if src.exists():
            shutil.copy2(src, dest_dir / fn)
            saved.append(fn)
        else:
            missing.append(fn)

    return jsonify({
        "saved":   len(saved),
        "missing": missing,
        "folder":  str(dest_dir),
    })


@app.route("/photos/<trip>/<path:filename>")
def serve_photo(trip: str, filename: str):
    """Serve original photos from the mytrips directory."""
    return send_from_directory(MYTRIPS_DIR / trip, filename)


if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5000)

"""
Agentic Photo Curator — Flask web UI.

Run from repo root:  python -m agentic.app
Then open:           http://localhost:5001
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from .pipeline import run as run_pipeline
from .ingest import MYTRIPS_DIR, load_images, get_photo_datetime
from .cache import SkillCache
from .skills.score import AestheticScoreSkill
from .skills.classify import ClassifySkill
from .skills.blur import BlurSkill
from .skills.exposure import ExposureSkill

app = Flask(__name__)

# job_id → Queue; None sentinel = stream done
_jobs: dict[str, queue.Queue] = {}
_cancel: dict[str, threading.Event] = {}


# ── Background worker ────────────────────────────────────────────────────────

def _worker(job_id: str, trip: str, top_n: int, threshold: int, api_key: str) -> None:
    q = _jobs[job_id]
    ev = _cancel[job_id]

    def emit(event: str, data: dict) -> None:
        q.put({"event": event, "data": data})

    run_pipeline(
        trip=trip,
        top_n=top_n,
        threshold=threshold,
        api_key=api_key,
        on_event=emit,
        should_stop=ev.is_set,
    )
    _cancel.pop(job_id, None)
    q.put(None)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/config")
def api_config():
    return jsonify({"has_api_key": bool(os.environ.get("ANTHROPIC_API_KEY"))})


@app.route("/api/trips")
def api_trips():
    if not MYTRIPS_DIR.is_dir():
        return jsonify([])
    trips = sorted(
        {"name": p.name, "count": len(load_images(p))}
        for p in MYTRIPS_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    return jsonify(trips)


@app.route("/api/curate", methods=["POST"])
def api_curate():
    body = request.get_json(force=True)
    trip = body.get("trip", "").strip()
    top_n = int(body.get("top_n", 20))
    threshold = int(body.get("threshold", 24))
    api_key = body.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
    force = bool(body.get("force", False))

    if not trip:
        return jsonify({"error": "trip name required"}), 400
    if not (MYTRIPS_DIR / trip).is_dir():
        return jsonify({"error": f"Trip '{trip}' not found"}), 404

    cache_path = MYTRIPS_DIR / trip / f"{trip}_shortlist.json"
    if not force and cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if "narrative_title" in data:
                return jsonify({"cached": True, **data})
        except Exception:
            pass

    if not api_key:
        return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 400

    cache_path.unlink(missing_ok=True)
    job_id = str(uuid.uuid4())
    _jobs[job_id] = queue.Queue()
    _cancel[job_id] = threading.Event()
    threading.Thread(target=_worker, args=(job_id, trip, top_n, threshold, api_key), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/cancel/<job_id>", methods=["POST"])
def api_cancel(job_id: str):
    ev = _cancel.get(job_id)
    if ev:
        ev.set()
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
                _jobs.pop(job_id, None)
                yield "event: close\ndata: {}\n\n"
                break
            yield f"event: {msg['event']}\ndata: {json.dumps(msg['data'])}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


NEARBY_SECS = 90


@app.route("/api/nearby/<trip>/<path:filename>")
def api_nearby(trip: str, filename: str):
    trip_dir = MYTRIPS_DIR / trip
    if not trip_dir.is_dir():
        return jsonify({"error": "trip not found"}), 404
    ref_path = trip_dir / filename
    if not ref_path.exists():
        return jsonify({"error": "photo not found"}), 404
    ref_dt = get_photo_datetime(ref_path)
    if ref_dt is None:
        return jsonify({"error": "no timestamp available"}), 400

    cache = SkillCache(trip_dir)
    nearby = []
    for p in load_images(trip_dir):
        if p.name == filename:
            continue
        dt = get_photo_datetime(p)
        if dt is None:
            continue
        diff = abs((dt - ref_dt).total_seconds())
        if diff > NEARBY_SECS:
            continue
        score = cache.get(p.name, AestheticScoreSkill.name, AestheticScoreSkill.version) or {}
        classify = cache.get(p.name, ClassifySkill.name, ClassifySkill.version) or {}
        blur = cache.get(p.name, BlurSkill.name, BlurSkill.version) or {}
        exp = cache.get(p.name, ExposureSkill.name, ExposureSkill.version) or {}
        nearby.append({
            "filename":           p.name,
            "taken_at":           dt.isoformat(),
            "seconds_apart":      int(diff),
            "aggregate_score":    _agg(blur, exp, score),
            "ai_scores": {
                "composition":     score.get("composition"),
                "subject_clarity": score.get("subject_clarity"),
                "uniqueness":      score.get("uniqueness"),
            } if score else None,
            "scene_id":           classify.get("scene_id", ""),
            "narrative_category": classify.get("narrative_category", ""),
            "reason":             score.get("brief_reason", ""),
        })
    nearby.sort(key=lambda x: x["seconds_apart"])
    return jsonify({
        "reference": {"filename": filename, "taken_at": ref_dt.isoformat()},
        "nearby": nearby,
        "window_secs": NEARBY_SECS,
    })


@app.route("/api/save-shortlist/<trip>", methods=["POST"])
def api_save_shortlist(trip: str):
    trip_dir = MYTRIPS_DIR / trip
    if not trip_dir.is_dir():
        return jsonify({"error": f"Trip '{trip}' not found"}), 404
    filenames = request.get_json(force=True).get("filenames", [])
    if not filenames:
        return jsonify({"error": "no filenames provided"}), 400
    dest = trip_dir / "final shortlist"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir()
    saved, missing = [], []
    for fn in filenames:
        src = trip_dir / fn
        if src.exists():
            shutil.copy2(src, dest / fn)
            saved.append(fn)
        else:
            missing.append(fn)
    return jsonify({"saved": len(saved), "missing": missing, "folder": str(dest)})


@app.route("/photos/<trip>/<path:filename>")
def serve_photo(trip: str, filename: str):
    return send_from_directory(MYTRIPS_DIR / trip, filename)


def _agg(blur: dict, exp: dict, score: dict) -> float | None:
    if not score:
        return None
    tech = blur.get("sharpness_score", 5.0) * 0.6 + exp.get("exposure_score", 5.0) * 0.4
    ai = (score.get("composition", 5.0) * 0.35 +
          score.get("subject_clarity", 5.0) * 0.35 +
          score.get("uniqueness", 5.0) * 0.30)
    return round(tech * 0.25 + ai * 0.75, 2)


if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5001)

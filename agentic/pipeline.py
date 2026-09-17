"""End-to-end curation pipeline: Scanner → SelectionAgent → SequencingAgent."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import anthropic

from .ingest import MYTRIPS_DIR
from .cache import SkillCache
from .agents import ScannerAgent, SelectionAgent, SequencingAgent


def run(
    trip: str,
    top_n: int,
    threshold: int,
    api_key: str,
    on_event: Callable[[str, dict], None],
    should_stop: Callable[[], bool] | None = None,
) -> None:
    """Run the full agentic curation pipeline, emitting progress events."""
    trip_dir = MYTRIPS_DIR / trip
    cache = SkillCache(trip_dir)
    client = anthropic.Anthropic(api_key=api_key)

    try:
        # ── Phase 1: Scan ────────────────────────────────────────────────────
        on_event("scan_start", {"trip": trip})

        def on_scan(done: int, total: int, filename: str, phase: str) -> None:
            if should_stop and should_stop():
                raise StopIteration
            on_event("scanning", {"done": done, "total": total, "filename": filename, "phase": phase})

        manifest = ScannerAgent().scan(
            trip_dir, client, cache,
            on_progress=on_scan,
            should_stop=should_stop,
        )
        on_event("scan_complete", {"total": len(manifest)})

        # ── Phase 2: Selection Agent ─────────────────────────────────────────
        on_event("selecting", {"status": "start", "top_n": top_n})

        shortlist = SelectionAgent().select(
            manifest, top_n, threshold, client,
            on_event=on_event,
            should_stop=should_stop,
        )
        on_event("selection_complete", {"count": len(shortlist)})

        # ── Phase 3: Sequencing Agent ────────────────────────────────────────
        on_event("sequencing", {"status": "start"})

        result = SequencingAgent().sequence(
            shortlist, client,
            on_event=on_event,
            should_stop=should_stop,
        )

        # ── Persist ──────────────────────────────────────────────────────────
        out = trip_dir / f"{trip}_shortlist.json"
        out.write_text(
            json.dumps({
                "trip":                  trip,
                "top_n_requested":       top_n,
                "similarity_threshold":  threshold,
                "total_photos_scanned":  len(manifest),
                "shortlist_count":       len(shortlist),
                "narrative_title":       result["narrative_title"],
                "narrative_summary":     result["narrative_summary"],
                "chapter_breaks":        result["chapter_breaks"],
                "shortlist":             result["sequence"],
            }, indent=2),
            encoding="utf-8",
        )

        on_event("complete", {
            "shortlist":         result["sequence"],
            "narrative_title":   result["narrative_title"],
            "narrative_summary": result["narrative_summary"],
            "chapter_breaks":    result["chapter_breaks"],
            "total":             len(manifest),
        })

    except StopIteration:
        on_event("cancelled", {})
    except Exception as exc:
        on_event("error", {"message": str(exc)})

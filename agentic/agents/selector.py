"""SelectionAgent — Claude tool-use agent that picks the best photos."""
from __future__ import annotations

import json
from collections import Counter
from typing import Callable

import imagehash

_SYSTEM = """\
You are an expert photo editor curating a trip album.

Use the available tools to explore the photo data, then select the best photos.

Guidelines:
- Prioritise technically excellent photos (sharp, well-exposed)
- Ensure scene diversity — avoid too many photos from the same location
- Include a mix of subject types and narrative categories
- When photos are near-duplicates, keep only the best one
- Aim to tell the complete story of the trip

Workflow:
1. Call get_scene_breakdown to understand the collection
2. Call list_photos to see the top-scoring candidates
3. Use find_similar_photos to identify near-duplicates
4. Use get_photo_details for borderline cases
5. Call finalize_shortlist exactly once with your final selection"""


class _State:
    def __init__(self, manifest: list[dict], threshold: int):
        self.manifest = manifest
        self.threshold = threshold
        self._by_fn = {p["filename"]: p for p in manifest}
        self._shortlist: list[str] | None = None
        self._reasoning = ""

    def handle(self, name: str, inputs: dict) -> dict:
        match name:
            case "list_photos":
                return self._list(**inputs)
            case "get_photo_details":
                return self._details(**inputs)
            case "find_similar_photos":
                return self._similar(**inputs)
            case "get_scene_breakdown":
                return self._breakdown()
            case "finalize_shortlist":
                self._shortlist = inputs["filenames"]
                self._reasoning = inputs.get("reasoning", "")
                return {"status": "ok", "selected": len(self._shortlist)}
            case _:
                return {"error": f"unknown tool: {name}"}

    def _list(self, sort_by="aggregate_score", limit=25,
              filter_blurry=False, filter_overexposed=False) -> dict:
        photos = list(self.manifest)
        if filter_blurry:
            photos = [p for p in photos if not p.get("blur_detection", {}).get("is_blurry", False)]
        if filter_overexposed:
            photos = [p for p in photos if not p.get("exposure_analysis", {}).get("is_overexposed", False)]
        key_fn = {
            "aggregate_score": lambda p: p.get("aggregate_score", 0),
            "sharpness":       lambda p: p.get("blur_detection", {}).get("sharpness_score", 0),
            "exposure":        lambda p: p.get("exposure_analysis", {}).get("exposure_score", 0),
            "taken_at":        lambda p: p.get("taken_at") or "",
        }.get(sort_by, lambda p: p.get("aggregate_score", 0))
        photos.sort(key=key_fn, reverse=(sort_by != "taken_at"))
        return {"photos": [_summary(p) for p in photos[:limit]], "total": len(photos)}

    def _details(self, filename: str) -> dict:
        p = self._by_fn.get(filename)
        if p is None:
            return {"error": f"not found: {filename}"}
        return {k: v for k, v in p.items() if k != "path"}

    def _similar(self, filename: str, threshold: int | None = None) -> dict:
        thr = threshold if threshold is not None else self.threshold
        ref_str = self._by_fn.get(filename, {}).get("phash", {}).get("hash")
        if not ref_str:
            return {"similar": [], "note": "no hash available for this photo"}
        ref_h = imagehash.hex_to_hash(ref_str)
        similar = []
        for p in self.manifest:
            if p["filename"] == filename:
                continue
            h_str = p.get("phash", {}).get("hash")
            if not h_str:
                continue
            dist = ref_h - imagehash.hex_to_hash(h_str)
            if dist <= thr:
                similar.append({
                    "filename": p["filename"],
                    "hamming_distance": dist,
                    "aggregate_score": p.get("aggregate_score"),
                })
        similar.sort(key=lambda x: x["hamming_distance"])
        return {"reference": filename, "threshold": thr, "similar": similar}

    def _breakdown(self) -> dict:
        scores = [p.get("aggregate_score", 0) for p in self.manifest]
        return {
            "total_photos": len(self.manifest),
            "scenes": dict(Counter(
                p.get("narrative_classify", {}).get("scene_id", "unknown")
                for p in self.manifest
            ).most_common()),
            "subject_types": dict(Counter(
                p.get("narrative_classify", {}).get("subject_type", "other")
                for p in self.manifest
            ).most_common()),
            "narrative_categories": dict(Counter(
                p.get("narrative_classify", {}).get("narrative_category", "other")
                for p in self.manifest
            ).most_common()),
            "score_range": {
                "min":    round(min(scores, default=0), 2),
                "max":    round(max(scores, default=10), 2),
                "median": round(sorted(scores)[len(scores) // 2], 2) if scores else 5.0,
            },
        }

    def get_shortlist(self, top_n: int) -> list[dict]:
        if not self._shortlist:
            ranked = sorted(self.manifest, key=lambda p: p.get("aggregate_score", 0), reverse=True)
            self._shortlist = [p["filename"] for p in ranked[:top_n]]
        result = []
        for rank, fn in enumerate(self._shortlist, 1):
            p = self._by_fn.get(fn, {"filename": fn})
            classify = p.get("narrative_classify", {})
            score = p.get("aesthetic_score", {})
            entry = {
                "rank":             rank,
                "filename":         fn,
                "path":             p.get("path", ""),
                "aggregate_score":  p.get("aggregate_score", 0),
                "blur_detection":   p.get("blur_detection"),
                "exposure_analysis": p.get("exposure_analysis"),
                "face_detection":   p.get("face_detection"),
                "ai_scores": {
                    "composition":     score.get("composition", 5.0),
                    "subject_clarity": score.get("subject_clarity", 5.0),
                    "uniqueness":      score.get("uniqueness", 5.0),
                },
                "scene_id":           classify.get("scene_id", ""),
                "subject_type":       classify.get("subject_type", "other"),
                "narrative_category": classify.get("narrative_category", "other"),
                "emotional_tone":     classify.get("emotional_tone", ""),
                "setting":            classify.get("setting", ""),
                "taken_at":           p.get("taken_at"),
                "reason":             score.get("brief_reason", ""),
            }
            pq = score.get("portrait_quality")
            if pq:
                entry["portrait_quality"] = pq
            result.append(entry)
        return result


def _summary(p: dict) -> dict:
    classify = p.get("narrative_classify", {})
    return {
        "filename":           p["filename"],
        "aggregate_score":    p.get("aggregate_score"),
        "sharpness_score":    p.get("blur_detection", {}).get("sharpness_score"),
        "exposure_score":     p.get("exposure_analysis", {}).get("exposure_score"),
        "face_count":         p.get("face_detection", {}).get("face_count", 0),
        "scene_id":           classify.get("scene_id", ""),
        "narrative_category": classify.get("narrative_category", ""),
        "subject_type":       classify.get("subject_type", ""),
        "emotional_tone":     classify.get("emotional_tone", ""),
        "taken_at":           p.get("taken_at"),
    }


_TOOLS = [
    {
        "name": "list_photos",
        "description": "List photos sorted by a score metric. Use to survey candidates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sort_by": {
                    "type": "string",
                    "enum": ["aggregate_score", "sharpness", "exposure", "taken_at"],
                    "description": "Metric to sort by (default: aggregate_score)",
                },
                "limit": {"type": "integer", "description": "Max photos to return (default 25)"},
                "filter_blurry": {"type": "boolean", "description": "Exclude blurry photos"},
                "filter_overexposed": {"type": "boolean", "description": "Exclude overexposed photos"},
            },
        },
    },
    {
        "name": "get_photo_details",
        "description": "Get full metadata for a single photo — all skill scores.",
        "input_schema": {
            "type": "object",
            "properties": {"filename": {"type": "string"}},
            "required": ["filename"],
        },
    },
    {
        "name": "find_similar_photos",
        "description": "Find near-duplicate photos via perceptual hash distance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "threshold": {
                    "type": "integer",
                    "description": "Hamming distance (0=exact, 10=burst shots, 24=same-scene reframes)",
                },
            },
            "required": ["filename"],
        },
    },
    {
        "name": "get_scene_breakdown",
        "description": "Count photos by scene, subject type, narrative category, and score range. Call this first.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "finalize_shortlist",
        "description": "Submit your final photo selection. Call exactly once when ready.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filenames": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Selected photo filenames",
                },
                "reasoning": {"type": "string", "description": "Brief explanation of selection strategy"},
            },
            "required": ["filenames"],
        },
    },
]


class SelectionAgent:
    """Claude tool-use agent that selects the best photos from a trip manifest."""

    def select(
        self,
        manifest: list[dict],
        top_n: int,
        similarity_threshold: int,
        client,
        on_event: Callable[[str, dict], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[dict]:
        state = _State(manifest, similarity_threshold)
        top5 = sorted(manifest, key=lambda p: p.get("aggregate_score", 0), reverse=True)[:5]
        initial = (
            f"Please select the best {top_n} photos from this trip.\n"
            f"Total photos: {len(manifest)}\n"
            f"Top 5 by score:\n{json.dumps([_summary(p) for p in top5], indent=2)}\n\n"
            "Start with get_scene_breakdown to understand the full collection, then proceed."
        )
        messages = [{"role": "user", "content": initial}]

        while True:
            if should_stop and should_stop():
                raise StopIteration
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=_SYSTEM,
                tools=_TOOLS,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": resp.content})
            if on_event:
                on_event("selecting", {"stop_reason": resp.stop_reason})
            if resp.stop_reason != "tool_use":
                break
            results = []
            finalized = False
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                if on_event:
                    on_event("selecting", {"tool": block.name, "input": block.input})
                outcome = state.handle(block.name, block.input)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(outcome),
                })
                if block.name == "finalize_shortlist":
                    finalized = True
            messages.append({"role": "user", "content": results})
            if finalized:
                break

        return state.get_shortlist(top_n)

"""SequencingAgent — Claude tool-use agent that orders selected photos into a narrative arc."""
from __future__ import annotations

import json
from typing import Callable

_SYSTEM = """\
You are a travel photo book editor arranging selected photos into a compelling narrative sequence.

Use the tools to examine the photos, then set a final display order.

Guidelines:
- Create an arc: establish the setting → build through activities → peak moments → reflective close
- Alternate subject types for visual rhythm (avoid clustering all landscapes together)
- Follow chronological order within a day but prioritise narrative impact over strict time order
- Strongest photos work best at the opening, at the 1/3 and 2/3 marks, and at the close
- Note natural chapter breaks between days or locations

Workflow:
1. Call get_shortlist to see all selected photos with their metadata
2. Use get_by_narrative_category or get_by_timeline to explore groupings
3. Call set_sequence exactly once with your final ordered list, narrative title, and 2-3 sentence summary"""


class _State:
    def __init__(self, shortlist: list[dict]):
        self.shortlist = shortlist
        self._by_fn = {p["filename"]: p for p in shortlist}
        self.sequence: list[str] | None = None
        self.narrative_title = ""
        self.narrative_summary = ""
        self.chapter_breaks: list[dict] = []

    def handle(self, name: str, inputs: dict) -> dict:
        match name:
            case "get_shortlist":
                return {
                    "count": len(self.shortlist),
                    "photos": [
                        {
                            "filename":           p["filename"],
                            "scene_id":           p.get("scene_id", ""),
                            "subject_type":       p.get("subject_type", ""),
                            "narrative_category": p.get("narrative_category", ""),
                            "emotional_tone":     p.get("emotional_tone", ""),
                            "taken_at":           p.get("taken_at"),
                            "aggregate_score":    p.get("aggregate_score"),
                            "reason":             p.get("reason", ""),
                        }
                        for p in self.shortlist
                    ],
                }
            case "get_by_narrative_category":
                cat = inputs.get("category", "")
                matching = [p["filename"] for p in self.shortlist if p.get("narrative_category") == cat]
                return {"category": cat, "count": len(matching), "filenames": matching}
            case "get_by_timeline":
                with_ts = sorted(
                    [(p["taken_at"], p["filename"]) for p in self.shortlist if p.get("taken_at")]
                )
                no_ts = [p["filename"] for p in self.shortlist if not p.get("taken_at")]
                return {"chronological": [fn for _, fn in with_ts], "no_timestamp": no_ts}
            case "set_sequence":
                self.sequence = inputs["filenames"]
                self.narrative_title = inputs.get("narrative_title", "")
                self.narrative_summary = inputs.get("narrative_summary", "")
                self.chapter_breaks = inputs.get("chapter_breaks", [])
                return {"status": "ok", "count": len(self.sequence)}
            case _:
                return {"error": f"unknown tool: {name}"}

    def get_result(self) -> dict:
        if not self.sequence:
            self.sequence = [p["filename"] for p in self.shortlist]
        ordered = []
        for i, fn in enumerate(self.sequence):
            if fn not in self._by_fn:
                continue
            p = dict(self._by_fn[fn])
            p["rank"] = i + 1
            ordered.append(p)
        return {
            "sequence":          ordered,
            "narrative_title":   self.narrative_title,
            "narrative_summary": self.narrative_summary,
            "chapter_breaks":    self.chapter_breaks,
        }


_TOOLS = [
    {
        "name": "get_shortlist",
        "description": "Get all selected photos with their metadata. Call this first.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_by_narrative_category",
        "description": "Get filenames belonging to a specific narrative category.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["arrival", "exploration", "activity", "meal", "portrait",
                             "landmark", "candid", "atmosphere", "departure", "other"],
                }
            },
            "required": ["category"],
        },
    },
    {
        "name": "get_by_timeline",
        "description": "Get photos sorted chronologically by capture timestamp.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "set_sequence",
        "description": "Set the final display order, narrative title, and summary. Call exactly once.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filenames": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "All selected filenames in final display order",
                },
                "narrative_title": {
                    "type": "string",
                    "description": "Short evocative title for the album (e.g. 'Three Days in Vegas')",
                },
                "narrative_summary": {
                    "type": "string",
                    "description": "2-3 sentences describing the story arc of the sequence",
                },
                "chapter_breaks": {
                    "type": "array",
                    "description": "Optional chapter breaks between days or locations",
                    "items": {
                        "type": "object",
                        "properties": {
                            "after_filename": {"type": "string"},
                            "chapter_title":  {"type": "string"},
                        },
                    },
                },
            },
            "required": ["filenames", "narrative_title", "narrative_summary"],
        },
    },
]


class SequencingAgent:
    """Claude tool-use agent that orders selected photos into a narrative arc."""

    def sequence(
        self,
        shortlist: list[dict],
        client,
        on_event: Callable[[str, dict], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict:
        state = _State(shortlist)
        categories: dict[str, int] = {}
        for p in shortlist:
            cat = p.get("narrative_category", "other")
            categories[cat] = categories.get(cat, 0) + 1

        initial = (
            f"Please arrange these {len(shortlist)} selected photos into the best narrative sequence.\n"
            f"Narrative categories present: {json.dumps(categories)}\n\n"
            "Start with get_shortlist to see all photos with their metadata, then call set_sequence."
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
                on_event("sequencing", {"stop_reason": resp.stop_reason})
            if resp.stop_reason != "tool_use":
                break
            results = []
            finalized = False
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                if on_event:
                    on_event("sequencing", {"tool": block.name, "input": block.input})
                outcome = state.handle(block.name, block.input)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(outcome),
                })
                if block.name == "set_sequence":
                    finalized = True
            messages.append({"role": "user", "content": results})
            if finalized:
                break

        return state.get_result()

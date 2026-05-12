"""Retrieval Trace Tab — visual pipeline flow for the RAG debug dashboard.

Renders each retrieval stage as a card connected by vertical arrows:

    Query
      ↓
    Vector Search   [42ms]  [20 docs]
      ↓
    BM25 Search     [3ms]   [20 docs]   (only when hybrid ON)
      ↓
    RRF Merge       [1ms]   [20 docs]   (only when hybrid ON)
      ↓
    Rerank          [8ms]   [5 docs]    (only when reranker ON)
      ↓
    Final Context           [5 docs]

Each stage card shows:
  • badge: elapsed ms
  • badge: doc count
  • score stats: min / avg / max
  • top-3 doc previews (collapsible)

Entry point:

    refresh_fn = build(ctrl)

``refresh_fn`` is the ``render_pipeline.refresh`` callable; ``dashboard.py``
appends it to the ``on_search`` list so the trace updates whenever a search
runs from the Search tab.
"""

from __future__ import annotations

import statistics
from typing import Callable

from nicegui import ui

from ui.search_controller import SearchController, TraceStep

# ---------------------------------------------------------------------------
# Stage appearance config
# ---------------------------------------------------------------------------

_STAGE_STYLE: dict[str, dict] = {
    "Vector Search": {
        "icon":       "search",
        "color":      "blue",
        "header_cls": "bg-blue-50 border-blue-200",
        "badge_cls":  "bg-blue-100 text-blue-700",
        "score_cls":  "text-blue-600",
    },
    "BM25 Search": {
        "icon":       "text_fields",
        "color":      "orange",
        "header_cls": "bg-orange-50 border-orange-200",
        "badge_cls":  "bg-orange-100 text-orange-700",
        "score_cls":  "text-orange-600",
    },
    "RRF Merge": {
        "icon":       "merge",
        "color":      "amber",
        "header_cls": "bg-amber-50 border-amber-200",
        "badge_cls":  "bg-amber-100 text-amber-700",
        "score_cls":  "text-amber-600",
    },
    "Rerank": {
        "icon":       "sort",
        "color":      "purple",
        "header_cls": "bg-purple-50 border-purple-200",
        "badge_cls":  "bg-purple-100 text-purple-700",
        "score_cls":  "text-purple-600",
    },
    "Final Context": {
        "icon":       "done_all",
        "color":      "green",
        "header_cls": "bg-green-50 border-green-200",
        "badge_cls":  "bg-green-100 text-green-700",
        "score_cls":  "text-green-600",
    },
}

_DEFAULT_STYLE = {
    "icon":       "radio_button_unchecked",
    "color":      "gray",
    "header_cls": "bg-gray-50 border-gray-200",
    "badge_cls":  "bg-gray-100 text-gray-600",
    "score_cls":  "text-gray-600",
}


def _style(stage: str) -> dict:
    return _STAGE_STYLE.get(stage, _DEFAULT_STYLE)


# ---------------------------------------------------------------------------
# Score stats helper
# ---------------------------------------------------------------------------

def _score_stats(docs: list[tuple]) -> str | None:
    """Return 'min x.xx  avg x.xx  max x.xx' or None for empty list."""
    scores = [s for _, s in docs if isinstance(s, (int, float))]
    if not scores:
        return None
    lo  = min(scores)
    hi  = max(scores)
    avg = statistics.mean(scores)
    return f"min {lo:.3f}   avg {avg:.3f}   max {hi:.3f}"


# ---------------------------------------------------------------------------
# Single stage card
# ---------------------------------------------------------------------------

def _step_card(step: TraceStep, step_num: int) -> None:
    """Render one pipeline stage as a NiceGUI card."""
    st = _style(step.stage)

    with ui.card().classes(
        f"w-full border {st['header_cls']} rounded-lg p-0 overflow-hidden"
    ):
        # Header row
        with ui.row().classes(
            f"w-full items-center gap-2 px-3 py-2 {st['header_cls']} border-b"
        ):
            ui.icon(st["icon"]).classes(f"text-{st['color']}-500 text-base")
            ui.label(f"{step_num}. {step.stage}").classes(
                "text-sm font-semibold text-gray-700 flex-1"
            )

            # Timing badge (hidden for Final Context)
            if step.elapsed_ms > 0:
                ui.label(f"{step.elapsed_ms:.1f} ms").classes(
                    f"text-xs font-mono rounded-full px-2 py-0.5 {st['badge_cls']}"
                )

            # Doc count badge
            ui.label(f"{step.out_count} docs").classes(
                f"text-xs font-mono rounded-full px-2 py-0.5 {st['badge_cls']}"
            )

        # Body
        with ui.column().classes("w-full px-3 py-2 gap-1"):

            # Stage-specific secondary info
            params = step.params
            if step.stage == "RRF Merge":
                overlap      = params.get("overlap", 0)
                total_unique = params.get("total_unique", 0)
                ui.label(
                    f"overlap: {overlap} / {total_unique} unique chunks across vector + BM25"
                ).classes("text-xs text-gray-500 italic")
            elif step.stage == "Final Context":
                ui.label("→ context window sent to LLM").classes(
                    "text-xs text-green-600 font-medium"
                )
            elif step.in_count > 0 and step.stage != "Vector Search" and step.stage != "BM25 Search":
                ui.label(
                    f"from {step.in_count} candidates → {step.out_count} kept"
                ).classes("text-xs text-gray-500 italic")

            # Score distribution
            stats_str = _score_stats(step.docs)
            if stats_str:
                ui.label(stats_str).classes(
                    f"text-xs font-mono {st['score_cls']}"
                )

            # Top docs (expandable)
            if step.docs:
                with ui.expansion(
                    f"Top {min(len(step.docs), 5)} results",
                    icon="expand_more",
                ).classes("w-full text-xs text-gray-500").props("dense"):
                    for rank, (doc, score) in enumerate(step.docs[:5]):
                        meta    = doc.metadata
                        cid     = meta.get("chunk_id", "?")
                        doc_id  = meta.get("doc_id", "?")
                        preview = doc.page_content[:120].replace("\n", " ")
                        ellip   = "…" if len(doc.page_content) > 120 else ""

                        with ui.row().classes(
                            "items-start gap-2 py-1 border-b border-gray-100 last:border-0"
                        ):
                            ui.label(f"#{rank + 1}").classes(
                                "text-xs font-bold text-gray-400 w-5 flex-shrink-0"
                            )
                            ui.label(f"{score:.4f}").classes(
                                f"text-xs font-mono flex-shrink-0 {st['score_cls']}"
                            )
                            with ui.column().classes("flex-1 gap-0"):
                                ui.label(
                                    f"c{cid}  ·  {doc_id}"
                                ).classes("text-xs text-gray-400")
                                ui.label(preview + ellip).classes(
                                    "text-xs text-gray-700 leading-snug"
                                )


# ---------------------------------------------------------------------------
# Arrow connector
# ---------------------------------------------------------------------------

def _arrow() -> None:
    with ui.column().classes("items-center w-full my-0.5"):
        ui.icon("arrow_downward").classes("text-gray-300 text-xl")


# ---------------------------------------------------------------------------
# Tab entry point
# ---------------------------------------------------------------------------

def build(ctrl: SearchController) -> Callable:
    """Build the Retrieval Trace tab into the current NiceGUI context.

    Returns ``render_pipeline.refresh`` so the caller (dashboard.py) can
    trigger a redraw after each search without importing NiceGUI internals.
    """

    with ui.scroll_area().style(
        "height: 100%; width: 100%; padding: 0.75rem;"
    ):
        @ui.refreshable
        def render_pipeline():
            trace = ctrl.trace

            if not trace:
                with ui.column().classes("items-center justify-center w-full mt-16"):
                    ui.icon("account_tree").classes("text-5xl text-gray-200")
                    ui.label("Run a search to see the pipeline trace.").classes(
                        "text-gray-400 italic text-sm mt-2"
                    )
                return

            # ── Query card ────────────────────────────────────────────────
            with ui.card().classes(
                "w-full border bg-gray-50 border-gray-200 rounded-lg px-3 py-2"
            ):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("search").classes("text-gray-400 text-base")
                    ui.label("Query").classes(
                        "text-xs font-semibold uppercase tracking-wide text-gray-400 w-16"
                    )
                    ui.label(ctrl.last_query).classes(
                        "text-sm text-gray-800 font-medium flex-1"
                    )

            # ── Pipeline steps ────────────────────────────────────────────
            total_ms = sum(s.elapsed_ms for s in trace)
            for idx, step in enumerate(trace):
                _arrow()
                _step_card(step, idx + 1)

            # ── Total time footer ─────────────────────────────────────────
            if total_ms > 0:
                ui.label(f"Total retrieval time: {total_ms:.1f} ms").classes(
                    "text-xs text-gray-400 italic text-right w-full mt-2 pr-1"
                )

        render_pipeline()

    return render_pipeline.refresh

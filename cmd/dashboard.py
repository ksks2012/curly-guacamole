"""
RAG Debug Dashboard  —  Phase 1 MVP

Layout:
┌─────────────────────────────────────────────────────┐
│  QUERY BAR  [ input ]  [ top-k ]  [ Search ]        │  ← pinned top
├──────────────────────────────┬──────────────────────┤
│  RESULT LIST  (left, scroll) │  CHUNK DETAIL (right)│
│  #1  score  page  chunk      │  metadata key:value  │
│  <content preview>           │  content preview     │
│  #2 ...                      │                      │
└──────────────────────────────┴──────────────────────┘

Run:
    python dashboard.py
"""

from nicegui import ui

from utils.config import AppConfig
from rag.client import LocalLlamaClient

# ---------------------------------------------------------------------------
# Bootstrap client (shared across all UI interactions)
# ---------------------------------------------------------------------------
_config = AppConfig()
_client = LocalLlamaClient(_config)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_results: list[tuple] = []          # list of (rank, doc, score)
_selected_metadata: dict = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_color(score: float) -> str:
    """Return a Tailwind text-color class based on relevance score."""
    if score >= 0.75:
        return "text-green-600"
    if score >= 0.50:
        return "text-yellow-600"
    return "text-red-500"


def _run_search(query: str, k: int, result_area, metadata_area) -> None:
    """Run similarity_search_with_scores and refresh the result area."""
    global _results, _selected_metadata

    query = query.strip()
    if not query:
        ui.notify("Please enter a query.", type="warning")
        return

    _selected_metadata = {}
    _results = []

    try:
        raw = _client.similarity_search_with_scores(query, k=k)
    except Exception as e:
        ui.notify(f"Search error: {e}", type="negative")
        return

    _results = [(rank + 1, doc, score) for rank, (doc, score) in enumerate(raw)]

    _render_results(result_area, metadata_area)
    _render_metadata(metadata_area)


def _render_results(result_area, metadata_area) -> None:
    """Re-render the result list cards."""
    result_area.clear()

    if not _results:
        with result_area:
            ui.label("No results.").classes("text-gray-400 italic mt-4")
        return

    with result_area:
        for rank, doc, score in _results:
            color = _score_color(score)
            page = doc.metadata.get("page", "?")
            chunk_id = doc.metadata.get("chunk_id", "?")
            filename = doc.metadata.get("filename", "?")
            preview = doc.page_content[:300].replace("\n", " ")

            with ui.card().classes("w-full cursor-pointer hover:shadow-md mb-2").on(
                "click", lambda d=doc, s=score, ma=metadata_area: _select_chunk(d, s, ma)
            ):
                with ui.row().classes("items-center gap-4"):
                    ui.label(f"#{rank}").classes("text-lg font-bold w-8 text-gray-500")
                    ui.label(f"score: {score}").classes(f"font-mono font-semibold {color}")
                    ui.label(f"page {int(page) + 1}").classes("text-gray-500 text-sm")
                    ui.label(f"chunk {chunk_id}").classes("text-gray-400 text-sm")
                    ui.label(filename).classes("text-blue-400 text-xs truncate max-w-xs")
                ui.separator()
                ui.label(preview + ("…" if len(doc.page_content) > 300 else "")).classes(
                    "text-sm text-gray-700 whitespace-pre-wrap"
                )


def _select_chunk(doc, score, metadata_area) -> None:
    """Show full metadata for the clicked chunk."""
    global _selected_metadata
    _selected_metadata = {
        "score": score,
        **{k: v for k, v in doc.metadata.items()},
        "_content_length": len(doc.page_content),
        "_content_preview": doc.page_content[:500],
    }
    _render_metadata(metadata_area)


def _render_metadata(metadata_area) -> None:
    """Re-render the metadata panel."""
    metadata_area.clear()
    with metadata_area:
        if not _selected_metadata:
            ui.label("Click a result to inspect its metadata.").classes(
                "text-gray-400 italic"
            )
            return

        ui.label("Selected chunk metadata").classes("text-base font-semibold mb-2")
        with ui.grid(columns=2).classes("gap-x-4 gap-y-1 text-sm w-full"):
            for key, val in _selected_metadata.items():
                if key == "_content_preview":
                    continue
                ui.label(key).classes("font-mono text-gray-500 text-right")
                ui.label(str(val)).classes("font-mono text-gray-800 break-all")

        ui.separator().classes("my-2")
        ui.label("Content preview").classes("text-sm font-semibold text-gray-500")
        ui.label(_selected_metadata.get("_content_preview", "")).classes(
            "text-sm text-gray-700 whitespace-pre-wrap bg-gray-50 rounded p-2 w-full"
        )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

@ui.page("/")
def dashboard():
    ui.page_title("RAG Debug Dashboard")

    # Populated after the body row is built; accessed only at click-time.
    areas: dict = {}

    # Full-viewport column: query bar pinned top, body fills the rest.
    with ui.column().style(
        "width: 100%; height: 100vh; display: flex; flex-direction: column;"
        " gap: 0; padding: 0; overflow: hidden;"
    ):
        # ── Query bar (pinned top) ─────────────────────────────────────────
        with ui.card().classes("w-full rounded-none shadow-md p-4").style(
            "flex-shrink: 0; z-index: 10;"
        ):
            ui.label("RAG Debug Dashboard").classes(
                "text-xl font-bold text-gray-800 mb-2"
            )
            with ui.row().classes("w-full items-end gap-3"):
                query_input = ui.input(
                    placeholder="Enter your query…"
                ).classes("flex-1")

                top_k_input = ui.number(
                    label="top-k", value=5, min=1, max=20, step=1
                ).classes("w-24")

                ui.button(
                    "Search",
                    on_click=lambda: _run_search(
                        query_input.value,
                        int(top_k_input.value or 5),
                        areas["result"],
                        areas["meta"],
                    ),
                ).classes("bg-blue-600 text-white px-6")

            query_input.on(
                "keydown.enter",
                lambda: _run_search(
                    query_input.value,
                    int(top_k_input.value or 5),
                    areas["result"],
                    areas["meta"],
                ),
            )

        # ── Body: result list (left) | chunk detail (right) ───────────────
        with ui.row().style(
            "flex: 1; min-height: 0; gap: 1rem; padding: 1rem; overflow: hidden;"
        ):
            # Left — scrollable result cards
            with ui.column().style(
                "flex: 1; min-height: 0; overflow-y: auto; gap: 0.5rem;"
            ) as result_area:
                ui.label("Results will appear here after you search.").classes(
                    "text-gray-400 italic mt-2"
                )
            areas["result"] = result_area

            # Right — selected chunk metadata & content
            with ui.card().style(
                "width: 24rem; flex-shrink: 0; min-height: 0;"
                " overflow-y: auto; padding: 1rem;"
            ) as metadata_area:
                ui.label("Click a result to inspect its metadata.").classes(
                    "text-gray-400 italic"
                )
            areas["meta"] = metadata_area


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="RAG Debug Dashboard", port=8888, reload=False)

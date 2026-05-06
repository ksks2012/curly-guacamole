"""
RAG Debug Dashboard  —  Phase 2: Rerank Visualization

Layout (rerank OFF):
┌────────────────────────────────────────────────────────────────┐
│  QUERY BAR  [ input ] [fetch-k] [top-k] [☐ Rerank] [Search]  │
├───────────────────────────────────┬────────────────────────────┤
│  VECTOR RESULTS (scroll)          │  CHUNK DETAIL              │
│  #1  vscore:0.82  p3  chunk12     │  key: value…               │
│  #2  …                            │  content preview           │
└───────────────────────────────────┴────────────────────────────┘

Layout (rerank ON):
┌────────────────────────────────────────────────────────────────┐
│  QUERY BAR  [ input ] [fetch-k] [top-k] [☑ Rerank] [Search]  │
├──────────────────┬──────────────────────┬──────────────────────┤
│  VECTOR (fetch_k)│  RERANKED (top_k)    │  CHUNK DETAIL        │
│  #1  0.82        │  #1  ▲7  rs:3.21     │  key: val            │
│  #2  0.71  ←blue │  #2  ▼1  rs:2.90     │  …                   │
│  #3  0.65        │  …                   │  content preview     │
│  …               │                      │                      │
└──────────────────┴──────────────────────┴──────────────────────┘
  blue border = chunk passed rerank filter

Run:
    python cmd/dashboard.py
"""

from nicegui import ui

from utils.logger import AppLogger

from utils.config import AppConfig
from rag.client import LocalLlamaClient

# ---------------------------------------------------------------------------
# Logging (configure before anything else so client init messages appear)
# ---------------------------------------------------------------------------
_config = AppConfig()
AppLogger.setup(
    level=_config.log_level,
    fmt=_config.log_format,
    datefmt=_config.log_datefmt,
)
log = AppLogger.get("dashboard")

# ---------------------------------------------------------------------------
# Bootstrap client (expensive — loaded once at module start)
# ---------------------------------------------------------------------------
log.info("Loading config and building RAG client…")
_client = LocalLlamaClient(_config)
log.info("RAG client ready")

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
_state: dict = {
    "vector": [],      # list[tuple[Document, float]]  — (doc, relevance_score)
    "reranked": None,  # list[tuple[Document, float]] | None  — (doc, rerank_score)
    "metadata": {},    # currently selected chunk (displayed in detail panel)
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vscore_color(score: float) -> str:
    """Tailwind text-color class for a 0-1 vector relevance score."""
    if score >= 0.75:
        return "text-green-600"
    if score >= 0.50:
        return "text-yellow-600"
    return "text-red-500"


def _rank_change(chunk_id, vector_results: list, rerank_pos: int) -> tuple[str, str]:
    """Return (label, css_class) showing how a chunk's rank changed after reranking.

    delta > 0  →  moved up   (▲N, green)
    delta < 0  →  moved down (▼N, red)
    delta == 0 →  unchanged  (—, gray)
    not found  →  new entry  (★, blue) — shouldn't happen in practice
    """
    for v_pos, (doc, _) in enumerate(vector_results):
        if doc.metadata.get("chunk_id") == chunk_id:
            delta = v_pos - rerank_pos
            if delta > 0:
                return f"▲{delta}", "text-green-600 font-bold"
            if delta < 0:
                return f"▼{abs(delta)}", "text-red-500 font-bold"
            return "—", "text-gray-400"
    return "★", "text-blue-400"


def _run_search(
    query: str,
    k: int,
    fetch_k: int,
    use_rerank: bool,
    on_done,
) -> None:
    """Execute search_for_debug, store results in _state, then call on_done()."""
    query = query.strip()
    if not query:
        ui.notify("Please enter a query.", type="warning")
        return

    log.info("Search: query=%r  k=%d  fetch_k=%d  use_rerank=%s", query, k, fetch_k, use_rerank)
    _state["vector"] = []
    _state["reranked"] = None
    _state["metadata"] = {}

    ui.notify("Searching…", type="info", timeout=1500)
    try:
        result = _client.search_for_debug(
            query, k=k, fetch_k=fetch_k, use_rerank=use_rerank
        )
    except Exception as e:
        log.error("Search failed: %s", e, exc_info=True)
        ui.notify(f"Search error: {e}", type="negative")
        return

    _state["vector"] = result["vector"]
    _state["reranked"] = result["reranked"]
    log.info("Search done: %d vector results, reranked=%s",
             len(_state["vector"]),
             len(_state["reranked"]) if _state["reranked"] is not None else "off")

    if use_rerank and _state["reranked"] is None:
        ui.notify(
            "Reranker not available — check config.reranker_type.", type="warning"
        )

    on_done()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

@ui.page("/")
def dashboard():
    ui.page_title("RAG Debug Dashboard")

    with ui.column().style(
        "width: 100%; height: 100vh; display: flex; flex-direction: column;"
        " gap: 0; padding: 0; overflow: hidden;"
    ):
        # ── Query bar (pinned top) ─────────────────────────────────────────
        with ui.card().classes("w-full rounded-none shadow-md p-3").style(
            "flex-shrink: 0;"
        ):
            ui.label("RAG Debug Dashboard").classes(
                "text-lg font-bold text-gray-800 mb-2"
            )
            with ui.row().classes("w-full items-end gap-3 flex-wrap"):
                query_input = ui.input(
                    placeholder="Enter your query…"
                ).classes("flex-1 min-w-[12rem]")
                fetch_k_input = ui.number(
                    label="fetch-k", value=20, min=5, max=100, step=5
                ).classes("w-24")
                top_k_input = ui.number(
                    label="top-k", value=5, min=1, max=20, step=1
                ).classes("w-24")
                rerank_toggle = ui.checkbox("Rerank")

                def do_search():
                    rerank_on = rerank_toggle.value
                    k = int(top_k_input.value or 5)
                    fetch_k = int(fetch_k_input.value or 20)

                    def on_done():
                        render_results.refresh(rerank_on)
                        render_detail.refresh()

                    _run_search(query_input.value, k, fetch_k, rerank_on, on_done)

                ui.button("Search", on_click=do_search).classes(
                    "bg-blue-600 text-white px-6"
                )

            query_input.on("keydown.enter", do_search)

        # ── Body ──────────────────────────────────────────────────────────
        with ui.row().style(
            "flex: 1; min-height: 0; gap: 0.75rem; padding: 0.75rem; overflow: hidden;"
        ):
            # Left / center: result columns (refreshable — rebuilt on every search)
            @ui.refreshable
            def render_results(rerank_on: bool = False):
                vector = _state["vector"]
                reranked = _state["reranked"]
                reranked_ids = {
                    doc.metadata.get("chunk_id") for doc, _ in (reranked or [])
                }

                # ── Vector column ──────────────────────────────────────────
                with ui.column().style(
                    "flex: 1; min-height: 0; overflow-y: auto;"
                ):
                    header = f"VECTOR  ({len(vector)})"
                    if reranked_ids:
                        header += f"  ·  {len(reranked_ids)} passed rerank"
                    ui.label(header).classes(
                        "text-xs font-semibold uppercase tracking-wide text-gray-500 mb-2"
                    )

                    if not vector:
                        ui.label("Results will appear after search.").classes(
                            "text-gray-400 italic text-sm"
                        )
                    else:
                        for v_rank, (doc, vscore) in enumerate(vector):
                            chunk_id = doc.metadata.get("chunk_id", "?")
                            page = doc.metadata.get("page", "?")
                            filename = doc.metadata.get("filename", "?")
                            preview = doc.page_content[:180].replace("\n", " ")
                            in_reranked = chunk_id in reranked_ids
                            border = (
                                "border-l-4 border-blue-300"
                                if in_reranked
                                else "border-l-4 border-transparent"
                            )
                            page_str = str(int(page) + 1) if page != "?" else "?"

                            with ui.card().classes(
                                f"w-full cursor-pointer hover:shadow-md mb-1 {border}"
                            ).on(
                                "click",
                                lambda d=doc, s=vscore: (
                                    _state.update({"metadata": {
                                        "vscore": s,
                                        **d.metadata,
                                        "_content_len": len(d.page_content),
                                        "_content": d.page_content[:600],
                                    }}),
                                    render_detail.refresh(),
                                ),
                            ):
                                with ui.row().classes("items-center gap-2 flex-wrap"):
                                    ui.label(f"#{v_rank + 1}").classes(
                                        "text-xs font-bold w-5 text-gray-400"
                                    )
                                    ui.label(f"{vscore}").classes(
                                        f"font-mono text-xs font-semibold {_vscore_color(vscore)}"
                                    )
                                    ui.label(f"p{page_str}").classes("text-gray-400 text-xs")
                                    ui.label(f"c{chunk_id}").classes("text-gray-400 text-xs")
                                    ui.label(filename).classes(
                                        "text-blue-400 text-xs truncate max-w-[9rem]"
                                    )
                                ui.label(
                                    preview + ("…" if len(doc.page_content) > 180 else "")
                                ).classes("text-xs text-gray-600 leading-snug mt-0.5")

                # ── Reranked column (only when rerank is ON) ───────────────
                if rerank_on:
                    with ui.column().style(
                        "flex: 1; min-height: 0; overflow-y: auto;"
                    ):
                        count = len(reranked) if reranked else 0
                        ui.label(f"RERANKED  ({count})").classes(
                            "text-xs font-semibold uppercase tracking-wide text-gray-500 mb-2"
                        )

                        if not reranked:
                            ui.label("No rerank results.").classes(
                                "text-gray-400 italic text-sm"
                            )
                        else:
                            for r_rank, (doc, rscore) in enumerate(reranked):
                                chunk_id = doc.metadata.get("chunk_id", "?")
                                page = doc.metadata.get("page", "?")
                                filename = doc.metadata.get("filename", "?")
                                preview = doc.page_content[:180].replace("\n", " ")
                                change_label, change_color = _rank_change(
                                    chunk_id, vector, r_rank
                                )
                                page_str = str(int(page) + 1) if page != "?" else "?"

                                with ui.card().classes(
                                    "w-full cursor-pointer hover:shadow-md mb-1"
                                    " border-l-4 border-green-300"
                                ).on(
                                    "click",
                                    lambda d=doc, s=rscore: (
                                        _state.update({"metadata": {
                                            "rscore": s,
                                            **d.metadata,
                                            "_content_len": len(d.page_content),
                                            "_content": d.page_content[:600],
                                        }}),
                                        render_detail.refresh(),
                                    ),
                                ):
                                    with ui.row().classes("items-center gap-2 flex-wrap"):
                                        ui.label(f"#{r_rank + 1}").classes(
                                            "text-xs font-bold w-5 text-gray-700"
                                        )
                                        ui.label(change_label).classes(
                                            f"text-xs font-mono {change_color} w-7"
                                        )
                                        ui.label(f"{rscore:.2f}").classes(
                                            "font-mono text-xs font-semibold text-purple-600"
                                        )
                                        ui.label(f"p{page_str}").classes("text-gray-400 text-xs")
                                        ui.label(f"c{chunk_id}").classes("text-gray-400 text-xs")
                                        ui.label(filename).classes(
                                            "text-blue-400 text-xs truncate max-w-[9rem]"
                                        )
                                    ui.label(
                                        preview + ("…" if len(doc.page_content) > 180 else "")
                                    ).classes("text-xs text-gray-600 leading-snug mt-0.5")

            render_results(False)

            # Right: chunk detail (independent refreshable inside the card)
            with ui.card().style(
                "width: 22rem; flex-shrink: 0; min-height: 0;"
                " overflow-y: auto; padding: 0.75rem;"
            ):
                @ui.refreshable
                def render_detail():
                    meta = _state["metadata"]
                    if not meta:
                        ui.label("Click a result to inspect.").classes(
                            "text-gray-400 italic text-sm"
                        )
                        return

                    ui.label("Chunk detail").classes(
                        "text-sm font-semibold text-gray-700 mb-2"
                    )
                    with ui.grid(columns=2).classes("gap-x-3 gap-y-0.5 text-xs w-full"):
                        for key, val in meta.items():
                            if key == "_content":
                                continue
                            ui.label(key).classes(
                                "font-mono text-gray-400 text-right truncate"
                            )
                            ui.label(str(val)).classes("font-mono text-gray-800 break-all")

                    ui.separator().classes("my-2")
                    ui.label("Content").classes("text-xs font-semibold text-gray-400 mb-1")
                    ui.label(meta.get("_content", "")).classes(
                        "text-xs text-gray-700 whitespace-pre-wrap bg-gray-50 rounded p-2 w-full"
                    )

                render_detail()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="RAG Debug Dashboard", port=8888, reload=False)


"""
RAG Debug Dashboard  —  Phase 2: Rerank Visualization

Display layer only — no business logic, no RAG state.
All logic is delegated to SearchController (search_controller.py).
All data access is through the RAG backend (rag/).

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

from utils.config import AppConfig
from utils.logger import AppLogger
from rag.client import LocalLlamaClient
from ui.search_controller import RERANKER_UNAVAILABLE, SearchController
from index_controller import IndexController

# ---------------------------------------------------------------------------
# Bootstrap: config → logging → client → controller
# ---------------------------------------------------------------------------
_config = AppConfig()
AppLogger.setup(
    level=_config.log_level,
    fmt=_config.log_format,
    datefmt=_config.log_datefmt,
)
log = AppLogger.get("dashboard")

log.info("Loading config and building RAG client…")
_client = LocalLlamaClient(_config)
_ctrl = SearchController(_client)
_idx_ctrl = IndexController(_client)
log.info("RAG client ready")


# ---------------------------------------------------------------------------
# Page (display layer — NiceGUI only)
# ---------------------------------------------------------------------------

@ui.page("/")
def dashboard():
    ui.page_title("RAG Debug Dashboard")

    with ui.column().style(
        "width: 100%; height: 100vh; display: flex; flex-direction: column;"
        " gap: 0; padding: 0; overflow: hidden;"
    ):
        # ── Tab bar (pinned top) ───────────────────────────────────────────
        with ui.tabs().classes("w-full bg-white shadow-sm").style(
            "flex-shrink: 0;"
        ) as tabs:
            search_tab = ui.tab("Search").props("no-caps")
            index_tab = ui.tab("Index").props("no-caps")

        with ui.tab_panels(tabs, value=search_tab).classes("p-0 w-full").style(
            "flex: 1; min-height: 0; overflow: hidden;"
            " display: flex; flex-direction: column;"
        ):
            # ─────────────────────────────────────────────────────────────
            # SEARCH tab
            # ─────────────────────────────────────────────────────────────
            with ui.tab_panel(search_tab).classes("p-0").style(
                "height: 100%; display: flex; flex-direction: column; overflow: hidden;"
            ):
                # ── Query bar ─────────────────────────────────────────────
                with ui.card().classes("w-full rounded-none shadow-md p-3").style(
                    "flex-shrink: 0;"
                ):
                    ui.label("RAG Debug Dashboard").classes(
                        "text-lg font-bold text-gray-800 mb-2"
                    )
                    # Row 1: search controls
                    with ui.row().classes("w-full items-end gap-3 flex-wrap"):
                        query_input = ui.input(
                            placeholder="Enter your query\u2026"
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

                            ui.notify("Searching\u2026", type="info", timeout=1500)
                            error = _ctrl.run_search(query_input.value, k, fetch_k, rerank_on)

                            if error == "Query is empty.":
                                ui.notify("Please enter a query.", type="warning")
                                return
                            if error == RERANKER_UNAVAILABLE:
                                ui.notify(
                                    "Reranker not available \u2014 check config.reranker_type.",
                                    type="warning",
                                )
                            elif error:
                                ui.notify(f"Search error: {error}", type="negative")
                                return

                            render_results.refresh(rerank_on)
                            render_detail.refresh()

                        ui.button("Search", on_click=do_search).classes(
                            "bg-blue-600 text-white px-6"
                        )

                    query_input.on("keydown.enter", do_search)

                    # Row 2: filter controls
                    with ui.row().classes("w-full items-center gap-3 flex-wrap mt-2"):
                        filter_toggle = ui.checkbox("Filter by doc")

                        _doc_ids = _ctrl.list_doc_ids()
                        filter_select = ui.select(
                            options=_doc_ids,
                            label="source doc",
                            value=_doc_ids[0] if _doc_ids else None,
                        ).classes("w-64").props("outlined dense")
                        filter_select.set_visibility(False)

                        @ui.refreshable
                        def render_filter_status():
                            active = _ctrl.filter_doc_id
                            if active:
                                ui.label(f"Filter: doc_id = {active}").classes(
                                    "text-xs font-mono bg-blue-50 text-blue-700"
                                    " border border-blue-200 rounded px-2 py-0.5"
                                )
                            else:
                                ui.label("Filter: off \u2014 searching all documents").classes(
                                    "text-xs text-gray-400 italic"
                                )

                        render_filter_status()

                        def on_filter_toggle(enabled: bool):
                            filter_select.set_visibility(enabled)
                            if enabled:
                                _ctrl.set_filter(filter_select.value)
                            else:
                                _ctrl.set_filter(None)
                            render_filter_status.refresh()

                        def on_filter_select(doc_id: str):
                            if filter_toggle.value:
                                _ctrl.set_filter(doc_id)
                                render_filter_status.refresh()

                        filter_toggle.on_value_change(lambda e: on_filter_toggle(e.value))
                        filter_select.on_value_change(lambda e: on_filter_select(e.value))

                # ── Body ──────────────────────────────────────────────────
                with ui.row().style(
                    "flex: 1; min-height: 0; gap: 0.75rem; padding: 0.75rem;"
                    " overflow: hidden; align-items: stretch;"
                ):
                    @ui.refreshable
                    def render_results(rerank_on: bool = False):
                        vector = _ctrl.vector_results
                        reranked = _ctrl.reranked_results
                        reranked_ids = _ctrl.reranked_chunk_ids

                        with ui.element("div").style(
                            "display: flex; flex-direction: row; height: 100%;"
                            " gap: 0.75rem; overflow: hidden; flex: 1;"
                        ):
                            # ── Vector column ──────────────────────────────
                            with ui.column().style("flex: 1; min-height: 0; overflow-y: auto;"):
                                header = f"VECTOR  ({len(vector)})"
                                if reranked_ids:
                                    header += f"  \u00b7  {len(reranked_ids)} passed rerank"
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
                                                _ctrl.select_chunk(d, s, "vscore"),
                                                render_detail.refresh(),
                                            ),
                                        ):
                                            with ui.row().classes("items-center gap-2 flex-wrap"):
                                                ui.label(f"#{v_rank + 1}").classes(
                                                    "text-xs font-bold w-5 text-gray-400"
                                                )
                                                ui.label(f"{vscore}").classes(
                                                    f"font-mono text-xs font-semibold"
                                                    f" {SearchController.score_color(vscore)}"
                                                )
                                                ui.label(f"p{page_str}").classes("text-gray-400 text-xs")
                                                ui.label(f"c{chunk_id}").classes("text-gray-400 text-xs")
                                                ui.label(filename).classes(
                                                    "text-blue-400 text-xs truncate max-w-[9rem]"
                                                )
                                            ui.label(
                                                preview + ("\u2026" if len(doc.page_content) > 180 else "")
                                            ).classes("text-xs text-gray-600 leading-snug mt-0.5")

                            # ── Reranked column (only when rerank is ON) ───
                            if rerank_on:
                                with ui.column().style("flex: 1; min-height: 0; overflow-y: auto;"):
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
                                            change_label, change_color = SearchController.rank_change(
                                                chunk_id, vector, r_rank
                                            )
                                            page_str = str(int(page) + 1) if page != "?" else "?"

                                            with ui.card().classes(
                                                "w-full cursor-pointer hover:shadow-md mb-1"
                                                " border-l-4 border-green-300"
                                            ).on(
                                                "click",
                                                lambda d=doc, s=rscore: (
                                                    _ctrl.select_chunk(d, s, "rscore"),
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
                                                    preview + ("\u2026" if len(doc.page_content) > 180 else "")
                                                ).classes("text-xs text-gray-600 leading-snug mt-0.5")

                    render_results(False)

                    # Right — chunk detail panel
                    with ui.card().style(
                        "width: 22rem; flex-shrink: 0; min-height: 0;"
                        " overflow-y: auto; padding: 0.75rem;"
                    ):
                        @ui.refreshable
                        def render_detail():
                            meta = _ctrl.selected_metadata
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

            # ─────────────────────────────────────────────────────────────
            # INDEX tab
            # ─────────────────────────────────────────────────────────────
            with ui.tab_panel(index_tab).style(
                "height: 100%; overflow-y: auto; padding: 1rem;"
            ):
                ui.label("Index Documents").classes("text-lg font-bold text-gray-800 mb-4")

                # ── Chunking options ───────────────────────────────────────
                with ui.card().classes("w-full max-w-2xl mb-4 p-4"):
                    ui.label("Chunking options").classes(
                        "text-sm font-semibold text-gray-600 mb-3"
                    )
                    with ui.row().classes("items-end gap-4 flex-wrap"):
                        chunk_size_input = ui.number(
                            label="Chunk size", value=500, min=100, max=2000, step=100
                        ).classes("w-36")
                        overlap_input = ui.number(
                            label="Overlap", value=100, min=0, max=500, step=50
                        ).classes("w-36")
                        doc_id_input = ui.input(
                            label="doc-id override (optional)",
                            placeholder="defaults to filename",
                        ).classes("flex-1 min-w-[14rem]")

                # ── Upload ─────────────────────────────────────────────────
                with ui.card().classes("w-full max-w-2xl mb-4 p-4"):
                    ui.label("Upload PDF").classes(
                        "text-sm font-semibold text-gray-600 mb-3"
                    )

                    @ui.refreshable
                    def render_index_status():
                        result = _idx_ctrl.last_result
                        if result:
                            color = (
                                "text-green-700 bg-green-50 border-green-200"
                                if _idx_ctrl.last_ok
                                else "text-red-700 bg-red-50 border-red-200"
                            )
                            ui.label(result).classes(
                                f"text-xs font-mono {color} border rounded px-2 py-1 w-full"
                            )
                        else:
                            ui.label("No file indexed yet.").classes(
                                "text-xs text-gray-400 italic"
                            )

                    def handle_upload(e):
                        file_bytes = e.content.read()
                        ok = _idx_ctrl.index_pdf(
                            file_name=e.name,
                            file_bytes=file_bytes,
                            chunk_size=int(chunk_size_input.value or 500),
                            chunk_overlap=int(overlap_input.value or 100),
                            doc_id=doc_id_input.value or None,
                        )
                        ui.notify(
                            _idx_ctrl.last_result,
                            type="positive" if ok else "negative",
                        )
                        render_index_status.refresh()
                        render_doc_list.refresh()

                    ui.upload(
                        label="Drop PDF here or click to select",
                        on_upload=handle_upload,
                        max_files=1,
                        auto_upload=True,
                    ).props("accept=.pdf flat").classes("w-full")

                    ui.separator().classes("my-3")
                    render_index_status()

                # ── Indexed doc list ───────────────────────────────────────
                with ui.card().classes("w-full max-w-2xl p-4"):
                    with ui.row().classes("items-center justify-between mb-3"):
                        ui.label("Indexed documents").classes(
                            "text-sm font-semibold text-gray-600"
                        )
                        ui.button(
                            icon="refresh",
                            on_click=lambda: render_doc_list.refresh(),
                        ).props("flat dense round")

                    @ui.refreshable
                    def render_doc_list():
                        docs = _idx_ctrl.list_docs()
                        if not docs:
                            ui.label("No documents indexed yet.").classes(
                                "text-gray-400 italic text-sm"
                            )
                        else:
                            with ui.column().classes("gap-1 w-full"):
                                for doc in docs:
                                    ui.label(f"\u2022 {doc}").classes(
                                        "text-sm font-mono text-gray-700"
                                    )

                    render_doc_list()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="RAG Debug Dashboard", port=8888, reload=False)



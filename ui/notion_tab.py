"""
Notion tab UI — sync controls, page list, hybrid search and RAG query.

Layout:
┌─────────────────────┬──────────────────────────────────────────┐
│ LEFT SIDEBAR        │ RIGHT PANEL (tabbed)                     │
│ ─────────────────── │ ┌─ Search ──────────────────────────────┐│
│ [Sync]  [Embed]     │ │ [query input] [k] [fetch-k] [mode]   ││
│ [Sync+Embed] [Full] │ │ → result cards                        ││
│                     │ └───────────────────────────────────────┘│
│ ─────────────────── │ ┌─ Pages ───────────────────────────────┐│
│ Pages (N)           │ │ table: title, last_edited, synced     ││
│  • page title       │ └───────────────────────────────────────┘│
│  • …                │                                          │
└─────────────────────┴──────────────────────────────────────────┘

Exposes a single entry point:

    build(ctrl)
"""

from __future__ import annotations

import asyncio
from typing import Callable

from nicegui import ui

from ui.notion_controller import NotionController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _status_badge(text: str, ok: bool) -> None:
    color = "text-green-700 bg-green-50 border-green-200" if ok else "text-red-700 bg-red-50 border-red-200"
    ui.label(text).classes(
        f"text-xs font-mono {color} border rounded px-2 py-1 w-full break-words"
    )


def _dict_to_lines(d: dict) -> str:
    return "  ".join(f"{k}:{v}" for k, v in d.items() if k != "error")


def _result_card(chunk: dict) -> None:
    meta    = chunk["meta"]
    content = chunk["content"]
    score   = chunk["score"]

    title = (
        meta.get("title") or meta.get("page_title") or meta.get("source") or ""
    ).strip()
    section = (meta.get("section") or "").strip()
    heading = section or title or "(no heading)"
    doc_type = meta.get("document_type", "")
    workspace = meta.get("workspace", "")

    with ui.card().classes(
        "w-full border border-gray-100 hover:shadow-sm transition-shadow"
    ):
        with ui.row().classes("items-start justify-between w-full gap-2 mb-1"):
            with ui.column().classes("gap-0 flex-1 min-w-0"):
                ui.label(heading[:70]).classes(
                    "text-sm font-semibold text-gray-800 truncate"
                )
                if title and section:
                    ui.label(title[:50]).classes("text-xs text-blue-500 truncate")
            ui.label(f"{score:.3f}").classes(
                "text-xs font-mono text-indigo-400 flex-shrink-0"
            )
        ui.label(content[:300] + ("…" if len(content) > 300 else "")).classes(
            "text-xs text-gray-600 leading-relaxed whitespace-pre-wrap"
        )
        if doc_type or workspace:
            ui.label(f"{doc_type}  {workspace}").classes(
                "text-xs text-gray-300 mt-1"
            )


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build(ctrl: NotionController) -> None:
    """Build the complete Notion tab content into the current NiceGUI context."""

    if not ctrl.is_configured():
        with ui.column().classes("p-6 gap-2"):
            ui.label("Notion not configured").classes(
                "text-base font-semibold text-red-600"
            )
            ui.label(
                "Set notion_token in etc/config.yaml to enable this tab."
            ).classes("text-sm text-gray-500")
        return

    # ── Load pages from RawStore on first render (no network call) ─────────
    ctrl.load_pages()

    # ── Outer layout: left sidebar + right panel ───────────────────────────
    with ui.element("div").style(
        "display: flex; flex-direction: row; width: 100%; height: 100%;"
        " overflow: hidden;"
    ):
        # ==================================================================
        # LEFT SIDEBAR — sync controls + page list
        # ==================================================================
        with ui.column().style(
            "width: 22rem; flex-shrink: 0; height: 100%; overflow-y: auto;"
            " padding: 1rem; gap: 0.75rem; border-right: 1px solid #e5e7eb;"
        ):
            ui.label("Notion Sync").classes("text-lg font-bold text-gray-800")

            # ── Status area ────────────────────────────────────────────────
            @ui.refreshable
            def render_sync_status():
                if ctrl.last_sync:
                    ok = "error" not in ctrl.last_sync
                    _status_badge(_dict_to_lines(ctrl.last_sync), ok)
                if ctrl.last_embed:
                    ok = "error" not in ctrl.last_embed
                    _status_badge(_dict_to_lines(ctrl.last_embed), ok)

            # ── Action buttons ─────────────────────────────────────────────
            with ui.card().classes("w-full p-3"):
                ui.label("Actions").classes(
                    "text-xs font-semibold text-gray-500 mb-2"
                )

                full_sync_toggle = ui.checkbox("Full sync (re-process all pages)").classes(
                    "text-xs text-gray-600"
                )

                async def _do_sync():
                    btn_sync.props("loading")
                    client_ctx = ui.context.client
                    result = await asyncio.to_thread(
                        ctrl.sync, full_sync_toggle.value
                    )
                    with client_ctx:
                        render_sync_status.refresh()
                        render_page_list.refresh()
                        ok = "error" not in result
                        ui.notify(
                            _dict_to_lines(result) if ok else result.get("error", "error"),
                            type="positive" if ok else "negative",
                        )
                        btn_sync.props(remove="loading")

                async def _do_embed():
                    btn_embed.props("loading")
                    client_ctx = ui.context.client
                    result = await asyncio.to_thread(ctrl.embed)
                    with client_ctx:
                        render_sync_status.refresh()
                        ok = "error" not in result
                        ui.notify(
                            _dict_to_lines(result) if ok else result.get("error", "error"),
                            type="positive" if ok else "negative",
                        )
                        btn_embed.props(remove="loading")

                async def _do_sync_and_embed():
                    btn_all.props("loading")
                    client_ctx = ui.context.client
                    result = await asyncio.to_thread(
                        ctrl.sync_and_embed, full_sync_toggle.value
                    )
                    with client_ctx:
                        render_sync_status.refresh()
                        render_page_list.refresh()
                        ok = "error" not in result
                        ui.notify(
                            str(result) if ok else str(result.get("error", "error")),
                            type="positive" if ok else "negative",
                        )
                        btn_all.props(remove="loading")

                with ui.row().classes("gap-2 flex-wrap mt-1"):
                    btn_sync  = ui.button("Sync",        on_click=_do_sync).props(
                        "outlined dense no-caps"
                    ).classes("text-xs")
                    btn_embed = ui.button("Embed",       on_click=_do_embed).props(
                        "outlined dense no-caps"
                    ).classes("text-xs")
                    btn_all   = ui.button("Sync + Embed", on_click=_do_sync_and_embed).props(
                        "unelevated dense no-caps color=primary"
                    ).classes("text-xs")

            render_sync_status()

            # ── Page list ──────────────────────────────────────────────────
            @ui.refreshable
            def render_page_list():
                pages = ctrl.pages
                with ui.column().classes("gap-1 w-full"):
                    ui.label(f"Pages ({len(pages)})").classes(
                        "text-xs font-semibold text-gray-500 mt-1"
                    )
                    if not pages:
                        ui.label("No pages synced yet.").classes(
                            "text-xs text-gray-400 italic"
                        )
                        return
                    for page in pages:
                        title = getattr(page, "title", "") or "(untitled)"
                        edited = str(
                            getattr(page, "last_edited_time", "")
                            or getattr(page, "updated_at", "")
                            or ""
                        )[:10]
                        with ui.row().classes(
                            "items-center gap-1 py-0.5 px-1 rounded hover:bg-gray-50"
                        ):
                            ui.icon("article").classes("text-gray-300 text-sm")
                            with ui.column().classes("gap-0 flex-1 min-w-0"):
                                ui.label(title[:40]).classes(
                                    "text-xs text-gray-700 truncate leading-tight"
                                )
                                if edited:
                                    ui.label(edited).classes(
                                        "text-xs text-gray-300"
                                    )

            render_page_list()

        # ==================================================================
        # RIGHT PANEL — search + results
        # ==================================================================
        with ui.column().style(
            "flex: 1; min-width: 0; height: 100%; overflow: hidden;"
            " display: flex; flex-direction: column;"
        ):
            # ── Search bar ─────────────────────────────────────────────────
            with ui.card().classes("w-full p-3 m-2").style("flex-shrink: 0;"):
                ui.label("Search Notion").classes(
                    "text-sm font-semibold text-gray-700 mb-2"
                )
                with ui.row().classes("items-end gap-3 flex-wrap w-full"):
                    query_input = ui.input(
                        placeholder="Enter a query…",
                    ).classes("flex-1 min-w-0").props("outlined dense clearable")

                    k_input = ui.number(
                        label="top-k", value=5, min=1, max=20, step=1,
                    ).classes("w-20").props("outlined dense")

                    fetch_k_input = ui.number(
                        label="fetch-k", value=20, min=5, max=100, step=5,
                    ).classes("w-24").props("outlined dense")

                    mode_select = ui.select(
                        options={"hybrid": "Hybrid search", "rag": "RAG answer"},
                        value="hybrid",
                        label="Mode",
                    ).classes("w-36").props("outlined dense")

                    search_btn = ui.button("Search", icon="search").props(
                        "unelevated no-caps color=primary"
                    )

            # ── Results area ───────────────────────────────────────────────
            with ui.element("div").style(
                "flex: 1; min-height: 0; overflow-y: auto; padding: 0.5rem 0.75rem;"
            ):
                @ui.refreshable
                def render_results():
                    r = ctrl.last_result
                    if not r.query:
                        ui.label("Run a search to see results.").classes(
                            "text-sm text-gray-400 italic p-2"
                        )
                        return

                    if r.error:
                        ui.label(f"Error: {r.error}").classes(
                            "text-sm text-red-600 font-mono p-2"
                        )
                        return

                    # ── RAG answer box ─────────────────────────────────────
                    if r.mode == "rag" and r.answer:
                        with ui.card().classes(
                            "w-full border-l-4 border-indigo-400 bg-indigo-50 p-3 mb-3"
                        ):
                            ui.label("Answer").classes(
                                "text-xs font-semibold text-indigo-600 mb-1"
                            )
                            ui.label(r.answer).classes(
                                "text-sm text-gray-800 whitespace-pre-wrap"
                            )

                    # ── Source chunks ──────────────────────────────────────
                    if r.chunks:
                        ui.label(
                            f"{'Sources' if r.mode == 'rag' else 'Results'}"
                            f"  ({len(r.chunks)})"
                        ).classes(
                            "text-xs font-semibold text-gray-500 mb-1"
                        )
                        for chunk in r.chunks:
                            _result_card(chunk)
                    else:
                        ui.label("No results found.").classes(
                            "text-sm text-gray-400 italic p-2"
                        )

                render_results()

            # ── Wire search button ─────────────────────────────────────────
            async def _do_search():
                q = (query_input.value or "").strip()
                if not q:
                    ui.notify("Enter a query first.", type="warning")
                    return
                k      = int(k_input.value or 5)
                fk     = int(fetch_k_input.value or 20)
                mode   = mode_select.value
                search_btn.props("loading")
                client_ctx = ui.context.client

                if mode == "rag":
                    result = await asyncio.to_thread(ctrl.rag_query, q, k, fk)
                else:
                    result = await asyncio.to_thread(ctrl.hybrid_search, q, k, fk)

                with client_ctx:
                    render_results.refresh()
                    if result.error:
                        ui.notify(result.error, type="negative")
                    search_btn.props(remove="loading")

            search_btn.on("click", _do_search)
            query_input.on("keydown.enter", _do_search)

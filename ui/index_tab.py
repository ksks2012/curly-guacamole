"""Index tab UI — chunking options, document metadata, file upload, document list.

Exposes a single entry point:

    build(idx_ctrl, log, on_doc_indexed=callback)

`on_doc_indexed` is called after a file is successfully embedded so the
caller (dashboard) can refresh cross-tab state such as filter dropdowns.
"""

import asyncio
import logging
from collections.abc import Callable

from nicegui import ui

from ui.index_controller import IndexController

# Mutable dict to hold per-doc enrich state visible to render_doc_list closure
_enrich_state: dict[str, str] = {}  # doc_id -> "running" | "done" | "error: ..."


def build(
    idx_ctrl: IndexController,
    log: logging.Logger,
    *,
    on_doc_indexed: Callable[[], None],
) -> None:
    """Build the complete Index tab content into the current NiceGUI context.

    Args:
        idx_ctrl:        controller for save + embed operations.
        log:             logger for the upload handler.
        on_doc_indexed:  callback fired after a successful embed (cross-tab refresh).
    """
    _selected_doc: dict = {"id": None}

    # ── Left column: forms ─────────────────────────────────────────────────
    with ui.column().style(
        "width: 28rem; flex-shrink: 0; height: 100%; overflow-y: auto;"
        " padding: 1rem; gap: 1rem; border-right: 1px solid #e5e7eb;"
    ):
        ui.label("Index Documents").classes("text-lg font-bold text-gray-800 mb-2")

        # ── Chunking options ───────────────────────────────────────────────
        with ui.card().classes("w-full p-4"):
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
                ).classes("w-full")
                strategy_select = ui.select(
                    options={
                        "auto":      "auto (by file type)",
                        "recursive": "recursive",
                        "heading":   "heading-aware",
                        "semantic":  "semantic (slow)",
                    },
                    value="auto",
                    label="Chunk strategy",
                ).classes("w-44").props("outlined dense")

        # ── Document metadata ──────────────────────────────────────────────
        with ui.card().classes("w-full p-4"):
            ui.label("Document metadata (optional)").classes(
                "text-sm font-semibold text-gray-600 mb-3"
            )
            with ui.row().classes("items-end gap-4 flex-wrap"):
                title_input = ui.input(
                    label="Title",
                    placeholder="defaults to doc-id",
                ).classes("w-full")
                workspace_input = ui.input(
                    label="Workspace",
                    placeholder="e.g. work / personal",
                ).classes("w-40")
            with ui.row().classes("items-end gap-4 flex-wrap mt-2"):
                tags_input = ui.input(
                    label="Tags (comma-separated)",
                    placeholder="e.g. llm, notes, paper",
                ).classes("w-full")
                importance_input = ui.number(
                    label="Importance", value=0.0, min=0.0, max=1.0, step=0.1,
                    format="%.1f",
                ).classes("w-36")

        # ── Upload ─────────────────────────────────────────────────────────
        with ui.card().classes("w-full p-4"):
            ui.label("Upload document").classes(
                "text-sm font-semibold text-gray-600 mb-3"
            )

            @ui.refreshable
            def render_index_status():
                result = idx_ctrl.last_result
                if result:
                    color = (
                        "text-green-700 bg-green-50 border-green-200"
                        if idx_ctrl.last_ok
                        else "text-red-700 bg-red-50 border-red-200"
                    )
                    ui.label(result).classes(
                        f"text-xs font-mono {color} border rounded px-2 py-1 w-full"
                    )
                else:
                    ui.label("No file indexed yet.").classes(
                        "text-xs text-gray-400 italic"
                    )

            async def handle_upload(e):
                file_bytes     = await e.file.read()
                file_name      = e.file.name
                raw_doc_id     = doc_id_input.value or None
                resolved_doc_id = (raw_doc_id or "").strip() or file_name

                try:
                    save_path = idx_ctrl.save_file(file_name, file_bytes, raw_doc_id)
                except Exception as ex:
                    ui.notify(f"Save failed: {ex}", type="negative")
                    log.error("Failed to save uploaded file: %s", ex, exc_info=True)
                    return

                ui.notify(
                    f"Saved {file_name} ({len(file_bytes) // 1024} KB) \u2014 embedding\u2026",
                    type="info",
                    timeout=0,
                    close_button=True,
                )

                raw_tags = tags_input.value or ""
                tags     = [t.strip() for t in raw_tags.split(",") if t.strip()]

                strategy = strategy_select.value
                if strategy == "auto":
                    strategy = None

                ok = await asyncio.to_thread(
                    idx_ctrl.embed_file,
                    save_path,
                    resolved_doc_id,
                    int(chunk_size_input.value or 500),
                    int(overlap_input.value    or 100),
                    title_input.value.strip() or "",
                    tags or None,
                    workspace_input.value.strip() or "",
                    float(importance_input.value or 0.0),
                    strategy,
                )

                ui.notify(
                    idx_ctrl.last_result,
                    type="positive" if ok else "negative",
                )
                render_index_status.refresh()
                render_doc_list.refresh()
                on_doc_indexed()

            ui.upload(
                label="Drop a PDF, Markdown, or text file here (or click to select)",
                on_upload=handle_upload,
                max_files=1,
                auto_upload=True,
            ).props("accept=.pdf,.md,.markdown,.txt,.text flat").classes("w-full")

            ui.separator().classes("my-3")
            render_index_status()

    # ── Middle column: indexed document list ───────────────────────────────
    with ui.column().style(
        "flex: 1; height: 100%; overflow-y: auto; padding: 1rem; gap: 0;"
        " border-right: 1px solid #e5e7eb;"
    ):
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
            docs = idx_ctrl.list_docs_with_titles()
            if not docs:
                ui.label("No documents indexed yet.").classes(
                    "text-gray-400 italic text-sm"
                )
            else:
                for doc_id, display_title in docs:
                    is_sel = _selected_doc["id"] == doc_id
                    border = (
                        "border-l-4 border-blue-400"
                        if is_sel
                        else "border-l-4 border-transparent"
                    )
                    enrich_status = _enrich_state.get(doc_id, "")
                    with ui.card().classes(
                        f"w-full mb-1 {border}"
                    ):
                        with ui.row().classes("items-center justify-between w-full gap-2"):
                            ui.label(display_title).classes(
                                "text-sm font-mono text-gray-700 flex-1 min-w-0 truncate"
                                " cursor-pointer"
                            ).on(
                                "click",
                                lambda d=doc_id: (
                                    _selected_doc.update({"id": d}),
                                    render_doc_detail.refresh(),
                                    render_doc_list.refresh(),
                                ),
                            )

                            if enrich_status == "running":
                                ui.spinner(size="xs").classes("text-blue-400")
                            elif enrich_status.startswith("error"):
                                ui.label("✗").classes("text-xs text-red-400").tooltip(
                                    enrich_status
                                )
                            elif enrich_status == "done":
                                ui.label("✔").classes("text-xs text-green-500").tooltip(
                                    "Enrichment complete"
                                )

                            async def _do_enrich(d=doc_id):
                                # Capture the client BEFORE any refresh; refreshing the
                                # list destroys all child elements (including this slot),
                                # so the context must be restored after the await.
                                client = ui.context.client
                                _enrich_state[d] = "running"
                                render_doc_list.refresh()
                                stats = await asyncio.to_thread(
                                    idx_ctrl.enrich_doc, d, False
                                )
                                if stats.get("failed", 0) == -1:
                                    _enrich_state[d] = f"error: {stats.get('error', '?')}"
                                    msg, kind = f"Enrich failed for {d}", "negative"
                                else:
                                    _enrich_state[d] = "done"
                                    msg = (
                                        f"Enriched {d}: "
                                        f"+{stats['enriched']} "
                                        f"skip={stats['skipped']} "
                                        f"fail={stats['failed']}"
                                    )
                                    kind = "positive"
                                with client:
                                    render_doc_list.refresh()
                                    ui.notify(msg, type=kind)

                            ui.button(
                                "✦", on_click=_do_enrich
                            ).props("flat dense").classes(
                                "text-xs text-purple-500 px-1"
                            ).tooltip("Run knowledge extraction (enrich)")

        render_doc_list()

    # ── Right column: selected document detail ─────────────────────────────
    with ui.element("div").style(
        "width: 22rem; flex-shrink: 0; height: 100%; overflow-y: auto;"
        " padding: 0.75rem; border-left: 1px solid #e5e7eb;"
    ):
        @ui.refreshable
        def render_doc_detail():
            doc_id = _selected_doc["id"]
            if not doc_id:
                ui.label("Click a document to inspect.").classes(
                    "text-gray-400 italic text-sm"
                )
                return

            info = idx_ctrl.get_doc_info(doc_id)
            ui.label("Document info").classes(
                "text-sm font-semibold text-gray-700 mb-2"
            )
            with ui.grid(columns=2).classes("gap-x-3 gap-y-1 text-xs w-full"):
                for key, val in info.items():
                    if str(val) == "":
                        continue
                    ui.label(key).classes(
                        "font-mono text-gray-400 text-right truncate"
                    )
                    ui.label(str(val)).classes("font-mono text-gray-800 break-all")

        render_doc_detail()

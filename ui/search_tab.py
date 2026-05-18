"""Search tab UI — query bar, filter panel, result columns, chunk detail.

Exposes a single entry point:

    fi_source = build(ctrl)

`fi_source` is the Source-filter Select widget; the caller (dashboard)
passes it back to the Index tab so it can refresh its options after
a new document is embedded.
"""

from __future__ import annotations

from types import SimpleNamespace

from nicegui import ui

from ui.search_controller import RERANKER_UNAVAILABLE, SearchController


# ---------------------------------------------------------------------------
# Card rendering helpers (module-level so they are re-usable)
# ---------------------------------------------------------------------------

def _source_label(meta: dict) -> tuple[str, str]:
    """Return (position_label, source_label) for a result card, document-type aware.

    Notion chunks  → (section[:18], title[:22])
    File-based     → (pN,           filename)
    """
    if meta.get("document_type") == "notion":
        section = (meta.get("section") or "")[:18]
        title   = (meta.get("title")   or "?")[:22]
        return section, title
    pg  = meta.get("page", "?")
    pos = f"p{int(pg) + 1}" if isinstance(pg, int) else "p?"
    src = meta.get("filename") or meta.get("title") or "?"
    return pos, src


def _result_card(
    doc,
    rank: int,
    *,
    border: str,
    rank_cls: str,
    score_str: str,
    score_cls: str,
    extra_label: str | None = None,
    extra_cls: str = "",
    on_click,
) -> None:
    """Render a single result card into the current NiceGUI container."""
    chunk_id = doc.metadata.get("chunk_id", "?")
    pos, src = _source_label(doc.metadata)
    preview  = doc.page_content[:180].replace("\n", " ")
    with ui.card().classes(
        f"w-full cursor-pointer hover:shadow-md mb-1 border-l-4 {border}"
    ).on("click", on_click):
        with ui.row().classes("items-center gap-2 flex-wrap"):
            ui.label(f"#{rank + 1}").classes(f"text-xs font-bold w-5 {rank_cls}")
            if extra_label is not None:
                ui.label(extra_label).classes(f"text-xs font-mono {extra_cls} w-7")
            ui.label(score_str).classes(f"font-mono text-xs font-semibold {score_cls}")
            if pos:
                ui.label(pos).classes("text-gray-400 text-xs")
            ui.label(f"c{chunk_id}").classes("text-gray-400 text-xs")
            ui.label(src).classes("text-blue-400 text-xs truncate max-w-[9rem]")
        ui.label(
            preview + ("\u2026" if len(doc.page_content) > 180 else "")
        ).classes("text-xs text-gray-600 leading-snug mt-0.5")


# ---------------------------------------------------------------------------
# Tab entry point
# ---------------------------------------------------------------------------

def build(ctrl: SearchController, on_search: list | None = None) -> ui.Select:
    """Build the complete Search tab content into the current NiceGUI context.

    Args:
        on_search : Optional list of zero-argument callables that will be
                    invoked after every successful search.  Callers append
                    refresh functions here to keep other tabs in sync.

    Returns the Source-filter Select widget so the Index tab can refresh its
    options after a document is embedded.
    """
    _on_search = on_search if on_search is not None else []

    # ── Query bar ─────────────────────────────────────────────────────────
    with ui.card().classes("w-full rounded-none shadow-md p-3").style("flex-shrink: 0;"):
        ui.label("RAG Debug Dashboard").classes("text-lg font-bold text-gray-800 mb-2")

        # Row 1: search controls
        with ui.row().classes("w-full items-end gap-3 flex-wrap"):
            query_input   = ui.input(placeholder="Enter your query\u2026").classes(
                "flex-1 min-w-[12rem]"
            )
            fetch_k_input = ui.number(
                label="fetch-k", value=20, min=5, max=100, step=5
            ).classes("w-24")
            top_k_input   = ui.number(
                label="top-k", value=5, min=1, max=20, step=1
            ).classes("w-24")
            rerank_toggle = ui.checkbox("Rerank")
            hybrid_toggle = ui.checkbox("Hybrid")

            def do_search():
                rerank_on = rerank_toggle.value
                hybrid_on = hybrid_toggle.value
                k         = int(top_k_input.value   or 5)
                fetch_k   = int(fetch_k_input.value or 20)

                ui.notify("Searching\u2026", type="info", timeout=1500)
                error = ctrl.run_search(
                    query_input.value, k, fetch_k,
                    use_rerank=rerank_on, use_hybrid=hybrid_on,
                )

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

                render_results.refresh(rerank_on, hybrid_on)
                render_detail.refresh()
                for cb in _on_search:
                    cb()

            ui.button("Search", on_click=do_search).classes("bg-blue-600 text-white px-6")

        query_input.on("keydown.enter", do_search)

        # Row 2: filter toggle + active-filter summary
        with ui.row().classes("w-full items-center gap-3 flex-wrap mt-2"):
            filter_toggle = ui.checkbox("Filters")

            @ui.refreshable
            def render_filter_status():
                if ctrl.filter_active:
                    ui.label(f"Active: {ctrl.filter_summary}").classes(
                        "text-xs font-mono bg-blue-50 text-blue-700"
                        " border border-blue-200 rounded px-2 py-0.5"
                    )
                    ui.button(
                        "\u00d7 Clear",
                        on_click=lambda: (
                            ctrl.clear_filter(),
                            _refresh_filter_inputs(),
                            render_filter_status.refresh(),
                        ),
                    ).props("flat dense").classes("text-xs text-red-400 px-1 py-0")
                else:
                    ui.label("No filters active \u2014 searching all documents").classes(
                        "text-xs text-gray-400 italic"
                    )

            render_filter_status()
            filter_toggle.on_value_change(lambda e: filter_panel.set_visibility(e.value))

        # Filter panel (hidden by default)
        filter_panel = ui.element("div").classes(
            "w-full border border-gray-200 rounded bg-gray-50 p-3 mt-1"
        )
        filter_panel.set_visibility(False)

        with filter_panel:
            _fi_source = ui.select(
                options={"": ""} | ctrl.list_doc_title_map(),
                value="",
                label="Source (title)",
            ).classes("w-52").props("outlined dense clearable")
            _fi_workspace = ui.select(
                options=[""] + ctrl.list_workspaces(),
                value="",
                label="Workspace",
            ).classes("w-44").props("outlined dense clearable")
            _fi_doctype = ui.select(
                options=[""] + ctrl.list_document_types(),
                value="",
                label="Doc type",
            ).classes("w-36").props("outlined dense clearable")
            _fi_tag    = ui.select(
                options=[""]
                + ctrl.list_tags(),
                value="",
                label="Tags",
            ).classes("w-40").props("outlined dense clearable")
            _fi_after  = ui.input(
                label="Created after",  placeholder="YYYY-MM-DD"
            ).classes("w-36")
            _fi_before = ui.input(
                label="Created before", placeholder="YYYY-MM-DD"
            ).classes("w-36")

            def _apply_filter_field(field: str, value):
                ctrl.set_filter_field(field, value)
                render_filter_status.refresh()

            def _refresh_filter_inputs():
                """Reset all filter inputs to reflect controller state."""
                f = ctrl.filter
                _fi_source.set_value(f.source_id or "")
                _fi_workspace.set_value(f.workspace or "")
                _fi_doctype.set_value(f.document_type or "")
                _fi_tag.set_value(f.tag or "")
                _fi_after.set_value(f.created_after or "")
                _fi_before.set_value(f.created_before or "")

            _fi_source.on_value_change(
                lambda e: _apply_filter_field("source_id", e.value)
            )
            _fi_workspace.on_value_change(
                lambda e: _apply_filter_field("workspace", e.value)
            )
            _fi_doctype.on_value_change(
                lambda e: _apply_filter_field("document_type", e.value)
            )
            _fi_tag.on_value_change(
                lambda e: _apply_filter_field("tag", e.value)
            )
            _fi_after.on("keydown.enter",
                lambda: _apply_filter_field("created_after", _fi_after.value)
            )
            _fi_after.on("blur",
                lambda: _apply_filter_field("created_after", _fi_after.value)
            )
            _fi_before.on("keydown.enter",
                lambda: _apply_filter_field("created_before", _fi_before.value)
            )
            _fi_before.on("blur",
                lambda: _apply_filter_field("created_before", _fi_before.value)
            )

    # ── Body: results columns + detail panel ──────────────────────────────
    with ui.row().style(
        "flex: 1; min-height: 0; gap: 0.75rem; padding: 0.75rem;"
        " overflow: hidden; align-items: stretch;"
    ):
        @ui.refreshable
        def render_results(rerank_on: bool = False, hybrid_on: bool = False):
            vector       = ctrl.vector_results
            reranked     = ctrl.reranked_results
            reranked_ids = ctrl.reranked_chunk_ids
            bm25         = ctrl.bm25_results
            hybrid       = ctrl.hybrid_results
            hybrid_ids   = ctrl.hybrid_chunk_ids

            with ui.element("div").style(
                "display: flex; flex-direction: row; height: 100%;"
                " gap: 0.75rem; overflow: hidden; flex: 1;"
            ):
                # ── Vector column ──────────────────────────────────────────
                with ui.column().style("flex: 1; min-height: 0; overflow-y: auto;"):
                    header = f"VECTOR  ({len(vector)})"
                    if hybrid_ids:
                        header += f"  \u00b7  {len(hybrid_ids)} in fused"
                    elif reranked_ids:
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
                            chunk_id    = doc.metadata.get("chunk_id", "?")
                            in_fused    = chunk_id in hybrid_ids
                            in_reranked = chunk_id in reranked_ids
                            border = (
                                "border-amber-400" if in_fused
                                else "border-blue-300" if in_reranked
                                else "border-transparent"
                            )
                            _result_card(
                                doc, v_rank,
                                border=border,
                                rank_cls="text-gray-400",
                                score_str=f"{vscore}",
                                score_cls=SearchController.score_color(vscore),
                                on_click=lambda d=doc, s=vscore: (
                                    ctrl.select_chunk(d, s, "vscore"),
                                    render_detail.refresh(),
                                ),
                            )

                # ── BM25 column (only when hybrid is ON) ───────────────────
                if hybrid_on:
                    with ui.column().style("flex: 1; min-height: 0; overflow-y: auto;"):
                        bm25_count = len(bm25) if bm25 else 0
                        ui.label(f"BM25  ({bm25_count})").classes(
                            "text-xs font-semibold uppercase tracking-wide text-orange-500 mb-2"
                        )
                        if not bm25:
                            ui.label("No BM25 results.").classes(
                                "text-gray-400 italic text-sm"
                            )
                        else:
                            for b_rank, (doc, bscore) in enumerate(bm25):
                                _result_card(
                                    doc, b_rank,
                                    border="border-orange-300",
                                    rank_cls="text-orange-400",
                                    score_str=f"{bscore:.3f}",
                                    score_cls="text-orange-600",
                                    on_click=lambda d=doc, s=bscore: (
                                        ctrl.select_chunk(d, s, "bm25score"),
                                        render_detail.refresh(),
                                    ),
                                )

                    # ── Fused column (hybrid ON, rerank OFF) ────────────────
                    if not rerank_on:
                        with ui.column().style("flex: 1; min-height: 0; overflow-y: auto;"):
                            fused_count = len(hybrid) if hybrid else 0
                            ui.label(f"FUSED RRF  ({fused_count})").classes(
                                "text-xs font-semibold uppercase tracking-wide text-amber-600 mb-2"
                            )
                            if not hybrid:
                                ui.label("No fused results.").classes(
                                    "text-gray-400 italic text-sm"
                                )
                            else:
                                for f_rank, (doc, fscore) in enumerate(hybrid):
                                    _result_card(
                                        doc, f_rank,
                                        border="border-amber-400",
                                        rank_cls="text-amber-600",
                                        score_str=f"{fscore:.4f}",
                                        score_cls="text-amber-700",
                                        on_click=lambda d=doc, s=fscore: (
                                            ctrl.select_chunk(d, s, "rrf_score"),
                                            render_detail.refresh(),
                                        ),
                                    )

                # ── Reranked column (rerank ON) ─────────────────────────────
                if rerank_on:
                    with ui.column().style("flex: 1; min-height: 0; overflow-y: auto;"):
                        count        = len(reranked) if reranked else 0
                        label_prefix = "RERANKED (hybrid)" if hybrid_on else "RERANKED"
                        ui.label(f"{label_prefix}  ({count})").classes(
                            "text-xs font-semibold uppercase tracking-wide text-gray-500 mb-2"
                        )
                        if not reranked:
                            ui.label("No rerank results.").classes(
                                "text-gray-400 italic text-sm"
                            )
                        else:
                            for r_rank, (doc, rscore) in enumerate(reranked):
                                chunk_id = doc.metadata.get("chunk_id", "?")
                                change_label, change_color = SearchController.rank_change(
                                    chunk_id, vector, r_rank
                                )
                                _result_card(
                                    doc, r_rank,
                                    border="border-green-300",
                                    rank_cls="text-gray-700",
                                    score_str=f"{rscore:.2f}",
                                    score_cls="text-purple-600",
                                    extra_label=change_label,
                                    extra_cls=change_color,
                                    on_click=lambda d=doc, s=rscore: (
                                        ctrl.select_chunk(d, s, "rscore"),
                                        render_detail.refresh(),
                                    ),
                                )

        render_results(False, False)

        # Right — chunk detail panel
        with ui.card().style(
            "width: 22rem; flex-shrink: 0; min-height: 0; height: 100%;"
            " overflow-y: auto; padding: 0.75rem;"
        ):
            @ui.refreshable
            def render_detail():
                meta = ctrl.selected_metadata
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

    return SimpleNamespace(fi_source=_fi_source, query_input=query_input)

"""Search tab UI — query bar, filter panel, result columns, chunk detail.

Exposes a single entry point:

    fi_source = build(ctrl)

`fi_source` is the Source-filter Select widget; the caller (dashboard)
passes it back to the Index tab so it can refresh its options after
a new document is embedded.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
from typing import Literal

from nicegui import ui
from rag.retrieval.base import RetrievalResult

from ui.search_controller import RERANKER_UNAVAILABLE, SearchController
from ui import CodeGraphAdapter, GraphLimits


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


def _code_block_card(row: dict, *, on_click) -> None:
    """Render one code-block card for the Code Search browser list."""
    meta = dict(row.get("metadata") or {})
    content = str(row.get("content") or "")
    chunk_id = str(meta.get("chunk_id", "?"))
    chunk_type = str(meta.get("chunk_type", "?"))
    name = str(meta.get("name") or "?")
    repo_id = str(meta.get("repo_id") or "?")
    file_path = str(meta.get("file_path") or "?")
    line_span = f"L{meta.get('start_line', '?')}-{meta.get('end_line', '?')}"
    preview = content[:220].replace("\n", " ")

    with ui.card().classes(
        "w-full cursor-pointer hover:shadow-md mb-1 border-l-4 border-cyan-300"
    ).on("click", on_click):
        with ui.row().classes("items-center gap-2 flex-wrap"):
            ui.label(chunk_type).classes(
                "text-[10px] font-mono bg-cyan-50 text-cyan-700 border border-cyan-200 rounded px-1 py-0"
            )
            ui.label(name).classes("text-xs font-semibold text-gray-700")
            ui.label(f"c{chunk_id}").classes("text-xs text-gray-400 font-mono")
            ui.label(line_span).classes("text-xs text-gray-400 font-mono")
        ui.label(file_path).classes("text-xs text-blue-500 font-mono truncate")
        ui.label(f"repo: {repo_id}").classes("text-xs text-gray-400")
        ui.label(preview + ("\u2026" if len(content) > 220 else "")).classes(
            "text-xs text-gray-600 leading-snug mt-0.5"
        )


def _code_block_card_expanded(row: dict, *, expanded: bool, on_click) -> None:
    """Render one code-block card with optional inline expanded details."""
    _code_block_card(row, on_click=on_click)
    if not expanded:
        return

    meta = dict(row.get("metadata") or {})
    content = str(row.get("content") or "")
    with ui.card().classes("w-full mb-3 border border-cyan-100 bg-cyan-50/30"):
        with ui.grid(columns=2).classes("gap-x-3 gap-y-0.5 text-xs w-full"):
            for key in [
                "repo_id", "file_path", "chunk_type", "name",
                "start_line", "end_line", "language", "branch", "chunk_id",
            ]:
                if key not in meta:
                    continue
                ui.label(key).classes("font-mono text-gray-400 text-right truncate")
                ui.label(str(meta.get(key, ""))).classes("font-mono text-gray-800 break-all")

        ui.separator().classes("my-2")
        ui.label("Content").classes("text-xs font-semibold text-gray-500 mb-1")
        ui.label(content).classes(
            "text-xs text-gray-700 whitespace-pre-wrap bg-white rounded p-2 w-full"
        )


# ---------------------------------------------------------------------------
# Tab entry point
# ---------------------------------------------------------------------------

def build(
    ctrl: SearchController,
    on_search: list | None = None,
    *,
    result_scope: Literal["all", "document", "code"] = "document",
    title: str = "RAG Debug Dashboard",
    enable_graph: bool = False,
) -> SimpleNamespace:
    """Build the complete Search tab content into the current NiceGUI context.

    Args:
        on_search : Optional list of zero-argument callables that will be
                    invoked after every successful search.  Callers append
                    refresh functions here to keep other tabs in sync.

    Returns the Source-filter Select widget so the Index tab can refresh its
    options after a document is embedded.
    """
    _on_search = on_search if on_search is not None else []
    _fi_source = None

    def _source_options() -> dict[str, str]:
        options = {str(k): str(v) for k, v in ctrl.list_doc_title_map().items()}
        return {"": "(All sources)"} | options

    # ── Query bar ─────────────────────────────────────────────────────────
    with ui.card().classes("w-full rounded-none shadow-md p-3").style("flex-shrink: 0;"):
        ui.label(title).classes("text-lg font-bold text-gray-800 mb-2")

        # Row 1: search controls
        with ui.row().classes("w-full items-end gap-3 flex-wrap"):
            query_placeholder = "Enter your query\u2026"
            if result_scope == "code":
                query_placeholder = (
                    "Enter code query, e.g. repo:payments path:src/payments "
                    "module:payments.service find caller"
                )

            query_input   = ui.input(placeholder=query_placeholder).classes(
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
                    result_scope=result_scope,
                    apply_filter=(result_scope == "document"),
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

                graph_state["rerank_on"] = bool(rerank_on)
                graph_state["hybrid_on"] = bool(hybrid_on)
                render_results.refresh(rerank_on, hybrid_on)
                if result_scope == "code" and enable_graph:
                    render_graph.refresh()
                render_detail.refresh()
                for cb in _on_search:
                    cb()

            ui.button("Search", on_click=do_search).classes("bg-blue-600 text-white px-6")

        query_input.on("keydown.enter", do_search)

        if result_scope == "code":
            ui.label(
                "Soft scope hints: repo:<repo-id> path:<path-fragment> module:<module-prefix>. These boost ranking only."
            ).classes("text-xs text-gray-500 mt-2")

        if result_scope == "document":
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
                    options=_source_options(),
                    value="",
                    label="Source (title)",
                ).classes("w-64").props(
                    "outlined dense clearable use-input input-debounce=0 options-dense"
                )
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
                    options=[""] + ctrl.list_tags(),
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
        else:
            ui.label(
                "Code Search uses independent controls below (repo, file path, and text)."
            ).classes("text-xs text-gray-500 mt-2")

    # ── Body: results + graph + detail panel ──────────────────────────────
    graph_adapter = CodeGraphAdapter(limits=GraphLimits(max_nodes=120, max_edges=240))
    graph_lookup: dict[str, tuple[object, float, str]] = {}
    graph_state = {"rerank_on": False, "hybrid_on": False, "serial": 0}

    def _active_primary_results() -> list[tuple[object, float, str]]:
        if graph_state["rerank_on"] and ctrl.reranked_results:
            return [(doc, score, "rscore") for doc, score in ctrl.reranked_results]
        if graph_state["hybrid_on"] and ctrl.hybrid_results:
            return [(doc, score, "rrf_score") for doc, score in ctrl.hybrid_results]
        return [(doc, score, "vscore") for doc, score in ctrl.vector_results]

    def _all_result_lookup() -> dict[str, tuple[object, float, str]]:
        lookup: dict[str, tuple[object, float, str]] = {}
        pools = [
            (ctrl.vector_results, "vscore"),
            (ctrl.bm25_results or [], "bm25score"),
            (ctrl.hybrid_results or [], "rrf_score"),
            (ctrl.reranked_results or [], "rscore"),
        ]
        for docs, score_key in pools:
            for doc, score in docs:
                cid = str(doc.metadata.get("chunk_id", "")).strip()
                if cid and cid not in lookup:
                    lookup[cid] = (doc, score, score_key)
        return lookup

    def _to_retrieval_results(rows: list[tuple[object, float, str]]) -> list[RetrievalResult]:
        out: list[RetrievalResult] = []
        for doc, score, _ in rows:
            metadata = dict(doc.metadata or {})
            source = "code" if (metadata.get("repo_id") or metadata.get("chunk_type")) else "document"
            out.append(
                RetrievalResult(
                    content=doc.page_content,
                    score=float(score),
                    source=source,
                    metadata=metadata,
                )
            )
        return out

    with ui.element("div").style(
        "flex: 1; min-height: 0; display: flex; flex-direction: row;"
        " gap: 0.75rem; padding: 0.75rem; overflow: hidden; align-items: stretch;"
    ):
        with ui.column().style(
            "flex: 1 1 48rem; min-width: 0; min-height: 0; display: flex; flex-direction: column;"
            " gap: 0.75rem; overflow: hidden;"
        ):
            with ui.card().style(
                "flex: 1 1 58%; min-height: 0; overflow: hidden; padding: 0.75rem;"
                " display: flex; flex-direction: column;"
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
                        "display: flex; flex-direction: row; flex: 1; min-height: 0; height: 100%;"
                        " gap: 0.75rem; overflow-x: auto; overflow-y: hidden;"
                    ):
                        with ui.element("div").style(
                            "flex: 1 0 20rem; min-width: 20rem; min-height: 0; max-height: 100%;"
                            " display: flex; flex-direction: column; overflow-y: auto;"
                        ):
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

                        if hybrid_on:
                            with ui.element("div").style(
                                "flex: 1 0 20rem; min-width: 20rem; min-height: 0; max-height: 100%;"
                                " display: flex; flex-direction: column; overflow-y: auto;"
                            ):
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

                            if not rerank_on:
                                with ui.element("div").style(
                                    "flex: 1 0 20rem; min-width: 20rem; min-height: 0; max-height: 100%;"
                                    " display: flex; flex-direction: column; overflow-y: auto;"
                                ):
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

                        if rerank_on:
                            with ui.element("div").style(
                                "flex: 1 0 20rem; min-width: 20rem; min-height: 0; max-height: 100%;"
                                " display: flex; flex-direction: column; overflow-y: auto;"
                            ):
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

            if result_scope == "code" and enable_graph:
                with ui.card().style(
                    "flex: 1 1 42%; min-height: 0; overflow: hidden; padding: 0.75rem;"
                    " display: flex; flex-direction: column;"
                ):
                    ui.label("Code Graph (Cytoscape)").classes(
                        "text-xs font-semibold uppercase tracking-wide text-gray-500 mb-1"
                    )

                    @ui.refreshable
                    def render_graph():
                        nonlocal graph_lookup
                        rows = _active_primary_results()
                        if not rows:
                            ui.label("Graph appears after search results are available.").classes(
                                "text-gray-400 italic text-sm"
                            )
                            return

                        retrieval_rows = _to_retrieval_results(rows)
                        payload = graph_adapter.build(retrieval_rows, query=ctrl.last_query)

                        all_lookup = _all_result_lookup()
                        graph_lookup = {
                            node.id: all_lookup[node.id]
                            for node in payload.nodes
                            if node.id in all_lookup
                        }

                        graph_state["serial"] += 1
                        serial = graph_state["serial"]
                        container_id = f"code-graph-canvas-{serial}"
                        payload_json = json.dumps(payload.to_cytoscape(), ensure_ascii=True)
                        ui.label(
                            f"Nodes: {payload.meta.get('node_count', len(payload.nodes))}"
                            f"  Edges: {payload.meta.get('edge_count', len(payload.edges))}"
                        ).classes("text-xs text-gray-500 mb-2")

                        ui.html(
                            f'<div id="{container_id}" '
                            'style="width:100%;height:100%;min-height:280px;border:1px solid #e5e7eb;border-radius:8px;"></div>'
                        ).classes("w-full h-full")

                        init_graph_js = f"""
                        (function() {{
                            const payload = {payload_json};
                            const serial = {serial};
                            const root = document.getElementById({json.dumps(container_id)});
                            if (!root) return;
                            if (!window.cytoscape) {{
                                root.innerHTML = '<div style="padding:12px;color:#6b7280;font-size:12px;">Cytoscape.js is loading...</div>';
                                return;
                            }}
                            window.__codeGraphQueue = window.__codeGraphQueue || [];
                            window.__codeGraphState = window.__codeGraphState || {{ cy: null }};
                            if (window.__codeGraphState.cy) {{
                                window.__codeGraphState.cy.destroy();
                            }}

                            const edgeColor = (t) => ({{
                                CALLS: '#2563eb',
                                EXTENDS: '#16a34a',
                                IMPLEMENTS: '#d97706',
                                IMPORTS: '#0d9488',
                                NEARBY: '#64748b',
                            }}[t] || '#64748b');

                            const cy = window.cytoscape({{
                                container: root,
                                elements: [...payload.nodes, ...payload.edges],
                                style: [
                                    {{ selector: 'node', style: {{
                                        'label': 'data(label)',
                                        'font-size': '10px',
                                        'color': '#111827',
                                        'text-wrap': 'wrap',
                                        'text-max-width': 120,
                                        'background-color': '#dbeafe',
                                        'border-width': 1,
                                        'border-color': '#60a5fa',
                                        'width': 28,
                                        'height': 28,
                                    }} }},
                                    {{ selector: 'node[is_primary = true]', style: {{
                                        'background-color': '#bfdbfe',
                                        'border-color': '#2563eb',
                                        'border-width': 2,
                                        'width': 34,
                                        'height': 34,
                                    }} }},
                                    {{ selector: 'edge', style: {{
                                        'curve-style': 'bezier',
                                        'target-arrow-shape': 'triangle',
                                        'line-color': (e) => edgeColor(e.data('edge_type')),
                                        'target-arrow-color': (e) => edgeColor(e.data('edge_type')),
                                        'width': (e) => Math.max(1.5, Math.min(6, (Number(e.data('score')) || 0.1) * 6)),
                                        'label': 'data(edge_type)',
                                        'font-size': '8px',
                                        'color': '#6b7280',
                                        'text-background-color': '#ffffff',
                                        'text-background-opacity': 0.75,
                                        'text-background-padding': 1,
                                    }} }},
                                ],
                                layout: {{ name: 'breadthfirst', directed: true, padding: 20, spacingFactor: 1.05 }},
                            }});
                            window.__codeGraphState.cy = cy;
                            cy.on('tap', 'node', (evt) => {{
                                window.__codeGraphQueue.push({{
                                    serial,
                                    node_id: evt.target.id(),
                                    data: evt.target.data(),
                                }});
                            }});
                        }})();
                        """
                        ui.timer(0.05, lambda js=init_graph_js: ui.run_javascript(js), once=True)

                    render_graph()
            else:
                @ui.refreshable
                def render_graph():
                    return

        with ui.card().style(
            "flex: 1 1 22rem; min-width: 20rem; max-width: 30rem;"
            " min-height: 0; height: 100%; overflow-y: auto; padding: 0.75rem;"
        ):
            @ui.refreshable
            def render_detail():
                meta = ctrl.selected_metadata
                if not meta:
                    empty_hint = (
                        "Click a result to inspect."
                        if result_scope == "document" or (result_scope == "code" and not enable_graph)
                        else "Click a result or graph node to inspect."
                    )
                    ui.label(empty_hint).classes("text-gray-400 italic text-sm")
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

    if result_scope == "code" and enable_graph:
        async def _poll_graph_click_queue() -> None:
            try:
                event = await ui.run_javascript(
                    "(function(){const q=window.__codeGraphQueue||[]; return q.length ? q.shift() : null;})()"
                )
            except Exception:
                return
            if not event:
                return
            if int(event.get("serial", -1)) != int(graph_state["serial"]):
                return

            node_id = str(event.get("node_id", "")).strip()
            if not node_id:
                return

            if ctrl.select_chunk_by_id(node_id):
                render_detail.refresh()
                return

            node_data = dict(event.get("data", {}) or {})
            ctrl.select_graph_node(node_data)
            render_detail.refresh()

        ui.timer(0.25, _poll_graph_click_queue)

    return SimpleNamespace(
        fi_source=_fi_source,
        query_input=query_input,
        source_options=_source_options,
    )


def build_code_list(ctrl: SearchController, *, title: str = "Code List") -> None:
    """Build a dedicated paginated Code Block list view."""
    state = {
        "repo_id": "",
        "file_path": "",
        "text": "",
        "page": 1,
        "page_size": 25,
        "selected_chunk_id": "",
    }

    with ui.card().classes("w-full rounded-none shadow-md p-3").style("flex-shrink: 0;"):
        ui.label(title).classes("text-lg font-bold text-gray-800 mb-2")
        with ui.row().classes("w-full items-end gap-3 flex-wrap"):
            cb_repo = ui.input(label="Repo ID", placeholder="e.g. redesigned-eureka").classes("w-44")
            cb_file = ui.input(label="File path contains", placeholder="e.g. src/").classes("w-56")
            cb_text = ui.input(label="Text search", placeholder="e.g. function name").classes("flex-1 min-w-[12rem]")
            cb_page_size = ui.select(
                options=[10, 25, 50, 100],
                value=state["page_size"],
                label="Page size",
            ).classes("w-28").props("dense outlined")

            def _apply_filters() -> None:
                state["repo_id"] = (cb_repo.value or "").strip()
                state["file_path"] = (cb_file.value or "").strip()
                state["text"] = (cb_text.value or "").strip()
                state["page_size"] = int(cb_page_size.value or 25)
                state["page"] = 1
                state["selected_chunk_id"] = ""
                render_code_blocks.refresh()

            ui.button("Apply", on_click=_apply_filters).classes("bg-cyan-600 text-white")
            ui.button(
                "Clear",
                on_click=lambda: (
                    cb_repo.set_value(""),
                    cb_file.set_value(""),
                    cb_text.set_value(""),
                    cb_page_size.set_value(25),
                    _apply_filters(),
                ),
            ).props("outline")

            cb_repo.on("keydown.enter", _apply_filters)
            cb_file.on("keydown.enter", _apply_filters)
            cb_text.on("keydown.enter", _apply_filters)
            cb_page_size.on_value_change(lambda _: _apply_filters())

    def _toggle_code_block(state: dict, rr: dict, cid: str) -> None:
        """Toggle the expanded code block card and keep the detail panel in sync."""
        if state["selected_chunk_id"] == cid:
            state["selected_chunk_id"] = ""
        else:
            state["selected_chunk_id"] = cid
        ctrl.select_code_block(rr)

    with ui.element("div").style(
        "flex: 1; min-height: 0; display: flex; flex-direction: row;"
        " gap: 0.75rem; padding: 0.75rem; overflow: hidden; align-items: stretch;"
    ):
        with ui.element("div").style("flex: 1; min-height: 0; overflow-y: auto;"):
            @ui.refreshable
            def render_code_blocks():
                rows = ctrl.list_code_blocks(
                    repo_id=state["repo_id"],
                    text=state["text"],
                    limit=500,
                )
                if state["file_path"]:
                    needle = state["file_path"].lower()
                    rows = [
                        r for r in rows
                        if needle in str((r.get("metadata") or {}).get("file_path", "")).lower()
                    ]

                total = len(rows)
                page_size = max(1, int(state["page_size"]))
                page_count = max(1, int(math.ceil(total / page_size)) if total else 1)
                state["page"] = max(1, min(int(state["page"]), page_count))
                start = (state["page"] - 1) * page_size
                page_rows = rows[start:start + page_size]

                with ui.row().classes("w-full items-center justify-between mb-2"):
                    ui.label(
                        f"Blocks: {total}  ·  Page {state['page']}/{page_count}"
                    ).classes("text-xs text-gray-500")
                    with ui.row().classes("items-center gap-2"):
                        ui.button("Prev", on_click=lambda: (
                            state.__setitem__("page", max(1, state["page"] - 1)),
                            render_code_blocks.refresh(),
                        )).props("dense outline")
                        ui.button("Next", on_click=lambda: (
                            state.__setitem__("page", min(page_count, state["page"] + 1)),
                            render_code_blocks.refresh(),
                        )).props("dense outline")

                if not rows:
                    ui.label("No code blocks found.").classes("text-gray-400 italic text-sm")
                    return

                for row in page_rows:
                    chunk_id = str((row.get("metadata") or {}).get("chunk_id", "")).strip()
                    _code_block_card_expanded(
                        row,
                        expanded=(chunk_id == state["selected_chunk_id"]),
                        on_click=lambda rr=row, cid=chunk_id: (
                            _toggle_code_block(state, rr, cid),
                            render_detail.refresh(),
                            render_code_blocks.refresh(),
                        ),
                    )

            render_code_blocks()

        with ui.card().style(
            "width: 22rem; flex-shrink: 0; min-height: 0; height: 100%;"
            " overflow-y: auto; padding: 0.75rem;"
        ):
            @ui.refreshable
            def render_detail():
                meta = ctrl.selected_metadata
                if not meta:
                    ui.label("Click a code block card to inspect.").classes(
                        "text-gray-400 italic text-sm"
                    )
                    return

                ui.label("Code block detail").classes(
                    "text-sm font-semibold text-gray-700 mb-2"
                )
                ordered_keys = [
                    "source_type",
                    "repo_id",
                    "file_path",
                    "chunk_type",
                    "name",
                    "language",
                    "branch",
                    "start_line",
                    "end_line",
                    "chunk_id",
                ]
                seen_keys = set()
                with ui.grid(columns=2).classes("gap-x-3 gap-y-0.5 text-xs w-full"):
                    for key in ordered_keys:
                        if key not in meta:
                            continue
                        seen_keys.add(key)
                        ui.label(key).classes("font-mono text-gray-400 text-right truncate")
                        ui.label(str(meta.get(key, ""))).classes("font-mono text-gray-800 break-all")
                    for key in sorted(k for k in meta.keys() if k not in seen_keys and k != "_content"):
                        ui.label(key).classes("font-mono text-gray-400 text-right truncate")
                        ui.label(str(meta.get(key, ""))).classes("font-mono text-gray-800 break-all")

                ui.separator().classes("my-2")
                ui.label("Content").classes("text-xs font-semibold text-gray-400 mb-1")
                ui.label(meta.get("_content", "")).classes(
                    "text-xs text-gray-700 whitespace-pre-wrap bg-gray-50 rounded p-2 w-full"
                )

            render_detail()

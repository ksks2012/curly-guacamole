"""Code Graph tab UI.

Dedicated view for code graph rendering and node-focused inspection.
"""

from __future__ import annotations

import json
import time
from typing import Any

from nicegui import ui
from rag.retrieval.base import RetrievalResult

from ui import CodeGraphAdapter, GraphLimits, normalize_event
from ui.code_graph_metrics import interaction_latency_p95_ms
from ui.code_graph_presets import PRESETS
from ui.code_graph_staged_view import DEFAULT_STAGE_KEY, apply_stage, get_stage, next_stage_key
from ui.search_controller import RERANKER_UNAVAILABLE, SearchController


def build(ctrl: SearchController, *, title: str = "Code Graph") -> None:
    """Build the standalone code graph tab."""

    graph_adapter = CodeGraphAdapter(limits=GraphLimits(max_nodes=80, max_edges=160))
    graph_lookup: dict[str, tuple[object, float, str]] = {}
    edge_lookup: dict[str, dict[str, Any]] = {}
    graph_state = {
        "rerank_on": False,
        "hybrid_on": False,
        "serial": 0,
        "layout": "breadthfirst",
        "stage_mode": DEFAULT_STAGE_KEY,
        "edge_type_filter": None,
        "max_nodes_val": 80,
        "max_edges_val": 160,
        "dense_max_edges_val": 160,
        "latency_samples_ms": {"sparse": [], "dense": []},
        "latency_p95_ms": {"sparse": 0.0, "dense": 0.0},
        "last_payload": None,
    }

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
            out.append(
                RetrievalResult(
                    content=doc.page_content,
                    score=float(score),
                    source="code",
                    metadata=metadata,
                )
            )
        return out

    with ui.card().classes("w-full rounded-none shadow-md p-3").style("flex-shrink: 0;"):
        ui.label(title).classes("text-lg font-bold text-gray-800 mb-2")

        with ui.row().classes("w-full items-end gap-3 flex-wrap"):
            query_input = ui.input(placeholder="Enter your code query...").classes(
                "flex-1 min-w-[12rem]"
            )
            fetch_k_input = ui.number(
                label="fetch-k", value=20, min=5, max=100, step=5
            ).classes("w-24")
            top_k_input = ui.number(
                label="top-k", value=5, min=1, max=20, step=1
            ).classes("w-24")
            rerank_toggle = ui.checkbox("Rerank")
            hybrid_toggle = ui.checkbox("Hybrid")
            relation_toggle = ui.checkbox("Relations", value=True)

            ui.label("View Flow:").classes("text-xs text-gray-600")
            stage_select = ui.select(
                label=None,
                options={
                    "sparse": "Stage 1: Sparse",
                    "dense": "Stage 2: Enriched",
                },
            ).classes("w-40").props("outlined dense")
            stage_select.set_value(DEFAULT_STAGE_KEY)

            stage_status_label = ui.label("").classes("text-xs text-amber-700 min-w-[16rem]")
            stage_switch_button = ui.button("Switch to Stage 2").props("outline")

            ui.label("Preset:").classes("text-xs text-gray-600")
            preset_select = ui.select(
                label=None,
                options={"small": "Small", "medium": "Medium", "large": "Large", "custom": "Custom"}
            ).classes("w-32").props("outlined dense")
            preset_select.set_value("medium")

            ui.label("Layout:").classes("text-xs text-gray-600")
            layout_select = ui.select(
                label=None,
                options={"breadthfirst": "Breadth-first", "cose": "COSE"}
            ).classes("w-32").props("outlined dense")
            layout_select.set_value("breadthfirst")

            ui.label("Max Nodes:").classes("text-xs text-gray-600")
            max_nodes_input = ui.number(
                label=None, value=120, min=10, max=500, step=10
            ).classes("w-24").props("outlined dense")

            ui.label("Max Edges:").classes("text-xs text-gray-600")
            max_edges_input = ui.number(
                label=None, value=160, min=10, max=1000, step=10
            ).classes("w-24").props("outlined dense")

            def _sync_stage_controls(*, update_dense_base: bool) -> None:
                stage_key = str(stage_select.value or DEFAULT_STAGE_KEY)
                graph_state["stage_mode"] = stage_key

                if update_dense_base and stage_key == "dense":
                    graph_state["dense_max_edges_val"] = int(max_edges_input.value or 160)

                resolved = apply_stage(
                    stage_key=stage_key,
                    max_nodes=int(max_nodes_input.value or 80),
                    dense_max_edges=int(graph_state["dense_max_edges_val"]),
                    preferred_layout=str(layout_select.value or "breadthfirst"),
                )

                graph_state["max_nodes_val"] = resolved.max_nodes
                graph_state["max_edges_val"] = resolved.max_edges
                graph_state["layout"] = resolved.layout

                max_nodes_input.set_value(resolved.max_nodes)
                max_edges_input.set_value(resolved.max_edges)
                layout_select.set_value(resolved.layout)
                relation_toggle.set_value(resolved.relations_enabled)

                stage = get_stage(stage_key)
                p95 = float(graph_state["latency_p95_ms"].get(stage_key, 0.0) or 0.0)
                stage_status_label.text = (
                    f"{stage.label} | target overlap <= {stage.target_overlap_rate:.0%} | "
                    f"readable >= {stage.target_readable_nodes} | p95 <= {stage.target_latency_p95_ms}ms | "
                    f"current p95: {p95:.1f}ms"
                )

                if stage_key == "sparse":
                    stage_switch_button.text = "Switch to Stage 2"
                    stage_switch_button.props("outline color=amber")
                else:
                    stage_switch_button.text = "Stage 2 Active"
                    stage_switch_button.props("outline color=green")

            def _go_next_stage() -> None:
                stage_select.set_value(next_stage_key(str(stage_select.value or DEFAULT_STAGE_KEY)))
                _sync_stage_controls(update_dense_base=False)
                ui.notify("Switched to enriched relation view.", type="positive", timeout=1200)

            def _apply_preset() -> None:
                """Apply selected preset to UI controls."""
                preset_key = str(preset_select.value or "custom").lower()
                if preset_key in PRESETS:
                    preset = PRESETS[preset_key]
                    max_nodes_input.set_value(preset.max_nodes)
                    max_edges_input.set_value(preset.max_edges)
                    graph_state["dense_max_edges_val"] = preset.max_edges
                    layout_select.set_value(preset.layout)
                    relation_toggle.set_value(preset.relations_enabled)
                    stage_select.set_value("dense" if preset.relations_enabled else "sparse")
                    _sync_stage_controls(update_dense_base=False)

            def do_search() -> None:
                rerank_on = bool(rerank_toggle.value)
                hybrid_on = bool(hybrid_toggle.value)
                k = int(top_k_input.value or 5)
                fetch_k = int(fetch_k_input.value or 20)

                if str(stage_select.value or DEFAULT_STAGE_KEY) == "dense":
                    graph_state["dense_max_edges_val"] = int(max_edges_input.value or 160)
                _sync_stage_controls(update_dense_base=False)

                max_nodes = int(graph_state["max_nodes_val"] or 80)
                max_edges = int(graph_state["max_edges_val"] or 160)
                layout_val = str(graph_state["layout"] or "breadthfirst")
                include_relations = bool(relation_toggle.value)

                ui.notify("Searching...", type="info", timeout=1500)
                error = ctrl.run_search(
                    query_input.value,
                    k,
                    fetch_k,
                    use_rerank=rerank_on,
                    use_hybrid=hybrid_on,
                    result_scope="code",
                    apply_filter=False,
                    include_relations=include_relations,
                )

                if error == "Query is empty.":
                    ui.notify("Please enter a query.", type="warning")
                    return
                if error == RERANKER_UNAVAILABLE:
                    ui.notify(
                        "Reranker not available - check config.reranker_type.",
                        type="warning",
                    )
                elif error:
                    ui.notify(f"Search error: {error}", type="negative")
                    return

                graph_state["rerank_on"] = rerank_on
                graph_state["hybrid_on"] = hybrid_on
                graph_state["layout"] = layout_val
                graph_state["max_nodes_val"] = max_nodes
                graph_state["max_edges_val"] = max_edges
                graph_adapter._limits = GraphLimits(max_nodes=max_nodes, max_edges=max_edges)

                if str(stage_select.value or DEFAULT_STAGE_KEY) == "sparse":
                    ui.notify("Sparse stage active. Switch to Stage 2 for enriched relation detail.", type="warning", timeout=1700)
                render_graph.refresh()
                render_detail.refresh()

            preset_select.on_value_change(lambda _: _apply_preset())
            stage_select.on_value_change(lambda _: _sync_stage_controls(update_dense_base=False))
            stage_switch_button.on_click(_go_next_stage)
            max_edges_input.on_value_change(lambda _: _sync_stage_controls(update_dense_base=True))
            ui.button("Search", on_click=do_search).classes("bg-blue-600 text-white px-6")
            _sync_stage_controls(update_dense_base=False)

        query_input.on("keydown.enter", do_search)

    with ui.element("div").style(
        "flex: 1; min-height: 0; display: flex; flex-direction: row;"
        " gap: 0.75rem; padding: 0.75rem; overflow: hidden; align-items: stretch;"
    ):
        with ui.card().style(
            "flex: 1 1 42rem; min-height: 0; overflow: hidden; padding: 0.75rem;"
            " display: flex; flex-direction: column;"
        ):
            ui.label("Code Graph (Cytoscape)").classes(
                "text-xs font-semibold uppercase tracking-wide text-gray-500 mb-1"
            )

            @ui.refreshable
            def render_graph() -> None:
                nonlocal graph_lookup, edge_lookup
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
                edge_lookup = {
                    edge.id: {
                        "id": edge.id,
                        "source": edge.source,
                        "target": edge.target,
                        "edge_type": edge.edge_type,
                        "direction": edge.direction,
                        "score": edge.score,
                        "explain": edge.explain,
                        "metadata": dict(edge.metadata or {}),
                    }
                    for edge in payload.edges
                }

                graph_state["serial"] += 1
                serial = graph_state["serial"]
                graph_state["last_payload"] = payload
                container_id = f"code-graph-canvas-tab-{serial}"
                payload_json = json.dumps(payload.to_cytoscape(), ensure_ascii=True)

                meta = payload.meta or {}
                node_count = len(payload.nodes)
                edge_count = len(payload.edges)
                truncated_nodes = meta.get("truncated_nodes", False)
                truncated_edges = meta.get("truncated_edges", False)
                related_hit_rate = meta.get("related_count_hit_rate", 0.0)
                relation_mode = "enriched" if bool(relation_toggle.value) else "vector-only"
                stage = get_stage(str(graph_state["stage_mode"]))
                current_stage_p95 = float(graph_state["latency_p95_ms"].get(stage.key, 0.0) or 0.0)

                meta_lines = [
                    f"Nodes: {node_count}",
                    f"Edges: {edge_count}",
                    f"Layout: {graph_state['layout']}",
                    f"Relation: {relation_mode}",
                    f"Stage: {stage.label}",
                ]
                if truncated_nodes:
                    meta_lines.append("⚠ nodes truncated")
                if truncated_edges:
                    meta_lines.append("⚠ edges truncated")
                if related_hit_rate >= 0:
                    meta_lines.append(f"hit-rate: {related_hit_rate:.1%}")
                if current_stage_p95 > 0:
                    meta_lines.append(f"latency p95: {current_stage_p95:.1f}ms")

                ui.label(" | ".join(meta_lines)).classes("text-xs text-gray-500 mb-2")

                ui.html(
                    f'<div id="{container_id}" '
                    'style="width:100%;height:100%;min-height:300px;border:1px solid #e5e7eb;border-radius:8px;"></div>'
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
                        layout: {{ name: {json.dumps(graph_state["layout"])}, directed: true, padding: 20, spacingFactor: 1.05 }},
                    }});
                    window.__codeGraphState.cy = cy;
                    cy.on('tap', 'node', (evt) => {{
                        window.__codeGraphQueue.push({{
                            serial,
                            event: 'node_click',
                            node_id: evt.target.id(),
                            edge_id: '',
                            payload: evt.target.data(),
                        }});
                    }});
                    cy.on('tap', 'edge', (evt) => {{
                        window.__codeGraphQueue.push({{
                            serial,
                            event: 'edge_click',
                            node_id: '',
                            edge_id: evt.target.id(),
                            payload: evt.target.data(),
                        }});
                    }});
                    cy.on('tap', (evt) => {{
                        if (evt.target !== cy) return;
                        window.__codeGraphQueue.push({{
                            serial,
                            event: 'canvas_click',
                            node_id: '',
                            edge_id: '',
                            payload: {{}},
                        }});
                    }});
                }})();
                """
                ui.timer(0.05, lambda js=init_graph_js: ui.run_javascript(js), once=True)

            render_graph()

        with ui.card().style(
            "flex: 1 1 22rem; min-width: 20rem; max-width: 30rem;"
            " min-height: 0; height: 100%; overflow-y: auto; padding: 0.75rem;"
        ):
            @ui.refreshable
            def render_detail() -> None:
                meta = ctrl.selected_metadata
                if not meta:
                    ui.label("Click a graph node or search result to inspect.").classes(
                        "text-gray-400 italic text-sm"
                    )
                    return

                detail_kind = str(meta.get("_detail_kind", "chunk"))
                detail_title = "Edge detail" if detail_kind == "edge" else "Chunk detail"
                ui.label(detail_title).classes("text-sm font-semibold text-gray-700 mb-2")
                with ui.grid(columns=2).classes("gap-x-3 gap-y-0.5 text-xs w-full"):
                    for key, val in meta.items():
                        if key in {"_content", "_detail_kind"}:
                            continue
                        ui.label(key).classes("font-mono text-gray-400 text-right truncate")
                        ui.label(str(val)).classes("font-mono text-gray-800 break-all")

                ui.separator().classes("my-2")
                ui.label("Content").classes("text-xs font-semibold text-gray-400 mb-1")
                ui.label(meta.get("_content", "")).classes(
                    "text-xs text-gray-700 whitespace-pre-wrap bg-gray-50 rounded p-2 w-full"
                )

            render_detail()

    async def _poll_graph_click_queue() -> None:
        event_start = time.perf_counter()
        try:
            raw_event = await ui.run_javascript(
                "(function(){const q=window.__codeGraphQueue||[]; return q.length ? q.shift() : null;})()"
            )
        except Exception:
            return

        if not raw_event:
            return

        if int(raw_event.get("serial", -1)) != int(graph_state["serial"]):
            return

        try:
            event = normalize_event(raw_event)
        except Exception:
            return

        if event.event == "node_click":
            node_id = str(event.node_id).strip()
            if not node_id:
                return
            node_data: dict[str, Any] = dict(event.payload or {})
            if node_id in graph_lookup:
                doc, score, score_key = graph_lookup[node_id]
                ctrl.select_chunk(doc, score, score_key)
            else:
                ctrl.handle_graph_node_click(node_id, node_data)
            render_detail.refresh()
            elapsed_ms = (time.perf_counter() - event_start) * 1000.0
            _record_latency(elapsed_ms)
            return

        if event.event == "edge_click":
            edge_id = str(event.edge_id).strip()
            edge_data: dict[str, Any] = dict(event.payload or {})
            if edge_id and edge_id in edge_lookup:
                edge_data = dict(edge_lookup[edge_id])
            ctrl.handle_graph_edge_click(edge_data)
            render_detail.refresh()
            elapsed_ms = (time.perf_counter() - event_start) * 1000.0
            _record_latency(elapsed_ms)
            return

        if event.event == "canvas_click":
            ctrl.handle_graph_canvas_click()
            render_detail.refresh()
            elapsed_ms = (time.perf_counter() - event_start) * 1000.0
            _record_latency(elapsed_ms)
            return

    def _record_latency(elapsed_ms: float) -> None:
        stage_key = str(graph_state["stage_mode"])
        bucket = graph_state["latency_samples_ms"].setdefault(stage_key, [])
        bucket.append(float(elapsed_ms))
        if len(bucket) > 300:
            del bucket[:-300]
        graph_state["latency_p95_ms"][stage_key] = interaction_latency_p95_ms(bucket)

    ui.timer(0.25, _poll_graph_click_queue)

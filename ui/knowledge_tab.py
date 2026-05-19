"""
Knowledge Card tab UI — Smart Note Cards from indexed chunks.

Chip types (each independently filterable and toggle-able):
  * Blue   (keyword)       -> ka_keywords   — B.1 KnowledgeExtractor
  * Purple (entity)        -> ka_entities   — B.1 KnowledgeExtractor
  * Green  (topic_cluster) -> topic_id      — B.3 TopicClusterer
  * Orange (ka_topic)      -> ka_topics     — B.1 KnowledgeExtractor

Exposes a single entry point:

    build(ctrl, on_ask=None)

`on_ask` is a zero-or-one-arg callable; the dashboard wires it so that
clicking a suggested question pre-fills the Search tab query input.
"""

from __future__ import annotations

import asyncio
from typing import Callable

from nicegui import ui

from ui.knowledge_controller import KnowledgeController


# ---------------------------------------------------------------------------
# Card rendering helper
# ---------------------------------------------------------------------------

def _parse_csv(raw: str) -> list[str]:
    return [s.strip() for s in str(raw or "").split(",") if s.strip()]


def _knowledge_card(
    chunk: dict,
    *,
    on_ask: Callable[[str], None],
    on_filter_keyword:       Callable[[str], None],
    on_filter_entity:        Callable[[str], None],
    on_filter_topic_cluster: Callable[[str], None],
    on_filter_ka_topic:      Callable[[str], None],
    on_click: Callable[[], None],
    show_keywords:       bool = True,
    show_entities:       bool = True,
    show_topic_clusters: bool = True,
    show_ka_topics:      bool = True,
) -> None:
    """Render one knowledge card into the current NiceGUI container."""
    meta    = chunk["metadata"]
    content = chunk["content"]

    summary        = str(meta.get("ka_summary", "")).strip()
    keywords       = _parse_csv(meta.get("ka_keywords", ""))
    entities       = _parse_csv(meta.get("ka_entities", ""))
    topic_clusters = [(meta.get("topic_id") or "").strip()]   # scalar -> list
    topic_clusters = [t for t in topic_clusters if t]         # drop empty
    ka_topics      = _parse_csv(meta.get("ka_topics", ""))
    questions      = _parse_csv(meta.get("ka_questions", ""))[:5]

    section    = (meta.get("section") or "").strip()
    title      = (meta.get("title") or meta.get("page_title") or "").strip()
    chunk_id   = meta.get("chunk_id", "")
    page       = meta.get("page", "")
    date       = str(meta.get("created_time") or meta.get("created_at") or "")[:10]
    importance = float(meta.get("importance", 0.0))

    heading = section or title or "(no heading)"

    # Chips to render (respects visibility toggles)
    visible_kw = keywords[:6]       if show_keywords       else []
    visible_en = entities[:3]       if show_entities        else []
    visible_tc = topic_clusters[:3] if show_topic_clusters else []
    visible_kt = ka_topics[:3]      if show_ka_topics       else []

    with ui.card().classes(
        "w-full hover:shadow-md transition-shadow border border-gray-100 cursor-pointer"
    ).on("click", on_click):
        # Header
        with ui.row().classes("items-start justify-between w-full gap-2"):
            with ui.column().classes("gap-0 flex-1 min-w-0"):
                ui.label(heading[:60]).classes(
                    "text-sm font-semibold text-gray-800 leading-tight truncate"
                )
                if title and section:
                    ui.label(title[:40]).classes("text-xs text-blue-500 truncate")
            ui.label(f"c{chunk_id}").classes(
                "text-xs font-mono text-gray-300 flex-shrink-0"
            )

        # Summary
        if summary:
            ui.label(summary).classes(
                "text-xs text-gray-600 leading-relaxed mt-1"
            ).style("display: -webkit-box; -webkit-line-clamp: 3;"
                    " -webkit-box-orient: vertical; overflow: hidden;")
        else:
            preview = content[:140].replace("\n", " ")
            ui.label(
                preview + ("…" if len(content) > 140 else "")
            ).classes("text-xs text-gray-400 italic mt-1")

        # Chips: Blue=keyword, Purple=entity, Green=cluster(topic_id), Orange=ka_topic
        if visible_kw or visible_en or visible_tc or visible_kt:
            with ui.row().classes("flex-wrap gap-1 mt-1.5"):
                for kw in visible_kw:
                    ui.label(kw).classes(
                        "text-xs bg-blue-50 text-blue-600 border border-blue-200"
                        " rounded-full px-2 py-0 cursor-pointer hover:bg-blue-100"
                    ).on("click.stop", lambda v=kw: on_filter_keyword(v))
                for en in visible_en:
                    ui.label(en).classes(
                        "text-xs bg-purple-50 text-purple-600 border border-purple-200"
                        " rounded-full px-2 py-0 cursor-pointer hover:bg-purple-100"
                    ).on("click.stop", lambda v=en: on_filter_entity(v))
                for tc in visible_tc:
                    ui.label(tc).classes(
                        "text-xs bg-green-50 text-green-600 border border-green-200"
                        " rounded-full px-2 py-0 cursor-pointer hover:bg-green-100"
                    ).on("click.stop", lambda v=tc: on_filter_topic_cluster(v))
                for kt in visible_kt:
                    ui.label(kt).classes(
                        "text-xs bg-orange-50 text-orange-600 border border-orange-200"
                        " rounded-full px-2 py-0 cursor-pointer hover:bg-orange-100"
                    ).on("click.stop", lambda v=kt: on_filter_ka_topic(v))

        # Suggested Questions
        if questions:
            ui.separator().classes("my-1.5")
            with ui.column().classes("gap-0.5"):
                for q in questions[:3]:
                    with ui.row().classes(
                        "items-start gap-1 cursor-pointer hover:bg-blue-50"
                        " rounded px-1 py-0.5"
                    ).on("click.stop", lambda qq=q: on_ask(qq)):
                        ui.icon("chat_bubble_outline").classes(
                            "text-xs text-blue-400 mt-0.5 flex-shrink-0"
                        ).style("font-size: 0.75rem;")
                        ui.label(q).classes("text-xs text-blue-600 leading-tight")

        # Footer
        footer_parts: list[str] = []
        if page != "" and page is not None:
            footer_parts.append(f"p.{page}")
        if date:
            footer_parts.append(date)
        if importance:
            footer_parts.append(f"★{importance:.1f}")
        if footer_parts:
            ui.label("  ·  ".join(footer_parts)).classes("text-xs text-gray-300 mt-1.5")


# ---------------------------------------------------------------------------
# Tab entry point
# ---------------------------------------------------------------------------

def build(
    ctrl: KnowledgeController,
    on_ask: Callable[[str], None] | None = None,
) -> None:
    """Build the complete Knowledge Card tab into the current NiceGUI context."""

    _on_ask = on_ask if on_ask is not None else lambda q: ui.notify(
        f"Search: {q}", type="info"
    )
    _view_mode: dict = {"grid": True}
    _chip_vis: dict = {
        "keywords":       True,
        "entities":       True,
        "topic_clusters": True,
        "ka_topics":      True,
    }

    # Toolbar
    with ui.card().classes("w-full rounded-none shadow-md p-3").style("flex-shrink: 0;"):
        ui.label("Knowledge Cards").classes("text-lg font-bold text-gray-800 mb-2")

        with ui.row().classes("w-full items-end gap-3 flex-wrap"):
            fi_doc = ui.select(
                options={"": "All Documents"} | ctrl.list_doc_title_map(),
                value="",
                label="Document",
            ).classes("w-52").props("outlined dense clearable")

            fi_keyword = ui.select(
                options=[""] + ctrl.list_keywords(),
                value="",
                label="🔵 Keyword",
            ).classes("w-36").props("outlined dense clearable")

            fi_entity = ui.select(
                options=[""] + ctrl.list_entities(),
                value="",
                label="🟣 Entity",
            ).classes("w-36").props("outlined dense clearable")

            fi_cluster = ui.select(
                options=[""] + ctrl.list_topic_clusters(),
                value="",
                label="🟢 Cluster Topic",
            ).classes("w-40").props("outlined dense clearable")

            fi_ka_topic = ui.select(
                options=[""] + ctrl.list_ka_topics(),
                value="",
                label="🟠 KA Topic",
            ).classes("w-40").props("outlined dense clearable")

            fi_text = ui.input(
                placeholder="Text search…"
            ).classes("flex-1 min-w-[10rem]").props("outlined dense clearable")

            def _apply_filters():
                ctrl.set_filter_doc_id(fi_doc.value or "")
                ctrl.set_filter_keyword(fi_keyword.value or "")
                ctrl.set_filter_entity(fi_entity.value or "")
                ctrl.set_filter_topic_cluster(fi_cluster.value or "")
                ctrl.set_filter_ka_topic(fi_ka_topic.value or "")
                ctrl.set_filter_text(fi_text.value or "")
                render_status.refresh()
                render_cards.refresh()

            def _do_reload():
                ui.notify("Loading chunks…", type="info", timeout=1500)
                ctrl.reload()
                fi_doc.set_options({"": "All Documents"} | ctrl.list_doc_title_map())
                fi_keyword.set_options([""] + ctrl.list_keywords())
                fi_entity.set_options([""] + ctrl.list_entities())
                fi_cluster.set_options([""] + ctrl.list_topic_clusters())
                fi_ka_topic.set_options([""] + ctrl.list_ka_topics())
                ctrl.clear_filter()
                _reset_dropdowns()
                render_status.refresh()
                render_cards.refresh()

            fi_doc.on_value_change(lambda _: _apply_filters())
            fi_keyword.on_value_change(lambda _: _apply_filters())
            fi_entity.on_value_change(lambda _: _apply_filters())
            fi_cluster.on_value_change(lambda _: _apply_filters())
            fi_ka_topic.on_value_change(lambda _: _apply_filters())
            fi_text.on("keydown.enter", _apply_filters)
            fi_text.on("blur", _apply_filters)

            ui.button("Reload", on_click=_do_reload).props("outlined").classes(
                "text-blue-600 border-blue-300"
            )

            _enrich_running: dict = {"running": False}

            async def _do_enrich_all():
                if _enrich_running["running"]:
                    ui.notify("Enrichment already in progress.", type="warning")
                    return
                _enrich_running["running"] = True
                enrich_btn.props("loading")
                ui.notify("Running knowledge extraction on all docs…", type="info", timeout=0,
                          close_button=True)
                try:
                    stats = await asyncio.to_thread(ctrl.enrich_all, False)
                    ui.notify(
                        f"Enrich all done — "
                        f"docs={stats['docs_processed']}  "
                        f"+{stats['enriched']}  "
                        f"skip={stats['skipped']}  "
                        f"fail={stats['failed']}",
                        type="positive",
                        timeout=8000,
                    )
                except Exception as exc:
                    ui.notify(f"Enrich all failed: {exc}", type="negative")
                finally:
                    _enrich_running["running"] = False
                    enrich_btn.props(remove="loading")

            enrich_btn = ui.button(
                "✦ Enrich All", on_click=_do_enrich_all
            ).props("outlined").classes("text-purple-600 border-purple-300")

            _cluster_running: dict = {"running": False}

            async def _do_cluster():
                if _cluster_running["running"]:
                    ui.notify("Clustering already in progress.", type="warning")
                    return
                _cluster_running["running"] = True
                cluster_btn.props("loading")
                ui.notify("Running topic clustering…", type="info", timeout=0, close_button=True)
                try:
                    topic_map = await asyncio.to_thread(ctrl.cluster_topics)
                    # Refresh cluster dropdown after assignment
                    fi_cluster.set_options([""] + ctrl.list_topic_clusters())
                    ui.notify(
                        f"Clustering done — "
                        f"{topic_map.n_clusters} topics  "
                        f"{topic_map.n_chunks} chunks",
                        type="positive",
                        timeout=6000,
                    )
                    render_cards.refresh()
                except Exception as exc:
                    ui.notify(f"Clustering failed: {exc}", type="negative")
                finally:
                    _cluster_running["running"] = False
                    cluster_btn.props(remove="loading")

            cluster_btn = ui.button(
                "⬡ Cluster Topics", on_click=_do_cluster
            ).props("outlined").classes("text-green-600 border-green-300")

            def _toggle_view():
                _view_mode["grid"] = not _view_mode["grid"]
                render_cards.refresh()

            ui.button(icon="view_module", on_click=_toggle_view).props(
                "flat round dense"
            ).tooltip("Toggle Grid / List view")

        # Row 2: chip visibility toggles
        with ui.row().classes("w-full items-center gap-4 mt-2 flex-wrap"):
            ui.label("Show chips:").classes("text-xs text-gray-400")

            def _make_toggle(key: str, label: str):
                ck = ui.checkbox(label, value=True).classes("text-xs")
                def _on_change(e, k=key):
                    _chip_vis[k] = e.value
                    render_cards.refresh()
                ck.on_value_change(_on_change)

            _make_toggle("keywords",       "🔵 Keywords")
            _make_toggle("entities",       "🟣 Entities")
            _make_toggle("topic_clusters", "🟢 Cluster Topics")
            _make_toggle("ka_topics",      "🟠 KA Topics")

        # Filter status row
        @ui.refreshable
        def render_status():
            total    = ctrl.total_count
            filtered = len(ctrl.filtered_chunks())
            with ui.row().classes("w-full items-center gap-3 flex-wrap mt-1"):
                if not ctrl.is_loaded:
                    ui.label("Click Reload to load knowledge cards.").classes(
                        "text-xs text-gray-400 italic"
                    )
                elif ctrl.filter.is_empty():
                    ui.label(f"{total} chunks total").classes(
                        "text-xs text-gray-400 italic"
                    )
                else:
                    ui.label(
                        f"{filtered} / {total}  ·  filter: {ctrl.filter.summary()}"
                    ).classes(
                        "text-xs font-mono bg-blue-50 text-blue-700"
                        " border border-blue-200 rounded px-2 py-0.5"
                    )
                    ui.button(
                        "× Clear",
                        on_click=lambda: (
                            ctrl.clear_filter(),
                            _reset_dropdowns(),
                            render_status.refresh(),
                            render_cards.refresh(),
                        ),
                    ).props("flat dense").classes("text-xs text-red-400 px-1 py-0")

        render_status()

    def _reset_dropdowns():
        fi_doc.set_value("")
        fi_keyword.set_value("")
        fi_entity.set_value("")
        fi_cluster.set_value("")
        fi_ka_topic.set_value("")
        fi_text.set_value("")

    def _click_keyword(kw: str) -> None:
        fi_keyword.set_options([""] + ctrl.list_keywords())
        ctrl.set_filter_keyword(kw)
        fi_keyword.set_value(kw)
        render_status.refresh()
        render_cards.refresh()

    def _click_entity(en: str) -> None:
        fi_entity.set_options([""] + ctrl.list_entities())
        ctrl.set_filter_entity(en)
        fi_entity.set_value(en)
        render_status.refresh()
        render_cards.refresh()

    def _click_topic_cluster(tc: str) -> None:
        fi_cluster.set_options([""] + ctrl.list_topic_clusters())
        ctrl.set_filter_topic_cluster(tc)
        fi_cluster.set_value(tc)
        render_status.refresh()
        render_cards.refresh()

    def _click_ka_topic(kt: str) -> None:
        fi_ka_topic.set_options([""] + ctrl.list_ka_topics())
        ctrl.set_filter_ka_topic(kt)
        fi_ka_topic.set_value(kt)
        render_status.refresh()
        render_cards.refresh()

    # ── Body: card grid + detail panel ──────────────────────────────────
    _selected: dict = {"chunk": None}

    # Use a plain div instead of ui.row() to avoid Quasar's flex-wrap:wrap
    # which prevents the inner scrollable area from getting a fixed height.
    with ui.element("div").style(
        "flex: 1; min-height: 0; display: flex; flex-direction: row;"
        " gap: 0.75rem; padding: 0.75rem; overflow: hidden; align-items: stretch;"
    ):
        # Left — scrollable card grid
        with ui.element("div").style("flex: 1; min-height: 0; overflow-y: auto;"):

            @ui.refreshable
            def render_cards():
                chunks = ctrl.filtered_chunks()

                if not ctrl.is_loaded:
                    with ui.column().classes("items-center justify-center w-full py-16"):
                        ui.icon("auto_stories").classes("text-5xl text-gray-300")
                        ui.label("Click Reload to browse knowledge cards.").classes(
                            "text-gray-400 mt-2"
                        )
                    return

                if not chunks:
                    with ui.column().classes("items-center justify-center w-full py-16"):
                        ui.icon("search_off").classes("text-5xl text-gray-300")
                        ui.label("No chunks match the current filter.").classes(
                            "text-gray-400 mt-2"
                        )
                    return

                if _view_mode["grid"]:
                    container = ui.element("div").style(
                        "display: grid;"
                        " grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));"
                        " gap: 0.75rem;"
                    )
                else:
                    container = ui.column().classes("gap-2 w-full")

                with container:
                    for chunk in chunks:
                        _knowledge_card(
                            chunk,
                            on_ask=_on_ask,
                            on_filter_keyword=_click_keyword,
                            on_filter_entity=_click_entity,
                            on_filter_topic_cluster=_click_topic_cluster,
                            on_filter_ka_topic=_click_ka_topic,
                            on_click=lambda c=chunk: (
                                _selected.__setitem__("chunk", c),
                                render_detail.refresh(),
                            ),
                            show_keywords=_chip_vis["keywords"],
                            show_entities=_chip_vis["entities"],
                            show_topic_clusters=_chip_vis["topic_clusters"],
                            show_ka_topics=_chip_vis["ka_topics"],
                        )

            render_cards()

        # Right — chunk detail panel
        with ui.card().style(
            "width: 22rem; flex-shrink: 0; min-height: 0; height: 100%;"
            " overflow-y: auto; padding: 0.75rem;"
        ):
            @ui.refreshable
            def render_detail():
                c = _selected["chunk"]
                if c is None:
                    ui.label("Click a card to inspect.").classes(
                        "text-gray-400 italic text-sm"
                    )
                    return

                meta    = c["metadata"]
                content = c["content"]

                ui.label("Chunk detail").classes(
                    "text-sm font-semibold text-gray-700 mb-2"
                )

                # AI knowledge fields
                ka_fields = [
                    ("Summary",       meta.get("ka_summary",   "")),
                    ("Keywords",      meta.get("ka_keywords",  "")),
                    ("Entities",      meta.get("ka_entities",  "")),
                    ("KA Topics",     meta.get("ka_topics",    "")),
                    ("Cluster Topic", meta.get("topic_id",     "")),
                    ("Questions",     meta.get("ka_questions", "")),
                ]
                for label, val in ka_fields:
                    if val:
                        ui.label(label).classes(
                            "text-xs font-semibold text-gray-500 mt-1"
                        )
                        ui.label(str(val)).classes(
                            "text-xs text-gray-700 bg-gray-50 rounded p-1 w-full"
                        )

                ui.separator().classes("my-2")

                # Raw metadata
                skip = {"ka_summary", "ka_keywords", "ka_entities",
                        "ka_topics", "ka_questions", "topic_id"}
                with ui.grid(columns=2).classes("gap-x-3 gap-y-0.5 text-xs w-full"):
                    for key, val in meta.items():
                        if key in skip:
                            continue
                        ui.label(key).classes(
                            "font-mono text-gray-400 text-right truncate"
                        )
                        ui.label(str(val)).classes(
                            "font-mono text-gray-800 break-all"
                        )

                ui.separator().classes("my-2")
                ui.label("Content").classes(
                    "text-xs font-semibold text-gray-400 mb-1"
                )
                ui.label(content).classes(
                    "text-xs text-gray-700 whitespace-pre-wrap"
                    " bg-gray-50 rounded p-2 w-full"
                )

            render_detail()

"""
Knowledge Card tab UI — Smart Note Cards from indexed chunks.

Each card shows:
  - Section heading / page title
  - AI-generated summary (ka_summary)
  - Keyword chips (ka_keywords) — clickable → filter by tag
  - Entity chips   (ka_entities) — clickable → filter by tag
  - Topic chips    (topics) — clickable → filter by topic
  - Suggested questions (ka_questions) — clickable → ask in Search tab
  - Metadata footer (page, date, importance)

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
    on_filter_tag: Callable[[str], None],
    on_filter_topic: Callable[[str], None],
    on_click: Callable[[], None],
) -> None:
    """Render one knowledge card into the current NiceGUI container."""
    meta    = chunk["metadata"]
    content = chunk["content"]

    summary   = str(meta.get("ka_summary", "")).strip()
    keywords  = _parse_csv(meta.get("ka_keywords", ""))
    entities  = _parse_csv(meta.get("ka_entities", ""))
    topics    = _parse_csv(meta.get("topics", ""))
    questions = _parse_csv(meta.get("ka_questions", ""))[:5]

    section   = (meta.get("section") or "").strip()
    title     = (meta.get("title") or meta.get("page_title") or "").strip()
    chunk_id  = meta.get("chunk_id", "")
    page      = meta.get("page", "")
    date      = str(meta.get("created_time") or meta.get("created_at") or "")[:10]
    importance = float(meta.get("importance", 0.0))

    heading = section or title or "(no heading)"

    with ui.card().classes(
        "w-full hover:shadow-md transition-shadow border border-gray-100 cursor-pointer"
    ).on("click", on_click):
        # ── Header ───────────────────────────────────────────────────────
        with ui.row().classes("items-start justify-between w-full gap-2"):
            with ui.column().classes("gap-0 flex-1 min-w-0"):
                ui.label(heading[:60]).classes(
                    "text-sm font-semibold text-gray-800 leading-tight truncate"
                )
                if title and section:
                    ui.label(title[:40]).classes(
                        "text-xs text-blue-500 truncate"
                    )
            ui.label(f"c{chunk_id}").classes(
                "text-xs font-mono text-gray-300 flex-shrink-0"
            )

        # ── Summary ──────────────────────────────────────────────────────
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

        # ── Chips: keywords / entities / topics ──────────────────────────
        if keywords or entities or topics:
            with ui.row().classes("flex-wrap gap-1 mt-1.5"):
                for kw in keywords[:6]:
                    ui.label(kw).classes(
                        "text-xs bg-blue-50 text-blue-600 border border-blue-200"
                        " rounded-full px-2 py-0 cursor-pointer hover:bg-blue-100"
                    ).on("click.stop", lambda k=kw: on_filter_tag(k))
                for en in entities[:3]:
                    ui.label(en).classes(
                        "text-xs bg-purple-50 text-purple-600 border border-purple-200"
                        " rounded-full px-2 py-0 cursor-pointer hover:bg-purple-100"
                    ).on("click.stop", lambda e=en: on_filter_tag(e))
                for tp in topics[:3]:
                    ui.label(tp).classes(
                        "text-xs bg-green-50 text-green-600 border border-green-200"
                        " rounded-full px-2 py-0 cursor-pointer hover:bg-green-100"
                    ).on("click.stop", lambda t=tp: on_filter_topic(t))

        # ── Suggested Questions ───────────────────────────────────────────
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
                        ui.label(q).classes(
                            "text-xs text-blue-600 leading-tight"
                        )

        # ── Footer ────────────────────────────────────────────────────────
        footer_parts: list[str] = []
        if page != "" and page is not None:
            footer_parts.append(f"p.{page}")
        if date:
            footer_parts.append(date)
        if importance:
            footer_parts.append(f"★{importance:.1f}")
        if footer_parts:
            ui.label("  ·  ".join(footer_parts)).classes(
                "text-xs text-gray-300 mt-1.5"
            )


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
    _view_mode: dict = {"grid": True}  # mutable closure cell

    # ── Toolbar ───────────────────────────────────────────────────────────
    with ui.card().classes("w-full rounded-none shadow-md p-3").style("flex-shrink: 0;"):
        ui.label("Knowledge Cards").classes("text-lg font-bold text-gray-800 mb-2")

        with ui.row().classes("w-full items-end gap-3 flex-wrap"):
            fi_doc = ui.select(
                options={"": "All Documents"} | ctrl.list_doc_title_map(),
                value="",
                label="Document",
            ).classes("w-52").props("outlined dense clearable")

            fi_topic = ui.select(
                options=[""] + ctrl.list_topics(),
                value="",
                label="Topic",
            ).classes("w-40").props("outlined dense clearable")

            fi_tag = ui.select(
                options=[""] + ctrl.list_tags(),
                value="",
                label="Tag / Keyword",
            ).classes("w-40").props("outlined dense clearable")

            fi_text = ui.input(
                placeholder="Text search…"
            ).classes("flex-1 min-w-[10rem]").props("outlined dense clearable")

            def _apply_filters():
                ctrl.set_filter_doc_id(fi_doc.value or "")
                ctrl.set_filter_topic(fi_topic.value or "")
                ctrl.set_filter_tag(fi_tag.value or "")
                ctrl.set_filter_text(fi_text.value or "")
                render_status.refresh()
                render_cards.refresh()

            def _do_reload():
                ui.notify("Loading chunks…", type="info", timeout=1500)
                ctrl.reload()
                # Refresh dropdown options after reload
                fi_doc.set_options({"": "All Documents"} | ctrl.list_doc_title_map())
                fi_topic.set_options([""] + ctrl.list_topics())
                fi_tag.set_options([""] + ctrl.list_tags())
                ctrl.clear_filter()
                fi_doc.set_value("")
                fi_topic.set_value("")
                fi_tag.set_value("")
                fi_text.set_value("")
                render_status.refresh()
                render_cards.refresh()

            fi_doc.on_value_change(lambda _: _apply_filters())
            fi_topic.on_value_change(lambda _: _apply_filters())
            fi_tag.on_value_change(lambda _: _apply_filters())
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

            def _toggle_view():
                _view_mode["grid"] = not _view_mode["grid"]
                render_cards.refresh()

            ui.button(icon="view_module", on_click=_toggle_view).props(
                "flat round dense"
            ).tooltip("Toggle Grid / List view")

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
        fi_topic.set_value("")
        fi_tag.set_value("")
        fi_text.set_value("")

    def _click_filter_tag(tag: str) -> None:
        ctrl.set_filter_tag(tag)
        fi_tag.set_value(tag)
        render_status.refresh()
        render_cards.refresh()

    def _click_filter_topic(topic: str) -> None:
        ctrl.set_filter_topic(topic)
        fi_topic.set_value(topic)
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
                            on_filter_tag=_click_filter_tag,
                            on_filter_topic=_click_filter_topic,
                            on_click=lambda c=chunk: (
                                _selected.__setitem__("chunk", c),
                                render_detail.refresh(),
                            ),
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
                    ("Summary",   meta.get("ka_summary",   "")),
                    ("Keywords",  meta.get("ka_keywords",  "")),
                    ("Entities",  meta.get("ka_entities",  "")),
                    ("Questions", meta.get("ka_questions", "")),
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
                skip = {"ka_summary", "ka_keywords", "ka_entities", "ka_questions"}
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

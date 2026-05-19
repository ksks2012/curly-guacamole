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

import asyncio

from nicegui import ui

from utils.config import AppConfig
from utils.logger import AppLogger
from rag.client import LocalLlamaClient
from ui.search_controller import SearchController
from ui.index_controller import IndexController
from ui.config_controller import ConfigController
from ui.knowledge_controller import KnowledgeController
import ui.search_tab as search_tab_ui
import ui.index_tab as index_tab_ui
import ui.trace_tab as trace_tab_ui
import ui.config_tab as config_tab_ui
import ui.knowledge_tab as knowledge_tab_ui
import ui.notion_tab as notion_tab_ui
from ui.notion_controller import NotionController

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
_cfg_ctrl = ConfigController(_config)
_know_ctrl = KnowledgeController(_client)
_notion_ctrl = NotionController(_config)
log.info("RAG client ready")


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
        # ── Tab bar (pinned top) ───────────────────────────────────────────
        with ui.tabs().classes("w-full bg-white shadow-sm").style(
            "flex-shrink: 0;"
        ) as tabs:
            search_tab = ui.tab("Search").props("no-caps")
            trace_tab  = ui.tab("Trace").props("no-caps")
            index_tab  = ui.tab("Index").props("no-caps")
            know_tab   = ui.tab("Knowledge").props("no-caps")
            notion_tab = ui.tab("Notion").props("no-caps")
            config_tab = ui.tab("Config").props("no-caps")

        # Shared callback list — trace tab appends its refresh here
        _on_search: list = []

        with ui.tab_panels(tabs, value=search_tab).classes("p-0 w-full").style(
            "flex: 1; min-height: 0; overflow: hidden;"
            " display: flex; flex-direction: column;"
        ):
            with ui.tab_panel(search_tab).classes("p-0").style(
                "height: 100%; display: flex; flex-direction: column; overflow: hidden;"
            ):
                search_result = search_tab_ui.build(_ctrl, on_search=_on_search)
                fi_source = search_result.fi_source

            with ui.tab_panel(trace_tab).classes("p-0").style(
                "height: 100%; display: flex; flex-direction: column; overflow: hidden;"
            ):
                refresh_trace = trace_tab_ui.build(_ctrl)
                _on_search.append(refresh_trace)

            with ui.tab_panel(index_tab).classes("p-0").style(
                "height: 100%; display: flex; flex-direction: row; overflow: hidden;"
            ):
                index_tab_ui.build(
                    _idx_ctrl, log,
                    on_doc_indexed=lambda: fi_source.set_options(
                        {"": ""} | _ctrl.list_doc_title_map()
                    ),
                )

            with ui.tab_panel(config_tab).classes("p-0").style(
                "height: 100%; display: flex; flex-direction: row; overflow: hidden;"
            ):
                config_tab_ui.build(_cfg_ctrl)

            with ui.tab_panel(know_tab).classes("p-0").style(
                "height: 100%; display: flex; flex-direction: column; overflow: hidden;"
            ):
                def _ask_from_knowledge(question: str) -> None:
                    search_result.query_input.set_value(question)
                    tabs.set_value(search_tab)

                knowledge_tab_ui.build(_know_ctrl, on_ask=_ask_from_knowledge)

            with ui.tab_panel(notion_tab).classes("p-0").style(
                "height: 100%; display: flex; flex-direction: row; overflow: hidden;"
            ):
                notion_tab_ui.build(_notion_ctrl)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="RAG Debug Dashboard", port=8888, reload=False)

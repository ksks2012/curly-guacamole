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
import ui.search_tab as search_tab_ui
import ui.index_tab as index_tab_ui

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
            index_tab  = ui.tab("Index").props("no-caps")

        with ui.tab_panels(tabs, value=search_tab).classes("p-0 w-full").style(
            "flex: 1; min-height: 0; overflow: hidden;"
            " display: flex; flex-direction: column;"
        ):
            with ui.tab_panel(search_tab).classes("p-0").style(
                "height: 100%; display: flex; flex-direction: column; overflow: hidden;"
            ):
                fi_source = search_tab_ui.build(_ctrl)

            with ui.tab_panel(index_tab).classes("p-0").style(
                "height: 100%; display: flex; flex-direction: row; overflow: hidden;"
            ):
                index_tab_ui.build(
                    _idx_ctrl, log,
                    on_doc_indexed=lambda: fi_source.set_options(
                        [""] + _ctrl.list_doc_ids()
                    ),
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="RAG Debug Dashboard", port=8888, reload=False)

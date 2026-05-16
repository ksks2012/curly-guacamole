"""
Config tab UI — form-based editor for application settings.

Exposes a single entry point:

    build(cfg_ctrl)

Each setting is rendered as a labelled row with an appropriate input widget
(text field, number, boolean toggle, dropdown, or JSON input).
Fields are grouped by section in collapsible panels.

Layout:
┌─ toolbar ─────────────────────────────────────────────────────────┐
│  Configuration   [path]          [status]  [↺ Reload]  [Save]    │
├─ scrollable body ─────────────────────────────────────────────────┤
│  ▶ LLM Server                                                    │
│    Embed Base URL   OpenAI-compatible…   [input............] ⚠   │
│    Embed Model      Model name…          [input............] ⚠   │
│    ───────────────────────────────────────────────────────────── │
│  ▶ Query Expansion                                               │
│    Enable Expansion  Generate alt…       [toggle]         ●      │
└───────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import json

from nicegui import ui

from ui.config_controller import ConfigController, FIELD_SCHEMA


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build(cfg_ctrl: ConfigController) -> None:
    """Build the Config tab into the current NiceGUI context."""
    _widgets: dict[str, ui.element] = {}

    def _collect() -> dict:
        data: dict = {}
        for section in FIELD_SCHEMA:
            for key, _, field_type, _reload, _desc in section["fields"]:
                if key not in _widgets:
                    continue
                w = _widgets[key]
                if field_type == "bool":
                    data[key] = bool(w.value)
                elif field_type == "int":
                    try:
                        data[key] = int(w.value or 0)
                    except (TypeError, ValueError):
                        data[key] = 0
                elif field_type == "json":
                    try:
                        data[key] = json.loads(w.value or "{}")
                    except json.JSONDecodeError:
                        data[key] = {}
                else:
                    data[key] = str(w.value or "")
        return data

    def _reload_widgets() -> None:
        data = cfg_ctrl.get_data()
        for section in FIELD_SCHEMA:
            for key, _, field_type, _reload, _desc in section["fields"]:
                if key not in _widgets:
                    continue
                w   = _widgets[key]
                val = data.get(key, "")
                if field_type == "bool":
                    w.set_value(bool(val))
                elif field_type == "int":
                    try:
                        w.set_value(int(val or 0))
                    except (TypeError, ValueError):
                        w.set_value(0)
                elif field_type == "json":
                    w.set_value(json.dumps(val) if isinstance(val, (dict, list)) else str(val or "{}"))
                elif field_type.startswith("select:"):
                    opts = field_type.split(":", 1)[1].split(",")
                    w.set_value(str(val) if str(val) in opts else opts[0])
                else:
                    w.set_value(str(val or ""))

    with ui.column().classes("w-full").style(
        "height: 100%; display: flex; flex-direction: column; overflow: hidden;"
    ):
        # ── Toolbar (pinned) ──────────────────────────────────────────
        with ui.row().classes(
            "w-full items-center px-4 py-2 gap-3 border-b border-gray-200 bg-white"
        ).style("flex-shrink: 0;"):
            ui.label("Configuration").classes("text-base font-bold text-gray-800")
            ui.label(cfg_ctrl.config_path()).classes(
                "text-xs text-gray-400 font-mono flex-1"
            ).style("overflow: hidden; text-overflow: ellipsis; white-space: nowrap;")

            status_label = ui.label("").classes("text-sm text-gray-500")

            def _on_reload() -> None:
                _reload_widgets()
                status_label.set_text("Reloaded from disk.")
                status_label.classes(remove="text-green-600 text-red-600 text-orange-500")
                status_label.classes(add="text-gray-500")

            def _on_save() -> None:
                data = _collect()
                hot_ch, restart_ch = cfg_ctrl.changed_keys_from_data(data)
                ok, msg = cfg_ctrl.save_data(data)
                if ok:
                    if restart_ch:
                        status_label.set_text(
                            f"✓ Saved — restart required for: {', '.join(restart_ch)}"
                        )
                        status_label.classes(remove="text-green-600 text-red-600 text-gray-500")
                        status_label.classes(add="text-orange-500")
                    else:
                        status_label.set_text("✓ Saved — all changes are active.")
                        status_label.classes(remove="text-orange-500 text-red-600 text-gray-500")
                        status_label.classes(add="text-green-600")
                else:
                    status_label.set_text(f"✗ {msg}")
                    status_label.classes(remove="text-green-600 text-orange-500 text-gray-500")
                    status_label.classes(add="text-red-600")

            ui.button("↺ Reload", on_click=_on_reload).props("flat color=grey dense")
            ui.button("Save", on_click=_on_save).props("unelevated color=primary dense").classes("px-5")

        # ── Scrollable body ───────────────────────────────────────────
        with ui.scroll_area().classes("w-full").style("flex: 1; min-height: 0;"):
            with ui.column().classes("w-full p-4 gap-3"):
                current_data = cfg_ctrl.get_data()
                for section_info in FIELD_SCHEMA:
                    _build_section(section_info, current_data, _widgets)


# ---------------------------------------------------------------------------
# Section / field builders
# ---------------------------------------------------------------------------

def _build_section(section_info: dict, current_data: dict, widgets: dict) -> None:
    with ui.expansion(section_info["section"], value=True).classes(
        "w-full border border-gray-200 rounded-lg shadow-none"
    ).props("dense"):
        with ui.column().classes("w-full px-4 pb-3 gap-0"):
            for i, (key, label, field_type, reload_mode, desc) in enumerate(section_info["fields"]):
                value   = current_data.get(key, "")
                divider = " border-t border-gray-100" if i > 0 else ""
                with ui.row().classes(f"w-full items-center gap-3 py-2{divider}"):
                    with ui.column().classes("gap-0").style("flex: 1; min-width: 0;"):
                        ui.label(label).classes("text-sm font-medium text-gray-800")
                        if desc:
                            ui.label(desc).classes("text-xs text-gray-400 leading-tight")
                    _make_widget(key, field_type, value, widgets)
                    _badge(reload_mode)


def _make_widget(key: str, field_type: str, value, widgets: dict) -> ui.element:
    if field_type == "bool":
        w = ui.switch(value=bool(value))
    elif field_type == "int":
        try:
            v = int(value or 0)
        except (TypeError, ValueError):
            v = 0
        w = ui.number(value=v, min=0).classes("w-28").props("outlined dense")
    elif field_type.startswith("select:"):
        opts = field_type.split(":", 1)[1].split(",")
        cur  = str(value) if str(value) in opts else opts[0]
        w    = ui.select(options=opts, value=cur).classes("w-44").props("outlined dense")
    elif field_type == "json":
        disp = json.dumps(value) if isinstance(value, (dict, list)) else str(value or "{}")
        w    = ui.input(value=disp).classes("w-56 font-mono").props("outlined dense")
    else:
        w = ui.input(value=str(value or "")).classes("w-72").props("outlined dense")
    widgets[key] = w
    return w


def _badge(reload_mode: str) -> None:
    if reload_mode == "hot":
        ui.label("● active").classes(
            "text-xs px-2 py-0.5 rounded-full bg-green-100 text-green-700 font-medium whitespace-nowrap"
        )
    elif reload_mode == "restart":
        ui.label("⚠ restart").classes(
            "text-xs px-2 py-0.5 rounded-full bg-orange-100 text-orange-600 font-medium whitespace-nowrap"
        )

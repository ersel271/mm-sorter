# src/ui/keybinds.py
"""
Keyboard shortcut handling for the CV display window.

For registering a new key:
    1. add an entry to _BINDS: ord("x"): "action_name"
    2. implement _on_action_name(ov: Overlay) -> bool
    3. add a case "action_name" block in handle_key

Usage:
    if handle_key(ov.show(frame), ov):
        break
"""

import logging

from src.ui.overlay import Overlay

log = logging.getLogger(__name__)

_BINDS: dict[int, str] = {
    ord("q"): "quit",
    ord("d"): "debug",
    ord("t"): "sidebar",
    ord("l"): "log",
}

def handle_key(key: int, ov: Overlay) -> bool:
    """process a keypress and return True if the app should quit."""
    match _BINDS.get(key & 0xFF):
        case "quit":
            return _on_quit(ov)
        case "debug":
            return _on_debug(ov)
        case "sidebar":
            return _on_sidebar(ov)
        case "log":
            return _on_log(ov)
        case _:
            return False

# actions

def _on_quit(ov: Overlay) -> bool:
    return True

def _on_debug(ov: Overlay) -> bool:
    ov.toggle_debug()
    log.info("debug overlay %s", "on" if ov.debug else "off")
    return False

def _on_sidebar(ov: Overlay) -> bool:
    ov.toggle_sidebar()
    return False

def _on_log(ov: Overlay) -> bool:
    ov.toggle_log()
    return False

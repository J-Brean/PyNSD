"""
gui/errors.py
-------------
Turns an unhandled exception into something the user can see and send on.

PyQt6 hands an exception raised inside a slot to ``sys.excepthook`` and then
carries on, so without this a button simply does nothing: the traceback goes to
a console that a user who launched PyNSD from a shortcut never sees.  Every one
of those is a "silent crash" from the outside.
"""
from __future__ import annotations

import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QStandardPaths
from PyQt6.QtWidgets import QApplication, QMessageBox

_MAX_REPEATS = 3                                   # don't let a repeating fault bury the screen
_seen: dict[str, int] = {}


def log_path() -> Path:
    """Where the traceback is written, so a user can attach it to a bug report."""
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    folder = Path(base) if base else Path.home() / ".pynsd"
    if folder.name.lower() != "pynsd":                       # before the app name is set
        folder = folder / "PyNSD"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "errors.log"


def _record(text: str) -> Path | None:
    try:
        path = log_path()
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} =====\n{text}")
        return path
    except Exception:
        return None                                # logging must never itself raise


def _report(exc_type, exc, tb) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, tb)
        return

    text = "".join(traceback.format_exception(exc_type, exc, tb))
    path = _record(text)
    sys.__excepthook__(exc_type, exc, tb)          # keep the console behaviour too

    key = f"{exc_type.__name__}:{exc}"
    _seen[key] = _seen.get(key, 0) + 1
    if _seen[key] > _MAX_REPEATS or QApplication.instance() is None:
        return

    box = QMessageBox()
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("PyNSD hit a problem")
    box.setText("That action stopped part way through, so nothing was changed.")
    box.setInformativeText(
        f"{exc_type.__name__}: {exc}\n\n"
        "This usually means the data was not the shape the step expected. "
        "Check the file and settings for that step, then try again."
        + (f"\n\nDetails saved to:\n{path}" if path else ""))
    box.setDetailedText(text)
    if _seen[key] == _MAX_REPEATS:
        box.setInformativeText(box.informativeText()
                               + "\n\nThis has happened repeatedly; further reports "
                                 "of the same fault will be logged but not shown.")
    box.exec()


def install() -> None:
    """Route unhandled exceptions, on the main thread and on workers, to a dialog."""
    sys.excepthook = _report
    threading.excepthook = lambda args: _report(args.exc_type, args.exc_value, args.exc_traceback)

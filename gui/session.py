"""
gui/session.py
--------------
Remembers what was open, so a restart picks up where you left off.

Only the file paths and the import settings are stored: the data itself is
re-read from disk on restore.  That means a session survives a crash, because it
is written whenever the file list changes rather than only on a clean exit.

Corrections applied inside the Load panel (QC, line loss, normalise, filters) are
recorded by name for reference but are not replayed, since replaying them blindly
onto freshly-read data could double-apply them.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QSettings, QStandardPaths

MAX_RECENT = 8


def _folder() -> Path:
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    folder = Path(base) if base else Path.home() / ".pynsd"
    if folder.name.lower() != "pynsd":
        folder = folder / "PyNSD"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def session_path() -> Path:
    return _folder() / "session.json"


# ---- Recent files ------------------------------------------------------- #

def recent_files() -> list[str]:
    stored = QSettings().value("recent/files", [], type=list) or []
    return [p for p in stored if Path(p).exists()]


def remember_recent(paths) -> None:
    """Push these to the top of the recent list, most recent first, no repeats."""
    current = [p for p in (QSettings().value("recent/files", [], type=list) or [])]
    for path in reversed([str(p) for p in paths]):
        if path in current:
            current.remove(path)
        current.insert(0, path)
    QSettings().setValue("recent/files", current[:MAX_RECENT])


def clear_recent() -> None:
    QSettings().remove("recent/files")


# ---- The session itself -------------------------------------------------- #

def save(files: list[dict], settings: dict, confirmed: bool, corrections: list[str]) -> None:
    """Write the current session. Called whenever the file list changes."""
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "files": files,
        "settings": settings,
        "confirmed": confirmed,
        "corrections": corrections,
    }
    try:
        session_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass                                      # never let bookkeeping break the app


def load() -> dict | None:
    """The last session, or None if there is not a usable one."""
    try:
        payload = json.loads(session_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    payload["files"] = [f for f in payload.get("files", []) if Path(f.get("path", "")).exists()]
    return payload if payload["files"] else None


def clear() -> None:
    session_path().unlink(missing_ok=True)

"""
gui/filedialogs.py
------------------
File dialogs that open where the user last was.

Every dialog in PyNSD used to start in the working directory, so each load and
each export meant navigating back to the data folder again.  These wrappers take
and return exactly what the Qt static methods do, so call sites are unchanged
apart from the name.
"""
from __future__ import annotations

import os

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QFileDialog

_LAST_DIR = "paths/last_dir"


def _settings() -> QSettings:
    return QSettings("PyNSD", "PyNSD")


def _start_from(hint: str) -> str:
    """Resolve a caller's hint against the folder last used.

    An empty hint means "wherever we were"; a bare filename is a suggested name
    to place there; anything with a directory in it is left alone.
    """
    remembered = _settings().value(_LAST_DIR, "", type=str)
    if not hint:
        return remembered
    if os.path.dirname(hint) or not remembered:
        return hint
    return os.path.join(remembered, hint)


def _remember(path: str) -> None:
    if path:
        _settings().setValue(_LAST_DIR, os.path.dirname(os.path.abspath(path)))


def get_open_file_name(parent, caption, start="", filter="", **kw) -> tuple[str, str]:
    path, selected = QFileDialog.getOpenFileName(parent, caption, _start_from(start), filter, **kw)
    _remember(path)
    return path, selected


def get_open_file_names(parent, caption, start="", filter="", **kw) -> tuple[list[str], str]:
    paths, selected = QFileDialog.getOpenFileNames(parent, caption, _start_from(start), filter, **kw)
    if paths:
        _remember(paths[0])
    return paths, selected


def get_save_file_name(parent, caption, start="", filter="", **kw) -> tuple[str, str]:
    path, selected = QFileDialog.getSaveFileName(parent, caption, _start_from(start), filter, **kw)
    _remember(path)
    return path, selected


def get_existing_directory(parent, caption, start="", **kw) -> str:
    path = QFileDialog.getExistingDirectory(parent, caption, _start_from(start), **kw)
    if path:
        _settings().setValue(_LAST_DIR, path)
    return path

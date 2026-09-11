from contextlib import contextmanager

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication


@contextmanager
def busy_cursor():
    """Show the wait cursor while a slow step runs, so the app does not look stuck."""
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        yield
    finally:
        QApplication.restoreOverrideCursor()


def available_geometry(widget=None):
    """Usable area (taskbar excluded) of the screen the widget is on."""
    screen = widget.screen() if widget is not None else None
    if screen is None:
        screen = QApplication.primaryScreen()
    return screen.availableGeometry() if screen else None


def fit_to_screen(widget, width=None, height=None, margin=0.94, centre=True):
    """Resize a window to the requested size, capped to the screen it sits on.

    Windows laptops are routinely 1366x768, or 1920x1080 at 150% scaling
    (1280x720 logical), so any hard-coded pixel size has to be clamped or the
    window opens larger than the desktop and cannot be reached.
    """
    if width is None:
        width = widget.width()
    if height is None:
        height = widget.height()

    geo = available_geometry(widget)
    if geo is not None:
        width = min(int(width), int(geo.width() * margin))
        height = min(int(height), int(geo.height() * margin))

    widget.resize(int(width), int(height))

    if centre and geo is not None and not widget.isVisible():          # Don't yank a window the user is already using
        frame = widget.frameGeometry()
        frame.moveCenter(geo.center())
        widget.move(max(geo.left(), frame.left()), max(geo.top(), frame.top()))

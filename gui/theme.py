"""
gui/theme.py
------------
Single source of truth for PyNSD's visual identity.

Defines the colour palette, spacing scale, typography and button classes, and
loads the shared stylesheet (``gui/style.qss``).  Panels should never call
``setStyleSheet`` directly: they set object names / ``class`` properties and let
the central stylesheet do the styling.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QPainter, QPixmap
from PyQt6.QtWidgets import QSplashScreen, QWidget

# ─────────────────────────────────────────────────────────────────────────── #
# Colour palette (the warm cream brand)
# ─────────────────────────────────────────────────────────────────────────── #
PALETTE = {
    "bg":            "#fff1e5",   # warm cream background
    "surface":       "#ffffff",   # inputs / tables
    "ink":           "#33302e",   # primary text
    "muted":         "#6f655d",   # secondary text / hints
    "border":        "#d8cabb",   # tan hairline borders
    "tan":           "#e2d5cb",   # default button / accent
    "tan_hover":     "#d1c4ba",   # hover accent
    "tan_active":    "#b3a8a0",   # pressed / strong accent
    "selection":     "#f3e7da",   # row / nav selection wash
    "primary":       "#33302e",   # dominant action (ink)
    "primary_hover": "#4a4540",
    "primary_text":  "#fff1e5",
    "destructive":   "#9c4a3c",   # quiet brick red
    "green":         "#c4d1ba",   # subtle "apply / go" accent
}

# ─────────────────────────────────────────────────────────────────────────── #
# Spacing scale (px) — use these instead of ad-hoc setFixedWidth/margins
# ─────────────────────────────────────────────────────────────────────────── #
SPACE_XS, SPACE_SM, SPACE_MD, SPACE_LG, SPACE_XL = 4, 8, 12, 16, 24

# Sensible shared minimum widths so rows of controls line up in columns
FIELD_MIN_WIDTH = 160
NARROW_FIELD_MIN_WIDTH = 90

SERIF_FAMILY = "Georgia"


def preferred_sans() -> str:
    """Return Inter if installed, else the best available clean system sans."""
    families = set(QFontDatabase.families())
    for candidate in ("Inter", "Segoe UI", "Helvetica Neue", "Arial", "Roboto"):
        if candidate in families:
            return candidate
    return QFont().defaultFamily()


_QSS_PATH = Path(__file__).with_name("style.qss")


def load_stylesheet() -> str:
    """Read style.qss and substitute the @token@ palette/font placeholders."""
    qss = _QSS_PATH.read_text(encoding="utf-8")
    for key, value in PALETTE.items():
        qss = qss.replace(f"@{key}@", value)
    qss = qss.replace("@sans@", preferred_sans())
    qss = qss.replace("@serif@", SERIF_FAMILY)
    return qss


def apply_app_theme(app) -> None:
    """Install the sans font and shared stylesheet on the whole application."""
    app.setFont(QFont(preferred_sans(), 10))
    app.setStyleSheet(load_stylesheet())


def repolish(widget: QWidget) -> None:
    """Re-evaluate the stylesheet for a widget after a dynamic property change."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def make_splash() -> QSplashScreen:
    """A proper QSplashScreen rendered in the brand palette."""
    pix = QPixmap(520, 260)
    pix.fill(QColor(PALETTE["bg"]))

    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    painter.setPen(QColor(PALETTE["tan_active"]))
    painter.drawRoundedRect(1, 1, pix.width() - 3, pix.height() - 3, 12, 12)

    painter.setPen(QColor(PALETTE["ink"]))
    painter.setFont(QFont(SERIF_FAMILY, 30, QFont.Weight.Bold))
    painter.drawText(pix.rect().adjusted(0, -28, 0, -28),
                     Qt.AlignmentFlag.AlignCenter, "🍌 PyNSD 🍌")

    painter.setFont(QFont(preferred_sans(), 12))
    painter.setPen(QColor(PALETTE["muted"]))
    painter.drawText(pix.rect().adjusted(0, 60, 0, 60),
                     Qt.AlignmentFlag.AlignCenter, "The PNSD Toolkit")
    painter.end()

    splash = QSplashScreen(pix, Qt.WindowType.WindowStaysOnTopHint)
    splash.showMessage(
        "Loading toolkits…",
        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
        QColor(PALETTE["muted"]),
    )
    return splash

"""
gui/widgets.py
--------------
Small reusable, theme-aware building blocks shared across panels.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QFormLayout, QLayout, QSizePolicy, QToolButton,
                             QVBoxLayout, QWidget)

from gui.theme import SPACE_MD, SPACE_SM


class CollapsibleSection(QWidget):
    """A labelled section with a clickable header that expands/collapses.

    Usage::

        sec = CollapsibleSection("1 · Add files")
        sec.set_content_layout(my_layout)
    """

    def __init__(self, title: str, expanded: bool = True, parent=None):
        super().__init__(parent)

        self._toggle = QToolButton(text=title, checkable=True, checked=expanded)
        self._toggle.setObjectName("SectionHeader")
        self._toggle.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle.setSizePolicy(QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Fixed)            # full-width banner
        self._toggle.toggled.connect(self._on_toggled)

        self._content = QWidget()
        self._content.setVisible(expanded)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(SPACE_SM)
        root.addWidget(self._toggle)
        root.addWidget(self._content)

    def _on_toggled(self, checked: bool) -> None:
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        self._content.setVisible(checked)

    def set_content_layout(self, layout: QLayout) -> None:
        layout.setContentsMargins(SPACE_MD, SPACE_SM, SPACE_MD, SPACE_MD)
        self._content.setLayout(layout)

    @property
    def content(self) -> QWidget:
        return self._content


def make_form(label_width: int | None = None) -> QFormLayout:
    """A QFormLayout pre-configured for the shared look (aligned columns)."""
    form = QFormLayout()
    form.setHorizontalSpacing(SPACE_MD)
    form.setVerticalSpacing(SPACE_SM)
    form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    return form

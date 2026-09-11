"""
gui/widgets.py
--------------
Small reusable, theme-aware building blocks shared across panels.
"""
from __future__ import annotations

from PyQt6.QtCore import QElapsedTimer, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QFormLayout, QHBoxLayout, QLabel, QLayout,
                             QProgressBar, QPushButton, QSizePolicy,
                             QToolButton, QVBoxLayout, QWidget)

from gui.theme import SPACE_MD, SPACE_SM, repolish


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

    def set_expanded(self, expanded: bool) -> None:
        self._toggle.setChecked(expanded)


def validate_number(edit, minimum=None, maximum=None, integer=False, allow_blank=True):
    """Outline a text box in red while what is typed is not a usable number.

    Catching a stray letter or an out-of-range value as it is typed beats a
    dialog after the fact, or worse, a value that silently falls back to a
    default the user never chose.
    """
    def problem() -> str:
        text = edit.text().strip()
        if not text:
            return "" if allow_blank else "This cannot be empty."
        try:
            value = int(text) if integer else float(text)
        except ValueError:
            return "Whole number expected." if integer else "Number expected."
        if minimum is not None and value < minimum:
            return f"Must be at least {minimum:g}."
        if maximum is not None and value > maximum:
            return f"Must be at most {maximum:g}."
        return ""

    def check():
        message = problem()
        edit.setProperty("invalid", bool(message))
        edit.setToolTip(message or edit.property("_valid_tip") or "")
        repolish(edit)

    edit.setProperty("_valid_tip", edit.toolTip())
    edit.textChanged.connect(check)
    check()
    return edit


class RunProgress(QWidget):
    """Progress bar, elapsed time and a Cancel button for one background job.

    Hidden until a job starts.  Shows a moving bar while the total is unknown
    and switches to a real percentage as soon as the job reports one.
    """
    cancel_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._elapsed = QElapsedTimer()

        self.bar = QProgressBar()
        self.bar.setObjectName("RunBar")
        self.bar.setTextVisible(True)
        self.bar.setMinimumWidth(220)

        self.label = QLabel()
        self.label.setObjectName("FileStatus")

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setProperty("class", "destructive")
        self.btn_cancel.clicked.connect(self._on_cancel)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACE_SM)
        row.addWidget(self.bar)
        row.addWidget(self.label)
        row.addWidget(self.btn_cancel)
        row.addStretch()                               # keep the group together on the left

        self._tick = QTimer(self)
        self._tick.timeout.connect(self._refresh_elapsed)
        self.setVisible(False)

    def start(self, message: str = "Working…") -> None:
        self._message = message
        self.bar.setRange(0, 0)                        # indeterminate until told otherwise
        self.bar.setValue(0)
        self.label.setText(message)
        self.btn_cancel.setEnabled(True)
        self.btn_cancel.setText("Cancel")
        self._elapsed.start()
        self._tick.start(1000)
        self.setVisible(True)

    def update(self, done: int, total: int, message: str = "") -> None:
        if total > 0:
            self.bar.setRange(0, total)
            self.bar.setValue(done)
        if message:
            self._message = message
        self._refresh_elapsed()

    def stop(self) -> None:
        self._tick.stop()
        self.setVisible(False)

    def _on_cancel(self) -> None:
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setText("Stopping…")
        self.label.setText("Stopping after the current step…")
        self.cancel_requested.emit()

    def _refresh_elapsed(self) -> None:
        if not self.btn_cancel.isEnabled():
            return                                     # keep the "stopping" message
        seconds = self._elapsed.elapsed() // 1000
        stamp = f"{seconds // 60}m {seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"
        self.label.setText(f"{self._message}   ({stamp})")


def make_form(label_width: int | None = None) -> QFormLayout:
    """A QFormLayout pre-configured for the shared look (aligned columns)."""
    form = QFormLayout()
    form.setHorizontalSpacing(SPACE_MD)
    form.setVerticalSpacing(SPACE_SM)
    form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    return form

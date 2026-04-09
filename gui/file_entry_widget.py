from pathlib import Path
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, 
                             QGroupBox, QComboBox, QLineEdit, QGridLayout)
from utils.data_loader import DATE_COLUMN_OPTIONS, DATE_FORMAT_OPTIONS, DataFile

class FileEntryWidget(QFrame):
    removed = pyqtSignal(str)
    selected = pyqtSignal(str)
    reparse_requested = pyqtSignal()

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("FileEntryWidget { background-color: #fdfdfd; border: 1px solid #ddd; border-radius: 4px; }")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # --- Top Row (Summary) ---
        top_row = QHBoxLayout()
        self.lbl_name = QLabel(Path(self.path).name)
        self.lbl_name.setStyleSheet("font-weight: bold; font-size: 13px;")
        top_row.addWidget(self.lbl_name)
        
        self.lbl_status = QLabel("Parsing...")
        self.lbl_status.setStyleSheet("color: #777; font-size: 12px;")
        top_row.addWidget(self.lbl_status)
        top_row.addStretch()

        self.btn_override = QPushButton("⚙️ Overrides")
        self.btn_override.setCheckable(True)
        self.btn_override.clicked.connect(self._toggle_overrides)
        top_row.addWidget(self.btn_override)

        self.btn_reparse = QPushButton("↻ Reparse")
        self.btn_reparse.clicked.connect(self.reparse_requested.emit)
        top_row.addWidget(self.btn_reparse)

        self.btn_remove = QPushButton("❌")
        self.btn_remove.setFixedWidth(30)
        self.btn_remove.clicked.connect(lambda: self.removed.emit(self.path))
        top_row.addWidget(self.btn_remove)
        layout.addLayout(top_row)

        # --- Override Panel (Hidden by default) ---
        self.override_panel = QGroupBox("Override Global Settings")
        self.override_panel.setCheckable(True)
        self.override_panel.setChecked(False)
        self.override_panel.setVisible(False)
        
        grid = QGridLayout(self.override_panel)
        
        # Date Col & Format
        grid.addWidget(QLabel("Date Col:"), 0, 0)
        self.col_combo = QComboBox()
        self.col_combo.addItems(DATE_COLUMN_OPTIONS)
        grid.addWidget(self.col_combo, 0, 1)
        self.custom_col = QLineEdit()
        self.custom_col.setPlaceholderText("Custom col...")
        self.custom_col.setVisible(False)
        self.col_combo.currentTextChanged.connect(lambda t: self.custom_col.setVisible(t=="Custom..."))
        grid.addWidget(self.custom_col, 0, 2)

        grid.addWidget(QLabel("Format:"), 1, 0)
        self.fmt_combo = QComboBox()
        for disp, _ in DATE_FORMAT_OPTIONS: self.fmt_combo.addItem(disp)
        grid.addWidget(self.fmt_combo, 1, 1)
        self.custom_fmt = QLineEdit()
        self.custom_fmt.setPlaceholderText("Custom fmt...")
        self.custom_fmt.setVisible(False)
        self.fmt_combo.currentIndexChanged.connect(lambda i: self.custom_fmt.setVisible(DATE_FORMAT_OPTIONS[i][1]=="custom"))
        grid.addWidget(self.custom_fmt, 1, 2)

        # Timezone & NAs
        grid.addWidget(QLabel("Timezone:"), 2, 0)
        self.tz_input = QLineEdit("UTC")
        grid.addWidget(self.tz_input, 2, 1)

        grid.addWidget(QLabel("NAs:"), 2, 2)
        self.na_combo = QComboBox()
        self.na_combo.addItems(["Drop Rows", "Fill (Fwd/Bwd)", "Interpolate", "Fill Min (1e-4)"])
        grid.addWidget(self.na_combo, 2, 3)

        # Avg & Drop Cols
        grid.addWidget(QLabel("Avg:"), 3, 0)
        avg_layout = QHBoxLayout()
        self.resample_val = QLineEdit()
        self.resample_val.setFixedWidth(40)
        avg_layout.addWidget(self.resample_val)
        self.resample_unit = QComboBox()
        self.resample_unit.addItems(["Minutes", "Hours", "Days"])
        avg_layout.addWidget(self.resample_unit)
        grid.addLayout(avg_layout, 3, 1)

        grid.addWidget(QLabel("Drop Cols:"), 3, 2)
        self.drop_cols = QLineEdit()
        grid.addWidget(self.drop_cols, 3, 3)

        layout.addWidget(self.override_panel)

    def _toggle_overrides(self):
        self.override_panel.setVisible(self.btn_override.isChecked())

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.selected.emit(self.path)
        super().mousePressEvent(event)

    def set_selected_style(self, selected: bool):
        if selected:
            self.setStyleSheet("FileEntryWidget { background-color: #e6f2ff; border: 2px solid #0078d7; border-radius: 4px; }")
        else:
            self.setStyleSheet("FileEntryWidget { background-color: #fdfdfd; border: 1px solid #ddd; border-radius: 4px; }")

    def set_result(self, result: DataFile):
        if result.ok:
            color = "green"
            status_text = f"✓ {result.size_str} | {result.n_rows} rows | {result.n_bins} bins"
        else:
            color = "red"
            status_text = "❌ Error"
        
        self.lbl_status.setText(status_text)
        self.lbl_status.setStyleSheet(f"color: {color}; font-size: 12px;")

    # --- Effective Value Methods ---
    def overrides_active(self) -> bool:
        return self.override_panel.isChecked()

    def effective_col(self, global_val: str) -> str:
        if not self.overrides_active(): return global_val
        txt = self.col_combo.currentText()
        return self.custom_col.text().strip() if txt == "Custom..." else txt

    def effective_fmt(self, global_val: str) -> str:
        if not self.overrides_active(): return global_val
        val = DATE_FORMAT_OPTIONS[self.fmt_combo.currentIndex()][1]
        return self.custom_fmt.text().strip() if val == "custom" else val

    def effective_tz(self, global_val: str) -> str:
        return self.tz_input.text().strip() if self.overrides_active() else global_val

    def effective_na(self, global_val: str) -> str:
        return self.na_combo.currentText() if self.overrides_active() else global_val

    def effective_drop(self, global_val: str) -> str:
        return self.drop_cols.text().strip() if self.overrides_active() else global_val

    def effective_resample(self, global_val: str, global_unit: str) -> tuple[str, str]:
        if not self.overrides_active(): return global_val, global_unit
        return self.resample_val.text().strip(), self.resample_unit.currentText()
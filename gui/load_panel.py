from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
import re
from PyQt6.QtCore import Qt, pyqtSignal, QDate, QPoint
from PyQt6.QtGui import QFont, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QFileDialog, QFrame, QGroupBox, QHBoxLayout, 
                             QHeaderView, QLabel, QLineEdit, QProgressBar, QPushButton, QScrollArea, QSplitter, 
                             QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QDialog, QMessageBox, QInputDialog,
                             QDateEdit, QSpinBox, QCheckBox, QToolTip)
from gui.file_entry_widget import FileEntryWidget
from gui.widgets import CollapsibleSection, make_form
from gui.theme import (FIELD_MIN_WIDTH, NARROW_FIELD_MIN_WIDTH,
                       SPACE_SM, SPACE_MD, SPACE_LG)
from utils.data_loader import (DATE_COLUMN_OPTIONS, DATE_FORMAT_OPTIONS, DEFAULT_DATE_COL, DEFAULT_DATE_FMT,
                               DataFile, load_pnsd_file, apply_qc_filter, calculate_line_losses, align_bins)

# ─────────────────────────────────────────────────────────────────────────── #
# Hoverable preview icon label
# ─────────────────────────────────────────────────────────────────────────── #
class _PreviewIconLabel(QLabel):
    """QLabel that shows a rich tooltip reliably via QToolTip.showText on hover."""
    def __init__(self, tooltip_html: str, parent=None):
        super().__init__("📄", parent)
        self._tip = tooltip_html

    def enterEvent(self, event):
        QToolTip.showText(self.mapToGlobal(QPoint(self.width() + 4, 0)), self._tip, self)
        super().enterEvent(event)

    def leaveEvent(self, event):
        QToolTip.hideText()
        super().leaveEvent(event)


# ─────────────────────────────────────────────────────────────────────────── #
# Spline Harmonisation Diagnostic Dialog
# ─────────────────────────────────────────────────────────────────────────── #
class HarmoniseDialog(QDialog):
    def __init__(self, valid_res_dict, target_keys, parent=None):
        super().__init__(parent)
        self.valid_res_dict = valid_res_dict
        self.target_keys = target_keys
        self.setWindowTitle("Harmonise Diameters (Cubic Spline Check)")
        self.resize(900, 600)
        
        layout = QVBoxLayout(self)
        
        # --- Settings Row ---
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Target Dataset (Reference Diameters):"))
        self.combo = QComboBox()
        self.combo.addItems([Path(p).name for p in self.target_keys])
        self.combo.currentIndexChanged.connect(self.update_plot)
        ctrl.addWidget(self.combo)
        
        ctrl.addSpacing(20)
        ctrl.addWidget(QLabel("Out-of-range boundaries:"))
        self.oor_combo = QComboBox()
        self.oor_combo.addItems(["Fill with 1.0 (Safe for Log plots)", "Drop Out-of-Bounds Columns"])
        ctrl.addWidget(self.oor_combo)
        layout.addLayout(ctrl)
        
        # --- Plotting Canvas ---
        self.fig = Figure(figsize=(8, 5))
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas)
        
        # --- Execution Buttons ---
        btn_box = QHBoxLayout()
        self.btn_apply = QPushButton("✨ Apply Harmonisation")
        self.btn_apply.setProperty("class", "primary")
        self.btn_apply.clicked.connect(self.accept)
        btn_box.addStretch()
        btn_box.addWidget(self.btn_apply)
        layout.addLayout(btn_box)
        
        self.update_plot()
        
    def update_plot(self):
        """Plots the raw mean PNSD against the spline to visually validate the maths."""
        target_path = self.target_keys[self.combo.currentIndex()]
        target_data = self.valid_res_dict[target_path]
        t_diams = np.array(target_data.diameters)
        
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Diameter (nm)")
        ax.set_ylabel("Mean dN/dlogDp")
        ax.set_title("Spline Interpolation Check (Mean PNSD)")
        
        for path in self.target_keys:
            data = self.valid_res_dict[path]
            o_diams = np.array(data.diameters)
            
            # Strip NaNs for the diagnostic mean plot
            mean_raw = data.df.mean(skipna=True).values
            p = ax.plot(o_diams, mean_raw, 'o', alpha=0.6, label=f"{Path(path).name} (Raw)")
            
            if path == target_path or np.array_equal(o_diams, t_diams):
                ax.plot(o_diams, mean_raw, '-', color=p[0].get_color(), lw=2, alpha=0.5, label=f"{Path(path).name} (Target Spline)")
                continue
            
            # Clean the raw mean for a stable 1D spline plot
            valid_mask = ~np.isnan(mean_raw)
            if not valid_mask.any(): continue
            clean_o_diams = o_diams[valid_mask]
            clean_mean = mean_raw[valid_mask]
            
            try:
                f = interp1d(clean_o_diams, clean_mean, kind='cubic', bounds_error=False, fill_value=np.nan)
                mean_new = f(t_diams)
                # Plot the new spline curve over the target bin range
                ax.plot(t_diams, mean_new, '-', color=p[0].get_color(), lw=2, label=f"{Path(path).name} (Spline)")
            except Exception as e:
                print(f"Plot spline failed for {path}: {e}")
                
        ax.legend(fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw()
        
    def get_target_path(self): return self.target_keys[self.combo.currentIndex()]
    def get_oor_action(self): return "fill" if self.oor_combo.currentIndex() == 0 else "drop"


# ─────────────────────────────────────────────────────────────────────────── #
# DateTime Filter Dialog
# ─────────────────────────────────────────────────────────────────────────── #
class DateTimeFilterDialog(QDialog):
    def __init__(self, df: pd.DataFrame, diams: list, parent=None):
        super().__init__(parent)
        self.df = df.copy()
        self.diams = diams
        self.mask = pd.Series(True, index=df.index)
        
        self.setWindowTitle("Filter by Date/Time")
        self.resize(900, 650)
        
        layout = QVBoxLayout(self)
        
        # --- Filter Mode Selection ---
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Filter Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Custom Date Range", "By Day of Week", "By Month", "By Hour of Day", "By Year"])
        self.mode_combo.currentIndexChanged.connect(self.update_controls)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)
        
        # --- Dynamic Controls Container ---
        self.controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_widget)
        layout.addWidget(self.controls_widget)
        
        # --- Visualization Canvas ---
        self.fig = Figure(figsize=(8, 3))
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas)
        
        # --- Stats Label ---
        self.stats_label = QLabel()
        self.stats_label.setObjectName("StatsLabel")
        layout.addWidget(self.stats_label)

        # --- Buttons ---
        btn_box = QHBoxLayout()
        self.btn_apply = QPushButton("✨ Apply Filter")
        self.btn_apply.setProperty("class", "primary")
        self.btn_apply.clicked.connect(self.accept)
        btn_box.addStretch()
        btn_box.addWidget(self.btn_apply)
        layout.addLayout(btn_box)
        
        self.update_controls()
        
    def update_controls(self):
        """Rebuild control panel based on selected filter mode."""
        # Clear existing controls — items may be widgets OR sub-layouts
        def _clear_layout(layout):
            while layout.count():
                item = layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.deleteLater()
                elif item.layout() is not None:
                    _clear_layout(item.layout())
        _clear_layout(self.controls_layout)
        
        mode = self.mode_combo.currentText()
        
        if mode == "Custom Date Range":
            self._build_date_range_controls()
        elif mode == "By Day of Week":
            self._build_dow_controls()
        elif mode == "By Month":
            self._build_month_controls()
        elif mode == "By Hour of Day":
            self._build_hour_controls()
        elif mode == "By Year":
            self._build_year_controls()
        
        self.update_filter()
    
    def _build_date_range_controls(self):
        row = QHBoxLayout()
        row.addWidget(QLabel("Start Date:"))
        self.date_start = QDateEdit()
        self.date_start.setDate(QDate(self.df.index.min().year, self.df.index.min().month, self.df.index.min().day))
        self.date_start.dateChanged.connect(self.update_filter)
        row.addWidget(self.date_start)
        
        row.addWidget(QLabel("End Date:"))
        self.date_end = QDateEdit()
        self.date_end.setDate(QDate(self.df.index.max().year, self.df.index.max().month, self.df.index.max().day))
        self.date_end.dateChanged.connect(self.update_filter)
        row.addWidget(self.date_end)
        
        row.addWidget(QLabel("Action:"))
        self.action_combo = QComboBox()
        self.action_combo.addItems(["Keep Inside Range", "Remove Inside Range"])
        self.action_combo.currentIndexChanged.connect(self.update_filter)
        row.addWidget(self.action_combo)
        row.addStretch()
        
        self.controls_layout.addLayout(row)
    
    def _build_dow_controls(self):
        row = QHBoxLayout()
        row.addWidget(QLabel("Keep Days:"))
        self.dow_checks = {}
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for i, day in enumerate(days):
            cb = QCheckBox(day)
            cb.setChecked(True)
            cb.stateChanged.connect(self.update_filter)
            self.dow_checks[i] = cb
            row.addWidget(cb)
        row.addStretch()
        self.controls_layout.addLayout(row)
    
    def _build_month_controls(self):
        row = QHBoxLayout()
        row.addWidget(QLabel("Keep Months:"))
        self.month_checks = {}
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        for i, month in enumerate(months):
            cb = QCheckBox(month)
            cb.setChecked(True)
            cb.stateChanged.connect(self.update_filter)
            self.month_checks[i+1] = cb
            row.addWidget(cb)
        row.addStretch()
        self.controls_layout.addLayout(row)
    
    def _build_hour_controls(self):
        row = QHBoxLayout()
        row.addWidget(QLabel("Hour Range:"))
        row.addWidget(QLabel("From:"))
        self.hour_start = QSpinBox()
        self.hour_start.setRange(0, 23)
        self.hour_start.setValue(0)
        self.hour_start.valueChanged.connect(self.update_filter)
        row.addWidget(self.hour_start)
        
        row.addWidget(QLabel("To:"))
        self.hour_end = QSpinBox()
        self.hour_end.setRange(0, 23)
        self.hour_end.setValue(23)
        self.hour_end.valueChanged.connect(self.update_filter)
        row.addWidget(self.hour_end)
        row.addStretch()
        self.controls_layout.addLayout(row)
    
    def _build_year_controls(self):
        row = QHBoxLayout()
        row.addWidget(QLabel("Keep Years:"))
        self.year_checks = {}
        years = sorted(self.df.index.year.unique())
        for year in years:
            cb = QCheckBox(str(year))
            cb.setChecked(True)
            cb.stateChanged.connect(self.update_filter)
            self.year_checks[year] = cb
            row.addWidget(cb)
        row.addStretch()
        self.controls_layout.addLayout(row)
    
    def update_filter(self):
        """Update the filter mask based on current control state."""
        mode = self.mode_combo.currentText()
        self.mask = pd.Series(True, index=self.df.index)
        
        if mode == "Custom Date Range":
            start = self.date_start.date().toPyDate()
            end = self.date_end.date().toPyDate()
            in_range = (self.df.index.date >= start) & (self.df.index.date <= end)
            
            if self.action_combo.currentText() == "Keep Inside Range":
                self.mask = in_range
            else:
                self.mask = ~in_range
        
        elif mode == "By Day of Week":
            selected_days = [day for day, cb in self.dow_checks.items() if cb.isChecked()]
            self.mask = self.df.index.dayofweek.isin(selected_days)
        
        elif mode == "By Month":
            selected_months = [month for month, cb in self.month_checks.items() if cb.isChecked()]
            self.mask = self.df.index.month.isin(selected_months)
        
        elif mode == "By Hour of Day":
            start_hr = self.hour_start.value()
            end_hr = self.hour_end.value()
            if start_hr <= end_hr:
                self.mask = (self.df.index.hour >= start_hr) & (self.df.index.hour <= end_hr)
            else:
                self.mask = (self.df.index.hour >= start_hr) | (self.df.index.hour <= end_hr)
        
        elif mode == "By Year":
            selected_years = [year for year, cb in self.year_checks.items() if cb.isChecked()]
            self.mask = self.df.index.year.isin(selected_years)
        
        self.update_visualization()
    
    def update_visualization(self):
        """Update the plot showing what will be kept/removed."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Create a time series showing kept vs removed
        kept_series = self.mask.astype(int)
        dates = mdates.date2num(self.df.index)
        
        colors = np.where(np.asarray(self.mask, dtype=bool), 'green', 'red')
        ax.scatter(dates, kept_series, c=colors, s=20, alpha=0.6)
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Remove", "Keep"])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.figure.autofmt_xdate(rotation=45)
        ax.set_ylabel("Filter Result")
        ax.set_title("Date/Time Filter Preview")
        
        # Update stats
        n_keep = self.mask.sum()
        n_total = len(self.mask)
        pct = 100 * n_keep / n_total if n_total > 0 else 0
        self.stats_label.setText(f"Keeping {n_keep}/{n_total} rows ({pct:.1f}%)")
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def get_filtered_data(self):
        """Return the filtered DataFrame."""
        return self.df[self.mask]
    
    def get_mask(self):
        """Return the filter mask."""
        return self.mask


# ─────────────────────────────────────────────────────────────────────────── #
# Diameter Filter Dialog
# ─────────────────────────────────────────────────────────────────────────── #
class DiameterFilterDialog(QDialog):
    def __init__(self, df: pd.DataFrame, diams: list, parent=None):
        super().__init__(parent)
        self.df = df
        self.diams = np.array(diams, dtype=float)

        self.setWindowTitle("Filter Diameters")
        self.resize(900, 620)

        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Min Dp (nm):"))
        self.min_input = QLineEdit(f"{self.diams.min():.2f}")
        self.min_input.setFixedWidth(90)
        self.min_input.textChanged.connect(self.update_plot)
        ctrl.addWidget(self.min_input)

        ctrl.addWidget(QLabel("Max Dp (nm):"))
        self.max_input = QLineEdit(f"{self.diams.max():.2f}")
        self.max_input.setFixedWidth(90)
        self.max_input.textChanged.connect(self.update_plot)
        ctrl.addWidget(self.max_input)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.fig = Figure(figsize=(8, 4))
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas)

        self.stats_lbl = QLabel()
        self.stats_lbl.setObjectName("StatsLabel")
        layout.addWidget(self.stats_lbl)

        btns = QHBoxLayout()
        self.btn_apply = QPushButton("✨ Apply Diameter Filter")
        self.btn_apply.setProperty("class", "primary")
        self.btn_apply.clicked.connect(self.accept)
        btns.addStretch()
        btns.addWidget(self.btn_apply)
        layout.addLayout(btns)

        self._mask = np.ones_like(self.diams, dtype=bool)
        self.update_plot()

    def _read_bounds(self):
        try:
            dmin = float(self.min_input.text())
            dmax = float(self.max_input.text())
        except ValueError:
            return None, None
        if dmax < dmin:
            dmin, dmax = dmax, dmin
        return dmin, dmax

    def update_plot(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Diameter (nm)")
        ax.set_ylabel("Mean dN/dlogDp")
        ax.set_title("Diameter Filter Preview")

        mean_all = np.clip(self.df.mean(skipna=True).to_numpy(dtype=float), 1e-6, None)
        ax.plot(self.diams, mean_all, color='gray', alpha=0.45, lw=1.5, label='All diameters')

        dmin, dmax = self._read_bounds()
        if dmin is None:
            self.stats_lbl.setText("Enter valid numeric min/max diameters.")
            self.canvas.draw()
            return

        self._mask = (self.diams >= dmin) & (self.diams <= dmax)

        if dmin > self.diams.min():
            ax.axvspan(self.diams.min(), dmin, color='red', alpha=0.12)
        if dmax < self.diams.max():
            ax.axvspan(dmax, self.diams.max(), color='red', alpha=0.12)

        if self._mask.any():
            kept = np.where(self._mask, mean_all, np.nan)
            ax.plot(self.diams, kept, color='blue', lw=2.0, label='Kept range')

        cut_mask = ~self._mask
        if cut_mask.any():
            ax.scatter(self.diams[cut_mask], mean_all[cut_mask], color='red', s=12, alpha=0.7, label='Cut bins')

        kept_n = int(self._mask.sum())
        total_n = int(len(self._mask))
        self.stats_lbl.setText(f"Keeping {kept_n}/{total_n} diameter bins ({100*kept_n/max(total_n,1):.1f}%).")

        ax.legend(fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw()

    def get_kept_mask(self):
        return self._mask


# ─────────────────────────────────────────────────────────────────────────── #
# Load Panel Main Class
# ─────────────────────────────────────────────────────────────────────────── #


def detect_timezone_from_csv(file_path: str) -> str | None:
    """
    Auto-detect timezone offset embedded in CSV timestamps.
    Looks for patterns like: 2022-08-01 12:30:45+02:00 or 2022-08-01T12:30:45-0500
    Returns a UTC offset string pandas understands (e.g. 'UTC+02:00') or None.
    """
    # Match a tz offset that immediately follows a time component (digit or second digit group)
    # e.g. "...45+02:00"  "...30-05:00"  "...00+0000"
    tz_in_datetime = re.compile(r'\d{2}([+-])(\d{2}):?(\d{2})$')
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                for token in line.split(','):
                    token = token.strip().strip('"').strip("'")
                    m = tz_in_datetime.search(token)
                    if m:
                        sign, hh, mm = m.group(1), m.group(2), m.group(3)
                        h, mn = int(hh), int(mm)
                        if 0 <= h <= 14 and 0 <= mn < 60:
                            return f"UTC{sign}{h:02d}:{mn:02d}"
    except Exception:
        pass
    return None



class LoadPanel(QWidget):
    data_confirmed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries: dict[str, FileEntryWidget] = {}
        self._results: dict[str, DataFile] = {}
        self._selected_paths: set[str] = set()
        self._active_preview_path: str | None = None
        self._merged_df = None
        self._merged_diams = None
        
        self._build_ui()
        self.setAcceptDrops(True)

    def _info_btn(self, text, title="Information"):
        btn = QPushButton("ℹ️")
        btn.setObjectName("InfoButton")
        btn.setFixedSize(24, 24)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(lambda: QMessageBox.information(self, title, text))
        return btn

    def _field_row(self, *widgets):
        """Pack widgets (fields + info buttons) into a left-aligned row for forms."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACE_SM)
        for w in widgets:
            row.addWidget(w)
        row.addStretch()
        container = QWidget()
        container.setLayout(row)
        return container

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(SPACE_MD)
        root.setContentsMargins(SPACE_LG, SPACE_LG, SPACE_LG, SPACE_MD)

        # ── Header strip: title + how-to + "Editing: <file>" ──────────── #
        header = QHBoxLayout()
        title = QLabel("Load & prepare data")
        title.setObjectName("H2")
        intro_text = (
            "Data Pipeline:\n"
            "1. Add files and parse datetimes/timezones.\n"
            "2. Apply QC to strip spikes, or correct for line-losses.\n"
            "3. Ctrl-click multiple files in the list, then use the Combine buttons.\n"
            "4. Export any dataset, or proceed with the selected one."
        )
        header.addWidget(title)
        header.addWidget(self._info_btn(intro_text, "How to use this tool"))
        header.addStretch()
        self._editing_lbl = QLabel("Editing: —")
        self._editing_lbl.setObjectName("EditingStrip")
        header.addWidget(self._editing_lbl)
        root.addLayout(header)

        # ── Accordion of workflow steps inside a scroll area ──────────── #
        # Order: set import settings first, then corrections, then combine.
        scroll_outer = QScrollArea()
        scroll_outer.setWidgetResizable(True)
        inner = QWidget()
        sections = QVBoxLayout(inner)
        sections.setContentsMargins(0, 0, 0, 0)
        sections.setSpacing(SPACE_MD)
        sections.addWidget(self._build_import_section())
        sections.addWidget(self._build_corrections_section())
        sections.addWidget(self._build_combine_section())
        sections.addStretch()
        scroll_outer.setWidget(inner)
        root.addWidget(scroll_outer, stretch=5)

        # ── Files + preview: pinned to the bottom, always visible ─────── #
        root.addWidget(self._build_files_area(), stretch=2)

        # ── Sticky footer: Proceed is always visible bottom-right ─────── #
        root.addWidget(self._build_footer())

    # ── Files area (add/clear + list + preview), always visible ───────── #
    def _build_files_area(self) -> QWidget:
        container = QWidget()
        container.setObjectName("FilesPane")
        container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        body = QVBoxLayout(container)
        body.setContentsMargins(SPACE_MD, SPACE_SM, SPACE_MD, SPACE_SM)
        body.setSpacing(SPACE_SM)

        files_title = QLabel("Files")
        files_title.setObjectName("H2")
        body.addWidget(files_title)

        file_row = QHBoxLayout()
        add_btn = QPushButton("+ Add files…")
        add_btn.setProperty("class", "secondary")
        add_btn.clicked.connect(self._browse_files)
        file_row.addWidget(add_btn)
        clear_btn = QPushButton("Clear all")
        clear_btn.setProperty("class", "destructive")
        clear_btn.clicked.connect(self._clear_all)
        file_row.addWidget(clear_btn)
        file_row.addStretch()
        body.addLayout(file_row)

        list_splitter = QSplitter(Qt.Orientation.Vertical)

        self._file_list_inner = QWidget()
        self._file_list_layout = QVBoxLayout(self._file_list_inner)
        self._file_list_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._file_list_inner)
        scroll.setMinimumHeight(110)
        list_splitter.addWidget(scroll)

        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self._preview_hint = QLabel("Preview")
        self._preview_hint.setObjectName("Hint")
        preview_layout.addWidget(self._preview_hint)
        self._preview_table = QTableWidget()
        self._preview_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._preview_table.setMinimumHeight(130)
        preview_layout.addWidget(self._preview_table)
        list_splitter.addWidget(preview_container)

        list_splitter.setSizes([320, 260])
        list_splitter.setStretchFactor(0, 1)
        list_splitter.setStretchFactor(1, 1)
        body.addWidget(list_splitter, stretch=1)

        return container

    # ── Section 1 · Import settings ──────────────────────────────────── #
    def _build_import_section(self) -> CollapsibleSection:
        sec = CollapsibleSection("1 · Import settings")
        body = QVBoxLayout()
        form = make_form()

        self._col_combo = QComboBox()
        self._col_combo.addItems(DATE_COLUMN_OPTIONS)
        self._col_combo.setMinimumWidth(FIELD_MIN_WIDTH)
        self._col_combo.currentTextChanged.connect(self._on_col_changed)
        self._custom_col = QLineEdit()
        self._custom_col.setPlaceholderText("Enter custom column name…")
        self._custom_col.setVisible(False)
        form.addRow("Date column", self._field_row(
            self._col_combo, self._custom_col,
            self._info_btn("The exact name of the column in your CSV containing the timestamp.", "Date Column")))

        self._fmt_combo = QComboBox()
        self._fmt_combo.setMinimumWidth(FIELD_MIN_WIDTH)
        sorted_fmts = sorted(DATE_FORMAT_OPTIONS, key=lambda x: 0 if str(x[0]).upper().startswith('Y') else (1 if str(x[0]).upper().startswith('D') else 2))
        for disp, _ in sorted_fmts: self._fmt_combo.addItem(disp)
        self._fmt_combo.currentIndexChanged.connect(self._on_fmt_changed)
        self._custom_fmt = QLineEdit()
        self._custom_fmt.setPlaceholderText("e.g., yyyy/MM/dd HH:mm:ss")
        self._custom_fmt.setVisible(False)
        form.addRow("Date format", self._field_row(
            self._fmt_combo, self._custom_fmt,
            self._info_btn("The structural format of your dates. Use 'Custom...' if none match perfectly.", "Date Format")))

        self._tz_input = QLineEdit("UTC")
        self._tz_input.setMinimumWidth(FIELD_MIN_WIDTH)
        form.addRow("Timezone", self._field_row(
            self._tz_input,
            self._info_btn("Standardises your data to a specific timezone (e.g., 'UTC', 'Europe/London'). Protects against DST gaps.", "Timezone")))

        self._resample_val = QLineEdit()
        self._resample_val.setPlaceholderText("Value")
        self._resample_val.setMinimumWidth(NARROW_FIELD_MIN_WIDTH)
        self._resample_unit = QComboBox()
        self._resample_unit.addItems(["Minutes", "Hours", "Days"])
        form.addRow("Average to timebase", self._field_row(
            self._resample_val, self._resample_unit,
            self._info_btn("Optional: Average your data to a new timebase (e.g., enter '15' and select 'Minutes' to downsample).", "Timebase Averaging")))

        self._na_combo = QComboBox()
        self._na_combo.setMinimumWidth(FIELD_MIN_WIDTH)
        self._na_combo.addItems(["Drop Rows", "Fill (Fwd/Bwd)", "Interpolate", "Fill Min (1e0)"])
        form.addRow("Missing data", self._field_row(
            self._na_combo,
            self._info_btn("How to handle missing data.\nDrop: Removes the row.\nFill: Copies last valid value.\nInterpolate: Draws a line.\nFill Min: Replaces with 1e0.", "Missing Data Handling")))

        self._drop_cols = QLineEdit()
        self._drop_cols.setPlaceholderText("Comma separated…")
        self._drop_cols.setMinimumWidth(FIELD_MIN_WIDTH)
        form.addRow("Drop columns", self._field_row(
            self._drop_cols,
            self._info_btn("Comma-separated list of columns to ignore completely (e.g., 'Status, Error Code').", "Drop Columns")))

        self._flag_col = QLineEdit()
        self._flag_col.setPlaceholderText("e.g., flag")
        self._flag_col.setMinimumWidth(FIELD_MIN_WIDTH)
        self._flag_val = QLineEdit("1")
        self._flag_val.setMaximumWidth(NARROW_FIELD_MIN_WIDTH)
        form.addRow("Error flag column", self._field_row(
            self._flag_col, QLabel("Error value"), self._flag_val,
            self._info_btn(
                "Optional: name of a 'flag' column in your CSV.\n"
                "Rows where that column equals the Error Value will be deleted before any other processing.\n"
                "Leave blank to skip. Default error value is 1 (e.g. 0 = clean, 1 = flagged).",
                "Error Flag Filtering")))

        body.addLayout(form)

        apply_btn = QPushButton("Apply to all loaded files")
        apply_btn.setProperty("class", "secondary")
        apply_btn.clicked.connect(self._apply_global_to_all)
        apply_row = QHBoxLayout()
        apply_row.addStretch()
        apply_row.addWidget(apply_btn)
        body.addLayout(apply_row)

        sec.set_content_layout(body)
        return sec

    # ── Section 2 · Corrections & QC ─────────────────────────────────── #
    def _build_corrections_section(self) -> CollapsibleSection:
        sec = CollapsibleSection("2 · Corrections & QC", expanded=False)
        body = QVBoxLayout()
        body.setSpacing(SPACE_MD)

        # --- QC spike removal ---
        qc_box = QGroupBox("QC spike removal")
        qc_form = make_form()
        self._qc_win = QLineEdit("20")
        self._qc_win.setMaximumWidth(NARROW_FIELD_MIN_WIDTH)
        qc_form.addRow("QC window (points)", self._field_row(
            self._qc_win,
            self._info_btn("Size of the rolling window (number of data points) used to calculate the moving median baseline.", "QC Window")))
        self._qc_thresh = QLineEdit("3.0")
        self._qc_thresh.setMaximumWidth(NARROW_FIELD_MIN_WIDTH)
        qc_form.addRow("Threshold (std dev)", self._field_row(
            self._qc_thresh,
            self._info_btn("How many standard deviations away from the baseline a point must be to be flagged as a spike.", "QC Threshold")))
        self._qc_action = QComboBox()
        self._qc_action.setMinimumWidth(FIELD_MIN_WIDTH)
        self._qc_action.addItems(["Replace with NA", "Replace with Mean"])
        qc_form.addRow("Action on spikes", self._field_row(
            self._qc_action,
            self._info_btn("What to do with identified spikes or negative values.", "QC Action")))
        qc_box_layout = QVBoxLayout(qc_box)
        qc_box_layout.addLayout(qc_form)
        qc_btn = QPushButton("Run QC")
        qc_btn.clicked.connect(self._run_qc)
        qc_btn_row = QHBoxLayout(); qc_btn_row.addStretch(); qc_btn_row.addWidget(qc_btn)
        qc_box_layout.addLayout(qc_btn_row)
        body.addWidget(qc_box)

        # --- Floor & ceiling ---
        thresh_box = QGroupBox("Floor & ceiling")
        thresh_layout = QHBoxLayout(thresh_box)
        thresh_layout.addWidget(QLabel("Minimum floor"))
        self._floor_thresh = QLineEdit("1.0")
        self._floor_thresh.setMaximumWidth(NARROW_FIELD_MIN_WIDTH)
        thresh_layout.addWidget(self._floor_thresh)
        btn_floor = QPushButton("Apply floor")
        btn_floor.clicked.connect(self._run_floor_threshold)
        thresh_layout.addWidget(btn_floor)
        floor_info = ("MPSS instruments cannot reliably measure < 1 particle/cm3.\n"
                      "Values below this are often non-physical. Low values are sometimes also used as error flags.\n"
                      "This tool replaces all values below your threshold with the threshold.")
        thresh_layout.addWidget(self._info_btn(floor_info, "Minimum Floor Threshold"))
        thresh_layout.addSpacing(SPACE_LG)
        thresh_layout.addWidget(QLabel("Maximum ceiling"))
        self._ceil_thresh = QLineEdit("1e6")
        self._ceil_thresh.setMaximumWidth(NARROW_FIELD_MIN_WIDTH)
        thresh_layout.addWidget(self._ceil_thresh)
        btn_ceil = QPushButton("Apply ceiling")
        btn_ceil.clicked.connect(self._run_ceil_threshold)
        thresh_layout.addWidget(btn_ceil)
        thresh_layout.addStretch()
        body.addWidget(thresh_box)

        # --- Line loss (Gormley-Kennedy) ---
        ll_box = QGroupBox("Line loss (Gormley-Kennedy)")
        ll_outer = QVBoxLayout(ll_box)
        ll_form = make_form()
        self._ll_len = QLineEdit("2.0")
        self._ll_len.setMaximumWidth(NARROW_FIELD_MIN_WIDTH)
        ll_form.addRow("Tube length (m)", self._ll_len)
        self._ll_id = QLineEdit("0.006")
        self._ll_id.setMaximumWidth(NARROW_FIELD_MIN_WIDTH)
        ll_form.addRow("Inner diameter (m)", self._ll_id)
        self._ll_temp = QLineEdit("293")
        self._ll_temp.setMaximumWidth(NARROW_FIELD_MIN_WIDTH)
        ll_form.addRow("Temperature (K)", self._ll_temp)
        self._ll_flow = QLineEdit("1.0")
        self._ll_flow.setMaximumWidth(NARROW_FIELD_MIN_WIDTH)
        ll_form.addRow("Flow (LPM)", self._ll_flow)
        ll_outer.addLayout(ll_form)
        ll_btn = QPushButton("Line loss correct")
        ll_btn.clicked.connect(self._run_line_loss)
        ll_btn_row = QHBoxLayout(); ll_btn_row.addStretch()
        ll_btn_row.addWidget(ll_btn)
        ll_btn_row.addWidget(self._info_btn("Applies Gormley-Kennedy diffusional loss corrections to the active dataset.", "Line Loss Correction"))
        ll_outer.addLayout(ll_btn_row)
        body.addWidget(ll_box)

        # --- Normalisation ---
        norm_box = QGroupBox("Normalisation")
        norm_layout = QHBoxLayout(norm_box)
        norm_layout.addWidget(QLabel("dlogDp bin width"))
        self._norm_dlogdp = QLineEdit("1.0")
        self._norm_dlogdp.setMaximumWidth(NARROW_FIELD_MIN_WIDTH)
        norm_layout.addWidget(self._norm_dlogdp)
        norm_layout.addWidget(self._info_btn("The calculated logarithmic width of your diameter bins. Required to normalise between dN and dN/dlogDp. n.b., the subsequent calculations will anticipate that your values are in dNdlogdp", "dlogDp Normalisation"))
        btn_norm = QPushButton("Normalise")
        btn_norm.clicked.connect(self._run_normalise)
        norm_layout.addWidget(btn_norm)
        btn_unnorm = QPushButton("Un-normalise")
        btn_unnorm.clicked.connect(self._run_unnormalise)
        norm_layout.addWidget(btn_unnorm)
        norm_layout.addStretch()
        body.addWidget(norm_box)

        # --- Filters ---
        filt_box = QGroupBox("Filters")
        filt_layout = QHBoxLayout(filt_box)
        dt_btn = QPushButton("⏰ Filter by date/time")
        dt_btn.clicked.connect(self._run_datetime_filter)
        filt_layout.addWidget(dt_btn)
        dp_btn = QPushButton("📏 Filter diameters")
        dp_btn.clicked.connect(self._run_diameter_filter)
        filt_layout.addWidget(dp_btn)
        filt_layout.addWidget(self._info_btn("Filter data by custom date range, day of week, month, hour of day, or year.", "Date/Time Filter"))
        filt_layout.addStretch()
        body.addWidget(filt_box)

        sec.set_content_layout(body)
        return sec

    # ── Section 3 · Combine & proceed ────────────────────────────────── #
    def _build_combine_section(self) -> CollapsibleSection:
        sec = CollapsibleSection("3 · Combine & proceed")
        body = QVBoxLayout()
        body.setSpacing(SPACE_MD)

        combine_box = QGroupBox("Combine files")
        combine_layout = QVBoxLayout(combine_box)
        merge_row = QHBoxLayout()

        self.merge_append_btn = QPushButton("Simple append (keep bins)")
        self.merge_append_btn.setProperty("class", "secondary")
        self.merge_append_btn.setEnabled(False)
        self.merge_append_btn.clicked.connect(lambda: self._execute_merge(mode="append"))
        merge_row.addWidget(self.merge_append_btn)
        merge_row.addWidget(self._info_btn("Appends selected files as they are. Missing bins will become NAs. Used for when you have multiple .csv files from the same instrument but different times", "Simple Append"))

        self.merge_splice_btn = QPushButton("Splice datasets (align bins)")
        self.merge_splice_btn.setProperty("class", "secondary")
        self.merge_splice_btn.setEnabled(False)
        self.merge_splice_btn.clicked.connect(lambda: self._execute_merge(mode="splice"))
        merge_row.addWidget(self.merge_splice_btn)
        merge_row.addWidget(self._info_btn("Joins two datasets with overlapping times, but different bins. Used to, for example, merge NanoSMPS and LongSMPS data.", "Splice Datasets"))

        self.harmonise_btn = QPushButton("Harmonise Dp")
        self.harmonise_btn.setProperty("class", "secondary")
        self.harmonise_btn.setEnabled(False)
        self.harmonise_btn.clicked.connect(self._run_harmonise)
        merge_row.addWidget(self.harmonise_btn)
        merge_row.addWidget(self._info_btn("Interpolates selected datasets onto a common set of diameters via cubic splines.", "Harmonise Dp"))
        merge_row.addStretch()
        combine_layout.addLayout(merge_row)

        self._merge_hint = QLabel("Select 2 or more valid files (Ctrl-click in the list) to enable merging.")
        self._merge_hint.setObjectName("Hint")
        combine_layout.addWidget(self._merge_hint)
        body.addWidget(combine_box)

        export_row = QHBoxLayout()
        save_btn = QPushButton("Export active datafile to CSV")
        save_btn.setProperty("class", "secondary")
        save_btn.clicked.connect(self._export_csv)
        export_row.addWidget(save_btn)
        export_row.addStretch()
        body.addLayout(export_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        body.addWidget(self.progress_bar)

        sec.set_content_layout(body)
        return sec

    # ── Sticky footer: the primary action, always pinned bottom-right ─── #
    def _build_footer(self) -> QWidget:
        container = QWidget()
        footer = QHBoxLayout(container)
        footer.setContentsMargins(SPACE_MD, SPACE_SM, SPACE_MD, 0)
        footer.addStretch()
        self._confirm_btn = QPushButton("📊  Proceed with selected dataset  →")
        self._confirm_btn.setProperty("class", "primary")
        self._confirm_btn.clicked.connect(self._confirm)
        footer.addWidget(self._confirm_btn)
        return container

    def _refresh_editing_label(self):
        if self._active_preview_path:
            self._editing_lbl.setText(f"Editing: {Path(self._active_preview_path).name}")
        else:
            self._editing_lbl.setText("Editing: —")

    def _on_col_changed(self, text):
        self._custom_col.setVisible(text == "Custom...")
        
    def _on_fmt_changed(self, idx):
        disp = self._fmt_combo.itemText(idx) 
        val = next(v for d, v in DATE_FORMAT_OPTIONS if d == disp) 
        self._custom_fmt.setVisible(val == "custom")

    def _browse_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select PNSD files", "", "Data (*.csv *.xlsx *.xls *.txt *.dat *.tsv)")
        if not paths: return
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(paths))
        for i, path in enumerate(paths):
            self._inject_file_to_list(path)
            self._parse_and_update(path)
            self.progress_bar.setValue(i + 1)
            QApplication.processEvents()
        self.progress_bar.setVisible(False)

    def _inject_file_to_list(self, path: str):
        if path not in self._entries:
            entry = FileEntryWidget(path)                                                # Create widget
            entry.removed.connect(self._remove_file)                                     # Bind removal
            entry.selected.connect(self._select_file)                                    # Bind selection
            entry.reparse_requested.connect(lambda p=path: self._parse_and_update(p))    # Bind reparse
            
            row_widget = QWidget()                                                       # Create wrapper
            row_layout = QHBoxLayout(row_widget)                                         # Add layout
            row_layout.setContentsMargins(0, 0, 0, 0)                                    # Strip margins
            
            # Build file preview snippet
            preview_lines = []
            try:
                if path.lower().endswith(('.csv', '.txt', '.dat', '.tsv')):
                    with open(path, 'r', encoding='utf-8', errors='replace') as f:
                        for _ in range(8):
                            try:
                                line = next(f).strip()
                                line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                                preview_lines.append(line[:120] + "…" if len(line) > 120 else line)
                            except StopIteration:
                                break
            except Exception as e:
                preview_lines = [f"Could not read file: {e}"]
            
            fname_safe = Path(path).name.replace('&', '&amp;').replace('<', '&lt;')
            body = "<br/>".join(preview_lines) if preview_lines else "<i>File is empty.</i>"
            tooltip_html = (
                f"<html><body style='font-family:Consolas,monospace; font-size:9pt; "
                f"background:#000000; color:#ffffff; padding:6px; white-space:pre;'>"
                f"<b style='color:#ffffff;'>📄 {fname_safe}</b><br/>"
                f"<hr style='border:1px solid #555; margin:3px 0;'/>"
                f"{body}</body></html>"
            )
            info_icon = _PreviewIconLabel(tooltip_html)                                  # Create icon
            info_icon.setObjectName("PreviewIcon")                                       # Style via QSS
            info_icon.setFixedSize(24, 24)                                               # Size icon
            info_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)                         # Centre icon
            info_icon.setCursor(Qt.CursorShape.PointingHandCursor)                       # Add pointer cursor
            
            row_layout.addWidget(info_icon)                                              # Pack icon
            row_layout.addWidget(entry)                                                  # Pack entry
            
            self._entries[path] = entry                                                  # Store reference
            entry._container = row_widget                                                # Link wrapper
            
            self._file_list_layout.insertWidget(self._file_list_layout.count()-1, row_widget) # Add to UI
            
    def _parse_and_update(self, path: str):
        entry = self._entries.get(path)
        if not entry: return
        
        if path.startswith("MERGED_"):
            if path in self._results: entry.set_result(self._results[path])
            return
        
        # Auto-detect timezone from file
        detected_tz = detect_timezone_from_csv(path)
        if detected_tz:
            self._tz_input.setText(detected_tz)
        
        g_col_text = self._col_combo.currentText()
        g_col = self._custom_col.text() if g_col_text == "Custom..." else g_col_text
        g_fmt_disp = self._fmt_combo.currentText() 
        g_fmt_val = next(v for d, v in DATE_FORMAT_OPTIONS if d == g_fmt_disp) 
        g_fmt = self._custom_fmt.text() if g_fmt_val == "custom" else g_fmt_val
        g_tz = self._tz_input.text()
        g_drop = self._drop_cols.text()
        g_na = self._na_combo.currentText()
        g_res_val = self._resample_val.text().strip()
        g_res_unit = self._resample_unit.currentText()
        
        col = entry.effective_col(g_col)
        fmt = entry.effective_fmt(g_fmt)
        tz = entry.effective_tz(g_tz)
        drop = entry.effective_drop(g_drop)
        na_text = entry.effective_na(g_na)
        res_val, res_unit = entry.effective_resample(g_res_val, g_res_unit)

        g_flag_col = self._flag_col.text().strip()
        g_flag_val = self._flag_val.text().strip() or "1"
        
        na_map = {"Drop Rows": "drop", "Fill (Fwd/Bwd)": "ffill", "Interpolate": "interpolate", "Fill Min (1e0)": "zero"}
        na_method = na_map.get(na_text, "drop")
        
        resample_rule = None
        if res_val.isdigit():
            resample_rule = f"{res_val}min" if res_unit == "Minutes" else f"{res_val}h" if res_unit == "Hours" else f"{res_val}D"
            
        result = load_pnsd_file(path, col, fmt, resample_rule, na_method, tz, drop, g_flag_col, g_flag_val)
        self._results[path] = result
        entry.set_result(result)
        
        if path in self._selected_paths or len(self._selected_paths) == 0: 
            self._active_preview_path = path
            if result.ok:
                self._update_dlogdp_box(result.diameters)
                self._populate_preview(result.df, result.diameters, f"Previewing: {Path(path).name}")
            else:
                self._populate_preview(None, None, f"Error loading {Path(path).name}: {result.error}")
                
        self._update_merge_buttons()

    def _apply_global_to_all(self):
        for path in list(self._entries): self._parse_and_update(path)

    def _select_file(self, path: str):
        modifiers = QApplication.keyboardModifiers()
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            if path in self._selected_paths: self._selected_paths.remove(path)
            else: self._selected_paths.add(path)
        else:
            self._selected_paths = {path}
            
        for p, entry in self._entries.items(): 
            entry.set_selected_style(p in self._selected_paths)
            
        if path in self._selected_paths: self._active_preview_path = path
        elif len(self._selected_paths) > 0: self._active_preview_path = next(iter(self._selected_paths))
        else: self._active_preview_path = None
            
        if self._active_preview_path:
            res = self._results.get(self._active_preview_path)
            if res and res.ok:
                self._update_dlogdp_box(res.diameters)
                title = f"Previewing: {Path(self._active_preview_path).name}"
                if len(self._selected_paths) > 1: title += f" (Part of {len(self._selected_paths)} selected files)"
                self._populate_preview(res.df, res.diameters, title)
        else:
            self._preview_table.clear()
            self._preview_hint.setText("No file selected.")
            self._refresh_editing_label()
        self._update_merge_buttons()

    def _remove_file(self, path: str):
        entry = self._entries.pop(path, None)
        if entry: 
            if hasattr(entry, '_container'): entry._container.deleteLater()
            else: entry.deleteLater()
        self._results.pop(path, None)
        if path in self._selected_paths: self._selected_paths.remove(path)
        if self._active_preview_path == path:
            self._preview_table.clear()
            self._active_preview_path = None
            self._refresh_editing_label()
        self._update_merge_buttons()

    def _clear_all(self):
        for path in list(self._entries): self._remove_file(path)
        self._selected_paths.clear()

    def _run_floor_threshold(self):
        df, diams = self._get_active_data()                                                                              
        if df is None: return                                                                                            
        try: threshold = float(self._floor_thresh.text())                                                                
        except ValueError: return                                                                                        
        
        num_replaced = (df < threshold).sum().sum()                                                                      
        df_floored = df.clip(lower=threshold)                                                                            
        
        if self._active_preview_path == "MERGED": self._merged_df = df_floored                                           
        else: self._results[self._active_preview_path].df = df_floored                                                   
            
        self._populate_preview(df_floored, diams, f"Values < {threshold} floored")                                       
        QMessageBox.information(self, "Sorted!", f"Successfully replaced {num_replaced} values that were below {threshold}. 🧹") 

    def _run_ceil_threshold(self): 
        df, diams = self._get_active_data() 
        if df is None: return 
        try: threshold = float(self._ceil_thresh.text()) 
        except ValueError: return 
        
        num_replaced = (df > threshold).sum().sum() 
        df_ceiled = df.clip(upper=threshold) 
        
        if self._active_preview_path == "MERGED": self._merged_df = df_ceiled 
        else: self._results[self._active_preview_path].df = df_ceiled 
        
        self._populate_preview(df_ceiled, diams, f"Values > {threshold} ceiling applied") 
        QMessageBox.information(self, "Sorted!", f"Successfully replaced {num_replaced} values that were above {threshold}. 🧹") 

    def _update_merge_buttons(self):
        valid_total = sum(1 for r in self._results.values() if r.ok)
        if len(self._selected_paths) > 0:
            valid_selected = sum(1 for p in self._selected_paths if self._results.get(p) and self._results[p].ok)
            can_merge = valid_selected >= 2
        else:
            can_merge = valid_total >= 2
        self.merge_append_btn.setEnabled(can_merge)
        self.merge_splice_btn.setEnabled(can_merge)
        self.harmonise_btn.setEnabled(can_merge)
        if hasattr(self, "_merge_hint"):
            self._merge_hint.setVisible(not can_merge)

    def _run_harmonise(self): 
        """Harmonises datasets onto a common target bin set using cubic splines."""
        target_keys = [p for p in self._results if p in self._selected_paths] if len(self._selected_paths) >= 2 else list(self._results.keys()) 
        valid_res_keys = [k for k in target_keys if self._results[k].ok] 
        if len(valid_res_keys) < 2: return 
        
        dlg = HarmoniseDialog(self._results, valid_res_keys, self)
        
        if dlg.exec():
            target_path = dlg.get_target_path()
            oor_action = dlg.get_oor_action()
            target_data = self._results[target_path]
            t_diams = np.array(target_data.diameters)
            
            harmonised_dfs = {}
            
            for path in valid_res_keys:
                if path == target_path:
                    harmonised_dfs[path] = self._results[path].df.copy()
                    continue
                
                data = self._results[path]
                o_diams = np.array(data.diameters)
                if np.array_equal(o_diams, t_diams):
                    harmonised_dfs[path] = data.df.copy()
                    continue
                
                # Math safe interpolation strictly for the spline application
                df_clean = data.df.interpolate(method='linear', axis=1, limit_direction='both').fillna(1e-4)
                try:
                    f = interp1d(o_diams, df_clean.values, axis=1, kind='cubic', bounds_error=False, fill_value=np.nan)
                    new_vals = f(t_diams)
                    harmonised_dfs[path] = pd.DataFrame(new_vals, index=data.df.index, columns=t_diams)
                except Exception as e:
                    QMessageBox.warning(self, "Spline Error", f"Failed to harmonise {Path(path).name}:\n{e}")
                    harmonised_dfs[path] = None

            # --- Out of Range (OOR) Handler ---
            if oor_action == "drop":
                valid_cols = pd.Series(True, index=t_diams)
                for df in harmonised_dfs.values():
                    if df is not None:
                        valid_cols = valid_cols & df.notna().any()
                
                t_diams = t_diams[valid_cols]
                for path in harmonised_dfs:
                    if harmonised_dfs[path] is not None:
                        harmonised_dfs[path] = harmonised_dfs[path].loc[:, t_diams]
            else:
                for path in harmonised_dfs:
                    if harmonised_dfs[path] is not None:
                        harmonised_dfs[path] = harmonised_dfs[path].fillna(1.0).clip(lower=1e-4)

            # Apply final assignments back to standard workflow
            for path in valid_res_keys:
                if harmonised_dfs[path] is not None:
                    self._results[path].df = harmonised_dfs[path]
                    self._results[path].diameters = list(t_diams)
                    self._results[path].n_bins = len(t_diams)
                    
            self._update_merge_buttons()
            if self._active_preview_path in valid_res_keys:
                self._populate_preview(self._results[self._active_preview_path].df, list(t_diams), "Harmonised")
            
            QMessageBox.information(self, "Success", "Datasets harmonised successfully! 📏")

    def _get_active_data(self):
        if self._active_preview_path == "MERGED": return self._merged_df, self._merged_diams
        res = self._results.get(self._active_preview_path)
        if res and res.ok: return res.df, res.diameters
        return None, None

    def _update_dlogdp_box(self, diams: list):
        if not diams: return
        log_diams = np.log10(diams)
        dlogdp = np.mean(np.diff(log_diams)) if len(log_diams) > 1 else 1.0
        self._norm_dlogdp.setText(f"{dlogdp:.4f}")

    def _run_qc(self):
        df, diams = self._get_active_data()
        if df is None: return
        try:
            win = int(self._qc_win.text())
            thresh = float(self._qc_thresh.text())
        except ValueError: return
        
        act = "na" if "NA" in self._qc_action.currentText() else "mean"
        df_clean, num_corrected, outliers = apply_qc_filter(df, win, thresh, act)
        
        if self._active_preview_path == "MERGED": self._merged_df = df_clean
        else: self._results[self._active_preview_path].df = df_clean
        
        self._populate_preview(df_clean, diams, "QC Filter Applied")
        QMessageBox.information(self, "QC Complete", f"Identified and corrected {num_corrected} anomalous data points.")
        self._show_qc_diagnostic_plot(df, df_clean, diams)

    def _show_qc_diagnostic_plot(self, df_raw, df_clean, diams):
        dlg = QDialog(self)
        dlg.setWindowTitle("QC Filter Diagnostic")
        layout = QVBoxLayout(dlg)
        fig = Figure(figsize=(8, 4))
        canvas = FigureCanvasQTAgg(fig)
        ax = fig.add_subplot(111)
        
        log_diams = np.log10(diams)
        dlogdp = np.mean(np.diff(log_diams)) if len(log_diams) > 1 else 1.0
        raw_n = df_raw.sum(axis=1) * dlogdp
        clean_n = df_clean.sum(axis=1) * dlogdp
        dates = mdates.date2num(df_raw.index)
        
        ax.plot(dates, raw_n, 'r-', alpha=0.5, label="Raw (Flagged spikes in red)")
        ax.plot(dates, clean_n, 'b-', label="Cleaned Baseline")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.set_ylabel("Total Number Conc.")
        ax.legend()
        fig.tight_layout()
        layout.addWidget(canvas)
        dlg.exec()

    def _run_line_loss(self):
        df, diams = self._get_active_data()
        if df is None: return
        try:
            L = float(self._ll_len.text())
            ID = float(self._ll_id.text())
            T = float(self._ll_temp.text())
            Q = float(self._ll_flow.text())
        except ValueError: return
        
        pen = calculate_line_losses(np.array(diams), L, ID, T, Q)
        corrected_df = df.div(pen, axis=1)
        
        if self._active_preview_path == "MERGED": self._merged_df = corrected_df
        else: self._results[self._active_preview_path].df = corrected_df
            
        self._populate_preview(corrected_df, diams, "Line Loss Corrected")
        self._show_line_loss_plot(diams, pen, df.mean(), corrected_df.mean())

    def _show_line_loss_plot(self, diams, pen, mean_raw, mean_corr):
        dlg = QDialog(self)
        dlg.setWindowTitle("Line Loss Diagnostic")
        layout = QVBoxLayout(dlg)
        fig = Figure(figsize=(8, 4))
        canvas = FigureCanvasQTAgg(fig)
        ax1 = fig.add_subplot(121)
        ax1.plot(diams, pen, 'b-')
        ax1.set_xscale('log')
        ax1.set_title("Penetration Function")
        ax2 = fig.add_subplot(122)
        ax2.plot(diams, mean_raw, 'k--', label="Raw")
        ax2.plot(diams, mean_corr, 'r-', label="Corrected")
        ax2.set_xscale('log')
        ax2.legend()
        ax2.set_title("Mean Distribution Impact")
        fig.tight_layout()
        layout.addWidget(canvas)
        dlg.exec()

    def _run_normalise(self):
        df, diams = self._get_active_data()
        if df is None: return
        try: val = float(self._norm_dlogdp.text())
        except ValueError: return
        
        new_df = df / val
        if self._active_preview_path == "MERGED": self._merged_df = new_df
        else: self._results[self._active_preview_path].df = new_df
        
        self._populate_preview(new_df, diams, "Normalised (dN/dlogDp)")
        QMessageBox.information(self, "Yeehaw!", "Data successfully normalised to dN/dlogDp! 🤠🚀")

    def _run_unnormalise(self):
        df, diams = self._get_active_data()
        if df is None: return
        try: val = float(self._norm_dlogdp.text())
        except ValueError: return
        
        new_df = df * val
        if self._active_preview_path == "MERGED": self._merged_df = new_df
        else: self._results[self._active_preview_path].df = new_df
        
        self._populate_preview(new_df, diams, "Un-normalised (dN)")
        QMessageBox.information(self, "Yeehaw!", "Data successfully converted back to N! (No normalisation) 🤠🐎")

    def _run_datetime_filter(self):
        """Open the date/time filter dialog."""
        df, diams = self._get_active_data()
        if df is None: return
        
        dlg = DateTimeFilterDialog(df, diams, self)
        
        if dlg.exec():
            filtered_df = dlg.get_filtered_data()
            n_removed = len(df) - len(filtered_df)
            
            if self._active_preview_path == "MERGED": 
                self._merged_df = filtered_df
            else: 
                self._results[self._active_preview_path].df = filtered_df
            
            self._populate_preview(filtered_df, diams, f"Date/Time Filtered ({n_removed} rows removed)")
            QMessageBox.information(self, "Filter Applied", f"Successfully filtered data. Removed {n_removed} rows. ✂️")

    def _run_diameter_filter(self):
        """Open the diameter filter dialog and keep a chosen sub-range of Dp bins."""
        df, diams = self._get_active_data()
        if df is None or diams is None or len(diams) == 0:
            return

        dlg = DiameterFilterDialog(df, diams, self)
        if dlg.exec():
            keep_mask = dlg.get_kept_mask()
            if keep_mask is None or int(np.sum(keep_mask)) < 1:
                QMessageBox.warning(self, "Filter Diameters", "No diameter bins selected.")
                return

            if len(keep_mask) != len(df.columns):
                QMessageBox.warning(self, "Filter Diameters", "Diameter/bin mismatch; could not apply filter.")
                return

            keep_cols = list(df.columns[keep_mask]) if hasattr(df.columns, '__getitem__') else [c for i, c in enumerate(df.columns) if keep_mask[i]]
            filtered_df = df.loc[:, keep_cols].copy()
            kept_diams = [float(d) for d in np.array(diams, dtype=float)[keep_mask]]

            if self._active_preview_path and self._active_preview_path.startswith("MERGED_"):
                self._results[self._active_preview_path].df = filtered_df
                self._results[self._active_preview_path].diameters = kept_diams
                self._results[self._active_preview_path].n_bins = len(kept_diams)
            elif self._active_preview_path in self._results:
                self._results[self._active_preview_path].df = filtered_df
                self._results[self._active_preview_path].diameters = kept_diams
                self._results[self._active_preview_path].n_bins = len(kept_diams)

            self._update_dlogdp_box(kept_diams)
            self._populate_preview(filtered_df, kept_diams, f"Diameter filtered ({len(kept_diams)} bins kept)")
            QMessageBox.information(self, "Filter Diameters", f"Kept {len(kept_diams)} diameter bins.")

    def _execute_merge(self, mode: str):
        if len(self._selected_paths) >= 2:
            target_keys = [p for p in self._results if p in self._selected_paths]
        else:
            target_keys = list(self._results.keys())

        valid_res = [self._results[k] for k in target_keys if self._results[k].ok]
        if len(valid_res) < 2: return
            
        name, ok = QInputDialog.getText(self, "Name Dataset", "Enter a name for the combined dataset:")
        if not ok or not name: return
        
        if mode == "splice":
            from gui.merger_dialogue import InstrumentMergerDialog
            if len(valid_res) == 2:
                dlg = InstrumentMergerDialog(valid_res[0], valid_res[1], self)   
                if dlg.exec():
                    merged_df = dlg.final_df                                     
                    final_diams = dlg.final_diams                                
                else: return
            else:
                QMessageBox.information(self, "Info", "Interactive splicing is only available when exactly 2 files are selected. Defaulting to algorithmic alignment.")
                base_diams = np.array(valid_res[0].diameters)
                aligned_dfs = []
                for r in valid_res:
                    if not np.array_equal(r.diameters, base_diams):
                        aligned = align_bins(r.df, np.array(r.diameters), base_diams)
                        aligned_dfs.append(aligned)
                    else: aligned_dfs.append(r.df)
                merged_df = pd.concat(aligned_dfs).sort_index()
                final_diams = base_diams
        else:
            merged_df = pd.concat([r.df for r in valid_res]).sort_index()
            final_diams = np.array([float(c) for c in merged_df.columns if pd.notna(c)])

        merged_df = merged_df[merged_df.index.notna()] 
        merged_df = merged_df[merged_df.index.year > 1990]
            
        fake_path = f"MERGED_{name}.csv"
        new_data = DataFile(path=Path(fake_path), df=merged_df, df_raw=merged_df, 
                            n_rows=len(merged_df), n_bins=len(final_diams),
                            diameters=list(final_diams), date_min=merged_df.index.min(), date_max=merged_df.index.max())
                            
        self._results[fake_path] = new_data
        self._inject_file_to_list(fake_path)
        
        self._entries[fake_path].set_result(new_data)                        
        
        self._selected_paths = {fake_path}
        for p, entry in self._entries.items(): entry.set_selected_style(p == fake_path)
        self._active_preview_path = fake_path
        self._update_dlogdp_box(list(final_diams))
        self._populate_preview(merged_df, list(final_diams), f"Previewing: {fake_path}")
        self._update_merge_buttons()
        
    def _export_csv(self):
        df, _ = self._get_active_data()
        if df is None: return
        path, _ = QFileDialog.getSaveFileName(self, "Export Dataset", "", "CSV (*.csv)")
        if path: df.to_csv(path)

    def _populate_preview(self, df: pd.DataFrame | None, diams: list, title: str):
        self._refresh_editing_label()
        if df is None:
            self._preview_hint.setText(title)
            self._preview_table.clear()
            self._preview_table.setRowCount(0)
            self._preview_table.setColumnCount(0)
            return
            
        self._preview_hint.setText(f"{title} | {len(df)} rows")
        self._preview_table.clear()
        preview_df = df.head(15)
        cols = list(preview_df.columns)[:20]
        self._preview_table.setRowCount(len(preview_df))
        self._preview_table.setColumnCount(len(cols) + 1)
        self._preview_table.setHorizontalHeaderLabels(["Datetime"] + [str(c) for c in cols])
        
        for r_idx, (ts, row) in enumerate(preview_df.iterrows()):
            self._preview_table.setItem(r_idx, 0, QTableWidgetItem(str(ts)))
            for c_idx, val in enumerate(row[cols]):
                self._preview_table.setItem(r_idx, c_idx+1, QTableWidgetItem(f"{val:.2f}" if val==val else ""))
        self._preview_table.resizeColumnsToContents()

    def _confirm(self):
        target_keys = [p for p in self._results if p in self._selected_paths]
        
        if not target_keys:
            if self._active_preview_path:
                target_keys = [self._active_preview_path]
            elif len(self._results) == 1:
                target_keys = list(self._results.keys())
            else:
                QMessageBox.warning(self, "No Selection", "Please click on a file in the list to select it before continuing.")
                return

        ok_res = {p: self._results[p] for p in target_keys if self._results.get(p) and self._results[p].ok}
        
        if ok_res: 
            self.data_confirmed.emit(ok_res)
        else:
            QMessageBox.warning(self, "Invalid Data", "The selected file(s) contain errors and cannot be used.")
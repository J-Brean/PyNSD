from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QFileDialog, QFrame, QGroupBox, QHBoxLayout, 
                             QHeaderView, QLabel, QLineEdit, QProgressBar, QPushButton, QScrollArea, QSplitter, 
                             QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QDialog, QMessageBox, QInputDialog)
from gui.file_entry_widget import FileEntryWidget
from utils.data_loader import (DATE_COLUMN_OPTIONS, DATE_FORMAT_OPTIONS, DEFAULT_DATE_COL, DEFAULT_DATE_FMT, 
                               DataFile, load_pnsd_file, apply_qc_filter, calculate_line_losses, align_bins)

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

    # --- FIX 1 & 2: Robust Information Buttons ---
    def _info_btn(self, text, title="Information"):
        """Creates a reliable clickable info button for tooltips."""
        btn = QPushButton("ℹ️")  # <-- Changed to emoji here!
        btn.setFixedSize(24, 24)   # Made slightly bigger to fit the emoji
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        # Removed the orange text color so the OS renders the native blue/grey emoji
        btn.setStyleSheet("border: none; font-size: 16px;") 
        btn.clicked.connect(lambda: QMessageBox.information(self, title, text))
        return btn

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(14, 14, 14, 10)

        intro_layout = QHBoxLayout()
        intro_lbl = QLabel("Load & Prepare Data")
        intro_lbl.setStyleSheet("font-size: 16px; font-weight: bold;")
        
        intro_text = (
            "Data Pipeline:\n"
            "1. Add files and parse datetimes/timezones.\n"
            "2. Apply QC to strip spikes, or correct for line-losses.\n"
            "3. Ctrl-click multiple files in the list, then use the Merge buttons to combine them.\n"
            "4. Export any dataset from the preview to CSV."
        )
        intro_layout.addWidget(intro_lbl)
        intro_layout.addWidget(self._info_btn(intro_text, "How to use this tool"))
        intro_layout.addStretch()
        root.addLayout(intro_layout)

        # --- Settings & Corrections Panel ---
        settings_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left: Global Import Settings
        import_box = QGroupBox("Import Settings")
        import_layout = QVBoxLayout(import_box)
        
        # --- FIX 5: Restored Custom Text Inputs ---
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Date Col:"))
        row1.addWidget(self._info_btn("The exact name of the column in your CSV containing the timestamp.", "Date Column"))
        
        self._col_combo = QComboBox()
        self._col_combo.addItems(DATE_COLUMN_OPTIONS)
        self._col_combo.currentTextChanged.connect(self._on_col_changed)
        row1.addWidget(self._col_combo)
        
        self._custom_col = QLineEdit()
        self._custom_col.setPlaceholderText("Enter custom column name...")
        self._custom_col.setVisible(False)
        row1.addWidget(self._custom_col)
        
        row1.addWidget(QLabel("Format:"))
        row1.addWidget(self._info_btn("The structural format of your dates. Use 'Custom...' if none match perfectly.", "Date Format"))
        self._fmt_combo = QComboBox()
        for disp, _ in DATE_FORMAT_OPTIONS: self._fmt_combo.addItem(disp)
        self._fmt_combo.currentIndexChanged.connect(self._on_fmt_changed)
        row1.addWidget(self._fmt_combo)
        
        self._custom_fmt = QLineEdit()
        self._custom_fmt.setPlaceholderText("e.g., yyyy/MM/dd HH:mm:ss")
        self._custom_fmt.setVisible(False)
        row1.addWidget(self._custom_fmt)
        
        import_layout.addLayout(row1)
        
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Timezone:"))
        row2.addWidget(self._info_btn("Standardises your data to a specific timezone (e.g., 'UTC', 'Europe/London'). Protects against DST gaps.", "Timezone"))
        self._tz_input = QLineEdit("UTC")
        row2.addWidget(self._tz_input)
        
        row2.addWidget(QLabel("Avg:"))
        row2.addWidget(self._info_btn("Optional: Average your data to a new timebase (e.g., enter '15' and select 'Minutes' to downsample).", "Timebase Averaging"))
        self._resample_val = QLineEdit()
        self._resample_val.setPlaceholderText("Val")
        self._resample_val.setFixedWidth(40)
        row2.addWidget(self._resample_val)
        self._resample_unit = QComboBox()
        self._resample_unit.addItems(["Minutes", "Hours", "Days"])
        row2.addWidget(self._resample_unit)
        import_layout.addLayout(row2)
        
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("NAs:"))
        row3.addWidget(self._info_btn("How to handle missing data.\nDrop: Removes the row.\nFill: Copies last valid value.\nInterpolate: Draws a line.\nFill Min: Replaces with 1e0.", "Missing Data Handling"))
        self._na_combo = QComboBox()
        self._na_combo.addItems(["Drop Rows", "Fill (Fwd/Bwd)", "Interpolate", "Fill Min (1e0)"])
        row3.addWidget(self._na_combo)
        
        row3.addWidget(QLabel("Drop Cols:"))
        row3.addWidget(self._info_btn("Comma-separated list of columns to ignore completely (e.g., 'Status, Error Code').", "Drop Columns"))
        self._drop_cols = QLineEdit()
        self._drop_cols.setPlaceholderText("Comma sep...")
        row3.addWidget(self._drop_cols)
        import_layout.addLayout(row3)
        
        apply_btn = QPushButton("Apply to all loaded files")
        apply_btn.clicked.connect(self._apply_global_to_all)
        import_layout.addWidget(apply_btn)
        settings_splitter.addWidget(import_box)

        # Right: Corrections & QC
        corr_box = QGroupBox("Corrections & QC (Applies to active preview)")
        corr_layout = QVBoxLayout(corr_box)
        
        qc_row = QHBoxLayout()
        qc_row.addWidget(QLabel("QC Win:"))
        qc_row.addWidget(self._info_btn("Size of the rolling window (number of data points) used to calculate the moving median baseline.", "QC Window"))
        self._qc_win = QLineEdit("20")
        self._qc_win.setFixedWidth(30)
        qc_row.addWidget(self._qc_win)
        
        qc_row.addWidget(QLabel("Std:"))
        qc_row.addWidget(self._info_btn("How many standard deviations away from the baseline a point must be to be flagged as a spike.", "QC Threshold"))
        self._qc_thresh = QLineEdit("3.0")
        self._qc_thresh.setFixedWidth(30)
        qc_row.addWidget(self._qc_thresh)
        
        self._qc_action = QComboBox()
        self._qc_action.addItems(["Replace with NA", "Replace with Mean"])
        qc_row.addWidget(self._qc_action)
        qc_row.addWidget(self._info_btn("What to do with identified spikes or negative values.", "QC Action"))
        
        qc_btn = QPushButton("Run QC")
        qc_btn.clicked.connect(self._run_qc)
        qc_row.addWidget(qc_btn)
        corr_layout.addLayout(qc_row)

        thresh_row = QHBoxLayout()                                                                 # Create new layout row
        thresh_row.addWidget(QLabel("Min Floor:"))                                                 # Label for threshold
        self._floor_thresh = QLineEdit("1.0")                                                      # Default to 1.0
        self._floor_thresh.setFixedWidth(40)                                                       # Keep UI tight
        thresh_row.addWidget(self._floor_thresh)                                                   # Add to layout

        btn_floor = QPushButton("Apply Floor")                                                     # Create action button
        btn_floor.clicked.connect(self._run_floor_threshold)                                       # Connect to logic
        thresh_row.addWidget(btn_floor)                                                            # Add to layout
        
        floor_info = ("MPSS instruments cannot reliably measure < 1 particle/cm3.\n"               # Define physical context
                      "Values below this are often non-physical. Low values are sometimes also used as error flags.\n"
                      "This tool replaces all values below your threshold with the threshold.")
        thresh_row.addWidget(self._info_btn(floor_info, "Minimum Floor Threshold"))                # Add info button
        thresh_row.addStretch()                                                                    # Push elements left
        corr_layout.addLayout(thresh_row)                                                          # Add to main corrections box
        
        ll_row = QHBoxLayout()
        
        ll_row = QHBoxLayout()
        ll_row.addWidget(QLabel("L(m):"))
        self._ll_len = QLineEdit("2.0")
        self._ll_len.setFixedWidth(30)
        ll_row.addWidget(self._ll_len)
        ll_row.addWidget(QLabel("ID(m):"))
        self._ll_id = QLineEdit("0.006")
        self._ll_id.setFixedWidth(40)
        ll_row.addWidget(self._ll_id)
        ll_row.addWidget(QLabel("T(K):"))
        self._ll_temp = QLineEdit("293")
        self._ll_temp.setFixedWidth(35)
        ll_row.addWidget(self._ll_temp)
        ll_row.addWidget(QLabel("Q(LPM):"))
        self._ll_flow = QLineEdit("1.0")
        self._ll_flow.setFixedWidth(30)
        ll_row.addWidget(self._ll_flow)
        
        ll_btn = QPushButton("Line Loss Correct")
        ll_btn.clicked.connect(self._run_line_loss)
        ll_row.addWidget(ll_btn)
        ll_row.addWidget(self._info_btn("Applies Gormley-Kennedy diffusional loss corrections to the active dataset.", "Line Loss Correction"))
        corr_layout.addLayout(ll_row)
        
        norm_row = QHBoxLayout()
        norm_row.addWidget(QLabel("dlogDp:"))
        norm_row.addWidget(self._info_btn("The calculated logarithmic width of your diameter bins. Required to normalise between dN and dN/dlogDp. n.b., the subsequent calculations will anticipate that your values are in dNdlogdp", "dlogDp Normalisation"))
        self._norm_dlogdp = QLineEdit("1.0")
        self._norm_dlogdp.setFixedWidth(50)
        norm_row.addWidget(self._norm_dlogdp)
        
        btn_norm = QPushButton("Normalise")
        btn_norm.clicked.connect(self._run_normalise)
        norm_row.addWidget(btn_norm)
        
        btn_unnorm = QPushButton("Un-normalise")
        btn_unnorm.clicked.connect(self._run_unnormalise)
        norm_row.addWidget(btn_unnorm)
        corr_layout.addLayout(norm_row)
        
        settings_splitter.addWidget(corr_box)
        root.addWidget(settings_splitter)

        # --- File Lists & Preview ---
        file_row = QHBoxLayout()
        add_btn = QPushButton("+ Add files…")
        add_btn.clicked.connect(self._browse_files)
        file_row.addWidget(add_btn)
        clear_btn = QPushButton("Clear all")
        clear_btn.clicked.connect(self._clear_all)
        file_row.addWidget(clear_btn)
        root.addLayout(file_row)

        list_splitter = QSplitter(Qt.Orientation.Vertical)
        
        self._file_list_inner = QWidget()
        self._file_list_layout = QVBoxLayout(self._file_list_inner)
        self._file_list_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._file_list_inner)
        list_splitter.addWidget(scroll)
        
        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        self._preview_hint = QLabel("Preview")
        preview_layout.addWidget(self._preview_hint)
        self._preview_table = QTableWidget()
        self._preview_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        preview_layout.addWidget(self._preview_table)
        list_splitter.addWidget(preview_container)
        
        list_splitter.setSizes([600, 200])   # Forces top to start at 600px and bottom at 200px
        list_splitter.setStretchFactor(0, 1) # If window grows, give all extra space to the file list
        list_splitter.setStretchFactor(1, 0) # Tell the preview table NOT to grow
        root.addWidget(list_splitter, stretch=1)
        # --- Action Row ---
        action_row = QHBoxLayout()
        
        self.merge_append_btn = QPushButton("Simple Append (Keep Bins)")
        self.merge_append_btn.setStyleSheet("font-weight: bold; background-color: #e2d5cb;")
        self.merge_append_btn.setEnabled(False)
        self.merge_append_btn.clicked.connect(lambda: self._execute_merge(mode="append"))
        action_row.addWidget(self.merge_append_btn)
        action_row.addWidget(self._info_btn("Appends selected files as they are. Missing bins will become NAs. Used for when you have multiple .csv files from the same instrument but different times", "Simple Append"))
        
        self.merge_splice_btn = QPushButton("Splice Datasets (Align Bins)")
        self.merge_splice_btn.setStyleSheet("font-weight: bold; background-color: #d1c4ba;")
        self.merge_splice_btn.setEnabled(False)
        self.merge_splice_btn.clicked.connect(lambda: self._execute_merge(mode="splice"))
        action_row.addWidget(self.merge_splice_btn)
        action_row.addWidget(self._info_btn("Joins two datasets with overlapping times, but different bins. Used to, for example, merge NanoSMPS and LongSMPS data. The spline feature will give the datasets a consistent set of diameters with consistent dlogdp values (note: check that this maintains total N!).", "Splice Datasets"))
        
        save_btn = QPushButton("Export Active Preview to CSV")
        save_btn.clicked.connect(self._export_csv)
        action_row.addWidget(save_btn)
        action_row.addStretch()
        
        self._confirm_btn = QPushButton("📊 Proceed with selected dataframe  →")
        self._confirm_btn.setStyleSheet("font-size: 16pt; font-weight: bold;") # Sets size and boldness
        self._confirm_btn.clicked.connect(self._confirm)
        action_row.addWidget(self._confirm_btn)
        root.addLayout(action_row)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)

    # --- UI Events for Custom Fields ---
    def _on_col_changed(self, text):
        self._custom_col.setVisible(text == "Custom...")
        
    def _on_fmt_changed(self, idx):
        _, val = DATE_FORMAT_OPTIONS[idx]
        self._custom_fmt.setVisible(val == "custom")

    # --- Loading & UI Handling ---

    def _browse_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select PNSD files", "", "Data (*.csv *.xlsx *.xls)")
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
            entry = FileEntryWidget(path)
            entry.removed.connect(self._remove_file)
            entry.selected.connect(self._select_file)
            
            # FIX: Use a lambda to explicitly pass the path to the reparse function
            entry.reparse_requested.connect(lambda p=path: self._parse_and_update(p)) 
            
            self._entries[path] = entry
            self._file_list_layout.insertWidget(self._file_list_layout.count()-1, entry)
            
    def _parse_and_update(self, path: str):
        entry = self._entries.get(path)
        if not entry: return
        
        if path.startswith("MERGED_"):
            if path in self._results: entry.set_result(self._results[path])
            return
        
        # 1. Read Global Settings
        g_col_text = self._col_combo.currentText()
        g_col = self._custom_col.text() if g_col_text == "Custom..." else g_col_text
        g_fmt_val = DATE_FORMAT_OPTIONS[self._fmt_combo.currentIndex()][1]
        g_fmt = self._custom_fmt.text() if g_fmt_val == "custom" else g_fmt_val
        g_tz = self._tz_input.text()
        g_drop = self._drop_cols.text()
        g_na = self._na_combo.currentText()
        g_res_val = self._resample_val.text().strip()
        g_res_unit = self._resample_unit.currentText()
        
        # 2. Ask file entry for its effective settings (Overrides win)
        col = entry.effective_col(g_col)
        fmt = entry.effective_fmt(g_fmt)
        tz = entry.effective_tz(g_tz)
        drop = entry.effective_drop(g_drop)
        na_text = entry.effective_na(g_na)
        res_val, res_unit = entry.effective_resample(g_res_val, g_res_unit)
        
        # 3. Process the logic
        na_map = {"Drop Rows": "drop", "Fill (Fwd/Bwd)": "ffill", "Interpolate": "interpolate", "Fill Min (1e0)": "zero"}
        na_method = na_map.get(na_text, "drop")
        
        resample_rule = None
        if res_val.isdigit():
            resample_rule = f"{res_val}min" if res_unit == "Minutes" else f"{res_val}h" if res_unit == "Hours" else f"{res_val}D"
            
        result = load_pnsd_file(path, col, fmt, resample_rule, na_method, tz, drop)
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
        self._update_merge_buttons()

    def _remove_file(self, path: str):
        entry = self._entries.pop(path, None)
        if entry: entry.deleteLater()
        self._results.pop(path, None)
        if path in self._selected_paths: self._selected_paths.remove(path)
        if self._active_preview_path == path: 
            self._preview_table.clear()
            self._active_preview_path = None
        self._update_merge_buttons()

    def _clear_all(self):
        for path in list(self._entries): self._remove_file(path)
        self._selected_paths.clear()

    # --- Corrections & Splicing ---

    def _run_floor_threshold(self):
        df, diams = self._get_active_data()                                                        # Fetch active dataset
        if df is None: return                                                                      # Escape if empty
        
        try: threshold = float(self._floor_thresh.text())                                          # Parse user input
        except ValueError: return                                                                  # Ignore text/typos
        
        num_replaced = (df < threshold).sum().sum()                                                # Count values below threshold before modifying
        
        df_floored = df.clip(lower=threshold)                                                      # Pandas replaces all values below threshold
        
        if self._active_preview_path == "MERGED": self._merged_df = df_floored                     # Save to merged state
        else: self._results[self._active_preview_path].df = df_floored                             # Save to individual file state
            
        self._populate_preview(df_floored, diams, f"Values < {threshold} floored")                 # Update GUI table
        QMessageBox.information(self, "Sorted!", f"Successfully replaced {num_replaced} values that were below {threshold}. 🧹") # Success pop-up
        
    def _update_merge_buttons(self):
        valid_total = sum(1 for r in self._results.values() if r.ok)
        if len(self._selected_paths) > 0:
            valid_selected = sum(1 for p in self._selected_paths if self._results.get(p) and self._results[p].ok)
            can_merge = valid_selected >= 2
        else:
            can_merge = valid_total >= 2
        self.merge_append_btn.setEnabled(can_merge)
        self.merge_splice_btn.setEnabled(can_merge)

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

    def _execute_merge(self, mode: str):
        if len(self._selected_paths) >= 2:
            target_keys = [p for p in self._results if p in self._selected_paths]
        else:
            target_keys = list(self._results.keys())

        valid_res = [self._results[k] for k in target_keys if self._results[k].ok]
        if len(valid_res) < 2: return
            
        name, ok = QInputDialog.getText(self, "Name Dataset", "Enter a name for the combined dataset:")
        if not ok or not name: return
        
        # Dynamic import to avoid circular dependency
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
        
        self._entries[fake_path].set_result(new_data)                        # Tell the UI row it's finished!
        
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
        # 1. Grab only the files that are currently highlighted (Ctrl-clicked)
        target_keys = [p for p in self._results if p in self._selected_paths]
        
        # 2. Fallbacks: If nothing is highlighted, try the active preview, or auto-select if there's only 1 file
        if not target_keys:
            if self._active_preview_path:
                target_keys = [self._active_preview_path]
            elif len(self._results) == 1:
                target_keys = list(self._results.keys())
            else:
                QMessageBox.warning(self, "No Selection", "Please click on a file in the list to select it before continuing.")
                return

        # 3. Filter for valid files and emit them to the main application
        ok_res = {p: self._results[p] for p in target_keys if self._results.get(p) and self._results[p].ok}
        
        if ok_res: 
            self.data_confirmed.emit(ok_res)
        else:
            QMessageBox.warning(self, "Invalid Data", "The selected file(s) contain errors and cannot be used.")
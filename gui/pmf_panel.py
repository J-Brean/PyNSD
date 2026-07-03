import os                                                                # For path manipulation
import shutil                                                            # For copying the key file
import subprocess                                                        # For running external .exe
import copy                                                              # For copying colormaps
import re                                                                # For parsing the Fortran LOG files
import json                                                              # For reading/writing archive metadata
import pandas as pd                                                      # For data handling
import numpy as np                                                       # For maths operations
import matplotlib.pyplot as plt                                          # Global Matplotlib import
import matplotlib as mpl                                                 # Global Matplotlib settings
import seaborn as sns                                                    # Global Seaborn import
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg         # Embedded plotting for Qt
from matplotlib.figure import Figure                                     # Figure object 
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFileDialog, QLabel, 
                             QLineEdit, QComboBox, QDoubleSpinBox, 
                             QSlider, QGroupBox, QGridLayout, 
                             QMessageBox, QTableWidget, QSpinBox, 
                             QTableWidgetItem, QDialog, QProgressBar, 
                             QApplication, QCheckBox, QTabWidget,
                             QInputDialog, QListWidget)
from PyQt6.QtGui import QFont                                            # For monospace fonts
from PyQt6.QtCore import Qt, QSettings  # Added QSettings back to the import list
from utils.pmf_ini_generator import generate_pmf_ini                     # External INI generator
from utils.data_loader import DATE_COLUMN_OPTIONS, DATE_FORMAT_OPTIONS   # Shared datetime parse options
from utils.tracer_join import build_tracer_frame, align_to_index         # External tracer helpers

class CowProgressDialog(QDialog):
    def __init__(self, total_steps, parent=None):
        super().__init__(parent)                                         
        self.setWindowTitle("PMF Batch Execution")                       
        self.setMinimumWidth(400)                                        
        self.main_layout = QVBoxLayout(self)                             
        self.cow_label = QLabel()                                        
        self.cow_label.setFont(QFont("Courier", 10))                     
        self.progress = QProgressBar()                                   
        self.progress.setMaximum(total_steps)                            
        self.main_layout.addWidget(self.cow_label)                       
        self.main_layout.addWidget(self.progress)                        
        
    def update_progress(self, step, factors, fpeak):
        cow = f"""
 __________________________________________________
< Running {factors} factor solution with FPEAK = {fpeak} >
 --------------------------------------------------
        \\   ^__^
         \\  (oo)\\_______
            (__)\\       )\\/\\
                ||----w |
                ||     ||
"""
        self.cow_label.setText(cow)                                      
        self.progress.setValue(step)                                     
        QApplication.processEvents()                                     

class OptimiseProgressDialog(QDialog):
    def __init__(self, max_iterations, parent=None):
        super().__init__(parent)                                         
        self.setWindowTitle("Error Coefficient Optimisation")            
        self.setMinimumWidth(450)                                        
        self.main_layout = QVBoxLayout(self)                             
        self.info_label = QLabel("Initialising optimisation sequence...\nThis may take a while.") 
        self.info_label.setFont(QFont("Courier", 10))                    
        self.progress = QProgressBar()                                   
        self.progress.setMaximum(max_iterations)                         
        self.main_layout.addWidget(self.info_label)                      
        self.main_layout.addWidget(self.progress)                        

    def update_status(self, step, coeff, q_ratio):
        text = (f"Optimising Model...\n"
                f"Iteration: {step}\n"
                f"Current Error Fraction (C3): {coeff:.4f}\n"
                f"Achieved Q/Qexp: {q_ratio:.4f}\n\n"
                f"Hunting for Q/Qexp = 1.0...")                          
        self.info_label.setText(text)                                    
        self.progress.setValue(step)                                     
        QApplication.processEvents()                                     

class BootstrapProgressDialog(QDialog):
    def __init__(self, total, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bootstrap Uncertainty")
        self.setMinimumWidth(470)
        self.cancelled = False
        lay = QVBoxLayout(self)
        self.info = QLabel("Preparing bootstrap runs...")
        self.info.setFont(QFont("Courier", 10))
        self.progress = QProgressBar(); self.progress.setMaximum(total)
        btn = QPushButton("Cancel"); btn.clicked.connect(self._cancel)
        lay.addWidget(self.info); lay.addWidget(self.progress); lay.addWidget(btn)

    def _cancel(self):
        self.cancelled = True

    def update(self, step, total, eta_s, ok_frac):
        eta = f"{int(eta_s // 60)}m {int(eta_s % 60)}s" if eta_s > 0 else "estimating..."
        self.info.setText(f"Bootstrap run {step}/{total}\n"
                          f"Estimated time remaining: {eta}\n"
                          f"Factor mapping success so far: {ok_frac * 100:.0f}%")
        self.progress.setValue(step)
        QApplication.processEvents()

class BootstrapResultsDialog(QDialog):
    def __init__(self, diams, base_F, F_boot, contrib_boot, names, success_rate, n_used, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Bootstrap Results ({n_used} runs, mapping success {success_rate * 100:.0f}%)")
        self.resize(1100, 760)
        self.diams = np.asarray(diams, dtype=float)
        self.base_F = base_F; self.F_boot = F_boot; self.contrib = contrib_boot; self.names = names
        lay = QVBoxLayout(self)
        self.fig = Figure(figsize=(11, 7)); self.canvas = FigureCanvasQTAgg(self.fig)
        lay.addWidget(self.canvas)
        btn = QPushButton("Save Figure"); btn.clicked.connect(self._save)
        lay.addWidget(btn)
        self._plot()

    def _plot(self):
        k = len(self.names)
        self.fig.clear()
        gs = self.fig.add_gridspec(2, k)
        for a in range(k):
            ax = self.fig.add_subplot(gs[0, a])
            boot = self.F_boot[a]                                        # (n_used, n_variables)
            lo = np.nanpercentile(boot, 5, axis=0)
            hi = np.nanpercentile(boot, 95, axis=0)
            med = np.nanpercentile(boot, 50, axis=0)
            on_diams = (len(self.diams) == boot.shape[1])
            x = self.diams if on_diams else np.arange(boot.shape[1])
            ax.fill_between(x, lo, hi, color='steelblue', alpha=0.3)
            ax.plot(x, med, color='steelblue', lw=1.5, label='Bootstrap median')
            ax.plot(x, self.base_F[:, a], color='red', lw=1.2, ls='--', label='Base')
            if on_diams:
                ax.set_xscale('log')
            ax.set_title(self.names[a], fontsize=9)
            if a == 0:
                ax.set_ylabel('F (dN/dlogDp)'); ax.legend(fontsize=7)

        axb = self.fig.add_subplot(gs[1, :])
        axb.boxplot([self.contrib[:, a] for a in range(k)], showfliers=False)
        axb.set_xticks(range(1, k + 1)); axb.set_xticklabels(self.names, rotation=20)
        axb.set_ylabel('Mean contribution (cm⁻³)')
        axb.set_title('Bootstrap distribution of factor mean contribution')
        self.fig.tight_layout(); self.canvas.draw()

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Figure", "Bootstrap_Uncertainty.png",
                                              "PNG (*.png);;PDF (*.pdf)")
        if path:
            self.fig.savefig(path, dpi=300, bbox_inches='tight')

class TracerLoadDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load External Tracers")
        self.setMinimumWidth(560)
        self.raw_df = None
        self.tracer_df = None                                            # Result frame on accept

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Load a co-located gas/met file (NOx, BC, O3, solar radiation, WD, WS, ...).\n"
                                "It is aligned to the PMF timeline by nearest timestamp when plotted."))

        row1 = QHBoxLayout()
        self.btn_browse = QPushButton("+ Load Tracer File")
        self.btn_browse.clicked.connect(self._browse)
        self.lbl_path = QLabel("No file loaded")
        row1.addWidget(self.btn_browse); row1.addWidget(self.lbl_path); row1.addStretch()
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Date Col:"))
        self.date_col = QComboBox(); self.date_col.addItems(DATE_COLUMN_OPTIONS)
        self.date_col.currentTextChanged.connect(lambda t: self.custom_date.setVisible(t == "Custom..."))
        row2.addWidget(self.date_col)
        self.custom_date = QLineEdit(); self.custom_date.setPlaceholderText("Custom date column"); self.custom_date.setVisible(False)
        row2.addWidget(self.custom_date)
        row2.addWidget(QLabel("Format:"))
        self.fmt = QComboBox()
        for disp, _ in DATE_FORMAT_OPTIONS:
            self.fmt.addItem(disp)
        self.fmt.currentIndexChanged.connect(self._on_fmt)
        row2.addWidget(self.fmt)
        self.custom_fmt = QLineEdit(); self.custom_fmt.setPlaceholderText("Custom format"); self.custom_fmt.setVisible(False)
        row2.addWidget(self.custom_fmt)
        row2.addWidget(QLabel("TZ:"))
        self.tz = QLineEdit("UTC"); self.tz.setFixedWidth(80)
        row2.addWidget(self.tz)
        layout.addLayout(row2)

        layout.addWidget(QLabel("Select tracer columns to keep (Ctrl/Shift for multiple):"))
        self.col_list = QListWidget()
        self.col_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        layout.addWidget(self.col_list)

        btn_ok = QPushButton("Parse & Use Tracers")
        btn_ok.setStyleSheet("background-color: #4CAF50; color: white;")
        btn_ok.clicked.connect(self._accept)
        layout.addWidget(btn_ok)

    def _on_fmt(self, idx):
        val = next(v for d, v in DATE_FORMAT_OPTIONS if d == self.fmt.itemText(idx))
        self.custom_fmt.setVisible(val == "custom")

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Tracer File", "",
                                              "Data (*.csv *.xlsx *.xls *.txt *.tsv *.dat)")
        if not path:
            return
        try:
            if path.lower().endswith((".xlsx", ".xls")):
                raw = pd.read_excel(path, dtype=str)
            else:
                raw = pd.read_csv(path, dtype=str)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Failed to read file:\n{exc}")
            return
        self.raw_df = raw
        self.lbl_path.setText(os.path.basename(path))
        cols = [str(c) for c in raw.columns]
        self.col_list.clear(); self.col_list.addItems(cols)
        low = {c.lower(): c for c in cols}
        for cand in ["date", "datetime", "timestamp", "time"]:             # Guess the datetime column
            if cand in low:
                self.date_col.setCurrentText(low[cand] if low[cand] in DATE_COLUMN_OPTIONS else "Custom...")
                if low[cand] not in DATE_COLUMN_OPTIONS:
                    self.custom_date.setText(low[cand])
                break

    def _accept(self):
        if self.raw_df is None:
            return QMessageBox.warning(self, "No File", "Load a tracer file first.")
        date_choice = self.date_col.currentText()
        date_col = self.custom_date.text().strip() if date_choice == "Custom..." else date_choice
        match = [c for c in self.raw_df.columns if str(c).strip().lower() == date_col.strip().lower()]
        if not match:
            return QMessageBox.warning(self, "Date Column", f"Column '{date_col}' not found in the file.")
        fmt_val = next(v for d, v in DATE_FORMAT_OPTIONS if d == self.fmt.currentText())
        date_fmt = self.custom_fmt.text().strip() if fmt_val == "custom" else fmt_val
        value_cols = [i.text() for i in self.col_list.selectedItems() if i.text() != match[0]]
        if not value_cols:
            return QMessageBox.warning(self, "Columns", "Select at least one tracer column.")
        try:
            self.tracer_df = build_tracer_frame(self.raw_df, match[0], date_fmt,
                                                self.tz.text().strip() or "UTC", value_cols)
        except Exception as exc:
            return QMessageBox.critical(self, "Parse Error", f"Failed to parse tracers:\n{exc}")
        if self.tracer_df.empty:
            return QMessageBox.warning(self, "Empty", "No valid tracer rows parsed. Check date format/timezone.")
        self.accept()

class RenameDialog(QDialog):
    def __init__(self, current_factors, factor_names, parent=None):
        super().__init__(parent)                                         
        self.setWindowTitle("Rename Active Factors")                     
        self.setMinimumWidth(300)                                        
        self.factor_names = factor_names                                 
        self.main_layout = QVBoxLayout(self)                             
        
        self.table = QTableWidget(current_factors, 2)                    
        self.table.setHorizontalHeaderLabels(["Raw ID", "Custom Name"])  
        
        for i in range(current_factors):                                 
            raw_str = f"Factor {i+1}"                                    
            raw_item = QTableWidgetItem(raw_str)                         
            raw_item.setFlags(raw_item.flags() ^ Qt.ItemFlag.ItemIsEditable) 
            self.table.setItem(i, 0, raw_item)                           
            
            existing_name = self.factor_names.get(raw_str, f"F{i+1}")    
            self.table.setItem(i, 1, QTableWidgetItem(existing_name))    
            
        self.main_layout.addWidget(self.table)                           
        
        btn_apply = QPushButton("Apply Names & Close")                   
        btn_apply.setStyleSheet("background-color: #4CAF50; color: white;") 
        btn_apply.clicked.connect(self.apply_names)                      
        self.main_layout.addWidget(btn_apply)                            
        
    def apply_names(self):
        self.factor_names.clear()                                        
        for i in range(self.table.rowCount()):                           
            raw = self.table.item(i, 0).text()                           
            custom = self.table.item(i, 1).text()                        
            self.factor_names[raw] = custom                              
        self.accept()                                                    

class CombineFactorsDialog(QDialog):
    def __init__(self, panel, parent=None):
        super().__init__(parent)
        self.panel = panel
        self.selected = None                                             # (idx_a, idx_b) on accept
        self.setWindowTitle("Combine Factors")
        self.setMinimumWidth(340)
        layout = QVBoxLayout(self)

        info = QLabel("Select two factors to merge. Their G (time) and\n"
                      "F (profile) columns are summed into one factor.")
        layout.addWidget(info)

        row = QHBoxLayout()
        self.cb_a = QComboBox(); self.cb_b = QComboBox()
        for i in range(panel.current_factors):
            name = panel._get_factor_name(i)
            self.cb_a.addItem(name, i); self.cb_b.addItem(name, i)
        if panel.current_factors > 1:
            self.cb_b.setCurrentIndex(1)                                 # Default to a different pair
        row.addWidget(self.cb_a); row.addWidget(QLabel("+")); row.addWidget(self.cb_b)
        layout.addLayout(row)

        btn = QPushButton("Combine")
        btn.setStyleSheet("background-color: #4CAF50; color: white;")
        btn.clicked.connect(self._apply)
        layout.addWidget(btn)

    def _apply(self):
        a = self.cb_a.currentData(); b = self.cb_b.currentData()
        if a == b:
            QMessageBox.warning(self, "Invalid", "Please select two different factors.")
            return
        self.selected = (a, b)
        self.accept()

class TabbedVisualizer(QDialog):
    def __init__(self, panel, parent=None):
        super().__init__(parent)
        self.panel = panel
        self.resize(1400, 850)
        self._building = False                                           # Re-entrancy guard for slider-driven rebuilds
        self._pending = None                                            # Latest slider request deferred during a build

        self.tabs = QTabWidget()
        self.main_layout = QVBoxLayout(self)
        self._build_controls()                                           # Live FPEAK / factor sliders
        self.main_layout.addWidget(self.tabs)

        self._reload_and_build()                                         # Populate tabs from active model

    def _build_controls(self):
        # Discover the computed batch runs so the sliders can only snap to models that exist on disk.
        self.runs = {}                                                   # (factors, fpeak) -> combo index
        for i in range(self.panel.combo_fpeak.count()):
            text = self.panel.combo_fpeak.itemText(i)
            fm = re.search(r"Factors:?\s*(\d+)", text, re.IGNORECASE)
            pm = re.search(r"FPEAK:?\s*([-0-9.]+)", text, re.IGNORECASE)
            if fm and pm:
                self.runs[(int(fm.group(1)), float(pm.group(1)))] = i

        if not self.runs:                                                # Library loads have no batch grid
            note = QLabel("Live FPEAK / factor switching is unavailable (no batch runs in this session).")
            note.setStyleSheet("color: gray; font-style: italic;")
            self.main_layout.addWidget(note)
            self.slider_fac = self.slider_fpk = None
            return

        self.factors_list = sorted({f for f, p in self.runs})
        self.fpeaks_list = sorted({p for f, p in self.runs})

        bar = QHBoxLayout()
        self.slider_fac = QSlider(Qt.Orientation.Horizontal)
        self.slider_fac.setRange(0, len(self.factors_list) - 1)
        self.slider_fac.setToolTip("Slide to switch the number of factors in the active solution.")
        self.lbl_fac_val = QLabel()
        self.slider_fpk = QSlider(Qt.Orientation.Horizontal)
        self.slider_fpk.setRange(0, len(self.fpeaks_list) - 1)
        self.slider_fpk.setToolTip("Slide to switch the FPEAK rotation of the active solution.")
        self.lbl_fpk_val = QLabel()

        # Start the sliders on whatever model is already active.
        try: fi = self.factors_list.index(self.panel.current_factors)
        except ValueError: fi = 0
        try: pi = self.fpeaks_list.index(self.panel.current_fpeak)
        except ValueError: pi = 0
        for s, v in ((self.slider_fac, fi), (self.slider_fpk, pi)):
            s.blockSignals(True); s.setValue(v); s.blockSignals(False)   # Avoid firing a rebuild during setup
        self._sync_control_labels()

        self.slider_fac.valueChanged.connect(self._on_controls_changed)
        self.slider_fpk.valueChanged.connect(self._on_controls_changed)

        bar.addWidget(QLabel("Factors:")); bar.addWidget(self.slider_fac); bar.addWidget(self.lbl_fac_val)
        bar.addSpacing(20)
        bar.addWidget(QLabel("FPEAK:")); bar.addWidget(self.slider_fpk); bar.addWidget(self.lbl_fpk_val)
        self.main_layout.addLayout(bar)

    def _sync_control_labels(self):
        self.lbl_fac_val.setText(str(self.factors_list[self.slider_fac.value()]))
        self.lbl_fpk_val.setText(f"{self.fpeaks_list[self.slider_fpk.value()]:g}")

    def _on_controls_changed(self):
        self._sync_control_labels()
        f_val = self.factors_list[self.slider_fac.value()]
        p_val = self.fpeaks_list[self.slider_fpk.value()]
        idx = self.runs.get((f_val, p_val))
        if idx is None:                                                  # This factor/FPEAK pair was not computed
            self.setWindowTitle(f"PyNSD Visualisation Suite | No run for Factors={f_val}, FPEAK={p_val:g}")
            return
        if self._building:                                              # A rebuild is already running: defer, don't re-enter
            self._pending = (f_val, p_val)
            return
        self.panel.combo_fpeak.setCurrentIndex(idx)                      # Synchronously reloads F/G via update_fpeak
        self._reload_and_build()

    def _reload_and_build(self):
        if self.panel.f_matrix is None or self.panel.g_matrix is None:   # Guard against a failed reload
            return
        if self._building:                                              # Never rebuild on top of an in-progress rebuild
            return
        self._building = True
        try:
            prev = self.tabs.currentIndex()
            while self.tabs.count():                                     # Dispose old pages (and their figures)
                w = self.tabs.widget(0); self.tabs.removeTab(0); w.deleteLater()

            n_rows = len(self.panel.g_matrix); n_cols = len(self.panel.f_matrix)
            q_ratio = self.panel._get_q_ratio(self.panel.current_factors, self.panel.current_fpeak, n_rows, n_cols)
            active_text = self.panel.combo_fpeak.currentText()
            self.setWindowTitle(f"PyNSD Visualisation Suite ({active_text}) | Q/Qexp={q_ratio:.3f}")

            self.g_number = self.panel.get_scaled_g()

            builders = [self._build_size_tab, self._build_time_tab, self._build_seasonal_tab,
                        self._build_dow_tab, self._build_diurnal_tab, self._build_mass_tab,
                        self._build_resid_recon_tab, self._build_diag_tab, self._build_qq_tab,
                        self._build_summary_tab, self._build_gspace_tab, self._build_tracer_tab,
                        self._build_polar_tab, self._build_nucsplit_tab]
            if self.panel.chk_wide_pmf.isChecked():
                builders += [self._build_wide_profiles_tab, self._build_widepmf_tab]

            for build in builders:                                      # Isolate each tab: one failure must not kill the suite
                try:
                    build()
                except Exception as e:
                    print(f"Visualiser tab '{build.__name__}' failed: {e}")

            if 0 <= prev < self.tabs.count():                           # Keep the user on the same tab
                self.tabs.setCurrentIndex(prev)
        finally:
            self._building = False

        # If slider moves arrived during the build, rebuild once for the latest position.
        if self._pending is not None and getattr(self, 'slider_fac', None) is not None:
            self._pending = None
            f_val = self.factors_list[self.slider_fac.value()]
            p_val = self.fpeaks_list[self.slider_fpk.value()]
            idx = self.runs.get((f_val, p_val))
            if idx is not None and (f_val, p_val) != (self.panel.current_factors, self.panel.current_fpeak):
                self.panel.combo_fpeak.setCurrentIndex(idx)
                self._reload_and_build()

    def _add_save_button(self, layout, fig, default_name):
        btn_save = QPushButton("Save Figure")                            
        btn_save.clicked.connect(lambda: self._save_figure(fig, default_name)) 
        layout.addWidget(btn_save)                                       
        
    def _save_figure(self, fig, default_name):
        path, _ = QFileDialog.getSaveFileName(self, "Save Figure", default_name, "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)") 
        if path: fig.savefig(path, dpi=300, bbox_inches='tight')         

    def _get_mean_se_pnsd(self, factor_idx):
        diams = self.panel.diams
        raw_vals = self.panel.f_matrix.iloc[:, factor_idx].values
        if self.panel.chk_wide_pmf.isChecked():
            n_bins = len(diams)
            n_hours = len(raw_vals) // n_bins
            if n_hours >= 1 and n_hours * n_bins == len(raw_vals):       # Only reshape when it divides cleanly
                reshaped = raw_vals.reshape(n_hours, n_bins)
                return np.mean(reshaped, axis=0), np.std(reshaped, axis=0) / np.sqrt(n_hours)
            return np.full(n_bins, np.nan), np.zeros(n_bins)             # Mismatch: return diam-length NaNs
        return raw_vals, np.zeros_like(raw_vals)

    def _build_size_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab)                       
        fig = Figure(figsize=(8, 6)); canvas = FigureCanvasQTAgg(fig)
        ax1 = fig.add_subplot(111); ax2 = ax1.twinx()                    
        diams = self.panel.diams
        mass_factor = (np.pi / 6) * (diams ** 3) * 1e-9                  
        for i in range(self.panel.current_factors):
            name = self.panel._get_factor_name(i)
            mean_n, se_n = self._get_mean_se_pnsd(i)
            if len(mean_n) != len(diams):                                # Skip factors that don't match the diameter axis
                continue
            line, = ax1.plot(diams, mean_n, label=name, lw=2.5)
            ax1.fill_between(diams, mean_n - se_n, mean_n + se_n, color=line.get_color(), alpha=0.2)
            ax2.plot(diams, mean_n * mass_factor, color=line.get_color(), lw=1.5, alpha=0.4, ls='--')
        ax1.set_xscale('log'); ax1.set_xlabel('Mobility Diameter (nm)', fontsize=14) 
        ax1.set_ylabel(r'dN/dlogD$_p$ (cm$^{-3}$)', fontsize=14)
        ax2.set_ylabel(r'dM/dlogD$_p$ ($\mu$g m$^{-3}$)', fontsize=14)
        ax1.legend(loc='upper right'); ax1.grid(True, which="both", ls="--", alpha=0.3)
        fig.tight_layout(); layout.addWidget(canvas)
        self._add_save_button(layout, fig, "PNSD_Profiles.png")
        self.tabs.addTab(tab, "Size & Mass Dists")

    def _build_time_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        ctrl = QHBoxLayout(); self.cb_agg = QComboBox()
        self.cb_agg.addItems(["None (Raw)", "Hourly", "Daily", "Weekly", "Monthly"])
        self.cb_agg.currentIndexChanged.connect(lambda: self._update_time_plot())
        ctrl.addWidget(QLabel("Aggregate:")); ctrl.addWidget(self.cb_agg); ctrl.addStretch()
        layout.addLayout(ctrl)
        self.time_fig = Figure(figsize=(10, 5)); self.time_canvas = FigureCanvasQTAgg(self.time_fig)
        self.time_ax = self.time_fig.add_subplot(111); layout.addWidget(self.time_canvas)
        self._update_time_plot(); self.tabs.addTab(tab, "Time Series")

    def _update_time_plot(self):
        df = self.g_number.copy(); agg = self.cb_agg.currentText(); df.index = pd.to_datetime(df.index)
        if "None" not in agg:
            rule = {'Hourly':'h','Daily':'D','Weekly':'W','Monthly':'ME'}[agg.split()[0]]
            df = df.resample(rule).mean()
        df = df.dropna(); self.time_ax.clear()
        x_years = (df.index - df.index[0]).total_seconds() / (365.25 * 24 * 3600) if len(df) > 1 else np.zeros(len(df))
        for i in range(self.panel.current_factors):
            name = self.panel._get_factor_name(i); y = df.iloc[:, i].values
            line, = self.time_ax.plot(df.index, y, lw=1.5); color = line.get_color()
            if len(df) > 1 and x_years[-1] > 0:
                m, c = np.polyfit(x_years, y, 1); y_p = m * x_years + c
                r2 = 1 - (np.sum((y - y_p)**2) / np.sum((y - np.mean(y))**2)) if np.var(y) > 0 else 0
                self.time_ax.plot(df.index, y_p, color=color, ls='--', alpha=0.5)
                label = f"{name}\n(m={m:.1e} yr$^{{-1}}$, R$^2$={r2:.2f})"
            else:
                label = name
            line.set_label(label)
        self.time_ax.set_ylabel(r'Particle Number (cm$^{-3}$)'); self.time_ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        self.time_fig.tight_layout(); self.time_canvas.draw()

    def _build_seasonal_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab); fig = Figure(); canvas = FigureCanvasQTAgg(fig); ax = fig.add_subplot(111)
        df = self.g_number.copy(); df.columns = [self.panel._get_factor_name(i) for i in range(self.panel.current_factors)]
        df['Month'] = df.index.month; melt = df.melt(id_vars='Month')
        grp = melt.groupby(['Month', 'variable'])['value']; mu = grp.mean().unstack(); se = grp.sem().unstack()
        for c in mu.columns:
            l, = ax.plot(mu.index, mu[c], label=c, marker='o')
            ax.fill_between(mu.index, mu[c]-se[c], mu[c]+se[c], color=l.get_color(), alpha=0.2)
        ax.set_xticks(range(1,13)); ax.set_ylabel(r'Particle Number (cm$^{-3}$)')
        ax.legend(); fig.tight_layout(); layout.addWidget(canvas); self.tabs.addTab(tab, "Seasonal")

    def _build_dow_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab); fig = Figure(); canvas = FigureCanvasQTAgg(fig); ax = fig.add_subplot(111)
        df = self.g_number.copy(); df.columns = [self.panel._get_factor_name(i) for i in range(self.panel.current_factors)]
        df['DOW'] = df.index.day_name(); days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        df['DOW'] = pd.Categorical(df['DOW'], categories=days, ordered=True); melt = df.melt(id_vars='DOW')
        grp = melt.groupby(['DOW', 'variable'])['value']; mu = grp.mean().unstack(); se = grp.sem().unstack()
        for c in mu.columns:
            l, = ax.plot(range(7), mu[c], label=c, marker='o')
            ax.fill_between(range(7), mu[c]-se[c], mu[c]+se[c], color=l.get_color(), alpha=0.2)
        ax.set_xticks(range(7)); ax.set_xticklabels(days, rotation=45); ax.set_ylabel(r'Particle Number (cm$^{-3}$)')
        ax.legend(); fig.tight_layout(); layout.addWidget(canvas); self.tabs.addTab(tab, "Day of Week")

    def _build_diurnal_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        fig = Figure(figsize=(8, 6)); canvas = FigureCanvasQTAgg(fig)
        ax = fig.add_subplot(111)

        diams = self.panel.diams
        dlogdp = np.log10(diams[1] / diams[0]) if len(diams) > 1 else 1.0

        for i in range(self.panel.current_factors):
            name = self.panel._get_factor_name(i)
            try:
                if self.panel.chk_wide_pmf.isChecked():
                    n_bins = len(diams); n_hours = len(self.panel.f_matrix) // n_bins
                    f_reshaped = self.panel.f_matrix.iloc[:, i].values.reshape(n_hours, n_bins)
                    y_vals = (np.sum(f_reshaped, axis=1) * dlogdp) * self.panel.g_matrix.iloc[:, i].mean()
                    ax.plot(np.arange(n_hours), y_vals, label=name, lw=2, marker='o')
                else:
                    df = self.g_number.iloc[:, i].copy()
                    if not isinstance(df.index, pd.DatetimeIndex):               # Ensure index is datetime
                        df.index = pd.to_datetime(df.index)                      # Convert if needed
                    diurnal = df.groupby(df.index.hour).mean()                   # Compute diurnal mean
                    ax.plot(diurnal.index, diurnal.values, label=name, lw=2, marker='o')
            except Exception as e: print(f"Diurnal plot fail: {e}")              # Prevent tab crash

        ax.set_xlabel("Hour of Day", fontsize=14)
        ax.set_ylabel("Mean Particle Number (cm-3)", fontsize=14)
        ax.set_xticks(np.arange(0, 24, 2))                                       # Set logical hour ticks
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left'); ax.grid(True, ls="--", alpha=0.4)
        fig.tight_layout(); layout.addWidget(canvas)
        self._add_save_button(layout, fig, "Diurnal_Cycle.png")
        self.tabs.addTab(tab, "Diurnal Cycle")

    def _build_mass_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab); fig = Figure(figsize=(8, 6)); canvas = FigureCanvasQTAgg(fig); ax = fig.add_subplot(111)
        diams = self.panel.diams; mass_factor = (np.pi / 6) * (diams ** 3) * 1e-9
        for i in range(self.panel.current_factors):
            name = self.panel._get_factor_name(i)
            mean_n, se_n = self._get_mean_se_pnsd(i)
            if len(mean_n) != len(diams):                                # Skip factors that don't match the diameter axis
                continue
            mean_m = mean_n * mass_factor
            ax.plot(diams, mean_m, label=name, lw=2.5)
        ax.set_xscale('log'); ax.set_xlabel('Mobility Diameter (nm)', fontsize=14)
        ax.set_ylabel(r'dM/dlogD$_p$ ($\mu$g m$^{-3}$)', fontsize=14)
        ax.legend(loc='upper right'); ax.grid(True, which="both", ls="--", alpha=0.4)
        fig.tight_layout(); layout.addWidget(canvas)
        self._add_save_button(layout, fig, "Mass_Distributions.png")
        self.tabs.addTab(tab, "Mass Distributions")
        
    def _build_resid_recon_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        fig = Figure(figsize=(10, 8)); canvas = FigureCanvasQTAgg(fig)
        ax1 = fig.add_subplot(211); ax2 = fig.add_subplot(212)
        
        try:
            X_recon = self.panel.g_matrix.values @ self.panel.f_matrix.values    # Math recon
            mat_path = os.path.join(self.panel.run_dir, "MATRIX.DAT")
            X_orig = np.array(open(mat_path).read().replace(',', ' ').split(), dtype=float).reshape(X_recon.shape)
            
            diams = self.panel.diams; n_bins = len(diams)
            
            if self.panel.chk_wide_pmf.isChecked():
                n_hours = X_recon.shape[1] // n_bins
                n_days = X_recon.shape[0]
                X_recon = X_recon.reshape(n_days * n_hours, n_bins)
                X_orig = X_orig.reshape(n_days * n_hours, n_bins)
                time_idx = pd.date_range(start=self.panel.g_matrix.index[0], periods=n_days * n_hours, freq='h')
            else:
                time_idx = self.panel.g_matrix.index
                
            mean_orig = X_orig.mean(axis=0); mean_recon = X_recon.mean(axis=0)
            ax1.plot(diams, mean_orig, label="Original Data", color='black', lw=2)
            ax1.plot(diams, mean_recon, label="PMF Reconstructed", color='red', ls='--', lw=2)
            ax1.set_xscale('log'); ax1.set_xlabel("Mobility Diameter (nm)", fontsize=12)
            ax1.set_ylabel("Mean dN/dlogDp", fontsize=12); ax1.legend(); ax1.grid(True, ls='--', alpha=0.5)
            
            total_orig = np.where(X_orig.sum(axis=1) == 0, 1e-9, X_orig.sum(axis=1))
            total_recon = X_recon.sum(axis=1)
            ratio = (total_orig - total_recon) / total_orig
            
            ax2.plot(time_idx, ratio, color='royalblue', lw=1)
            ax2.axhline(0, color='red', ls='--', lw=1.5)
            ax2.set_xlabel("Date", fontsize=12)
            ax2.set_ylabel("(Original - Reconstructed) / Original", fontsize=12)
            ax2.grid(True, ls='--', alpha=0.5)
            
        except Exception as e: ax1.text(0.5, 0.5, f"Reconstruction Error: {e}", ha='center')
        
        fig.tight_layout(); layout.addWidget(canvas)
        self._add_save_button(layout, fig, "Residual_Reconstruction.png")
        self.tabs.addTab(tab, "Residuals & Recon")

    def _build_diag_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab); fig = Figure(); canvas = FigureCanvasQTAgg(fig); ax = fig.add_subplot(111)
        path = os.path.join(self.panel.run_dir, f"ScaledResid_{self.panel.current_factors}_{self.panel.current_fpeak}.dat") 
        try:
            res = np.array(open(path).read().replace(',',' ').split(), dtype=float)
            res = res[~np.isnan(res)]
            ax.hist(res, bins=50, range=(-10,10), color='royalblue', edgecolor='black')
            ax.axvline(x=-3, color='red', ls='--'); ax.axvline(x=3, color='red', ls='--')
            ax.set_xlabel('Scaled Residual'); ax.set_ylabel('Frequency')
        except: ax.text(0.5,0.5,"Missing Data", ha='center')
        fig.tight_layout(); layout.addWidget(canvas); self.tabs.addTab(tab, "Diagnostics")

    def _build_qq_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        fig = Figure(figsize=(8, 6)); canvas = FigureCanvasQTAgg(fig)
        ax = fig.add_subplot(111)

        # Gather Q/Qexp for every computed run straight from the batch combo box.
        data = {}                                                        # fpeak -> [(factors, q), ...]
        for i in range(self.panel.combo_fpeak.count()):
            text = self.panel.combo_fpeak.itemText(i)
            fm = re.search(r"Factors:?\s*(\d+)", text, re.IGNORECASE)
            pm = re.search(r"FPEAK:?\s*([-0-9.]+)", text, re.IGNORECASE)
            qm = re.search(r"Q/Qexp:?\s*([-0-9.]+)", text, re.IGNORECASE)
            if not (fm and pm):
                continue
            q = float(qm.group(1)) if qm else np.nan
            data.setdefault(float(pm.group(1)), []).append((int(fm.group(1)), q))

        if not data:
            ax.text(0.5, 0.5, "Q/Qexp scree needs a completed PMF batch\n(no runs found in this session).",
                    ha='center', va='center')
        else:
            for fpeak in sorted(data):
                pts = sorted(data[fpeak])
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker='o', lw=1.8, label=f"FPEAK={fpeak:g}")
            ax.axhline(1.0, color='red', ls='--', lw=1.2, alpha=0.7, label='Q/Qexp = 1')

            active_q = next((q for fac, q in data.get(self.panel.current_fpeak, [])
                             if fac == self.panel.current_factors), None)
            if active_q is not None and np.isfinite(active_q):
                ax.scatter([self.panel.current_factors], [active_q], s=180, facecolors='none',
                           edgecolors='black', linewidths=2, zorder=5, label='Active run')

            ax.set_xticks(sorted({fac for pts in data.values() for fac, _ in pts}))
            ax.set_xlabel("Number of Factors", fontsize=13)
            ax.set_ylabel("Q/Qexp", fontsize=13)
            ax.legend(fontsize=9); ax.grid(True, ls='--', alpha=0.4)

        fig.tight_layout(); layout.addWidget(canvas)
        self._add_save_button(layout, fig, "Q_Qexp_Scree.png")
        self.tabs.addTab(tab, "Q/Qexp Scree")

    def _build_summary_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        diams = self.panel.diams
        dlogdp = np.log10(diams[1] / diams[0]) if len(diams) > 1 else 1.0
        mass_factor = (np.pi / 6) * (diams ** 3) * 1e-9

        # Real mean dN/dlogDp contributed by each factor = mean(raw G) * F profile.
        profiles, number_concs = [], []
        for i in range(self.panel.current_factors):
            f_mean, _ = self._get_mean_se_pnsd(i)
            real = f_mean * self.panel.g_matrix.iloc[:, i].mean()
            profiles.append(real)
            number_concs.append(float(np.sum(real) * dlogdp))
        total_n = np.sum(number_concs)

        rows = []
        for i in range(self.panel.current_factors):
            real = profiles[i]
            w = np.clip(real, 0, None)
            n_conc = number_concs[i]
            pct = 100.0 * n_conc / total_n if total_n else np.nan
            modal = diams[int(np.argmax(real))] if len(real) else np.nan
            gmd = np.exp(np.sum(w * np.log(diams)) / np.sum(w)) if np.sum(w) > 0 else np.nan
            mass = float(np.sum(real * mass_factor) * dlogdp)
            rows.append([self.panel._get_factor_name(i), n_conc, pct, modal, gmd, mass])

        headers = ["Factor", "Mean N (cm⁻³)", "% of N", "Modal Dp (nm)", "GMD (nm)", "Mass (µg m⁻³)"]
        self.summary_df = pd.DataFrame([r[1:] for r in rows],
                                       index=[r[0] for r in rows], columns=headers[1:])

        table = QTableWidget(len(rows), len(headers))
        table.setHorizontalHeaderLabels(headers)
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                table.setItem(r, c, QTableWidgetItem(val if c == 0 else f"{val:.3g}"))
        table.resizeColumnsToContents()
        layout.addWidget(table)

        btn = QPushButton("Export Summary to CSV")
        btn.clicked.connect(self._export_summary)
        layout.addWidget(btn)
        self.tabs.addTab(tab, "Factor Summary")

    def _export_summary(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Factor Summary", "Factor_Summary.csv", "CSV (*.csv)")
        if path:
            self.summary_df.to_csv(path, index_label="Factor")

    def _build_gspace_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        ctrl = QHBoxLayout()
        self.gs_x = QComboBox(); self.gs_y = QComboBox()
        for i in range(self.panel.current_factors):
            n = self.panel._get_factor_name(i)
            self.gs_x.addItem(n, i); self.gs_y.addItem(n, i)
        if self.panel.current_factors > 1:
            self.gs_y.setCurrentIndex(1)
        self.gs_norm = QCheckBox("Normalise to max"); self.gs_norm.setChecked(True)
        ctrl.addWidget(QLabel("X factor:")); ctrl.addWidget(self.gs_x)
        ctrl.addWidget(QLabel("Y factor:")); ctrl.addWidget(self.gs_y)
        ctrl.addWidget(self.gs_norm); ctrl.addStretch()
        layout.addLayout(ctrl)

        self.gs_fig = Figure(figsize=(7, 6)); self.gs_canvas = FigureCanvasQTAgg(self.gs_fig)
        layout.addWidget(self.gs_canvas)
        trigger = lambda: self._update_gspace_plot()
        self.gs_x.currentIndexChanged.connect(trigger)
        self.gs_y.currentIndexChanged.connect(trigger)
        self.gs_norm.stateChanged.connect(trigger)
        self._update_gspace_plot()
        self._add_save_button(layout, self.gs_fig, "G_Space_Scatter.png")
        self.tabs.addTab(tab, "G-Space Scatter")

    def _update_gspace_plot(self):
        self.gs_fig.clear()
        ax = self.gs_fig.add_subplot(111)
        ix = self.gs_x.currentData(); iy = self.gs_y.currentData()
        x = self.panel.g_matrix.iloc[:, ix].to_numpy(dtype=float)
        y = self.panel.g_matrix.iloc[:, iy].to_numpy(dtype=float)
        if self.gs_norm.isChecked():
            if x.max() > 0: x = x / x.max()
            if y.max() > 0: y = y / y.max()
        ax.scatter(x, y, s=10, alpha=0.4, edgecolors='none')
        r = np.corrcoef(x, y)[0, 1] if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0 else np.nan
        lim = max(x.max(), y.max()) if len(x) else 1.0
        ax.plot([0, lim], [0, lim], color='red', ls='--', lw=1, alpha=0.6, label='1:1')
        ax.set_xlabel(f"{self.panel._get_factor_name(ix)} contribution", fontsize=12)
        ax.set_ylabel(f"{self.panel._get_factor_name(iy)} contribution", fontsize=12)
        ax.set_title(f"Edge plot   (Pearson r = {r:.3f})", fontsize=12)
        ax.legend(fontsize=9); ax.grid(True, ls='--', alpha=0.3)
        self.gs_fig.tight_layout(); self.gs_canvas.draw()

    def _build_tracer_tab(self):
        if self.panel.tracer_df is None:                                 # Only present once tracers are loaded
            return
        tab = QWidget(); layout = QVBoxLayout(tab)
        self._tracer_aligned = align_to_index(self.panel.tracer_df, self.panel.g_matrix.index)

        ctrl = QHBoxLayout()
        self.tr_method = QComboBox(); self.tr_method.addItems(["Pearson", "Spearman"])
        self.tr_factor = QComboBox()
        for i in range(self.panel.current_factors):
            self.tr_factor.addItem(self.panel._get_factor_name(i), i)
        self.tr_col = QComboBox(); self.tr_col.addItems([str(c) for c in self._tracer_aligned.columns])
        ctrl.addWidget(QLabel("Method:")); ctrl.addWidget(self.tr_method)
        ctrl.addWidget(QLabel("Factor:")); ctrl.addWidget(self.tr_factor)
        ctrl.addWidget(QLabel("Tracer:")); ctrl.addWidget(self.tr_col); ctrl.addStretch()
        layout.addLayout(ctrl)

        self.tr_fig = Figure(figsize=(11, 8)); self.tr_canvas = FigureCanvasQTAgg(self.tr_fig)
        layout.addWidget(self.tr_canvas)
        trigger = lambda: self._update_tracer_plot()
        for w in [self.tr_method, self.tr_factor, self.tr_col]:
            w.currentIndexChanged.connect(trigger)
        self._update_tracer_plot()
        self._add_save_button(layout, self.tr_fig, "Tracer_Correlation.png")
        self.tabs.addTab(tab, "Tracer Correlation")

    def _update_tracer_plot(self):
        self.tr_fig.clear()
        g = self.g_number                                               # Scaled particle number per factor
        tr = self._tracer_aligned
        method = self.tr_method.currentText().lower()
        factor_names = [self.panel._get_factor_name(i) for i in range(self.panel.current_factors)]
        cols = [str(c) for c in tr.columns]

        corr = np.full((len(factor_names), len(cols)), np.nan)
        for i in range(len(factor_names)):
            gi = pd.Series(g.iloc[:, i].to_numpy(dtype=float))
            for j, c in enumerate(tr.columns):
                tj = pd.Series(pd.to_numeric(tr[c], errors='coerce').to_numpy(dtype=float))
                mask = gi.notna() & tj.notna()
                if mask.sum() > 2 and gi[mask].std() > 0 and tj[mask].std() > 0:
                    corr[i, j] = gi[mask].corr(tj[mask], method=method)

        grid = self.tr_fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])
        ax_h = self.tr_fig.add_subplot(grid[0, :])
        im = ax_h.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax_h.set_xticks(range(len(cols))); ax_h.set_xticklabels(cols, rotation=45, ha='right')
        ax_h.set_yticks(range(len(factor_names))); ax_h.set_yticklabels(factor_names)
        for i in range(len(factor_names)):
            for j in range(len(cols)):
                if np.isfinite(corr[i, j]):
                    ax_h.text(j, i, f"{corr[i, j]:.2f}", ha='center', va='center', fontsize=8,
                              color='white' if abs(corr[i, j]) > 0.5 else 'black')
        self.tr_fig.colorbar(im, ax=ax_h, label=f"{method.title()} r", fraction=0.025)
        ax_h.set_title("Factor (scaled G) vs tracer correlation")

        fi = self.tr_factor.currentData(); col = self.tr_col.currentText()
        x = pd.Series(pd.to_numeric(tr[col], errors='coerce').to_numpy(dtype=float))
        y = pd.Series(g.iloc[:, fi].to_numpy(dtype=float))
        mask = x.notna() & y.notna()

        ax_s = self.tr_fig.add_subplot(grid[1, 0])
        if mask.sum() > 2:
            xs = x[mask].to_numpy(); ys = y[mask].to_numpy()
            ax_s.scatter(xs, ys, s=8, alpha=0.4, edgecolors='none')
            if np.std(xs) > 0:
                m, b = np.polyfit(xs, ys, 1)
                xr = np.array([xs.min(), xs.max()])
                ax_s.plot(xr, m * xr + b, color='red', lw=1.5)
                r = np.corrcoef(xs, ys)[0, 1]
                ax_s.set_title(f"{self.panel._get_factor_name(fi)} vs {col}  (r={r:.2f}, R²={r**2:.2f})", fontsize=10)
        ax_s.set_xlabel(col); ax_s.set_ylabel(f"{self.panel._get_factor_name(fi)} (cm⁻³)")
        ax_s.grid(True, ls='--', alpha=0.3)

        ax_t = self.tr_fig.add_subplot(grid[1, 1])
        idx = self.panel.g_matrix.index
        ax_t.plot(idx, y.to_numpy(), color='steelblue', lw=1)
        ax_t2 = ax_t.twinx()
        ax_t2.plot(idx, x.to_numpy(), color='darkorange', lw=1, alpha=0.7)
        ax_t.set_ylabel(self.panel._get_factor_name(fi), color='steelblue', fontsize=9)
        ax_t2.set_ylabel(col, color='darkorange', fontsize=9)
        ax_t.tick_params(axis='x', labelrotation=30, labelsize=8)
        ax_t.set_title("Time-series overlay", fontsize=10)

        self.tr_fig.tight_layout(); self.tr_canvas.draw()

    def _build_polar_tab(self):
        if self.panel.tracer_df is None:                                 # Needs WD (and ideally WS) among tracers
            return
        self._polar_aligned = align_to_index(self.panel.tracer_df, self.panel.g_matrix.index)
        cols = [str(c) for c in self._polar_aligned.columns]

        tab = QWidget(); layout = QVBoxLayout(tab)
        ctrl = QGridLayout()
        self.pl_factor = QComboBox()
        for i in range(self.panel.current_factors):
            self.pl_factor.addItem(self.panel._get_factor_name(i), i)
        self.pl_wd = QComboBox(); self.pl_wd.addItems(cols)
        self.pl_ws = QComboBox(); self.pl_ws.addItem("(none)"); self.pl_ws.addItems(cols)
        low = {c.lower(): c for c in cols}
        for cand in ["wd", "wind_dir", "wind_direction", "wdir", "direction"]:
            if cand in low: self.pl_wd.setCurrentText(low[cand]); break
        for cand in ["ws", "wind_speed", "windspeed", "speed"]:
            if cand in low: self.pl_ws.setCurrentText(low[cand]); break
        self.pl_pct = QSpinBox(); self.pl_pct.setRange(50, 99); self.pl_pct.setValue(75)
        ctrl.addWidget(QLabel("Factor:"), 0, 0); ctrl.addWidget(self.pl_factor, 0, 1)
        ctrl.addWidget(QLabel("WD col:"), 0, 2); ctrl.addWidget(self.pl_wd, 0, 3)
        ctrl.addWidget(QLabel("WS col:"), 1, 2); ctrl.addWidget(self.pl_ws, 1, 3)
        ctrl.addWidget(QLabel("CBPF percentile:"), 1, 0); ctrl.addWidget(self.pl_pct, 1, 1)
        layout.addLayout(ctrl)

        self.pl_fig = Figure(figsize=(12, 5)); self.pl_canvas = FigureCanvasQTAgg(self.pl_fig)
        layout.addWidget(self.pl_canvas)
        trigger = lambda: self._update_polar_plot()
        for w in [self.pl_factor, self.pl_wd, self.pl_ws]:
            w.currentIndexChanged.connect(trigger)
        self.pl_pct.valueChanged.connect(trigger)
        self._update_polar_plot()
        self._add_save_button(layout, self.pl_fig, "Factor_Polar.png")
        self.tabs.addTab(tab, "Polar / CBPF")

    def _update_polar_plot(self):
        self.pl_fig.clear()
        fi = self.pl_factor.currentData()
        name = self.panel._get_factor_name(fi)
        contrib = self.g_number.iloc[:, fi].to_numpy(dtype=float)
        wd = pd.to_numeric(self._polar_aligned[self.pl_wd.currentText()], errors='coerce').to_numpy(dtype=float)
        ws_col = self.pl_ws.currentText()
        ws = (pd.to_numeric(self._polar_aligned[ws_col], errors='coerce').to_numpy(dtype=float)
              if ws_col in self._polar_aligned.columns else None)

        n_sec = 16
        edges = np.linspace(0, 360, n_sec + 1)
        centers = np.deg2rad((edges[:-1] + edges[1:]) / 2)
        wd_mod = np.mod(wd, 360.0)

        # Left: mean factor contribution by wind sector (pollutant rose).
        ax1 = self.pl_fig.add_subplot(121, projection='polar')
        means = []
        for k in range(n_sec):
            m = (wd_mod >= edges[k]) & (wd_mod < edges[k + 1]) & np.isfinite(contrib)
            means.append(float(np.nanmean(contrib[m])) if m.any() else 0.0)
        peak = max(means) or 1.0
        ax1.bar(centers, means, width=np.deg2rad(360 / n_sec),
                color=plt.cm.viridis(np.array(means) / peak), edgecolor='k', linewidth=0.5, alpha=0.85)
        ax1.set_theta_zero_location("N"); ax1.set_theta_direction(-1)
        ax1.set_title(f"{name}: mean contribution by wind sector", fontsize=10)

        # Right: CBPF (WD x WS grid) where WS is available, else CPF by WD only.
        finite = np.isfinite(contrib)
        thr = np.nanpercentile(contrib[finite], self.pl_pct.value()) if finite.any() else np.nan
        ax2 = self.pl_fig.add_subplot(122, projection='polar')
        if ws is not None and np.isfinite(ws).any():
            n_rings = 6
            ws_max = np.nanpercentile(ws[np.isfinite(ws)], 95)
            ring_edges = np.linspace(0, ws_max if ws_max > 0 else 1.0, n_rings + 1)
            prob = np.full((n_rings, n_sec), np.nan)
            for a in range(n_sec):
                for rr in range(n_rings):
                    m = ((wd_mod >= edges[a]) & (wd_mod < edges[a + 1]) &
                         (ws >= ring_edges[rr]) & (ws < ring_edges[rr + 1]) & finite)
                    if m.sum() >= 5:
                        prob[rr, a] = np.mean(contrib[m] > thr)
            T, R = np.meshgrid(np.deg2rad(edges), ring_edges)
            mesh = ax2.pcolormesh(T, R, prob, cmap='turbo', vmin=0, vmax=1, shading='flat')
            self.pl_fig.colorbar(mesh, ax=ax2, label=f"P(> P{self.pl_pct.value()})", fraction=0.045)
            ax2.set_title(f"CBPF: {name}  (WD x WS)", fontsize=10)
        else:
            cpf = []
            for k in range(n_sec):
                m = (wd_mod >= edges[k]) & (wd_mod < edges[k + 1]) & finite
                cpf.append(float(np.mean(contrib[m] > thr)) if m.sum() >= 5 else 0.0)
            ax2.bar(centers, cpf, width=np.deg2rad(360 / n_sec), color='indianred', edgecolor='k', linewidth=0.5)
            ax2.set_title(f"CPF: P(contribution > P{self.pl_pct.value()}) by WD", fontsize=10)
        ax2.set_theta_zero_location("N"); ax2.set_theta_direction(-1)

        self.pl_fig.tight_layout(); self.pl_canvas.draw()

    def _build_nucsplit_tab(self):
        # Split the Nucleation factor into Traffic and Photochemical sources after
        # Rodriguez & Cuevas (2007), using NOx as a traffic proxy scaled per day at the morning peak.
        if self.panel.tracer_df is None:
            return
        tab = QWidget(); layout = QVBoxLayout(tab)
        if self.panel.chk_wide_pmf.isChecked():
            msg = QLabel("Nucleation splitting needs continuous hourly data.\nDisable WidePMF mode to use this tool.")
            msg.setStyleSheet("color: gray; font-style: italic;")
            layout.addWidget(msg); self.tabs.addTab(tab, "Nucleation Split"); return

        self._nuc_aligned = align_to_index(self.panel.tracer_df, self.panel.g_matrix.index)
        cols = [str(c) for c in self._nuc_aligned.columns]

        ctrl = QGridLayout()
        self.ns_factor = QComboBox()
        for i in range(self.panel.current_factors):
            self.ns_factor.addItem(self.panel._get_factor_name(i), i)
        self.ns_nox = QComboBox(); self.ns_nox.addItems(cols)
        self.ns_sr = QComboBox(); self.ns_sr.addItems(cols)
        for c in cols:                                                    # Guess NOx and solar-radiation columns
            if "nox" in c.lower():
                self.ns_nox.setCurrentText(c); break
        for c in cols:
            cl = c.lower()
            if "solar" in cl or "rad" in cl or cl == "sr" or "global" in cl:
                self.ns_sr.setCurrentText(c); break

        self.ns_peak = QSpinBox(); self.ns_peak.setRange(0, 23); self.ns_peak.setValue(8)
        self.ns_daystart = QSpinBox(); self.ns_daystart.setRange(0, 23); self.ns_daystart.setValue(8)
        self.ns_dayend = QSpinBox(); self.ns_dayend.setRange(0, 23); self.ns_dayend.setValue(21)
        self.ns_thr = QDoubleSpinBox(); self.ns_thr.setRange(0, 2000); self.ns_thr.setValue(10.0)
        self.ns_lowsr = QComboBox(); self.ns_lowsr.addItems(["Assign all to Traffic", "Apply NOx formula anyway"])

        ctrl.addWidget(QLabel("Nucleation factor:"), 0, 0); ctrl.addWidget(self.ns_factor, 0, 1)
        ctrl.addWidget(QLabel("NOx col:"), 0, 2); ctrl.addWidget(self.ns_nox, 0, 3)
        ctrl.addWidget(QLabel("Solar rad col:"), 0, 4); ctrl.addWidget(self.ns_sr, 0, 5)
        ctrl.addWidget(QLabel("Peak hour:"), 1, 0); ctrl.addWidget(self.ns_peak, 1, 1)
        ctrl.addWidget(QLabel("Day window:"), 1, 2)
        day_box = QHBoxLayout(); day_box.addWidget(self.ns_daystart); day_box.addWidget(QLabel("to")); day_box.addWidget(self.ns_dayend)
        ctrl.addLayout(day_box, 1, 3)
        ctrl.addWidget(QLabel("SR threshold (W m⁻²):"), 1, 4); ctrl.addWidget(self.ns_thr, 1, 5)
        ctrl.addWidget(QLabel("Daytime when SR ≤ threshold:"), 2, 0); ctrl.addWidget(self.ns_lowsr, 2, 1, 1, 3)
        layout.addLayout(ctrl)

        self.ns_status = QLabel(""); layout.addWidget(self.ns_status)
        self.ns_fig = Figure(figsize=(12, 5)); self.ns_canvas = FigureCanvasQTAgg(self.ns_fig)
        layout.addWidget(self.ns_canvas)

        trigger = lambda: self._update_nucsplit()
        for w in [self.ns_factor, self.ns_nox, self.ns_sr, self.ns_lowsr]:
            w.currentIndexChanged.connect(trigger)
        for w in [self.ns_peak, self.ns_daystart, self.ns_dayend]:
            w.valueChanged.connect(trigger)
        self.ns_thr.valueChanged.connect(trigger)

        btn = QPushButton("Export Sources to CSV"); btn.clicked.connect(self._export_nucsplit)
        layout.addWidget(btn)
        self._add_save_button(layout, self.ns_fig, "Nucleation_Split.png")
        self._update_nucsplit()
        self.tabs.addTab(tab, "Nucleation Split")

    def _compute_nuc_split(self):
        fi = self.ns_factor.currentData()
        idx = pd.DatetimeIndex(self.panel.g_matrix.index)
        nuc = self.g_number.iloc[:, fi].to_numpy(dtype=float)            # Scaled particle number of the factor
        nox = pd.to_numeric(self._nuc_aligned[self.ns_nox.currentText()], errors='coerce').to_numpy(dtype=float)
        sr = pd.to_numeric(self._nuc_aligned[self.ns_sr.currentText()], errors='coerce').to_numpy(dtype=float)

        work = pd.DataFrame({'Nuc': nuc, 'NOx': nox, 'SR': sr}, index=idx)
        work['date'] = work.index.normalize()                           # Hours taken from the PMF timeline
        work['hour'] = work.index.hour

        # Day-specific scale factor kd = Nucleation / NOx at the morning peak hour.
        peak = work[work['hour'] == self.ns_peak.value()].copy()
        with np.errstate(divide='ignore', invalid='ignore'):
            peak['kd'] = (peak['Nuc'] / peak['NOx']).replace([np.inf, -np.inf], np.nan)
        work['kd'] = work['date'].map(peak.groupby('date')['kd'].mean())

        day_mask = (work['hour'] >= self.ns_daystart.value()) & (work['hour'] <= self.ns_dayend.value())
        sr_high = work['SR'] > self.ns_thr.value()
        low_sr_all_traffic = (self.ns_lowsr.currentIndex() == 0)
        formula_mask = (day_mask & sr_high) if low_sr_all_traffic else day_mask
        formula_mask = formula_mask & work['kd'].notna()                # Fall back to all-traffic when kd is undefined

        traffic = work['Nuc'].copy()                                    # Default: all nucleation assigned to traffic
        traffic[formula_mask] = work['kd'][formula_mask] * work['NOx'][formula_mask]
        traffic = np.minimum(np.clip(traffic.to_numpy(dtype=float), 0, None), work['Nuc'].to_numpy(dtype=float))
        photo = work['Nuc'].to_numpy(dtype=float) - traffic

        return pd.DataFrame({'Nucleation': work['Nuc'].to_numpy(dtype=float),
                             'Traffic_nucleation': traffic, 'Photonucleation': photo}, index=idx)

    def _update_nucsplit(self):
        self.ns_fig.clear()
        res = self._compute_nuc_split()
        self._nucsplit_df = res

        tot_t = np.nansum(res['Traffic_nucleation']); tot_p = np.nansum(res['Photonucleation'])
        tot = tot_t + tot_p
        if tot > 0:
            self.ns_status.setText(f"Mean split over record: Traffic {100 * tot_t / tot:.1f}%, "
                                   f"Photochemical {100 * tot_p / tot:.1f}%")

        ax1 = self.ns_fig.add_subplot(121)
        dm = res.groupby(res.index.hour).mean(numeric_only=True)
        ax1.stackplot(dm.index, dm['Traffic_nucleation'], dm['Photonucleation'],
                      labels=['Traffic nucleation', 'Photonucleation'], colors=['#555555', 'gold'], alpha=0.85)
        ax1.plot(dm.index, dm['Nucleation'], color='black', lw=1.5, ls='--', label='Total Nucleation')
        ax1.set_xlabel("Hour of day"); ax1.set_ylabel("Particle number (cm⁻³)")
        ax1.set_xticks(range(0, 24, 3)); ax1.legend(fontsize=8); ax1.grid(True, ls='--', alpha=0.3)
        ax1.set_title("Mean diurnal split", fontsize=10)

        ax2 = self.ns_fig.add_subplot(122)
        ax2.stackplot(res.index, res['Traffic_nucleation'].fillna(0), res['Photonucleation'].fillna(0),
                      labels=['Traffic nucleation', 'Photonucleation'], colors=['#555555', 'gold'], alpha=0.8)
        ax2.set_ylabel("Particle number (cm⁻³)"); ax2.set_title("Source time series", fontsize=10)
        ax2.tick_params(axis='x', labelrotation=30, labelsize=8)

        self.ns_fig.tight_layout(); self.ns_canvas.draw()

    def _export_nucsplit(self):
        if getattr(self, '_nucsplit_df', None) is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Nucleation Sources", "Nucleation_Sources.csv", "CSV (*.csv)")
        if path:
            self._nucsplit_df.to_csv(path, index_label="datetime")

    def _build_wide_profiles_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        ctrl = QHBoxLayout()
        self.spin_vmin_p = QDoubleSpinBox(); self.spin_vmin_p.setRange(0.01, 1e6); self.spin_vmin_p.setValue(1.0)
        self.spin_vmax_p = QDoubleSpinBox(); self.spin_vmax_p.setRange(1.0, 1e9); self.spin_vmax_p.setValue(5000.0)
        ctrl.addWidget(QLabel("Vmin:")); ctrl.addWidget(self.spin_vmin_p); ctrl.addWidget(QLabel("Vmax:")); ctrl.addWidget(self.spin_vmax_p); ctrl.addStretch()
        layout.addLayout(ctrl)
        
        self.prof_fig = Figure(figsize=(10, 8)); self.prof_canvas = FigureCanvasQTAgg(self.prof_fig)
        layout.addWidget(self.prof_canvas)
        
        self.spin_vmin_p.valueChanged.connect(self._update_wide_profiles_plot)
        self.spin_vmax_p.valueChanged.connect(self._update_wide_profiles_plot)
        self._update_wide_profiles_plot()
        self.tabs.addTab(tab, "WidePMF Diurnals")
        
    def _update_wide_profiles_plot(self):
        try:
            self.prof_fig.clear()
            n_fac = self.panel.current_factors
            cols = 2
            rows = int(np.ceil(n_fac / cols))
            
            axes = self.prof_fig.subplots(rows, cols, sharex=True, sharey=True)
            if n_fac == 1: axes = np.array([axes])
            axes = axes.flatten()
            
            diams = self.panel.diams
            n_h = len(self.panel.f_matrix) // len(diams)
            norm = mpl.colors.LogNorm(vmin=self.spin_vmin_p.value(), vmax=self.spin_vmax_p.value())
            
            for i in range(n_fac):
                ax = axes[i]
                data = self.panel.f_matrix.iloc[:, i].values.reshape(n_h, len(diams))
                mesh = ax.pcolormesh(np.arange(n_h), diams, data.T, cmap='turbo', norm=norm, shading='nearest')
                ax.set_yscale('log')
                ax.set_title(self.panel._get_factor_name(i), fontweight='bold')
                ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
                ax.set_yticks([10, 20, 50, 100, 200, 500])
                
                if i >= len(axes) - cols: ax.set_xlabel("Hour")
                if i % cols == 0: ax.set_ylabel(r"D$_p$ (nm)")
                
            for i in range(n_fac, len(axes)):
                axes[i].axis('off')
                
            self.prof_fig.subplots_adjust(right=0.88, wspace=0.1, hspace=0.3)
            cbar_ax = self.prof_fig.add_axes([0.90, 0.15, 0.02, 0.7])
            cbar = self.prof_fig.colorbar(mesh, cax=cbar_ax, label="dN/dlogDp")
            cbar.formatter = mpl.ticker.LogFormatterMathtext()               # Log Ticks Formatting
            cbar.update_ticks()
            
            self.prof_canvas.draw()
        except Exception as e: print(f"Profile Error: {e}")

    def _build_widepmf_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        ctrl = QGridLayout()
        self.cb_d1 = QComboBox(); self.cb_d2 = QComboBox()
        self.pal1 = QComboBox(); self.pal2 = QComboBox()
        palettes = ['turbo', 'jet', 'viridis', 'plasma', 'inferno', 'magma', 'GnBu', 'YlOrRd']
        self.pal1.addItems(palettes); self.pal2.addItems(palettes)
        for i in range(self.panel.current_factors):
            n = self.panel._get_factor_name(i); self.cb_d1.addItem(n, i); self.cb_d2.addItem(n, i)
        self.svmin = QDoubleSpinBox(); self.svmin.setRange(0.01, 1e5); self.svmin.setValue(1.0)
        self.svmax = QDoubleSpinBox(); self.svmax.setRange(1.0, 1e7); self.svmax.setValue(5000.0)
        
        ctrl.addWidget(QLabel("Factor 1:"),0,0); ctrl.addWidget(self.cb_d1,0,1); ctrl.addWidget(QLabel("Palette:"),0,2); ctrl.addWidget(self.pal1,0,3)
        ctrl.addWidget(QLabel("Factor 2:"),1,0); ctrl.addWidget(self.cb_d2,1,1); ctrl.addWidget(QLabel("Palette:"),1,2); ctrl.addWidget(self.pal2,1,3)
        ctrl.addWidget(QLabel("Vmin:"),0,4); ctrl.addWidget(self.svmin,0,5); ctrl.addWidget(QLabel("Vmax:"),1,4); ctrl.addWidget(self.svmax,1,5)
        layout.addLayout(ctrl)
        
        self.wide_fig = Figure(figsize=(12, 6)); self.wide_canvas = FigureCanvasQTAgg(self.wide_fig); layout.addWidget(self.wide_canvas)
        trigger = lambda: self._update_wide_plot()
        for w in [self.cb_d1, self.cb_d2, self.pal1, self.pal2, self.svmin, self.svmax]: w.currentIndexChanged.connect(trigger) if isinstance(w, QComboBox) else w.valueChanged.connect(trigger)
        self._update_wide_plot(); self.tabs.addTab(tab, "WidePMF 48h Combiner")

    def _update_wide_plot(self):
        try:
            self.wide_fig.clear()
            self.wide_fig.subplots_adjust(wspace=0)                              
            ax1 = self.wide_fig.add_subplot(121)
            ax2 = self.wide_fig.add_subplot(122, sharey=ax1) 
            
            diams = self.panel.diams; n_h = len(self.panel.f_matrix) // len(diams)
            norm = mpl.colors.LogNorm(vmin=self.svmin.value(), vmax=self.svmax.value())
            
            configs = [
                (ax1, self.cb_d1.currentData(), self.pal1.currentText(), True),  
                (ax2, self.cb_d2.currentData(), self.pal2.currentText(), False)  
            ]
            
            for ax, idx, pal, is_left in configs:
                data = self.panel.f_matrix.iloc[:, idx].values.reshape(n_h, len(diams))
                mesh = ax.pcolormesh(np.arange(n_h), diams, data.T, cmap=pal, norm=norm, shading='nearest')
                ax.set_yscale('log')
                ax.set_title(self.panel._get_factor_name(idx), fontweight='bold')
                ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
                ax.set_yticks([10, 20, 50, 100, 200, 500])
                ax.set_xlabel("Hour")
                
                if is_left:
                    ax.set_ylabel(r"D$_p$ (nm)", fontsize=12)
                else:
                    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
                    ax.set_xticks(np.arange(0, n_h + 1, 6))
                    ax.set_xticklabels([str(h + n_h) for h in np.arange(0, n_h + 1, 6)])

            cbar_ax = self.wide_fig.add_axes([0.92, 0.15, 0.02, 0.7])            
            cbar = self.wide_fig.colorbar(mesh, cax=cbar_ax, label="dN/dlogDp")
            cbar.formatter = mpl.ticker.LogFormatterMathtext()                   # Log Ticks Formatting
            cbar.update_ticks()
            
            self.wide_fig.subplots_adjust(right=0.9)
            self.wide_canvas.draw()
            
        except Exception as e: print(f"Combiner Error: {e}")

class PMFPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)                                         
        
        self.batch_diams = None
        self.df = None                                                   
        self.dates = None                                                
        self.diams = None                                                
        self.f_matrix = None
        self.g_matrix = None
        self.tracer_df = None                                            # External gas/met tracers (datetime-indexed)

        self.settings = QSettings("PyNSD", "PMF_Config")
        self.pmf_exe_path = self.settings.value("exe_path", "")          
        self.pmf_key_path = self.settings.value("key_path", "")          
        
        self.working_dir = os.path.dirname(self.pmf_exe_path) if self.pmf_exe_path else "" 
        self.run_dir = os.path.join(self.working_dir, "latest_output") if self.working_dir else "" 
        
        self.current_factors = 0                                         
        self.current_fpeak = 0.0                                         
        self.factor_names = {}                                           
        
        self._setup_ui()                                                 
        
    def load_data(self, data_file):
        self.df = data_file.df.copy()                                    
        self.df.index = pd.to_datetime(self.df.index)                    
        self.dates = self.df.index                                       
        self.diams = np.array(data_file.diameters)

        self.combo_fpeak.clear()                                                 
        self.f_matrix = None
        self.g_matrix = None
        self.batch_diams = None                       
    
    def prepare_run_directory(self):
        if not self.run_dir: return                                      
        if not os.path.exists(self.run_dir):                             
            os.makedirs(self.run_dir)                                    
        else:                                                            
            for f in os.listdir(self.run_dir):                           
                f_path = os.path.join(self.run_dir, f)                   
                if os.path.isfile(f_path): os.remove(f_path)             
        if self.pmf_key_path:                                            
            shutil.copy(self.pmf_key_path, os.path.join(self.run_dir, "pmf2key.key")) 
            
    def write_fkey(self, factors, n_cols):
        fkey_path = os.path.join(self.run_dir, "FKEY.DAT")               
        row_str = " ".join(["0"] * n_cols)                               
        with open(fkey_path, 'w', newline='\r\n') as f:                  
            for _ in range(factors):                                     
                f.write(row_str + "\r\n")                                
        print(f"FKEY.DAT updated for {factors} factors.")                

    def get_scaled_g(self):                                              
        g_num = self.g_matrix.copy()
        diams = self.diams
        dlogdp = np.log10(diams[1] / diams[0]) if len(diams) > 1 else 1.0
        f_sums = []
        for i in range(self.current_factors):
            if self.chk_wide_pmf.isChecked():
                n_bins = len(diams)
                n_hours = len(self.f_matrix) // n_bins
                f_reshaped = self.f_matrix.iloc[:, i].values.reshape(n_hours, n_bins)
                f_sums.append((f_reshaped.sum(axis=1) * dlogdp).mean())
            else:
                f_sums.append(self.f_matrix.iloc[:, i].sum() * dlogdp)
        for i in range(self.current_factors):
            g_num.iloc[:, i] = g_num.iloc[:, i] * f_sums[i]
        return g_num

    def _get_q_ratio(self, factors, fpeak, n_rows, n_cols, is_batch=False): 
        if is_batch:
            resid_path = os.path.join(self.run_dir, "ScaledResid.dat") 
        else:
            resid_path = os.path.join(self.run_dir, f"ScaledResid_{factors}_{fpeak}.dat")
            
        try:
            if not os.path.exists(resid_path): return 0.0                
            with open(resid_path, 'r') as f:                             
                resids = np.array(f.read().replace(',', ' ').split(), dtype=float)
            resids = np.abs(resids)                                      
            q_values = np.where(resids <= 4.0, resids**2, 8.0 * resids - 16.0) 
            Q_robust = np.sum(q_values)                                  
            Q_exp = (n_rows * n_cols) - (factors * (n_rows + n_cols))    
            return Q_robust / Q_exp                                      
        except Exception as e:
            print(f"Q-Calc Error: {e}")                                  
            return 0.0

    def export_pmf_data(self, error_fraction):
        mat_path = os.path.join(self.run_dir, "MATRIX.DAT")              
        err_t_path = os.path.join(self.run_dir, "T_MATRIX.DAT")          
        err_v_path = os.path.join(self.run_dir, "V_MATRIX.DAT")          
        
        df_out = self.df.copy()                                          
        df_out = df_out.apply(pd.to_numeric, errors='coerce').fillna(0)  
        
        if self.chk_wide_pmf.isChecked():                                
            df_out = df_out.resample('h').mean().dropna(how='all')
            df_out['Date'] = df_out.index.date                           
            df_out['Hour'] = df_out.index.hour                           
            df_out = df_out.pivot(index='Date', columns='Hour')          
            
            existing_hours = df_out.columns.get_level_values('Hour').unique() 
            missing_hours = [h for h in range(24) if h not in existing_hours] 
            for h in missing_hours:                                      
                for col in self.diams:                                   
                    df_out[(col, h)] = 0.0                               
                    
            df_out = df_out.swaplevel(axis=1).sort_index(axis=1)         
            df_out = df_out.dropna()                                     

        df_error = (df_out * error_fraction).clip(lower=1e-2)            
        df_dummy_v = pd.DataFrame(0, index=df_out.index, columns=df_out.columns) 

        df_out.to_csv(mat_path, sep=' ', header=False, index=False)      
        df_error.to_csv(err_t_path, sep=' ', header=False, index=False)  
        df_dummy_v.to_csv(err_v_path, sep=' ', header=False, index=False)
        
        print(f"Exported PMF matrices to isolated dir. Dimensions: {df_out.shape}") 
        return df_out.shape[0], df_out.shape[1]                          

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)                                  
        
        dir_group = QGroupBox("1. PMF2 Software Setup")                  
        dir_layout = QGridLayout()                                       
        
        exe_str = os.path.basename(self.pmf_exe_path) if self.pmf_exe_path else "Not Selected" 
        key_str = os.path.basename(self.pmf_key_path) if self.pmf_key_path else "Not Selected" 
        
        self.lbl_exe = QLabel(f"Exe: {exe_str}")                         
        btn_exe = QPushButton("Locate pmf2.exe")                         
        btn_exe.setToolTip("Step 1: Point to your pmf2.exe file to authorise the batch execution engine.")
        btn_exe.clicked.connect(self.select_pmf_exe)                     
        btn_info_exe = QPushButton("?")                                  
        btn_info_exe.setMaximumWidth(30); btn_info_exe.clicked.connect(self.info_exe)
        
        self.lbl_key = QLabel(f"Key: {key_str}")                         
        btn_key = QPushButton("Locate pmf2key.key")                      
        btn_key.setToolTip("Step 2: Locate your license key. This will be dynamically fed to PMF2.")
        btn_key.clicked.connect(self.select_pmf_key)                     
        btn_info_key = QPushButton("?")                                  
        btn_info_key.setMaximumWidth(30); btn_info_key.clicked.connect(self.info_key)
        
        dir_layout.addWidget(self.lbl_exe, 0, 0); dir_layout.addWidget(btn_exe, 0, 1); dir_layout.addWidget(btn_info_exe, 0, 2) 
        dir_layout.addWidget(self.lbl_key, 1, 0); dir_layout.addWidget(btn_key, 1, 1); dir_layout.addWidget(btn_info_key, 1, 2) 
        dir_group.setLayout(dir_layout); main_layout.addWidget(dir_group)
        
        settings_group = QGroupBox("2. INI Settings & Batch Execution")  
        settings_layout = QGridLayout()                                  
        
        settings_layout.addWidget(QLabel("Min Factors:"), 0, 0)          
        self.spin_fac_min = QSpinBox()                                   
        self.spin_fac_min.setRange(2, 10); self.spin_fac_min.setValue(3) 
        self.spin_fac_min.setToolTip("Step 3: Define the minimum number of source profiles to seek.")
        settings_layout.addWidget(self.spin_fac_min, 0, 1)               
        
        settings_layout.addWidget(QLabel("Max Factors:"), 1, 0)          
        self.spin_fac_max = QSpinBox()                                   
        self.spin_fac_max.setRange(2, 10); self.spin_fac_max.setValue(6) 
        self.spin_fac_max.setToolTip("Step 3: Define the upper threshold of source profiles.")
        settings_layout.addWidget(self.spin_fac_max, 1, 1)               
        
        btn_info_fac = QPushButton("?"); btn_info_fac.setMaximumWidth(30)
        btn_info_fac.clicked.connect(self.info_factors); settings_layout.addWidget(btn_info_fac, 0, 2, 2, 1)
        
        settings_layout.addWidget(QLabel("Error Frac:"), 2, 0)           
        self.spin_error = QDoubleSpinBox()                               
        self.spin_error.setDecimals(3); self.spin_error.setValue(0.100)  
        self.spin_error.setToolTip("Step 4: Set a baseline uncertainty fraction (e.g. 0.1 for 10%). Optimize this later.")
        settings_layout.addWidget(self.spin_error, 2, 1)                 
        
        btn_info_err = QPushButton("?"); btn_info_err.setMaximumWidth(30)
        btn_info_err.clicked.connect(self.info_error); settings_layout.addWidget(btn_info_err, 2, 2)
        
        settings_layout.addWidget(QLabel("FPEAK Min:"), 0, 3)            
        self.spin_fpeak_min = QDoubleSpinBox()                           
        self.spin_fpeak_min.setRange(-10.0, 10.0); self.spin_fpeak_min.setValue(-1.0)
        self.spin_fpeak_min.setToolTip("Step 5: Define the lower boundary for rotational ambiguity testing.")
        settings_layout.addWidget(self.spin_fpeak_min, 0, 4)             
        
        settings_layout.addWidget(QLabel("FPEAK Max:"), 1, 3)            
        self.spin_fpeak_max = QDoubleSpinBox()                           
        self.spin_fpeak_max.setRange(-10.0, 10.0); self.spin_fpeak_max.setValue(1.0) 
        self.spin_fpeak_max.setToolTip("Step 5: Define the upper boundary for rotational ambiguity testing.")
        settings_layout.addWidget(self.spin_fpeak_max, 1, 4)             
        
        settings_layout.addWidget(QLabel("FPEAK Step:"), 2, 3)           
        self.spin_fpeak_step = QDoubleSpinBox()                          
        self.spin_fpeak_step.setSingleStep(0.1); self.spin_fpeak_step.setValue(0.5) 
        self.spin_fpeak_step.setToolTip("Step 5: The increment between your FPEAK Min and Max limits.")
        settings_layout.addWidget(self.spin_fpeak_step, 2, 4)            
        
        btn_info_fpk = QPushButton("?"); btn_info_fpk.setMaximumWidth(30)
        btn_info_fpk.clicked.connect(self.info_fpeak); settings_layout.addWidget(btn_info_fpk, 0, 5, 3, 1)
        
        self.chk_wide_pmf = QCheckBox("Enable WidePMF Mode")             
        self.chk_wide_pmf.setToolTip("Optional: Reshape the data block into a [Days x (Hours*Bins)] matrix to yield cyclical diurnal profiles.")
        settings_layout.addWidget(self.chk_wide_pmf, 3, 0, 1, 2)         
        
        lbl_beddows = QLabel("Ref: Beddows et al. (2025) Sci. Total Environ. 998:180231")
        lbl_beddows.setStyleSheet("font-size: 10px; color: gray;")       
        settings_layout.addWidget(lbl_beddows, 4, 0, 1, 3)
        
        btn_info_wide = QPushButton("?"); btn_info_wide.setMaximumWidth(30)
        btn_info_wide.clicked.connect(self.info_wide); settings_layout.addWidget(btn_info_wide, 3, 2)
        
        btn_run = QPushButton("Generate INI & Run PMF Batch!")           
        btn_run.setStyleSheet("background-color: #4CAF50; color: white;")
        btn_run.setToolTip("Step 6: Trigger the execution sequence. May take several minutes.")
        btn_run.clicked.connect(self.run_pmf_batch); settings_layout.addWidget(btn_run, 0, 6, 4, 1)
        
        settings_group.setLayout(settings_layout); main_layout.addWidget(settings_group)
        
        explore_group = QGroupBox("3. Model Explorer & Workflow")        
        explore_layout = QVBoxLayout()                                   
        
        sel_layout = QHBoxLayout()                                       
        self.lbl_fpeak = QLabel("Active Data: None")                     
        self.combo_fpeak = QComboBox()                                   
        self.combo_fpeak.setToolTip("Step 7: Select a computed matrix array to inspect.")
        self.combo_fpeak.currentTextChanged.connect(self.update_fpeak)   
        sel_layout.addWidget(self.lbl_fpeak); sel_layout.addWidget(self.combo_fpeak)
        explore_layout.addLayout(sel_layout)                             
        
        action_layout = QHBoxLayout()                                            # Container
        btn_vis = QPushButton("1. Open Visualisation Suite")                     # Button 1
        btn_vis.clicked.connect(self.open_visualiser)                            # Link
        btn_opt = QPushButton("2. Optimise Error Fraction")                     # Button 2
        btn_opt.clicked.connect(self.optimize_coefficients)                      # Link
        btn_rename = QPushButton("3. Rename Factors")                            # Button 3
        btn_rename.clicked.connect(self.open_renamer)                            # Link

        btn_combine = QPushButton("Combine Factors")                             # Merge two factors
        btn_combine.setStyleSheet("background-color: #673AB7; color: white;")    # Purple style
        btn_combine.clicked.connect(self.combine_factors)                        # Link logic

        btn_boot = QPushButton("Bootstrap Uncertainty")                          # Resample + re-run PMF
        btn_boot.setStyleSheet("background-color: #E91E63; color: white;")       # Pink style
        btn_boot.setToolTip("Re-run PMF on resampled data many times to estimate factor uncertainty. Slow.")
        btn_boot.clicked.connect(self.bootstrap_uncertainty)                     # Link logic

        btn_archive = QPushButton("4. Archive to Library")                       # New Archive Button
        btn_archive.setStyleSheet("background-color: #009688; color: white;")    # Teal style
        btn_archive.clicked.connect(self.save_current_model)                     # Link logic

        btn_load_lib = QPushButton("5. Load from Library")                       # New Load Button
        btn_load_lib.setStyleSheet("background-color: #795548; color: white;")   # Brown style
        btn_load_lib.clicked.connect(self.load_from_library)                     # Link logic
        
        for b in [btn_vis, btn_opt, btn_rename, btn_combine, btn_boot, btn_archive, btn_load_lib]:
            action_layout.addWidget(b)                                           # Add all to layout
        action_layout.addWidget(btn_vis)                                                 # Existing
        action_layout.addWidget(btn_opt)                                                 # Existing
        action_layout.addWidget(btn_rename)                                              # Existing
        action_layout.addWidget(btn_archive)                                             # Existing
        action_layout.addWidget(btn_load_lib)                                            # Add new button
        
        explore_layout.addLayout(action_layout)                                          # Pack layout

        tracer_row = QHBoxLayout()                                                        # External tracer loader
        btn_tracer = QPushButton("Load External Tracers (gas/met)")
        btn_tracer.setStyleSheet("background-color: #3F51B5; color: white;")
        btn_tracer.setToolTip("Load co-located NOx/BC/O3/solar/wind data for correlation, polar and nucleation-split tools.")
        btn_tracer.clicked.connect(self.open_tracer_loader)
        self.lbl_tracer = QLabel("No tracers loaded")
        self.lbl_tracer.setStyleSheet("color: gray;")
        tracer_row.addWidget(btn_tracer); tracer_row.addWidget(self.lbl_tracer); tracer_row.addStretch()
        explore_layout.addLayout(tracer_row)

        btn_export = QPushButton("4. Export Final Array to CSV")
        btn_export.setStyleSheet("background-color: #607D8B; color: white;")
        btn_export.setToolTip("Step 9: Compile active matrices (F, Raw G, Scaled G) to local disk for reporting.")
        btn_export.clicked.connect(self.export_final_data)               
        explore_layout.addWidget(btn_export)                             
        
        explore_group.setLayout(explore_layout); main_layout.addWidget(explore_group)
        main_layout.addStretch()                                         

    def export_final_data(self):
        if self.f_matrix is None or self.g_matrix is None:               
            QMessageBox.warning(self, "Error", "No matrices available to export.") 
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Export Data", "PyNSD_Final_Factors.csv", "CSV Files (*.csv)") 
        if not path: return                                              
        base, _ = os.path.splitext(path)                                 
        
        f_df = self.f_matrix.copy()
        f_df.columns = [self._get_factor_name(i) for i in range(self.current_factors)] 
        f_df.to_csv(f"{base}_F_Profiles.csv", index_label="Diameter_nm") 

        g_raw = self.g_matrix.copy()
        g_raw.columns = f_df.columns                                     
        g_raw.to_csv(f"{base}_G_Raw.csv", index_label="Date")            

        g_scaled = self.get_scaled_g()                                   
        g_scaled.columns = f_df.columns                                  
        g_scaled.to_csv(f"{base}_G_Scaled_ParticleNumber.csv", index_label="Date") 

        with open(f"{base}_Data_Structure.txt", "w") as f:               
            f.write("PyNSD PMF Array Output Manifest\n")                 
            f.write("===============================\n")                 
            f.write("1. _F_Profiles.csv: The final F matrix defining source size distributions. Under WidePMF, these rows represent contiguous hourly bins.\n")
            f.write("2. _G_Raw.csv: The unmodified, dimensionless G matrix corresponding to timesteps.\n")
            f.write("3. _G_Scaled_ParticleNumber.csv: The G matrix uniformly scaled against the dlogDp integral of each F profile to reflect pure particle numbers (cm-3).\n")
            
        QMessageBox.information(self, "Success", "Export matrices successfully compiled.") 

    def info_exe(self):
        msg = ("Locate the 'pmf2.exe' or 'pmf2wopt.exe' file. This defines the core computational engine and working directory for the batch runs.\n\n"
               "CRITICAL NOTE: PyNSD does NOT include PMF2. You must legally provide your own licensed copy of the executable.") 
        QMessageBox.warning(self, "PMF Executable", msg)                 
        
    def info_key(self):
        msg = ("PMF2 requires a valid 'pmf2key.key' file to authorise execution. "
               "This software will automatically copy it into the working directory to prevent execution crashes.") 
        QMessageBox.information(self, "PMF Key", msg)                    

    def info_factors(self):
        msg = ("Positive matrix factorization resolves the data matrix into a linear "
               "combination of factor profiles and time series contributions. The number "
               "of factors determines the dimensionality of the solution space.") 
        QMessageBox.information(self, "Number of Factors", msg)          

    def info_error(self):
        msg = ("The 'Error Fraction' represents the C3 coefficient in PMF2's uncertainty equation: s_ij = C1 + C2 * sqrt(y_ij) + C3 * y_ij.\n\n"
               "While C1 and C2 are typically fixed by your raw data matrix, C3 defines the baseline percentage uncertainty (e.g. 0.1 = 10%). "
               "You can manually set this, or use the 'Optimise' button to automatically scale it until the solution Q roughly equals Q-theoretical.") 
        QMessageBox.information(self, "Error Fraction (C3)", msg)        

    def info_fpeak(self):
        msg = ("FPEAK introduces a penalty term to the Q-value to control rotational "
               "ambiguity. Positive FPEAK values force factor profiles towards zero, "
               "while negative FPEAK values force factor contributions towards zero.") 
        QMessageBox.information(self, "FPEAK", msg)                      
        
    def info_wide(self):
        msg = ("WidePMF mode restructures the data from a continuous time series into daily chunks, "
               "where each hourly PNSD for each day is stacked side-by-side, such that 1 row represents 1 day of data. "
               "Each factor then represents a full 24-hour heatmap showing how a source changes throughout the day, "
               "better capturing events like New Particle Formation.\n\n"
               "Sometimes particles evolve on 48-hour cycles, so WidePMF picks up the particles being formed on day 1, "
               "and growing on day 2. There is a tool in the visualisation suite to visually pair the factors up side-by-side.\n\n"
               "Ref: Beddows et al. (2025). Science of The Total Environment, 998, 180231.") 
        QMessageBox.information(self, "WidePMF Mode", msg)               

    def select_pmf_exe(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select pmf2.exe", "", "Executables (*.exe)") 
        if file_path:                                                    
            self.pmf_exe_path = file_path                                
            self.working_dir = os.path.dirname(file_path)                        
            self.settings.setValue("exe_path", file_path)                
            self.working_dir = os.path.dirname(file_path)                
            self.run_dir = os.path.join(self.working_dir, "latest_output") 
            self.lbl_exe.setText(f"Exe: {os.path.basename(file_path)}")  
            
            potential_key = os.path.join(self.working_dir, "pmf2key.key")
            if os.path.exists(potential_key):                            
                self.pmf_key_path = potential_key                        
                self.settings.setValue("key_path", potential_key)        
                self.lbl_key.setText("Key: pmf2key.key (Auto-found)")    

    def select_pmf_key(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select pmf2key.key", "", "Key Files (*.key);;All Files (*)") 
        if file_path:                                                    
            self.pmf_key_path = file_path                                
            self.settings.setValue("key_path", file_path)                
            self.lbl_key.setText(f"Key: {os.path.basename(file_path)}")  
            
    def run_pmf_batch(self):
        if not self.pmf_exe_path or not self.pmf_key_path:               
            QMessageBox.warning(self, "Error", "Please locate both pmf2.exe and pmf2key.key before running.") 
            return
        
        self.batch_diams = self.diams.copy()
        self.prepare_run_directory()                                     
            
        fac_min = self.spin_fac_min.value()                              
        fac_max = self.spin_fac_max.value()                              
        f_min = self.spin_fpeak_min.value()                              
        f_max = self.spin_fpeak_max.value()                              
        f_step = self.spin_fpeak_step.value()                            
        
        if f_step <= 0:                                                  
            fpeaks = [f_min]                                             
        else:                                                            
            fpeaks = np.arange(f_min, f_max + f_step, f_step)            
            
        factors_range = range(fac_min, fac_max + 1)                      
        total_runs = len(factors_range) * len(fpeaks)                    
        self.combo_fpeak.clear()                                         
        
        dialog = CowProgressDialog(total_runs, self)                     
        dialog.show()                                                    
        
        task_name = "PyNSD"                                              
        current_step = 0                                                 
        current_error = self.spin_error.value()                          
        
        n_rows, n_cols = self.export_pmf_data(current_error)             
        
        for factors in factors_range:                                    
            for fpeak in fpeaks:                                         
                fpeak = round(fpeak, 2)                                  
                self.write_fkey(factors, n_cols)                         
                
                dialog.update_progress(current_step, factors, fpeak)     
                self.generate_ini(n_rows, n_cols, factors, fpeak, current_error) 
                
                try:
                    process = subprocess.Popen(
                        [self.pmf_exe_path, task_name],                  
                        cwd=self.run_dir,                                
                        creationflags=subprocess.CREATE_NEW_CONSOLE      
                    )
                    process.wait()                                       
                    
                    q_ratio = self._get_q_ratio(factors, fpeak, n_rows, n_cols, is_batch=True)
                    q_val_str = f"{q_ratio:.2f}" if q_ratio > 0 else "N/A"
                    
                    self._rename_output_files(factors, fpeak)            
                    self.combo_fpeak.addItem(f"Factors: {factors}, FPEAK: {fpeak}, Q/Qexp: {q_val_str}") 
                    
                except Exception as e:
                    dialog.close()                                       
                    print(f"Subprocess Error: {e}")                      
                    return
                current_step += 1                                        
                
        if self.combo_fpeak.count() > 0:
            self.combo_fpeak.setCurrentIndex(0)                                  # Trigger update_fpeak
            self.update_fpeak(self.combo_fpeak.currentText())                   # Double-check trigger
        
        dialog.update_progress(total_runs, "Complete", "")               
        dialog.close()                                                   
        QMessageBox.information(self, "Success", "Batch PMF runs complete!")

    def optimize_coefficients(self):
        if not self.pmf_exe_path or self.current_factors == 0:           
            QMessageBox.warning(self, "Error", "Please run a batch and select an active model from the Explorer first.") 
            return

        target_ratio = 1.0                                               
        current_error = self.spin_error.value()                          
        max_iters = 8                                                    
        q_ratio = 0.0                                                    
        
        dialog = OptimiseProgressDialog(max_iters, self)                 
        dialog.show()                                                    
        
        self.prepare_run_directory()                                     
        
        for step in range(1, max_iters + 1):                             
            n_rows, n_cols = self.export_pmf_data(current_error)         
            self.generate_ini(n_rows, n_cols, self.current_factors, self.current_fpeak, current_error)

            try:
                process = subprocess.Popen(
                    [self.pmf_exe_path, "PyNSD"],                        
                    cwd=self.run_dir,                                    
                    creationflags=subprocess.CREATE_NEW_CONSOLE          
                )
                process.wait()                                           
            except Exception as e:
                dialog.close()                                           
                QMessageBox.critical(self, "Execution Error", str(e))    
                return

            q_ratio = self._get_q_ratio(self.current_factors, self.current_fpeak, n_rows, n_cols, is_batch=True)
            
            if q_ratio == 0.0:                                           
                print("Could not calculate Q.")                     
                break
                
            dialog.update_status(step, current_error, q_ratio)       
            
            if abs(q_ratio - target_ratio) < 0.05: break             
                
            current_error = current_error * np.sqrt(q_ratio)         
                
        dialog.close()                                                   
        
        if q_ratio != 0.0:                                               
            self._rename_output_files(self.current_factors, f"{self.current_fpeak}_opt") 
            self.spin_error.setValue(current_error)                      
            QMessageBox.information(self, "Optimisation Complete", f"Optimal Error Fraction found: {current_error:.4f}\nAchieved Q/Qexp: {q_ratio:.4f}") 
        else:
            QMessageBox.warning(self, "Optimisation Failed", "Could not calculate Q/Qexp from the residuals.")

    def _read_matrix(self, path, shape):
        vals = np.array(open(path).read().replace(',', ' ').split(), dtype=float)
        return vals.reshape(shape)

    def _match_factors(self, base_F, boot_F):
        # Greedy assignment of bootstrap factors to base factors by profile correlation.
        k = base_F.shape[1]
        corr = np.zeros((k, k))
        for a in range(k):
            for c in range(k):
                ca, cb = base_F[:, a], boot_F[:, c]
                if np.std(ca) > 0 and np.std(cb) > 0:
                    corr[a, c] = np.corrcoef(ca, cb)[0, 1]
        mapping = [-1] * k; scores = [0.0] * k; used = set()
        for a in np.argsort(-corr.max(axis=1)):                          # Assign the most distinctive base factor first
            for c in np.argsort(-corr[a]):
                if c not in used:
                    mapping[a] = int(c); scores[a] = float(corr[a, c]); used.add(int(c)); break
        return mapping, scores

    def bootstrap_uncertainty(self):
        if not self.pmf_exe_path or self.current_factors == 0 or self.f_matrix is None:
            return QMessageBox.warning(self, "Error", "Run a batch and select an active model first.")
        if not self.pmf_key_path:
            return QMessageBox.warning(self, "Error", "Locate pmf2key.key first.")

        n_boot, ok = QInputDialog.getInt(self, "Bootstrap Runs", "Number of bootstrap resamples:", 50, 5, 500)
        if not ok:
            return

        lo_min = max(1, int(n_boot * 5 / 60)); hi_min = max(1, int(n_boot * 30 / 60))
        warn = ("☠️  HEADS UP\n\n"
                f"This re-runs pmf2.exe {n_boot} times from scratch on resampled data.\n"
                f"Each run takes several seconds to about a minute, so the whole bootstrap will "
                f"likely take on the order of {lo_min}-{hi_min} MINUTES.\n\n"
                "The interface stays busy the entire time (a progress bar with a live ETA is shown, "
                "and there is a Cancel button). Continue?")
        if QMessageBox.question(self, "Bootstrap Uncertainty", warn,
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                                ) != QMessageBox.StandardButton.Yes:
            return

        import time
        current_error = self.spin_error.value()
        factors, fpeak = self.current_factors, self.current_fpeak

        # Base matrices (written to run_dir; this does not delete the F_FACTOR files of the active model).
        n_rows, n_cols = self.export_pmf_data(current_error)
        base_mat = self._read_matrix(os.path.join(self.run_dir, "MATRIX.DAT"), (n_rows, n_cols))
        base_err = self._read_matrix(os.path.join(self.run_dir, "T_MATRIX.DAT"), (n_rows, n_cols))
        base_v = self._read_matrix(os.path.join(self.run_dir, "V_MATRIX.DAT"), (n_rows, n_cols))
        base_F = self.f_matrix.to_numpy(dtype=float)                     # (n_variables, k)

        # Isolated working dir so the current model's outputs stay intact.
        boot_dir = os.path.join(self.run_dir, "bootstrap")
        if os.path.exists(boot_dir):
            for f in os.listdir(boot_dir):
                fp = os.path.join(boot_dir, f)
                if os.path.isfile(fp): os.remove(fp)
        else:
            os.makedirs(boot_dir)
        shutil.copy(self.pmf_key_path, os.path.join(boot_dir, "pmf2key.key"))
        with open(os.path.join(boot_dir, "FKEY.DAT"), 'w', newline='\r\n') as f:
            row = " ".join(["0"] * n_cols)
            for _ in range(factors):
                f.write(row + "\r\n")

        dialog = BootstrapProgressDialog(n_boot, self); dialog.show()
        dlogdp = np.log10(self.diams[1] / self.diams[0]) if len(self.diams) > 1 else 1.0

        F_boot = [[] for _ in range(factors)]
        contrib = []
        n_used = 0; n_scores = 0; n_ok = 0; thresh = 0.6
        start = time.time()

        for b in range(1, n_boot + 1):
            if dialog.cancelled:
                break
            idx = np.random.randint(0, n_rows, n_rows)                   # Resample rows with replacement
            np.savetxt(os.path.join(boot_dir, "MATRIX.DAT"), base_mat[idx], fmt="%.6g")
            np.savetxt(os.path.join(boot_dir, "T_MATRIX.DAT"), base_err[idx], fmt="%.6g")
            np.savetxt(os.path.join(boot_dir, "V_MATRIX.DAT"), base_v[idx], fmt="%.6g")
            generate_pmf_ini(boot_dir, n_rows, n_cols, factors, fpeak, current_error, task_name="PyNSD")

            try:
                proc = subprocess.Popen([self.pmf_exe_path, "PyNSD"], cwd=boot_dir,
                                        creationflags=subprocess.CREATE_NEW_CONSOLE)
                proc.wait()
                bf = self._read_matrix(os.path.join(boot_dir, "F_FACTOR.TXT"), (factors, n_cols)).T   # (n_vars, k)
                bg = np.array(open(os.path.join(boot_dir, "G_FACTOR.TXT")).read().replace(',', ' ').split(),
                              dtype=float).reshape(-1, factors)                                       # (n_rows, k)
            except Exception as e:
                print(f"Bootstrap {b} failed: {e}")
                elapsed = time.time() - start
                dialog.update(b, n_boot, (elapsed / b) * (n_boot - b), (n_ok / n_scores) if n_scores else 0.0)
                continue

            mapping, scores = self._match_factors(base_F, bf)
            for a in range(factors):
                F_boot[a].append(bf[:, mapping[a]])
            contrib.append(np.array([bg[:, mapping[a]].mean() * (bf[:, mapping[a]].sum() * dlogdp)
                                     for a in range(factors)]))
            n_used += 1
            n_scores += factors; n_ok += int(np.sum(np.array(scores) >= thresh))

            elapsed = time.time() - start
            dialog.update(b, n_boot, (elapsed / b) * (n_boot - b), (n_ok / n_scores) if n_scores else 0.0)

        dialog.close()

        if n_used < 2:
            return QMessageBox.warning(self, "Bootstrap", "Too few successful bootstrap runs to summarise.")

        F_boot_arr = [np.array(F_boot[a]) for a in range(factors)]       # each (n_used, n_variables)
        contrib_arr = np.array(contrib)                                  # (n_used, k)
        success = n_ok / n_scores if n_scores else 0.0
        names = [self._get_factor_name(i) for i in range(factors)]
        BootstrapResultsDialog(self.diams, base_F, F_boot_arr, contrib_arr, names, success, n_used, self).exec()

    def generate_ini(self, n_rows, n_cols, factors, fpeak, error_fraction=0.1):
        generate_pmf_ini(self.run_dir, n_rows, n_cols, factors, fpeak, error_fraction, task_name="PyNSD") 

    def _rename_output_files(self, factors, fpeak):
        files = ["F_FACTOR.TXT", "G_FACTOR.TXT", "ScaledResid.dat"]      
        for f in files:                                                  
            src = os.path.join(self.run_dir, f)                          
            base, ext = os.path.splitext(f)                              
            new_name = f"{base}_{factors}_{fpeak}{ext}"                  
            dst = os.path.join(self.run_dir, new_name)                   
            if os.path.exists(src):                                      
                if os.path.exists(dst): os.remove(dst)                   
                os.rename(src, dst)                                      

    def update_fpeak(self, text):
        if text:                                                                 # Check if text exists
            f_match = re.search(r"Factors:?\s*(\d+)", text, re.IGNORECASE)       # Regex for factor count
            p_match = re.search(r"FPEAK:?\s*([-0-9.]+)", text, re.IGNORECASE)    # Regex for fpeak value
            
            if f_match and p_match:
                self.current_factors = int(f_match.group(1))                     # Extract factors
                fpeak_str = p_match.group(1)                                     # KEEP STRING for filename match
                self.current_fpeak = float(fpeak_str)                            # Convert to float for logic
                self.lbl_fpeak.setText(f"Active Data: {text}")                   # Update the UI label
                
                self.factor_names.clear()                                        # Wipe custom names
                
                # Path construction using the string '0.0' to match the disk exactly
                f_path = os.path.join(self.run_dir, f"F_FACTOR_{self.current_factors}_{fpeak_str}.TXT")
                g_path = os.path.join(self.run_dir, f"G_FACTOR_{self.current_factors}_{fpeak_str}.TXT")
                
                try:
                    if not os.path.exists(f_path) or not os.path.exists(g_path): 
                        raise FileNotFoundError(f"File mismatch! Looking for: {f_path}") 
                    
                    with open(f_path, 'r') as f:
                        f_vals = np.array(f.read().replace(',', ' ').split(), dtype=float)
                    self.f_matrix = pd.DataFrame(f_vals.reshape(self.current_factors, -1).T)
                    
                    with open(g_path, 'r') as f:
                        g_vals = np.array(f.read().replace(',', ' ').split(), dtype=float)
                    self.g_matrix = pd.DataFrame(g_vals.reshape(-1, self.current_factors))
                    
                    if self.diams is not None and len(self.f_matrix) == len(self.diams): 
                        self.f_matrix.index = self.diams                         # Apply diameter index
                    
                    if self.dates is not None:                                   
                        if len(self.g_matrix) == len(self.dates):        
                            self.g_matrix.index = self.dates                     # Apply datetime index
                        else:                                            
                            self.g_matrix.index = pd.date_range(self.dates.min(), periods=len(self.g_matrix), freq='D')
                            
                except Exception as e:                                   
                    self.f_matrix = None                                         # Reset on fail
                    self.g_matrix = None                                         # Reset on fail
                    print(f"DEBUG: Load Failed: {e}")                            # Log to terminal

    def open_visualiser(self):
        print("DEBUG: Visualiser button clicked.")                                 # Log the click
        if self.f_matrix is None or self.g_matrix is None:
            print("DEBUG: Guard failed! f_matrix or g_matrix is None.")           # This is why it's failing
            QMessageBox.warning(self, "No Model Active", "Please select a run from the dropdown first.")
            return
        dialog = TabbedVisualizer(self, self)                            
        dialog.exec()                                                    

    def open_renamer(self):
        print("DEBUG: Renamer button clicked.")
        if self.current_factors == 0:
            return QMessageBox.warning(self, "No Data", "Select a model first.")
        from gui.pmf_panel import RenameDialog                                   # Double check import path
        dialog = RenameDialog(self.current_factors, self.factor_names, self)
        dialog.exec()

    def open_tracer_loader(self):
        dialog = TracerLoadDialog(self)
        if dialog.exec() and dialog.tracer_df is not None:
            self.tracer_df = dialog.tracer_df
            cols = ", ".join(str(c) for c in self.tracer_df.columns)
            self.lbl_tracer.setText(f"Tracers: {cols}  ({len(self.tracer_df)} rows)")
            self.lbl_tracer.setStyleSheet("color: green;")

    def combine_factors(self):
        if self.f_matrix is None or self.g_matrix is None:
            return QMessageBox.warning(self, "No Model Active", "Please select a run from the dropdown first.")
        if self.current_factors < 2:
            return QMessageBox.warning(self, "Error", "Need at least two factors to combine.")

        dialog = CombineFactorsDialog(self, self)
        if not dialog.exec() or dialog.selected is None:
            return
        idx_a, idx_b = dialog.selected
        lo, hi = sorted([idx_a, idx_b])                                          # Merged factor takes the lower slot

        combined_f = self.f_matrix.iloc[:, idx_a].values + self.f_matrix.iloc[:, idx_b].values
        combined_g = self.g_matrix.iloc[:, idx_a].values + self.g_matrix.iloc[:, idx_b].values
        combined_name = f"{self._get_factor_name(idx_a)}+{self._get_factor_name(idx_b)}"

        # Rebuild F, G and names positionally: drop the higher slot, sum into the lower one,
        # keep every other factor (and its current name) in order.
        f_cols, g_cols, names = [], [], {}
        pos = 0
        for i in range(self.current_factors):
            if i == hi:
                continue
            if i == lo:
                f_cols.append(combined_f); g_cols.append(combined_g); label = combined_name
            else:
                f_cols.append(self.f_matrix.iloc[:, i].values)
                g_cols.append(self.g_matrix.iloc[:, i].values)
                label = self._get_factor_name(i)
            names[f"Factor {pos + 1}"] = label
            pos += 1

        self.f_matrix = pd.DataFrame(np.column_stack(f_cols), index=self.f_matrix.index)
        self.g_matrix = pd.DataFrame(np.column_stack(g_cols), index=self.g_matrix.index)
        self.current_factors -= 1
        self.factor_names = names

        self.lbl_fpeak.setText(f"Active Data (COMBINED): {self.current_factors} factors")
        QMessageBox.information(self, "Factors Combined",
            f"Merged into '{combined_name}'.\nModel now has {self.current_factors} factors.\n\n"
            "This is an in-memory edit. Re-selecting a run from the dropdown reloads the original solution.")

    def _get_factor_name(self, col_idx):
        raw_name = f"Factor {col_idx + 1}"                               
        return self.factor_names.get(raw_name, raw_name)

    def save_current_model(self):
        if self.f_matrix is None or self.g_matrix is None:
            return QMessageBox.warning(self, "Error", "Nothing to save. Select a model first.")

        name, ok = QInputDialog.getText(self, "Archive Model", "Name this solution:")
        if not (ok and name): return

        safe_name = re.sub(r'[<>:"/\\|?*]', '_', name).strip(' .')                       # Strip path-hostile chars
        if not safe_name:
            return QMessageBox.warning(self, "Invalid Name", "Please enter a usable name.")

        library_root = os.path.join(self.working_dir, "saved_library") if self.working_dir else "saved_library"
        archive_dir = os.path.join(library_root, safe_name)

        if os.path.isdir(archive_dir) and os.listdir(archive_dir):                       # Confirm before clobbering
            reply = QMessageBox.question(self, "Overwrite?",
                f"An archive named '{safe_name}' already exists. Overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes: return
        os.makedirs(archive_dir, exist_ok=True)

        try:
            # Persist the in-memory matrices (the source of truth) with their own indices,
            # so combined/optimised solutions are saved exactly as displayed.
            self.f_matrix.to_csv(os.path.join(archive_dir, "F_matrix.csv"), index_label="Diameter_nm")
            self.g_matrix.to_csv(os.path.join(archive_dir, "G_matrix.csv"), index_label="Timestamp")

            # Best-effort: copy residuals + Q/Qexp so the Diagnostics tab survives a reload.
            q_ratio = 0.0
            try:
                fpeak_str = self.combo_fpeak.currentText().split("FPEAK:")[1].split(",")[0].strip()
            except Exception:
                fpeak_str = str(self.current_fpeak)
            resid_src = os.path.join(self.run_dir, f"ScaledResid_{self.current_factors}_{fpeak_str}.dat") if self.run_dir else ""
            if resid_src and os.path.exists(resid_src):
                shutil.copy(resid_src, os.path.join(archive_dir, "ScaledResid.dat"))
                q_ratio = self._get_q_ratio(self.current_factors, self.current_fpeak,
                                            len(self.g_matrix), len(self.f_matrix))

            meta = {
                "format": "csv_v2",
                "factors": self.current_factors,
                "fpeak": self.current_fpeak,
                "error_fraction": self.spin_error.value(),
                "names": self.factor_names,
                "chk_wide": self.chk_wide_pmf.isChecked(),
                "diams": [float(d) for d in self.diams] if self.diams is not None else [],
                "q_ratio": q_ratio,
            }
            with open(os.path.join(archive_dir, "metadata.json"), 'w') as f:
                json.dump(meta, f, indent=2)

            QMessageBox.information(self, "Success", f"Model archived to saved_library/{safe_name}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not archive model: {e}")

    def load_from_library(self):
        library_root = os.path.join(self.working_dir, "saved_library") if self.working_dir else "saved_library"
        selected_dir = QFileDialog.getExistingDirectory(self, "Select Archived Model", library_root)
        if not selected_dir: return

        try:
            with open(os.path.join(selected_dir, "metadata.json"), 'r') as f:
                meta = json.load(f)

            # 1. Restore scalar state
            self.current_factors = meta.get("factors", 0)
            self.current_fpeak = meta.get("fpeak", 0.0)
            self.factor_names = meta.get("names", {})
            self.chk_wide_pmf.setChecked(meta.get("chk_wide", False))
            self.diams = np.array(meta.get("diams", []))
            if "error_fraction" in meta:
                self.spin_error.setValue(meta["error_fraction"])

            # 2. Load matrices, supporting both the new CSV format and legacy TXT dumps
            f_csv = os.path.join(selected_dir, "F_matrix.csv")
            g_csv = os.path.join(selected_dir, "G_matrix.csv")

            if os.path.exists(f_csv) and os.path.exists(g_csv):
                self.f_matrix = pd.read_csv(f_csv, index_col=0)
                self.g_matrix = pd.read_csv(g_csv, index_col=0)
                self.g_matrix.index = pd.to_datetime(self.g_matrix.index)               # Rehydrate datetimes
                self.dates = self.g_matrix.index
                # Restore integer column labels to match freshly-loaded solutions (positional access).
                self.f_matrix.columns = range(self.f_matrix.shape[1])
                self.g_matrix.columns = range(self.g_matrix.shape[1])
                if len(self.diams) == 0 and not self.chk_wide_pmf.isChecked():
                    self.diams = np.array(self.f_matrix.index, dtype=float)              # Fall back to F index
            else:
                ts_list = meta.get("timestamps", [])
                if ts_list:
                    self.dates = pd.to_datetime(ts_list)
                with open(os.path.join(selected_dir, "F_MATRIX.TXT")) as fh:
                    f_vals = np.array(fh.read().replace(',', ' ').split(), dtype=float)
                with open(os.path.join(selected_dir, "G_MATRIX.TXT")) as fh:
                    g_vals = np.array(fh.read().replace(',', ' ').split(), dtype=float)
                self.f_matrix = pd.DataFrame(f_vals.reshape(self.current_factors, -1).T)
                self.g_matrix = pd.DataFrame(g_vals.reshape(-1, self.current_factors))
                if len(self.diams) > 0 and not self.chk_wide_pmf.isChecked() and len(self.f_matrix) == len(self.diams):
                    self.f_matrix.index = self.diams
                if self.dates is not None and len(self.g_matrix) == len(self.dates):
                    self.g_matrix.index = self.dates

            # 3. Best-effort: drop residuals back where the Diagnostics tab looks for them
            resid_arch = os.path.join(selected_dir, "ScaledResid.dat")
            if os.path.exists(resid_arch) and self.run_dir:
                os.makedirs(self.run_dir, exist_ok=True)
                shutil.copy(resid_arch, os.path.join(self.run_dir, f"ScaledResid_{self.current_factors}_{self.current_fpeak}.dat"))

            self.lbl_fpeak.setText(f"Active Data (LIBRARY): {os.path.basename(selected_dir)}")
            QMessageBox.information(self, "Loaded", "Model restored from library.")

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to reconstruct model: {e}")
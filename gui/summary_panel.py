import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.colors import LogNorm
import matplotlib
from matplotlib import rcParams
from PyQt6.QtWidgets import (QWidget,
                             QVBoxLayout,
                             QHBoxLayout,
                             QLabel,
                             QLineEdit,
                             QComboBox,
                             QSplitter,
                             QTableWidget,
                             QTableWidgetItem,
                             QHeaderView,
                             QPushButton,
                             QMessageBox,
                             QDialog,
                             QSizePolicy)
from PyQt6.QtCore import Qt, QTimer
from utils.calculations import dlogdp_per_bin, integrate_pnsd, resolve_dlogdp
from utils.helpers import fit_to_screen
from gui.widgets import validate_number
from gui.filedialogs import get_save_file_name

# Mesh columns drawn at once. Comfortably above screen width and a 300 dpi
# export, but far below the row count of a long high-resolution deployment.
MAX_MESH_COLUMNS = 4000

# Style settings - Force LaTeX to use the same serif font as the UI
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Georgia', 'Times New Roman']
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Georgia'
rcParams['mathtext.it'] = 'Georgia:italic'
rcParams['mathtext.bf'] = 'Georgia:bold'
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['figure.facecolor'] = '#fff1e5'
rcParams['axes.facecolor'] = '#fff1e5'
rcParams['savefig.facecolor'] = '#fff1e5'

# --- Custom Export Dialog (Ported from Trend Analysis) ---
class ExportDialog(QDialog):
    def __init__(self, plot_name, canvas, fig, restore_callback, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.fig = fig
        self.restore_callback = restore_callback
        self.setWindowTitle(f"Export: {plot_name}")
        
        self.layout = QVBoxLayout(self)
        
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Width (px):"))
        self.val_w = QLineEdit(str(int(fig.get_figwidth() * fig.dpi)))
        ctrl_layout.addWidget(self.val_w)
        
        ctrl_layout.addWidget(QLabel("Height (px):"))
        self.val_h = QLineEdit(str(int(fig.get_figheight() * fig.dpi)))
        ctrl_layout.addWidget(self.val_h)
        
        self.btn_apply = QPushButton("Apply Size")
        self.btn_apply.clicked.connect(self.apply_size)
        ctrl_layout.addWidget(self.btn_apply)
        
        self.btn_save = QPushButton("💾 Save Image")
        self.btn_save.clicked.connect(self.save_plot)
        ctrl_layout.addWidget(self.btn_save)
        
        self.layout.addLayout(ctrl_layout)
        self.layout.addWidget(self.canvas, stretch=1)
        
        w = int(fig.get_figwidth() * fig.dpi)
        h = int(fig.get_figheight() * fig.dpi) + 50
        fit_to_screen(self, w, h)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.val_w.setText(str(self.canvas.width()))
        self.val_h.setText(str(self.canvas.height()))

    def apply_size(self):
        try:
            w, h = int(self.val_w.text()), int(self.val_h.text())
            fit_to_screen(self, w, h + 50)
        except ValueError:
            pass

    def save_plot(self):
        path, _ = get_save_file_name(self, "Save Plot", "", "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)")
        if path:
            w_in = self.canvas.width() / self.fig.dpi
            h_in = self.canvas.height() / self.fig.dpi
            self.fig.set_size_inches(w_in, h_in)
            self.fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
            QMessageBox.information(self, "Success", "Plot saved successfully!")
            self.accept()

    def closeEvent(self, event):
        self.restore_callback()
        super().closeEvent(event)

# Multi-Gaussian model for log10-space fitting
def multi_lognormal(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        H, mu, sigma = params[i:i+3]
        y += H * np.exp(-((x - mu)**2) / (2 * sigma**2))
    return y


class SummaryPanel(QWidget):                                                 
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.df = None
        self.diams = None
        
        layout = QVBoxLayout(self)
        
        # --- Info Header ---
        intro_layout = QHBoxLayout()
        intro_lbl = QLabel("Dataset Summary & Interactive Exploration")
        intro_lbl.setStyleSheet("font-size: 14px; font-weight: bold;")
        
        info_text = (
            "<b>How to use this panel:</b><br><br>"
            "This panel provides a complete overview of your dataset. <br><br>"
            "• <b>Interactive Zooming:</b> Use the magnifying glass tool on the top contour plot to highlight a specific time period. "
            "As you zoom in, the bottom plots (the average size distribution and diurnal cycle) will automatically recalculate and update to reflect <i>only</i> the data within your zoomed window!<br><br>"
            "• <b>Dynamic Time-Averaging:</b> The thin blue line (Total N) on the top plot automatically resamples itself to monthly or daily averages when viewing multi-year datasets, preventing the plot from becoming a solid block of ink. It returns to high-resolution as you zoom in.<br><br>"
            "• <b>Standard Errors:</b> The shaded translucent regions on the bottom plots represent ±1 Standard Error of the Mean (SEM), providing a clean confidence interval for your averages.<br><br>"
            "• <b>Lognormal Fitting:</b> The system automatically isolates up to 5 individual modes in the active size distribution and reports their peak diameter and height. Width limits are constrained to prevent unrealistically broad modes.<br><br>"
            "• <b>Exporting:</b> Use the 'Export Plots' button to save high-resolution, custom-sized copies of these figures."
        )
        intro_layout.addWidget(intro_lbl)
        
        self.btn_info = QPushButton("ℹ️")
        self.btn_info.setFixedSize(24, 24)
        self.btn_info.clicked.connect(lambda: QMessageBox.information(self, "Summary Panel Info", info_text))
        intro_layout.addWidget(self.btn_info)
        intro_layout.addStretch()
        layout.addLayout(intro_layout)

        # --- Top Control Bar ---
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("dlogDp:"))
        self.dlogdp_input = QLineEdit("0.0")
        self.dlogdp_input.editingFinished.connect(lambda: self.update_top())
        ctrl_layout.addWidget(self.dlogdp_input)
        
        ctrl_layout.addWidget(QLabel("Density (g/cm³):"))
        self.density_input = QLineEdit("1.5")
        self.density_input.editingFinished.connect(lambda: self.update_bottom())
        ctrl_layout.addWidget(self.density_input)
        
        ctrl_layout.addWidget(QLabel("Colour Map:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["turbo", "viridis", "plasma", "inferno"]) 
        self.cmap_combo.currentTextChanged.connect(lambda: self.update_top())
        ctrl_layout.addWidget(self.cmap_combo)
        
        ctrl_layout.addWidget(QLabel("Min Colour:"))
        self.cbar_min = QLineEdit("1")
        self.cbar_min.editingFinished.connect(lambda: self.update_top())
        ctrl_layout.addWidget(self.cbar_min)
        
        ctrl_layout.addWidget(QLabel("Max Colour:"))
        self.cbar_max = QLineEdit("")
        self.cbar_max.editingFinished.connect(lambda: self.update_top())
        ctrl_layout.addWidget(self.cbar_max)
        
        self.export_btn = QPushButton("Export Plots")
        self.export_btn.clicked.connect(self.export_plots)
        ctrl_layout.addWidget(self.export_btn)
        
        layout.addLayout(ctrl_layout)
        
        # --- Top Section (Heatmap) ---
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        top_widget = QWidget()
        self.top_layout = QVBoxLayout(top_widget)
        
        top_bar = QHBoxLayout()
        self.fig_top = Figure(figsize=(12, 5))
        self.canvas_top = FigureCanvasQTAgg(self.fig_top)
        self.canvas_top.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas_top.setMinimumHeight(400)
        self.canvas_top.updateGeometry()
        
        self.toolbar_top = NavigationToolbar2QT(self.canvas_top, self)
        top_bar.addWidget(self.toolbar_top)
        
        self.btn_log_top = QPushButton("Log Y (Total N)")
        self.btn_log_top.setCheckable(True)
        self.btn_log_top.clicked.connect(lambda: self._update_top_line())
        top_bar.addWidget(self.btn_log_top)
        self.top_layout.addLayout(top_bar)
        
        self.ax_contour = self.fig_top.add_subplot(111)
        self.ax_line = self.ax_contour.twinx()
        self.top_layout.addWidget(self.canvas_top, stretch=1)
        splitter.addWidget(top_widget)
        
        # --- Bottom Section (Plots & Stats) ---
        bot_widget = QWidget()
        bot_v_layout = QVBoxLayout(bot_widget)
        
        bot_bar = QHBoxLayout()
        self.btn_log_dist = QPushButton("Log Y (Dist)")
        self.btn_log_dist.setCheckable(True)
        self.btn_log_dist.clicked.connect(lambda: self.update_bottom())
        bot_bar.addWidget(self.btn_log_dist)
        
        #self.btn_log_diur = QPushButton("Log Y (Diurnal)")
        #self.btn_log_diur.setCheckable(True)
        #self.btn_log_diur.clicked.connect(lambda: self.update_bottom())
        #bot_bar.addWidget(self.btn_log_diur)
        bot_bar.addStretch()
        bot_v_layout.addLayout(bot_bar)
        
        self.bot_plots_layout = QHBoxLayout()
        
        self.fig_dist = Figure(figsize=(5, 3))
        self.canvas_dist = FigureCanvasQTAgg(self.fig_dist)
        self.ax_dist_num = self.fig_dist.add_subplot(111)
        self.ax_dist_mass = self.ax_dist_num.twinx()
        
        self.fig_diur = Figure(figsize=(5, 3))
        self.canvas_diur = FigureCanvasQTAgg(self.fig_diur)
        self.ax_diur_num = self.fig_diur.add_subplot(111)
        self.ax_diur_mass = self.ax_diur_num.twinx()
        
        self.bot_plots_layout.addWidget(self.canvas_dist, stretch=2)
        self.bot_plots_layout.addWidget(self.canvas_diur, stretch=2)
        
        self.table = QTableWidget()
        self.table.setColumnCount(1)
        self.table.setHorizontalHeaderLabels(["Value"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch) 
        self.bot_plots_layout.addWidget(self.table, stretch=1)
        
        bot_v_layout.addLayout(self.bot_plots_layout)
        
        splitter.addWidget(bot_widget)
        layout.addWidget(splitter, stretch=1)
        
        self.cbar = None
        self.zoom_cid = None
        self.line_total_n = None
        self.total_n_series = None
        self._date_nums = np.array([])

        self._pending_xlim = None
        self._zoom_timer = QTimer(self)
        self._zoom_timer.setSingleShot(True)
        self._zoom_timer.timeout.connect(self._apply_zoom)

        # Red outline while a box holds something unusable.
        validate_number(self.dlogdp_input, minimum=0, maximum=2)
        validate_number(self.density_input, minimum=0.1, maximum=25)
        validate_number(self.cbar_min, minimum=0)
        validate_number(self.cbar_max, minimum=0)




    def _downsample_for_mesh(self, df: pd.DataFrame) -> pd.DataFrame:
        """Average the time axis down to what a screen can actually resolve.

        A year of 5 min data is 105 000 mesh columns behind a plot about 1200 px
        wide, and matplotlib rasterises every one of them.  The cap is well above
        both screen and 300 dpi export resolution, so the picture is unchanged.
        Zooming re-draws from the full data, so detail is not lost.
        """
        if len(df) <= MAX_MESH_COLUMNS:
            return df

        step = int(np.ceil(len(df) / MAX_MESH_COLUMNS))
        blocks = np.arange(len(df)) // step
        out = df.groupby(blocks).mean()                    # all-NaN blocks stay NaN, so gaps survive
        out.index = df.index[::step][:len(out)]
        return out

    def _dlogdp_enabled(self) -> bool:
        """A zero or unreadable dlogDp box means 'do not draw total number'."""
        try:
            return float(self.dlogdp_input.text()) > 0
        except ValueError:
            return False

    def _insert_time_gaps_for_plot(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reindex to expected cadence so missing periods become NaN gaps in plots."""
        if df is None or df.empty or len(df.index) < 3:
            return df

        df_sorted = df.sort_index()
        deltas = pd.Series(df_sorted.index).diff().dropna()
        cadence = deltas.median()
        if pd.isna(cadence) or cadence <= pd.Timedelta(0):
            return df_sorted

        full_index = pd.date_range(df_sorted.index.min(), df_sorted.index.max(), freq=cadence)
        if len(full_index) > len(df_sorted) * 20:
            return df_sorted

        return df_sorted.reindex(full_index)

    def load_data(self, data_file):
        temp_df = data_file.df.copy()
        temp_df = temp_df.clip(lower=1e-4)
        temp_df.index = pd.to_datetime(temp_df.index, errors='coerce')
        self.df = temp_df[temp_df.index.notna()]
        self.diams = np.array(data_file.diameters)
        self._date_nums = mdates.date2num(self.df.index)     # reused by every zoom event
        
        # Shown for information: leaving it alone means the real per-bin widths
        # are used, typing a different number forces that uniform width instead.
        self.dlogdp_input.setText(f"{float(np.mean(dlogdp_per_bin(self.diams))):.4f}")
        
        self.update_top()
        self.update_bottom()

    def update_top(self):
        if self.df is None or self.df.empty: return

        if self.cbar: self.cbar.remove()
        self.cbar = None
        if self.zoom_cid: self.ax_contour.callbacks.disconnect(self.zoom_cid)
            
        self.ax_contour.set_yscale('linear') 
        self.ax_line.set_yscale('linear')
        self.ax_contour.clear()
        self.ax_line.clear()
        self.line_total_n = None

        plot_df = self._downsample_for_mesh(self._insert_time_gaps_for_plot(self.df))
        pnsd = plot_df.to_numpy(dtype=float)
        pnsd_safe = np.where(np.isnan(pnsd), np.nan, np.clip(pnsd, 1e-4, None))
        if not np.any(np.isfinite(pnsd_safe)):
            return

        v_min = float(self.cbar_min.text()) if self.cbar_min.text() else 1.0
        v_max = float(self.cbar_max.text()) if self.cbar_max.text() else float(np.nanmax(pnsd_safe))
        if v_max <= v_min: v_max = v_min * 10

        cmap = matplotlib.colormaps[self.cmap_combo.currentText()].copy()
        cmap.set_bad(color='white')
        
        mesh = self.ax_contour.pcolormesh(plot_df.index, self.diams, pnsd_safe.T, 
                                   cmap=cmap, 
                                   shading='auto', 
                                   norm=LogNorm(vmin=v_min, vmax=v_max))
        
        # Reduced padding so it doesn't compress the plot width!
        cb_ax = self.ax_contour.inset_axes([0.02, 0.80, 0.25, 0.04]) 
        
        self.cbar = self.fig_top.colorbar(mesh, cax=cb_ax, orientation='horizontal')    
        self.cbar.set_label(r'$\mathrm{dN/dlogD_p \ (cm^{-3})}$', size=10)            
        
        # Move ticks and label to the top of the little bar so they are easy to read
        self.cbar.ax.xaxis.set_ticks_position('top')
        self.cbar.ax.xaxis.set_label_position('top')
        self.cbar.ax.tick_params(labelsize=8)
        
        # Add a semi-transparent white background to the colorbar axes so the text stands out
        cb_ax.patch.set_facecolor('white')
        cb_ax.patch.set_alpha(0.7)
        
        self.ax_contour.set_yscale('log')
        self.ax_contour.set_xlim(plot_df.index.min(), plot_df.index.max())
        self.ax_contour.xaxis_date()
        
        locator = mdates.AutoDateLocator()
        self.ax_contour.xaxis.set_major_locator(locator)
        self.ax_contour.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        self.ax_contour.set_ylabel("Diameter (nm)")

        if self._dlogdp_enabled():
            total_n = integrate_pnsd(pnsd, resolve_dlogdp(self.diams, self.dlogdp_input.text()))
        else:
            total_n = np.zeros(len(pnsd))
        all_missing_rows = np.all(np.isnan(pnsd), axis=1)
        total_n = total_n.astype(float)
        total_n[all_missing_rows] = np.nan
        self.total_n_series = pd.Series(total_n, index=plot_df.index)
        
        self._update_top_line()
        
        self.zoom_cid = self.ax_contour.callbacks.connect('xlim_changed', self.on_zoom)
        self.fig_top.tight_layout()
        self.canvas_top.draw()

    def _update_top_line(self, xlim=None):
        if self.total_n_series is None or self.total_n_series.empty: return
        
        if xlim is None: delta_days = (self.total_n_series.index.max() - self.total_n_series.index.min()).days
        else: delta_days = xlim[1] - xlim[0]

        if delta_days > 365: resampled = self.total_n_series.resample('ME').mean()
        elif delta_days > 31: resampled = self.total_n_series.resample('D').mean()
        else: resampled = self.total_n_series

        x_data = mdates.date2num(resampled.index)
        y_data = resampled.values

        if self.line_total_n is None:
            self.line_total_n, = self.ax_line.plot(x_data, y_data, color='blue', alpha=0.8, linewidth=0.8)
            self.ax_line.set_ylabel(r'$\mathrm{Total \ N \ (cm^{-3})}$', color='blue', labelpad=20, rotation=90) 
            self.ax_line.yaxis.set_label_position("right")                        
            self.ax_line.tick_params(axis='y', colors='blue')
        else:
            self.line_total_n.set_xdata(x_data)
            self.line_total_n.set_ydata(y_data)
            
        if self.btn_log_top.isChecked(): self.ax_line.set_yscale('log')
        else: self.ax_line.set_yscale('linear')
            
        self.ax_line.relim()
        self.ax_line.autoscale_view(scalex=False, scaley=True)
        self.canvas_top.draw_idle()

    def on_zoom(self, ax):
        """Queue a refresh of the lower plots rather than recomputing per event.

        A single drag or wheel gesture emits many xlim_changed signals, and each
        one re-fits the lognormal modes; doing that per event is what makes
        zooming feel stuck on a large dataset.
        """
        if self.df is None: return
        self._pending_xlim = ax.get_xlim()
        self._zoom_timer.start(120)

    def _apply_zoom(self):
        if self.df is None or self._pending_xlim is None: return
        xlim = self._pending_xlim

        self._update_top_line(xlim)

        idx_start = np.searchsorted(self._date_nums, xlim[0])
        idx_end = np.searchsorted(self._date_nums, xlim[1])
        self.update_bottom(int(max(0, idx_start)), int(min(len(self._date_nums) - 1, idx_end)))

    def update_bottom(self, idx_start=None, idx_end=None):                   
        if self.df is None or self.df.empty: return                          
        if isinstance(idx_start, bool): idx_start = None                     
            
        try: density = float(self.density_input.text())
        except ValueError: density = 1.5
        dlogdp = resolve_dlogdp(self.diams, self.dlogdp_input.text())        # per-bin widths

        if idx_start is None: idx_start, idx_end = 0, len(self.df) - 1       
        subset = self.df.iloc[idx_start:idx_end]                             
        if subset.empty: return                                              
        
        for ax in [self.ax_dist_num, self.ax_dist_mass, self.ax_diur_num, self.ax_diur_mass]:
            # Back to linear on both axes first: clearing a log axis resets its
            # limits to (0, 1), which warns and leaves the axis in a bad state.
            ax.set_yscale('linear')
            ax.set_xscale('linear')
            ax.clear()
            
        volume = (np.pi / 6) * (self.diams ** 3)                             
        mass_factor = density * 1e-9                                         
        
        # --- Plot 1: Average Size Distributions & Fitting ---
        avg_pnsd = subset.mean(axis=0).to_numpy()                            
        avg_mass = avg_pnsd * volume * mass_factor                           
        
        # Changed to Standard Error (.sem)
        se_pnsd = subset.sem(axis=0).to_numpy()
        se_mass = se_pnsd * volume * mass_factor
        
        self.ax_dist_num.plot(self.diams, avg_pnsd, color='blue')            
        self.ax_dist_num.fill_between(self.diams, np.clip(avg_pnsd - se_pnsd, 1e-4, None), avg_pnsd + se_pnsd, color='blue', alpha=0.2)
        
        self.ax_dist_num.set_xscale('log')                                   
        self.ax_dist_num.set_ylabel(r'$\mathrm{dN/dlogD_p \ (cm^{-3})}$', color='blue') 
        self.ax_dist_num.tick_params(axis='y', colors='blue')                
        if self.btn_log_dist.isChecked(): self.ax_dist_num.set_yscale('log')                               
        
        self.ax_dist_mass.plot(self.diams, avg_mass, color='#2ca02c')   
        self.ax_dist_mass.fill_between(self.diams, np.clip(avg_mass - se_mass, 1e-9, None), avg_mass + se_mass, color='#2ca02c', alpha=0.2)
        
        self.ax_dist_mass.set_ylabel(r'$\mathrm{dM/dlogD_p \ (\mu g \ m^{-3})}$', color='#2ca02c', labelpad=15) 
        self.ax_dist_mass.yaxis.set_label_position("right")                  
        self.ax_dist_mass.tick_params(axis='y', colors='#2ca02c')            
        if self.btn_log_dist.isChecked(): self.ax_dist_mass.set_yscale('log')                              
        
        # Multi-Gaussian Curve Fitting with strict width bounds
        x_log10 = np.log10(self.diams)
        y_data = avg_pnsd
        max_y = np.max(y_data)

        # Zooming into a gap gives an all-NaN window, and a single-bin file gives
        # a degenerate fit range. Either used to stop update_bottom part way with
        # no message, leaving the lower plots half drawn.
        fittable = (np.isfinite(y_data).sum() >= 4 and np.isfinite(max_y)
                    and max_y > 0 and len(np.unique(x_log10)) > 3)
        if not fittable:
            # A log axis with nothing positive on it refuses to draw at all, so
            # put these back to linear before touching the canvas.
            for ax in (self.ax_dist_num, self.ax_dist_mass, self.ax_diur_num, self.ax_diur_mass):
                ax.set_xscale('linear')
                ax.set_yscale('linear')
            self.table.setRowCount(1)
            self.table.setVerticalHeaderLabels(["Status"])
            self.table.setItem(0, 0, QTableWidgetItem("Not enough data in view"))
            self.canvas_dist.draw()
            self.canvas_diur.draw()
            return

        # Seed the solver with 3 standard atmospheric modes instead of using find_peaks
        # (Nucleation ~15nm, Aitken ~50nm, Accumulation ~150nm)
        guess_dps = [15.0, 50.0, 150.0]
        
        p0, bounds_low, bounds_up = [], [], []
        for dp in guess_dps:
            # Ensure the guess is within the instrument's measurement bounds
            dp_safe = np.clip(dp, self.diams.min() * 1.1, self.diams.max() * 0.9)
            idx = (np.abs(self.diams - dp_safe)).argmin()
            
            p0.extend([y_data[idx] + (max_y * 0.1), np.log10(dp_safe), 0.25])
            bounds_low.extend([0, np.min(x_log10), 0.05])
            # Sigma capped at 0.45. This allows modes to be broad enough to capture 
            # accumulation modes, but prevents a single mode from swallowing the whole plot.
            bounds_up.extend([max_y * 1.5, np.max(x_log10), 0.45]) 
            
        try:
            popt, _ = curve_fit(multi_lognormal, x_log10, y_data, p0=p0, bounds=(bounds_low, bounds_up), maxfev=5000)
        except RuntimeError:
            popt = p0 
            
        modes_info = []
        total_fit = np.zeros_like(x_log10)
        
        for i in range(0, len(popt), 3):
            H, mu, sigma = popt[i:i+3]
            # Only keep the mode if its height is > 2% of the peak concentration 
            # This automatically filters out "dead" or non-existent modes
            if H > max_y * 0.02:
                mode_dp = 10**mu
                modes_info.append((H, mode_dp))
                y_mode = H * np.exp(-((x_log10 - mu)**2) / (2 * sigma**2))
                self.ax_dist_num.plot(self.diams, y_mode, color='black', alpha=0.4, ls='--')
                total_fit += y_mode
                
        modes_info.sort(key=lambda x: x[1])
        
        # Plot the combined sum of the fitted modes as a thick dotted line
        # If this dotted line matches your blue line, the fit is perfect!
        if np.any(total_fit > 0):
            self.ax_dist_num.plot(self.diams, total_fit, color='black', alpha=0.8, ls=':', lw=1.5, label="Total Fit")

        # --- Plot 2: Average Diurnal Cycle ---
        subset_n = (subset * dlogdp).sum(axis=1)
        subset_mass = (subset * volume * mass_factor * dlogdp).sum(axis=1)
        
        diurnal_n = subset_n.groupby(subset_n.index.hour).mean().clip(lower=1e-4)             
        diurnal_m = subset_mass.groupby(subset_mass.index.hour).mean().clip(lower=1e-9)       
        
        # Changed to Standard Error (.sem)
        se_n = subset_n.groupby(subset_n.index.hour).sem().fillna(0)
        se_m = subset_mass.groupby(subset_mass.index.hour).sem().fillna(0)
        
        self.ax_diur_num.plot(diurnal_n.index, diurnal_n.values, color='blue') 
        self.ax_diur_num.fill_between(diurnal_n.index, np.clip(diurnal_n - se_n, 1e-4, None), diurnal_n + se_n, color='blue', alpha=0.2)
        self.ax_diur_num.set_ylabel(r'$\mathrm{Mean \ N \ (cm^{-3})}$', color='blue') 
        self.ax_diur_num.tick_params(axis='y', colors='blue')                
        self.ax_diur_num.set_xticks(np.arange(0, 25, 4))                     
        #if self.btn_log_diur.isChecked(): self.ax_diur_num.set_yscale('log')                               
        
        self.ax_diur_mass.plot(diurnal_m.index, diurnal_m.values, color='#2ca02c') 
        self.ax_diur_mass.fill_between(diurnal_m.index, np.clip(diurnal_m - se_m, 1e-9, None), diurnal_m + se_m, color='#2ca02c', alpha=0.2)
        self.ax_diur_mass.set_ylabel(r'$\mathrm{Mean \ Mass \ (\mu g \ m^{-3})}$', color='#2ca02c', labelpad=15) 
        self.ax_diur_mass.yaxis.set_label_position("right")                  
        self.ax_diur_mass.tick_params(axis='y', colors='#2ca02c')            
        #if self.btn_log_diur.isChecked(): self.ax_diur_mass.set_yscale('log')                              
        
        mean_tot_n = subset_n.mean()                                         
        mean_tot_m = subset_mass.mean()                                      
        global_mode_dp = self.diams[np.argmax(avg_pnsd)]                            
        global_mean_dp = np.average(self.diams, weights=avg_pnsd) if np.sum(avg_pnsd) > 0 else 0 

        v_labels = ["Mean N (cm⁻³)", "Mean Mass (μg m⁻³)", "Global Mode Dp (nm)", "Global Mean Dp (nm)"]
        for i in range(len(modes_info)):
            v_labels.extend([f"Mode {i+1} Height", f"Mode {i+1} Dp (nm)"])
            
        self.table.setRowCount(len(v_labels))
        self.table.setVerticalHeaderLabels(v_labels)

        self.table.setItem(0, 0, QTableWidgetItem(f"{mean_tot_n:.1f}"))      
        self.table.setItem(1, 0, QTableWidgetItem(f"{mean_tot_m:.2f}"))      
        self.table.setItem(2, 0, QTableWidgetItem(f"{global_mode_dp:.1f}"))         
        self.table.setItem(3, 0, QTableWidgetItem(f"{global_mean_dp:.1f}"))         

        row_idx = 4
        for H, dp in modes_info:
            self.table.setItem(row_idx, 0, QTableWidgetItem(f"{H:.1f}"))
            self.table.setItem(row_idx+1, 0, QTableWidgetItem(f"{dp:.1f}"))
            row_idx += 2

        self.fig_dist.tight_layout()
        self.fig_diur.tight_layout()
        
        self.canvas_dist.draw()
        self.canvas_diur.draw()

    def export_plots(self):
        def restore_top():
            self.top_layout.insertWidget(1, self.canvas_top, stretch=1)
            self.fig_top.set_size_inches(self.orig_top_size)
            self.canvas_top.draw()
            
        self.orig_top_size = self.fig_top.get_size_inches()
        dlg_top = ExportDialog("Contour Plot", self.canvas_top, self.fig_top, restore_top, self)
        dlg_top.exec()

        def restore_dist():
            self.bot_plots_layout.insertWidget(0, self.canvas_dist, stretch=2)
            self.fig_dist.set_size_inches(self.orig_dist_size)
            self.canvas_dist.draw()
            
        self.orig_dist_size = self.fig_dist.get_size_inches()
        dlg_dist = ExportDialog("Size Distribution", self.canvas_dist, self.fig_dist, restore_dist, self)
        dlg_dist.exec()

        def restore_diur():
            self.bot_plots_layout.insertWidget(1, self.canvas_diur, stretch=2)
            self.fig_diur.set_size_inches(self.orig_diur_size)
            self.canvas_diur.draw()
            
        self.orig_diur_size = self.fig_diur.get_size_inches()
        dlg_diur = ExportDialog("Diurnal Cycle", self.canvas_diur, self.fig_diur, restore_diur, self)
        dlg_diur.exec()
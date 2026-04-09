import math
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import LogNorm
from matplotlib import rcParams
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QComboBox, QSplitter,
                             QPushButton, QTextEdit, QScrollArea, QCheckBox,
                             QFileDialog, QMessageBox, QDialog)
from PyQt6.QtCore import Qt

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Georgia', 'Times New Roman']
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['figure.facecolor'] = '#fff1e5'
rcParams['axes.facecolor'] = '#fff1e5'
rcParams['savefig.facecolor'] = '#fff1e5'

# --- Custom Export Dialog ---
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
        self.resize(w, h)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.val_w.setText(str(self.canvas.width()))
        self.val_h.setText(str(self.canvas.height()))

    def apply_size(self):
        try:
            w, h = int(self.val_w.text()), int(self.val_h.text())
            self.resize(w, h + 50)
        except ValueError:
            pass

    def save_plot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Plot", "", "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)")
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


class TrendPanel(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.df = None
        self.diams = None
        
        layout = QVBoxLayout(self)
        
        intro_layout = QHBoxLayout()
        intro_lbl = QLabel("Trend Analysis: Explore temporal changes and repeating cycles.")
        intro_lbl.setStyleSheet("font-size: 14px; font-weight: bold;")
        
        info_text = r"""
        <h3>Trend Analysis Models</h3>
        
        <b>1. Linear Regression (Ordinary Least Squares)</b><br>
        Fits a linear trend line to the data using the equation:<br>
        <i>Y = slope &times; X + intercept</i><br>
        • <b>Slope:</b> The average rate of change per year.<br>
        • <b>R&sup2; (Coefficient of Determination):</b> The proportion of the variance in the dependent variable that is predictable from the independent variable (0 to 1).<br>
        • <b>p-value:</b> Indicates the statistical significance of the trend. A value &lt; 0.05 is typically considered significant.<br><br>
        
        <b>2. Mann-Kendall Trend Test</b><br>
        A non-parametric statistical test used to assess whether there is a monotonic upward or downward trend over time. It does not assume a linear trend or a normal distribution.<br>
        • <b>Tau (&tau;):</b> Kendall's rank correlation coefficient (-1 to 1). Measures the strength of the monotonic relationship.<br>
        • <b>p-value:</b> Assesses the significance of the trend. If &lt; 0.05, the trend is considered statistically significant.<br><br>
        
        <b>3. Seasonal Decomposition (Additive)</b><br>
        Separates the time series (<i>Y<sub>t</sub></i>) into three distinct components:<br>
        <i>Y<sub>t</sub> = T<sub>t</sub> + S<sub>t</sub> + R<sub>t</sub></i><br>
        • <b>Trend (T<sub>t</sub>):</b> The long-term progression of the series.<br>
        • <b>Seasonal (S<sub>t</sub>):</b> Repeating short-term cycles (e.g., daily or annual patterns).<br>
        • <b>Residual (R<sub>t</sub>):</b> The random noise or anomalies remaining after extracting the trend and seasonality.
        The slope, R^sup2;, and p-value of the trend component are shown in the right hand side.<br><br>

        <b>Gallery:</b><br>
        Breaks down the average diurnal cycles by your chosen timeframe (Year, Season, Month) so you can compare how daily patterns shift over longer periods.
        """
        intro_layout.addWidget(intro_lbl)
        
        self.btn_info = QPushButton("ℹ️")
        self.btn_info.setFixedSize(24, 24)
        self.btn_info.clicked.connect(lambda: QMessageBox.information(self, "Trend Analysis Info", info_text))
        intro_layout.addWidget(self.btn_info)
        
        intro_layout.addStretch()
        layout.addLayout(intro_layout)
        
        ctrl_layout1 = QHBoxLayout()
        ctrl_layout1.addWidget(QLabel("Size Fractions (nm):"))
        self.size_bins = QLineEdit("10-50, 50-800")
        ctrl_layout1.addWidget(self.size_bins)
        
        self.chk_mass = QCheckBox("Plot Total Mass (Secondary Y)")
        ctrl_layout1.addWidget(self.chk_mass)
        
        self.chk_log_num = QCheckBox("Log Scale (N)")
        ctrl_layout1.addWidget(self.chk_log_num)

        ctrl_layout1.addWidget(QLabel("Time Avg:"))
        self.time_avg = QComboBox()
        self.time_avg.addItems(["Original", "Hourly", "Daily", "Monthly"])
        ctrl_layout1.addWidget(self.time_avg)
        
        ctrl_layout1.addWidget(QLabel("Trend Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Linear Regression", "Mann-Kendall", "Decomposition"])
        ctrl_layout1.addWidget(self.model_combo)
        
        ctrl_layout1.addWidget(QLabel("Gallery:"))
        self.breakdown = QComboBox()
        self.breakdown.addItems(["Year", "Season", "Month"])
        ctrl_layout1.addWidget(self.breakdown)
        layout.addLayout(ctrl_layout1)
        
        ctrl_layout2 = QHBoxLayout()
        ctrl_layout2.addWidget(QLabel("Colour Map:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["turbo", "viridis", "plasma", "inferno"])
        ctrl_layout2.addWidget(self.cmap_combo)
        
        ctrl_layout2.addWidget(QLabel("Min Colour:"))
        self.cbar_min = QLineEdit("1")
        ctrl_layout2.addWidget(self.cbar_min)
        
        ctrl_layout2.addWidget(QLabel("Max Colour:"))
        self.cbar_max = QLineEdit("")
        ctrl_layout2.addWidget(self.cbar_max)
        
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.clicked.connect(self.run_analysis)
        ctrl_layout2.addWidget(self.run_btn)
        
        self.export_btn = QPushButton("Export Plots")
        self.export_btn.clicked.connect(self.export_plots)
        ctrl_layout2.addWidget(self.export_btn)
        layout.addLayout(ctrl_layout2)
        
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        top_widget = QWidget()
        self.top_layout = QHBoxLayout(top_widget)
        self.fig_trend = Figure(figsize=(6, 3))
        self.canvas_trend = FigureCanvasQTAgg(self.fig_trend)
        self.top_layout.addWidget(self.canvas_trend, stretch=3)
        
        self.stats_out = QTextEdit()
        self.stats_out.setReadOnly(True)
        self.top_layout.addWidget(self.stats_out, stretch=1)
        splitter.addWidget(top_widget)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.gallery_widget = QWidget()
        self.gallery_layout = QVBoxLayout(self.gallery_widget)
        self.fig_gal = Figure()
        self.canvas_gal = FigureCanvasQTAgg(self.fig_gal)
        self.gallery_layout.addWidget(self.canvas_gal)
        
        self.scroll.setWidget(self.gallery_widget)
        splitter.addWidget(self.scroll)
        layout.addWidget(splitter)

    def load_data(self, data_file):
        self.df = data_file.df
        self.diams = np.array(data_file.diameters)

    def run_analysis(self):
        if self.df is None: return
        
        bin_texts = self.size_bins.text().split(',')
        bin_ranges = []
        for bt in bin_texts:
            parts = bt.split('-')
            if len(parts) == 2:
                try: bin_ranges.append((float(parts[0].strip()), float(parts[1].strip())))
                except ValueError: pass
        
        if not bin_ranges: return
        
        log_diams = np.log10(self.diams)
        dlogdp = np.mean(np.diff(log_diams)) if len(log_diams) > 1 else 1.0
        calc_mass = self.chk_mass.isChecked()
        
        resample_map = {"Hourly": "h", "Daily": "D", "Monthly": "ME"}
        avg_choice = self.time_avg.currentText()
        
        series_dict = {}
        for dmin, dmax in bin_ranges:
            mask = (self.diams >= dmin) & (self.diams <= dmax)
            if not np.any(mask): continue
            
            n_cm3 = self.df.iloc[:, mask] * dlogdp
            subset_num = n_cm3.sum(axis=1)
            
            if avg_choice in resample_map:
                subset_num = subset_num.resample(resample_map[avg_choice]).mean().dropna()
                
            series_dict[f"{dmin}-{dmax}nm"] = subset_num
            
        total_mass_series = None
        if calc_mass:
            d_m = self.diams * 1e-9
            vol_m3 = (np.pi / 6.0) * (d_m ** 3)
            mass_kg = (self.df * dlogdp) * 1e6 * vol_m3 * 1.5e3
            total_mass_ug = (mass_kg * 1e9).sum(axis=1)
            
            if avg_choice in resample_map:
                total_mass_ug = total_mass_ug.resample(resample_map[avg_choice]).mean().dropna()
                
            total_mass_series = total_mass_ug

        self.plot_trend(series_dict, total_mass_series)
        self.plot_gallery(dlogdp)

    def plot_trend(self, series_dict, total_mass_series):
        self.fig_trend.clear()
        model = self.model_combo.currentText()
        colours = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
        stats_text = ""
        calc_mass = self.chk_mass.isChecked()
        
        if model in ["Linear Regression", "Mann-Kendall"]:
            ax = self.fig_trend.add_subplot(111)
            ax2 = ax.twinx() if total_mass_series is not None else None
            
            if ax2:
                ax.set_zorder(ax2.get_zorder() + 1)
                ax.patch.set_visible(False)
            
            for i, (name, num_vals) in enumerate(series_dict.items()):
                c = colours[i % len(colours)]
                x_days = mdates.date2num(num_vals.index)
                x_years = x_days / 365.25
                
                ax.plot(num_vals.index, num_vals.values, alpha=0.4, color=c, marker='.', ls='', label=f"{name}", zorder=3)
                
                if model == "Linear Regression":
                    res_num = stats.linregress(x_years, num_vals.values)
                    trend_num = res_num.intercept + res_num.slope * x_years
                    ax.plot(num_vals.index, trend_num, color=c, lw=2.5, zorder=10, label=f"{name} Trend")
                    stats_text += f"[{name}]\nSlope: {res_num.slope:.2e} cm⁻³/yr\nR²: {res_num.rvalue**2:.3f}\np: {res_num.pvalue:.2e}\n\n"
                    
                elif model == "Mann-Kendall":
                    tau, p_val = stats.kendalltau(x_years, num_vals.values)
                    sig = 'Yes' if p_val < 0.05 else 'No'
                    stats_text += f"[{name}]\nTau: {tau:.4f}\np: {p_val:.2e}\nSig? {sig}\n\n"

            if total_mass_series is not None:
                mass_vals = total_mass_series.values
                x_days = mdates.date2num(total_mass_series.index)
                x_years = x_days / 365.25
                
                ax2.plot(total_mass_series.index, mass_vals, alpha=0.4, color='gray', marker='.', ls='', label="Total Mass", zorder=1)
                
                if model == "Linear Regression":
                    res_mass = stats.linregress(x_years, mass_vals)
                    trend_mass = res_mass.intercept + res_mass.slope * x_years
                    ax2.plot(total_mass_series.index, trend_mass, color='black', ls='--', lw=2.5, zorder=5, label="Mass Trend")
                    stats_text += f"[Total Mass]\nSlope: {res_mass.slope:.2e} μg m⁻³/yr\nR²: {res_mass.rvalue**2:.3f}\np: {res_mass.pvalue:.2e}\n\n"
                elif model == "Mann-Kendall":
                    tau_m, p_val_m = stats.kendalltau(x_years, mass_vals)
                    sig_m = 'Yes' if p_val_m < 0.05 else 'No'
                    stats_text += f"[Total Mass]\nTau: {tau_m:.4f}\np: {p_val_m:.2e}\nSig? {sig_m}\n\n"

            if self.chk_log_num.isChecked(): ax.set_yscale('log')
            
            ax.set_ylabel(r"Number Conc. ($cm^{-3}$)")
            if ax2: ax2.set_ylabel(r"Total Mass ($\mu g \ m^{-3}$)", color='black')
            
            lines, labels = ax.get_legend_handles_labels()
            if ax2:
                lines2, labels2 = ax2.get_legend_handles_labels()
                lines += lines2; labels += labels2
            ax.legend(lines, labels, loc='best', fontsize=8)
            
            self.stats_out.setText(stats_text)
            
        elif model == "Decomposition":
            period_map = {"Hourly": 24, "Daily": 365, "Monthly": 12, "Original": 24}
            period = period_map.get(self.time_avg.currentText(), 24)
            axes = self.fig_trend.subplots(4, 1, sharex=True)
            
            for i, (name, num_vals) in enumerate(series_dict.items()):
                c = colours[i % len(colours)]
                try:
                    clean_series = num_vals.interpolate(method='time').bfill().ffill()
                    dec = seasonal_decompose(clean_series, model='additive', period=period)
                    
                    axes[0].plot(clean_series.index, clean_series, color=c, label=f"{name}")
                    axes[1].plot(dec.trend.index, dec.trend, color=c)
                    axes[2].plot(dec.seasonal.index, dec.seasonal, color=c)
                    axes[3].plot(dec.resid.index, dec.resid, color=c, marker='.', ls='', alpha=0.5)
                    
                    valid_trend = dec.trend.dropna()
                    if len(valid_trend) > 2:
                        x_years = mdates.date2num(valid_trend.index) / 365.25
                        res_dec = stats.linregress(x_years, valid_trend.values)
                        stats_text += f"[{name} - Deseasonalised Trend]\nSlope: {res_dec.slope:.2e} cm⁻³/yr\nR²: {res_dec.rvalue**2:.3f}\np: {res_dec.pvalue:.2e}\n\n"
                    else:
                        stats_text += f"[{name}] Not enough data for trend regression.\n\n"
                        
                except ValueError as e:
                    stats_text += f"[{name}] Failed: {e}\n\n"
            
            axes[0].set_ylabel("Original"); axes[0].legend(loc='best', fontsize=8)
            axes[1].set_ylabel("Trend")
            axes[2].set_ylabel("Seasonal")
            axes[3].set_ylabel("Residual")
            self.stats_out.setText(stats_text + "(Mass omitted for clarity in Decomp)")

        self.fig_trend.tight_layout(pad=2.0)
        self.canvas_trend.draw()

    def plot_gallery(self, dlogdp):
        self.fig_gal.clf()
        mode = self.breakdown.currentText()
        if mode == "Year": groups = self.df.groupby(self.df.index.year)
        elif mode == "Month": groups = self.df.groupby(self.df.index.month)
        else: groups = self.df.groupby((self.df.index.month % 12 // 3) + 1)
        
        n_plots = len(groups)
        if n_plots == 0: return
        
        cols = 2
        rows = math.ceil(n_plots / cols)
        
        # Give a little extra height to accommodate the color bar at the bottom
        req_height = max(450, 250 * rows + 50)
        self.canvas_gal.setMinimumHeight(req_height)
        self.fig_gal.set_figheight(req_height / self.fig_gal.dpi)
        
        axes = self.fig_gal.subplots(rows, cols, sharex=True, sharey=True)
        if n_plots == 1: axes = np.array([axes])
        axes = axes.flatten()
        
        labels = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Autumn"}
        try: v_min = float(self.cbar_min.text())
        except ValueError: v_min = 1.0
        v_max = v_min
        max_tot_n = 0
        diurnals = []
        
        for name, group in groups:
            diurnal = group.groupby(group.index.hour).mean()
            v_max = max(v_max, diurnal.to_numpy().max())
            tot_n = np.sum(diurnal.to_numpy(), axis=1) * dlogdp
            max_tot_n = max(max_tot_n, tot_n.max())
            diurnals.append((name, diurnal, tot_n))
            
        try: v_max = float(self.cbar_max.text())
        except ValueError: pass
        if v_max <= v_min: v_max = v_min * 10
        cmap = self.cmap_combo.currentText()
        
        im = None
        
        for i, ax in enumerate(axes):
            if i >= n_plots:
                ax.set_visible(False)
                continue
                
            name, diurnal, tot_n = diurnals[i]
            pnsd_safe = np.clip(diurnal.to_numpy(), 1e-4, None)
            
            im = ax.pcolormesh(diurnal.index, self.diams, pnsd_safe.T,
                          cmap=cmap, shading='auto',
                          norm=LogNorm(vmin=v_min, vmax=v_max))
            ax.set_yscale('log')
            
            title = labels.get(name, str(name)) if mode == "Season" else str(name)
            ax.set_title(f"{mode}: {title}", fontsize=10)
            ax.set_xticks(np.arange(0, 25, 6))
            
            if i % cols == 0: ax.set_ylabel("Dp (nm)")
            if i >= (rows - 1) * cols: ax.set_xlabel("Hour")
            
            ax2 = ax.twinx()
            ax2.plot(diurnal.index, tot_n, color='red', alpha=0.8)
            ax2.set_ylim(0, max_tot_n * 1.05)
            
            if i % cols == cols - 1 or i == n_plots - 1:
                ax2.set_ylabel(r"Total N ($cm^{-3}$)", color='red')
            else:
                ax2.set_yticklabels([])
            ax2.tick_params(axis='y', colors='red')
            
        # Draw the sleeker, horizontal color bar at the bottom
        if im:
            cbar = self.fig_gal.colorbar(im, ax=axes.ravel().tolist(), 
                                         orientation='horizontal', 
                                         pad=0.25,      # Pushes it away from the plots
                                         fraction=0.03, # Makes it take up less overall space
                                         aspect=50,     # Makes the bar thinner
                                         shrink=0.6)    # Shortens it so it doesn't span edge-to-edge
            cbar.set_label(r"dN/dlogDp ($cm^{-3}$)")
            
        # Force the figure to leave physical space at the bottom for the new color bar
        self.fig_gal.subplots_adjust(hspace=0.4, wspace=0.15, bottom=0.15)
        self.canvas_gal.draw()

    def export_plots(self):
        def restore_trend():
            self.top_layout.insertWidget(0, self.canvas_trend, stretch=3)
            self.fig_trend.set_size_inches(self.orig_trend_size)
            self.canvas_trend.draw()
            
        self.orig_trend_size = self.fig_trend.get_size_inches()
        dlg_trend = ExportDialog("Trend Analysis", self.canvas_trend, self.fig_trend, restore_trend, self)
        dlg_trend.exec()

        def restore_gal():
            self.gallery_layout.addWidget(self.canvas_gal)
            self.fig_gal.set_size_inches(self.orig_gal_size)
            self.canvas_gal.draw()
            
        self.orig_gal_size = self.fig_gal.get_size_inches()
        dlg_gal = ExportDialog("Diurnal Gallery", self.canvas_gal, self.fig_gal, restore_gal, self)
        dlg_gal.exec()
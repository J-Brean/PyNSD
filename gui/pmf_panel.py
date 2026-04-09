import os                                                                # For path manipulation
import shutil                                                            # For copying the key file
import subprocess                                                        # For running external .exe
import copy                                                              # For copying colormaps
import re                                                                # For parsing the Fortran LOG files
import pandas as pd                                                      # For data handling
import numpy as np                                                       # For maths operations
import matplotlib.pyplot as plt                                          # Global Matplotlib import
import matplotlib as mpl                                                 # Global Matplotlib settings
import seaborn as sns                                                    # Global Seaborn import
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg         # Embedded plotting for Qt
from matplotlib.figure import Figure                                     # Figure object 
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,          # Core UI layouts
                             QPushButton, QFileDialog, QLabel,           # Core UI elements
                             QLineEdit, QComboBox, QDoubleSpinBox,       # Inputs
                             QSlider, QGroupBox, QGridLayout,            # Advanced UI
                             QMessageBox, QTableWidget, QSpinBox,        # Dialogues and tables
                             QTableWidgetItem, QDialog, QProgressBar,    # Dialogues and bars
                             QApplication, QCheckBox, QTabWidget)        # App control and tabs
from PyQt6.QtCore import Qt, QSettings                                   # For alignment flags and memory
from PyQt6.QtGui import QFont                                            # For monospace fonts
from utils.pmf_ini_generator import generate_pmf_ini                     # External INI generator

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

class TabbedVisualizer(QDialog):
    def __init__(self, panel, parent=None):
        super().__init__(parent)                                         
        self.panel = panel                                               
        
        n_rows = len(self.panel.g_matrix)
        n_cols = len(self.panel.f_matrix)
        q_ratio = self.panel._get_q_ratio(self.panel.current_factors, self.panel.current_fpeak, n_rows, n_cols)
        
        active_text = self.panel.combo_fpeak.currentText()
        self.setWindowTitle(f"PyNSD Visualisation Suite ({active_text}) | Q/Qexp={q_ratio:.3f}") 
        self.resize(1400, 850)                                           
        
        self.g_number = self.panel.get_scaled_g()                        
        
        self.tabs = QTabWidget()                                         
        self.main_layout = QVBoxLayout(self)
        self.main_layout.addWidget(self.tabs)                            
        
        self._build_size_tab()                                           
        self._build_time_tab()                                           
        self._build_seasonal_tab()                                       
        self._build_dow_tab()                                            
        self._build_mass_tab()                                           
        self._build_resid_recon_tab()                                    
        self._build_diag_tab()                                           
        
        if self.panel.chk_wide_pmf.isChecked():                          
            self._build_wide_profiles_tab()                              
            self._build_widepmf_tab()                                    

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
            reshaped = raw_vals.reshape(n_hours, n_bins)
            return np.mean(reshaped, axis=0), np.std(reshaped, axis=0) / np.sqrt(n_hours)
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

    def _build_mass_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab); fig = Figure(figsize=(8, 6)); canvas = FigureCanvasQTAgg(fig); ax = fig.add_subplot(111)
        diams = self.panel.diams; mass_factor = (np.pi / 6) * (diams ** 3) * 1e-9
        for i in range(self.panel.current_factors):
            name = self.panel._get_factor_name(i)
            mean_n, se_n = self._get_mean_se_pnsd(i)                     
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
        
        self.df = None                                                   
        self.dates = None                                                
        self.diams = None                                                
        self.f_matrix = None                                             
        self.g_matrix = None                                             
        
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
        self.spin_fpeak_min.setRange(-2.0, 2.0); self.spin_fpeak_min.setValue(-1.0) 
        self.spin_fpeak_min.setToolTip("Step 5: Define the lower boundary for rotational ambiguity testing.")
        settings_layout.addWidget(self.spin_fpeak_min, 0, 4)             
        
        settings_layout.addWidget(QLabel("FPEAK Max:"), 1, 3)            
        self.spin_fpeak_max = QDoubleSpinBox()                           
        self.spin_fpeak_max.setRange(-2.0, 2.0); self.spin_fpeak_max.setValue(1.0) 
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
        
        action_layout = QHBoxLayout()                                    
        btn_vis = QPushButton("1. Open Visualisation Suite")             
        btn_vis.setStyleSheet("background-color: #9C27B0; color: white;")
        btn_vis.setToolTip("Step 8a: Inspect the selected array across dynamic plotting metrics.")
        btn_vis.clicked.connect(self.open_visualiser); action_layout.addWidget(btn_vis)
        
        btn_opt = QPushButton("2. Optimise Error Fraction (Q/Qexp ≈ 1)") 
        btn_opt.setStyleSheet("background-color: #2196F3; color: white;")
        btn_opt.setToolTip("Step 8b: Select an ideal factor count, then iteratively drive its Q/Qexp toward unity.")
        btn_opt.clicked.connect(self.optimize_coefficients); action_layout.addWidget(btn_opt)
        
        btn_rename = QPushButton("3. Rename Factors")                    
        btn_rename.setStyleSheet("background-color: #FF9800; color: white;")
        btn_rename.setToolTip("Step 8c: Assign final physical source nomenclature to factors.")
        btn_rename.clicked.connect(self.open_renamer); action_layout.addWidget(btn_rename)
        
        explore_layout.addLayout(action_layout)                          
        
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
        if text:                                                         
            f_match = re.search(r"Factors:\s*(\d+)", text)
            p_match = re.search(r"FPEAK:\s*([-0-9.]+)", text)
            
            if f_match and p_match:
                self.current_factors = int(f_match.group(1))             
                self.current_fpeak = float(p_match.group(1))             
                self.lbl_fpeak.setText(f"Active Data: {text}")           
                
                self.factor_names.clear()                                
                
                f_path = os.path.join(self.run_dir, f"F_FACTOR_{self.current_factors}_{self.current_fpeak}.TXT")
                g_path = os.path.join(self.run_dir, f"G_FACTOR_{self.current_factors}_{self.current_fpeak}.TXT")
                
                try:
                    if not os.path.exists(f_path) or not os.path.exists(g_path): 
                        raise FileNotFoundError(f"Cannot find:\n{f_path}\n{g_path}") 
                    
                    with open(f_path, 'r') as f:
                        f_vals = np.array(f.read().replace(',', ' ').split(), dtype=float)
                    self.f_matrix = pd.DataFrame(f_vals.reshape(self.current_factors, -1).T)
                    
                    with open(g_path, 'r') as f:
                        g_vals = np.array(f.read().replace(',', ' ').split(), dtype=float)
                    self.g_matrix = pd.DataFrame(g_vals.reshape(-1, self.current_factors))
                    
                    if self.diams is not None and len(self.f_matrix) == len(self.diams): 
                        self.f_matrix.index = self.diams                 
                    
                    if self.dates is not None:                           
                        if len(self.g_matrix) == len(self.dates):        
                            self.g_matrix.index = self.dates             
                        else:                                            
                            self.g_matrix.index = pd.date_range(self.dates.min(), periods=len(self.g_matrix), freq='D') 
                            
                except Exception as e:                                   
                    self.f_matrix = None                                 
                    self.g_matrix = None                                 
                    QMessageBox.critical(self, "Data Load Error", f"Could not load the data for {text}.\n\nReason:\n{e}")

    def open_renamer(self):
        if self.current_factors == 0:                                    
            QMessageBox.warning(self, "Error", "Please select a model from the Explorer first.") 
            return
        dialog = RenameDialog(self.current_factors, self.factor_names, self) 
        dialog.exec()                                                    

    def open_visualiser(self):
        if self.f_matrix is None or self.g_matrix is None:               
            QMessageBox.warning(self, "Error", "Data matrices not loaded. Make sure the batch ran successfully.") 
            return
        dialog = TabbedVisualizer(self, self)                            
        dialog.exec()                                                    

    def _get_factor_name(self, col_idx):
        raw_name = f"Factor {col_idx + 1}"                               
        return self.factor_names.get(raw_name, raw_name)
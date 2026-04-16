from __future__ import annotations

import numpy as np                                                           # Array operations
import pandas as pd                                                          # Data manipulation
import scipy.stats as stats                                                  # For KDE fitting
from scipy.signal import find_peaks                                          # For peak finding
import matplotlib.dates as mdates                                            # Date formatting
import matplotlib.cm as cm                                                   # Colormaps
from matplotlib.figure import Figure                                         # Matplotlib figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg              # Canvas elements
from matplotlib.colors import LogNorm                                        # Log scaling
from matplotlib import rcParams                                              # Global plot parameters

from PyQt6.QtCore import Qt, QThread, pyqtSignal                             # Core Qt
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,      # UI Layouts
                             QLineEdit, QComboBox, QSplitter, QTabWidget,    # UI Widgets
                             QPushButton, QProgressBar, QGroupBox, QScrollArea,
                             QDialog, QMessageBox, QFileDialog)

# ── Matplotlib global style ─────────────────────────────────────────────── #
rcParams['font.family'] = 'serif'                                            
rcParams['font.serif'] = ['Georgia', 'Times New Roman']                      
rcParams['figure.facecolor'] = '#fff1e5'                                     
rcParams['axes.facecolor'] = '#fff1e5'                                       
rcParams['savefig.facecolor'] = '#fff1e5'                                    

_FT_BG = '#fff1e5'                                                           
_ACCENT_COLORS = ['#0f6e56', '#185fa5', '#854f0b', '#a32d2d', '#533ab7']      

# ─────────────────────────────────────────────────────────────────────────── #
# Custom Export Dialog
# ─────────────────────────────────────────────────────────────────────────── #
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

# ─────────────────────────────────────────────────────────────────────────── #
# Worker Thread (Calculates Delta N)
# ─────────────────────────────────────────────────────────────────────────── #
class NanoRankingWorker(QThread):                                            
    finished = pyqtSignal(object)                                            
    error = pyqtSignal(str)                                                  

    def __init__(self, df: pd.DataFrame, diams: np.ndarray, kwargs: dict):   
        super().__init__()                                                   
        self.df = df                                                         
        self.diams = diams                                                   
        self.kwargs = kwargs                                                 

    def run(self):                                                           
        try:                                                                 
            res = self._process_ranking()                                    
            self.finished.emit(res)                                          
        except Exception as exc:                                             
            self.error.emit(str(exc))                                        

    def _process_ranking(self) -> dict:                                      
        dp_min = self.kwargs['dp_min']                                       
        dp_max = self.kwargs['dp_max']                                       
        bg_start, bg_end = self.kwargs['bg_start'], self.kwargs['bg_end']    
        act_start, act_end = self.kwargs['act_start'], self.kwargs['act_end']
        smooth_hrs = self.kwargs['smooth']                                   

        mask = (self.diams >= dp_min) & (self.diams <= dp_max)               
        if not mask.any(): raise ValueError("No bins in diameter range")     
        
        log_d = np.log10(self.diams)                                         
        dlogdp = np.mean(np.diff(log_d)) if len(log_d) > 1 else 0.1          
        
        n_target = self.df.iloc[:, mask].sum(axis=1) * dlogdp                
        n_smooth = n_target.rolling(f"{smooth_hrs}h", center=True).median()  
        
        daily_stats = []                                                     
        for date, group in n_smooth.groupby(n_smooth.index.date):            
            if len(group) < 12: continue                                     
            
            if bg_start > bg_end:                                            
                bg_mask = (group.index.hour >= bg_start) | (group.index.hour < bg_end)
            else:                                                            
                bg_mask = (group.index.hour >= bg_start) & (group.index.hour < bg_end)
                
            act_mask = (group.index.hour >= act_start) & (group.index.hour < act_end) 
            
            bg_data = group[bg_mask]                                         
            act_data = group[act_mask]                                       
            
            n_bg = bg_data.median() if len(bg_data) > 0 else np.nan          
            n_act = act_data.max() if len(act_data) > 0 else np.nan          
            
            daily_stats.append({                                             
                'date': pd.Timestamp(date),                                  
                'N_bg': n_bg,                                                
                'N_act': n_act,                                              
                'delta_N': n_act - n_bg                                      
            })                                                               

        results_df = pd.DataFrame(daily_stats).set_index('date').dropna()    
        results_df['rank'] = results_df['delta_N'].rank(pct=True) * 100      
        
        return {                                                             
            "daily_df": results_df,                                          
            "n_smooth": n_smooth,                                            
            "params": self.kwargs                                            
        }                                                                    

# ─────────────────────────────────────────────────────────────────────────── #
# Main Panel
# ─────────────────────────────────────────────────────────────────────────── #
class NanoRankingPanel(QWidget):                                             
    def __init__(self, parent=None):                                         
        super().__init__(parent)                                             
        self._df = None                                                      
        self._diams = None                                                   
        self._results = None                                                 
        self._build_ui()                                                     

    def _show_math_popup(self, topic: str):
        pop = QMessageBox(self)
        pop.setWindowTitle(f"Mathematics: {topic}")
        pop.setTextFormat(Qt.TextFormat.RichText)
        
        if topic == "Intensity":
            pop.setText(
                "<b>Formation Intensity (dN)</b><br><br>"
                "dN = max(N_active) - median(N_background)<br><br>"
                "This metric calculates the absolute intensity of an NPF event. It takes the maximum smoothed "
                "particle concentration in the target size range during the day (active window) and subtracts "
                "the median background concentration from the preceding night. This step removes the influence "
                "of pre-existing accumulation mode particles to isolate newly formed nanoparticles."
            )
        elif topic == "Gaussian KDE":
            pop.setText(
                "<b>KDE Mode Fitting</b><br><br>"
                "f(x) = (1 / nh) * SUM[ K((x - x_i) / h) ]<br><br>"
                "We fit a Gaussian Kernel Density Estimate (KDE) to the log10-transformed dN distribution. "
                "The algorithm then identifies the local maxima (modes) of this density curve. These modes "
                "represent naturally occurring clusters of event strengths, allowing us to classify events "
                "into distinct kinetic regimes (e.g., weak vs. strong)."
            )
        elif topic == "Percentiles":
            pop.setText(
                "<b>Percentile Ranking</b><br><br>"
                "Rank = CDF(dN) * 100<br><br>"
                "Each daily event is ranked against the empirical Cumulative Distribution Function (CDF) of "
                "the entire dataset. A rank of 90% means that the formation intensity on that day was stronger "
                "than 90% of all events recorded during the campaign. This standardizes intensity across extreme variations."
            )
        pop.exec()

    # --- Export Handlers ---
    def _export_meth(self):
        if self._results is None: return QMessageBox.warning(self, "No Data", "Calculate ranking first!")
        def restore():
            self._meth_wrapper.addWidget(self.canvas_method)
            self.canvas_method.draw()
        ExportDialog("Methodology Check", self.canvas_method, self.fig_method, restore, self).exec()

    def _export_perc(self):
        if self._results is None: return QMessageBox.warning(self, "No Data", "Calculate ranking first!")
        def restore():
            self._perc_layout.addWidget(self.canvas_perc)
            self.canvas_perc.draw()
        ExportDialog("Percentiles", self.canvas_perc, self.fig_perc, restore, self).exec()

    def _export_gaus(self):
        if self._results is None: return QMessageBox.warning(self, "No Data", "Calculate ranking first!")
        def restore():
            self._gaus_wrapper.addWidget(self.canvas_gauss)
            self.canvas_gauss.draw()
        ExportDialog("Gaussian Modes", self.canvas_gauss, self.fig_gauss, restore, self).exec()

    def _export_temp(self):
        if self._results is None: return QMessageBox.warning(self, "No Data", "Calculate ranking first!")
        def restore():
            self._temp_wrapper.addWidget(self.canvas_temp)
            self.canvas_temp.draw()
        ExportDialog("Temporal Climatology", self.canvas_temp, self.fig_temp, restore, self).exec()

    def _export_csv(self):
        if self._results is None: return QMessageBox.warning(self, "No Data", "Calculate ranking first!")
        path, _ = QFileDialog.getSaveFileName(self, "Save Nano Ranking Bins", "", "CSV Files (*.csv)")
        if not path: return
        
        res_df = self._results['daily_df']
        bins = [f"N_{i*5}_{(i+1)*5}" for i in range(20)]
        
        df_export = pd.DataFrame(index=res_df.index, columns=['date', 'percentile'] + bins)
        df_export['date'] = df_export.index.strftime('%Y-%m-%d')
        df_export['percentile'] = res_df['rank']
        
        for idx, row in res_df.iterrows():
            rank = row['rank']
            bin_idx = int(rank // 5)
            if bin_idx >= 20: bin_idx = 19  # In case rank == 100
            bin_name = bins[bin_idx]
            df_export.loc[idx, bin_name] = row['delta_N']
        
        df_export.to_csv(path, index=False)
        QMessageBox.information(self, "Success", "Nano ranking bins exported successfully!")

    # --- UI Builder ---
    def _build_ui(self):                                                     
        root = QVBoxLayout(self)                                             
        root.setContentsMargins(12, 12, 12, 8)                               
        root.addWidget(self._build_settings_box())                           
        
        self._progress_bar = QProgressBar()                                  
        self._progress_bar.setRange(0, 0)                                    
        self._progress_bar.setVisible(False)                                 
        self._progress_bar.setFixedHeight(6)                                 
        root.addWidget(self._progress_bar)                                   

        self._workspace = QTabWidget()                                       
        self._workspace.addTab(self._build_method_tab(), "① Methodology")    
        self._workspace.addTab(self._build_percentile_tab(), "② Percentiles")
        self._workspace.addTab(self._build_gaussian_tab(), "③ Gaussian Modes")
        self._workspace.addTab(self._build_temporal_tab(), "④ Temporal")     
        root.addWidget(self._workspace, stretch=1)                           

    def _build_settings_box(self) -> QGroupBox:                              
        box = QGroupBox("Nano Ranking Settings")                             
        layout = QHBoxLayout(box)                                            
        
        layout.addWidget(QLabel("D<sub>p</sub> Range (nm):"))                
        self._dp_min = QLineEdit("2.5")
        self._dp_min.setFixedWidth(40)      
        self._dp_max = QLineEdit("5.0")
        self._dp_max.setFixedWidth(40)      
        layout.addWidget(self._dp_min)
        layout.addWidget(QLabel("-"))        
        layout.addWidget(self._dp_max)                                       
        
        layout.addSpacing(15)                                                
        layout.addWidget(QLabel("Background (hrs):"))                        
        self._bg_start = QLineEdit("21")
        self._bg_start.setFixedWidth(30)   
        self._bg_end = QLineEdit("6")
        self._bg_end.setFixedWidth(30)        
        layout.addWidget(self._bg_start)
        layout.addWidget(QLabel("-"))      
        layout.addWidget(self._bg_end)                                       

        layout.addSpacing(15)                                                
        layout.addWidget(QLabel("Active (hrs):"))                            
        self._act_start = QLineEdit("6")
        self._act_start.setFixedWidth(30)  
        self._act_end = QLineEdit("18")
        self._act_end.setFixedWidth(30)      
        layout.addWidget(self._act_start)
        layout.addWidget(QLabel("-"))      
        layout.addWidget(self._act_end)                                      

        layout.addSpacing(15)                                                
        layout.addWidget(QLabel("Smooth (hrs):"))                            
        self._smooth = QLineEdit("2")
        self._smooth.setFixedWidth(30)        
        layout.addWidget(self._smooth)                                       
        
        layout.addSpacing(15)                                                
        layout.addWidget(QLabel("Colour Map:"))                              
        self._cmap_combo = QComboBox()                                       
        self._cmap_combo.addItems(["turbo", "viridis", "plasma", "inferno"]) 
        self._cmap_combo.currentTextChanged.connect(self._redraw_visuals)    
        layout.addWidget(self._cmap_combo)                                   
        
        layout.addWidget(QLabel("Min Colour:"))                              
        self._cbar_min = QLineEdit("1")                                      
        self._cbar_min.textChanged.connect(self._redraw_visuals)             
        layout.addWidget(self._cbar_min)                                     
        
        layout.addWidget(QLabel("Max Colour:"))                              
        self._cbar_max = QLineEdit("")                                       
        self._cbar_max.textChanged.connect(self._redraw_visuals)             
        layout.addWidget(self._cbar_max)                                     

        layout.addStretch()                                                  
        self._run_btn = QPushButton("Calculate Ranking")                     
        self._run_btn.clicked.connect(self._start_ranking)                   
        layout.addWidget(self._run_btn)                                      
        
        self._export_btn = QPushButton("Export CSV")                        
        self._export_btn.clicked.connect(self._export_csv)                  
        self._export_btn.setEnabled(False)                                  
        layout.addWidget(self._export_btn)                                  
        
        return box                                                           

    def _build_method_tab(self) -> QWidget:                                  
        w = QWidget()                                                        
        layout = QVBoxLayout(w)
        
        toolbar = QHBoxLayout()
        btn_math = QPushButton("ℹ️ Math")
        btn_math.clicked.connect(lambda: self._show_math_popup("Intensity"))
        btn_export = QPushButton("⤢ Export Plot")
        btn_export.clicked.connect(self._export_meth)
        toolbar.addWidget(btn_math)
        toolbar.addStretch()
        toolbar.addWidget(btn_export)
        layout.addLayout(toolbar)

        self._meth_wrapper = QVBoxLayout()
        self.fig_method = Figure(figsize=(10, 6))                            
        self.canvas_method = FigureCanvasQTAgg(self.fig_method)              
        self._meth_wrapper.addWidget(self.canvas_method)                     
        layout.addLayout(self._meth_wrapper)
        return w                                                             

    def _build_percentile_tab(self) -> QWidget:                              
        w = QWidget()                                                        
        layout = QVBoxLayout(w)
        
        toolbar = QHBoxLayout()
        btn_math = QPushButton("ℹ️ Rank Math")
        btn_math.clicked.connect(lambda: self._show_math_popup("Percentiles"))
        btn_export = QPushButton("⤢ Export Plot")
        btn_export.clicked.connect(self._export_perc)
        toolbar.addWidget(btn_math)
        toolbar.addStretch()
        toolbar.addWidget(btn_export)
        layout.addLayout(toolbar)

        scroll = QScrollArea()                                               
        scroll.setWidgetResizable(True)                                      
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)                      
        
        inner = QWidget()                                                    
        self._perc_layout = QVBoxLayout(inner)                               
        scroll.setWidget(inner)                                              
        layout.addWidget(scroll)                                             
        
        # Define figure early so export button can access it safely
        self.fig_perc = Figure(figsize=(14, 32))
        self.canvas_perc = FigureCanvasQTAgg(self.fig_perc)
        
        return w                                                             

    def _build_gaussian_tab(self) -> QWidget:                                
        w = QWidget()                                                        
        layout = QVBoxLayout(w) 
        
        toolbar = QHBoxLayout()
        btn_math = QPushButton("ℹ️ KDE Math")
        btn_math.clicked.connect(lambda: self._show_math_popup("Gaussian KDE"))
        btn_export = QPushButton("⤢ Export Plot")
        btn_export.clicked.connect(self._export_gaus)
        toolbar.addWidget(btn_math)
        toolbar.addStretch()
        toolbar.addWidget(btn_export)
        layout.addLayout(toolbar)

        self._gaus_wrapper = QVBoxLayout()
        self.fig_gauss = Figure(figsize=(10, 5))                             
        self.canvas_gauss = FigureCanvasQTAgg(self.fig_gauss)                
        self._gaus_wrapper.addWidget(self.canvas_gauss)                      
        layout.addLayout(self._gaus_wrapper)
        return w                                                             

    def _build_temporal_tab(self) -> QWidget:                                
        w = QWidget()                                                        
        layout = QVBoxLayout(w)
        
        toolbar = QHBoxLayout()
        btn_export = QPushButton("⤢ Export Plot")
        btn_export.clicked.connect(self._export_temp)
        toolbar.addStretch()
        toolbar.addWidget(btn_export)
        layout.addLayout(toolbar)
        
        scroll = QScrollArea()                                               
        scroll.setWidgetResizable(True)                                      
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)                      
        
        inner = QWidget()                                                    
        inner_layout = QVBoxLayout(inner)                                    
        
        self._temp_wrapper = QVBoxLayout()
        self.fig_temp = Figure(figsize=(10, 12))                             
        self.canvas_temp = FigureCanvasQTAgg(self.fig_temp)                  
        self.canvas_temp.setMinimumHeight(800)                               
        
        self._temp_wrapper.addWidget(self.canvas_temp)                       
        inner_layout.addLayout(self._temp_wrapper)
        
        scroll.setWidget(inner)                                              
        layout.addWidget(scroll)                                             
        return w

    def load_data(self, data_file):                                          
        self._df = data_file.df.copy()                                       
        self._df.index = pd.to_datetime(self._df.index)                      
        self._diams = np.array(data_file.diameters)                          

    def _start_ranking(self):                                                
        if self._df is None: return                                          
        self._progress_bar.setVisible(True)                                  
        
        kwargs = {                                                           
            'dp_min': float(self._dp_min.text()),                            
            'dp_max': float(self._dp_max.text()),                            
            'bg_start': int(self._bg_start.text()),                          
            'bg_end': int(self._bg_end.text()),                              
            'act_start': int(self._act_start.text()),                        
            'act_end': int(self._act_end.text()),                            
            'smooth': int(self._smooth.text())                               
        }                                                                    
        
        self._worker = NanoRankingWorker(self._df, self._diams, kwargs)      
        self._worker.finished.connect(self._on_worker_done)                  
        self._worker.start()                                                 

    def _on_worker_done(self, result: dict):                                 
        self._progress_bar.setVisible(False)                                 
        self._results = result                                               
        self._export_btn.setEnabled(True)                                   
        self._redraw_visuals()                                               

    def _redraw_visuals(self):                                               
        if self._results is None: return                                     
        self._plot_methodology()                                             
        self._plot_gaussian()                                                
        self._plot_temporal()                                                
        self._plot_percentiles()                                             

    # ── Plotting Logic ─────────────────────────────────────────────────── #

    def _plot_methodology(self):                                             
        self.fig_method.clear()                                              
        res_df = self._results['daily_df']                                   
        n_smooth = self._results['n_smooth']                                 
        params = self._results['params']                                     
        
        ax1 = self.fig_method.add_subplot(211)                               
        ax1.plot(n_smooth.index, n_smooth.values, color=_ACCENT_COLORS[1], lw=0.5) 
        ax1.set_ylabel('N (cm-3)')                 
        ax1.set_yscale('log')                                                
        ax1.set_title(f"Target Particle Range ({params['dp_min']}-{params['dp_max']} nm)") 
        
        ax2 = self.fig_method.add_subplot(212)                               
        if not res_df.empty:                                                 
            example_date = res_df.index[len(res_df)//2]                      
            day_data = n_smooth[n_smooth.index.date == example_date.date()]  
            
            ax2.plot(day_data.index.hour + day_data.index.minute/60,         
                     day_data.values, color=_ACCENT_COLORS[0], lw=2)         
            
            ax2.axvspan(params['bg_start'], params['bg_end'] if params['bg_start'] < params['bg_end'] else 24, color='blue', alpha=0.1, label='Background') 
            if params['bg_start'] > params['bg_end']:
                ax2.axvspan(0, params['bg_end'], color='blue', alpha=0.1)

            ax2.axvspan(params['act_start'], params['act_end'], color='red', alpha=0.1, label='Active') 
            
            ax2.set_ylabel('N (cm-3)')             
            ax2.set_xlabel("Hour of day")                                    
            ax2.set_yscale('log')                                            
            ax2.legend()                                                     
            ax2.set_title(f"Example Day: {example_date.strftime('%Y-%m-%d')}") 

        self.fig_method.tight_layout()                                       
        self.canvas_method.draw()                                            

    def _plot_gaussian(self):                                                
        self.fig_gauss.clear()                                               
        res_df = self._results['daily_df'].copy()                            
        
        delta_n = res_df['delta_N'].clip(lower=1e-4)                         
        log_dn = np.log10(delta_n)                                           
        
        ax = self.fig_gauss.add_subplot(111)                                 
        
        counts, bins = np.histogram(log_dn, bins=40, density=True)           
        ax.stairs(counts, 10**bins, fill=True, color='lightgrey', edgecolor='white') 
        
        kde = stats.gaussian_kde(log_dn)                                     
        x_log = np.linspace(log_dn.min(), log_dn.max(), 200)                 
        y_kde = kde(x_log)                                                   
        
        ax.plot(10**x_log, y_kde, color=_ACCENT_COLORS[1], lw=2, label="KDE Fit") 
        
        peaks, _ = find_peaks(y_kde, prominence=0.05)                        
        for i, p in enumerate(peaks):                                        
            ax.axvline(10**x_log[p], color=_ACCENT_COLORS[3], linestyle='--', lw=1) 
            ax.text(10**x_log[p], y_kde[p] + 0.05, f"Mode {i+1}", color=_ACCENT_COLORS[3], ha='center', va='bottom') 
        
        ax.set_xscale('log')                                                 
        ax.set_xlabel('dN (cm-3)')            
        ax.set_ylabel("Density")                                             
        ax.set_title("Distribution of Intensity (dN) and Detected Modes")        
        ax.legend()                                                          
        
        self.fig_gauss.tight_layout()                                        
        self.canvas_gauss.draw()                                             

    def _plot_temporal(self):                                                
        self.fig_temp.clear()                                                
        res_df = self._results['daily_df'].copy()                            
        if res_df.empty: return                                              
        
        # --- 1. Full Time Series Plot ---
        ax1 = self.fig_temp.add_subplot(311)                                 
        ax1.plot(res_df.index, res_df['delta_N'], marker='.', linestyle='', color='red', alpha=0.3, ms=5) 
        
        roll = res_df['delta_N'].rolling('30D', min_periods=1).median()      
        ax1.plot(roll.index, roll.values, color='blue', lw=2, label="30-Day Median") 
        
        ax1.set_yscale('log')                                                
        ax1.set_ylabel('dN (cm-3)')                  
        ax1.set_title("Full Time Series")                                    
        ax1.legend()                                                         
        
        # Harmonised Quartile configuration
        quartile_bins = [0, 25, 50, 75, 100]                                 
        quartile_labels = ["0-25%", "25-50%", "50-75%", "75-100%"]            
        res_df['quartile'] = pd.cut(res_df['rank'], bins=quartile_bins, labels=quartile_labels) 
        colors_blue = ['#c6dbef', '#6baed6', '#2171b5', '#08306b']

        # --- 2. Continuous Monthly Line Chart (Harmonised) ---
        ax2 = self.fig_temp.add_subplot(312)                                 
        monthly_quart = res_df.groupby([pd.Grouper(freq='ME'), 'quartile'], observed=False).size().unstack(fill_value=0) 
        
        for i, col in enumerate(monthly_quart.columns):                      
            ax2.plot(monthly_quart.index, monthly_quart[col],                
                     marker='o', lw=2, ms=4, color=colors_blue[i], label=col)    
            
        # Top 1% Trace
        top_1pct = res_df[res_df['rank'] >= 95].groupby(pd.Grouper(freq='ME')).size().reindex(monthly_quart.index, fill_value=0)
        ax2.plot(top_1pct.index, top_1pct.values, color='black', linestyle='--', lw=1.5, label='95-100% (Extreme)')
            
        ax2.set_ylabel("Days per Month")                                     
        ax2.set_title("Monthly Frequency by Quartile (Harmonised)")              
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())                
        ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator())) 
        ax2.legend(title="Percentile", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9) 
        
        # --- 3. Seasonal Climatology Bar Chart ---
        ax3 = self.fig_temp.add_subplot(313)                                 
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        seasonal_counts = res_df.groupby([res_df.index.month, 'quartile'], observed=False).size().unstack(fill_value=0) 
        
        bottom = np.zeros(12)                                                
        
        for i, col in enumerate(seasonal_counts.columns):                    
            ax3.bar(month_names, seasonal_counts[col].reindex(range(1,13), fill_value=0), 
                    bottom=bottom, label=col, color=colors_blue[i], edgecolor='white') 
            bottom += seasonal_counts[col].reindex(range(1,13), fill_value=0).values 
            
        ax3.set_ylabel("Total Days")                                         
        ax3.set_title("Seasonal Climatology (Quartiles by Month)")            
        ax3.legend(title="Percentile", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9) 
        
        self.fig_temp.tight_layout()                                         
        self.canvas_temp.draw()
        
    def _plot_percentiles(self):                                             
        while self._perc_layout.count():                                     
            item = self._perc_layout.takeAt(0)                               
            if item.widget(): item.widget().deleteLater()                    

        res_df = self._results['daily_df']                                   
        params = self._results['params']                                     
        df_norm = self._df.index.normalize().date                            
        
        try: v_min = float(self._cbar_min.text())                            
        except ValueError: v_min = 1.0                                       
        try: v_max_user = float(self._cbar_max.text())                       
        except ValueError: v_max_user = None                                 

        log_d = np.log10(self._diams)                                        
        dlogdp = np.mean(np.diff(log_d)) if len(log_d) > 1 else 0.1          
        mask_target = (self._diams >= params['dp_min']) & (self._diams <= params['dp_max']) 
        
        self.fig_perc.clear()
        self.canvas_perc.setMinimumHeight(3500)                              
        
        gs = self.fig_perc.add_gridspec(10, 2, hspace=0.6, wspace=0.4)       
        
        # Compute global v_max
        v_max_list = []
        for i in range(20):                                                  
            p_min, p_max = i * 5, (i + 1) * 5                                
            days_in_bin = res_df[(res_df['rank'] > p_min) & (res_df['rank'] <= p_max)].index.date 
            if len(days_in_bin) == 0: continue
            mask = np.isin(df_norm, days_in_bin)                             
            subset = self._df[mask]                                          
            diurnal_mean = subset.groupby(subset.index.hour).mean()         
            pnsd_vals = np.clip(diurnal_mean.values, 1e-4, None)            
            v_max = v_max_user if v_max_user else max(pnsd_vals.max(), v_min*10)
            v_max_list.append(v_max)
        v_max_global = max(v_max_list) if v_max_list else v_min*10
        
        for i in range(20):                                                  
            row, col = i // 2, i % 2                                         
            ax = self.fig_perc.add_subplot(gs[row, col])                     
            
            p_min, p_max = i * 5, (i + 1) * 5                                
            days_in_bin = res_df[(res_df['rank'] > p_min) & (res_df['rank'] <= p_max)].index.date 
            
            if len(days_in_bin) == 0:                                        
                ax.set_title(f"({p_min}, {p_max}] - No Data")                
                ax.axis('off')                                               
                continue                                                     
                
            mask = np.isin(df_norm, days_in_bin)                             
            subset = self._df[mask]                                          
            
            diurnal_mean = subset.groupby(subset.index.hour).mean()          
            pnsd_vals = np.clip(diurnal_mean.values, 1e-4, None)             
            
            mesh = ax.pcolormesh(diurnal_mean.index, self._diams, pnsd_vals.T, 
                                 cmap=self._cmap_combo.currentText(), 
                                 norm=LogNorm(vmin=v_min, vmax=v_max_global), shading='auto') 
            
            ax.set_yscale('log')                                             
            if col == 0:
                ax.set_ylabel("Dp (nm)")                                         
            else:
                ax.set_ylabel("")
            ax.set_title(f"Percentile ({p_min}, {p_max}%]")                  
            
            # --- Right Hand Axes ---
            if col == 1:
                total_n = diurnal_mean.sum(axis=1) * dlogdp                      
                target_n = diurnal_mean.iloc[:, mask_target].sum(axis=1) * dlogdp 
                fraction = np.where(total_n > 0, target_n / total_n, 0)          
                
                ax_num = ax.twinx()                                              
                ax_num.plot(diurnal_mean.index, total_n, color='red', lw=1.5, alpha=0.8) 
                ax_num.set_ylabel(r"Total N", color='red')                       
                ax_num.set_yscale('log')                                         
                ax_num.tick_params(axis='y', colors='red')                       
                
                ax_frac = ax.twinx()                                             
                ax_frac.spines['right'].set_position(('outward', 45))            
                ax_frac.plot(diurnal_mean.index, fraction, color='green', lw=1.5, alpha=0.8) 
                ax_frac.set_ylabel(r"Fraction", color='green')                   
                ax_frac.set_ylim(0, 1.0)                                         
                ax_frac.tick_params(axis='y', colors='green')                    
                ax_frac.spines['right'].set_color('green')                       

        self.fig_perc.colorbar(mesh, ax=self.fig_perc.axes, orientation='vertical', fraction=0.02, pad=0.08, label="dN/dlogDp (cm-3)") 
        
        self._perc_layout.addWidget(self.canvas_perc)                        
        self._perc_layout.addStretch()
        self.canvas_perc.draw()
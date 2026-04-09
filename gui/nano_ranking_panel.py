from __future__ import annotations

import numpy as np                                                           # Array operations
import pandas as pd                                                          # Data manipulation
import scipy.stats as stats                                                  # For KDE fitting
from scipy.signal import find_peaks                                          # For peak finding
import matplotlib.dates as mdates                                            # Date formatting
from matplotlib.figure import Figure                                         # Matplotlib figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg              # Canvas elements
from matplotlib.colors import LogNorm                                        # Log scaling
from matplotlib import rcParams                                              # Global plot parameters

from PyQt6.QtCore import Qt, QThread, pyqtSignal                             # Core Qt
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,      # UI Layouts
                             QLineEdit, QComboBox, QSplitter, QTabWidget,    # UI Widgets
                             QPushButton, QProgressBar, QGroupBox, QScrollArea)

# ── Matplotlib global style ─────────────────────────────────────────────── #
rcParams['font.family'] = 'serif'                                            
rcParams['font.serif'] = ['Georgia', 'Times New Roman']                      
rcParams['mathtext.fontset'] = 'custom'                                      
rcParams['mathtext.rm'] = 'Georgia'                                          
rcParams['mathtext.it'] = 'Georgia:italic'                                   
rcParams['figure.facecolor'] = '#fff1e5'                                     
rcParams['axes.facecolor'] = '#fff1e5'                                       
rcParams['savefig.facecolor'] = '#fff1e5'                                    

_FT_BG = '#fff1e5'                                                           
_ACCENT_COLORS = ['#0f6e56', '#185fa5', '#854f0b', '#a32d2d', '#533ab7']     

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
        self._workspace.addTab(self._build_temporal_tab(), "④ Temporal")     # New Tab added
        root.addWidget(self._workspace, stretch=1)                           

    def _build_settings_box(self) -> QGroupBox:                              
        box = QGroupBox("Nano Ranking Settings")                             
        layout = QHBoxLayout(box)                                            
        
        layout.addWidget(QLabel("D<sub>p</sub> Range (nm):"))                
        self._dp_min = QLineEdit("2.5"); self._dp_min.setFixedWidth(40)      
        self._dp_max = QLineEdit("5.0"); self._dp_max.setFixedWidth(40)      
        layout.addWidget(self._dp_min); layout.addWidget(QLabel("-"))        
        layout.addWidget(self._dp_max)                                       
        
        layout.addSpacing(15)                                                
        layout.addWidget(QLabel("Background (hrs):"))                        
        self._bg_start = QLineEdit("21"); self._bg_start.setFixedWidth(30)   
        self._bg_end = QLineEdit("6"); self._bg_end.setFixedWidth(30)        
        layout.addWidget(self._bg_start); layout.addWidget(QLabel("-"))      
        layout.addWidget(self._bg_end)                                       

        layout.addSpacing(15)                                                
        layout.addWidget(QLabel("Active (hrs):"))                            
        self._act_start = QLineEdit("6"); self._act_start.setFixedWidth(30)  
        self._act_end = QLineEdit("18"); self._act_end.setFixedWidth(30)     
        layout.addWidget(self._act_start); layout.addWidget(QLabel("-"))     
        layout.addWidget(self._act_end)                                      

        layout.addSpacing(15)                                                
        layout.addWidget(QLabel("Smooth (hrs):"))                            
        self._smooth = QLineEdit("2"); self._smooth.setFixedWidth(30)        
        layout.addWidget(self._smooth)                                       
        
        layout.addSpacing(15)                                                
        layout.addWidget(QLabel("Colour Map:"))                              
        self._cmap_combo = QComboBox()                                       
        self._cmap_combo.addItems(["turbo", "viridis", "plasma", "inferno"]) 
        self._cmap_combo.currentTextChanged.connect(self._redraw_visuals)    # Connect to instant redraw
        layout.addWidget(self._cmap_combo)                                   
        
        layout.addWidget(QLabel("Min Colour:"))                              
        self._cbar_min = QLineEdit("1")                                      
        self._cbar_min.textChanged.connect(self._redraw_visuals)             # Connect to instant redraw
        layout.addWidget(self._cbar_min)                                     
        
        layout.addWidget(QLabel("Max Colour:"))                              
        self._cbar_max = QLineEdit("")                                       
        self._cbar_max.textChanged.connect(self._redraw_visuals)             # Connect to instant redraw
        layout.addWidget(self._cbar_max)                                     

        layout.addStretch()                                                  
        self._run_btn = QPushButton("Calculate Ranking")                     
        self._run_btn.clicked.connect(self._start_ranking)                   
        layout.addWidget(self._run_btn)                                      
        
        return box                                                           

    def _build_method_tab(self) -> QWidget:                                  
        w = QWidget()                                                        
        layout = QVBoxLayout(w)                                              
        self.fig_method = Figure(figsize=(10, 6))                            
        self.canvas_method = FigureCanvasQTAgg(self.fig_method)              
        layout.addWidget(self.canvas_method)                                 
        return w                                                             

    def _build_percentile_tab(self) -> QWidget:                              
        w = QWidget()                                                        
        layout = QVBoxLayout(w)                                              
        scroll = QScrollArea()                                               # Add Scroll Area
        scroll.setWidgetResizable(True)                                      
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)                      
        
        inner = QWidget()                                                    
        self._perc_layout = QVBoxLayout(inner)                               
        scroll.setWidget(inner)                                              
        layout.addWidget(scroll)                                             
        return w                                                             

    def _build_gaussian_tab(self) -> QWidget:                                
        w = QWidget()                                                        
        layout = QVBoxLayout(w)                                              
        self.fig_gauss = Figure(figsize=(10, 5))                             
        self.canvas_gauss = FigureCanvasQTAgg(self.fig_gauss)                
        layout.addWidget(self.canvas_gauss)                                  
        return w                                                             

    def _build_temporal_tab(self) -> QWidget:                                
        w = QWidget()                                                        
        layout = QVBoxLayout(w)                                              
        
        scroll = QScrollArea()                                               # Add scroll area for 3 plots
        scroll.setWidgetResizable(True)                                      
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)                      
        
        inner = QWidget()                                                    
        inner_layout = QVBoxLayout(inner)                                    
        
        self.fig_temp = Figure(figsize=(10, 12))                             # Increased height
        self.canvas_temp = FigureCanvasQTAgg(self.fig_temp)                  
        self.canvas_temp.setMinimumHeight(800)                               # Force minimum height
        
        inner_layout.addWidget(self.canvas_temp)                             
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
        self._redraw_visuals()                                               # Draw all tabs

    def _redraw_visuals(self):                                               # Central redraw function
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
        ax1.set_ylabel(r'$\mathrm{N_{target} \ (cm^{-3})}$')                 
        ax1.set_yscale('log')                                                
        ax1.set_title(f"Target Particle Range ({params['dp_min']}-{params['dp_max']} nm)") 
        
        ax2 = self.fig_method.add_subplot(212)                               
        if not res_df.empty:                                                 
            example_date = res_df.index[len(res_df)//2]                      
            day_data = n_smooth[n_smooth.index.date == example_date.date()]  
            
            ax2.plot(day_data.index.hour + day_data.index.minute/60,         
                     day_data.values, color=_ACCENT_COLORS[0], lw=2)         
            
            ax2.axvspan(params['bg_start'], params['bg_end'] if params['bg_start'] < params['bg_end'] else 24, color='blue', alpha=0.1, label='Background') 
            ax2.axvspan(params['act_start'], params['act_end'], color='red', alpha=0.1, label='Active') 
            
            ax2.set_ylabel(r'$\mathrm{N_{target} \ (cm^{-3})}$')             
            ax2.set_xlabel("Hour of day")                                    
            ax2.set_yscale('log')                                            
            ax2.legend()                                                     
            ax2.set_title(f"Example Day: {example_date.strftime('%Y-%m-%d')}") # Removed Fig reference

        self.fig_method.tight_layout()                                       
        self.canvas_method.draw()                                            

    def _plot_gaussian(self):                                                
        self.fig_gauss.clear()                                               
        res_df = self._results['daily_df'].copy()                            
        
        delta_n = res_df['delta_N'].clip(lower=1e-4)                         # Get valid dN values
        log_dn = np.log10(delta_n)                                           # Log transform for math
        
        ax = self.fig_gauss.add_subplot(111)                                 
        
        # Plot visual histogram using steps mapped to linear X axis
        counts, bins = np.histogram(log_dn, bins=40, density=True)           
        ax.stairs(counts, 10**bins, fill=True, color='lightgrey', edgecolor='white') 
        
        # Fit KDE and find peaks
        kde = stats.gaussian_kde(log_dn)                                     
        x_log = np.linspace(log_dn.min(), log_dn.max(), 200)                 
        y_kde = kde(x_log)                                                   
        
        ax.plot(10**x_log, y_kde, color=_ACCENT_COLORS[1], lw=2, label="KDE Fit") # Plot KDE
        
        peaks, _ = find_peaks(y_kde, prominence=0.05)                        # Find modes
        for i, p in enumerate(peaks):                                        
            ax.axvline(10**x_log[p], color=_ACCENT_COLORS[3], linestyle='--', lw=1) 
            ax.text(10**x_log[p], y_kde[p] + 0.05, f"Mode {i+1}", color=_ACCENT_COLORS[3], ha='center') 
        
        ax.set_xscale('log')                                                 # Log10 transform X axis
        ax.set_xlabel(r'$\mathrm{\Delta N_{2.5-5} \ (cm^{-3})}$')            
        ax.set_ylabel("Density")                                             
        ax.set_title("Distribution of $\Delta N$ and Detected Modes")        # Removed Fig reference
        ax.legend()                                                          
        
        self.fig_gauss.tight_layout()                                        
        self.canvas_gauss.draw()                                             

    def _plot_temporal(self):                                                
        self.fig_temp.clear()                                                
        res_df = self._results['daily_df'].copy()                            
        if res_df.empty: return                                              
        
        import matplotlib.cm as cm                                           
        
        # --- 1. Full Time Series Plot ---
        ax1 = self.fig_temp.add_subplot(311)                                 
        ax1.plot(res_df.index, res_df['delta_N'], marker='.', linestyle='', color='red', alpha=0.3, ms=5) 
        
        roll = res_df['delta_N'].rolling('30D', min_periods=1).median()      
        ax1.plot(roll.index, roll.values, color='blue', lw=2, label="30-Day Median") 
        
        ax1.set_yscale('log')                                                
        ax1.set_ylabel(r'$\mathrm{\Delta N \ (cm^{-3})}$')                   
        ax1.set_title("Full Time Series")                                    
        ax1.legend()                                                         
        
        # --- 2. Continuous Monthly Line Chart (Deciles) ---
        ax2 = self.fig_temp.add_subplot(312)                                 
        
        decile_bins = np.arange(0, 101, 10)                                  
        decile_labels = [f"{i}-{i+10}%" for i in range(0, 100, 10)]          
        res_df['decile'] = pd.cut(res_df['rank'], bins=decile_bins, labels=decile_labels) 
        
        monthly_decile = res_df.groupby([pd.Grouper(freq='MS'), 'decile'], observed=False).size().unstack(fill_value=0) 
        
        try: cmap_paired = cm.get_cmap('Paired')                             
        except AttributeError: 
            import matplotlib
            cmap_paired = matplotlib.colormaps['Paired']                     
            
        colors_paired = [cmap_paired(i) for i in range(10)]                  
        
        for i, col in enumerate(monthly_decile.columns):                     
            ax2.plot(monthly_decile.index, monthly_decile[col],              
                     marker='o', lw=2, ms=4, color=colors_paired[i], label=col)    
            
        ax2.set_ylabel("Days per Month")                                     
        ax2.set_title("Continuous Monthly Frequency by Decile")              
        
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())                
        ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator())) 
        
        ax2.legend(title="Percentile", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9) 
        
        # --- 3. Seasonal Climatology Bar Chart (Jan-Dec) ---
        ax3 = self.fig_temp.add_subplot(313)                                 
        
        quartile_bins = [0, 25, 50, 75, 100]                                 
        quartile_labels = ["0-25%", "25-50%", "50-75%", "75-100%"]           
        res_df['quartile'] = pd.cut(res_df['rank'], bins=quartile_bins, labels=quartile_labels) 
        
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        
        seasonal_counts = res_df.groupby([res_df.index.month, 'quartile'], observed=False).size().unstack(fill_value=0) 
        
        bottom = np.zeros(12)                                                
        colors_blue = ['#c6dbef', '#6baed6', '#2171b5', '#08306b']           
        
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
        while self._perc_layout.count():                                     # Clear old plots
            item = self._perc_layout.takeAt(0)                               
            if item.widget(): item.widget().deleteLater()                    

        res_df = self._results['daily_df']                                   
        params = self._results['params']                                     
        df_norm = self._df.index.normalize().date                            # Normalized dates for matching
        
        try: v_min = float(self._cbar_min.text())                            # Get colour limits
        except ValueError: v_min = 1.0                                       
        try: v_max_user = float(self._cbar_max.text())                       
        except ValueError: v_max_user = None                                 

        log_d = np.log10(self._diams)                                        
        dlogdp = np.mean(np.diff(log_d)) if len(log_d) > 1 else 0.1          
        
        mask_target = (self._diams >= params['dp_min']) & (self._diams <= params['dp_max']) 
        
        # 10x2 Grid for the 20 percentile bins
        fig = Figure(figsize=(14, 32))                                       
        fig.patch.set_facecolor(_FT_BG)                                      
        canvas = FigureCanvasQTAgg(fig)                                      
        canvas.setMinimumHeight(3500)                                        # Force scroll height
        
        gs = fig.add_gridspec(10, 2, hspace=0.6, wspace=0.4)                 # Space for right axes
        
        for i in range(20):                                                  
            row, col = i // 2, i % 2                                         
            ax = fig.add_subplot(gs[row, col])                               
            
            p_min, p_max = i * 5, (i + 1) * 5                                
            days_in_bin = res_df[(res_df['rank'] > p_min) & (res_df['rank'] <= p_max)].index.date 
            
            if len(days_in_bin) == 0:                                        
                ax.set_title(f"({p_min}, {p_max}] - No Data")                
                ax.axis('off')                                               
                continue                                                     
                
            mask = np.isin(df_norm, days_in_bin)                             # Match hourly rows
            subset = self._df[mask]                                          
            
            diurnal_mean = subset.groupby(subset.index.hour).mean()          # Calculate diurnal cycle
            pnsd_vals = np.clip(diurnal_mean.values, 1e-4, None)             
            
            v_max = v_max_user if v_max_user else max(pnsd_vals.max(), v_min*10) 
            mesh = ax.pcolormesh(diurnal_mean.index, self._diams, pnsd_vals.T, 
                                 cmap=self._cmap_combo.currentText(), 
                                 norm=LogNorm(vmin=v_min, vmax=v_max), shading='auto') 
            
            ax.set_yscale('log')                                             
            ax.set_ylabel("Dp (nm)")                                         
            ax.set_title(f"Percentile ({p_min}, {p_max}]")                   
            
            # --- Right Hand Axes (Total N and Fraction) ---
            total_n = diurnal_mean.sum(axis=1) * dlogdp                      # Total concentration
            target_n = diurnal_mean.iloc[:, mask_target].sum(axis=1) * dlogdp # Target range concentration
            fraction = np.where(total_n > 0, target_n / total_n, 0)          # Avoid zero division
            
            ax_num = ax.twinx()                                              # First twin for Total N (Red)
            ax_num.plot(diurnal_mean.index, total_n, color='red', lw=1.5, alpha=0.8) 
            ax_num.set_ylabel(r"Total N", color='red')                       
            ax_num.set_yscale('log')                                         
            ax_num.tick_params(axis='y', colors='red')                       
            
            ax_frac = ax.twinx()                                             # Second twin for Fraction (Green)
            ax_frac.spines['right'].set_position(('outward', 45))            # Offset it so it doesn't overlap
            ax_frac.plot(diurnal_mean.index, fraction, color='green', lw=1.5, alpha=0.8) 
            ax_frac.set_ylabel(r"Fraction", color='green')                   
            ax_frac.set_ylim(0, 1.0)                                         # Fraction is strictly 0 to 1
            ax_frac.tick_params(axis='y', colors='green')                    
            ax_frac.spines['right'].set_color('green')                       

        fig.colorbar(mesh, ax=fig.axes, orientation='vertical', fraction=0.02, pad=0.08, label=r"dN/dlogD$_p$ (cm$^{-3}$)") 
        
        self._perc_layout.addWidget(canvas)                                  
        self._perc_layout.addStretch()
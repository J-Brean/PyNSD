import os                                                                        
import numpy as np                                                               
import pandas as pd                                                              
from scipy import stats, signal                                                  
import matplotlib.dates as mdates                                                
import matplotlib.pyplot as plt                                                  
from matplotlib.figure import Figure                                             
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg                  
from matplotlib.colors import LogNorm                                            
from matplotlib.widgets import RectangleSelector                                 
from matplotlib import rcParams                                                  

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QPushButton, QSlider, QDialog,
                             QMessageBox, QGroupBox, QLineEdit, QFileDialog,
                             QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QInputDialog)

from utils.calculations import (calc_condensation_sink, calc_coagulation_sink,   
                                calc_formation_rate, fit_modes_to_pnsd, calc_growth_rate,
                                calculate_j1_5, calculate_m)                     

try:
    from fastai.vision.all import load_learner                                       
except ImportError:
    load_learner = None                                                              

rcParams['font.family'] = 'serif'                                                
rcParams['font.serif'] = ['Georgia', 'Times New Roman']                          
rcParams['xtick.direction'] = 'out'                                              
rcParams['ytick.direction'] = 'out'                                              
rcParams['figure.facecolor'] = '#fff1e5'                                         
rcParams['axes.facecolor'] = '#fff1e5'                                           
rcParams['savefig.facecolor'] = '#fff1e5'                                        

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
# Coagulation Sink & Lifetime Heatmap Tool
# ─────────────────────────────────────────────────────────────────────────── #
class CoagSWindow(QDialog):                                                      
    def __init__(self, day_df, diams, dlogdp, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Coagulation Matrix Analysis")                       
        self.resize(1200, 650)                                                   
        self.day_df = day_df                                                     
        self.diams = diams                                                       
        self.dlogdp = dlogdp                                                     
        self._build_ui()                                                         

    def _build_ui(self):
        layout = QVBoxLayout(self)                                               
        
        # --- Inline Export Controls ---
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Width (px):"))
        self.val_w = QLineEdit(str(self.width()))
        ctrl_layout.addWidget(self.val_w)
        ctrl_layout.addWidget(QLabel("Height (px):"))
        self.val_h = QLineEdit(str(self.height() - 50))
        ctrl_layout.addWidget(self.val_h)
        self.btn_apply = QPushButton("Apply Size")
        self.btn_apply.clicked.connect(self.apply_size)
        ctrl_layout.addWidget(self.btn_apply)
        self.btn_save = QPushButton("💾 Save Image")
        self.btn_save.clicked.connect(self.save_plot)
        ctrl_layout.addWidget(self.btn_save)
        layout.addLayout(ctrl_layout)
        
        self.fig = Figure(figsize=(12, 6))                                       
        self.canvas = FigureCanvasQTAgg(self.fig)                                
        layout.addWidget(self.canvas)                                            

        ax1, ax2 = self.fig.subplots(1, 2, sharex=True, sharey=True)             
        pnsd = self.day_df.to_numpy()                                            
        
        coags_matrix = calc_coagulation_sink(self.diams, pnsd, self.dlogdp)      

        dates = mdates.date2num(self.day_df.index)                               
        coags_safe = np.clip(coags_matrix, 1e-7, None)                           
        
        lifetime_hours = 1.0 / (coags_safe * 3600.0)

        mesh1 = ax1.pcolormesh(dates, self.diams, coags_safe.T, cmap='inferno', norm=LogNorm(), shading='nearest') 
        ax1.set_yscale('log')                                                    
        ax1.set_ylabel("Diameter (nm)")                                          
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))              
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))             
        self.fig.colorbar(mesh1, ax=ax1, label="CoagS (s⁻¹)")               
        ax1.set_title("Coagulation Sink", fontweight='bold') 

        mesh2 = ax2.pcolormesh(dates, self.diams, lifetime_hours.T, cmap='viridis', norm=LogNorm(), shading='nearest') 
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=4))              
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))             
        self.fig.colorbar(mesh2, ax=ax2, label="Lifetime (Hours)")               
        ax2.set_title("Coagulation Lifetime", fontweight='bold')

        self.fig.suptitle(f"Matrix Analysis - {self.day_df.index.date[0]}", fontsize=14, fontweight='bold')
        self.fig.tight_layout()                                                  
        self.canvas.draw()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.val_w.setText(str(self.canvas.width()))
        self.val_h.setText(str(self.canvas.height()))

    def apply_size(self):
        try:
            w, h = int(self.val_w.text()), int(self.val_h.text())
            self.resize(w, h + 50)
        except ValueError: pass

    def save_plot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Plot", "", "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)")
        if path:
            w_in = self.canvas.width() / self.fig.dpi
            h_in = self.canvas.height() / self.fig.dpi
            self.fig.set_size_inches(w_in, h_in)
            self.fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
            QMessageBox.information(self, "Success", "Plot saved successfully!")


class MLClassifierWorker(QThread):
    progress = pyqtSignal(int, str)                                              
    finished = pyqtSignal(object)                                                
    error = pyqtSignal(str)                                                      

    def __init__(self, df: pd.DataFrame, diams: np.ndarray, model_path: str, threshold: float, file_name: str = "Dataset"):
        super().__init__()
        self.df = df
        self.diams = diams
        self.model_path = model_path
        self.threshold = threshold
        self.file_name = file_name                                               

    def run(self):
        try:
            if load_learner is None: raise ImportError("fastai is not installed.")
                
            learner = load_learner(self.model_path)                                  
            results = []
            
            class_map = {'NPF': 'NPF', 'NO': 'non-NPF', 'BAD': 'bad data'}           
            
            groups = list(self.df.groupby(self.df.index.date))                       
            total_days = len(groups)
            
            debug_dir = os.path.join(os.getcwd(), "ML_contour_plots")
            os.makedirs(debug_dir, exist_ok=True)
            
            for f in os.listdir(debug_dir):                                      
                os.remove(os.path.join(debug_dir, f))                            
            
            for idx, (date, day_df) in enumerate(groups):
                pct = int((idx / total_days) * 100)
                self.progress.emit(pct, f"Analysing {date}...")                  
                
                if len(day_df) < 12:
                    results.append({
                        'date': pd.Timestamp(date), 'raw_class': 'bad data', 
                        'prob': 1.0, 'prob_NPF': 0.0, 'prob_NonNPF': 0.0, 'prob_Bad': 1.0
                    })
                    continue
                
                date_str = pd.Timestamp(date).strftime('%Y-%m-%d')                
                img_name = f"{self.file_name}_{date_str}.png"                    
                img_path = os.path.join(debug_dir, img_name)                     
                
                self._generate_temp_plot(day_df, img_path)                       
                
                pred_class, pred_idx, outputs = learner.predict(img_path)        
                vocab = list(learner.dls.vocab)
                all_probs = {str(k): v.item() for k, v in zip(vocab, outputs)}
                
                mapped_class = class_map.get(pred_class, 'bad data')
                
                results.append({
                    'date': pd.Timestamp(date), 
                    'raw_class': mapped_class, 
                    'prob': outputs[pred_idx].item(),
                    'prob_NPF': all_probs.get('NPF', 0.0),
                    'prob_NonNPF': all_probs.get('NO', 0.0),
                    'prob_Bad': all_probs.get('BAD', 0.0)
                })
                    
            self.progress.emit(100, "Classification complete.")
            res_df = pd.DataFrame(results).set_index('date')
            self.finished.emit(res_df)
            
        except Exception as exc:
            self.error.emit(str(exc))

    def _generate_temp_plot(self, day_df, save_path):
        import matplotlib as mpl                                            
        import copy                                                         
        
        orig_font = mpl.rcParams['font.family']                             
        orig_tick = mpl.rcParams['xtick.direction']                         
        
        mpl.rcParams['font.family'] = 'sans-serif'                          
        mpl.rcParams['font.sans-serif'] = ['Arial']                         
        mpl.rcParams['xtick.direction'] = 'in'                              
        mpl.rcParams['ytick.direction'] = 'in'                              
        
        fig = plt.figure(figsize=(6, 6), dpi=100)                           
        ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])                           
        
        day_df = day_df.groupby(day_df.index.hour).mean()                   
        day_df = day_df.reindex(range(24))                                  
        day_df = day_df.bfill(limit=1).ffill(limit=1).fillna(1e-4)          
        
        pnsd_log = np.log10(np.clip(day_df.values, 1.0, None))              
        log_diams = np.log10(self.diams)                                    
        time_centers = np.arange(0.5, 24.5)                                 
        
        cmap = copy.copy(mpl.colormaps['jet'])                              
        cmap.set_under('black')                                             
        
        mesh = ax.pcolormesh(time_centers, log_diams, pnsd_log.T, 
                             cmap=cmap, vmin=0.0, vmax=5.0, 
                             shading='nearest')                             
        
        ax.set_ylim(log_diams.min(), log_diams.max())                       
        ax.set_xlim(0, 24)                                                  
        
        ax.set_xticks([5, 10, 15, 20])                                      
        ax.tick_params(axis='x', labelsize=14)                              
        ax.set_xlabel("X", fontsize=18)                                     
        
        potential_y_ticks = [1.0, 1.5, 2.0, 2.5, 3.0]                       
        valid_y_ticks = [t for t in potential_y_ticks if log_diams.min() <= t <= log_diams.max()]
        ax.set_yticks(valid_y_ticks)                                        
        ax.tick_params(axis='y', labelsize=14)                              
        ax.set_ylabel("Y", fontsize=18)                                     
        
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02, aspect=15)               
        cbar.set_ticks([0, 1, 2, 3, 4, 5])                                  
        cbar.ax.tick_params(labelsize=14)                                   
        
        fig.patch.set_facecolor('white')                                    
        ax.set_facecolor('white')                                           
        
        fig.savefig(save_path, facecolor='white')                           
        plt.close(fig)                                                      
        
        mpl.rcParams['font.family'] = orig_font                             
        mpl.rcParams['xtick.direction'] = orig_tick                         
        mpl.rcParams['ytick.direction'] = orig_tick                         
        

class MLSummaryWindow(QDialog):
    def __init__(self, raw_df, diams, results_df, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Deep Learning Classification Summary")
        self.resize(1100, 750)
        self.raw_df = raw_df
        self.diams = diams
        self.results = results_df
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        
        # --- Inline Export Controls ---
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Width (px):"))
        self.val_w = QLineEdit(str(self.width()))
        ctrl_layout.addWidget(self.val_w)
        ctrl_layout.addWidget(QLabel("Height (px):"))
        self.val_h = QLineEdit(str(self.height() - 100))
        ctrl_layout.addWidget(self.val_h)
        self.btn_apply = QPushButton("Apply Size")
        self.btn_apply.clicked.connect(self.apply_size)
        ctrl_layout.addWidget(self.btn_apply)
        self.btn_save = QPushButton("💾 Save Image")
        self.btn_save.clicked.connect(self.save_plot)
        ctrl_layout.addWidget(self.btn_save)
        layout.addLayout(ctrl_layout)

        total = len(self.results)
        counts = self.results['class'].value_counts()
        pct_npf = (counts.get('NPF', 0) / total) * 100
        pct_non = (counts.get('non-NPF', 0) / total) * 100
        pct_bad = (counts.get('bad data', 0) / total) * 100
        
        header = QLabel(f"<b>Total Evaluated Days:</b> {total} | <b>NPF:</b> {pct_npf:.1f}% | <b>Non-NPF:</b> {pct_non:.1f}% | <b>Bad Data:</b> {pct_bad:.1f}%")
        header.setStyleSheet("font-size: 16px; padding: 10px; background-color: #fff; border: 1px solid #ccc;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        self.fig = Figure(figsize=(12, 8))
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas)
        self._plot_summary()

    def _plot_summary(self):
        self.fig.clear()
        gs = self.fig.add_gridspec(2, 3, height_ratios=[1, 1.5], hspace=0.3, wspace=0.3)
        
        ax_freq = self.fig.add_subplot(gs[0, :])
        monthly = self.results.groupby([pd.Grouper(freq='MS'), 'class']).size().unstack(fill_value=0)
        colors = {'NPF': '#d62728', 'non-NPF': '#1f77b4', 'bad data': '#7f7f7f'}
        bottom = np.zeros(len(monthly))
        
        for cls in ['NPF', 'non-NPF', 'bad data']:
            if cls in monthly:
                ax_freq.bar(monthly.index, monthly[cls], width=20, bottom=bottom, label=cls, color=colors[cls])
                bottom += monthly[cls].values
                
        ax_freq.set_ylabel("Days per Month")
        ax_freq.legend(loc='upper right')
        ax_freq.set_title("Classification Frequency Over Time", fontweight='bold')
        
        log_d = np.log10(self.diams)
        dlogdp = np.mean(np.diff(log_d)) if len(log_d) > 1 else 0.1
        
        classes = ['NPF', 'non-NPF', 'bad data']
        for i, cls in enumerate(classes):
            ax_cont = self.fig.add_subplot(gs[1, i])
            days = self.results[self.results['class'] == cls].index.date
            if len(days) == 0:
                ax_cont.set_title(f"{cls}\n(0 days)", fontweight='bold'); ax_cont.axis('off')
                continue
                
            mask = np.isin(self.raw_df.index.normalize().date, days)
            subset = self.raw_df[mask]
            diurnal = subset.groupby(subset.index.hour).mean()
            
            pnsd = np.clip(diurnal.values, 1e-4, None)
            ax_cont.pcolormesh(diurnal.index, self.diams, pnsd.T, cmap='turbo', norm=LogNorm(vmin=10, vmax=5e4))
            ax_cont.set_yscale('log'); ax_cont.set_xlabel("Hour of Day")
            if i == 0: ax_cont.set_ylabel("Dp (nm)")
            
            ax_dn = ax_cont.twinx()
            total_n = diurnal.sum(axis=1) * dlogdp
            ax_dn.plot(diurnal.index, total_n, color='white', lw=2.5, alpha=0.9, label='Total N')
            if i == 2: ax_dn.set_ylabel(r"Total N $(cm^{-3})$", color='black')
            else: ax_dn.set_yticklabels([])
            
            ax_cont.set_title(f"Average {cls} Diurnal", fontweight='bold')

        self.fig.tight_layout()
        self.canvas.draw()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.val_w.setText(str(self.canvas.width()))
        self.val_h.setText(str(self.canvas.height()))

    def apply_size(self):
        try:
            w, h = int(self.val_w.text()), int(self.val_h.text())
            self.resize(w, h + 100)
        except ValueError: pass

    def save_plot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Plot", "", "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)")
        if path:
            w_in = self.canvas.width() / self.fig.dpi
            h_in = self.canvas.height() / self.fig.dpi
            self.fig.set_size_inches(w_in, h_in)
            self.fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
            QMessageBox.information(self, "Success", "Plot saved successfully!")


class DiurnalSummaryWindow(QDialog):
    def __init__(self, raw_df, diams, ml_results, j_min, j_max, dlogdp, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Average Diurnals (NPF vs Non-NPF)")             
        self.resize(1000, 950)                                               
        self.raw_df = raw_df
        self.diams = diams
        self.ml_results = ml_results
        self.j_min = j_min
        self.j_max = j_max
        self.dlogdp = dlogdp
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        
        # --- Inline Export Controls ---
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Width (px):"))
        self.val_w = QLineEdit(str(self.width()))
        ctrl_layout.addWidget(self.val_w)
        ctrl_layout.addWidget(QLabel("Height (px):"))
        self.val_h = QLineEdit(str(self.height() - 50))
        ctrl_layout.addWidget(self.val_h)
        self.btn_apply = QPushButton("Apply Size")
        self.btn_apply.clicked.connect(self.apply_size)
        ctrl_layout.addWidget(self.btn_apply)
        self.btn_save = QPushButton("💾 Save Image")
        self.btn_save.clicked.connect(self.save_plot)
        ctrl_layout.addWidget(self.btn_save)
        layout.addLayout(ctrl_layout)

        self.fig = Figure(figsize=(10, 12))                                  
        self.canvas = FigureCanvasQTAgg(self.fig)                            
        layout.addWidget(self.canvas)                                        
        self._plot_diurnals()

    def _calc_mass(self, pnsd_dndlogdp):
        N_cm3 = pnsd_dndlogdp * self.dlogdp                                  
        d_m = self.diams * 1e-9                                              
        vol_m3 = (np.pi / 6.0) * (d_m ** 3)                                  
        mass_kg_m3 = np.sum(N_cm3 * 1e6 * vol_m3 * 1.5e3, axis=1)            
        return np.sum(N_cm3, axis=1), mass_kg_m3 * 1e9                        

    def _plot_diurnals(self):
        gs = self.fig.add_gridspec(4, 2, hspace=0.4, wspace=0.15)
        classes = [('NPF', 1.5), ('non-NPF', 0.0)]                           
        shared_axes = {}  
        
        for col, (cls, assumed_gr) in enumerate(classes):
            days = self.ml_results[self.ml_results['class'] == cls].index.date 
            if len(days) == 0: continue
            
            mask = np.isin(self.raw_df.index.normalize().date, days)         
            diurnal_df = self.raw_df[mask].groupby(self.raw_df[mask].index.hour).mean()
            diurnal_df = diurnal_df.reindex(np.arange(24), fill_value=1e-4)  
            diurnal_pnsd = diurnal_df.to_numpy()
            
            cs_series = calc_condensation_sink(self.diams, diurnal_pnsd, self.dlogdp)
            coags_matrix = calc_coagulation_sink(self.diams, diurnal_pnsd, self.dlogdp)
            j_tot, _, _, _ = calc_formation_rate(self.diams, diurnal_pnsd, self.dlogdp, assumed_gr, self.j_min, self.j_max, coags_matrix)
            n_tot, mass_tot = self._calc_mass(diurnal_pnsd)
            
            time_axis = np.arange(1, 24)
            plot_pnsd = diurnal_pnsd[1:]
            cs_series = cs_series[1:]
            j_tot = j_tot[1:]
            n_tot = n_tot[1:]
            mass_tot = mass_tot[1:]
            
            ax_cont = self.fig.add_subplot(gs[0, col], sharey=shared_axes.get('cont'))
            ax_cont.pcolormesh(time_axis, self.diams, np.clip(plot_pnsd, 1e-4, None).T, cmap='turbo', norm=LogNorm(vmin=10, vmax=5e4), shading='nearest')
            ax_cont.set_yscale('log')
            ax_cont.set_title(f"{cls} Average ({len(days)} days)")
            
            if col == 0: 
                ax_cont.set_ylabel("Dp (nm)")
                shared_axes['cont'] = ax_cont                                
            else:
                ax_cont.tick_params(labelleft=False)                         
            
            ax_j = self.fig.add_subplot(gs[1, col], sharex=ax_cont, sharey=shared_axes.get('j'))
            ax_j.plot(time_axis, j_tot, color='blue', lw=2)
            if col == 0: 
                ax_j.set_ylabel(f"J (cm-3 s-1)\nAssumed GR={assumed_gr}")
                shared_axes['j'] = ax_j
            else:
                ax_j.tick_params(labelleft=False)
            
            ax_cs = self.fig.add_subplot(gs[2, col], sharex=ax_cont, sharey=shared_axes.get('cs'))
            ax_cs.plot(time_axis, cs_series, color='orange', lw=2)
            ax_cs.set_yscale('log')                                          
            if col == 0: 
                ax_cs.set_ylabel("CS (s-1)")
                shared_axes['cs'] = ax_cs
            else:
                ax_cs.tick_params(labelleft=False)
            
            ax_n = self.fig.add_subplot(gs[3, col], sharex=ax_cont, sharey=shared_axes.get('n'))
            ax_n.plot(time_axis, n_tot, color='red', label='Total N')
            
            ax_mass = ax_n.twinx()
            ax_mass.plot(time_axis, mass_tot, color='green', linestyle='--', label='Mass')
            
            if col == 0: 
                ax_n.set_ylabel("Total N (cm-3)", color='red')
                shared_axes['n'] = ax_n
                shared_axes['mass'] = ax_mass
                ax_mass.tick_params(labelright=False)                        
            
            if col == 1: 
                ax_mass.sharey(shared_axes['mass'])                          
                ax_mass.set_ylabel("Mass (ug/m3)", color='green')
                ax_n.tick_params(labelleft=False)                            
            
            ax_n.set_xlabel("Hour of Day")
            ax_cont.set_xlim(1, 23)

        self.fig.tight_layout()
        self.canvas.draw()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.val_w.setText(str(self.canvas.width()))
        self.val_h.setText(str(self.canvas.height()))

    def apply_size(self):
        try:
            w, h = int(self.val_w.text()), int(self.val_h.text())
            self.resize(w, h + 50)
        except ValueError: pass

    def save_plot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Plot", "", "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)")
        if path:
            w_in = self.canvas.width() / self.fig.dpi
            h_in = self.canvas.height() / self.fig.dpi
            self.fig.set_size_inches(w_in, h_in)
            self.fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
            QMessageBox.information(self, "Success", "Plot saved successfully!")


# ─────────────────────────────────────────────────────────────────────────── #
# Main Manual Panel
# ─────────────────────────────────────────────────────────────────────────── #
class NPFPanel(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.df = None
        self.diams = None
        
        self.daily_groups = []                                                       
        self.current_day_idx = 0                                                     
        self.classifications = {}                                                    
        self.gr_result = None                                                        
        self.j_cs_data = None                                                        
        self.scatter_overlay = None                                                  
        self.fit_snapshots = []                                                      
        self.selector = None                                                         
        
        self.last_box_bounds = None                                                  
        self.mode_overrides = {}                                                     
        self.refining_modes = False                                                  
        self.picking_points = False                                                  
        self.points = []                                                             
        self.last_csv_path = None                                                    
        
        self._build_ui()                                                             

    def _show_info_dialog(self, title, text):
        info = QMessageBox(self)
        info.setWindowTitle(title)
        info.setTextFormat(Qt.TextFormat.RichText)
        info.setText(text)
        info.exec()

    def export_plots(self):
        # 1. Heatmap
        def restore_hm():
            self.hm_wrapper.addWidget(self.canvas_hm)
            self.fig_hm.set_size_inches(self.orig_hm_size)
            self.canvas_hm.draw()
            
        self.orig_hm_size = self.fig_hm.get_size_inches()
        dlg_hm = ExportDialog("Daily Heatmap", self.canvas_hm, self.fig_hm, restore_hm, self)
        dlg_hm.exec()

        # 2. Browser
        def restore_br():
            self.browse_wrapper.addWidget(self.canvas_browse)
            self.fig_browse.set_size_inches(self.orig_br_size)
            self.canvas_browse.draw()
            
        self.orig_br_size = self.fig_browse.get_size_inches()
        dlg_br = ExportDialog("PNSD Browser", self.canvas_browse, self.fig_browse, restore_br, self)
        dlg_br.exec()

        # 3. Regression
        def restore_reg():
            self.reg_wrapper.addWidget(self.canvas_reg)
            self.fig_reg.set_size_inches(self.orig_reg_size)
            self.canvas_reg.draw()
            
        self.orig_reg_size = self.fig_reg.get_size_inches()
        dlg_reg = ExportDialog("Growth Rate Regression", self.canvas_reg, self.fig_reg, restore_reg, self)
        dlg_reg.exec()

        # 4. Diurnals
        def restore_diur():
            self.diur_wrapper.addWidget(self.canvas_diur)
            self.fig_diur.set_size_inches(self.orig_diur_size)
            self.canvas_diur.draw()
            
        self.orig_diur_size = self.fig_diur.get_size_inches()
        dlg_diur = ExportDialog("J & CS Kinetics", self.canvas_diur, self.fig_diur, restore_diur, self)
        dlg_diur.exec()

    def _build_ui(self):
        layout = QVBoxLayout(self)                                                   
        
        # --- Header ---
        intro_layout = QHBoxLayout()
        intro_lbl = QLabel("NPF Identifier: Classify events and calculate Formation & Growth Rates.")
        intro_lbl.setStyleSheet("font-size: 14px; font-weight: bold;")
        
        workflow_text = (
            "<b>Manual Analysis Workflow:</b><br><ol>"
            "<li><b>Find NPF Events:</b> Use the 'Previous' / 'Next' buttons to step through your dataset day by day.</li>"
            "<li><b>Classify Events:</b> Click an event type (NPF, Non-NPF, Burst, etc.).<br>"
            "<i>Note: If you click 'Non-NPF', the tool will automatically calculate the Condensation Sink (CS) and clear any Growth Rate fits.</i></li>"
            "<li><b>Fit (If NPF):</b> Draw a box around the growing particle mode on the heatmap. The algorithm will automatically trace the peak concentration.</li>"
            "<i>Note: See mode fitting tools for specifics.</i>"
            "<li><b>Calculate J & CS:</b> Verify your upper/lower diameter boundaries, then click 'Calc J & CS'.</li>"
            "<li><b>Export:</b> Click 'Send to CSV' to append the day's results to your master file.</li></ol>"
        )
        self.btn_intro_info = QPushButton("ℹ️ Workflow Guide")
        self.btn_intro_info.setStyleSheet("background-color: #e0f7fa; border-radius: 4px; padding: 4px;")
        self.btn_intro_info.clicked.connect(lambda: self._show_info_dialog("How to Use This Tool", workflow_text))
        
        intro_layout.addWidget(intro_lbl)
        intro_layout.addStretch()
        intro_layout.addWidget(self.btn_intro_info)
        layout.addLayout(intro_layout)
        
        # --- Top Navigation & Controls ---
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("◄ Previous")
        self.btn_prev.clicked.connect(self.prev_day)
        nav_layout.addWidget(self.btn_prev)
        
        self.date_dropdown = QComboBox()
        self.date_dropdown.setStyleSheet("font-size: 14px; font-weight: bold; padding: 2px 10px;")
        self.date_dropdown.currentIndexChanged.connect(self.jump_to_date)
        self.date_dropdown.setMinimumWidth(150)
        nav_layout.addWidget(self.date_dropdown)
        
        self.btn_next = QPushButton("Next ►")
        self.btn_next.clicked.connect(self.next_day)
        nav_layout.addWidget(self.btn_next)
        nav_layout.addStretch()
        
        nav_layout.addWidget(QLabel("Colour Map:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["turbo", "viridis", "plasma", "inferno"])
        self.cmap_combo.currentTextChanged.connect(self.update_heatmap)
        nav_layout.addWidget(self.cmap_combo)
        
        nav_layout.addWidget(QLabel("Min:")); self.cbar_min = QLineEdit("1"); self.cbar_min.setFixedWidth(40)
        nav_layout.addWidget(self.cbar_min)
        nav_layout.addWidget(QLabel("Max:")); self.cbar_max = QLineEdit(""); self.cbar_max.setFixedWidth(40)
        nav_layout.addWidget(self.cbar_max)
        
        nav_layout.addWidget(QLabel("dlogDp:"))                                      
        self.val_dlogdp = QLineEdit(""); self.val_dlogdp.setFixedWidth(40)           
        nav_layout.addWidget(self.val_dlogdp)
        
        self.btn_redraw = QPushButton("Redraw")
        self.btn_redraw.clicked.connect(self.update_heatmap)
        nav_layout.addWidget(self.btn_redraw)
        
        self.btn_export_plots = QPushButton("⤢ Export Plots")
        self.btn_export_plots.clicked.connect(self.export_plots)
        nav_layout.addWidget(self.btn_export_plots)
        
        layout.addLayout(nav_layout)
        
        main_splitter = QSplitter(Qt.Orientation.Vertical)                           
        
        top_widget = QWidget(); top_layout = QHBoxLayout(top_widget)
        
        # 1. HEATMAP CANVAS WRAPPER
        self.hm_wrapper = QVBoxLayout()
        self.hm_wrapper.setContentsMargins(0, 0, 0, 0)
        self.fig_hm = Figure(); self.canvas_hm = FigureCanvasQTAgg(self.fig_hm)
        self.ax_hm = self.fig_hm.add_subplot(111); self.ax_hm_line = self.ax_hm.twinx()
        self.hm_wrapper.addWidget(self.canvas_hm)                                    
        top_layout.addLayout(self.hm_wrapper, stretch=2)                             

        self.canvas_hm.mpl_connect('button_press_event', self._on_click)             
        
        browse_widget = QWidget(); browse_layout = QVBoxLayout(browse_widget); browse_layout.setContentsMargins(0,0,0,0)
        
        # 2. BROWSER CANVAS WRAPPER
        self.browse_wrapper = QVBoxLayout()
        self.browse_wrapper.setContentsMargins(0, 0, 0, 0)
        self.fig_browse = Figure(); self.canvas_browse = FigureCanvasQTAgg(self.fig_browse)
        self.ax_browse = self.fig_browse.add_subplot(111)
        self.browse_wrapper.addWidget(self.canvas_browse)                            
        browse_layout.addLayout(self.browse_wrapper)                                 
        
        self.slider_browse = QSlider(Qt.Orientation.Horizontal); self.slider_browse.setEnabled(False)
        self.slider_browse.valueChanged.connect(self.update_browser); browse_layout.addWidget(self.slider_browse)
        top_layout.addWidget(browse_widget, stretch=1)
        
        # 3. REGRESSION CANVAS WRAPPER
        self.reg_wrapper = QVBoxLayout()
        self.reg_wrapper.setContentsMargins(0, 0, 0, 0)
        self.fig_reg = Figure(); self.canvas_reg = FigureCanvasQTAgg(self.fig_reg)
        self.ax_reg = self.fig_reg.add_subplot(111)
        self.reg_wrapper.addWidget(self.canvas_reg)                                  
        top_layout.addLayout(self.reg_wrapper, stretch=1)                            
        
        main_splitter.addWidget(top_widget)
        
        bot_widget = QWidget(); bot_layout = QHBoxLayout(bot_widget)
        ctrl_panel = QWidget(); ctrl_layout = QVBoxLayout(ctrl_panel)
        
        # --- 1. Fit Controls (MOVED TO TOP) ---
        fit_layout = QHBoxLayout()
        self.chk_limit_jump = QCheckBox("Max jump (nm):"); self.chk_limit_jump.setChecked(True)
        self.val_max_jump = QLineEdit("15.0"); self.val_max_jump.setFixedWidth(40)
        fit_layout.addWidget(self.chk_limit_jump); fit_layout.addWidget(self.val_max_jump)
        
        self.btn_clear = QPushButton("Clear Fit"); self.btn_clear.clicked.connect(self._clear_points) 
        fit_layout.addWidget(self.btn_clear)
        self.btn_pick = QPushButton("Select Points"); self.btn_pick.setCheckable(True) 
        self.btn_pick.clicked.connect(self._toggle_picking)                  
        fit_layout.addWidget(self.btn_pick)
        self.btn_refine = QPushButton("Refine Fit"); self.btn_refine.setCheckable(True) 
        self.btn_refine.clicked.connect(self._toggle_refine)                 
        fit_layout.addWidget(self.btn_refine)
        
        fit_info = (                                                                 
            "<b>Mode Fitting Tools</b><br><br>"
            "To trace a particle growth event, simply click and drag a box around the 'banana' shape on the heatmap.<br><br>"
            "<b>🧹 Clear Fit:</b> Wipes all manual overrides, picked points, and the calculated GR.<br><br>"
            "<b>📍 Select Points:</b> Fully manual mode. Disables automation. Click exactly <b>two points</b> on the heatmap to force a linear GR between them. Note: This is probably a bad idea :)<br><br>"
            "<b>🎯 Refine Fit:</b> Hybrid mode. Keeps the automated fit, but allows you to click the heatmap to manually force the algorithm to snap to a specific diameter for that hour. Great for bypassing background noise.<br><br>"
            "<b>Max Jump:</b> Prevents the automatic tracker from jumping up or down by more than this amount (in nm) between consecutive hours."
        )
        self.btn_fit_info = QPushButton("ℹ️"); self.btn_fit_info.setFixedSize(24, 24)
        self.btn_fit_info.clicked.connect(lambda: self._show_info_dialog("Fit Tools Info", fit_info))
        fit_layout.addWidget(self.btn_fit_info)

        fit_layout.addStretch()
        fit_layout.addWidget(QLabel("GR (nm/hr):"))
        self.val_gr = QLineEdit(""); self.val_gr.setReadOnly(True); self.val_gr.setFixedWidth(60)
        fit_layout.addWidget(self.val_gr)
        ctrl_layout.addLayout(fit_layout)
        
        # --- 2. Classify Day (MOVED TO MIDDLE) ---
        class_layout = QHBoxLayout()
        class_layout.addWidget(QLabel("Classify Day:"))
        self.btn_npf = QPushButton("NPF"); self.btn_non_npf = QPushButton("Non-NPF")
        self.btn_undef = QPushButton("Undefined"); self.btn_burst = QPushButton("Burst"); self.btn_custom = QPushButton("Custom")
        self.class_buttons = [self.btn_npf, self.btn_non_npf, self.btn_undef, self.btn_burst, self.btn_custom]
        for btn in self.class_buttons:
            class_layout.addWidget(btn)
            btn.clicked.connect(lambda checked, b=btn: self.on_class_button_clicked(b.text()))
            
        class_info = (
            "<b>Classification Guide</b><br><br>"
            "<b>NPF:</b> A clear, sustained regional New Particle Formation event (the classic 'banana'). Requires a Growth Rate fit.<br><br>"
            "<b>Non-NPF:</b> No formation observed. Clicking this will <i>automatically</i> run the Condensation Sink (CS) physics and skip the J/GR steps.<br><br>"
            "<b>Undefined:</b> Messy or broken data where an event might be happening but cannot be reliably traced.<br><br>"
            "<b>Burst:</b> Particles appear in the smallest bins but fail to grow (no 'banana' tail). You can calculate a JR for this day (it will presume your GR is 1 nm/h)<br><br>"
            "<b>Custom:</b> Enter your own string."
        )
        self.btn_class_info = QPushButton("ℹ️"); self.btn_class_info.setFixedSize(24, 24)
        self.btn_class_info.clicked.connect(lambda: self._show_info_dialog("Classification Info", class_info))
        class_layout.addWidget(self.btn_class_info)
        
        ctrl_layout.addLayout(class_layout)

        # --- 3. J & CS Calculation (BOTTOM) ---
        j_layout = QHBoxLayout()
        j_layout.addWidget(QLabel("J Bounds: Min")); self.j_min_dp = QLineEdit("3"); self.j_min_dp.setFixedWidth(30); j_layout.addWidget(self.j_min_dp)
        j_layout.addWidget(QLabel("Max")); self.j_max_dp = QLineEdit("25"); self.j_max_dp.setFixedWidth(30); j_layout.addWidget(self.j_max_dp)
        
        self.chk_j15 = QCheckBox("Calc J1.5")
        j_layout.addWidget(self.chk_j15)

        self.btn_calc_j = QPushButton("✨Calc J & CS"); self.btn_calc_j.clicked.connect(self.calculate_j_and_cs); j_layout.addWidget(self.btn_calc_j)
        self.btn_set_csv = QPushButton("📁Choose CSV"); self.btn_set_csv.clicked.connect(self._choose_new_csv); j_layout.addWidget(self.btn_set_csv)
        self.btn_export = QPushButton("💾Send to CSV"); self.btn_export.clicked.connect(self.export_to_csv); j_layout.addWidget(self.btn_export)
        
        self.btn_coags_map = QPushButton("🔥CoagS Map")
        self.btn_coags_map.clicked.connect(self._show_coags_map)
        j_layout.addWidget(self.btn_coags_map)

        self.btn_diurnals = QPushButton("☀️🌑Show Diurnals"); self.btn_diurnals.clicked.connect(self._show_diurnals); j_layout.addWidget(self.btn_diurnals)
        
        calc_info = (
            "<b>Calculation & Export</b><br><br>"
            "<b>J Bounds (Min/Max):</b> The diameter range used to calculate the Formation Rate (J). Usually 3nm to 25nm, depending on your instrument. The final CSV will record this 'J_Window'.<br><br>"
            "<b>Calc J1.5:</b> If checked, back-calculates the theoretical Formation Rate at 1.5 nm using the Kerminen-Kulmala survival equation. This requires a valid GR.<br><br>"
            "<b>✨ Calc J & CS:</b> Runs the physics equations for Coagulation Sink, Condensation Sink, and Formation Rate based on your fitted Growth Rate.<br><br>"
            "<b>Burst Events:</b> If you classify a day as 'Burst' and do not calculate a GR, the algorithm will automatically assume a GR of 1.0 nm/h to estimate J.<br><br>"
            "<b>🔥 CoagS Map:</b> Pops up a full 24-hour heatmap showing the calculated Coagulation Sink matrix for the active day.<br><br>"
            "<b>💾 Send to CSV:</b> Writes the current day's calculated row into the CSV file and the table below. (Missing columns are handled safely via pandas concat)."
        )
        self.btn_calc_info = QPushButton("ℹ️"); self.btn_calc_info.setFixedSize(24, 24)
        self.btn_calc_info.clicked.connect(lambda: self._show_info_dialog("Calculation Info", calc_info))
        j_layout.addWidget(self.btn_calc_info)
        
        ctrl_layout.addLayout(j_layout)

        self.csv_table = QTableWidget(0, 13)
        self.csv_table.setHorizontalHeaderLabels(["Date", "Class", "In Window", "J_Window", "Mode Dp", "GR", "J", "J[dNdt]", "J[GR]", "J[coag]", "CS", "m", "J1.5"])
        self.csv_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        ctrl_layout.addWidget(self.csv_table)
        bot_layout.addWidget(ctrl_panel, stretch=1)

        # 4. DIURNALS CANVAS WRAPPER
        self.diur_wrapper = QVBoxLayout()
        self.diur_wrapper.setContentsMargins(0, 0, 0, 0)
        self.fig_diur = Figure(); self.canvas_diur = FigureCanvasQTAgg(self.fig_diur)
        self.ax_j = self.fig_diur.add_subplot(111); self.ax_cs = self.ax_j.twinx()
        self.diur_wrapper.addWidget(self.canvas_diur)                                
        bot_layout.addLayout(self.diur_wrapper, stretch=1)                           
        
        main_splitter.addWidget(bot_widget)
        main_splitter.setStretchFactor(0, 3)                                         
        main_splitter.setStretchFactor(1, 2)                                         
        layout.addWidget(main_splitter, stretch=1)                                   

    def load_data(self, data_file):
        self.df = data_file.df
        self.diams = np.array(data_file.diameters)
        log_diams = np.log10(self.diams)
        default_dlogdp = np.mean(np.diff(log_diams)) if len(log_diams) > 1 else 1.0  
        self.val_dlogdp.setText(f"{default_dlogdp:.3f}") 
        
        self.daily_groups = list(self.df.groupby(self.df.index.date))                
        
        self.classifications = {str(date_obj): "Non-NPF" for date_obj, _ in self.daily_groups}
        
        self.date_dropdown.blockSignals(True)
        self.date_dropdown.clear()
        for date_obj, _ in self.daily_groups: self.date_dropdown.addItem(str(date_obj))
        self.date_dropdown.blockSignals(False)
        
        self.current_day_idx = 0
        self.update_day()

    def update_day(self):
        if not self.daily_groups: return
        date_obj, self.day_df = self.daily_groups[self.current_day_idx]
        self.date_dropdown.blockSignals(True)
        self.date_dropdown.setCurrentIndex(self.current_day_idx)
        self.date_dropdown.blockSignals(False)
        
        saved_class = self.classifications.get(str(date_obj), "Non-NPF")
        self.update_class_button_styles(saved_class)
        
        self.gr_result = None; self.j_cs_data = None; self.fit_snapshots = []
        self.slider_browse.setEnabled(False)
        
        if getattr(self, 'scatter_overlay', None): self.scatter_overlay.remove(); self.scatter_overlay = None
        self.val_gr.setText("")
        self.ax_reg.clear(); self.ax_browse.clear(); self.ax_j.clear(); self.ax_cs.clear()
        self.canvas_reg.draw(); self.canvas_browse.draw(); self.canvas_diur.draw()
        self.update_heatmap()

        if saved_class == "Non-NPF":
            self.auto_calculate_non_npf()

    def update_heatmap(self):
        if getattr(self, 'selector', None):                                          
            self.selector.set_active(False)
            self.selector = None

        self.ax_hm.clear(); self.ax_hm_line.clear(); self.ax_hm_line.patch.set_visible(False)
        self.ax_hm.set_zorder(1); self.ax_hm_line.set_zorder(2)
        
        dates = mdates.date2num(self.day_df.index)
        pnsd_safe = np.clip(self.day_df.to_numpy(), 1e-4, None)
        
        try: v_min = float(self.cbar_min.text())
        except ValueError: v_min = 1.0
        try: v_max = float(self.cbar_max.text())
        except ValueError: v_max = pnsd_safe.max()
        if v_max <= v_min: v_max = v_min * 10
        
        self.ax_hm.pcolormesh(dates, self.diams, pnsd_safe.T, cmap=self.cmap_combo.currentText(), shading='nearest', norm=LogNorm(vmin=v_min, vmax=v_max))
        self.ax_hm.set_yscale('log'); self.ax_hm.set_ylabel("Diameter (nm)")
        if len(dates) > 1: self.ax_hm.set_xlim([dates[0], dates[-1]])
        self.ax_hm.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        self.ax_hm.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.ax_hm.set_ylim([self.diams.min(), self.diams.max()])
        
        log_d = np.log10(self.diams); dlogdp = np.mean(np.diff(log_d)) if len(log_d) > 1 else 1.0
        tot_n = np.sum(pnsd_safe, axis=1) * dlogdp
        self.ax_hm_line.plot(dates, tot_n, color='red', alpha=0.6)
        self.ax_hm_line.set_ylabel("Total N", color='red')
        self.ax_hm_line.yaxis.set_label_position("right")                            
        self.ax_hm_line.yaxis.tick_right()
        self.ax_hm_line.tick_params(axis='y', colors='red')
        
        self.selector = RectangleSelector(self.ax_hm, self.on_box_select, useblit=False, button=[1], minspanx=0.01, minspany=1.0, spancoords='data', interactive=True)
        self.fig_hm.tight_layout(pad=1.5); self.canvas_hm.draw()

    def _toggle_picking(self):
        self.picking_points = self.btn_pick.isChecked()
        if self.picking_points:
            self.btn_refine.setChecked(False); self.refining_modes = False; self.btn_refine.setText("Refine Fit")
            if not hasattr(self, 'points'): self.points = []
            self.btn_pick.setText(f"Click plot ({len(self.points)}/2)")
            if getattr(self, 'selector', None): self.selector.set_active(False)      
        else:
            self.btn_pick.setText("Select Points")
            if getattr(self, 'selector', None): self.selector.set_active(True)       

    def _toggle_refine(self):
        self.refining_modes = self.btn_refine.isChecked()
        if self.refining_modes:
            self.btn_pick.setChecked(False); self.picking_points = False; self.btn_pick.setText("Select Points")
            self.btn_refine.setText("Click to Correct")
            if getattr(self, 'selector', None): self.selector.set_active(False)      
        else:
            self.btn_refine.setText("Refine Fit")
            if getattr(self, 'selector', None): self.selector.set_active(True)       

    def _clear_points(self):
        self.points = []
        if getattr(self, 'scatter_pts', None): self.scatter_pts.remove(); self.scatter_pts = None
        if getattr(self, 'line_fit', None): self.line_fit.remove(); self.line_fit = None
        if getattr(self, 'scatter_overlay', None): self.scatter_overlay.remove(); self.scatter_overlay = None
        
        self.val_gr.setText("")
        self.btn_pick.setChecked(False); self.btn_pick.setText("Select Points"); self.picking_points = False
        self.btn_refine.setChecked(False); self.btn_refine.setText("Refine Fit"); self.refining_modes = False
        self.mode_overrides = {}; self.last_box_bounds = None
        if getattr(self, 'selector', None): self.selector.set_active(True)           
        self.ax_reg.clear(); self.canvas_reg.draw(); self.canvas_hm.draw()

    def on_box_select(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.last_box_bounds = (min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2))
        self.mode_overrides = {}                                                     
        self._run_mode_fitting() 

    def _run_mode_fitting(self):
        if not getattr(self, 'last_box_bounds', None): return
        t_min, t_max, d_min, d_max = self.last_box_bounds
        
        dates = mdates.date2num(self.day_df.index)
        t_mask = (dates >= t_min) & (dates <= t_max)
        d_mask = (self.diams >= d_min) & (self.diams <= d_max)
        if not np.any(t_mask) or not np.any(d_mask): return
        
        subset_pnsd = self.day_df.iloc[t_mask, d_mask].to_numpy()
        sub_dates = dates[t_mask]
        active_diams = self.diams[d_mask]
        limit_jump = self.chk_limit_jump.isChecked()
        try: max_jump = float(self.val_max_jump.text())
        except ValueError: max_jump = 15.0
        
        valid_indices, mode_diams, snapshots = fit_modes_to_pnsd(subset_pnsd, active_diams, limit_jump, max_jump, self.mode_overrides)
        if len(mode_diams) < 3: return QMessageBox.warning(self, "Fitting Failed", "Not enough distinct peaks found.")
            
        valid_dates = sub_dates[valid_indices]
        mode_diams = np.array(mode_diams)
        self.fit_snapshots = [(valid_dates[i], snap[0], snap[1], snap[2]) for i, snap in enumerate(snapshots)]
        
        if getattr(self, 'scatter_overlay', None): self.scatter_overlay.remove()
        self.scatter_overlay = self.ax_hm.scatter(valid_dates, mode_diams, c='magenta', marker='x', s=40, zorder=3)
        self.ax_hm.set_ylim([self.diams.min(), self.diams.max()]); self.canvas_hm.draw_idle()
        
        self.slider_browse.setEnabled(True); self.slider_browse.setRange(0, len(self.fit_snapshots) - 1); self.slider_browse.setValue(0)
        self.update_browser()
        
        time_hours = (valid_dates - valid_dates[0]) * 24
        self.gr_result, intercept = calc_growth_rate(time_hours, mode_diams)
        self.val_gr.setText(f"{self.gr_result:.2f}")
        
        self.ax_reg.clear()
        self.ax_reg.plot(time_hours, mode_diams, 'ko', label="Fitted Modes")
        self.ax_reg.plot(time_hours, intercept + self.gr_result * time_hours, 'r-', label=f"GR: {self.gr_result:.2f} nm/h")
        self.ax_reg.set_xlabel("Time (h)"); self.ax_reg.set_ylabel("Mode (nm)")
        self.ax_reg.legend(loc='upper left', fontsize=9)
        self.fig_reg.tight_layout(pad=1.5); self.canvas_reg.draw()

    def _on_click(self, event):
        if event.inaxes not in [self.ax_hm, self.ax_hm_line]: return                 
        xdata, ydata = self.ax_hm.transData.inverted().transform((event.x, event.y)) 
        
        if getattr(self, 'refining_modes', False):
            if not getattr(self, 'last_box_bounds', None): return
            t_min, t_max, _, _ = self.last_box_bounds
            dates = mdates.date2num(self.day_df.index)
            sub_dates = dates[(dates >= t_min) & (dates <= t_max)] 
            
            if len(sub_dates) == 0: return
            closest_time_idx = np.argmin(np.abs(sub_dates - xdata))                  
            self.mode_overrides[closest_time_idx] = ydata                            
            self._run_mode_fitting() 
            return
            
        if getattr(self, 'picking_points', False):
            if not hasattr(self, 'points'): self.points = [] 
            if len(self.points) < 2:                                             
                self.points.append((xdata, ydata)) 
                self.btn_pick.setText(f"Click plot ({len(self.points)}/2)")      
                
                x_vals = [p[0] for p in self.points]; y_vals = [p[1] for p in self.points] 
                if getattr(self, 'scatter_pts', None): self.scatter_pts.remove()                   
                self.scatter_pts = self.ax_hm.scatter(x_vals, y_vals, color='white', marker='x', s=100, zorder=5) 
                
                if len(self.points) == 2:                                        
                    self.picking_points = False
                    self.btn_pick.setChecked(False)
                    self.btn_pick.setText("Select Points")
                    if getattr(self, 'selector', None): self.selector.set_active(True) 
                    if getattr(self, 'line_fit', None): self.line_fit.remove()       
                    self.line_fit, = self.ax_hm.plot(x_vals, y_vals, color='white', linestyle='--', lw=2) 
                    self._calculate_growth_rate()
                self.canvas_hm.draw()

    def _calculate_growth_rate(self):
        if len(self.points) != 2: return
        t1, t2 = self.points[0][0], self.points[1][0]
        dp1, dp2 = self.points[0][1], self.points[1][1]
        dt_hours = abs(t2 - t1) * 24.0
        ddp = abs(dp2 - dp1)
        if dt_hours > 0:
            gr = ddp / dt_hours; self.val_gr.setText(f"{gr:.2f}"); self.gr_result = gr
        else: self.val_gr.setText("Error")

    def update_browser(self):
        if not self.fit_snapshots: return
        idx = self.slider_browse.value()
        time_val, diams, pnsd, peak_dp = self.fit_snapshots[idx]
        self.ax_browse.clear()
        self.ax_browse.plot(diams, pnsd, 'b-', label='PNSD')
        self.ax_browse.axvline(peak_dp, color='magenta', linestyle='--', label=f'Peak: {peak_dp:.1f}nm')
        self.ax_browse.set_xscale('log')
        self.ax_browse.set_title(mdates.num2date(time_val).strftime('%H:%M'))
        self.ax_browse.legend(fontsize=8)
        self.fig_browse.tight_layout(pad=1.5); self.canvas_browse.draw()

    def on_class_button_clicked(self, text):
        if text.startswith("Custom"):
            new_text, ok = QInputDialog.getText(self, "Custom Class", "Enter custom classification:")
            if ok and new_text: text = new_text
            else: return

        date_str = str(self.daily_groups[self.current_day_idx][0])
        self.classifications[date_str] = text                                        
        self.update_class_button_styles(text)
        if text == "Non-NPF": self.auto_calculate_non_npf()

    def update_class_button_styles(self, active_text):
        standard = "QPushButton { background-color: #e2d5cb; border: 1px solid #b3a8a0; padding: 5px; border-radius: 3px; }"
        active = "QPushButton { background-color: #4a4a4a; color: white; border: 1px solid #33302e; padding: 5px; border-radius: 3px; font-weight: bold; }"
        for btn in self.class_buttons:
            if btn.text() == active_text or (btn.text().startswith("Custom") and active_text not in ["NPF", "Non-NPF", "Undefined", "Burst"]):
                btn.setStyleSheet(active)
                if btn.text().startswith("Custom"): btn.setText(f"Custom ({active_text})")
            else:
                btn.setStyleSheet(standard)
                if btn.text().startswith("Custom"): btn.setText("Custom")

    def auto_calculate_non_npf(self):
        try: dlogdp = float(self.val_dlogdp.text())
        except ValueError: return QMessageBox.warning(self, "Error", "dlogDp must be a number.")
        cs_series = calc_condensation_sink(self.diams, self.day_df.to_numpy(), dlogdp)
        self.gr_result = np.nan
        
        self.j_cs_data = pd.DataFrame({
            'date': self.day_df.index, 
            'Class': "Non-NPF", 
            'In_GR_Window': 0,
            'J_Window': "N/A",
            'Mode_Dp': np.nan, 
            'GR': np.nan, 'J': np.nan, 'J[dNdt]': np.nan, 'J[GR]': np.nan, 'J[coag]': np.nan, 
            'CS': cs_series,
            'm': np.nan,
            'J1.5': np.nan
        }).set_index('date')

        self.ax_j.clear(); self.ax_cs.clear(); self.ax_reg.clear(); self.canvas_reg.draw()
        time_axis = self.day_df.index
        self.ax_cs.plot(time_axis, cs_series, color='orange', label='CS')
        self.ax_cs.set_ylabel("Condensational Sink (CS)", color='orange')
        self.ax_cs.yaxis.set_label_position("right"); self.ax_cs.yaxis.tick_right()
        self.ax_cs.tick_params(axis='y', colors='orange'); self.ax_cs.spines['right'].set_color('orange')
        self.ax_cs.legend(loc='upper right')
        
        if len(time_axis) > 1: self.ax_cs.set_xlim([time_axis[0], time_axis[-1]])
        self.ax_cs.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        self.ax_cs.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.fig_diur.tight_layout(pad=1.5); self.canvas_diur.draw()

    def calculate_j_and_cs(self):
        date_str = str(self.day_df.index.date[0])
        idx_date = pd.to_datetime(date_str)
        if date_str in self.classifications:
            active_class = self.classifications[date_str]
        elif not getattr(self, 'current_ml_results', pd.DataFrame()).empty and idx_date in self.current_ml_results.index:
            active_class = self.current_ml_results.loc[idx_date, 'class']
        else:
            active_class = "Undefined"
        
        gr_to_use = self.gr_result
        if active_class == "Burst" and (self.gr_result is None or np.isnan(self.gr_result)):
            gr_to_use = 1.0  
        elif self.gr_result is None or np.isnan(self.gr_result):
            return QMessageBox.warning(self, "No GR", "Calculate GR first.")
            
        try: 
            j_min = float(self.j_min_dp.text()); j_max = float(self.j_max_dp.text())
            dlogdp = float(self.val_dlogdp.text())
        except ValueError: return QMessageBox.warning(self, "Input Error", "Inputs must be numbers.")
        
        j_window_str = f"{j_min}-{j_max}"
        pnsd = self.day_df.to_numpy()
        cs_series = calc_condensation_sink(self.diams, pnsd, dlogdp)
        coags_matrix = calc_coagulation_sink(self.diams, pnsd, dlogdp)
        j_total, dN_dt, gr_term, coag_term = calc_formation_rate(self.diams, pnsd, dlogdp, gr_to_use, j_min, j_max, coags_matrix)
        
        m_series = np.full_like(j_total, np.nan)
        j15_series = np.full_like(j_total, np.nan)
        
        if self.chk_j15.isChecked() and gr_to_use > 0:
            idx_d1 = np.argmin(np.abs(self.diams - 1.5))
            idx_dx = np.argmin(np.abs(self.diams - j_min))
            
            coags_d1_series = coags_matrix[:, idx_d1]
            coags_dx_series = coags_matrix[:, idx_dx]
            
            if j_min != 1.5:
                m_series = calculate_m(coags_d1_series, coags_dx_series, 1.5, j_min)
                m_safe = np.where(m_series == -1, -0.999, m_series)
                j15_series = calculate_j1_5(1.5, j_total, j_min, coags_d1_series * 3600, gr_to_use, m_safe)
            else:
                j15_series = j_total
        
        dates = mdates.date2num(self.day_df.index)
        mode_dps = np.full(len(dates), np.nan)                                       
        in_window = np.zeros(len(dates), dtype=int)                                  
        
        if getattr(self, 'last_box_bounds', None):
            t_min, t_max, _, _ = self.last_box_bounds
            snapshot_dict = {snap[0]: snap[3] for snap in getattr(self, 'fit_snapshots', [])}
            
            for i, d in enumerate(dates):
                if t_min <= d <= t_max:                                              
                    in_window[i] = 1                                                 
                    if d in snapshot_dict:
                        mode_dps[i] = snapshot_dict[d]                               
                    else:
                        mode_dps[i] = self.diams[np.argmax(pnsd[i])]                 
        
        self.j_cs_data = pd.DataFrame({
            'date': self.day_df.index, 
            'Class': active_class,
            'In_GR_Window': in_window,
            'J_Window': j_window_str,
            'Mode_Dp': mode_dps, 
            'GR': gr_to_use,
            'J': j_total, 'J[dNdt]': dN_dt, 'J[GR]': gr_term, 'J[coag]': coag_term, 'CS': cs_series,
            'm': m_series,
            'J1.5': j15_series
        }).set_index('date')
        
        self.ax_j.clear(); self.ax_cs.clear(); time_axis = self.day_df.index
        self.ax_j.plot(time_axis, j_total, color='blue', label='J')
        
        if self.chk_j15.isChecked():
            self.ax_j.plot(time_axis, j15_series, color='purple', linestyle='--', label='J1.5')
            
        self.ax_j.set_ylabel("Formation Rate (J)", color='blue'); self.ax_j.tick_params(axis='y', colors='blue'); self.ax_j.spines['left'].set_color('blue')
        self.ax_cs.plot(time_axis, cs_series, color='orange', label='CS'); self.ax_cs.set_ylabel("Condensational Sink (CS)", color='orange')
        self.ax_cs.yaxis.set_label_position("right"); self.ax_cs.yaxis.tick_right(); self.ax_cs.tick_params(axis='y', colors='orange'); self.ax_cs.spines['right'].set_color('orange')
        
        self.ax_j.legend(loc='upper left')
        self.ax_cs.legend(loc='upper right')
        
        if len(time_axis) > 1: self.ax_j.set_xlim([time_axis[0], time_axis[-1]])
        self.ax_j.xaxis.set_major_locator(mdates.HourLocator(interval=4)); self.ax_j.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.fig_diur.tight_layout(pad=1.5); self.canvas_diur.draw()

    def _choose_new_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Set CSV Output File", "", "CSV Files (*.csv)")
        if path:
            self.last_csv_path = path
            self.btn_export.setText(f"Send to CSV ({os.path.basename(path)})")

    def export_to_csv(self):
        if self.j_cs_data is None: return
        
        if not self.last_csv_path: 
            path, _ = QFileDialog.getSaveFileName(self, "Append or Save Analysis to CSV", "", "CSV Files (*.csv)")
            if not path: return
            self.last_csv_path = path                                                
            self.btn_export.setText(f"Send to CSV ({os.path.basename(path)})")       
            
        path = self.last_csv_path 
        
        try:
            if os.path.exists(path):
                existing_df = pd.read_csv(path, index_col=0, parse_dates=True)
                combined_df = pd.concat([existing_df, self.j_cs_data])
                combined_df.to_csv(path)
            else:
                self.j_cs_data.to_csv(path)           
            
            for index, row in self.j_cs_data.iterrows():
                r_idx = self.csv_table.rowCount()                                    
                self.csv_table.insertRow(r_idx)                                      
                fmt = lambda v, f: "NA" if pd.isna(v) else f"{v:{f}}"                
                
                self.csv_table.setItem(r_idx, 0, QTableWidgetItem(str(index)))
                self.csv_table.setItem(r_idx, 1, QTableWidgetItem(str(row['Class'])))
                self.csv_table.setItem(r_idx, 2, QTableWidgetItem(str(int(row['In_GR_Window']))))
                self.csv_table.setItem(r_idx, 3, QTableWidgetItem(str(row['J_Window'])))
                self.csv_table.setItem(r_idx, 4, QTableWidgetItem(fmt(row['Mode_Dp'], '.2f')))
                self.csv_table.setItem(r_idx, 5, QTableWidgetItem(fmt(row['GR'], '.3f')))
                self.csv_table.setItem(r_idx, 6, QTableWidgetItem(fmt(row['J'], '.3e')))
                self.csv_table.setItem(r_idx, 7, QTableWidgetItem(fmt(row['J[dNdt]'], '.3e')))
                self.csv_table.setItem(r_idx, 8, QTableWidgetItem(fmt(row['J[GR]'], '.3e')))
                self.csv_table.setItem(r_idx, 9, QTableWidgetItem(fmt(row['J[coag]'], '.3e')))
                self.csv_table.setItem(r_idx, 10, QTableWidgetItem(fmt(row['CS'], '.3e')))
                self.csv_table.setItem(r_idx, 11, QTableWidgetItem(fmt(row['m'], '.3f')))
                self.csv_table.setItem(r_idx, 12, QTableWidgetItem(fmt(row['J1.5'], '.3e')))
        except Exception as e: 
            QMessageBox.critical(self, "Error", str(e))                              

    def _show_diurnals(self):
        try:
            j_min = float(self.j_min_dp.text()); j_max = float(self.j_max_dp.text())
            dlogdp = float(self.val_dlogdp.text())
        except ValueError: return QMessageBox.warning(self, "Error", "Check J bounds and dlogDp.")
        
        if not self.classifications: return QMessageBox.warning(self, "No Data", "Classify at least one day first!") 
        dummy_ml = pd.DataFrame([{'date': pd.to_datetime(d), 'class': c} for d, c in self.classifications.items()]).set_index('date') 

        self.diurnal_win = DiurnalSummaryWindow(self.df, self.diams, dummy_ml, j_min, j_max, dlogdp, self) 
        self.diurnal_win.show()

    def _show_coags_map(self):
        if self.df is None: return
        try: dlogdp = float(self.val_dlogdp.text())
        except ValueError: return QMessageBox.warning(self, "Error", "dlogDp must be a number.")
        self.coags_win = CoagSWindow(self.day_df, self.diams, dlogdp, self)
        self.coags_win.show()

    def jump_to_date(self, index):
        if index >= 0 and index != self.current_day_idx: self.current_day_idx = index; self.update_day()

    def prev_day(self):
        if self.current_day_idx > 0: self.current_day_idx -= 1; self.update_day()

    def next_day(self):
        if self.current_day_idx < len(self.daily_groups) - 1: self.current_day_idx += 1; self.update_day()
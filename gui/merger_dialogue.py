import numpy as np                                                           # Array operations
import pandas as pd                                                          # Data manipulation
from pathlib import Path                                                     # Path handling
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,      # UI Elements
                             QPushButton, QSlider, QLineEdit, QCheckBox, 
                             QMessageBox, QGridLayout, QComboBox)
from PyQt6.QtCore import Qt                                                  # Core Qt enums
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg              # Canvas
from matplotlib.figure import Figure                                         # Plot figure
from utils.data_loader import regrid_pnsd_cdf, DataFile                      # Import spline tool

class InstrumentMergerDialog(QDialog):
    def __init__(self, file1: DataFile, file2: DataFile, parent=None):       
        super().__init__(parent)
        self.setWindowTitle("Advanced Instrument Merger")                    
        self.resize(900, 700)                                                
        
        df1_aligned, df2_aligned = file1.df.align(file2.df, join='inner', axis=0) # Sync timebases
        if df1_aligned.empty:                                                
            QMessageBox.critical(self, "Error", "These files have no overlapping timestamps!")
            self.reject()
            return
            
        self.name1 = Path(file1.path).name                                   
        self.name2 = Path(file2.path).name                                   
        self.df1 = df1_aligned                                               
        self.df2 = df2_aligned                                               
        self.diams1 = np.array(file1.diameters)                              
        self.diams2 = np.array(file2.diameters)                              
        
        self.all_diams = np.unique(np.concatenate([self.diams1, self.diams2])) 
        self.all_diams.sort()                                                
        
        self.final_df = None                                                 
        self.final_diams = None                                              
        
        self._build_ui()                                                     
        self._update_plot()                                                  

    def _build_ui(self):
        layout = QVBoxLayout(self)                                           
        
        self.fig = Figure(figsize=(8, 4))                                    
        self.canvas = FigureCanvasQTAgg(self.fig)                            
        self.ax = self.fig.add_subplot(111)                                  
        layout.addWidget(self.canvas, stretch=1)                             
        
        ctrl_grid = QGridLayout()                                            
        
        # Upper/Lower Priority Dropdown
        ctrl_grid.addWidget(QLabel("Merge Priority:"), 0, 0)                 
        self.combo_order = QComboBox()                                       
        self.combo_order.addItems([                                          
            f"{self.name1} (Low Dp) + {self.name2} (High Dp)",
            f"{self.name2} (Low Dp) + {self.name1} (High Dp)"
        ])
        self.combo_order.currentIndexChanged.connect(self._update_plot)      
        ctrl_grid.addWidget(self.combo_order, 0, 1, 1, 2)                    
        
        # Factor 1 Slider
        ctrl_grid.addWidget(QLabel(f"Factor: {self.name1}"), 1, 0)           
        self.slider_f1 = QSlider(Qt.Orientation.Horizontal)                  
        self.slider_f1.setRange(-100, 100)                                   
        self.slider_f1.setValue(0)                                           
        self.slider_f1.valueChanged.connect(self._update_plot)               
        ctrl_grid.addWidget(self.slider_f1, 1, 1)                            
        self.lbl_f1_val = QLabel("1.00 x")                                   
        ctrl_grid.addWidget(self.lbl_f1_val, 1, 2)                           
        
        # Factor 2 Slider
        ctrl_grid.addWidget(QLabel(f"Factor: {self.name2}"), 2, 0)           
        self.slider_f2 = QSlider(Qt.Orientation.Horizontal)                  
        self.slider_f2.setRange(-100, 100)                                   
        self.slider_f2.setValue(0)                                           
        self.slider_f2.valueChanged.connect(self._update_plot)               
        ctrl_grid.addWidget(self.slider_f2, 2, 1)                            
        self.lbl_f2_val = QLabel("1.00 x")                                   
        ctrl_grid.addWidget(self.lbl_f2_val, 2, 2)                           
        
        # Merge Point Slider
        ctrl_grid.addWidget(QLabel("Merge Diameter (nm):"), 3, 0)            
        self.slider_dp = QSlider(Qt.Orientation.Horizontal)                  
        self.slider_dp.setRange(0, len(self.all_diams) - 1)                  
        self.slider_dp.setValue(len(self.all_diams) // 2)                    
        self.slider_dp.valueChanged.connect(self._update_plot)               
        ctrl_grid.addWidget(self.slider_dp, 3, 1)                            
        self.lbl_dp_val = QLabel(f"{self.all_diams[self.slider_dp.value()]:.1f}") 
        ctrl_grid.addWidget(self.lbl_dp_val, 3, 2)                           
        
        layout.addLayout(ctrl_grid)                                          
        
        # Spline Options
        opt_layout = QHBoxLayout()                                           
        self.chk_spline = QCheckBox("Apply CDF Spline Regridding")           
        self.chk_spline.setChecked(True)                                     
        self.chk_spline.stateChanged.connect(self._update_plot)              
        opt_layout.addWidget(self.chk_spline)                                
        
        opt_layout.addWidget(QLabel("Channels/Decade:"))                     
        self.val_cpd = QLineEdit("64")                                       
        self.val_cpd.setFixedWidth(40)                                       
        self.val_cpd.textChanged.connect(self._update_plot)                  
        opt_layout.addWidget(self.val_cpd)                                   
        opt_layout.addStretch()                                              
        layout.addLayout(opt_layout)                                         
        
        btn_layout = QHBoxLayout()                                           
        btn_cancel = QPushButton("Cancel")                                   
        btn_cancel.clicked.connect(self.reject)                              
        btn_layout.addWidget(btn_cancel)                                     
        btn_layout.addStretch()                                              
        
        btn_apply = QPushButton("Generate & Add to File List")               
        btn_apply.setStyleSheet("font-weight: bold; background-color: #d1c4ba;") 
        btn_apply.clicked.connect(self._apply_merge)                         
        btn_layout.addWidget(btn_apply)                                      
        layout.addLayout(btn_layout)                                         

    def _get_factors(self):                                                  
        f1 = 10 ** (self.slider_f1.value() / 100.0)                          
        f2 = 10 ** (self.slider_f2.value() / 100.0)                          
        merge_dp = self.all_diams[self.slider_dp.value()]                    
        return f1, f2, merge_dp                                              

    def _update_plot(self):                                                  
        f1, f2, merge_dp = self._get_factors()                               
        self.lbl_f1_val.setText(f"{f1:.2f} x")                               
        self.lbl_f2_val.setText(f"{f2:.2f} x")                               
        self.lbl_dp_val.setText(f"{merge_dp:.1f} nm")                        
        
        self.ax.clear()                                                      
        
        # Calculate Means AND Standard Deviations
        mean1 = self.df1.mean(axis=0).to_numpy() * f1
        std1 = self.df1.std(axis=0).to_numpy() * f1
        mean2 = self.df2.mean(axis=0).to_numpy() * f2
        std2 = self.df2.std(axis=0).to_numpy() * f2
        
        is_order_1_low = self.combo_order.currentIndex() == 0                
        keep1_mask = self.diams1 <= merge_dp if is_order_1_low else self.diams1 > merge_dp
        keep2_mask = self.diams2 > merge_dp if is_order_1_low else self.diams2 <= merge_dp
        
        # Plot discarded regions (dashed lines, very faint shading)
        self.ax.plot(self.diams1[~keep1_mask], mean1[~keep1_mask], 'b--', alpha=0.4) 
        self.ax.fill_between(self.diams1[~keep1_mask], (mean1 - std1)[~keep1_mask], (mean1 + std1)[~keep1_mask], color='blue', alpha=0.05)
        
        self.ax.plot(self.diams2[~keep2_mask], mean2[~keep2_mask], 'r--', alpha=0.4) 
        self.ax.fill_between(self.diams2[~keep2_mask], (mean2 - std2)[~keep2_mask], (mean2 + std2)[~keep2_mask], color='red', alpha=0.05)
        
        # Plot kept regions (solid lines, standard shading)
        self.ax.plot(self.diams1[keep1_mask], mean1[keep1_mask], 'b-', linewidth=2, label=f"{self.name1} (Kept)") 
        self.ax.fill_between(self.diams1[keep1_mask], (mean1 - std1)[keep1_mask], (mean1 + std1)[keep1_mask], color='blue', alpha=0.2)
        
        self.ax.plot(self.diams2[keep2_mask], mean2[keep2_mask], 'r-', linewidth=2, label=f"{self.name2} (Kept)") 
        self.ax.fill_between(self.diams2[keep2_mask], (mean2 - std2)[keep2_mask], (mean2 + std2)[keep2_mask], color='red', alpha=0.2)
        
        self.ax.axvline(merge_dp, color='k', linestyle=':', label='Merge Point') 
        
        try: cpd = float(self.val_cpd.text())                                
        except ValueError: cpd = 64.0                                        
        
        if self.chk_spline.isChecked() and cpd > 0:                          
            merged_temp, _ = self._process_merge(f1, f2, merge_dp, True, cpd)
            mean_merged = merged_temp.mean(axis=0).to_numpy()                
            self.ax.plot(merged_temp.columns, mean_merged, 'k-', linewidth=2, label='Final Spline') 
            
        self.ax.set_xscale('log')                                            
        self.ax.set_yscale('log')                                            
        self.ax.set_xlabel('Diameter (nm)')                                  
        self.ax.set_ylabel('dN/dlogDp')                                      
        self.ax.legend(loc='upper right', fontsize=8)                        
        self.fig.tight_layout()                                              
        self.canvas.draw()

    def _process_merge(self, f1, f2, merge_dp, apply_spline, cpd):           
        df1_scaled = self.df1 * f1                                           
        df2_scaled = self.df2 * f2                                           
        
        is_order_1_low = self.combo_order.currentIndex() == 0                
        keep_1 = self.diams1 <= merge_dp if is_order_1_low else self.diams1 > merge_dp
        keep_2 = self.diams2 > merge_dp if is_order_1_low else self.diams2 <= merge_dp
        
        splice_df = pd.concat([df1_scaled.loc[:, keep_1], df2_scaled.loc[:, keep_2]], axis=1) 
        splice_diams = np.concatenate([self.diams1[keep_1], self.diams2[keep_2]]) 
        
        sort_idx = np.argsort(splice_diams)                                  # Guarantee array is sorted
        splice_diams = splice_diams[sort_idx]                                # Sort diameters
        splice_df = splice_df.iloc[:, sort_idx]                              # Sort dataframe columns
        
        if apply_spline:                                                     
            return regrid_pnsd_cdf(splice_df, splice_diams, cpd)             
        return splice_df, splice_diams                                       

    def _apply_merge(self):                                                  
        f1, f2, merge_dp = self._get_factors()                               
        apply_spline = self.chk_spline.isChecked()                           
        try: cpd = float(self.val_cpd.text())                                
        except ValueError: cpd = 64.0                                        
        
        self.final_df, self.final_diams = self._process_merge(f1, f2, merge_dp, apply_spline, cpd) 
        self.accept()
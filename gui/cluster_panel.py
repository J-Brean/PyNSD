"""
gui/cluster_panel.py
--------------------
Cluster Analysis panel for PNSD data.

Workflow
--------
1. COMPARE  — Run multiple clustering algorithms side-by-side and score them.
2. TUNE     — Sweep k / eps for the chosen algorithm; elbow + silhouette plots.
3. CLUSTER  — Apply final settings; view per-cluster diurnal, contour,
              DoW, MoY and full time-series plots.
4. EXPORT   — Save the cluster-assignment time series as CSV.

Data modes
----------
• Daily   — one row per day; columns = all [bin × hour] combinations (wide).
• Hourly  — one row per hour; columns = diameter bins only.
"""

from __future__ import annotations

import math
import warnings
from io import StringIO

import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import (DBSCAN, AgglomerativeClustering, KMeans,
                             MiniBatchKMeans, SpectralClustering)
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, 
                             QFileDialog, QGroupBox, QHBoxLayout, QLabel, 
                             QLineEdit, QProgressBar, QPushButton, QScrollArea, 
                             QSplitter, QTabWidget, QTextEdit, QVBoxLayout, 
                             QWidget, QDialog, QMessageBox)

# ── Matplotlib global style ─────────────────────────────────────────────── #
rcParams['font.family'] = 'serif'
rcParams['font.serif']  = ['Georgia', 'Times New Roman']
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['figure.facecolor'] = '#fff1e5'
rcParams['axes.facecolor']   = '#fff1e5'
rcParams['savefig.facecolor'] = '#fff1e5'

_FT_BG = '#fff1e5'
_ACCENT_COLORS = ['#0f6e56', '#185fa5', '#854f0b', '#a32d2d',
                  '#533ab7', '#3b6d11', '#854f0b', '#4b1528', '#202020']

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
        ctrl_layout.addWidget(QLabel("Width (in):"))
        self.val_w = QLineEdit(str(fig.get_figwidth()))
        ctrl_layout.addWidget(self.val_w)
        
        ctrl_layout.addWidget(QLabel("Height (in):"))
        self.val_h = QLineEdit(str(fig.get_figheight()))
        ctrl_layout.addWidget(self.val_h)
        
        self.btn_save = QPushButton("💾 Save Image")
        self.btn_save.clicked.connect(self.save_plot)
        ctrl_layout.addWidget(self.btn_save)
        
        self.btn_skip = QPushButton("Skip")
        self.btn_skip.clicked.connect(self.reject)
        ctrl_layout.addWidget(self.btn_skip)
        
        self.layout.addLayout(ctrl_layout)
        self.layout.addWidget(self.canvas, stretch=1)
        
        w = int(fig.get_figwidth() * fig.dpi)
        h = int(fig.get_figheight() * fig.dpi) + 50
        self.resize(w, h)

    def save_plot(self):
        try:
            w, h = float(self.val_w.text()), float(self.val_h.text())
            self.fig.set_size_inches(w, h)
        except ValueError:
            pass
            
        path, _ = QFileDialog.getSaveFileName(self, "Save Plot", "", "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)")
        if path:
            self.fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
            self.accept()

    def closeEvent(self, event):
        self.restore_callback()
        super().closeEvent(event)


# ─────────────────────────────────────────────────────────────────────────── #
# Background worker thread
# ─────────────────────────────────────────────────────────────────────────── #

class ClusterWorker(QThread):
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, task: str, kwargs: dict):
        super().__init__()
        self.task   = task
        self.kwargs = kwargs

    def run(self):
        try:
            if self.task == "compare":
                result = _run_compare(**self.kwargs)
            elif self.task == "tune":
                result = _run_tune(**self.kwargs)
            elif self.task == "cluster":
                result = _run_cluster(**self.kwargs)
            else:
                raise ValueError(f"Unknown task: {self.task}")
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


# ─────────────────────────────────────────────────────────────────────────── #
# Pure-compute helpers (called inside worker thread)
# ─────────────────────────────────────────────────────────────────────────── #

def _scale(X: np.ndarray, method: str) -> np.ndarray:
    if method == "Standard (z-score)":
        return StandardScaler().fit_transform(X)
    if method == "MinMax [0,1]":
        return MinMaxScaler().fit_transform(X)
    if method == "Robust (median/IQR)":
        return RobustScaler().fit_transform(X)
    if method == "Log + Standard":
        X = np.log1p(np.clip(X, 0, None))
        return StandardScaler().fit_transform(X)
    return X


def _dim_reduce(X: np.ndarray, method: str, n: int) -> np.ndarray:
    if n <= 0 or n >= X.shape[1] or method == "None":
        return X
        
    if method == "PCA":
        return PCA(n_components=n, random_state=42).fit_transform(X)
        
    elif method == "Autoencoder (PyTorch)":
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise ImportError("PyTorch is required for Autoencoders! Please run 'pip install torch' in your terminal.")

        class AE(nn.Module):
            def __init__(self, input_dim, latent_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, latent_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, input_dim)
                )
            def forward(self, x):
                return self.decoder(self.encoder(x))

        tensor_X = torch.FloatTensor(X)
        model = AE(X.shape[1], n)
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.MSELoss()

        model.train()
        # Drop epochs slightly for faster sweep
        for _ in range(120):
            optimizer.zero_grad()
            loss = criterion(model(tensor_X), tensor_X)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            return model.encoder(tensor_X).numpy()
            
    return X


def _score(X: np.ndarray, labels: np.ndarray) -> dict:
    unique = set(labels[labels >= 0])
    if len(unique) < 2:
        return {"silhouette": np.nan, "calinski": np.nan, "davies": np.nan}
    mask = labels >= 0
    try:
        sil = silhouette_score(X[mask], labels[mask], sample_size=min(1000, mask.sum()))
    except: sil = np.nan
    try:
        cal = calinski_harabasz_score(X[mask], labels[mask])
    except: cal = np.nan
    try:
        dav = davies_bouldin_score(X[mask], labels[mask])
    except: dav = np.nan
    return {"silhouette": sil, "calinski": cal, "davies": dav}


def _run_compare(X_raw, k, scaler, dim_method, dim_n, models):
    X = _scale(X_raw, scaler)
    X = _dim_reduce(X, dim_method, dim_n)
    rows = []
    for name in models:
        labels = _fit_model(name, X, k=k)
        s = _score(X, labels)
        n_clusters = len(set(labels[labels >= 0]))
        rows.append({
            "Model": name,
            "Clusters": n_clusters,
            "Silhouette ↑": round(s["silhouette"], 4) if not np.isnan(s["silhouette"]) else "—",
            "Calinski ↑":   round(s["calinski"], 1)   if not np.isnan(s["calinski"])   else "—",
            "Davies-Bouldin ↓": round(s["davies"], 4) if not np.isnan(s["davies"])     else "—",
        })
    return {"type": "compare", "rows": rows}


def _fit_model(name: str, X: np.ndarray, k: int, eps: float = 0.5,
               min_samples: int = 5, linkage_method: str = "ward") -> np.ndarray:
    if name == "K-Means":
        return KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    if name == "Mini-Batch K-Means":
        return MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    if name == "Agglomerative (Ward)":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    if name == "Agglomerative (Average)":
        return AgglomerativeClustering(n_clusters=k, linkage="average").fit_predict(X)
    if name == "Agglomerative (Complete)":
        return AgglomerativeClustering(n_clusters=k, linkage="complete").fit_predict(X)
    if name == "Gaussian Mixture Model (GMM)":
        return GaussianMixture(n_components=k, random_state=42).fit_predict(X)
    if name == "Spectral Clustering":
        return SpectralClustering(n_clusters=k, random_state=42, affinity='nearest_neighbors').fit_predict(X)
    if name == "DBSCAN":
        return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    raise ValueError(f"Unknown model: {name}")


def _run_tune(X_raw, scaler, dim_method, dim_n, model, k_min, k_max, eps_min, eps_max, eps_steps):
    X = _scale(X_raw, scaler)
    X = _dim_reduce(X, dim_method, dim_n)

    if model == "DBSCAN":
        eps_vals   = np.linspace(eps_min, eps_max, eps_steps)
        silhouettes, n_clusters = [], []
        for eps in eps_vals:
            labels  = DBSCAN(eps=eps, min_samples=5).fit_predict(X)
            s       = _score(X, labels)
            silhouettes.append(s["silhouette"])
            n_clusters.append(len(set(labels[labels >= 0])))
        return {"type": "tune_dbscan", "eps": eps_vals,
                "silhouettes": silhouettes, "n_clusters": n_clusters}

    ks, inertias, silhouettes, calinskis, davies = [], [], [], [], []
    for k in range(k_min, k_max + 1):
        labels = _fit_model(model, X, k=k)
        s = _score(X, labels)
        ks.append(k)
        silhouettes.append(s["silhouette"])
        calinskis.append(s["calinski"])
        davies.append(s["davies"])
        if model in ("K-Means", "Mini-Batch K-Means"):
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
            inertias.append(km.inertia_)
        else:
            inertias.append(np.nan)

    dendro_data = None
    if "Agglomerative" in model:
        link_map = {"Ward": "ward", "Average": "average", "Complete": "complete"}
        lmethod  = link_map.get(model.split("(")[-1].rstrip(")"), "ward")
        Z = linkage(X[:min(500, len(X))], method=lmethod)
        dendro_data = Z

    return {"type": "tune_k", "ks": ks, "inertias": inertias,
            "silhouettes": silhouettes, "calinskis": calinskis,
            "davies": davies, "dendro": dendro_data, "model": model}


def _run_cluster(X_raw, index, diams, scaler, dim_method, dim_n, model, k,
                 eps, min_samples, mode, df_hourly):
    X = _scale(X_raw, scaler)
    X = _dim_reduce(X, dim_method, dim_n)
    labels = _fit_model(model, X, k=k, eps=eps, min_samples=min_samples)
    
    return {
        "type": "cluster", 
        "labels": labels, 
        "index": index,
        "diams": diams, 
        "mode": mode, 
        "df_hourly": df_hourly,
        "scaler": scaler, 
        "model": model,
        "X_raw": X_raw 
    }

# ─────────────────────────────────────────────────────────────────────────── #
# Main panel widget
# ─────────────────────────────────────────────────────────────────────────── #

class ClusterPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._df       = None
        self._diams    = None
        self._results  = None
        self._worker   = None
        self._active_figures = []

        self._build_ui()

    def _info_btn(self, text, title="Information"):
        """Creates a clickable info button."""
        btn = QPushButton("ℹ️")
        btn.setFixedSize(24, 24)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet("border: none; font-size: 16px;") 
        btn.clicked.connect(lambda: QMessageBox.information(self, title, text))
        return btn

    # ────────────────────────────────────────────────────────────────────── #
    # UI construction                                                        #
    # ────────────────────────────────────────────────────────────────────── #

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 8)

        root.addWidget(self._build_settings_box())
        root.addWidget(self._build_progress_row())

        self._workspace = QTabWidget()
        self._workspace.addTab(self._build_compare_tab(),  "① Compare models")
        self._workspace.addTab(self._build_tune_tab(),     "② Tune k / ε")
        self._workspace.addTab(self._build_results_tab(),  "③ Results")
        root.addWidget(self._workspace, stretch=1)

    def _build_settings_box(self) -> QGroupBox:
        box = QGroupBox("Clustering settings")
        outer = QVBoxLayout(box)
        outer.setSpacing(6)
        
        # --- Row 1: Mode & Scaling ---
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Data mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Daily (bins × hours wide)", "Hourly (bins only)"])
        r1.addWidget(self._mode_combo)
        
        info_mode = ("<b>Data Mode defines the shape of the data being clustered:</b><br><br>"
                     "<b>Daily Mode:</b> Flattens all 24 hours of data into a single row. The algorithm groups "
                     "<i>entire days</i> together based on their diurnal evolution (e.g., 'Traffic Days' vs 'Clean Days').<br><br>"
                     "<b>Hourly Mode:</b> Treats every single hour independently. Good for grouping specific aerosol states, regardless of what time of day they occurred.")
        r1.addWidget(self._info_btn(info_mode, "Data Mode"))

        r1.addSpacing(15)
        r1.addWidget(QLabel("Scaling:"))
        self._scaler_combo = QComboBox() 
        self._scaler_combo.addItems(["Standard (z-score)", "MinMax [0,1]",
                                     "Robust (median/IQR)", "Log + Standard", "None"])
        self._scaler_combo.setFixedWidth(150)
        r1.addWidget(self._scaler_combo)
        
        info_scale = ("<b>Scaling ensures large values don't dominate the clustering:</b><br><br>"
                      "• <b>Standard:</b> Scales to mean=0, variance=1. (Default)<br>"
                      "• <b>MinMax:</b> Forces all values between 0 and 1.<br>"
                      "• <b>Robust:</b> Ignores massive outlier spikes when scaling.<br>"
                      "• <b>Log + Standard:</b> Best if your data varies by orders of magnitude.")
        r1.addWidget(self._info_btn(info_scale, "Scaling Methods"))
        
        self._btn_compare_scale = QPushButton("Compare Scalers")
        self._btn_compare_scale.clicked.connect(self._show_scale_comparison)
        r1.addWidget(self._btn_compare_scale)
        r1.addStretch()
        outer.addLayout(r1)

        # --- Row 2: Dimensionality Reduction ---
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Dim Reduction:"))
        self._dim_combo = QComboBox()
        self._dim_combo.addItems(["None", "PCA", "Autoencoder (PyTorch)"])
        r2.addWidget(self._dim_combo)

        r2.addWidget(QLabel("Dimensions:")) 
        self._pca_edit = QLineEdit("3")
        self._pca_edit.setFixedWidth(30)
        r2.addWidget(self._pca_edit)
        
        info_dim = ("<b>Dimensionality Reduction (The Bottleneck):</b><br>"
                    "This defines the 'resolution' of your data before clustering.<br><br>"
                    "• <b>PCA:</b> Linear reduction. Best for capturing the most variance.<br>"
                    "• <b>Autoencoder:</b> Non-linear neural network. Can find complex hidden patterns that PCA might miss.<br><br>"
                    "<b>Analogy:</b> Think of this value like image resolution. 1x1 pixel tells you nothing. "
                    "4K shows every speck of dust (noise). A 64x64 thumbnail is usually perfect to see the 'big picture' patterns.<br><br>"
                    "<b>Auto-Find Tool:</b> Quickly sweeps PCA dimensions from 2 to 10 and automatically locks in the value that produces the highest Silhouette score.")
        r2.addWidget(self._info_btn(info_dim, "Dimensionality Reduction"))
        
        self._btn_auto_dim = QPushButton("✨ Auto-Find")
        self._btn_auto_dim.clicked.connect(self._auto_find_dim)
        r2.addWidget(self._btn_auto_dim)
        
        self._btn_compare_dim = QPushButton("Compare Dim-Red")
        self._btn_compare_dim.clicked.connect(self._show_dim_comparison)
        r2.addWidget(self._btn_compare_dim)
        r2.addStretch()
        outer.addLayout(r2)

        # --- Row 3: Algorithm & Clustering ---
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Algorithm:"))
        self._model_combo = QComboBox()
        self._model_combo.addItems(["K-Means", "Mini-Batch K-Means", "Agglomerative (Ward)", 
                                    "Agglomerative (Average)", "Agglomerative (Complete)", 
                                    "Gaussian Mixture Model (GMM)", "Spectral Clustering", "DBSCAN"])
        self._model_combo.currentTextChanged.connect(self._on_model_changed)
        r3.addWidget(self._model_combo)

        info_algo = ("<b>Algorithms:</b><br><br>"
                     "• <b>K-Means:</b> Fast, relies on spherical clusters. Good baseline.<br>"
                     "• <b>Agglomerative (Ward):</b> Hierarchical. Minimises variance within clusters.<br>"
                     "• <b>GMM:</b> 'Soft' or Fuzzy clustering. Models clusters as probabilistic Gaussian distributions.<br>"
                     "• <b>Spectral:</b> Excellent for non-convex (weirdly shaped) data distributions.<br>"
                     "• <b>DBSCAN:</b> Density-based. Good for isolating noise. Does not force points into clusters.")
        r3.addWidget(self._info_btn(info_algo, "Algorithms"))

        r3.addWidget(QLabel("k:"))
        self._k_edit = QLineEdit("5")
        self._k_edit.setFixedWidth(30)
        r3.addWidget(self._k_edit)

        self._lbl_eps = QLabel("ε:")
        self._eps_edit = QLineEdit("0.5")
        self._eps_edit.setFixedWidth(40)
        
        self._lbl_noise = QLabel("Noise Sens:") 
        self._noise_slider = QLineEdit("5")
        self._noise_slider.setFixedWidth(30)
        
        r3.addWidget(self._lbl_eps)
        r3.addWidget(self._eps_edit)
        r3.addWidget(self._lbl_noise)
        r3.addWidget(self._noise_slider)
        
        info_dbscan = ("<b>DBSCAN Parameters:</b><br><br>"
                       "• <b>ε (Epsilon):</b> The maximum distance between two samples for one to be considered as in the neighborhood of the other.<br>"
                       "• <b>Noise Sens (Min Samples):</b> The number of samples in a neighborhood for a point to be considered as a core point. "
                       "Increasing this forces the algorithm to be stricter, classifying more edge-cases as 'Noise' (-1).")
        r3.addWidget(self._info_btn(info_dbscan, "DBSCAN Settings"))
        
        r3.addSpacing(20)
        self._cluster_btn = QPushButton("③ Run Final Clustering")
        self._cluster_btn.setStyleSheet("font-weight: bold; background-color: #d1c4ba;")
        self._cluster_btn.clicked.connect(self._start_cluster)
        r3.addWidget(self._cluster_btn)
        r3.addStretch()
        outer.addLayout(r3)
        
        # --- Row 4: Map & Exports ---
        r4 = QHBoxLayout()
        r4.addWidget(QLabel("Colour Map:"))
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(["turbo", "viridis", "plasma", "inferno"])
        self._cmap_combo.setFixedWidth(70)
        r4.addWidget(self._cmap_combo)

        r4.addWidget(QLabel("Min/Max:"))
        self._cbar_min = QLineEdit("1")
        self._cbar_min.setFixedWidth(30)
        r4.addWidget(self._cbar_min)

        self._cbar_max = QLineEdit("")
        self._cbar_max.setFixedWidth(30)
        r4.addWidget(self._cbar_max)

        r4.addSpacing(20)
        self._export_btn = QPushButton("Export CSV")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_csv)
        r4.addWidget(self._export_btn)
        
        self._export_plots_btn = QPushButton("📷 Export Active Tab Plots")
        self._export_plots_btn.clicked.connect(self._export_active_plots)
        r4.addWidget(self._export_plots_btn)
        r4.addStretch()

        outer.addLayout(r4)
        return box

    def _on_model_changed(self, text: str):
        is_dbscan = text == "DBSCAN"
        self._k_edit.setEnabled(not is_dbscan)
        self._eps_edit.setEnabled(is_dbscan)
        self._lbl_eps.setEnabled(is_dbscan)
        self._lbl_noise.setEnabled(is_dbscan)
        self._noise_slider.setEnabled(is_dbscan)

    def _build_progress_row(self) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)   
        self._progress_bar.setVisible(False)
        self._progress_bar.setFixedHeight(6)
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("font-size: 11px; color: #555;")
        row.addWidget(self._progress_bar, stretch=1)
        row.addWidget(self._status_lbl)
        return w

    def _build_compare_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        hint_layout = QHBoxLayout()
        hint = QLabel(
            "<b>① Compare Tab:</b> Run all selected algorithms with the same settings to compare internal separation scores."
        )
        hint.setStyleSheet("font-size: 12px; color: #444;")
        
        info_metrics = ("<b>Silhouette Score (-1 to 1):</b> How similar an object is to its own cluster compared to others. Higher is better.<br><br>"
                        "<b>Calinski-Harabasz Score:</b> Ratio of between-cluster dispersion to within-cluster dispersion. Higher is better.<br><br>"
                        "<b>Davies-Bouldin Score:</b> Average similarity between clusters. Lower means clusters are better separated.")
                        
        hint_layout.addWidget(hint)
        hint_layout.addWidget(self._info_btn(info_metrics, "Evaluation Metrics"))
        hint_layout.addStretch()
        layout.addLayout(hint_layout)
        
        # Checkboxes for comparison
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Compare:"))
        self._model_checks: dict[str, QCheckBox] = {}
        for name in ["K-Means", "Mini-Batch K-Means", "Agglomerative (Ward)",
                     "Gaussian Mixture Model (GMM)", "Spectral Clustering", "DBSCAN"]:
            cb = QCheckBox(name)
            cb.setChecked(name not in ("DBSCAN", "Spectral Clustering"))
            self._model_checks[name] = cb
            r3.addWidget(cb)

        r3.addSpacing(20)                                                    
        self._subset_check = QCheckBox("Use subset:")                        
        self._subset_check.setChecked(True)                                  
        r3.addWidget(self._subset_check)

        self._subset_size = QLineEdit("1000")                                
        self._subset_size.setFixedWidth(50)                                  
        r3.addWidget(self._subset_size)
        r3.addWidget(QLabel("rows"))                                         

        r3.addStretch() 
        self._compare_btn = QPushButton("Run Comparison")
        self._compare_btn.clicked.connect(self._start_compare)
        r3.addWidget(self._compare_btn)
        
        layout.addLayout(r3)

        self._compare_output = QTextEdit()
        self._compare_output.setReadOnly(True)
        self._compare_output.setFont(self._mono_font())
        layout.addWidget(self._compare_output)
        return w

    def _build_tune_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        ctrls = QHBoxLayout()
        hint = QLabel("<b>② Tune Tab:</b> Sweep values of 'k' to find the 'elbow' or highest silhouette score.")
        hint.setStyleSheet("font-size: 12px; color: #444;")
        ctrls.addWidget(hint)
        
        info_tune = ("<b>Tuning:</b><br><br>"
                     "• <b>Elbow Plot:</b> Look for the 'k' where the inertia (error) stops dropping drastically and forms an elbow.<br>"
                     "• <b>Silhouette Plot:</b> Pick the 'k' that produces the highest peak.")
        ctrls.addWidget(self._info_btn(info_tune, "Tuning Guide"))
        ctrls.addStretch()

        ctrls.addWidget(QLabel("k range:"))
        self._k_min = QLineEdit("2");  self._k_min.setFixedWidth(40)
        self._k_max = QLineEdit("12"); self._k_max.setFixedWidth(40)
        ctrls.addWidget(self._k_min)
        ctrls.addWidget(QLabel("–"))
        ctrls.addWidget(self._k_max)

        ctrls.addSpacing(16)
        ctrls.addWidget(QLabel("ε range (DBSCAN):"))
        self._eps_min = QLineEdit("0.1"); self._eps_min.setFixedWidth(40)
        self._eps_max = QLineEdit("2.0"); self._eps_max.setFixedWidth(40)
        self._eps_steps = QLineEdit("20"); self._eps_steps.setFixedWidth(40)
        ctrls.addWidget(self._eps_min)
        ctrls.addWidget(QLabel("–"))
        ctrls.addWidget(self._eps_max)
        ctrls.addWidget(QLabel("steps:"))
        ctrls.addWidget(self._eps_steps)
        
        self._tune_btn = QPushButton("Run Tuning")
        self._tune_btn.clicked.connect(self._start_tune)
        ctrls.addWidget(self._tune_btn)
        
        layout.addLayout(ctrls)

        self._fig_tune = Figure(figsize=(10, 4))
        self._canvas_tune = FigureCanvasQTAgg(self._fig_tune)
        layout.addWidget(self._canvas_tune, stretch=1)
        return w

    def _build_results_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        
        header = QHBoxLayout()
        hint = QLabel("<b>③ Results Tab:</b> Deep dive into the physical properties of the generated clusters.")
        hint.setStyleSheet("font-size: 12px; color: #444;")
        header.addWidget(hint)
        
        info_res = ("<b>Results Summary:</b><br><br>"
                    "Scroll down past the individual cluster details to view the <b>Global Summary Plots</b>, "
                    "which overlay all diurnal and seasonal cycles so you can compare their temporal behaviours directly.")
        header.addWidget(self._info_btn(info_res, "Results Guide"))
        header.addStretch()
        layout.addLayout(header)
        
        self.results_warning = QLabel("")
        self.results_warning.setStyleSheet("font-size: 12px; font-weight: bold; color: #a32d2d;")
        layout.addWidget(self.results_warning)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.Shape.NoFrame)

        inner = QWidget()
        self._results_layout = QVBoxLayout(inner)
        self._results_layout.setSpacing(12)
        scroll.setWidget(inner)
        layout.addWidget(scroll)
        return w

    def load_data(self, data_file):
        self._df    = data_file.df
        self._diams = np.array(data_file.diameters)
        self._set_status(f"Data loaded: {len(self._df):,} rows × {len(self._diams)} bins.")

    def _prepare_X(self) -> tuple[np.ndarray, pd.Index]:
        if self._df is None:
            raise ValueError("No data loaded.")
        mode = self._mode_combo.currentText()
        if mode.startswith("Hourly"):
            X = self._df.fillna(0).values.astype(float)
            return X, self._df.index
            
        df = self._df.copy()
        df.index = pd.to_datetime(df.index)
        df = df.fillna(0)

        rows, dates = [], []
        for date, day_df in df.groupby(df.index.date):
            hourly = day_df.groupby(day_df.index.hour).mean()
            # FIX: Ensure 24 hour grid and smoothly interpolate missing hours
            hourly = hourly.reindex(range(24)).interpolate(limit_direction='both').fillna(0)
            rows.append(hourly.values.flatten())
            dates.append(pd.Timestamp(date))

        X = np.array(rows, dtype=float)
        index = pd.DatetimeIndex(dates)
        return X, index
        
    def _plot_comparator_pnsd(self, ax, X_raw, labels, title, scores, k_clusters):
        """Helper to plot cluster centroids for comparator tools."""
        ax.set_title(f"{title}\nSilh: {scores['silhouette']:.3f} | DB: {scores['davies']:.2f}", fontsize=10)
        for c in range(k_clusters):
            mask = labels == c
            if not mask.any(): continue
            mean_pnsd = X_raw[mask].mean(axis=0)
            if self._mode_combo.currentText().startswith("Daily"):
                mean_pnsd = mean_pnsd.reshape(24, len(self._diams)).mean(axis=0)
            ax.plot(self._diams, mean_pnsd, label=f"C{c+1}", color=_ACCENT_COLORS[c % len(_ACCENT_COLORS)])
        ax.set_xscale('log')
        ax.set_xlabel("Diameter (nm)")

    def _show_scale_comparison(self):
        """Compares different standardisation tools."""
        if self._df is None: 
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
            
        X_raw, _ = self._prepare_X()
        dim_method = self._dim_combo.currentText()
        n_dims = self._int("_pca_edit", 3)
        k_clusters = self._int("_k_edit", 4)
        
        self._set_status("Calculating Scale Comparisons...")
        QApplication.processEvents() 
        
        scalers = ["Standard (z-score)", "MinMax [0,1]", "Robust (median/IQR)", "Log + Standard", "None"]
        
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Scaler Quality -> {k_clusters} Clusters (using {dim_method})")
        layout = QVBoxLayout(dlg)
        
        fig = Figure(figsize=(14, 6))
        canvas = FigureCanvasQTAgg(fig)
        axes = fig.subplots(2, 3, sharey=True).flatten()
        
        for i, scaler in enumerate(scalers):
            X_scaled = _scale(X_raw, scaler)
            X_dim = _dim_reduce(X_scaled, dim_method, n_dims)
            labels = _fit_model("K-Means", X_dim, k=k_clusters)
            s = _score(X_dim, labels)
            self._plot_comparator_pnsd(axes[i], X_raw, labels, scaler, s, k_clusters)
            
        axes[5].set_visible(False) 
        axes[0].set_ylabel(r"dN/dlogD$_p$ (cm$^{-3}$)")
        axes[3].set_ylabel(r"dN/dlogD$_p$ (cm$^{-3}$)")
        axes[0].legend(fontsize=8)
        
        fig.tight_layout()
        layout.addWidget(canvas)
        dlg.resize(1200, 600)
        
        self._set_status("Ready.")
        dlg.exec()

    def _auto_find_dim(self):
        """Sweeps PCA dims from 2 to 10 to find best Silhouette score."""
        if self._df is None: return self._set_status("Load data first.")
        X_raw, _ = self._prepare_X()
        scaler_method = self._scaler_combo.currentText()
        X_scaled = _scale(X_raw, scaler_method)
        
        best_score = -1
        best_dim = 2
        
        self._set_status("Auto-finding best dimensionality...")
        QApplication.processEvents()
        
        for n in range(2, 11):
            X_pca = _dim_reduce(X_scaled, "PCA", n)
            labels = _fit_model("K-Means", X_pca, k=4)
            s = _score(X_pca, labels)
            if not np.isnan(s["silhouette"]) and s["silhouette"] > best_score:
                best_score = s["silhouette"]
                best_dim = n
                
        self._pca_edit.setText(str(best_dim))
        self._dim_combo.setCurrentText("PCA")
        self._set_status(f"Auto-Find Complete: Selected {best_dim} dimensions (Score: {best_score:.3f}).")

    def _show_dim_comparison(self):
        """Grid sweep comparing PCA and Autoencoder across dims 2,3,4,5."""
        if self._df is None: 
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
            
        X_raw, _ = self._prepare_X()
        scaler_method = self._scaler_combo.currentText()
        X_scaled = _scale(X_raw, scaler_method)
        k_clusters = self._int("_k_edit", 4)
        
        dims_to_test = [2, 3, 4, 5]
        
        self._set_status("Calculating Dim-Red Sweep (This will take a moment)...")
        QApplication.processEvents() 
        
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Dim-Red Sweep: PCA vs Autoencoder -> {k_clusters} Clusters")
        layout = QVBoxLayout(dlg)
        
        fig = Figure(figsize=(16, 8))
        canvas = FigureCanvasQTAgg(fig)
        axes = fig.subplots(2, 4, sharey=True, sharex=True)
        
        for i, d in enumerate(dims_to_test):
            # PCA Row (Top)
            X_pca = _dim_reduce(X_scaled, "PCA", d)
            labels_pca = _fit_model("K-Means", X_pca, k=k_clusters)
            s_pca = _score(X_pca, labels_pca)
            self._plot_comparator_pnsd(axes[0, i], X_raw, labels_pca, f"PCA ({d} Dims)", s_pca, k_clusters)
            
            # AE Row (Bottom)
            try:
                X_ae = _dim_reduce(X_scaled, "Autoencoder (PyTorch)", d)
                labels_ae = _fit_model("K-Means", X_ae, k=k_clusters)
                s_ae = _score(X_ae, labels_ae)
                self._plot_comparator_pnsd(axes[1, i], X_raw, labels_ae, f"Autoencoder ({d} Dims)", s_ae, k_clusters)
            except Exception:
                axes[1, i].set_title("AE Failed/Missing", fontsize=10)

        axes[0, 0].set_ylabel(r"dN/dlogD$_p$ (cm$^{-3}$)")
        axes[1, 0].set_ylabel(r"dN/dlogD$_p$ (cm$^{-3}$)")
        axes[0, 0].legend(fontsize=8)
        
        fig.tight_layout()
        layout.addWidget(canvas)
        dlg.resize(1400, 700)
        
        info = QLabel("<b>Goal:</b> Find the sweet spot where clusters are highly physically distinct. If lines overlap heavily, the clustering failed to find separate physical states.")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        self._set_status("Ready.")
        dlg.exec()

    def _start_compare(self):
        if self._df is None: return self._set_status("Load data first.")
        models = [n for n, cb in self._model_checks.items() if cb.isChecked()]
        if not models: return self._set_status("Select at least one model.")

        X, _ = self._prepare_X()
        if self._subset_check.isChecked():                                   
            n_samples = self._int("_subset_size", 1000)                      
            if n_samples < len(X):                                           
                rng = np.random.default_rng(42)                              
                idx = rng.choice(len(X), n_samples, replace=False)           
                X = X[idx]                                                   
                self._set_status(f"Comparing models on {n_samples} rows...") 
        
        self._launch_worker("compare", dict(
            X_raw=X, k=self._int("_k_edit", 5),
            scaler=self._scaler_combo.currentText(),
            dim_method=self._dim_combo.currentText(),
            dim_n=self._int("_pca_edit", 0),
            models=models,
        ))

    def _start_tune(self):
        if self._df is None: return self._set_status("Load data first.")
        X, _ = self._prepare_X()
        self._launch_worker("tune", dict(
            X_raw=X,
            scaler=self._scaler_combo.currentText(),
            dim_method=self._dim_combo.currentText(),
            dim_n=self._int("_pca_edit", 0),
            model=self._model_combo.currentText(),
            k_min=self._int("_k_min", 2),
            k_max=self._int("_k_max", 12),
            eps_min=self._float("_eps_min", 0.1),
            eps_max=self._float("_eps_max", 2.0),
            eps_steps=self._int("_eps_steps", 20),
        ))

    def _start_cluster(self):
        if self._df is None: return self._set_status("Load data first.")
        X, index = self._prepare_X()
        self._launch_worker("cluster", dict(
            X_raw=X, index=index, diams=self._diams,
            scaler=self._scaler_combo.currentText(),
            dim_method=self._dim_combo.currentText(),
            dim_n=self._int("_pca_edit", 0),
            model=self._model_combo.currentText(),
            k=self._int("_k_edit", 5),
            eps=self._float("_eps_edit", 0.5),
            min_samples=self._int("_noise_slider", 5), 
            mode=self._mode_combo.currentText(),
            df_hourly=self._df,
        ))

    def _launch_worker(self, task: str, kwargs: dict):
        self._progress_bar.setVisible(True)
        self._set_status(f"Running {task}…")
        self._worker = ClusterWorker(task, kwargs)
        self._worker.finished.connect(self._on_worker_done)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()

    def _on_worker_done(self, result: dict):
        self._progress_bar.setVisible(False)
        t = result.get("type")

        if t == "compare":
            self._show_compare(result)
            self._workspace.setCurrentIndex(0)
        elif t in ("tune_k", "tune_dbscan"):
            self._show_tune(result)
            self._workspace.setCurrentIndex(1)
            self._active_figures = [self._fig_tune] 
        elif t == "cluster":
            self._results = result
            self._show_cluster_results(result)
            self._export_btn.setEnabled(True)
            self._workspace.setCurrentIndex(2)

        self._set_status("Done.")

    def _on_worker_error(self, msg: str):
        self._progress_bar.setVisible(False)
        if "PyTorch is required" in msg:
            QMessageBox.critical(self, "Dependency Error", msg)
        self._set_status(f"Error: {msg}")

    # ────────────────────────────────────────────────────────────────────── #
    # Display Methods                                                        #
    # ────────────────────────────────────────────────────────────────────── #

    def _show_compare(self, result: dict):
        rows = result["rows"]
        if not rows:
            self._compare_output.setText("No results.")
            return

        cols = list(rows[0].keys())
        widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}

        header = "  ".join(c.ljust(widths[c]) for c in cols)
        sep    = "  ".join("─" * widths[c] for c in cols)
        lines  = [header, sep]
        for row in rows:
            lines.append("  ".join(str(row[c]).ljust(widths[c]) for c in cols))

        self._compare_output.setText("\n".join(lines))

    def _show_tune(self, result: dict):
        self._fig_tune.clear()

        if result["type"] == "tune_dbscan":
            ax1 = self._fig_tune.add_subplot(121)
            ax2 = self._fig_tune.add_subplot(122)
            eps = result["eps"]
            ax1.plot(eps, result["silhouettes"], marker='o', color=_ACCENT_COLORS[0])
            ax1.set_xlabel("ε"); ax1.set_ylabel("Silhouette score ↑")
            ax1.set_title("Silhouette vs ε (DBSCAN)")

            ax2.plot(eps, result["n_clusters"], marker='o', color=_ACCENT_COLORS[1])
            ax2.set_xlabel("ε"); ax2.set_ylabel("Number of clusters")
            ax2.set_title("Clusters vs ε (DBSCAN)")

        else:
            ks = result["ks"]
            has_inertia = not all(np.isnan(result["inertias"]))
            n_plots = 4 if has_inertia else 3
            dendro_data = result.get("dendro")
            if dendro_data is not None: n_plots += 1

            axes = self._fig_tune.subplots(1, n_plots)
            i = 0

            if has_inertia:
                axes[i].plot(ks, result["inertias"], marker='o', color=_ACCENT_COLORS[0])
                axes[i].set_xlabel("k"); axes[i].set_ylabel("Inertia ↓")
                axes[i].set_title("Elbow")
                axes[i].set_xticks(ks); i += 1

            axes[i].plot(ks, result["silhouettes"], marker='o', color=_ACCENT_COLORS[1])
            axes[i].set_xlabel("k"); axes[i].set_ylabel("Score ↑")
            axes[i].set_title("Silhouette")
            axes[i].set_xticks(ks); i += 1

            axes[i].plot(ks, result["calinskis"], marker='o', color=_ACCENT_COLORS[2])
            axes[i].set_xlabel("k"); axes[i].set_ylabel("Score ↑")
            axes[i].set_title("Calinski-Harabasz")
            axes[i].set_xticks(ks); i += 1

            axes[i].plot(ks, result["davies"], marker='o', color=_ACCENT_COLORS[3])
            axes[i].set_xlabel("k"); axes[i].set_ylabel("Score ↓")
            axes[i].set_title("Davies-Bouldin")
            axes[i].set_xticks(ks); i += 1

            if dendro_data is not None and i < len(axes):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dendrogram(dendro_data, ax=axes[i], no_labels=True,
                               color_threshold=0, above_threshold_color='grey')
                axes[i].set_title("Dendrogram (sample)")
                axes[i].set_xlabel("Samples"); axes[i].set_ylabel("Distance")

        self._fig_tune.tight_layout(pad=2.0)
        self._canvas_tune.draw()

    def _show_cluster_results(self, result: dict):
        while self._results_layout.count():                                  
            item = self._results_layout.takeAt(0)                            
            if item.widget(): item.widget().deleteLater()                    

        self._active_figures = [] 
        self.results_warning.setText("")

        labels = result["labels"]                                            
        index = result["index"]                                              
        diams = result["diams"]                                              
        df_hourly = result["df_hourly"]                                      
        mode = result["mode"]                                                
        X_raw = result.get("X_raw")                                          
        
        cluster_counts = pd.Series(labels).value_counts()
        unique_clusters = cluster_counts.index.tolist()
        
        valid_clusters = [c for c in unique_clusters if c != -1] 
        top_clusters = valid_clusters[:5] 
        
        if len(valid_clusters) > 5:
            self.results_warning.setText(f"Warning: Model generated {len(valid_clusters)} clusters! Showing full plots for the top 5 largest. Minor clusters omitted for UI stability.")
        
        log_d = np.log10(diams)                                              
        dlogdp = np.mean(np.diff(log_d)) if len(log_d) > 1 else 0.1          

        # Data collection for summary plots
        sum_diurnals = {}
        sum_ts = {}
        sum_seasonals = {}
        sum_freq = {}
        sum_N = {}
        sum_contours = {}

        # --- 1. Global PCA Separation Plot ---
        if X_raw is not None and len(valid_clusters) > 1:
            sep_fig = Figure(figsize=(10, 6))                                
            self._active_figures.append(sep_fig) 
            
            sep_canvas = FigureCanvasQTAgg(sep_fig)                          
            sep_canvas.setMinimumHeight(500)                                 
            ax_sep = sep_fig.add_subplot(111)                                
            
            pca = PCA(n_components=2)                                        
            X_scaled = _scale(X_raw, result['scaler'])                       
            X_pca = pca.fit_transform(X_scaled)                              
            
            minor_mask = (labels == -1) | (~np.isin(labels, top_clusters))
            if minor_mask.any():
                ax_sep.scatter(X_pca[minor_mask, 0], X_pca[minor_mask, 1],               
                               label="Noise / Minor Clusters", color='grey', 
                               alpha=0.3, s=15, edgecolors='none')
            
            for cl in top_clusters:
                mask = labels == cl                                          
                color = _ACCENT_COLORS[cl % len(_ACCENT_COLORS)]             
                ax_sep.scatter(X_pca[mask, 0], X_pca[mask, 1],               
                               label=f"Cluster {cl+1} (n={cluster_counts[cl]})", color=color, 
                               alpha=0.8, s=25, edgecolors='white', lw=0.5)
            
            ax_sep.set_title("Cluster Separation (PCA)", fontweight='bold')   
            ax_sep.set_xlabel("PC 1"); ax_sep.set_ylabel("PC 2")             
            ax_sep.legend(loc='upper right', fontsize=9)                     
            sep_fig.tight_layout(pad=3.0)                                    
            self._results_layout.addWidget(sep_canvas)                       
            self._results_layout.addSpacing(40)                              

        # --- 2. Per-Cluster Detail Plots ---
        try: v_min = float(self._cbar_min.text())                            
        except ValueError: v_min = 1.0                                       
        v_max_user = float(self._cbar_max.text()) if self._cbar_max.text() else None

        for cl in top_clusters:
            mask = labels == cl                                              
            cl_index = index[mask]                                           
            n_members = len(cl_index)                                        
            pct = 100 * n_members / len(index)                               
            color = _ACCENT_COLORS[cl % len(_ACCENT_COLORS)]                 

            if mode.startswith("Daily"):
                X_subset = X_raw[mask]                                       
                mean_row = X_subset.mean(axis=0)                             
                pnsd_diurnal = mean_row.reshape(24, len(diams))              
                diurnal_idx = np.arange(24)                                  
                
                ts_n_vals = (X_subset.sum(axis=1) * dlogdp) / 24             
                ts_n = pd.Series(ts_n_vals, index=cl_index)                  
            else:
                cl_hourly = df_hourly.loc[df_hourly.index.isin(cl_index)]    
                diurnal_mean = cl_hourly.groupby(cl_hourly.index.hour).mean() 
                # FIX: Ensure perfect 24h grid for hourly data to prevent pcolormesh crashes!
                diurnal_mean = diurnal_mean.reindex(range(24)).interpolate(limit_direction='both').fillna(1e-4)
                pnsd_diurnal = diurnal_mean.values                           
                diurnal_idx = np.arange(24)                             
                ts_n = cl_hourly.sum(axis=1) * dlogdp                        

            # Store for summary
            total_n_diurnal = pnsd_diurnal.sum(axis=1) * dlogdp
            moy_counts = pd.Series(cl_index).dt.month.value_counts().reindex(range(1, 13), fill_value=0)
            
            sum_diurnals[cl] = (diurnal_idx, total_n_diurnal)
            sum_ts[cl] = (ts_n.index, ts_n.values)
            sum_seasonals[cl] = (["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], moy_counts.values)
            sum_freq[cl] = n_members
            sum_N[cl] = ts_n.sum()
            sum_contours[cl] = pnsd_diurnal

            # Build Figure
            fig = Figure(figsize=(12, 12))                                   
            self._active_figures.append(fig) 
            
            canvas = FigureCanvasQTAgg(fig)                                  
            canvas.setMinimumHeight(900)                                     
            
            gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)              
            ax_cont  = fig.add_subplot(gs[0, 0])                             
            ax_diurn = fig.add_subplot(gs[0, 1])                             
            ax_ts    = fig.add_subplot(gs[1, :])                             
            ax_dow   = fig.add_subplot(gs[2, 0])                             
            ax_moy   = fig.add_subplot(gs[2, 1])                             

            pnsd_safe = np.clip(pnsd_diurnal, 1e-4, None)                    
            v_max = v_max_user if v_max_user else max(pnsd_safe.max(), v_min*10)
            
            mesh = ax_cont.pcolormesh(diurnal_idx, diams, pnsd_safe.T,       
                                      cmap=self._cmap_combo.currentText(), 
                                      norm=LogNorm(vmin=v_min, vmax=v_max), shading='auto')
            ax_cont.set_xlim(0, 23)
            ax_cont.set_yscale('log')                                        
            ax_cont.set_title("Mean Diurnal PNSD")                           
            ax_cont.set_ylabel("Dp (nm)")                                    
            fig.colorbar(mesh, ax=ax_cont, label=r"dN/dlogD$_p$ (cm$^{-3}$)")

            ax_diurn.plot(diurnal_idx, total_n_diurnal, color=color, lw=2.5) 
            ax_diurn.fill_between(diurnal_idx, 0, total_n_diurnal, alpha=0.15, color=color)
            ax_diurn.set_xlim(0, 23)
            ax_diurn.set_ylabel(r"Total N (cm$^{-3}$)")                      
            ax_diurn.set_title("Mean Diurnal Profile")                       

            ax_ts.plot(ts_n.index, ts_n.values, color=color, lw=0.7, alpha=0.5) 
            ax_ts.set_title(f"Cluster Time Series ({'Daily' if mode.startswith('Daily') else 'Hourly'})")
            ax_ts.set_ylabel(r"Total N (cm$^{-3}$)")                         

            dow_counts = pd.Series(cl_index).dt.dayofweek.value_counts().reindex(range(7), fill_value=0)
            dow_labs = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]     
            ax_dow.plot(dow_labs, dow_counts.values, marker='o', color=color, lw=2)
            ax_dow.fill_between(dow_labs, 0, dow_counts.values, alpha=0.1, color=color)
            ax_dow.set_title("Weekly Breakdown")                             

            ax_moy.plot(sum_seasonals[cl][0], sum_seasonals[cl][1], marker='o', color=color, lw=2)
            ax_moy.fill_between(sum_seasonals[cl][0], 0, sum_seasonals[cl][1], alpha=0.1, color=color)
            ax_moy.set_title("Seasonal Breakdown")                           

            fig.suptitle(f"Cluster {cl+1} ({pct:.1f}% total data)", fontsize=15, fontweight='bold')
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])                        
            self._results_layout.addWidget(canvas)                           
            self._results_layout.addSpacing(60)                              


        # --- 3. Global Summary Plots ---
        self._results_layout.addWidget(QLabel("<h2>Global Summary</h2>"))
        
        # Summary 1: Contours
        n_plots = len(top_clusters)
        cols = 2
        rows = math.ceil(n_plots / cols)
        
        fig_sum_cont = Figure(figsize=(10, 3 * rows + 1))
        self._active_figures.append(fig_sum_cont)
        canvas_sum_cont = FigureCanvasQTAgg(fig_sum_cont)
        canvas_sum_cont.setMinimumHeight(max(400, 250 * rows + 100))
        
        axes = fig_sum_cont.subplots(rows, cols, sharex=True, sharey=True)
        if n_plots == 1: axes = np.array([axes])
        axes = axes.flatten()
        
        im = None
        for i, ax in enumerate(axes):
            if i >= n_plots:
                ax.set_visible(False)
                continue
            cl = top_clusters[i]
            pnsd_safe = np.clip(sum_contours[cl], 1e-4, None)
            im = ax.pcolormesh(diurnal_idx, diams, pnsd_safe.T, cmap=self._cmap_combo.currentText(), norm=LogNorm(vmin=v_min, vmax=v_max), shading='auto')
            ax.set_yscale('log')
            ax.set_xlim(0, 23)
            ax.set_title(f"Cluster {cl+1}")
            if i % cols == 0: ax.set_ylabel("Dp (nm)")
            if i >= (rows - 1) * cols: ax.set_xlabel("Hour")
            
        if im:
            cbar = fig_sum_cont.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', pad=0.15, fraction=0.05, aspect=40, shrink=0.7)
            cbar.set_label(r"dN/dlogD$_p$ (cm$^{-3}$)")
            
        fig_sum_cont.suptitle("All Cluster Contours", fontweight='bold')
        fig_sum_cont.subplots_adjust(hspace=0.3, wspace=0.1, bottom=0.2)
        self._results_layout.addWidget(canvas_sum_cont)
        self._results_layout.addSpacing(20)
        
        # Summary 2: Line Overlays
        fig_sum_lines = Figure(figsize=(12, 10))
        self._active_figures.append(fig_sum_lines)
        canvas_sum_lines = FigureCanvasQTAgg(fig_sum_lines)
        canvas_sum_lines.setMinimumHeight(600)
        gs2 = fig_sum_lines.add_gridspec(2, 2)
        
        ax_s_diur = fig_sum_lines.add_subplot(gs2[0, 0])
        ax_s_seas = fig_sum_lines.add_subplot(gs2[0, 1])
        ax_s_ts = fig_sum_lines.add_subplot(gs2[1, :])
        
        for cl in top_clusters:
            color = _ACCENT_COLORS[cl % len(_ACCENT_COLORS)]
            lbl = f"C{cl+1}"
            
            x, y = sum_diurnals[cl]
            ax_s_diur.plot(x, y, color=color, lw=2, label=lbl)
            
            x, y = sum_seasonals[cl]
            ax_s_seas.plot(x, y, marker='o', color=color, lw=2, label=lbl)
            
            x, y = sum_ts[cl]
            ax_s_ts.plot(x, y, color=color, lw=0.7, alpha=0.7, label=lbl)
            
        ax_s_diur.set_title("Overlay: Diurnals"); ax_s_diur.set_ylabel("Total N"); ax_s_diur.set_xlim(0, 23)
        ax_s_seas.set_title("Overlay: Seasonals"); ax_s_seas.set_ylabel("Frequency")
        ax_s_ts.set_title("Overlay: Time Series"); ax_s_ts.set_ylabel("Total N")
        ax_s_diur.legend(); ax_s_seas.legend(); ax_s_ts.legend(loc='upper right')
        
        fig_sum_lines.tight_layout()
        self._results_layout.addWidget(canvas_sum_lines)
        self._results_layout.addSpacing(20)
        
        # Summary 3: Bar and Pie Stats
        fig_sum_stats = Figure(figsize=(10, 4))
        self._active_figures.append(fig_sum_stats)
        canvas_sum_stats = FigureCanvasQTAgg(fig_sum_stats)
        canvas_sum_stats.setMinimumHeight(300)
        
        ax_bar = fig_sum_stats.add_subplot(121)
        ax_pie = fig_sum_stats.add_subplot(122)
        
        labels = [f"C{cl+1}" for cl in top_clusters]
        colors = [_ACCENT_COLORS[cl % len(_ACCENT_COLORS)] for cl in top_clusters]
        freqs = [sum_freq[cl] for cl in top_clusters]
        Ns = [sum_N[cl] for cl in top_clusters]
        
        ax_bar.bar(labels, freqs, color=colors, alpha=0.8)
        ax_bar.set_title("Cluster Frequency (Occurrences)")
        ax_bar.set_ylabel("Count")
        
        # FIX: Added scientific notation counts to pie chart
        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = pct * total / 100.0
                return f'{pct:.1f}%\n({val:.1e})'
            return my_autopct
            
        ax_pie.pie(Ns, labels=labels, colors=colors, autopct=make_autopct(Ns), startangle=90)
        ax_pie.set_title("Contribution to Total Particle Number")
        
        fig_sum_stats.tight_layout()
        self._results_layout.addWidget(canvas_sum_stats)

        self._results_layout.addStretch()                                    

    # ────────────────────────────────────────────────────────────────────── #
    # Export Methods                                                         #
    # ────────────────────────────────────────────────────────────────────── #
    
    def _export_active_plots(self):
        tab_idx = self._workspace.currentIndex()
        
        if tab_idx == 0:
            QMessageBox.information(self, "No Plots", "The Compare tab only contains text output. Nothing to export here!")
            return
            
        if not self._active_figures:
            QMessageBox.information(self, "No Plots", "No plots have been generated yet!")
            return
            
        for i, fig in enumerate(self._active_figures):
            canvas = fig.canvas
            parent_layout = canvas.parent().layout() if canvas.parent() else None
            
            def make_restore_callback(c=canvas, l=parent_layout, orig_size=fig.get_size_inches()):
                def restore():
                    if l: l.addWidget(c)
                    c.figure.set_size_inches(orig_size)
                    c.draw()
                return restore

            dlg = ExportDialog(f"Plot {i+1} of {len(self._active_figures)}", canvas, fig, make_restore_callback())
            dlg.exec()

    def _export_csv(self):
        if self._results is None: return

        labels = self._results["labels"]
        index  = self._results["index"]
        mode   = self._results["mode"]

        col_name = "cluster_day" if mode.startswith("Daily") else "cluster_hour"
        export_labels = np.where(labels == -1, -1, labels + 1)

        out = pd.DataFrame({col_name: export_labels}, index=index)
        out.index.name = "datetime"

        path, _ = QFileDialog.getSaveFileName(self, "Save cluster assignments", "", "CSV files (*.csv)")
        if not path: return
        if not path.endswith(".csv"): path += ".csv"

        out.to_csv(path)
        self._set_status(f"Saved → {path}")

    def _int(self, attr: str, default: int) -> int:
        try: return int(getattr(self, attr).text())
        except ValueError: return default

    def _float(self, attr: str, default: float) -> float:
        try: return float(getattr(self, attr).text())
        except ValueError: return default

    def _set_status(self, msg: str):
        self._status_lbl.setText(msg)

    @staticmethod
    def _mono_font():
        from PyQt6.QtGui import QFont
        f = QFont("Courier New", 9)
        return f
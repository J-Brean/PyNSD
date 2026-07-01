"""
gui/main_window.py
------------------
Root application window.  Holds the left navigation, the stacked panels and the
shared data state.
"""

from PyQt6.QtWidgets import (QMainWindow, QStackedWidget, QListWidget,
                             QListWidgetItem, QWidget, QVBoxLayout, QLabel,
                             QPushButton, QHBoxLayout, QGraphicsOpacityEffect)
from PyQt6.QtCore import Qt, QUrl, QPropertyAnimation, QParallelAnimationGroup
from PyQt6.QtMultimedia import QSoundEffect
import pandas as pd

from gui.load_panel import LoadPanel
from gui.summary_panel import SummaryPanel
from gui.trend_panel import TrendPanel
from gui.npf_panel_manual import NPFPanel
from gui.npf_panel_deeplearning import NPFDeepLearningPanel
from gui.cluster_panel import ClusterPanel
from gui.nano_ranking_panel import NanoRankingPanel
from gui.wind_panel import WindPanel
from gui.pmf_panel import PMFPanel
from gui.pollution_flag_panel import PollutionFlagPanel
from utils.data_loader import DataFile


class LandingPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self._has_played = False                                       # guard: play sound/fade once
        layout = QVBoxLayout(self)
        layout.setContentsMargins(50, 50, 50, 50)

        # --- 1. TITLE ---
        self.title = QLabel("🍌 PyNSD 🍌\nThe PNSD Toolkit")
        self.title.setObjectName("LandingTitle")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- 2. SUBTITLE ---
        self.subtitle = QLabel("James Brean, University of Birmingham")
        self.subtitle.setObjectName("LandingSubtitle")
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- 3. DESCRIPTION ---
        self.description = QLabel(
            "An all-in-one toolbox to analyse PNSDs, identify NPF events,\n"
            "cluster data, and explore trends."
        )
        self.description.setObjectName("LandingDescription")
        self.description.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- 4. START BUTTON ---
        self.start_btn = QPushButton("Get started!")
        self.start_btn.setProperty("class", "primary")
        self.start_btn.setMinimumSize(300, 64)
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.clicked.connect(self._go_to_load_tab)

        # --- LAYOUT CONSTRUCTION ---
        layout.addStretch(2)
        layout.addWidget(self.title)
        layout.addWidget(self.subtitle)
        layout.addStretch(1)
        layout.addWidget(self.description)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch(); btn_layout.addWidget(self.start_btn); btn_layout.addStretch()
        layout.addLayout(btn_layout)
        layout.addStretch(3)

        # --- SOUND & ANIMATION SETUP ---
        self.startup_sound = QSoundEffect(self)
        self.startup_sound.setSource(QUrl.fromLocalFile("startup.wav"))
        self.startup_sound.setVolume(0.5)

        self.anim_group = QParallelAnimationGroup()
        for widget in [self.title, self.subtitle, self.description, self.start_btn]:
            self._setup_fade(widget)

    def _setup_fade(self, widget):
        """Prepares a widget to be invisible and attaches a fade-in animation."""
        eff = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(eff)

        anim = QPropertyAnimation(eff, b"opacity")
        anim.setDuration(2000)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        self.anim_group.addAnimation(anim)

    def showEvent(self, event):
        """Play the startup sound and fade-in exactly once, on first appearance."""
        super().showEvent(event)
        if self._has_played:
            return
        self._has_played = True
        self.startup_sound.play()
        self.anim_group.start()

    def _go_to_load_tab(self):
        if self.main_window:
            self.main_window.show_load()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyNSD - The PNSD Toolkit")
        self.resize(1600, 1000)

        self.loaded_data = {}

        # --- Panels (stacked) ---
        self.landing_panel = LandingPanel(self)
        self.load_panel = LoadPanel(self)
        self.summary_panel = SummaryPanel(self)
        self.trend_panel = TrendPanel(self)
        self.npf_panel_manual = NPFPanel(self)
        self.npf_dl_panel = NPFDeepLearningPanel(self)
        self.cluster_panel = ClusterPanel(self)
        self.nano_ranking_panel = NanoRankingPanel(self)
        self.wind_panel = WindPanel(self)
        self.pmf_panel = PMFPanel(self)
        self.pollution_flag_panel = PollutionFlagPanel(self)

        self.load_panel.data_confirmed.connect(self._on_data_confirmed)

        self.stack = QStackedWidget()
        for panel in (self.landing_panel, self.load_panel, self.summary_panel,
                      self.trend_panel, self.npf_panel_manual, self.npf_dl_panel,
                      self.cluster_panel, self.nano_ranking_panel, self.wind_panel,
                      self.pmf_panel, self.pollution_flag_panel):
            self.stack.addWidget(panel)

        # --- Left navigation ---
        self.nav = QListWidget()
        self.nav.setObjectName("NavList")
        self.nav.setMaximumWidth(240)
        self.nav.setMinimumWidth(200)
        self.nav.currentItemChanged.connect(self._on_nav_changed)

        # (label, stack index, is_analysis)
        self._analysis_items: list[QListWidgetItem] = []
        self._add_nav_item("Welcome", 0, analysis=False)
        self._add_nav_header("DATA")
        self._add_nav_item("Load data", 1, analysis=False)
        self._add_nav_header("OVERVIEW")
        self._add_nav_item("Summary", 2, analysis=True)
        self._add_nav_item("Trend analysis", 3, analysis=True)
        self._add_nav_header("NPF")
        self._add_nav_item("NPF identifier", 4, analysis=True)
        self._add_nav_item("NPF deep learning", 5, analysis=True)
        self._add_nav_header("ADVANCED")
        self._add_nav_item("Cluster", 6, analysis=True)
        self._add_nav_item("Nano ranking", 7, analysis=True)
        self._add_nav_item("Wind", 8, analysis=True)
        self._add_nav_item("PMF", 9, analysis=True)
        self._add_nav_item("Pollution flags", 10, analysis=True)

        self._set_analysis_enabled(False)

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.nav)
        root.addWidget(self.stack, stretch=1)
        self.setCentralWidget(central)

        self.nav.setCurrentRow(0)                                      # Welcome by default

    # ---- Navigation helpers ------------------------------------------- #
    def _add_nav_header(self, text: str) -> None:
        item = QListWidgetItem(text)
        item.setFlags(Qt.ItemFlag.NoItemFlags)                         # non-selectable header
        font = item.font(); font.setBold(True); font.setPointSize(8)
        item.setFont(font)
        self.nav.addItem(item)

    def _add_nav_item(self, text: str, stack_index: int, analysis: bool) -> None:
        item = QListWidgetItem(text)
        item.setData(Qt.ItemDataRole.UserRole, stack_index)
        self.nav.addItem(item)
        if analysis:
            self._analysis_items.append(item)

    def _set_analysis_enabled(self, enabled: bool) -> None:
        flags = (Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
                 if enabled else Qt.ItemFlag.NoItemFlags)
        for item in self._analysis_items:
            item.setFlags(flags)
            if not enabled:
                tip = "Load and confirm a dataset first to unlock this analysis."
            else:
                tip = ""
            item.setToolTip(tip)

    def _on_nav_changed(self, current: QListWidgetItem, _previous):
        if current is None:
            return
        idx = current.data(Qt.ItemDataRole.UserRole)
        if idx is not None:
            self.stack.setCurrentIndex(idx)

    def _select_nav_for_index(self, stack_index: int) -> None:
        for row in range(self.nav.count()):
            item = self.nav.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == stack_index:
                self.nav.setCurrentItem(item)
                return

    def show_load(self) -> None:
        self._select_nav_for_index(1)

    # ---- Data flow ----------------------------------------------------- #
    def _on_data_confirmed(self, results: dict):
        self.loaded_data = results                                     # Store raw results

        if results:
            combined_df = pd.concat([r.df for r in results.values()]).sort_index()
            first_file = list(results.values())[0]

            merged_data = DataFile(
                path=first_file.path,
                df=combined_df,
                diameters=first_file.diameters,                        # Assumes identical bins
                n_rows=len(combined_df),
                n_bins=first_file.n_bins,
            )

            self.summary_panel.load_data(merged_data)
            self.trend_panel.load_data(merged_data)
            self.npf_panel_manual.load_data(merged_data)
            self.npf_dl_panel.load_data(merged_data)
            self.cluster_panel.load_data(merged_data)
            self.nano_ranking_panel.load_data(merged_data)
            self.wind_panel.load_data(merged_data)
            self.pmf_panel.load_data(merged_data)
            self.pollution_flag_panel.load_data(merged_data)

            self._set_analysis_enabled(True)                           # Unlock analysis sections
            self._select_nav_for_index(2)                              # Jump to Summary

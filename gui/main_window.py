"""
gui/main_window.py
------------------
Root application window.  Holds the tab widget and shared state.
"""

from PyQt6.QtWidgets import (QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
                             QLabel, QPushButton, QHBoxLayout, QGraphicsOpacityEffect) # Added OpacityEffect
from PyQt6.QtCore import Qt, QUrl, QPropertyAnimation, QParallelAnimationGroup          # Added Animation classes
from PyQt6.QtMultimedia import QSoundEffect
import pandas as pd
import os

from gui.load_panel import LoadPanel                                 # Import existing panels
from gui.summary_panel import SummaryPanel                           # Import the updated file
from gui.trend_panel import TrendPanel                               # Import new panel
from gui.npf_panel_manual import NPFPanel # Add to top imports
from gui.npf_panel_deeplearning import NPFDeepLearningPanel
from gui.cluster_panel import ClusterPanel
from gui.nano_ranking_panel import NanoRankingPanel
from gui.wind_panel import WindPanel
from gui.pmf_panel import PMFPanel
                                                 # Required for merging
from utils.data_loader import DataFile                               # Required to create merged object

class LandingPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        layout = QVBoxLayout(self)
        layout.setContentsMargins(50, 50, 50, 50)                                      # Give the edges some breathing room

        # --- 1. TITLE ---
        self.title = QLabel("🍌 PyNSD 🍌\nThe PNSD Toolkit")
        self.title.setStyleSheet("font-size: 82px; font-weight: bold; color: #33302e;") # Massive title
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # --- 2. SUBTITLE ---
        self.subtitle = QLabel("James Brean, University of Birmingham")
        self.subtitle.setStyleSheet("font-size: 28px; color: #4a4a4a; margin-top: 10px;") # Larger subtitle
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # --- 3. DESCRIPTION ---
        self.description = QLabel(
            "An all-in-one toolbox to analyse PNSDs, identify NPF events,\n"
            "cluster data, and explore trends."
        )
        self.description.setStyleSheet("font-size: 22px; margin: 40px 0; line-height: 1.5;") 
        self.description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # --- 4. START BUTTON ---
        self.start_btn = QPushButton("Get started!")
        self.start_btn.setFixedSize(300, 70)                                           # Larger button
        self.start_btn.setStyleSheet("""
            QPushButton { font-size: 24px; font-weight: bold; background-color: #d1c4ba; border-radius: 10px; }
            QPushButton:hover { background-color: #b3a8a0; }
        """)
        self.start_btn.clicked.connect(self._go_to_load_tab)

        # --- LAYOUT CONSTRUCTION ---
        layout.addStretch(2)                                                           # Push content down
        layout.addWidget(self.title)
        layout.addWidget(self.subtitle)
        layout.addStretch(1)                                                           # Space between title/description
        layout.addWidget(self.description)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch(); btn_layout.addWidget(self.start_btn); btn_layout.addStretch()
        layout.addLayout(btn_layout)
        layout.addStretch(3)                                                           # Push content up

        # --- SOUND & ANIMATION SETUP ---
        self.startup_sound = QSoundEffect(self)
        self.startup_sound.setSource(QUrl.fromLocalFile("startup.wav"))
        self.startup_sound.setVolume(0.5)                                              # Set a sensible default volume

        # Apply opacity effects for fading
        self.anim_group = QParallelAnimationGroup()
        for widget in [self.title, self.subtitle, self.description, self.start_btn]:
            self._setup_fade(widget)

    def _setup_fade(self, widget):
        """Prepares a widget to be invisible and attaches a fade-in animation."""
        eff = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(eff)
        
        anim = QPropertyAnimation(eff, b"opacity")
        anim.setDuration(2000)                                                         # 2-second fade
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        self.anim_group.addAnimation(anim)

    def showEvent(self, event):
        """Triggers when the window actually appears on screen."""
        super().showEvent(event)
        self.startup_sound.play()                                                      # 1. Play sound immediately
        self.anim_group.start()                                                        # 2. Start all fades simultaneously

    def _go_to_load_tab(self):
        if self.main_window:
            self.main_window.centralWidget().setCurrentIndex(1)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyNSD - The PNSD Toolkit")
        self.resize(1600, 1000)                                                        # Slightly wider startup
        
        self.setStyleSheet("""
            QWidget { background-color: #fff1e5; font-family: Georgia, serif; color: #33302e; }
            QTabWidget::pane { border: 1px solid #ccc; }
            QTabBar::tab { background: #f2e5da; padding: 10px 25px; border: 1px solid #ccc; font-size: 14px; }
            QTabBar::tab:selected { background: #fff1e5; font-weight: bold; border-bottom: none; }
            QPushButton { background-color: #e2d5cb; border: 1px solid #b3a8a0; padding: 5px; border-radius: 3px; }
            QPushButton:hover { background-color: #d1c4ba; }
            QLineEdit, QComboBox, QTableWidget { background-color: #ffffff; border: 1px solid #ccc; }
        """)

        self.loaded_data = {}

        self.tabs = QTabWidget()
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

        self.load_panel.data_confirmed.connect(self._on_data_confirmed)
        
        self.tabs.addTab(self.landing_panel, "Welcome")
        self.tabs.addTab(self.load_panel, "1 · Load data")
        self.tabs.addTab(self.summary_panel, "2 · Summary")
        self.tabs.addTab(self.trend_panel, "3 · Trend Analysis")
        self.tabs.addTab(self.npf_panel_manual, "4 · NPF Identifier")
        self.tabs.addTab(self.npf_dl_panel, "5 · NPF Deep Learning")
        self.tabs.addTab(self.cluster_panel, "6 · Cluster Analysis")
        self.tabs.addTab(self.nano_ranking_panel, "7 · Nano Ranking")
        self.tabs.addTab(self.wind_panel, "8 · Wind Analysis")                          # Fixed numbering
        self.tabs.addTab(self.pmf_panel, "9 · PMF Analysis")

        self.setCentralWidget(self.tabs)

    def _on_data_confirmed(self, results: dict):
        self.loaded_data = results
        if results:
            combined_df = pd.concat([r.df for r in results.values()]).sort_index()
            first_file = list(results.values())[0]
            
            merged_data = DataFile(
                path=first_file.path,
                df=combined_df,
                diameters=first_file.diameters,
                n_rows=len(combined_df),
                n_bins=first_file.n_bins
            )
            
            # Pass to panels
            self.summary_panel.load_data(merged_data)
            self.trend_panel.load_data(merged_data)
            self.npf_panel_manual.load_data(merged_data)
            self.npf_dl_panel.load_data(merged_data)
            self.cluster_panel.load_data(merged_data)
            self.nano_ranking_panel.load_data(merged_data)
            self.wind_panel.load_data(merged_data)
            self.pmf_panel.load_data(merged_data)

            self.tabs.setCurrentIndex(2)                                               # Switch to Summary tab

    def _on_data_confirmed(self, results: dict):
        self.loaded_data = results                                   # Store raw results
        
        if results:                                                  
            combined_df = pd.concat([r.df for r in results.values()])# Merge all loaded dataframes
            combined_df = combined_df.sort_index()                   # Sort chronologically by date
            
            first_file = list(results.values())[0]                   # Use first file as a template
            
            merged_data = DataFile(                                  # Create a new combined DataFile
                path=first_file.path,                                
                df=combined_df,                                      
                diameters=first_file.diameters,                      # Assumes all files have identical bins
                n_rows=len(combined_df),                             
                n_bins=first_file.n_bins                             
            )
            
            self.summary_panel.load_data(merged_data)               # Pass merged data to Summary tab
            self.trend_panel.load_data(merged_data)                  # Pass merged data to Trend tab
            self.npf_panel_manual.load_data(merged_data)
            self.npf_dl_panel.load_data(merged_data)
            self.cluster_panel.load_data(merged_data)
            self.nano_ranking_panel.load_data(merged_data)
            self.wind_panel.load_data(merged_data)
            self.pmf_panel.load_data(merged_data)



        self.centralWidget().setCurrentIndex(2)                      # Automatically switch to Summary tab
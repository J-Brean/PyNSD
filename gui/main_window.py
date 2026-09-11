"""
gui/main_window.py
------------------
Root application window.  Holds the left navigation, the stacked panels and the
shared data state.
"""

from pathlib import Path

from PyQt6.QtWidgets import (QMainWindow, QStackedWidget, QListWidget,
                             QListWidgetItem, QWidget, QVBoxLayout, QLabel,
                             QPushButton, QHBoxLayout, QGraphicsOpacityEffect,
                             QScrollArea, QMessageBox)
from PyQt6.QtCore import Qt, QSettings, QUrl, QPropertyAnimation, QParallelAnimationGroup
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtMultimedia import QSoundEffect
import pandas as pd

from gui import session
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
from utils.helpers import available_geometry, busy_cursor, fit_to_screen


class LandingPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self._has_played = False                                       # guard: play sound/fade once

        # The 82 px title alone needs ~1400 px of width, which overflows a
        # 1366-wide laptop.  Drop to the compact type scale on small screens.
        geo = available_geometry(self)
        self._compact = geo is not None and geo.width() < 1500

        layout = QVBoxLayout(self)
        pad = 24 if self._compact else 50
        layout.setContentsMargins(pad, pad, pad, pad)

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

        for lbl in (self.title, self.subtitle, self.description):
            lbl.setWordWrap(True)
            lbl.setProperty("compact", self._compact)                  # Picked up by style.qss

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
        self.startup_sound.setSource(QUrl.fromLocalFile(
            str(Path(__file__).resolve().parent.parent / "startup.wav")))   # not the working directory
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
        fit_to_screen(self, 1600, 1000)                                # Never open bigger than the user's desktop

        geo = available_geometry(self)                                 # A floor, but never one a small screen cannot meet
        if geo is not None:
            self.setMinimumSize(min(960, int(geo.width() * 0.6)),
                                min(620, int(geo.height() * 0.6)))

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

        # Each panel goes in its own scroll area.  Some panels (PMF, clustering,
        # nano ranking) have control rows whose combined minimum width is wider
        # than a laptop screen; without this the window's layout minimum wins and
        # the user simply cannot drag it smaller.
        self.stack = QStackedWidget()
        for panel in (self.landing_panel, self.load_panel, self.summary_panel,
                      self.trend_panel, self.npf_panel_manual, self.npf_dl_panel,
                      self.cluster_panel, self.nano_ranking_panel, self.wind_panel,
                      self.pmf_panel, self.pollution_flag_panel):
            scroller = QScrollArea()
            scroller.setWidget(panel)
            scroller.setWidgetResizable(True)                           # Panel still fills the viewport when there is room
            scroller.setFrameShape(QScrollArea.Shape.NoFrame)
            self.stack.addWidget(scroller)

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
        self._add_nav_item("Automated NPF identifiers", 5, analysis=True)
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

        self._build_menus()
        self._restore_window_state()
        self.nav.setCurrentRow(0)                                      # Welcome by default

    # ---- Menus, shortcuts and recent files -------------------------------- #
    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        open_action = QAction("&Open data files…", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)          # Ctrl+O
        open_action.triggered.connect(self._open_files)
        file_menu.addAction(open_action)

        self.recent_menu = file_menu.addMenu("Open &recent")
        self._refresh_recent_menu()

        file_menu.addSeparator()
        self.restore_action = QAction("Restore last session", self)
        self.restore_action.triggered.connect(lambda: self._restore_session(ask=False))
        file_menu.addAction(self.restore_action)

        file_menu.addSeparator()
        quit_action = QAction("E&xit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        edit_menu = self.menuBar().addMenu("&Edit")
        self.undo_action = QAction("&Undo last correction", self)
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)     # Ctrl+Z
        self.undo_action.triggered.connect(self._undo)
        edit_menu.addAction(self.undo_action)

    def _open_files(self) -> None:
        self.show_load()
        self.load_panel._browse_files()

    def _undo(self) -> None:
        """Ctrl+Z anywhere takes back the last Load-panel correction."""
        if self.load_panel._undo_stack:
            self.show_load()
            self.load_panel._undo_correction()
        else:
            self.statusBar().showMessage("Nothing to undo.", 3000)

    def _refresh_recent_menu(self) -> None:
        self.recent_menu.clear()
        files = session.recent_files()
        for path in files:
            action = QAction(Path(path).name, self)
            action.setToolTip(path)
            action.triggered.connect(lambda _checked, p=path: self._open_recent(p))
            self.recent_menu.addAction(action)
        if not files:
            empty = QAction("No recent files", self)
            empty.setEnabled(False)
            self.recent_menu.addAction(empty)
        else:
            self.recent_menu.addSeparator()
            clear = QAction("Clear list", self)
            clear.triggered.connect(lambda: (session.clear_recent(), self._refresh_recent_menu()))
            self.recent_menu.addAction(clear)

    def _open_recent(self, path: str) -> None:
        self.show_load()
        self.load_panel._add_files([path])
        self._refresh_recent_menu()

    # ---- Session ---------------------------------------------------------- #
    def _restore_window_state(self) -> None:
        stored = QSettings().value("window/geometry")
        if stored is not None:
            self.restoreGeometry(stored)

    def offer_session_restore(self) -> None:
        """On startup, offer to reopen whatever was loaded last time."""
        payload = session.load()
        if not payload:
            return

        names = ", ".join(Path(f["path"]).name for f in payload["files"][:3])
        more = f" and {len(payload['files']) - 3} more" if len(payload["files"]) > 3 else ""
        answer = QMessageBox.question(
            self, "Restore your last session?",
            f"PyNSD was last using {names}{more}.\n\n"
            f"Saved {payload.get('saved_at', 'earlier').replace('T', ' at ')}.\n\n"
            "Reopen those files with the same import settings?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            self._restore_session(ask=False, payload=payload)

    def _restore_session(self, ask: bool = True, payload: dict | None = None) -> None:
        payload = payload or session.load()
        if not payload:
            QMessageBox.information(self, "No session saved",
                                    "There is no previous session to restore.")
            return
        self.show_load()
        self.load_panel.restore_session(payload)
        self._refresh_recent_menu()
        if payload.get("corrections"):
            QMessageBox.information(
                self, "Corrections not reapplied",
                "These corrections were applied last time and have not been redone:\n\n• "
                + "\n• ".join(payload["corrections"]))

    def closeEvent(self, event):
        QSettings().setValue("window/geometry", self.saveGeometry())
        super().closeEvent(event)

    # ---- Navigation helpers ------------------------------------------- #
    def _add_nav_header(self, text: str) -> None:
        item = QListWidgetItem(text)
        item.setFlags(Qt.ItemFlag.NoItemFlags)                         # non-selectable header
        font = self.nav.font(); font.setBold(True); font.setPointSize(8)  # item.font() is unset (-1 pt) before the item is added
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
            files = list(results.values())
            first_file = files[0]

            # Every panel trusts that df's columns are the declared diameters.
            # Concatenating files with different bins breaks that: the frame
            # gains the union of both bin sets, half of it NaN and out of size
            # order, while n_bins still describes the first file only.  Stop
            # here instead, and point at the tools that exist for the job.
            mismatched = [f.path.name for f in files[1:]
                          if list(f.diameters) != list(first_file.diameters)]
            if mismatched:
                QMessageBox.warning(
                    self, "Size bins do not match",
                    f"{first_file.path.name} has {first_file.n_bins} size bins, but "
                    f"{', '.join(mismatched)} does not use the same bins.\n\n"
                    "Harmonise or splice the files onto a common bin set in the "
                    "Load panel, then continue.")
                return

            combined_df = pd.concat([r.df for r in files]).sort_index()

            overlapping = int(combined_df.index.duplicated().sum())
            if overlapping:
                QMessageBox.information(
                    self, "Overlapping timestamps",
                    f"These files share {overlapping} timestamp(s), so those periods "
                    "are counted twice in averages and diurnals. Trim the overlap in "
                    "the Load panel if that is not what you want.")

            merged_data = DataFile(
                path=first_file.path,
                df=combined_df,
                diameters=first_file.diameters,                        # Bins verified identical above
                n_rows=len(combined_df),
                n_bins=first_file.n_bins,
            )

            with busy_cursor():                                        # this can take a few seconds
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

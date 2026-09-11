"""
gui/npf_auto.py
---------------
The automated NPF identifiers: an image model, a physical classifier, or both,
with an optional growth-rate tracker, and summary plots over the whole record.

The two identifiers
-------------------
The image model (:mod:`utils.npf_render`) works from a daily contour plot and
returns p(NPF). Because it was trained on hand-labelled pictures, it tends to
agree with whatever those labellers counted as an event, and its recall can only
be measured against the same labels.

The physical classifier (:mod:`utils.npf_classify`) works from the measurements
and tests three criteria: a selective rise in the smallest particles, a coherent
growing mode, and an early start. Each day is reported with the criterion it
failed on, so the result can be checked rather than taken on trust.

The two are worth running together. Days they both accept are usually
straightforward. Days they disagree on are worth looking at: a clean growth track
the model rejects is generally a day unlike its training set, and a day the model
accepts but the criteria reject is often a plume arriving already grown.

Growth tracking
---------------
Growth tracking can be switched off, which makes a run much faster, but nothing
then tests whether the mode grew, so no day can reach class I. Days that rise
without a track are reported as ``undefined``. The panel notes this when that
combination is selected.

Output
------
One row per day, holding the intermediate quantities as well as the verdict: the
rise ratio, the selectivity, both growth-rate estimates and whether they agree,
the CNN probability, and each identifier's call. The summary plots are drawn from
this table, and it can be exported as CSV.

The summary plots
-----------------
Classes             Day counts by class, and the share of class I days by month.
                    Season is not used as a criterion, so the seasonal cycle is a
                    useful check that the classifier is behaving sensibly.
Mean distributions  Mean size distribution and mean diurnal total number, for
                    event days against the rest.
Growth rates        The distribution of tracked growth rates, and the path fit
                    against the appearance-time estimate. Points away from the
                    one-to-one line are worth checking by hand.
Agreement           Where the two identifiers agree, and the spread of CNN
                    probability within each class from the classifier.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.ticker import NullFormatter, ScalarFormatter

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
                             QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                             QMessageBox, QPushButton, QSpinBox,
                             QTabWidget, QVBoxLayout, QWidget)

from gui.filedialogs import get_open_file_name, get_save_file_name
from gui.widgets import CollapsibleSection, RunProgress
from gui.workers import CancellableWorker
from utils.calculations import dlogdp_per_bin, integrate_pnsd
from utils.npf_classify import (CLASS_ORDER, ClassifierSettings, classify,
                                daily_metrics)
from utils.npf_render import render_day
from utils.npf_tracking import TrackerSettings, track_days

CLASS_COLOURS = {"Ia": "#0f6e56", "Ib": "#4c9f70", "II": "#e0a458",
                 "undefined": "#a0a0a0", "non-event": "#c9c2bb"}
EVENT_CLASSES = ["Ia", "Ib"]

METHOD_SUMMARY = """These tools are meant to give you a first pass over a long
record. Check the days that matter to your result by eye before you use any of
this in a paper.<br><br>

<b>The image model (CNN)</b><br>
This is a convolutional network from Kecorius et al. (2024), trained on 4,819
daily contour plots that were labelled by hand as NPF, ordinary, or bad data.
It works from a picture of the day rather than from the measurements, so it
tends to agree with whatever its labellers counted as an event. PyNSD draws
each day in the same style the model was trained on, using the same colour
palette, image size, axes and 0 to 5 colour range, with missing hours left
black. A day is counted as an event when p(NPF) reaches 0.10, rather than by
taking the most likely class. That threshold raises recall from 0.80 to 0.90
against the training labels, and precision only falls from 0.99 to 0.93. You can
also have each day scored twice, once mirrored left to right, and the two
probabilities averaged, which settles borderline days.<br><br>

<b>The physical classifier</b><br>
This works from the measurements, and follows Dal Maso et al. (2005). It tests
three things, and reports which one a day failed on.<br>
1. <i>New.</i> There is a rise in 20 to 25 nm particles between 10:00 and 16:00,
compared with the median night across the surrounding week. Using a week of
nights rather than the one before means a single polluted night does not remove
a real event.<br>
2. <i>Selective.</i> That rise is larger than the equivalent rise in 50 to 100 nm
particles. Formation lifts the small end on its own, whereas a shallow boundary
layer or an advected plume lifts the whole distribution.<br>
3. <i>Grows, and starts small and early.</i> There is a mode that climbs in
diameter for several hours, and it starts at the small end during the morning or
early afternoon. Without the timing test, a plume arriving already grown in the
afternoon is hard to tell from formation.<br>
Days are then sorted into Ia (all three, clearly), Ib (all three, less clearly),
II (growth or rise present but weak), undefined (small particles appear but do
not grow coherently), and non-event.<br><br>

<b>The growth rate tracker</b><br>
Each hour's spectrum is smoothed twice along log10 Dp, and the broader of the
two is subtracted from the narrower. This leaves local bumps, so a small growing
mode is not swamped by the accumulation mode. Each point is also scored on how
far it sits above the same size range overnight, otherwise the search tends to
follow the standing Aitken mode and return a growth rate near zero.<br><br>
The track itself is found with a dynamic programme. The state is the pair (size,
current growth rate), and the growth rate is allowed to change by one step per
hour at a small cost, which keeps the track smooth. No individual hour is
accepted or rejected on a threshold, so a noisy hour on its own does not move the
track. The mode diameter is then refined below the channel spacing by fitting a
parabola through the peak, the track is cut back to its longest continuous
rising section, and the growth rate is the slope of diameter against time.<br><br>
A second estimate is made from the appearance time of each size bin, which uses
none of the same machinery. Both are reported, along with a flag for whether
they agree. The path fit tends to read low when growth is faster than about
5 nm/h, so the two are worth comparing on any day you intend to quote.<br><br>

<b>What is not used</b><br>
Solar radiation, temperature and condensation sink are not part of any
criterion. If sunshine were required, a summer maximum would be built in rather
than found. The seasonal cycle on the Classes tab is therefore worth looking at
as a check that the classifier is behaving sensibly.<br><br>

<b>Limitations</b><br>
You can only see what the instrument measures. From about 20 nm upwards these
are events that have already grown past the nucleation mode, not the sub-10 nm
growth that needs a PSM or a NAIS. The default thresholds were tuned on UK urban
and rural SMPS records, and you may well want different ones for another site or
size range, so they are all adjustable.
"""


class AutoIdentifierWorker(CancellableWorker):
    """Runs the chosen identifiers over the whole record."""

    def __init__(self, df, diams, options: dict):
        super().__init__()
        self.df = df
        self.diams = np.asarray(diams, dtype=float)
        self.options = options

    def work(self):
        opts = self.options
        result = {"options": opts}
        use_cnn = opts["method"] in ("CNN", "Both")
        use_physical = opts["method"] in ("Physical", "Both")

        tracks = None
        if opts["track_growth"]:
            self.tick(0, 1, "Tracking growth…")
            tracks = track_days(self.df, self.diams, opts["tracker"],
                                progress=lambda d, t, m: self.tick(d, t, m))
            result["tracks"] = tracks

        if use_physical:
            self.tick(0, 1, "Measuring daily rise and selectivity…")
            metrics = daily_metrics(self.df, self.diams, opts["classifier"])
            result["classified"] = classify(metrics, tracks, opts["classifier"])

        if use_cnn:
            result["cnn"] = self._run_cnn()

        self.tick(1, 1, "Summarising…")
        result["daily"] = self._daily_summary(result)
        return result

    def _run_cnn(self) -> pd.DataFrame:
        """Render each day in the training style and score it."""
        try:
            from fastai.vision.all import load_learner
        except ImportError:
            raise RuntimeError("fastai is not installed, so the CNN cannot run.")
        model_path = self.options["model_path"]
        if not os.path.exists(model_path):
            raise RuntimeError(f"No CNN model at {model_path}.")

        learner = load_learner(model_path, cpu=True)
        learner.dls.num_workers = 0
        vocab = [str(v) for v in learner.dls.vocab]

        groups = list(self.df.groupby(self.df.index.date))
        folder = os.path.join(tempfile.gettempdir(), "pynsd_cnn_render")
        os.makedirs(folder, exist_ok=True)

        # Render everything first, then score in batches. Predicting one image
        # at a time spends most of its time in setup rather than in the network.
        views = [False, True] if self.options["mirror_tta"] else [False]
        files, days = [], []
        for i, (day, day_df) in enumerate(groups):
            self.tick(i, len(groups), f"Drawing {day}")
            if len(day_df) < 12:
                continue
            for mirror in views:
                path = os.path.join(folder, f"{day}{'_m' if mirror else ''}.png")
                render_day(day_df, self.diams, path, mirror=mirror)
                files.append(path)
            days.append(pd.Timestamp(day))

        if not days:
            return pd.DataFrame()

        self.tick(0, len(files), f"Scoring {len(days)} days")
        with learner.no_bar(), learner.no_logging():
            dl = learner.dls.test_dl(files, num_workers=0, bs=16)
            probs, _ = learner.get_preds(dl=dl)
        probs = np.asarray(probs, dtype=float).reshape(len(days), len(views), -1).mean(axis=1)

        rows = [{"day": day, **{f"p_{v}": probs[i, j] for j, v in enumerate(vocab)}}
                for i, day in enumerate(days)]
        out = pd.DataFrame(rows).set_index("day")
        # Argmax is not the model's best operating point: scoring a day as NPF
        # at p >= 0.10 raises recall against the training labels from 0.80 to
        # 0.90 while precision only falls from 0.99 to 0.93.
        if "p_NPF" in out.columns:
            out["cnn_npf"] = out["p_NPF"] >= self.options["cnn_threshold"]
        return out

    def _daily_summary(self, result: dict) -> pd.DataFrame:
        """One row per day, with whatever each identifier had to say about it."""
        widths = dlogdp_per_bin(self.diams)
        hourly = self.df.resample("h").mean()
        measured = hourly.notna().any(axis=1)
        total_n = pd.Series(integrate_pnsd(hourly.to_numpy(dtype=float), widths),
                            index=hourly.index).where(measured)

        daily = pd.DataFrame({
            "N_total": total_n.groupby(total_n.index.date).mean(),
            "hours": measured.groupby(measured.index.date).sum(),
        })
        daily.index = pd.to_datetime(daily.index)
        daily.index.name = "day"
        # A record with gaps resamples into days that were never measured. They
        # are not days, and left in they swamp any count of class shares.
        daily = daily[daily["hours"] > 0]

        if "classified" in result:
            daily = daily.join(result["classified"][["class", "ratio7", "selectivity"]])
        if "tracks" in result:
            daily = daily.join(result["tracks"][["gr", "gr_app", "r2", "from", "to",
                                                 "dp_from", "dp_to", "agree", "plausible"]])
        if "cnn" in result and not result["cnn"].empty:
            keep = [c for c in ["p_NPF", "cnn_npf"] if c in result["cnn"].columns]
            daily = daily.join(result["cnn"][keep])

        physical_event = (daily["class"].isin(EVENT_CLASSES)
                          if "class" in daily.columns else pd.Series(False, index=daily.index))
        cnn_event = (daily["cnn_npf"].fillna(False)
                     if "cnn_npf" in daily.columns else pd.Series(False, index=daily.index))
        daily["physical_event"] = physical_event
        daily["cnn_event"] = cnn_event
        daily["event"] = physical_event | cnn_event
        daily["agreement"] = np.select(
            [physical_event & cnn_event, physical_event & ~cnn_event, ~physical_event & cnn_event],
            ["both", "physical only", "CNN only"], default="neither")
        return daily


class AutoIdentifierSection(QGroupBox):
    """The controls, the run, and the summary plots."""

    def __init__(self, panel, parent=None):
        super().__init__("Automated NPF identifiers", parent)
        self.panel = panel                      # the host panel, for df/diams/model_path
        self.result = None
        self._worker = None
        self._build_ui()

    # ---- interface -------------------------------------------------------- #
    def _build_ui(self):
        root = QVBoxLayout(self)

        intro_row = QHBoxLayout()
        intro = QLabel(
            "In this module, you can use two different tools to automatically identify NPF "
            "events. There is also a module that will attempt to automatically calculate "
            "growth rates. Treat these as preliminary!")
        intro.setWordWrap(True)
        intro_row.addWidget(intro, stretch=1)

        btn_info = QPushButton("ℹ️")
        btn_info.setFixedSize(24, 24)
        btn_info.setToolTip("How each identifier works")
        btn_info.clicked.connect(self._show_method)
        intro_row.addWidget(btn_info, alignment=Qt.AlignmentFlag.AlignTop)
        root.addLayout(intro_row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Identifier:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Both", "CNN", "Physical"])
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        row.addWidget(self.method_combo)

        self.chk_growth = QCheckBox("Track growth rates")
        self.chk_growth.setChecked(True)
        self.chk_growth.setToolTip("Required for class I. Without it a day can only reach "
                                   "'undefined', since nothing tests whether the mode grew.")
        self.chk_growth.toggled.connect(self._on_method_changed)
        row.addWidget(self.chk_growth)
        row.addStretch()

        self.btn_run = QPushButton("▶ Run identifiers")
        self.btn_run.setProperty("class", "primary")
        self.btn_run.clicked.connect(self._run)
        row.addWidget(self.btn_run)
        root.addLayout(row)

        self.progress = RunProgress()
        self.progress.cancel_requested.connect(self._cancel)
        root.addWidget(self.progress)

        # Foldable, so the plots get the room once the tuning is settled.
        settings = QHBoxLayout()
        settings.addWidget(self._build_cnn_box())
        settings.addWidget(self._build_physical_box())
        settings.addWidget(self._build_tracker_box())
        self.settings_section = CollapsibleSection("Settings", expanded=True)
        self.settings_section.set_content_layout(settings)
        root.addWidget(self.settings_section)

        self.status = QLabel("Load and confirm a dataset, then run.")
        self.status.setObjectName("FileStatus")
        root.addWidget(self.status)

        self.tabs = QTabWidget()
        self.figures = {}
        for name in ["Classes", "Mean days", "Mean distributions", "Growth rates", "Agreement"]:
            fig = Figure(figsize=(7, 4.2))
            canvas = FigureCanvasQTAgg(fig)
            holder = QWidget()
            lay = QVBoxLayout(holder)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(canvas)
            self.figures[name] = (fig, canvas)
            self.tabs.addTab(holder, name)
        self.tabs.setMinimumHeight(430)          # the plots need room to be read
        root.addWidget(self.tabs, stretch=1)

        export = QHBoxLayout()
        export.addStretch()
        self.btn_export = QPushButton("💾 Export daily table")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export)
        export.addWidget(self.btn_export)
        root.addLayout(export)

        self._on_method_changed()

    def _spin(self, value, lo, hi, step=0.1, decimals=2):
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setDecimals(decimals)
        w.setValue(value)
        return w

    def _build_cnn_box(self) -> QGroupBox:
        self.cnn_box = QGroupBox("Image model (CNN)")
        form = QFormLayout(self.cnn_box)
        self.cnn_threshold = self._spin(0.10, 0.01, 1.0, 0.01, 2)
        self.cnn_threshold.setToolTip(
            "p(NPF) at or above which a day counts as an event. The argmax is not the "
            "model's best operating point; 0.10 is the recommended one, and below it "
            "precision collapses.")
        form.addRow("p(NPF) threshold", self.cnn_threshold)
        self.chk_mirror = QCheckBox("Average with the mirrored day")
        self.chk_mirror.setChecked(True)
        self.chk_mirror.setToolTip("Scores each day twice, once flipped left to right, and "
                                   "averages. Steadies borderline days. Twice as slow.")
        form.addRow("", self.chk_mirror)

        # PyNSD does not ship the model, so it has to be found once and remembered.
        self.lbl_model = QLabel()
        self.lbl_model.setObjectName("FileStatus")
        self.lbl_model.setWordWrap(True)
        btn_model = QPushButton("Locate model…")
        btn_model.clicked.connect(self._locate_model)
        form.addRow(btn_model, self.lbl_model)
        self._refresh_model_label()
        return self.cnn_box

    def _model_path(self) -> str:
        stored = QSettings().value("npf/model_path", "", type=str)
        if stored and os.path.exists(stored):
            return stored
        return getattr(self.panel, "model_path", "")

    def _refresh_model_label(self):
        path = self._model_path()
        if path and os.path.exists(path):
            self.lbl_model.setText(f"✓ {os.path.basename(path)}")
        else:
            self.lbl_model.setText("No model found. The CNN cannot run until you point at one.")

    def _locate_model(self):
        path, _ = get_open_file_name(self, "Select the CNN model", "",
                                     "fastai model (*.pkl);;All files (*)")
        if path:
            QSettings().setValue("npf/model_path", path)
            self._refresh_model_label()

    def _build_physical_box(self) -> QGroupBox:
        self.physical_box = QGroupBox("Physical classifier")
        grid = QGridLayout(self.physical_box)
        d = ClassifierSettings()

        self.p_widgets = {}
        fields = [
            ("r_strong", "Rise, strong (N_act/N_bg)", d.r_strong, 1.0, 20.0, 0.1, 1),
            ("r_weak", "Rise, weak", d.r_weak, 1.0, 20.0, 0.1, 1),
            ("selectivity_min", "Selectivity, small vs large", d.selectivity_min, 1.0, 10.0, 0.1, 2),
            ("r2_strong", "Track R², strong", d.r2_strong, 0.0, 1.0, 0.05, 2),
            ("r2_weak", "Track R², weak", d.r2_weak, 0.0, 1.0, 0.05, 2),
            ("dur_strong", "Hours of growth, strong", d.dur_strong, 1.0, 24.0, 1.0, 0),
            ("dur_weak", "Hours of growth, weak", d.dur_weak, 1.0, 24.0, 1.0, 0),
            ("dp_start_max", "Track must start below (nm)", d.dp_start_max, 5.0, 200.0, 1.0, 0),
            ("growth_factor_min", "Overall growth factor", d.growth_factor_min, 1.0, 10.0, 0.1, 2),
        ]
        for i, (key, label, value, lo, hi, step, dec) in enumerate(fields):
            grid.addWidget(QLabel(label), i, 0)
            w = self._spin(value, lo, hi, step, dec)
            self.p_widgets[key] = w
            grid.addWidget(w, i, 1)

        grid.addWidget(QLabel("Track must start between"), len(fields), 0)
        hours = QHBoxLayout()
        self.hour_min = QSpinBox(); self.hour_min.setRange(0, 23); self.hour_min.setValue(d.hour_start_min)
        self.hour_max = QSpinBox(); self.hour_max.setRange(0, 23); self.hour_max.setValue(d.hour_start_max)
        hours.addWidget(self.hour_min); hours.addWidget(QLabel("and")); hours.addWidget(self.hour_max)
        holder = QWidget(); holder.setLayout(hours)
        grid.addWidget(holder, len(fields), 1)

        btn = QPushButton("Reset to defaults")
        btn.clicked.connect(self._reset_physical)
        grid.addWidget(btn, len(fields) + 1, 0, 1, 2)
        return self.physical_box

    def _build_tracker_box(self) -> QGroupBox:
        self.tracker_box = QGroupBox("Growth rate tracker")
        form = QFormLayout(self.tracker_box)
        d = TrackerSettings()
        self.t_dp_lo = self._spin(d.dp_lo, 1.0, 100.0, 1.0, 0)
        self.t_dp_hi = self._spin(d.dp_hi, 20.0, 1000.0, 5.0, 0)
        self.t_min_hours = QSpinBox(); self.t_min_hours.setRange(2, 24); self.t_min_hours.setValue(d.min_hours)
        self.t_anom = self._spin(d.anom_floor, 0.0, 2.0, 0.05, 2)
        form.addRow("Track from (nm)", self.t_dp_lo)
        form.addRow("Track to (nm)", self.t_dp_hi)
        form.addRow("Minimum hours", self.t_min_hours)
        form.addRow("Above its own night by", self.t_anom)
        note = QLabel("The tracker reports two growth rates: the path fit and an\n"
                      "independent appearance-time estimate. Where they disagree,\n"
                      "the day is flagged rather than averaged.")
        note.setObjectName("FileStatus")
        form.addRow(note)
        return self.tracker_box

    def _show_method(self):
        """The method, in the panel, so it need not be read from the source."""
        QMessageBox.information(self, "How the identifiers work", METHOD_SUMMARY)

    def _reset_physical(self):
        d = ClassifierSettings()
        for key, w in self.p_widgets.items():
            w.setValue(getattr(d, key))
        self.hour_min.setValue(d.hour_start_min)
        self.hour_max.setValue(d.hour_start_max)

    def _on_method_changed(self, *_):
        method = self.method_combo.currentText()
        self.cnn_box.setEnabled(method in ("CNN", "Both"))
        self.physical_box.setEnabled(method in ("Physical", "Both"))
        self.tracker_box.setEnabled(self.chk_growth.isChecked())
        if method in ("Physical", "Both") and not self.chk_growth.isChecked():
            self.status.setText("Growth tracking is off, so no day can reach class I.")
        else:
            self.status.setText("Ready.")

    # ---- running ---------------------------------------------------------- #
    def _collect_options(self) -> dict:
        classifier = ClassifierSettings(
            **{k: w.value() for k, w in self.p_widgets.items()},
            hour_start_min=self.hour_min.value(),
            hour_start_max=self.hour_max.value())
        tracker = TrackerSettings(dp_lo=self.t_dp_lo.value(), dp_hi=self.t_dp_hi.value(),
                                  min_hours=self.t_min_hours.value(),
                                  anom_floor=self.t_anom.value())
        return {"method": self.method_combo.currentText(),
                "track_growth": self.chk_growth.isChecked(),
                "classifier": classifier, "tracker": tracker,
                "cnn_threshold": self.cnn_threshold.value(),
                "mirror_tta": self.chk_mirror.isChecked(),
                "model_path": self._model_path()}

    def _run(self):
        df = getattr(self.panel, "df", None)
        if df is None or df.empty:
            QMessageBox.warning(self, "No data", "Load and confirm a dataset first.")
            return
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "Already running", "Wait for the current run, or cancel it.")
            return

        self.btn_run.setEnabled(False)
        self._worker = AutoIdentifierWorker(df, self.panel.diams, self._collect_options())
        self._worker.progress.connect(self.progress.update)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.cancelled.connect(self._on_cancelled)
        self.progress.start("Running identifiers…")
        self._worker.start()

    def _cancel(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()

    def _on_cancelled(self):
        self.progress.stop()
        self.btn_run.setEnabled(True)
        self.status.setText("Stopped. Nothing was changed.")

    def _on_error(self, message):
        self.progress.stop()
        self.btn_run.setEnabled(True)
        self.status.setText(f"Failed: {message}")
        QMessageBox.warning(self, "Identifiers failed", message)

    def _on_done(self, result):
        self.progress.stop()
        self.btn_run.setEnabled(True)
        self.result = result
        self.btn_export.setEnabled(True)
        self._describe(result)
        self._draw(result)
        self.settings_section.set_expanded(False)       # give the plots the room

    def _describe(self, result):
        daily = result["daily"]
        bits = [f"{len(daily)} days"]
        if "class" in daily.columns:
            counts = daily["class"].value_counts()
            bits.append("physical: " + ", ".join(f"{c} {int(counts.get(c, 0))}" for c in CLASS_ORDER))
        if "cnn_npf" in daily.columns:
            bits.append(f"CNN: {int(daily['cnn_npf'].sum())} events")
        if "gr" in daily.columns and daily["gr"].notna().any():
            bits.append(f"median GR {daily['gr'].median():.2f} nm/h "
                        f"({int(daily['gr'].notna().sum())} tracked)")
        self.status.setText("   |   ".join(bits))

    def _export(self):
        if self.result is None:
            return
        path, _ = get_save_file_name(self, "Export daily identifications",
                                     "npf_identifications.csv", "CSV (*.csv)")
        if path:
            self.result["daily"].to_csv(path, index_label="day")
            QMessageBox.information(self, "Exported", f"Written to {path}")

    # ---- summary plots ----------------------------------------------------- #
    def _draw(self, result):
        daily = result["daily"]
        self._draw_classes(daily)
        self._draw_mean_days(daily)
        self._draw_distributions(daily)
        self._draw_growth(daily)
        self._draw_agreement(daily)

    def _mean_day(self, days) -> np.ndarray | None:
        """Mean contour over a set of days, as hour by diameter.

        Averaged in log space, so one heavily polluted day does not dominate the
        mean the way it would on a linear average.
        """
        if not days:
            return None
        df = self.panel.df
        subset = df[pd.Index(df.index.date).isin(days)]
        if subset.empty:
            return None
        hourly = subset.resample("h").mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            logged = np.log10(hourly.to_numpy(dtype=float))
        logged[~np.isfinite(logged)] = np.nan
        frame = pd.DataFrame(logged, index=hourly.index)
        mean_log = frame.groupby(frame.index.hour).mean().reindex(range(24))
        return 10 ** mean_log.to_numpy(dtype=float)

    def _draw_mean_days(self, daily):
        """The mean banana, by class."""
        fig, _ = self.figures["Mean days"]
        fig.clear()

        groups = []
        if "class" in daily.columns:
            for cls in ["Ia", "Ib"]:
                groups.append((f"Class {cls}", set(daily.index[daily["class"] == cls].date)))
            groups.append(("Class I (Ia + Ib)",
                           set(daily.index[daily["class"].isin(EVENT_CLASSES)].date)))
            groups.append(("Non-event", set(daily.index[daily["class"] == "non-event"].date)))
        if "cnn_npf" in daily.columns:
            cnn = daily["cnn_npf"].fillna(False).astype(bool)
            groups.append(("CNN: NPF", set(daily.index[cnn].date)))
            groups.append(("CNN: not NPF", set(daily.index[~cnn].date)))

        groups = [(name, days) for name, days in groups if days]
        if not groups:
            fig.add_subplot(111).text(0.5, 0.5, "No days to average", ha="center", va="center")
            return self._finish("Mean days")

        diams = np.asarray(self.panel.diams, dtype=float)
        means = [(name, days, self._mean_day(days)) for name, days in groups]
        means = [(n, d, m) for n, d, m in means if m is not None]
        finite = np.concatenate([m[np.isfinite(m) & (m > 0)].ravel() for _, _, m in means])
        vmin = max(float(np.nanpercentile(finite, 5)), 1.0)
        vmax = float(np.nanpercentile(finite, 99.5))
        if vmax <= vmin:
            vmax = vmin * 10

        cols = min(3, len(means))
        rows = int(np.ceil(len(means) / cols))
        axes = fig.subplots(rows, cols, squeeze=False)
        mesh = None
        for ax, (name, days, mean) in zip(axes.ravel(), means):
            mesh = ax.pcolormesh(np.arange(24), diams, np.ma.masked_invalid(mean).T,
                                 cmap="turbo", norm=LogNorm(vmin=vmin, vmax=vmax),
                                 shading="nearest")
            ax.set_yscale("log")
            ax.set_title(f"{name}  (n = {len(days)})", fontsize=9)
            ax.set_xticks([0, 6, 12, 18])
        for ax in axes.ravel()[len(means):]:
            ax.set_visible(False)
        for ax in axes[-1]:
            if ax.get_visible():
                ax.set_xlabel("Hour of day")
        for ax in axes[:, 0]:
            ax.set_ylabel("Diameter (nm)")
        if mesh is not None:
            fig.colorbar(mesh, ax=axes.ravel().tolist(), label="dN/dlogD$_p$ (cm$^{-3}$)",
                         fraction=0.03, pad=0.02)
        self.figures["Mean days"][1].draw()

    def _finish(self, name):
        fig, canvas = self.figures[name]
        fig.tight_layout()
        canvas.draw()

    def _draw_classes(self, daily):
        fig, _ = self.figures["Classes"]
        fig.clear()
        if "class" not in daily.columns:
            fig.add_subplot(111).text(0.5, 0.5, "The physical classifier was not run",
                                      ha="center", va="center")
            return self._finish("Classes")

        ax1, ax2 = fig.subplots(1, 2)
        counts = daily["class"].value_counts().reindex(CLASS_ORDER, fill_value=0)
        ax1.bar(range(len(counts)), counts.values,
                color=[CLASS_COLOURS[c] for c in counts.index])
        ax1.set_xticks(range(len(counts)))
        ax1.set_xticklabels(counts.index, rotation=30, ha="right")
        ax1.set_ylabel("days")
        ax1.set_title("Days by class", fontsize=10)

        month = daily.index.month
        share = (daily["class"].isin(EVENT_CLASSES).groupby(month).mean() * 100)
        ax2.plot(share.index, share.values, marker="o", color=CLASS_COLOURS["Ia"])
        ax2.set_xticks(range(1, 13))
        ax2.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
        ax2.set_ylabel("class I days (%)")
        ax2.set_title("Class I days by month", fontsize=10)
        self._finish("Classes")

    def _draw_distributions(self, daily):
        fig, _ = self.figures["Mean distributions"]
        fig.clear()
        ax1, ax2 = fig.subplots(1, 2)
        diams = self.panel.diams
        df = self.panel.df

        event_days = set(daily.index[daily["event"]].date)
        quiet_days = set(daily.index[~daily["event"]].date)
        day_index = pd.Index(df.index.date)

        for label, days, colour in [("event days", event_days, "#a32d2d"),
                                    ("other days", quiet_days, "#185fa5")]:
            if not days:
                continue
            subset = df[day_index.isin(days)]
            ax1.plot(diams, subset.mean(axis=0).to_numpy(), color=colour, lw=1.6, label=label)
            hourly = subset.resample("h").mean()
            total = pd.Series(integrate_pnsd(hourly.to_numpy(dtype=float),
                                             dlogdp_per_bin(np.asarray(diams, dtype=float))),
                              index=hourly.index)
            ax2.plot(total.groupby(total.index.hour).mean(), color=colour, lw=1.6, label=label)

        ax1.set_xscale("log")
        ax1.xaxis.set_major_formatter(ScalarFormatter())      # decades, not 2x10^1
        ax1.xaxis.set_minor_formatter(NullFormatter())
        ax1.set_xlabel("Diameter (nm)")
        ax1.set_ylabel("dN/dlogD$_p$ (cm$^{-3}$)")
        ax1.set_title("Mean size distribution", fontsize=10)
        ax1.legend(fontsize=8)
        ax2.set_xlabel("Hour of day")
        ax2.set_ylabel("Total N (cm$^{-3}$)")
        ax2.set_title("Mean diurnal cycle", fontsize=10)
        self._finish("Mean distributions")

    def _draw_growth(self, daily):
        fig, _ = self.figures["Growth rates"]
        fig.clear()
        if "gr" not in daily.columns or not daily["gr"].notna().any():
            fig.add_subplot(111).text(0.5, 0.5, "Growth tracking was not run",
                                      ha="center", va="center")
            return self._finish("Growth rates")

        ax1, ax2 = fig.subplots(1, 2)
        gr = daily["gr"].dropna()
        ax1.hist(gr[gr.between(0, 20)], bins=24, color=CLASS_COLOURS["Ia"], edgecolor="white")
        ax1.set_xlabel("Growth rate (nm h$^{-1}$)")
        ax1.set_ylabel("days")
        ax1.set_title(f"Tracked growth rates (median {gr.median():.2f})", fontsize=10)

        both = daily[["gr", "gr_app"]].dropna()
        if not both.empty:
            agree = daily.loc[both.index, "agree"].fillna(False).astype(bool)
            ax2.scatter(both.loc[agree, "gr"], both.loc[agree, "gr_app"], s=12,
                        color=CLASS_COLOURS["Ia"], label="agree")
            ax2.scatter(both.loc[~agree, "gr"], both.loc[~agree, "gr_app"], s=12,
                        color="#a32d2d", label="disagree")
            lim = [0, float(np.nanpercentile(both.to_numpy(), 98))]
            ax2.plot(lim, lim, ls="--", color="grey", lw=0.8)
            ax2.set_xlim(lim); ax2.set_ylim(lim)
            ax2.legend(fontsize=8)
        ax2.set_xlabel("path fit (nm h$^{-1}$)")
        ax2.set_ylabel("appearance time (nm h$^{-1}$)")
        ax2.set_title("Two independent estimates", fontsize=10)
        self._finish("Growth rates")

    def _draw_agreement(self, daily):
        fig, _ = self.figures["Agreement"]
        fig.clear()
        if "cnn_npf" not in daily.columns or "class" not in daily.columns:
            fig.add_subplot(111).text(
                0.5, 0.5, "Run both identifiers to compare them", ha="center", va="center")
            return self._finish("Agreement")

        ax1, ax2 = fig.subplots(1, 2)
        counts = daily["agreement"].value_counts().reindex(
            ["both", "physical only", "CNN only", "neither"], fill_value=0)
        ax1.bar(range(4), counts.values,
                color=["#0f6e56", "#185fa5", "#854f0b", "#c9c2bb"])
        ax1.set_xticks(range(4))
        ax1.set_xticklabels(counts.index, rotation=25, ha="right")
        ax1.set_ylabel("days")
        ax1.set_title("Where the two identifiers agree", fontsize=10)

        for cls in CLASS_ORDER:
            sub = daily[daily["class"] == cls]["p_NPF"].dropna()
            if len(sub):                                   # a class of one still matters
                ax2.scatter(np.full(len(sub), CLASS_ORDER.index(cls))
                            + np.random.default_rng(0).normal(0, 0.07, len(sub)),
                            sub, s=8, alpha=0.5, color=CLASS_COLOURS[cls])
        ax2.axhline(self.cnn_threshold.value(), ls="--", color="grey", lw=0.8)
        ax2.set_xticks(range(len(CLASS_ORDER)))
        ax2.set_xticklabels(CLASS_ORDER, rotation=25, ha="right")
        ax2.set_ylabel("p(NPF) from the CNN")
        ax2.set_title("Image model score by physical class", fontsize=10)
        self._finish("Agreement")

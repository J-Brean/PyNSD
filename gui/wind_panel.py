from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from PyQt6.QtWidgets import (QWidget,
                             QVBoxLayout,
                             QHBoxLayout,
                             QLabel,
                             QPushButton,
                             QLineEdit,
                             QComboBox,
                             QGroupBox,
                             QMessageBox,
                             QCheckBox,
                             QDialog)

from utils.data_loader import DATE_COLUMN_OPTIONS, DATE_FORMAT_OPTIONS, parse_datetime_series
from utils.helpers import fit_to_screen
from utils.calculations import dlogdp_per_bin, integrate_pnsd
from gui.widgets import validate_number
from gui.filedialogs import get_open_file_name, get_save_file_name



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
        fit_to_screen(self, w, h)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.val_w.setText(str(self.canvas.width()))
        self.val_h.setText(str(self.canvas.height()))

    def apply_size(self):
        try:
            w, h = int(self.val_w.text()), int(self.val_h.text())
            fit_to_screen(self, w, h + 50)
        except ValueError:
            pass

    def save_plot(self):
        path, _ = get_save_file_name(
            self,
            "Save Plot",
            "",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)",
        )
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


class WindPanel(QWidget):
    def __init__(self, main_df=None, diams=None, parent=None):
        super().__init__(parent)

        self.df = main_df
        self.diams = np.array(diams) if diams is not None else None

        self.met_raw_df: pd.DataFrame | None = None
        self.met_df: pd.DataFrame | None = None
        self.joined_df: pd.DataFrame | None = None

        self._wd_col: str | None = None
        self._ws_col: str | None = None

        self._rose_patches = []
        self._sector_lines = {}
        self._sector_labels = []
        self._hover_cid = None
        self._sector_stats_df: pd.DataFrame | None = None
        self._sector_pnsd_df: pd.DataFrame | None = None
        self._sector_colors = None

        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        title_row = QHBoxLayout()
        title = QLabel("Wind Direction Analysis")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        title_row.addWidget(title)

        info_text = (
            "<b>Wind Panel Workflow</b><br><br>"
            "This panel links your aerosol size-distribution data (PNSD) with meteorology (wind direction + wind speed), "
            "then summarizes how concentrations and size spectra vary by wind sector.<br><br>"
            "<b>1) Load and parse met data</b><br>"
            "Load your met file and set the datetime column/format/timezone, just like in the main data loader. "
            "Then choose which columns are WD (wind direction, in degrees) and WS (wind speed).<br><br>"
            "<b>2) Optional averaging of met data</b><br>"
            "If you set a met averaging interval, wind direction is averaged correctly using <i>vector averaging</i> (circular mean), "
            "not a simple arithmetic angle mean. That avoids nonsense values around the 0°/360° boundary.<br><br>"
            "<b>3) Join met + PNSD</b><br>"
            "The panel performs a nearest-time join between PNSD timestamps and met timestamps, with a tolerance based on your PNSD resolution. "
            "Timezone alignment is handled before joining so UTC/local mismatches do not silently break the merge.<br><br>"
            "<b>4) Define sectors and build plots</b><br>"
            "Use sector definitions like 0-45,45-90,...,315-360 (custom sectors are allowed). "
            "The left plot is a pollutant rose (mean Total N per sector), and the right plot overlays the mean PNSD for each sector.<br><br>"
            "<b>5) Interactive linking</b><br>"
            "Hover over a wind sector in the rose and the corresponding PNSD curve is highlighted automatically.<br><br>"
            "<b>6) Exports</b><br>"
            "Export Plots saves separate high-resolution images for the rose and sector PNSDs. "
            "Export Data saves two CSV files: sector summary stats and sector mean PNSDs."
        )
        btn_info = QPushButton("ℹ️")
        btn_info.setFixedSize(24, 24)
        btn_info.clicked.connect(lambda: QMessageBox.information(self, "Wind Panel Guide", info_text))
        title_row.addWidget(btn_info)
        title_row.addStretch()
        root.addLayout(title_row)

        met_box = QGroupBox("1) Load & Parse Met Data")
        met_layout = QVBoxLayout(met_box)

        row_a = QHBoxLayout()
        self.btn_load_met = QPushButton("+ Load Met File")
        self.btn_load_met.clicked.connect(self._browse_met_file)
        row_a.addWidget(self.btn_load_met)
        self.met_path_lbl = QLabel("No met file loaded")
        row_a.addWidget(self.met_path_lbl)
        row_a.addStretch()
        met_layout.addLayout(row_a)

        row_b = QHBoxLayout()
        row_b.addWidget(QLabel("Date Col:"))
        self.date_col_combo = QComboBox()
        self.date_col_combo.addItems(DATE_COLUMN_OPTIONS)
        self.date_col_combo.currentTextChanged.connect(self._on_date_col_changed)
        row_b.addWidget(self.date_col_combo)

        self.custom_date_col = QLineEdit()
        self.custom_date_col.setPlaceholderText("Custom date column...")
        self.custom_date_col.setVisible(False)
        row_b.addWidget(self.custom_date_col)

        row_b.addWidget(QLabel("Format:"))
        self.fmt_combo = QComboBox()
        for disp, _ in DATE_FORMAT_OPTIONS:
            self.fmt_combo.addItem(disp)
        self.fmt_combo.currentIndexChanged.connect(self._on_fmt_changed)
        row_b.addWidget(self.fmt_combo)

        self.custom_fmt = QLineEdit()
        self.custom_fmt.setPlaceholderText("Custom datetime format...")
        self.custom_fmt.setVisible(False)
        row_b.addWidget(self.custom_fmt)

        row_b.addWidget(QLabel("Timezone:"))
        self.tz_input = QLineEdit("UTC")
        self.tz_input.setFixedWidth(100)
        row_b.addWidget(self.tz_input)
        met_layout.addLayout(row_b)

        row_c = QHBoxLayout()
        row_c.addWidget(QLabel("Wind Dir Col:"))
        self.wd_combo = QComboBox()
        row_c.addWidget(self.wd_combo)
        self.btn_set_wd = QPushButton("Use as WD")
        self.btn_set_wd.clicked.connect(self._set_wd_column)
        row_c.addWidget(self.btn_set_wd)

        row_c.addWidget(QLabel("Wind Speed Col:"))
        self.ws_combo = QComboBox()
        row_c.addWidget(self.ws_combo)
        self.btn_set_ws = QPushButton("Use as WS")
        self.btn_set_ws.clicked.connect(self._set_ws_column)
        row_c.addWidget(self.btn_set_ws)

        row_c.addWidget(QLabel("Met Avg:"))
        self.avg_val = QLineEdit("")
        self.avg_val.setPlaceholderText("Val")
        self.avg_val.setFixedWidth(40)
        row_c.addWidget(self.avg_val)
        self.avg_unit = QComboBox()
        self.avg_unit.addItems(["Minutes", "Hours", "Days"])
        row_c.addWidget(self.avg_unit)

        self.btn_parse_join = QPushButton("Parse Met + Join to PNSD")
        self.btn_parse_join.clicked.connect(self._parse_and_join)
        row_c.addWidget(self.btn_parse_join)
        met_layout.addLayout(row_c)

        root.addWidget(met_box)

        sec_box = QGroupBox("2) Wind Sectors & Plot")
        sec_layout = QVBoxLayout(sec_box)

        row_d = QHBoxLayout()
        row_d.addWidget(QLabel("Sectors (deg):"))
        self.sector_input = QLineEdit("0-45,45-90,90-135,135-180,180-225,225-270,270-315,315-360")
        row_d.addWidget(self.sector_input)
        self.chk_logy = QCheckBox("Log10 Y")
        self.chk_logy.setChecked(False)
        self.chk_logy.stateChanged.connect(self._apply_distribution_axis_scale)
        row_d.addWidget(self.chk_logy)
        self.btn_plot = QPushButton("Build Wind Plots")
        self.btn_plot.clicked.connect(self._build_wind_plots)
        row_d.addWidget(self.btn_plot)

        self.btn_export_plots = QPushButton("Export Plots")
        self.btn_export_plots.clicked.connect(self._export_plots)
        row_d.addWidget(self.btn_export_plots)

        self.btn_export_data = QPushButton("Export Data")
        self.btn_export_data.clicked.connect(self._export_data)
        row_d.addWidget(self.btn_export_data)
        sec_layout.addLayout(row_d)

        self.status_lbl = QLabel("Load PNSD and met data to begin.")
        sec_layout.addWidget(self.status_lbl)
        root.addWidget(sec_box)

        self.fig = Figure(figsize=(12, 5))
        self.canvas = FigureCanvasQTAgg(self.fig)
        root.addWidget(self.canvas, stretch=1)

        # Red outline while a box holds something unusable.
        validate_number(self.avg_val, minimum=1, integer=True)



    def _on_date_col_changed(self, text: str):
        self.custom_date_col.setVisible(text == "Custom...")

    def _on_fmt_changed(self, idx: int):
        disp = self.fmt_combo.itemText(idx)
        val = next(v for d, v in DATE_FORMAT_OPTIONS if d == disp)
        self.custom_fmt.setVisible(val == "custom")

    def _browse_met_file(self):
        path, _ = get_open_file_name(self, "Select Met File", "", "Data (*.csv *.xlsx *.xls *.txt *.tsv *.dat)")
        if not path:
            return

        try:
            p = Path(path)
            if p.suffix.lower() in (".xlsx", ".xls"):
                raw = pd.read_excel(path, dtype=str)
            else:
                raw = pd.read_csv(path, dtype=str)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Failed to read met file:\n{exc}")
            return

        self.met_raw_df = raw
        self.met_path_lbl.setText(Path(path).name)

        cols = [str(c) for c in raw.columns]
        self.wd_combo.clear(); self.wd_combo.addItems(cols)
        self.ws_combo.clear(); self.ws_combo.addItems(cols)

        lower_cols = {c.lower(): c for c in cols}
        for candidate in ["wd", "wind_dir", "wind_direction", "wdir", "direction"]:
            if candidate in lower_cols:
                self.wd_combo.setCurrentText(lower_cols[candidate])
                break
        for candidate in ["ws", "wind_speed", "windspeed", "speed"]:
            if candidate in lower_cols:
                self.ws_combo.setCurrentText(lower_cols[candidate])
                break

        date_guess = None
        for c in cols:
            if c.strip().lower() in [x.strip().lower() for x in DATE_COLUMN_OPTIONS if x != "Custom..."]:
                date_guess = c
                break
        if date_guess:
            self.date_col_combo.setCurrentText(date_guess if date_guess in DATE_COLUMN_OPTIONS else "Custom...")
            if date_guess not in DATE_COLUMN_OPTIONS:
                self.custom_date_col.setText(date_guess)

        self.status_lbl.setText("Met file loaded. Confirm columns and click 'Parse Met + Join to PNSD'.")

    def _set_wd_column(self):
        self._wd_col = self.wd_combo.currentText().strip() if self.wd_combo.count() else None
        if self._wd_col:
            self.status_lbl.setText(f"WD column set to: {self._wd_col}")

    def _set_ws_column(self):
        self._ws_col = self.ws_combo.currentText().strip() if self.ws_combo.count() else None
        if self._ws_col:
            self.status_lbl.setText(f"WS column set to: {self._ws_col}")

    def _parse_and_join(self):
        if self.df is None or self.diams is None:
            QMessageBox.warning(self, "No PNSD Data", "Load and confirm a PNSD dataframe first.")
            return
        if self.met_raw_df is None:
            QMessageBox.warning(self, "No Met Data", "Load a met file first.")
            return

        met = self.met_raw_df.copy()

        date_col_choice = self.date_col_combo.currentText()
        date_col = self.custom_date_col.text().strip() if date_col_choice == "Custom..." else date_col_choice
        col_match = [c for c in met.columns if str(c).strip().lower() == date_col.strip().lower()]
        if not col_match:
            QMessageBox.warning(self, "Date Column", f"Date column '{date_col}' not found in met file.")
            return
        dt_col = col_match[0]

        fmt_disp = self.fmt_combo.currentText()
        fmt_val = next(v for d, v in DATE_FORMAT_OPTIONS if d == fmt_disp)
        date_fmt = self.custom_fmt.text().strip() if fmt_val == "custom" else fmt_val

        met[dt_col] = parse_datetime_series(met[dt_col], date_fmt)
        met = met.dropna(subset=[dt_col]).copy()

        tz = self.tz_input.text().strip() or "UTC"
        try:
            if met[dt_col].dt.tz is None:
                met[dt_col] = met[dt_col].dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
            else:
                met[dt_col] = met[dt_col].dt.tz_convert(tz)
        except Exception as exc:
            QMessageBox.warning(self, "Timezone", f"Timezone handling failed:\n{exc}")
            return

        met = met.dropna(subset=[dt_col]).set_index(dt_col).sort_index()

        if not self._wd_col:
            self._wd_col = self.wd_combo.currentText().strip() if self.wd_combo.count() else None
        if not self._ws_col:
            self._ws_col = self.ws_combo.currentText().strip() if self.ws_combo.count() else None
        if not self._wd_col or not self._ws_col:
            QMessageBox.warning(self, "WD/WS Columns", "Set wind direction and wind speed columns.")
            return
        if self._wd_col not in met.columns or self._ws_col not in met.columns:
            QMessageBox.warning(self, "WD/WS Columns", "Selected WD/WS columns not found in met data.")
            return

        # A met export with two columns of the same name gives back a frame with
        # more columns than were asked for, which used to fail with an opaque
        # length mismatch and no message at all.
        duplicated = [c for c in (self._wd_col, self._ws_col) if list(met.columns).count(c) > 1]
        if duplicated:
            QMessageBox.warning(
                self, "Repeated column name",
                f"The met file has more than one column called '{duplicated[0]}'. "
                "Rename them so each is unique, then load the file again.")
            return

        met = met[[self._wd_col, self._ws_col]].copy()
        met.columns = ["WD", "WS"]
        met["WD"] = pd.to_numeric(met["WD"], errors="coerce")
        met["WS"] = pd.to_numeric(met["WS"], errors="coerce")

        usable = met.dropna(subset=["WD", "WS"])
        if usable.empty:
            unreadable = "wind direction" if met["WD"].isna().all() else "wind speed"
            QMessageBox.warning(
                self, "No usable wind data",
                f"No row has both a direction and a speed: the {unreadable} column "
                f"('{self._wd_col if unreadable.startswith('wind d') else self._ws_col}') "
                "holds no numbers. Check you picked the right columns.")
            return
        if len(usable) < len(met):
            self.status_lbl.setText(f"Dropped {len(met) - len(usable)} met row(s) with no direction or speed.")

        met = usable
        met["WD"] = np.mod(met["WD"], 360.0)
        met["WS"] = np.clip(met["WS"], 0, None)

        resample_rule = None
        val = self.avg_val.text().strip()
        unit = self.avg_unit.currentText()
        if val.isdigit():
            resample_rule = f"{val}min" if unit == "Minutes" else (f"{val}h" if unit == "Hours" else f"{val}D")
        if resample_rule:
            met = self._vector_resample_wind(met, resample_rule)
            if met.empty:
                QMessageBox.warning(self, "Averaging emptied the met data",
                                    f"Averaging to {resample_rule} left no complete intervals. "
                                    "Try a longer averaging period, or none.")
                return

        self.met_df = met
        self._join_met_with_pnsd()

    def _vector_resample_wind(self, met: pd.DataFrame, rule: str) -> pd.DataFrame:
        def agg_one(g: pd.DataFrame) -> pd.Series:
            if g.empty:
                return pd.Series({"WD": np.nan, "WS": np.nan})
            ws = g["WS"].to_numpy(dtype=float)
            wd = g["WD"].to_numpy(dtype=float)
            rad = np.deg2rad(np.mod(wd, 360.0))

            w = np.clip(ws, 0, None)
            if np.nansum(w) > 0:
                s = np.nansum(np.sin(rad) * w) / np.nansum(w)
                c = np.nansum(np.cos(rad) * w) / np.nansum(w)
            else:
                s = np.nanmean(np.sin(rad))
                c = np.nanmean(np.cos(rad))

            wd_mean = (np.degrees(np.arctan2(s, c)) + 360.0) % 360.0
            ws_mean = np.nanmean(ws)
            return pd.Series({"WD": wd_mean, "WS": ws_mean})

        out = met.resample(rule).apply(agg_one)
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = [c[0] for c in out.columns]
        return out.dropna(subset=["WD", "WS"])

    def _join_met_with_pnsd(self):
        pnsd = self.df.copy()
        met = self.met_df.copy()

        # merge_asof compares the two time columns directly, and on empty frames
        # their dtypes need not match, which fails with an unreadable message.
        if met.empty or pnsd.empty:
            QMessageBox.warning(self, "Nothing to join",
                                "One of the two datasets is empty after parsing. "
                                "Check the met file's date column, format and timezone.")
            return

        if pnsd.index.tz is None and met.index.tz is not None:
            pnsd.index = pnsd.index.tz_localize(met.index.tz)
        elif pnsd.index.tz is not None and met.index.tz is None:
            met.index = met.index.tz_localize(pnsd.index.tz)
        elif pnsd.index.tz is not None and met.index.tz is not None and str(pnsd.index.tz) != str(met.index.tz):
            met.index = met.index.tz_convert(pnsd.index.tz)

        p = pnsd.sort_index().reset_index().rename(columns={pnsd.index.name or "index": "datetime"})
        m = met.sort_index().reset_index().rename(columns={met.index.name or "index": "datetime"})

        if len(pnsd.index) > 1:
            tol = pd.Series(pnsd.index).diff().median()
            if pd.isna(tol) or tol <= pd.Timedelta(0):
                tol = pd.Timedelta("30min")
        else:
            tol = pd.Timedelta("30min")

        merged = pd.merge_asof(
            p,
            m,
            on="datetime",
            direction="nearest",
            tolerance=tol,
        )

        merged = merged.dropna(subset=["WD", "WS"]).set_index("datetime")

        if merged.empty:
            QMessageBox.warning(self, "Join", "Join produced no overlapping rows. Check timezone and averaging settings.")
            return

        self.joined_df = merged
        self.status_lbl.setText(f"Joined rows: {len(self.joined_df)}. Ready to build wind plots.")
        self._build_wind_plots()

    def _parse_sectors(self, text: str):
        sectors = []
        for token in [x.strip() for x in text.split(",") if x.strip()]:
            parts = [p.strip() for p in token.split("-")]
            if len(parts) != 2:
                continue
            try:
                a, b = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            sectors.append((a % 360.0, b % 360.0 if b != 360 else 360.0))
        return sectors

    def _sector_label(self, a: float, b: float) -> str:
        return f"{int(a)}-{int(b)}"

    def _angle_in_sector(self, wd: np.ndarray, a: float, b: float) -> np.ndarray:
        wd = np.mod(wd, 360.0)
        if b == 360.0 and a < b:
            return (wd >= a) | (wd < 0.000001)
        if a < b:
            return (wd >= a) & (wd < b)
        return (wd >= a) | (wd < b)

    def _build_wind_plots(self):
        if self.joined_df is None or self.joined_df.empty:
            self.status_lbl.setText("No joined data yet. Parse met data and join first.")
            return

        sectors = self._parse_sectors(self.sector_input.text().strip())
        if not sectors:
            QMessageBox.warning(self, "Sectors", "Could not parse sectors. Example: 0-45,45-90,...,315-360")
            return

        df = self.joined_df.copy()
        diam_cols = [c for c in df.columns if isinstance(c, (float, int, np.floating, np.integer))]
        if not diam_cols:
            try:
                diam_cols = [c for c in df.columns if float(c) > 0]
            except Exception:
                diam_cols = []
        if not diam_cols:
            QMessageBox.warning(self, "PNSD", "No numeric diameter columns found in joined dataframe.")
            return

        self.fig.clear()
        ax_rose = self.fig.add_subplot(121, projection="polar")
        ax_dist = self.fig.add_subplot(122)

        dlogdp = dlogdp_per_bin(np.array(diam_cols, dtype=float))
        total_n = integrate_pnsd(df[diam_cols].to_numpy(dtype=float), dlogdp)
        df["_TOTAL_N"] = total_n

        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(sectors))))
        self._sector_colors = colors
        self._rose_patches = []
        self._sector_lines = {}
        self._sector_labels = []

        stats_rows = []
        pnsd_rows = []

        for i, (a, b) in enumerate(sectors):
            mask = self._angle_in_sector(df["WD"].to_numpy(dtype=float), a, b)
            sub = df[mask]
            label = self._sector_label(a, b)
            self._sector_labels.append(label)

            mean_total_n = np.nanmean(sub["_TOTAL_N"].to_numpy(dtype=float)) if len(sub) else np.nan
            mean_ws = np.nanmean(sub["WS"].to_numpy(dtype=float)) if len(sub) else np.nan
            stats_rows.append({
                "sector": label,
                "start_deg": a,
                "end_deg": b,
                "n_rows": int(len(sub)),
                "mean_total_n": mean_total_n,
                "mean_ws": mean_ws,
            })

            width_deg = (b - a) if b > a else (360.0 - a + b)
            if b == 360.0 and a < b:
                width_deg = b - a
            theta_center = np.deg2rad((a + width_deg / 2.0) % 360.0)
            bar = ax_rose.bar(
                theta_center,
                np.nanmean(sub["_TOTAL_N"].to_numpy(dtype=float)) if len(sub) else 0.0,
                width=np.deg2rad(width_deg),
                bottom=0.0,
                color=colors[i],
                edgecolor="black",
                alpha=0.65,
                linewidth=0.8,
            )[0]
            self._rose_patches.append(bar)

            if len(sub) > 0:
                mean_dist = sub[diam_cols].mean(axis=0).to_numpy(dtype=float)
                line, = ax_dist.plot(np.array(diam_cols, dtype=float), mean_dist, color=colors[i], alpha=0.45, lw=1.6, label=label)
                pnsd_rows.append(pd.Series(mean_dist, index=np.array(diam_cols, dtype=float), name=label))
            else:
                line, = ax_dist.plot([], [], color=colors[i], alpha=0.25, lw=1.2, label=label)
                pnsd_rows.append(pd.Series(np.nan, index=np.array(diam_cols, dtype=float), name=label))
            self._sector_lines[label] = line

        self._sector_stats_df = pd.DataFrame(stats_rows)
        self._sector_pnsd_df = pd.DataFrame(pnsd_rows)
        self._sector_pnsd_df.index.name = "sector"

        ax_rose.set_theta_zero_location("N")
        ax_rose.set_theta_direction(-1)
        ax_rose.set_title("Pollutant Rose (Mean Total N by Wind Sector)")

        ax_dist.set_xscale("log")
        ax_dist.set_yscale("linear")
        ax_dist.set_xlabel("Diameter (nm)")
        ax_dist.set_ylabel("Mean dN/dlogDp")
        ax_dist.set_title("Mean Size Distribution by Wind Sector")
        ax_dist.legend(fontsize=8, loc="best")

        if self._hover_cid is not None:
            try:
                self.canvas.mpl_disconnect(self._hover_cid)
            except Exception:
                pass
        self._hover_cid = self.canvas.mpl_connect("motion_notify_event", self._on_hover_sector)

        self.fig.tight_layout()
        self.canvas.draw()

    def _ensure_sector_products(self) -> bool:
        if self.joined_df is None or self.joined_df.empty:
            QMessageBox.warning(self, "Export", "No joined wind/PNSD data to export.")
            return False
        if self._sector_stats_df is None or self._sector_pnsd_df is None:
            self._build_wind_plots()
        return self._sector_stats_df is not None and self._sector_pnsd_df is not None

    def _build_export_rose_figure(self) -> Figure:
        fig_r = Figure(figsize=(6, 5))
        ax_r = fig_r.add_subplot(111, projection="polar")
        colors = self._sector_colors if self._sector_colors is not None else plt.cm.tab10(np.linspace(0, 1, max(1, len(self._sector_stats_df))))
        for i, row in self._sector_stats_df.iterrows():
            a = float(row["start_deg"])
            b = float(row["end_deg"])
            width_deg = (b - a) if b > a else (360.0 - a + b)
            if b == 360.0 and a < b:
                width_deg = b - a
            theta_center = np.deg2rad((a + width_deg / 2.0) % 360.0)
            val = float(row["mean_total_n"]) if pd.notna(row["mean_total_n"]) else 0.0
            ax_r.bar(
                theta_center,
                val,
                width=np.deg2rad(width_deg),
                bottom=0.0,
                color=colors[i],
                edgecolor="black",
                alpha=0.95,
                linewidth=1.2,
            )
        ax_r.set_theta_zero_location("N")
        ax_r.set_theta_direction(-1)
        ax_r.set_title("Pollutant Rose (Mean Total N by Wind Sector)")
        fig_r.tight_layout()
        return fig_r

    def _build_export_pnsd_figure(self) -> Figure:
        fig_p = Figure(figsize=(7, 5))
        ax_p = fig_p.add_subplot(111)
        colors = self._sector_colors if self._sector_colors is not None else plt.cm.tab10(np.linspace(0, 1, max(1, len(self._sector_pnsd_df))))
        diam_values = np.array(self._sector_pnsd_df.columns, dtype=float)
        for i, (sector, row) in enumerate(self._sector_pnsd_df.iterrows()):
            y = row.to_numpy(dtype=float)
            if np.all(np.isnan(y)):
                continue
            ax_p.plot(diam_values, y, color=colors[i], lw=2.3, alpha=0.95, label=str(sector))
        ax_p.set_xscale("log")
        ax_p.set_yscale("log" if self.chk_logy.isChecked() else "linear")
        ax_p.set_xlabel("Diameter (nm)")
        ax_p.set_ylabel("Mean dN/dlogDp")
        ax_p.set_title("Mean Size Distribution by Wind Sector")
        ax_p.legend(fontsize=8, loc="best")
        fig_p.tight_layout()
        return fig_p

    def _export_plots(self):
        if not self._ensure_sector_products():
            return

        # Same workflow as other panels: open export dialogs with custom size + manual save.
        fig_r = self._build_export_rose_figure()
        canvas_r = FigureCanvasQTAgg(fig_r)
        dlg_r = ExportDialog("Wind Rose", canvas_r, fig_r, lambda: None, self)
        dlg_r.exec()

        fig_p = self._build_export_pnsd_figure()
        canvas_p = FigureCanvasQTAgg(fig_p)
        dlg_p = ExportDialog("PNSD by Sector", canvas_p, fig_p, lambda: None, self)
        dlg_p.exec()

    def _export_data(self):
        if not self._ensure_sector_products():
            return

        base, _ = get_save_file_name(self, "Export Wind Data (Base Name)", "wind_analysis", "CSV (*.csv)")
        if not base:
            return
        base_path = Path(base)
        stem = base_path.stem
        parent = base_path.parent

        stats_path = parent / f"{stem}_sector_summary.csv"
        pnsd_path = parent / f"{stem}_sector_pnsd.csv"

        self._sector_stats_df.to_csv(stats_path, index=False)
        self._sector_pnsd_df.to_csv(pnsd_path)

        QMessageBox.information(self, "Export Data", f"Saved:\n{stats_path}\n{pnsd_path}")

    def _apply_distribution_axis_scale(self):
        if self.fig is None or not self.fig.axes:
            return

        # By construction, the second axis is the sector-overlaid size distribution.
        if len(self.fig.axes) < 2:
            return
        ax_dist = self.fig.axes[1]

        if self.chk_logy.isChecked():
            ax_dist.set_yscale("log")
        else:
            ax_dist.set_yscale("linear")

        self.canvas.draw_idle()

    def _on_hover_sector(self, event):
        if not self._rose_patches:
            return

        active_idx = None
        if event.inaxes is not None and event.inaxes.name == "polar":
            for i, p in enumerate(self._rose_patches):
                contains, _ = p.contains(event)
                if contains:
                    active_idx = i
                    break

        for i, p in enumerate(self._rose_patches):
            if i == active_idx:
                p.set_alpha(1.0)
                p.set_linewidth(2.0)
            else:
                p.set_alpha(0.35)
                p.set_linewidth(0.8)

        for i, label in enumerate(self._sector_labels):
            line = self._sector_lines.get(label)
            if line is None:
                continue
            if i == active_idx:
                line.set_alpha(1.0)
                line.set_linewidth(3.0)
                line.set_zorder(10)
            else:
                line.set_alpha(0.25)
                line.set_linewidth(1.2)
                line.set_zorder(1)

        self.canvas.draw_idle()

    def load_data(self, data_file):
        self.df = data_file.df.copy()
        self.diams = np.array(data_file.diameters, dtype=float)
        self.status_lbl.setText("PNSD data loaded. Now load met data and parse/join.")
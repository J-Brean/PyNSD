from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from PyQt6.QtCore import QDate
from PyQt6.QtWidgets import (QWidget,
                             QVBoxLayout,
                             QHBoxLayout,
                             QLabel,
                             QPushButton,
                             QLineEdit,
                             QComboBox,
                             QGroupBox,
                             QMessageBox,
                             QDialog,
                             QDateEdit)

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


class PollutionFlagPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.df: pd.DataFrame | None = None
        self.diams: np.ndarray | None = None

        self.aux_raw_df: pd.DataFrame | None = None
        self.aux_df: pd.DataFrame | None = None
        self.joined_df: pd.DataFrame | None = None

        self._pol1_col: str | None = None
        self._pol2_col: str | None = None
        self._flag_mask: np.ndarray | None = None

        self._diam_cols: list[float] = []
        self._day_positions: np.ndarray | None = None
        self._ax_top = None
        self._ax_n = None   # twinx for Total N — must be included in click test
        self._ax_mid = None
        self._ax_bot = None
        self._click_cid = None

        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        title = QLabel("Pollution Flag Panel")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        root.addWidget(title)

        aux_box = QGroupBox("1) Load & Align Auxiliary Pollutant Data")
        aux_layout = QVBoxLayout(aux_box)

        row_a = QHBoxLayout()
        self.btn_load_aux = QPushButton("+ Load Auxiliary File")
        self.btn_load_aux.clicked.connect(self._browse_aux_file)
        row_a.addWidget(self.btn_load_aux)
        self.aux_path_lbl = QLabel("No auxiliary file loaded")
        row_a.addWidget(self.aux_path_lbl)
        row_a.addStretch()
        aux_layout.addLayout(row_a)

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
        aux_layout.addLayout(row_b)

        row_c = QHBoxLayout()
        row_c.addWidget(QLabel("Pollutant 1:"))
        self.pol1_combo = QComboBox()
        row_c.addWidget(self.pol1_combo)
        row_c.addWidget(QLabel("Pollutant 2:"))
        self.pol2_combo = QComboBox()
        row_c.addWidget(self.pol2_combo)

        row_c.addWidget(QLabel("Avg:"))
        self.avg_val = QLineEdit("")
        self.avg_val.setPlaceholderText("Val")
        self.avg_val.setFixedWidth(42)
        row_c.addWidget(self.avg_val)
        self.avg_unit = QComboBox()
        self.avg_unit.addItems(["Minutes", "Hours", "Days"])
        row_c.addWidget(self.avg_unit)

        self.btn_parse_join = QPushButton("Parse Aux + Join to PNSD")
        self.btn_parse_join.clicked.connect(self._parse_and_join)
        row_c.addWidget(self.btn_parse_join)
        aux_layout.addLayout(row_c)
        root.addWidget(aux_box)

        ctrl_box = QGroupBox("2) Flagging & Export")
        ctrl_layout = QVBoxLayout(ctrl_box)

        row_d = QHBoxLayout()
        row_d.addWidget(QLabel("Day:"))
        self.day_picker = QDateEdit()
        self.day_picker.setCalendarPopup(True)
        self.day_picker.dateChanged.connect(self._build_plots)
        row_d.addWidget(self.day_picker)

        self.btn_prev_day = QPushButton("◀ Day")
        self.btn_prev_day.clicked.connect(lambda: self.day_picker.setDate(self.day_picker.date().addDays(-1)))
        row_d.addWidget(self.btn_prev_day)

        self.btn_next_day = QPushButton("Day ▶")
        self.btn_next_day.clicked.connect(lambda: self.day_picker.setDate(self.day_picker.date().addDays(1)))
        row_d.addWidget(self.btn_next_day)

        self.btn_build = QPushButton("Build / Refresh Plots")
        self.btn_build.clicked.connect(self._build_plots)
        row_d.addWidget(self.btn_build)

        self.btn_clear_flags = QPushButton("Clear Flags")
        self.btn_clear_flags.clicked.connect(self._clear_flags)
        row_d.addWidget(self.btn_clear_flags)

        self.btn_export_plot = QPushButton("Export Plot")
        self.btn_export_plot.clicked.connect(self._export_plot)
        row_d.addWidget(self.btn_export_plot)

        ctrl_layout.addLayout(row_d)

        row_e = QHBoxLayout()
        row_e.addWidget(QLabel("Export Data Mode:"))
        self.export_mode_combo = QComboBox()
        self.export_mode_combo.addItems([
            "Datetime + Flag (0/1)",
            "PNSD + Flag Column",
            "PNSD with Flagged Rows Removed",
        ])
        row_e.addWidget(self.export_mode_combo)

        self.btn_export_data = QPushButton("Export CSV")
        self.btn_export_data.clicked.connect(self._export_data)
        row_e.addWidget(self.btn_export_data)
        row_e.addStretch()
        ctrl_layout.addLayout(row_e)

        # --- Colour/plot controls (matching Summary panel style) ---
        row_f = QHBoxLayout()
        row_f.addWidget(QLabel("Colour Map:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["turbo", "viridis", "plasma", "inferno"])
        row_f.addWidget(self.cmap_combo)

        row_f.addWidget(QLabel("Min Colour:"))
        self.cbar_min = QLineEdit("1")
        self.cbar_min.setFixedWidth(60)
        row_f.addWidget(self.cbar_min)

        row_f.addWidget(QLabel("Max Colour (auto):"))
        self.cbar_max_lbl = QLabel("–")
        self.cbar_max_lbl.setStyleSheet("color: #555; font-style: italic;")
        row_f.addWidget(self.cbar_max_lbl)
        row_f.addStretch()
        ctrl_layout.addLayout(row_f)

        self.status_lbl = QLabel("Load and confirm PNSD data. Auxiliary pollutant data is optional.")
        ctrl_layout.addWidget(self.status_lbl)
        root.addWidget(ctrl_box)

        self.fig = Figure(figsize=(12, 7))
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

    def _browse_aux_file(self):
        path, _ = get_open_file_name(self, "Select Auxiliary File", "", "Data (*.csv *.xlsx *.xls *.txt *.tsv *.dat)")
        if not path:
            return

        try:
            p = Path(path)
            if p.suffix.lower() in (".xlsx", ".xls"):
                raw = pd.read_excel(path, dtype=str)
            else:
                raw = pd.read_csv(path, dtype=str)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Failed to read auxiliary file:\n{exc}")
            return

        self.aux_raw_df = raw
        self.aux_path_lbl.setText(Path(path).name)

        cols = [str(c) for c in raw.columns]
        self.pol1_combo.clear(); self.pol1_combo.addItem("NA"); self.pol1_combo.addItems(cols)
        self.pol2_combo.clear(); self.pol2_combo.addItem("NA"); self.pol2_combo.addItems(cols)
        if len(cols) >= 2:
            self.pol1_combo.setCurrentText(cols[0])
            self.pol2_combo.setCurrentText(cols[1])
        elif len(cols) == 1:
            self.pol1_combo.setCurrentText(cols[0])
            self.pol2_combo.setCurrentText("NA")

        date_guess = None
        for c in cols:
            if c.strip().lower() in [x.strip().lower() for x in DATE_COLUMN_OPTIONS if x != "Custom..."]:
                date_guess = c
                break
        if date_guess:
            self.date_col_combo.setCurrentText(date_guess if date_guess in DATE_COLUMN_OPTIONS else "Custom...")
            if date_guess not in DATE_COLUMN_OPTIONS:
                self.custom_date_col.setText(date_guess)

        self.status_lbl.setText("Auxiliary file loaded. Select pollutant columns, then parse/join.")

    def _parse_and_join(self):
        if self.df is None or self.diams is None:
            QMessageBox.warning(self, "No PNSD Data", "Load and confirm a PNSD dataframe first.")
            return
        # If no auxiliary data is loaded, proceed with just PNSD and flag as None.
        if self.aux_raw_df is None:
            self.aux_df = None
            self._pol1_col = None
            self._pol2_col = None
            self._join_aux_with_pnsd()
            self.status_lbl.setText("No auxiliary file loaded. Proceed with PNSD-only analysis.")
            return

        aux = self.aux_raw_df.copy()

        date_col_choice = self.date_col_combo.currentText()
        date_col = self.custom_date_col.text().strip() if date_col_choice == "Custom..." else date_col_choice
        col_match = [c for c in aux.columns if str(c).strip().lower() == date_col.strip().lower()]
        if not col_match:
            QMessageBox.warning(self, "Date Column", f"Date column '{date_col}' not found in auxiliary file.")
            return
        dt_col = col_match[0]

        fmt_disp = self.fmt_combo.currentText()
        fmt_val = next(v for d, v in DATE_FORMAT_OPTIONS if d == fmt_disp)
        date_fmt = self.custom_fmt.text().strip() if fmt_val == "custom" else fmt_val

        aux[dt_col] = parse_datetime_series(aux[dt_col], date_fmt)
        aux = aux.dropna(subset=[dt_col]).copy()

        tz = self.tz_input.text().strip() or "UTC"
        try:
            if aux[dt_col].dt.tz is None:
                aux[dt_col] = aux[dt_col].dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
            else:
                aux[dt_col] = aux[dt_col].dt.tz_convert(tz)
        except Exception as exc:
            QMessageBox.warning(self, "Timezone", f"Timezone handling failed:\n{exc}")
            return

        aux = aux.dropna(subset=[dt_col]).set_index(dt_col).sort_index()

        pol1 = self.pol1_combo.currentText().strip() if self.pol1_combo.count() else "NA"
        pol2 = self.pol2_combo.currentText().strip() if self.pol2_combo.count() else "NA"

        use_pol1 = (pol1 and pol1.upper() != "NA")
        use_pol2 = (pol2 and pol2.upper() != "NA")

        selected_cols = []
        if use_pol1:
            if pol1 not in aux.columns:
                QMessageBox.warning(self, "Pollutants", "Selected Pollutant 1 column was not found in auxiliary data.")
                return
            selected_cols.append(pol1)
        if use_pol2:
            if pol2 not in aux.columns:
                QMessageBox.warning(self, "Pollutants", "Selected Pollutant 2 column was not found in auxiliary data.")
                return
            selected_cols.append(pol2)

        # Selecting a repeated column name returns more columns than were asked
        # for, and renaming them then fails with no message at all.
        repeated = [c for c in selected_cols if list(aux.columns).count(c) > 1]
        if repeated:
            QMessageBox.warning(
                self, "Repeated column name",
                f"The auxiliary file has more than one column called '{repeated[0]}'. "
                "Rename them so each is unique, then load the file again.")
            return

        if selected_cols:
            aux = aux[selected_cols].copy()
            rename_map = {}
            if use_pol1:
                rename_map[pol1] = "POLLUTANT_1"
            if use_pol2:
                rename_map[pol2] = "POLLUTANT_2"
            aux = aux.rename(columns=rename_map)

            for slot, chosen in (("POLLUTANT_1", pol1), ("POLLUTANT_2", pol2)):
                if slot not in aux.columns:
                    continue
                aux[slot] = pd.to_numeric(aux[slot], errors="coerce")
                if aux[slot].isna().all():
                    QMessageBox.warning(
                        self, "Pollutant has no numbers",
                        f"Column '{chosen}' holds no numeric values, so nothing could be "
                        "flagged from it. Check you picked the right column, and that the "
                        "file does not use a comma as its decimal separator.")
                    return
        else:
            # No pollutant columns requested; keep only datetime index for alignment.
            aux = pd.DataFrame(index=aux.index)

        resample_rule = None
        val = self.avg_val.text().strip()
        unit = self.avg_unit.currentText()
        if val.isdigit():
            resample_rule = f"{val}min" if unit == "Minutes" else (f"{val}h" if unit == "Hours" else f"{val}D")
        if resample_rule:
            aux = aux.resample(resample_rule).mean().dropna(how="all")

        self.aux_df = aux
        self._pol1_col = pol1 if use_pol1 else None
        self._pol2_col = pol2 if use_pol2 else None
        self._join_aux_with_pnsd()

    def _join_aux_with_pnsd(self):
        if self.df is None:
            return
        # merge_asof compares the two time columns directly, and on empty frames
        # their dtypes need not match, which fails with an unreadable message.
        if self.aux_df is not None and (self.aux_df.empty or self.df.empty):
            QMessageBox.warning(self, "Nothing to join",
                                "One of the two datasets is empty after parsing. Check the "
                                "auxiliary file's date column, format and timezone.")
            return
        # If aux_df is None, just use PNSD data as-is for flagging.
        if self.aux_df is None:
            self.joined_df = self.df.copy()
            self._flag_mask = np.zeros(len(self.joined_df), dtype=bool)
            self._build_plots()
            return

        pnsd = self.df.copy()
        aux = self.aux_df.copy()

        if pnsd.index.tz is None and aux.index.tz is not None:
            pnsd.index = pnsd.index.tz_localize(aux.index.tz)
        elif pnsd.index.tz is not None and aux.index.tz is None:
            aux.index = aux.index.tz_localize(pnsd.index.tz)
        elif pnsd.index.tz is not None and aux.index.tz is not None and str(pnsd.index.tz) != str(aux.index.tz):
            aux.index = aux.index.tz_convert(pnsd.index.tz)

        p = pnsd.sort_index().reset_index().rename(columns={pnsd.index.name or "index": "datetime"})
        a = aux.sort_index().reset_index().rename(columns={aux.index.name or "index": "datetime"})

        if len(pnsd.index) > 1:
            tol = pd.Series(pnsd.index).diff().median()
            if pd.isna(tol) or tol <= pd.Timedelta(0):
                tol = pd.Timedelta("30min")
        else:
            tol = pd.Timedelta("30min")

        merged = pd.merge_asof(
            p,
            a,
            on="datetime",
            direction="nearest",
            tolerance=tol,
        )

        poll_cols = [c for c in ["POLLUTANT_1", "POLLUTANT_2"] if c in merged.columns]
        if poll_cols:
            merged = merged.dropna(subset=poll_cols, how="all")
        merged = merged.set_index("datetime")
        if merged.empty:
            QMessageBox.warning(self, "Join", "Join produced no overlapping rows. Check timezone/averaging settings.")
            return

        self.joined_df = merged
        self._diam_cols = [float(c) for c in merged.columns if isinstance(c, (float, int, np.floating, np.integer))]
        if not self._diam_cols:
            self._diam_cols = [float(c) for c in merged.columns if str(c).replace('.', '', 1).isdigit()]

        if not self._diam_cols:
            QMessageBox.warning(self, "PNSD", "No numeric diameter columns found after joining.")
            return

        self._flag_mask = np.zeros(len(self.joined_df), dtype=bool)

        dmin = self.joined_df.index.min().date()
        dmax = self.joined_df.index.max().date()
        self.day_picker.setMinimumDate(QDate(dmin.year, dmin.month, dmin.day))
        self.day_picker.setMaximumDate(QDate(dmax.year, dmax.month, dmax.day))
        self.day_picker.setDate(QDate(dmin.year, dmin.month, dmin.day))

        selected_pollutants = ", ".join([p for p in [self._pol1_col, self._pol2_col] if p]) or "none"
        self.status_lbl.setText(
            f"Joined rows: {len(self.joined_df)}. Pollutants: {selected_pollutants}. Click points in top panel to toggle flags."
        )
        self._build_plots()

    def _clear_flags(self):
        if self._flag_mask is None:
            return
        self._flag_mask[:] = False
        self._build_plots()

    def _build_plots(self):
        if self.joined_df is None or self.joined_df.empty:
            return
        if self._flag_mask is None or len(self._flag_mask) != len(self.joined_df):
            self._flag_mask = np.zeros(len(self.joined_df), dtype=bool)

        date_val = self.day_picker.date().toPyDate()
        day_mask = self.joined_df.index.date == date_val
        self._day_positions = np.where(day_mask)[0]

        if len(self._day_positions) == 0:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "No rows for selected day.", ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            self.canvas.draw()
            return

        day_df = self.joined_df.iloc[self._day_positions]
        day_flags = self._flag_mask[self._day_positions]

        dvals = np.array(self._diam_cols, dtype=float)
        z = day_df[self._diam_cols].to_numpy(dtype=float).T
        z = np.clip(z, 1e-6, None)
        # No masking — flagged rows are rendered dimmed via axvspan overlays instead.

        dlogdp = dlogdp_per_bin(dvals)
        total_n = integrate_pnsd(day_df[self._diam_cols].to_numpy(dtype=float), dlogdp)

        all_mean = self.joined_df[self._diam_cols].mean(axis=0).to_numpy(dtype=float)
        keep_mask = ~self._flag_mask
        if np.any(keep_mask):
            keep_mean = self.joined_df.iloc[keep_mask][self._diam_cols].mean(axis=0).to_numpy(dtype=float)
        else:
            keep_mean = np.full_like(all_mean, np.nan)

        # Save current y-limits so clicking doesn't reset the zoom level.
        _prev_ylim = None
        if self._ax_top is not None:
            try:
                _prev_ylim = self._ax_top.get_ylim()
            except Exception:
                pass

        # --- Colour limits ---
        try:
            v_min = float(self.cbar_min.text())
        except (ValueError, AttributeError):
            v_min = 1.0
        # Auto vmax: max of the *unflagged* PNSD values for the current day.
        unmasked_vals = z[:, ~day_flags] if day_flags.any() else z
        v_max = float(np.nanmax(unmasked_vals)) if unmasked_vals.size and np.isfinite(unmasked_vals).any() else v_min * 1000
        if v_max <= v_min:
            v_max = v_min * 10
        self.cbar_max_lbl.setText(f"{v_max:.1f}")

        try:
            cmap_name = self.cmap_combo.currentText()
        except AttributeError:
            cmap_name = "turbo"
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad('#bdbdbd')

        from matplotlib.colors import LogNorm

        self.fig.clear()
        # Left: PNSD contour (top) + pollutants (bottom).  Right: mean PNSD comparison.
        outer = self.fig.add_gridspec(1, 2, width_ratios=[2.2, 1.4], wspace=0.35)
        left_gs = outer[0, 0].subgridspec(2, 1, height_ratios=[2.8, 1.2], hspace=0.38)
        self._ax_top = self.fig.add_subplot(left_gs[0, 0])
        self._ax_bot = self.fig.add_subplot(left_gs[1, 0], sharex=self._ax_top)
        self._ax_mid = self.fig.add_subplot(outer[0, 1])

        t_nums = mdates.date2num(day_df.index.to_pydatetime())
        mesh = self._ax_top.pcolormesh(
            t_nums, dvals, z,
            shading='auto', cmap=cmap,
            norm=LogNorm(vmin=max(v_min, 1e-6), vmax=v_max)
        )
        self._ax_top.set_yscale('log')
        self._ax_top.set_ylabel('Diameter (nm)')
        self._ax_top.set_title('Daily PNSD  —  click any point to toggle row flag')

        # Inline colourbar (matches summary panel style)
        cb_ax = self._ax_top.inset_axes([0.02, 0.80, 0.25, 0.05])
        cb = self.fig.colorbar(mesh, cax=cb_ax, orientation='horizontal')
        cb.set_label(r'$\mathrm{dN/dlogD_p}$', size=8)
        cb_ax.xaxis.set_ticks_position('top')
        cb_ax.xaxis.set_label_position('top')
        cb_ax.tick_params(labelsize=7)
        cb_ax.patch.set_facecolor('white')
        cb_ax.patch.set_alpha(0.7)

        # twinx for Total N — sits naturally above the pcolormesh so the line is visible.
        self._ax_n = self._ax_top.twinx()
        self._ax_n.plot(day_df.index, total_n, color='black', lw=1.5, alpha=0.95, zorder=10)
        self._ax_n.set_ylabel('Total N')

        # Dim flagged timesteps with a semi-transparent grey overlay instead of hiding them.
        if np.any(day_flags):
            flag_t = t_nums[day_flags]
            # Estimate half a bin-width for the span edges.
            half_bin = (t_nums[1] - t_nums[0]) * 0.5 if len(t_nums) > 1 else 0.0
            for ft in flag_t:
                self._ax_top.axvspan(ft - half_bin, ft + half_bin,
                                     ymin=0, ymax=1, color='white', alpha=0.80, zorder=5)

        # Restore y-limits so clicking doesn't auto-rescale the contour.
        if _prev_ylim is not None:
            self._ax_top.set_ylim(_prev_ylim)

        n_flag = int(np.sum(self._flag_mask))
        self._ax_mid.plot(dvals, np.clip(all_mean, 1e-8, None), color='#2b6cb0', lw=2.3, label='All rows')
        self._ax_mid.plot(dvals, np.clip(keep_mean, 1e-8, None), color='#c53030', lw=2.1, label='Unflagged only')
        self._ax_mid.set_xscale('log')
        self._ax_mid.set_yscale('log')
        self._ax_mid.set_xlabel('Diameter (nm)')
        self._ax_mid.set_ylabel('Mean dN/dlogDp')
        self._ax_mid.set_title(f'Mean PNSD Comparison\nFlagged: {n_flag}/{len(self.joined_df)}')
        self._ax_mid.legend(fontsize=8, loc='best')

        plotted_any = False
        if 'POLLUTANT_1' in day_df.columns:
            self._ax_bot.plot(day_df.index, day_df['POLLUTANT_1'].to_numpy(dtype=float), color='#1f77b4', lw=1.5,
                              label=(self._pol1_col or 'Pollutant 1'))
            plotted_any = True
        if 'POLLUTANT_2' in day_df.columns:
            self._ax_bot.plot(day_df.index, day_df['POLLUTANT_2'].to_numpy(dtype=float), color='#ff7f0e', lw=1.5,
                              label=(self._pol2_col or 'Pollutant 2'))
            plotted_any = True
        self._ax_bot.set_ylabel('Pollutants')
        self._ax_bot.set_title('Aligned Pollutant Time Series')
        if plotted_any:
            self._ax_bot.legend(fontsize=8, loc='best')
        else:
            self._ax_bot.text(0.5, 0.5, 'No pollutant selected.', ha='center', va='center',
                              transform=self._ax_bot.transAxes)
        self._ax_bot.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        if self._click_cid is not None:
            try:
                self.canvas.mpl_disconnect(self._click_cid)
            except Exception:
                pass
        self._click_cid = self.canvas.mpl_connect('button_press_event', self._on_click_pnsd)

        self.fig.autofmt_xdate(rotation=35)
        self.fig.tight_layout()
        self.canvas.draw()

    def _on_click_pnsd(self, event):
        if self.joined_df is None or self._day_positions is None or len(self._day_positions) == 0:
            return
        # Accept clicks on either the contour axes OR its twinx (Total N), since the twinx
        # sits on top and can intercept mouse events even when the user aims at the PNSD.
        if event.inaxes not in (self._ax_top, self._ax_n):
            return
        if event.xdata is None:
            return

        day_idx = self.joined_df.index[self._day_positions]
        if len(day_idx) == 0:
            return

        t_nums = mdates.date2num(day_idx.to_pydatetime())
        nearest_local = int(np.argmin(np.abs(t_nums - event.xdata)))
        nearest_global = int(self._day_positions[nearest_local])
        self._flag_mask[nearest_global] = ~self._flag_mask[nearest_global]
        self._build_plots()

    def _export_plot(self):
        if self.joined_df is None or self.joined_df.empty:
            QMessageBox.warning(self, "Export Plot", "No plotted data available.")
            return
        fig_tmp = Figure(figsize=(12, 7))
        canvas_tmp = FigureCanvasQTAgg(fig_tmp)

        orig_fig = self.fig
        orig_canvas = self.canvas
        self.fig = fig_tmp
        self.canvas = canvas_tmp
        self._build_plots()

        def _restore():
            self.fig = orig_fig
            self.canvas = orig_canvas
            self._build_plots()

        dlg = ExportDialog("Pollution Flag Panel", canvas_tmp, fig_tmp, _restore, self)
        dlg.exec()

    def _export_data(self):
        if self.joined_df is None or self.joined_df.empty or self._flag_mask is None:
            QMessageBox.warning(self, "Export CSV", "No joined data available.")
            return

        mode = self.export_mode_combo.currentText()
        path, _ = get_save_file_name(self, "Export CSV", "", "CSV (*.csv)")
        if not path:
            return

        out_df = None
        if mode == "Datetime + Flag (0/1)":
            out_df = pd.DataFrame({
                'datetime': self.joined_df.index,
                'flag': self._flag_mask.astype(int)
            })
        elif mode == "PNSD + Flag Column":
            out_df = self.joined_df[self._diam_cols].copy()
            out_df['flag'] = self._flag_mask.astype(int)
            out_df = out_df.reset_index()
        elif mode == "PNSD with Flagged Rows Removed":
            out_df = self.joined_df.loc[~self._flag_mask, self._diam_cols].copy().reset_index()

        if out_df is None:
            QMessageBox.warning(self, "Export CSV", "Could not build output dataframe.")
            return

        out_df.to_csv(path, index=False)
        QMessageBox.information(self, "Export CSV", "CSV exported successfully.")

    def load_data(self, data_file):
        self.df = data_file.df.copy()
        self.diams = np.array(data_file.diameters, dtype=float)
        self.aux_df = None
        self.joined_df = None
        self._flag_mask = None
        self.status_lbl.setText("PNSD loaded. Now load auxiliary data and choose two pollutant columns.")

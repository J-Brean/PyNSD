from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import re
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator                             # For the CDF spline regridder

from utils.calculations import log_bin_edges

_TOKEN_MAP = [
    ("yyyy", "%Y"), ("MM", "%m"), ("dd", "%d"),
    ("HH", "%H"), ("mm", "%M"), ("ss", "%S"),
]

# Display order is the menu order: auto first, then year-first, day-first,
# month-first.  Panels populate their combo boxes straight from this list.
DATE_FORMAT_OPTIONS = [
    ("Auto-detect  —  infer from the file",         "auto"),
    ("YYYY/MM/DD HH:mm:ss  —  2021/01/31 00:15:00", "yyyy/MM/dd HH:mm:ss"),
    ("YYYY-MM-DD HH:mm:ss  —  2021-01-31 00:15:00", "yyyy-MM-dd HH:mm:ss"),
    ("YYYY-MM-DD HH:mm  —  2021-01-31 00:15",       "yyyy-MM-dd HH:mm"),
    ("YYYY/MM/DD  —  2021/01/31",                   "yyyy/MM/dd"),
    ("DD/MM/YYYY HH:mm  —  31/01/2021 00:15",       "dd/MM/yyyy HH:mm"),
    ("DD/MM/YYYY HH:mm:ss  —  31/01/2021 00:15:00", "dd/MM/yyyy HH:mm:ss"),
    ("DD-MM-YYYY HH:mm  —  31-01-2021 00:15",       "dd-MM-yyyy HH:mm"),
    ("DD-MM-YYYY HH:mm:ss  —  31-01-2021 00:15:00", "dd-MM-yyyy HH:mm:ss"),
    ("DD/MM/YYYY  —  31/01/2021",                   "dd/MM/yyyy"),
    ("MM/DD/YYYY HH:mm  —  01/31/2021 00:15",       "MM/dd/yyyy HH:mm"),
    ("MM/DD/YYYY HH:mm:ss  —  01/31/2021 00:15:00", "MM/dd/yyyy HH:mm:ss"),
    ("MM/DD/YYYY  —  01/31/2021",                   "MM/dd/yyyy"),
    ("Custom...",                                   "custom"),
]

DATE_COLUMN_OPTIONS = [
    "Auto-detect", "date", "Date", "DateTime", "datetime",
    "DATE", "Time", "time", "timestamp", "Timestamp", "Custom...",
]
DEFAULT_DATE_COL = "Auto-detect"
DEFAULT_DATE_FMT = "auto"

# One source of truth for the NA handling choices, so the global panel and the
# per-file override panel cannot drift apart and silently mean different things.
NA_OPTIONS = [
    ("Drop Rows",      "drop"),
    ("Fill (Fwd/Bwd)", "ffill"),
    ("Interpolate",    "interpolate"),
    ("Fill Min (1.0)", "zero"),
]

# Anything outside this stays suspicious: mobility/optical instruments live well
# inside it, and a stray metadata column usually does not.
MIN_PLAUSIBLE_DP_NM, MAX_PLAUSIBLE_DP_NM = 0.3, 1.0e5

_UNIT_FACTORS_NM = {                                                         # header unit -> multiplier to nm
    "nm": 1.0, "nanometre": 1.0, "nanometer": 1.0,
    "um": 1e3, "µm": 1e3, "μm": 1e3, "micron": 1e3,
    "micrometre": 1e3, "micrometer": 1e3,
    "m": 1e9,
}

# Column labels carrying a number that is an index or a species, not a size:
# whole-label species names such as PM10, NO2 or O3, and index words anywhere.
_NOT_A_DIAMETER = re.compile(
    r"^\s*(?:pm|nox?|o|so|co|ch|bc|ec|oc|temp|rh|ws|wd|press)\s*[\d.]+\s*$"
    r"|(?:bin|chan|channel|index|sample|scan|record|row|no\.|#|unnamed)",
    re.I,
)

# "10.6", "Dp_10.6nm", "dN/dlogDp 10.6", "N(0.0106 um)", "10,6" ...
_DIAMETER_LABEL = re.compile(
    r"^[^0-9]*?(?P<num>\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?)\s*"
    r"(?P<unit>nanometre|nanometer|micrometre|micrometer|micron|nm|µm|μm|um|m)?\s*[\)\]]?\s*$",
    re.I,
)

def fmt_to_strptime(token_fmt: str) -> str:
    result = token_fmt
    for token, code in _TOKEN_MAP:
        result = result.replace(token, code)
    return result

def strip_time_tokens(token_fmt: str) -> str:
    """Return a date-only token format by removing trailing time tokens."""
    if not token_fmt:
        return token_fmt
    fmt = token_fmt.strip()
    # Remove common trailing time portions (e.g. " HH:mm:ss", " HH:mm")
    for suffix in (" HH:mm:ss", " HH:mm", "THH:mm:ss", "THH:mm"):
        if fmt.endswith(suffix):
            return fmt[: -len(suffix)].strip()
    return fmt

@dataclass
class DataFile:
    path: Path
    date_col: str = DEFAULT_DATE_COL
    date_fmt: str = DEFAULT_DATE_FMT
    df: Optional[pd.DataFrame] = None
    df_raw: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    warning: Optional[str] = None
    notes: list = field(default_factory=list)                # every assumption and every row dropped
    n_rows: int = 0
    n_bins: int = 0
    diameters: list = field(default_factory=list)
    date_min: Optional[pd.Timestamp] = None
    date_max: Optional[pd.Timestamp] = None
    time_res_minutes: Optional[float] = None

    @property
    def ok(self) -> bool: 
        return self.df is not None

    @property
    def status(self) -> str:
        if self.df is None: return "error"
        if self.warning: return "warning"
        return "ok"

    @property
    def size_str(self) -> str:
        if not self.path.exists(): return "In-Memory"           # Prevent disk check for merged files
        size = self.path.stat().st_size                         # Normal size check
        if size >= 1_048_576: return f"{size / 1_048_576:.1f} MB"
        return f"{size / 1024:.0f} KB"                          # Return standard format

    @property
    def date_range_str(self) -> str:
        if self.date_min is None or self.date_max is None: return ""
        return f"{self.date_min.strftime('%Y-%m-%d')} → {self.date_max.strftime('%Y-%m-%d')}"

    @property
    def diam_range_str(self) -> str:
        if not self.diameters: return ""
        return f"{self.diameters[0]:.1f} – {self.diameters[-1]:.1f} nm"
    
# --- Aerosol Physics & Manipulation Functions ---

def apply_qc_filter(df: pd.DataFrame, window: int, stdev_thresh: float, action: str):
    """Applies a rolling median filter and returns (cleaned_df, num_corrected, outliers_mask)."""
    rolling_median = df.rolling(window=window, center=True, min_periods=1).median()
    rolling_std = df.rolling(window=window, center=True, min_periods=1).std()
    
    outliers = np.abs(df - rolling_median) > (stdev_thresh * rolling_std)
    outliers |= (df < 0)
    
    num_corrected = int(outliers.sum().sum())                                # Count total modified data points
    
    if action == "na": df_clean = df.mask(outliers)
    else: df_clean = df.mask(outliers, rolling_median)
    
    return df_clean, num_corrected, outliers                                 # Return extra data for diagnostics

def regrid_pnsd_cdf(df: pd.DataFrame, diams: np.ndarray, cpd: float = 64.0) -> tuple[pd.DataFrame, np.ndarray]:
    """Regrids PNSD via a CDF spline, properly handling variable dlogDp from spliced datasets."""
    bounds = log_bin_edges(diams)                                            # Bin edges in log space
    dlogdp_array = np.diff(bounds)                                           # Exact variable bin widths
    
    dn = df.to_numpy() * dlogdp_array                                        # Convert dN/dlogDp to absolute N
    cdf_old = np.column_stack((np.zeros(len(dn)), np.cumsum(dn, axis=1)))    # Build cumulative sum starting at 0
    
    dlogdp_new = 1.0 / cpd                                                   # Calculate target uniform bin width
    new_bounds = np.arange(bounds[0], bounds[-1] + dlogdp_new, dlogdp_new)   # Generate new uniform edges
    log_diams_new = new_bounds[:-1] + dlogdp_new / 2.0                       # Get new midpoints
    new_diams = 10 ** log_diams_new                                          # Convert back to linear nm
    
    new_dn = np.zeros((len(df), len(new_diams)))                             # Initialise empty array
    
    for i in range(len(df)):                                                 
        spline = PchipInterpolator(bounds, cdf_old[i, :])                    # Fit monotonic spline to true boundaries
        cdf_eval = np.clip(spline(new_bounds), 0, cdf_old[i, -1])            # Evaluate CDF on new edges and clamp
        new_dn[i, :] = np.maximum(0, np.diff(cdf_eval))                      # Difference CDF to get N in new bins
        
    new_df = pd.DataFrame(new_dn / dlogdp_new, index=df.index, columns=new_diams) # Normalise back to dN/dlogDp
    return new_df, new_diams

def calculate_line_losses(diams_nm: np.ndarray, length_m: float, id_m: float, temp_k: float, flow_lpm: float) -> np.ndarray:
    q_m3_s = flow_lpm * 1.66667e-5
    diams_m = diams_nm * 1e-9
    
    kb = 1.380649e-23
    visc = 1.81e-5 * ((temp_k / 293.15) ** 1.5) * (393.15 / (temp_k + 120))
    mfp = 6.65e-8 * (temp_k / 293.15)
    
    knudsen = 2 * mfp / diams_m
    cc = 1 + knudsen * (1.142 + 0.558 * np.exp(-0.999 / knudsen))
    diff_coeff = (kb * temp_k * cc) / (3 * np.pi * visc * diams_m)
    
    mu = (diff_coeff * length_m) / q_m3_s
    
    penetration = np.ones_like(diams_m)
    mask1 = mu < 0.02
    mask2 = mu >= 0.02
    
    penetration[mask1] = 1 - 5.5 * (mu[mask1] ** (2/3)) + 3.77 * mu[mask1]
    penetration[mask2] = 0.819 * np.exp(-11.5 * mu[mask2]) + 0.097 * np.exp(-70.1 * mu[mask2])
    
    return np.clip(penetration, 0.01, 1.0)

def rebin_pnsd(df_source: pd.DataFrame, diams_source: np.ndarray,
               diams_target: np.ndarray, min_coverage: float = 0.99) -> pd.DataFrame:
    """Move dN/dlogDp onto another bin set, conserving total number.

    Each source bin's number is divided between the target bins it overlaps, in
    log-diameter space.  That is exact for a histogram, cannot ring or go
    negative the way a spline through the concentrations can, and is a single
    matrix multiply rather than a per-row fit.

    Target bins that the source does not cover come back as NaN rather than
    zero: no measurement is not the same as no particles.
    """
    src, tgt = log_bin_edges(diams_source), log_bin_edges(diams_target)
    src_widths, tgt_widths = np.diff(src), np.diff(tgt)

    overlap = np.clip(np.minimum(src[1:, None], tgt[None, 1:])
                      - np.maximum(src[:-1, None], tgt[None, :-1]), 0.0, None)
    weights = overlap / src_widths[:, None]              # fraction of each source bin, per target bin

    values = df_source.to_numpy(dtype=float)
    missing = np.isnan(values)
    dn = np.where(missing, 0.0, values) * src_widths

    out = (dn @ weights) / tgt_widths
    out[(missing.astype(float) @ (overlap > 0)) > 0] = np.nan       # drew on a gap in the source
    out[:, overlap.sum(axis=0) / tgt_widths < min_coverage] = np.nan  # outside the measured range

    return pd.DataFrame(out, index=df_source.index, columns=diams_target)


def align_bins(df_source: pd.DataFrame, diams_source: np.ndarray, diams_target: np.ndarray) -> pd.DataFrame:
    """Back-compatible name for the conservative rebinning above."""
    return rebin_pnsd(df_source, diams_source, diams_target)

# --- Reading a table of unknown shape --------------------------------------- #

_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


def _decode_head(p: Path, n_bytes: int = 65536) -> tuple[str, str]:
    """Decode the first lines of a text export, to sniff its shape.

    Only the head is read, so a year of 1 min data costs nothing to inspect.  The
    buffer is cut back to the last newline so a truncated multi-byte character
    cannot make a UTF-8 file look like something else.
    """
    head = p.read_bytes()[:n_bytes]
    head = head[: head.rfind(b"\n") + 1] or head
    for enc in _ENCODINGS:
        try:
            return head.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return head.decode("latin-1", errors="replace"), "latin-1"


def _sniff_layout(text: str) -> tuple[str, int]:
    """Return (delimiter, header line index).

    The data block is whatever field count most lines agree on, and the header is
    the first line that agrees with it.  That walks past an instrument preamble
    of any length without needing to know the instrument.
    """
    lines = text.splitlines()
    sample = [(i, ln) for i, ln in enumerate(lines) if ln.strip()][:200]
    if not sample:
        return ",", 0

    best_delim, best_agree, best_header = ",", 0, 0
    for delim in (",", "\t", ";", "|"):
        counts = Counter(ln.count(delim) + 1 for _, ln in sample)
        n_fields, agree = counts.most_common(1)[0]
        if n_fields > 1 and agree > best_agree:
            best_delim = delim
            best_agree = agree
            best_header = next(i for i, ln in sample if ln.count(delim) + 1 == n_fields)

    if best_agree == 0:                                                      # space-aligned columns
        counts = Counter(len(ln.split()) for _, ln in sample)
        n_fields, agree = counts.most_common(1)[0]
        if n_fields > 1:
            header = next(i for i, ln in sample if len(ln.split()) == n_fields)
            return r"\s+", header

    return best_delim, best_header


def _promote_header(frame: pd.DataFrame) -> pd.DataFrame:
    """For spreadsheets: use the first row as wide as the data block as the header."""
    widths = frame.notna().sum(axis=1)
    modal = Counter(widths).most_common(1)[0][0]
    header_idx = int(widths[widths == modal].index[0])
    out = frame.loc[header_idx + 1:].copy()
    out.columns = [str(c).strip() for c in frame.loc[header_idx]]
    return out.reset_index(drop=True)


def _read_any_table(p: Path, notes: list) -> pd.DataFrame:
    """Read csv/txt/dat/xlsx into strings, whatever the delimiter or preamble."""
    if p.suffix.lower() in (".xlsx", ".xls", ".xlsm"):
        raw = pd.read_excel(p, dtype=str, header=None)
        promoted = _promote_header(raw)
        if len(promoted) < len(raw) - 1:
            notes.append(f"Skipped {len(raw) - len(promoted) - 1} preamble row(s) above the header.")
        return promoted

    head, encoding = _decode_head(p)
    if encoding != "utf-8-sig":
        notes.append(f"File is not UTF-8; read as {encoding}.")

    delim, header_line = _sniff_layout(head)
    if delim != ",":
        shown = {"\t": "tab", r"\s+": "whitespace"}.get(delim, delim)
        notes.append(f"Delimiter detected as '{shown}'.")
    if header_line:
        notes.append(f"Skipped {header_line} preamble line(s) above the header.")

    # Stream from the path with the C parser wherever possible; only the
    # whitespace-aligned case needs the slower Python engine.
    return pd.read_csv(p, encoding=encoding, sep=delim, skiprows=header_line,
                       dtype=str, skip_blank_lines=True,
                       engine="python" if len(delim) > 1 else "c")


_DECIMAL_COMMA = re.compile(r"^-?\d+,\d+$")


def _fix_decimal_commas(frame: pd.DataFrame, notes: list) -> pd.DataFrame:
    """European exports write 1,234 for 1.234.  Detect that and normalise."""
    sample = frame.head(200).to_numpy().ravel()
    values = [str(v).strip() for v in sample if v is not None and str(v).strip() not in ("", "nan")]
    if not values:
        return frame
    if sum(bool(_DECIMAL_COMMA.match(v)) for v in values) / len(values) < 0.3:
        return frame

    notes.append("Comma decimal separator detected and converted.")
    return frame.apply(lambda s: s.str.replace(",", ".", regex=False)
                       if hasattr(s, "str") else s)


# --- Diameter columns ------------------------------------------------------- #

def parse_diameter_label(label) -> tuple[Optional[float], Optional[str]]:
    """Return (value, unit) for a size-bin header, or (None, None) if it is not one.

    Handles bare numbers, decorated headers such as ``Dp_10.6nm`` or
    ``dN/dlogDp 0.0106 um``, and rejects indices such as ``Bin 12``.
    """
    text = str(label).strip()
    if not text or _NOT_A_DIAMETER.search(text) or "unnamed" in text.lower():
        return None, None

    match = _DIAMETER_LABEL.match(text)
    if not match:
        return None, None
    try:
        value = float(match.group("num").replace(",", "."))
    except ValueError:
        return None, None
    if value <= 0:
        return None, None

    unit = match.group("unit")
    return value, unit.lower() if unit else None


def _diameter_scale_to_nm(values: list, units: list, notes: list) -> float:
    """Work out whether the header numbers are nm, µm or m, and return the factor.

    Getting this wrong is a silent factor-of-1000 error through every downstream
    calculation, so the assumption is always recorded.
    """
    stated = [u for u in units if u]
    if stated:
        unit = Counter(stated).most_common(1)[0][0]
        if len(set(stated)) > 1:
            notes.append(f"⚠ Mixed diameter units in the header; assumed {unit} throughout.")
        factor = _UNIT_FACTORS_NM[unit]
        if factor != 1.0:
            notes.append(f"⚠ Diameters read as {unit} and converted to nm.")
        return factor

    vmin, vmax = min(values), max(values)
    if vmax < 1e-4:                                                          # 1e-8 m = 10 nm
        notes.append("⚠ Header carries no units; values look like metres, converted to nm.")
        return 1e9
    if vmax <= 20 and vmin < 0.9:                                            # a nm instrument rarely reports below 0.9
        notes.append("⚠ Header carries no units; values look like µm, converted to nm.")
        return 1e3
    return 1.0


def _reject_stray_bins(pairs: list, notes: list) -> list:
    """Drop edge columns that are numeric but far off the log-spaced bin ladder.

    A column literally named ``1`` (a bin index or a flag) would otherwise be
    ingested as a 1 nm size bin and pollute the distribution.
    """
    kept = list(pairs)

    # At most one column from each end, and never on a short ladder: a spliced
    # PSM + SMPS dataset can legitimately open with a wide gap, and losing a real
    # bin would be worse than keeping a stray one.
    for _ in range(2):
        if len(kept) <= 10:
            break
        logs = np.log10([d for _, d in kept])
        gaps = np.diff(logs)
        median_gap = float(np.median(gaps))
        if median_gap <= 0:
            break
        if gaps[0] > 5 * median_gap:
            col, dp = kept.pop(0)
        elif gaps[-1] > 5 * median_gap:
            col, dp = kept.pop()
        else:
            break
        notes.append(f"⚠ Ignored column '{col}': {dp:g} sits far off the size-bin "
                     f"ladder. Rename it if it is a real size bin.")

    return kept


def find_diameter_columns(frame: pd.DataFrame, notes: list) -> list:
    """Return [(column, diameter_nm), ...] sorted ascending by diameter."""
    parsed = [(col, *parse_diameter_label(col)) for col in frame.columns]
    candidates = [(col, val, unit) for col, val, unit in parsed if val is not None]
    if not candidates:
        return []

    factor = _diameter_scale_to_nm([v for _, v, _ in candidates],
                                   [u for _, _, u in candidates], notes)
    pairs = sorted(((col, val * factor) for col, val, _ in candidates), key=lambda x: x[1])

    in_range = [(c, d) for c, d in pairs if MIN_PLAUSIBLE_DP_NM <= d <= MAX_PLAUSIBLE_DP_NM]
    if len(in_range) < len(pairs):
        notes.append(f"⚠ Ignored {len(pairs) - len(in_range)} column(s) outside "
                     f"{MIN_PLAUSIBLE_DP_NM:g}–{MAX_PLAUSIBLE_DP_NM:g} nm.")

    return _reject_stray_bins(in_range, notes)


# --- Timestamps ------------------------------------------------------------- #

_DATE_NAME_HINT = re.compile(r"(date|time|stamp|dt$|^dt)", re.I)
_TIME_OF_DAY = re.compile(r"^\d{1,2}:\d{2}")
_EXCEL_EPOCH = "1899-12-30"                                                  # Excel's day 0


def _parse_free(sample: pd.Series, dayfirst: bool = False) -> pd.Series:
    """Parse mixed or unknown date forms without pandas raising on the odd row."""
    return pd.to_datetime(sample, errors="coerce", format="mixed", dayfirst=dayfirst)


def detect_date_column(frame: pd.DataFrame) -> Optional[str]:
    """Pick the column that most looks like a timestamp, by name and by parsing."""
    best, best_score = None, 0.0
    for col in frame.columns:
        sample = frame[col].dropna().astype(str).str.strip().head(200)
        sample = sample[sample != ""]
        if len(sample) < 3:
            continue
        try:
            hit_rate = _parse_free(sample).notna().mean()
        except Exception:
            hit_rate = 0.0
        score = hit_rate + (0.5 if _DATE_NAME_HINT.search(str(col)) else 0.0)
        if hit_rate >= 0.7 and score > best_score:
            best, best_score = col, score
    return best


def detect_date_format(sample: pd.Series) -> Optional[str]:
    """Return the token format from DATE_FORMAT_OPTIONS that parses the most rows."""
    best, best_rate = None, 0.0
    for _, token_fmt in DATE_FORMAT_OPTIONS:
        if token_fmt in ("auto", "custom"):
            continue
        rate = pd.to_datetime(sample, format=fmt_to_strptime(token_fmt),
                              errors="coerce").notna().mean()
        if rate > best_rate:
            best, best_rate = token_fmt, rate
    return best if best_rate >= 0.9 else None


def _dayfirst_hint(sample: pd.Series, notes: list) -> bool:
    """Decide 01/02/2021 for numeric dates, and say so when it is a guess."""
    parts = sample.str.extract(r"^(\d{1,2})[/-](\d{1,2})[/-]\d{2,4}")
    first = pd.to_numeric(parts[0], errors="coerce")
    second = pd.to_numeric(parts[1], errors="coerce")
    if first.notna().sum() == 0:
        return False
    if first.max() > 12:
        return True
    if second.max() > 12:
        return False
    notes.append("⚠ Day/month order is ambiguous in this file; read as month first. "
                 "Set the format explicitly if that is wrong.")
    return False


def parse_datetime_series(values, date_fmt: str = "auto", notes: Optional[list] = None) -> pd.Series:
    """Parse a column of timestamps under a token format, or work the format out.

    Shared by the PNSD loader and by every side-file loader (met, tracers,
    pollution flags) so that all of them accept the same date variants.
    """
    notes = notes if notes is not None else []
    text = pd.Series(values).astype(str).str.strip()

    numeric = pd.to_numeric(text, errors="coerce")                           # Excel serial dates
    if numeric.notna().mean() > 0.9 and numeric.between(20000, 60000).mean() > 0.9:
        notes.append("Timestamps read as Excel serial numbers.")
        return pd.to_datetime(numeric, unit="D", origin=_EXCEL_EPOCH, errors="coerce")

    token_fmt = date_fmt
    if token_fmt in ("", "auto", "custom", None):
        sample = text.head(500)
        token_fmt = detect_date_format(sample)
        if token_fmt:
            notes.append(f"Date format detected as {token_fmt}.")

            # 01/02/2021 parses cleanly under both day-first and month-first, so
            # detection alone cannot settle it. Say so rather than pick quietly.
            rivals = [f for _, f in DATE_FORMAT_OPTIONS
                      if f not in ("auto", "custom") and f[:2] != token_fmt[:2]
                      and pd.to_datetime(sample, format=fmt_to_strptime(f),
                                         errors="coerce").notna().mean() >= 0.99]
            if rivals:
                notes.append(f"⚠ Every date also parses as {rivals[0]}; read as "
                             f"{token_fmt}. Set the format explicitly if that is wrong.")

    parsed = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")
    if token_fmt:
        parsed = pd.to_datetime(text, format=fmt_to_strptime(token_fmt), errors="coerce")

        # A format with a time part still has to cope with date-only rows.
        if parsed.isna().any() and "HH" in str(token_fmt):
            date_only = strip_time_tokens(token_fmt)
            if date_only != token_fmt:
                gaps = parsed.isna()
                parsed.loc[gaps] = pd.to_datetime(
                    text[gaps], format=fmt_to_strptime(date_only), errors="coerce")

    if parsed.isna().mean() > 0.2:                                           # embedded tz, mixed forms
        parsed = _parse_free(text, dayfirst=_dayfirst_hint(text, notes))
    return parsed


def _build_timestamps(raw: pd.DataFrame, dt_col: str, date_fmt: str, notes: list) -> pd.Series:
    """Turn the chosen column (plus a separate time column, if any) into datetimes."""
    parsed = parse_datetime_series(raw[dt_col], date_fmt, notes)

    # A date-only column plus a separate time-of-day column is a common export.
    if parsed.notna().any() and (parsed.dt.floor("D") == parsed).all():
        for col in raw.columns:
            if col == dt_col or not _DATE_NAME_HINT.search(str(col)):
                continue
            candidate = raw[col].astype(str).str.strip()
            if candidate.str.match(_TIME_OF_DAY).mean() > 0.9:
                clock = pd.to_timedelta(candidate.where(
                    candidate.str.count(":") == 2, candidate + ":00"), errors="coerce")
                if clock.notna().mean() > 0.9:
                    notes.append(f"Time of day taken from the '{col}' column.")
                    parsed = parsed + clock.fillna(pd.Timedelta(0))
                break

    return parsed


# --- Loading Routine -------------------------------------------------------- #

def load_pnsd_file(
    path: str, date_col: str = DEFAULT_DATE_COL, date_fmt: str = DEFAULT_DATE_FMT,
    resample_rule: Optional[str] = None, na_method: str = "drop",
    timezone: str = "UTC", cols_to_drop: str = "",
    flag_col: str = "", flag_value: str = "1"
) -> DataFile:
    p = Path(path)
    result = DataFile(path=p, date_col=date_col, date_fmt=date_fmt)
    notes = result.notes

    try:
        raw = _read_any_table(p, notes)
    except Exception as exc:
        result.error = f"Read fail: {exc}"
        return result

    raw.columns = [str(c).strip() for c in raw.columns]
    drop_list = [c.strip().lower() for c in cols_to_drop.split(",") if c.strip()]
    raw = raw.drop(columns=[c for c in raw.columns if c.lower() in drop_list])
    raw = _fix_decimal_commas(raw, notes)

    n_read = len(raw)

    # --- Which column holds the timestamp ---
    if date_col.strip().lower() in ("", "auto", "auto-detect"):
        dt_col = detect_date_column(raw)
        if dt_col is None:
            result.error = "No column in this file parses as a date. Name one explicitly."
            return result
        notes.append(f"Date column detected as '{dt_col}'.")
    else:
        match = [c for c in raw.columns if c.lower() == date_col.strip().lower()]
        if not match:
            available = ", ".join(list(raw.columns)[:6])
            result.error = f"Date column '{date_col}' not found. Columns start: {available}"
            return result
        dt_col = match[0]

    try:
        raw = raw.copy()
        raw[dt_col] = _build_timestamps(raw, dt_col, date_fmt, notes)

        unparsed = int(raw[dt_col].isna().sum())
        if unparsed:
            notes.append(f"⚠ Dropped {unparsed} row(s) with an unreadable timestamp.")
        raw_good = raw.dropna(subset=[dt_col]).copy()
        if raw_good.empty:
            result.error = "No timestamps could be parsed. Check the date column and format."
            return result

        if raw_good[dt_col].dt.tz is None:
            localised = raw_good[dt_col].dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
            lost = int(localised.isna().sum())
            if lost:
                notes.append(f"⚠ Dropped {lost} row(s) that do not exist or repeat "
                             f"across a {timezone} daylight-saving change.")
            raw_good[dt_col] = localised
        else:
            # Already tz-aware (embedded in the string) — convert rather than
            # double-apply the timezone.
            raw_good[dt_col] = raw_good[dt_col].dt.tz_convert(timezone)

        raw_good = raw_good.dropna(subset=[dt_col]).set_index(dt_col)

        # --- Flag column filtering (must happen before diameter detection) ---
        if flag_col.strip():
            fc_match = [c for c in raw_good.columns if c.lower() == flag_col.strip().lower()]
            if fc_match:
                before = len(raw_good)
                flag_series = pd.to_numeric(raw_good[fc_match[0]], errors="coerce")
                try:
                    raw_good = raw_good[flag_series != float(flag_value)]
                except ValueError:
                    raw_good = raw_good[raw_good[fc_match[0]].astype(str).str.strip() != flag_value.strip()]
                notes.append(f"Flag column '{fc_match[0]}' removed {before - len(raw_good)} row(s).")
            else:
                notes.append(f"⚠ Flag column '{flag_col}' not found; no rows were flagged out.")

        # Out-of-order rows and repeated timestamps break resampling and any
        # diurnal average, so deal with them here rather than downstream.
        if not raw_good.index.is_monotonic_increasing:
            notes.append("⚠ Rows were not in time order; sorted by timestamp.")
            raw_good = raw_good.sort_index()
        duplicated = int(raw_good.index.duplicated().sum())
        if duplicated:
            notes.append(f"⚠ Removed {duplicated} repeated timestamp(s), keeping the first of each.")
            raw_good = raw_good[~raw_good.index.duplicated(keep="first")]

        raw_good.index.name = "datetime"
    except Exception as exc:
        result.error = f"Date parsing failed: {exc}"
        return result

    # --- Which columns are size bins ---
    diam_pairs = find_diameter_columns(raw_good, notes)
    if not diam_pairs:
        available = ", ".join(list(raw_good.columns)[:6])
        result.error = ("No size-bin columns found. Headers should carry the bin "
                        f"diameter, e.g. '10.6' or 'Dp_10.6nm'. Columns start: {available}")
        return result

    # Take the columns in file order, then relabel with the matching diameters.
    # Sorting the diameters without reordering the data silently reverses any
    # file whose bins run large to small.
    pnsd = raw_good[[col for col, _ in diam_pairs]].apply(pd.to_numeric, errors="coerce")
    diameters = [dp for _, dp in diam_pairs]
    pnsd.columns = diameters

    n_dated = len(pnsd)
    if na_method == "drop":
        pnsd = pnsd.dropna()
        if len(pnsd) < n_dated:
            notes.append(f"⚠ Dropped {n_dated - len(pnsd)} row(s) holding a missing value.")
    elif na_method == "ffill":
        pnsd = pnsd.ffill().bfill()
    elif na_method == "interpolate":
        pnsd = pnsd.interpolate(method="time").ffill().bfill()
    elif na_method == "zero":
        pnsd = pnsd.fillna(1.0)

    if resample_rule:
        pnsd = pnsd.resample(resample_rule).mean()
        empty = int(pnsd.iloc[:, 0].isna().sum())
        if empty:
            notes.append(f"⚠ {empty} of {len(pnsd)} resampled interval(s) held no data "
                         f"and were filled from the nearest one.")
        pnsd = pnsd.ffill().bfill()

    if pnsd.empty:
        result.error = "Dataframe became empty after NA dropping or parsing."
        return result

    result.df_raw = raw_good
    result.df = pnsd
    result.n_rows = len(pnsd)
    result.n_bins = len(diameters)
    result.diameters = diameters
    result.date_min = pnsd.index.min()
    result.date_max = pnsd.index.max()
    if n_read and len(pnsd) < n_read * 0.5:
        notes.append(f"⚠ Only {len(pnsd)} of {n_read} rows in the file survived loading.")
    alerts = [n for n in notes if n.startswith("⚠")]                         # notes worth acting on
    result.warning = " ".join(alerts) if alerts else None

    return result
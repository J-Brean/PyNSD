from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator                    # Added PchipInterpolator for CDF splines

_TOKEN_MAP = [
    ("yyyy", "%Y"), ("MM", "%m"), ("dd", "%d"),
    ("HH", "%H"), ("mm", "%M"), ("ss", "%S"),
]

DATE_FORMAT_OPTIONS = [
    ("YYYY/MM/DD HH:mm:ss  —  2021/01/31 00:15:00", "yyyy/MM/dd HH:mm:ss"),
    ("DD/MM/YYYY HH:mm  —  31/01/2021 00:15",       "dd/MM/yyyy HH:mm"),
    ("DD/MM/YYYY HH:mm:ss  —  31/01/2021 00:15:00", "dd/MM/yyyy HH:mm:ss"),
    ("MM/DD/YYYY HH:mm  —  01/31/2021 00:15",       "MM/dd/yyyy HH:mm"),
    ("MM/DD/YYYY HH:mm:ss  —  01/31/2021 00:15:00", "MM/dd/yyyy HH:mm:ss"),
    ("YYYY-MM-DD HH:mm:ss  —  2021-01-31 00:15:00", "yyyy-MM-dd HH:mm:ss"),
    ("YYYY-MM-DD HH:mm  —  2021-01-31 00:15",       "yyyy-MM-dd HH:mm"),
    ("DD-MM-YYYY HH:mm  —  31-01-2021 00:15",       "dd-MM-yyyy HH:mm"),
    ("DD-MM-YYYY HH:mm:ss  —  31-01-2021 00:15:00", "dd-MM-yyyy HH:mm:ss"),
    ("YYYY/MM/DD  —  2021/01/31",                   "yyyy/MM/dd"),
    ("MM/DD/YYYY  —  01/31/2021",                   "MM/dd/yyyy"),
    ("DD/MM/YYYY  —  31/01/2021",                   "dd/MM/yyyy"),
    ("Custom...",                                   "custom"),
]

DATE_COLUMN_OPTIONS = [
    "date", "Date", "DateTime", "datetime",
    "DATE", "Time", "time", "timestamp", "Timestamp", "Custom...",
]
DEFAULT_DATE_COL = "date"
DEFAULT_DATE_FMT = "yyyy/MM/dd HH:mm:ss"

def fmt_to_strptime(token_fmt: str) -> str:
    result = token_fmt
    for token, code in _TOKEN_MAP:
        result = result.replace(token, code)
    return result

@dataclass
class DataFile:
    path: Path
    date_col: str = DEFAULT_DATE_COL
    date_fmt: str = DEFAULT_DATE_FMT
    df: Optional[pd.DataFrame] = None
    df_raw: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    warning: Optional[str] = None
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
    log_diams = np.log10(diams)                                              # Convert midpoints to log space
    
    bounds = np.zeros(len(log_diams) + 1)                                    # Array for bin edges
    if len(log_diams) > 1:                                                   
        bounds[1:-1] = (log_diams[:-1] + log_diams[1:]) / 2.0                # Calculate interior edges
        bounds[0] = log_diams[0] - (bounds[1] - log_diams[0])                # Extrapolate first edge
        bounds[-1] = log_diams[-1] + (log_diams[-1] - bounds[-2])            # Extrapolate final edge
    else:                                                                    
        bounds[0], bounds[1] = log_diams[0] - 0.05, log_diams[0] + 0.05      # Fallback for single bin
        
    dlogdp_array = np.diff(bounds)                                           # Calculate exact variable bin widths
    
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

def align_bins(df_source: pd.DataFrame, diams_source: np.ndarray, diams_target: np.ndarray) -> pd.DataFrame:
    log_diams_src = np.log10(diams_source)
    dlogdp_src = np.mean(np.diff(log_diams_src)) if len(log_diams_src) > 1 else 1.0
    dn_dlogdp = df_source.to_numpy() / dlogdp_src
    
    log_diams_tgt = np.log10(diams_target)
    dlogdp_tgt = np.mean(np.diff(log_diams_tgt)) if len(log_diams_tgt) > 1 else 1.0
    
    interp_func = interp1d(log_diams_src, dn_dlogdp, axis=1, bounds_error=False, fill_value=0.0)
    aligned_dn_dlogdp = interp_func(log_diams_tgt)
    
    aligned_dn = aligned_dn_dlogdp * dlogdp_tgt
    return pd.DataFrame(aligned_dn, index=df_source.index, columns=diams_target)

# --- Loading Routine ---

def load_pnsd_file(
    path: str, date_col: str = DEFAULT_DATE_COL, date_fmt: str = DEFAULT_DATE_FMT,
    resample_rule: Optional[str] = None, na_method: str = "drop",
    timezone: str = "UTC", cols_to_drop: str = ""
) -> DataFile:
    p = Path(path)
    result = DataFile(path=p, date_col=date_col, date_fmt=date_fmt)

    try: raw = pd.read_excel(path, dtype=str) if p.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path, dtype=str)
    except Exception as exc: 
        result.error = f"Read fail: {exc}"
        return result

    drop_list = [c.strip() for c in cols_to_drop.split(",") if c.strip()]
    raw = raw.drop(columns=[c for c in drop_list if c in raw.columns])

    col_match = [c for c in raw.columns if c.strip().lower() == date_col.strip().lower()]
    if not col_match: 
        result.error = f"Date column '{date_col}' not found."
        return result
    dt_col = col_match[0]

    try: 
        parsed_dates = pd.to_datetime(raw[dt_col].str.strip(), format=fmt_to_strptime(date_fmt), errors="coerce")
        raw[dt_col] = parsed_dates
        raw_good = raw.dropna(subset=[dt_col]).copy() # Safely drop bad dates BEFORE timezone conversion
        
        if raw_good[dt_col].dt.tz is None: 
            raw_good[dt_col] = raw_good[dt_col].dt.tz_localize(timezone, ambiguous='NaT', nonexistent='NaT')
        else: 
            raw_good[dt_col] = raw_good[dt_col].dt.tz_convert(timezone)
            
        raw_good = raw_good.dropna(subset=[dt_col]).set_index(dt_col)
        raw_good.index.name = "datetime"
    except Exception as exc: 
        result.error = f"Date parsing failed: {exc}"
        return result

    diam_cols = {}
    for col in raw_good.columns:
        try:
            val = float(str(col).strip())
            if val > 0: diam_cols[col] = val
        except ValueError: pass

    if not diam_cols: 
        result.error = "No numeric diameter columns found."
        return result

    pnsd = raw_good[list(diam_cols.keys())].apply(pd.to_numeric, errors="coerce")
    
    if na_method == "drop": pnsd = pnsd.dropna()
    elif na_method == "ffill": pnsd = pnsd.ffill().bfill()
    elif na_method == "interpolate": pnsd = pnsd.interpolate(method='time').ffill().bfill()
    elif na_method == "zero": pnsd = pnsd.fillna(1)        
    if resample_rule: 
        pnsd = pnsd.resample(resample_rule).mean().ffill().bfill()

    if pnsd.empty:
        result.error = "Dataframe became empty after NA dropping or parsing."
        return result

    result.df_raw = raw_good
    result.df = pnsd
    result.n_rows = len(pnsd)
    result.n_bins = len(diam_cols)
    result.diameters = sorted(diam_cols.values())
    result.df.columns = result.diameters # Align names
    result.date_min = pnsd.index.min()
    result.date_max = pnsd.index.max()

    return result
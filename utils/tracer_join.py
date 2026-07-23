"""Shared helpers for parsing an external time series (gas/met tracers) and
aligning it onto an existing datetime index such as a PMF G matrix.

The datetime parsing mirrors the logic proven in gui/wind_panel.py, but is kept
here as standalone functions so multiple panels can reuse it without coupling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.data_loader import DATE_FORMAT_OPTIONS, fmt_to_strptime, strip_time_tokens


def parse_datetimes(raw_dt: pd.Series, date_fmt: str, tz: str) -> pd.Series:
    """Parse a raw string datetime column into a tz-aware Series.

    Falls back to a date-only format if times fail to parse, then to a loose
    parse if a fifth of rows are still NaT, matching the wind panel's behaviour.
    """
    raw = raw_dt.astype(str).str.strip()
    parsed = pd.to_datetime(raw, format=fmt_to_strptime(date_fmt), errors="coerce")

    if parsed.isna().any() and ("HH" in str(date_fmt) or "%H" in fmt_to_strptime(date_fmt)):
        date_only = strip_time_tokens(date_fmt)
        if date_only != date_fmt:
            miss = parsed.isna()
            parsed.loc[miss] = pd.to_datetime(raw[miss], format=fmt_to_strptime(date_only), errors="coerce")

    if parsed.isna().mean() > 0.2:
        parsed = pd.to_datetime(raw, errors="coerce", utc=False)

    tz = tz or "UTC"
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    else:
        parsed = parsed.dt.tz_convert(tz)
    return parsed


def build_tracer_frame(raw_df: pd.DataFrame, date_col: str, date_fmt: str, tz: str,
                       value_cols: list[str]) -> pd.DataFrame:
    """Return a datetime-indexed, numeric-coerced frame of the chosen tracer columns."""
    df = raw_df.copy()
    df[date_col] = parse_datetimes(df[date_col], date_fmt, tz)
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    out = pd.DataFrame(index=df.index)
    for col in value_cols:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    return out.dropna(how="all")


def align_to_index(tracer_df: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Nearest-time align a datetime-indexed tracer frame onto target_index.

    Returns a frame with exactly len(target_index) rows in target order, so it can
    be used positionally against a G matrix sharing that index. Timezones are
    reconciled first; the tolerance is the median spacing of target_index.
    """
    original = pd.DatetimeIndex(target_index)
    tracer = tracer_df.copy()
    target = original

    if target.tz is None and tracer.index.tz is not None:
        target = target.tz_localize(tracer.index.tz)
    elif target.tz is not None and tracer.index.tz is None:
        tracer.index = tracer.index.tz_localize(target.tz)
    elif target.tz is not None and tracer.index.tz is not None and str(target.tz) != str(tracer.index.tz):
        tracer.index = tracer.index.tz_convert(target.tz)

    tol = pd.Series(target).diff().median()
    if pd.isna(tol) or tol <= pd.Timedelta(0):
        tol = pd.Timedelta("30min")

    left = pd.DataFrame({"datetime": target}).sort_values("datetime")
    right = tracer.sort_index().reset_index()
    right = right.rename(columns={right.columns[0]: "datetime"})

    merged = pd.merge_asof(left, right, on="datetime", direction="nearest", tolerance=tol)
    merged = merged.drop(columns=["datetime"])
    merged.index = original                                             # Restore caller's exact labels/order
    return merged

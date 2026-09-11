"""
utils/npf_classify.py
---------------------
A physical classifier for new particle formation days, following the logic of
Dal Maso et al. (2005) as closely as an instrument with a 20 nm lower limit
allows.

Classifications from this module are automatic and should be treated as
preliminary. Check the days that matter to your result by eye.

Unlike the image model in :mod:`utils.npf_render`, this works from the
measurements. Three criteria are tested separately, and each day is reported
along with the criterion it failed on.

Criterion 1: a rise in the smallest particles
---------------------------------------------
Number concentration in the 20 to 25 nm band is integrated hourly using per-bin
widths, then smoothed with a two-hour centred median. The day's peak is the
maximum between 10:00 and 16:00, requiring at least three valid hours. The
background is the median of the night, taken as 16:00 to 08:00 and assigned to
the morning it runs into, requiring at least six valid hours.

The background is then pooled across the surrounding week, as a centred rolling
median of seven nightly values. Using the single preceding night instead makes
the test sensitive to that one night: a polluted night removes a real event, and
the test ends up favouring days whose own night happened to be cleaner than
usual.

Criterion 2: the rise is confined to the small end
--------------------------------------------------
Formation raises the smallest particles much more than the accumulation mode,
whereas a shallow boundary layer, an advected plume or a change of air mass
raises the whole distribution. The same peak-over-night ratio is calculated for
the 50 to 100 nm band, using the same windows and smoothing, and the quotient of
the two must exceed ``selectivity_min``. Both bands use their own night, so the
two ratios are measured in the same way.

Criterion 3: a mode that grows, starting small and early
--------------------------------------------------------
The tracker in :mod:`utils.npf_tracking` must find a mode climbing in diameter
for several hours, starting at the small end during the morning or early
afternoon. Without the timing requirement, a plume arriving already grown in the
afternoon is difficult to separate from formation, and the tracker produces a
large number of winter false positives. Growth is judged on four counts
together: a plausible growth rate, an overall growth factor, a straight enough
track, and sufficient duration.

Classes
-------
Ia         all three criteria met clearly.
Ib         all three met, less clearly.
II         growth or rise present, but weak or short.
undefined  small particles appear, but do not grow coherently.
non-event  neither.

A day with neither a measured rise nor a track is left unclassified rather than
recorded as a non-event, so that gaps in the record do not appear as a separate
category.

What is not used
----------------
Solar radiation, temperature and condensation sink are not used as criteria. If
sunshine were required, a summer maximum would be built into the result rather
than found in it. Where meteorology is available it is better used afterwards as
a check: a classifier working properly should give class Ia days that are sunny
without having been told to.

Limitations
-----------
The instrument's lower limit sets what can be seen. With bins starting near
20 nm these are events that have already grown past the nucleation mode, so the
rise test detects the arrival of a mode rather than its formation. An instrument
that starts above about 30 nm will find very little.

The default thresholds were tuned on UK urban and rural SMPS records. Another
site, size range or season may need different values, so all of them are
adjustable from the panel.

Reference
---------
Dal Maso, M. et al. (2005), Boreal Environ. Res. 10, 323.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from utils.calculations import dlogdp_per_bin

CLASS_ORDER = ["Ia", "Ib", "II", "undefined", "non-event"]


@dataclass
class ClassifierSettings:
    """Every threshold the classifier uses, in physical units."""
    r_strong: float = 3.0            # N_act/N_bg for a convincing rise
    r_weak: float = 2.0              # and for a marginal one
    gr_min: float = 0.5              # nm/h, plausible growth
    gr_max: float = 20.0
    r2_strong: float = 0.80          # straightness of the growth track
    r2_weak: float = 0.60
    hour_start_min: int = 6          # the track must begin in this hour range
    hour_start_max: int = 14
    dp_start_max: float = 30.0       # nm, and at or below this diameter
    dur_strong: float = 6.0          # hours of coherent growth
    dur_weak: float = 4.0
    selectivity_min: float = 1.4     # the small end must rise this much more than 50-100 nm
    growth_factor_min: float = 1.4   # and the mode must grow by this factor overall
    anom_strong: float = 0.25        # how far the mode sits above its own night
    anom_weak: float = 0.15
    background_days: int = 7         # nights pooled for the background

    # The bands and windows the rise is measured over.
    small_band: tuple = (20.0, 25.0)
    large_band: tuple = (50.0, 100.0)
    active_hours: tuple = (10, 16)
    night_hours: tuple = (16, 8)     # from 16:00 to 08:00 the next morning

    def as_dict(self) -> dict:
        return asdict(self)


def band_concentration(df: pd.DataFrame, diams: np.ndarray, lo: float, hi: float) -> pd.Series:
    """Hourly number concentration between two diameters, cm-3.

    Integrated with per-bin widths, so a spliced or variable-resolution
    instrument is handled correctly.
    """
    mask = (diams >= lo) & (diams <= hi)
    if not mask.any():
        return pd.Series(dtype=float)
    widths = dlogdp_per_bin(diams)[mask]
    hourly = df.iloc[:, mask].resample("h").mean()
    return (hourly * widths).sum(axis=1, skipna=True).where(hourly.notna().any(axis=1))


def _day_ratio(series: pd.Series, settings: ClassifierSettings) -> pd.DataFrame:
    """Per day: the midday peak over the surrounding night background."""
    if series.empty:
        return pd.DataFrame(columns=["N_bg", "N_act", "ratio"])

    smoothed = series.rolling(2, center=True, min_periods=1).median()
    hour = series.index.hour
    calendar_day = pd.Index(series.index.date, name="day")

    # A night belongs to the morning it runs into, so 16:00 onwards counts
    # towards the next day.
    night_start, night_end = settings.night_hours
    is_night = (hour >= night_start) | (hour < night_end)
    night_day = np.where(hour >= night_start,
                         calendar_day + pd.Timedelta(days=1), calendar_day)

    night = pd.DataFrame({"day": night_day, "v": smoothed.to_numpy()})[is_night]
    bg = night.groupby("day")["v"].agg(["median", "count"])
    bg = bg[bg["count"] >= 6]["median"].rename("N_bg")

    act_lo, act_hi = settings.active_hours
    active = pd.DataFrame({"day": calendar_day, "v": smoothed.to_numpy()})[
        (hour >= act_lo) & (hour < act_hi)]
    ac = active.groupby("day")["v"].agg(["max", "count"])
    ac = ac[ac["count"] >= 3]["max"].rename("N_act")

    out = pd.concat([bg, ac], axis=1).dropna()
    out = out[(out["N_bg"] > 0) & np.isfinite(out["N_bg"]) & np.isfinite(out["N_act"])]
    out["ratio"] = out["N_act"] / out["N_bg"]
    return out


def daily_metrics(df: pd.DataFrame, diams: np.ndarray,
                  settings: ClassifierSettings | None = None) -> pd.DataFrame:
    """Per-day rise and selectivity, the two measured inputs to the classifier.

    Returns ``N_bg`` and ``N_act`` for the small band with their ratio, ``bg7``
    and ``ratio7`` against the seven-night background, ``r_large`` for the
    accumulation band, and their quotient as ``selectivity``. Days without
    enough night or midday hours are dropped rather than estimated.
    """
    settings = settings or ClassifierSettings()
    diams = np.asarray(diams, dtype=float)

    small = _day_ratio(band_concentration(df, diams, *settings.small_band), settings)
    large = _day_ratio(band_concentration(df, diams, *settings.large_band), settings)
    if small.empty:
        return pd.DataFrame(columns=["N_bg", "N_act", "ratio", "bg7", "ratio7",
                                     "r_large", "selectivity"])

    out = small.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()

    # The rise is measured against the typical night of the surrounding week,
    # not against the one night before: a single loaded night otherwise divides
    # a real event out of existence.
    spine = pd.date_range(out.index.min(), out.index.max(), freq="D")
    out = out.reindex(spine)
    window = settings.background_days
    bg7 = out["N_bg"].rolling(window, center=True, min_periods=1).median()
    out["bg7"] = np.where(np.isfinite(bg7) & (bg7 > 0), bg7, out["N_bg"])
    out["ratio7"] = out["N_act"] / out["bg7"]

    # Selectivity: formation lifts the smallest particles far more than the
    # accumulation mode. A boundary-layer collapse or an advected plume lifts
    # everything together.
    if not large.empty:
        r_large = pd.Series(large["ratio"].to_numpy(),
                            index=pd.to_datetime(large.index), name="r_large")
        out["r_large"] = r_large.reindex(out.index)
        # Both bands use their own night as the denominator, so the two are
        # measured the same way and their ratio means something. The seven-day
        # background belongs to the rise test, not to this one.
        out["selectivity"] = out["ratio"] / out["r_large"]
    else:
        out["r_large"] = np.nan
        out["selectivity"] = np.nan

    return out.dropna(subset=["N_act"])


def classify(metrics: pd.DataFrame, tracks: pd.DataFrame | None = None,
             settings: ClassifierSettings | None = None) -> pd.DataFrame:
    """Apply the criteria and return one labelled row per day.

    Every intermediate test is kept as its own column, so a day can be
    interrogated rather than just read off: ``selective``, ``new_strong``,
    ``new_weak``, ``grows``, ``early``, ``grew``, ``coherent_strong`` and
    ``coherent_weak``. The class is the first of Ia, Ib, II, undefined that its
    combination satisfies, and non-event otherwise.

    ``metrics`` comes from :func:`daily_metrics`. ``tracks`` comes from
    :func:`utils.npf_tracking.track_days`, or None when growth tracking is off,
    in which case nothing tests growth and no day can reach class I.
    """
    settings = settings or ClassifierSettings()
    d = metrics.copy()
    if tracks is not None and not tracks.empty:
        d = d.join(tracks.reindex(d.index), how="left")
    for col in ["gr", "r2", "from", "to", "dp_from", "dp_to", "anom"]:
        if col not in d.columns:
            d[col] = np.nan

    finite = np.isfinite
    d["duration"] = d["to"] - d["from"]

    d["selective"] = finite(d["selectivity"]) & (d["selectivity"] >= settings.selectivity_min)
    d["new_strong"] = finite(d["ratio7"]) & (d["ratio7"] >= settings.r_strong) & d["selective"]
    d["new_weak"] = finite(d["ratio7"]) & (d["ratio7"] >= settings.r_weak) & d["selective"]

    d["grows"] = (finite(d["gr"]) & d["gr"].between(settings.gr_min, settings.gr_max)
                  & finite(d["r2"]))
    d["early"] = (finite(d["from"])
                  & d["from"].between(settings.hour_start_min, settings.hour_start_max)
                  & finite(d["dp_from"]) & (d["dp_from"] <= settings.dp_start_max))
    d["grew"] = (finite(d["dp_to"]) & finite(d["dp_from"])
                 & (d["dp_to"] / d["dp_from"] >= settings.growth_factor_min))

    d["coherent_strong"] = (d["grows"] & d["grew"] & (d["r2"] >= settings.r2_strong)
                            & (d["duration"] >= settings.dur_strong)
                            & (d["anom"] > settings.anom_strong))
    d["coherent_weak"] = (d["grows"] & d["grew"] & (d["r2"] >= settings.r2_weak)
                          & (d["duration"] >= settings.dur_weak)
                          & (d["anom"] > settings.anom_weak))

    d["class"] = np.select(
        [d["new_strong"] & d["coherent_strong"] & d["early"],
         d["new_weak"] & d["coherent_strong"] & d["early"],
         d["new_weak"] & d["coherent_weak"] & d["early"],
         d["new_weak"] & ~d["coherent_weak"]],
        ["Ia", "Ib", "II", "undefined"],
        default="non-event")
    d.loc[d["ratio7"].isna() & d["gr"].isna(), "class"] = None
    d["class"] = pd.Categorical(d["class"], categories=CLASS_ORDER, ordered=True)
    return d

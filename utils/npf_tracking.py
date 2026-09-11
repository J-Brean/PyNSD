"""
utils/npf_tracking.py
---------------------
Automatic growth rates for new particle formation days.

Growth rates from this module are automatic and should be treated as
preliminary. Check any day you intend to quote.

Method
------
The approach follows what a person does when fitting a mode by hand: identify
which bump in the spectrum is growing, and reject a track that jumps about. It
runs in four stages.

1. Ridge enhancement. The accumulation mode carries most of the number but says
nothing about formation, so it has to be suppressed. Each hour's spectrum, as
log10 dN/dlogDp, is smoothed along log10 Dp with Gaussian kernels of 0.05 and
0.30 decades, and the broader result is subtracted from the narrower. This
leaves local curvature, so a small growing mode is judged on shape rather than
on concentration. Smoothing ignores missing bins rather than filling them, and
hours with fewer than six valid bins are dropped.

2. An overnight comparison. Ridge strength on its own tends to follow the
standing Aitken mode through the day and return a growth rate near zero. Each
point is therefore also scored on how far it sits above the same size bin
overnight, using the median of 00:00 to 05:00, or the 15th percentile of the
whole day where fewer than three night hours are available. The two scores are
clipped, to 0.3 decades for the ridge and 1.2 decades for the overnight
difference, and combined with weights of 1.0 and 1.6. The result is smoothed in
time with a 1-2-1 kernel, so a single bad scan does not break a ridge.

3. A dynamic programme over size and growth rate. The state is the pair (size
bin, current growth rate), with growth rate quantised from -1 to 20 nm/h in
steps of 0.5. Between one hour and the next the growth rate may change by one
step, at a cost of 0.35, and the diameter follows from it: the next diameter is
the current one plus the growth rate times the elapsed time, snapped to the
nearest measured bin. The best path maximises the total score across the day.
Tracks start in the lowest quarter of the size window, at the first hour between
06:00 and 14:00, since formation events begin small and during daylight. No
individual hour is accepted or rejected on a threshold, and the growth-rate part
of the state carries enough memory that one noisy hour does not move the track.

4. Sub-bin refinement. Channel spacing is typically several per cent, which is
coarse when the quantity of interest is a slope. A parabola is fitted through
the path bin and its two neighbours in log10 Dp, and its vertex is taken as the
mode diameter.

The path spans the whole day, so it is trimmed twice. The first trim is for
continuity: an hour is kept only where the ridge exceeds 0.15 and the overnight
difference reaches ``anom_floor``, and a run is broken by a missing hour or by a
step implying more than ``gr_jump`` nm/h, which indicates a jump between modes
rather than growth. Only the longest unbroken run is kept, so two separate
plumes are not joined into one event. The second trim keeps the rising section
only, from the smallest diameter reached before the maximum to that maximum,
since a flat lead-in or tail pulls a straight-line slope towards zero.

The growth rate is the slope of a weighted least-squares fit of refined mode
diameter against hour, weighted by ridge strength, and is reported with its
standard error and R².

Second estimate
---------------
A separate estimate is made from appearance times: for each size bin within the
tracked range, the hour at which its enhancement over the night background
peaks. The slope of diameter against that hour gives a growth rate that uses
none of the same machinery. ``agree`` is set where the two estimates differ by
less than half the larger of them. Days where they disagree are flagged rather
than averaged.

Limitations
-----------
The tracker follows an already-formed mode across the measured range, typically
20 to 100 nm. It says nothing about sub-10 nm growth, which needs a PSM or a
NAIS, and results should be described accordingly.

The path fit reads low when growth is fast. On synthetic events the
appearance-time estimate stays close to the input value while the path fit falls
below it above roughly 5 nm/h, because the score saturates across a strong broad
mode and the programme has little gradient left to follow. Both estimates are
returned for this reason.

The overnight comparison assumes the night was cleaner than the day. A nocturnal
source, or an event continuing past midnight, weakens it and can lose the track.

References
----------
Kulmala, M. et al. (2012), Nat. Protoc. 7, 1651, for growth rates by mode
fitting and by appearance time.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class TrackerSettings:
    """Tuning, all in physical units."""
    dp_lo: float = 15.0              # nm, bottom of the tracking window
    dp_hi: float = 120.0             # above this a growing mode has merged with
                                     # the accumulation mode and the question is moot
    gr_min_level: float = -1.0       # nm/h the state may take
    gr_max_level: float = 20.0
    gr_level_step: float = 0.5
    gr_step: int = 1                 # levels GR may change per hour
    gr_penalty: float = 0.35         # reward charged per level of change
    w_ridge: float = 1.0             # weight on "is this a local bump"
    w_anom: float = 1.6              # weight on "were these absent overnight"
    start_hour_min: int = 6
    start_hour_max: int = 14
    min_hours: int = 5               # a shorter track is not a growth rate
    smooth_bg: float = 0.30          # log10 nm, background kernel subtracted
    smooth_sig: float = 0.05         # log10 nm, light smoothing kept
    anom_floor: float = 0.25         # the mode must be this far above its own night
    gr_jump: float = 15.0            # nm/h, a bigger step is a jump between modes
    min_bins: int = 8
    min_track_hours: int = 12        # hours of data before a day is worth trying

    def levels(self) -> np.ndarray:
        n = int(round((self.gr_max_level - self.gr_min_level) / self.gr_level_step)) + 1
        return self.gr_min_level + self.gr_level_step * np.arange(n)

    def as_dict(self) -> dict:
        return asdict(self)


TRACK_COLUMNS = ["gr", "se", "r2", "gr_app", "r2_app", "from", "to",
                 "dp_from", "dp_to", "ridge", "anom", "note"]


def _smooth_size(values: np.ndarray, log_dp: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing along log10 Dp that ignores gaps rather than filling them."""
    if sigma <= 0:
        return values
    w = np.exp(-0.5 * ((log_dp[:, None] - log_dp[None, :]) / sigma) ** 2)
    ok = np.isfinite(values).astype(float)
    num = w @ np.where(np.isfinite(values), values, 0.0)
    den = w @ ok
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _peak_subbin(log_dp: np.ndarray, y: np.ndarray, i: int) -> float:
    """Refine a peak position by fitting a parabola through it and its neighbours."""
    lo, hi = max(0, i - 2), min(len(log_dp), i + 3)
    xs, ys = log_dp[lo:hi], y[lo:hi]
    if len(xs) < 3 or not np.all(np.isfinite(ys)):
        return float(log_dp[i])
    try:
        c2, c1, _ = np.polyfit(xs, ys, 2)
    except (np.linalg.LinAlgError, ValueError):
        return float(log_dp[i])
    if not np.isfinite(c2) or c2 >= 0:
        return float(log_dp[i])
    m = -c1 / (2 * c2)
    if not np.isfinite(m) or m < xs.min() or m > xs.max():
        return float(log_dp[i])
    return float(m)


def _reward_surface(lz: np.ndarray, log_dp: np.ndarray, hours: np.ndarray,
                    s: TrackerSettings) -> tuple[np.ndarray, np.ndarray]:
    """The two rewards, added: local bump, and absence overnight.

    Without the novelty term the programme happily follows the standing Aitken
    mode all day and returns a growth rate of zero.
    """
    ridge = np.full_like(lz, np.nan)
    for t in range(lz.shape[0]):
        if np.isfinite(lz[t]).sum() >= 6:
            ridge[t] = (_smooth_size(lz[t], log_dp, s.smooth_sig)
                        - _smooth_size(lz[t], log_dp, s.smooth_bg))

    night = hours <= 5
    with np.errstate(invalid="ignore"):
        if night.sum() >= 3:
            base = np.nanmedian(lz[night], axis=0)
        else:
            base = np.nanquantile(lz, 0.15, axis=0)
    anom = np.where(np.isfinite(lz - base), lz - base, 0.0)

    A = np.clip(anom, 0.0, 1.2) / 1.2
    Rr = np.clip(ridge, -0.3, 0.3) / 0.3
    R = s.w_ridge * Rr + s.w_anom * A

    # A little smoothing in time as well, so one bad scan cannot break a ridge.
    kernel = np.array([0.25, 0.5, 0.25])
    finite = np.isfinite(R).astype(float)
    filled = np.where(np.isfinite(R), R, 0.0)
    num = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 0, filled)
    den = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 0, finite)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.where(den > 0, num / den, np.nan)
    R[~np.isfinite(R)] = -1.0
    return R, np.where(np.isfinite(A), A, 0.0)


def _best_path(R: np.ndarray, dp: np.ndarray, hours: np.ndarray,
               s: TrackerSettings) -> tuple[np.ndarray, np.ndarray] | None:
    """Dynamic programme over (size, growth rate); returns the path in bin indices."""
    levels = s.levels()
    S, V, T = len(dp), len(levels), len(hours)
    start = np.flatnonzero((hours >= s.start_hour_min) & (hours <= s.start_hour_max))
    if not len(start):
        return None
    t0 = int(start[0])

    NEG = -1e6
    val = np.full((S, V), NEG)
    # A growth event begins small: starting anywhere would find the accumulation
    # mode again.
    val[: max(3, S // 4), :] = R[t0, : max(3, S // 4)][:, None]

    back_s = np.full((T, S, V), -1, dtype=np.int32)
    back_v = np.full((T, S, V), -1, dtype=np.int32)

    for t in range(t0 + 1, T):
        dt = float(hours[t] - hours[t - 1])
        nv = np.full((S, V), NEG)
        for v in range(V):
            target = dp + levels[v] * dt
            in_range = (target >= dp[0]) & (target <= dp[-1])
            idx = np.clip(np.searchsorted(dp, target), 0, S - 1)
            lower = np.clip(idx - 1, 0, S - 1)
            take_lower = np.abs(dp[lower] - target) <= np.abs(dp[idx] - target)
            idx = np.where(take_lower, lower, idx)

            best = np.full(S, NEG)
            best_s = np.full(S, -1, dtype=np.int32)
            best_v = np.full(S, -1, dtype=np.int32)
            for vp in range(max(0, v - s.gr_step), min(V, v + s.gr_step + 1)):
                cand = val[:, vp] + R[t, idx] - s.gr_penalty * abs(v - vp)
                ok = in_range & (val[:, vp] > NEG / 2)
                if not ok.any():
                    continue
                # Sorting ascending means the largest candidate is written last,
                # so a plain scatter leaves the maximum in place.
                order = np.argsort(cand[ok], kind="stable")
                tgt = idx[ok][order]
                tmp = np.full(S, NEG)
                tmp_s = np.full(S, -1, dtype=np.int32)
                tmp[tgt] = cand[ok][order]
                tmp_s[tgt] = np.flatnonzero(ok)[order].astype(np.int32)
                better = tmp > best
                best[better] = tmp[better]
                best_s[better] = tmp_s[better]
                best_v[better] = vp
            nv[:, v] = best
            back_s[t, :, v] = best_s
            back_v[t, :, v] = best_v

        val = nv
        if not (val > NEG / 2).any():
            return None

    s_i, v_i = np.unravel_index(int(np.argmax(val)), val.shape)
    path_s = np.zeros(T, dtype=int)
    for t in range(T - 1, t0, -1):
        path_s[t] = s_i
        s_new, v_new = back_s[t, s_i, v_i], back_v[t, s_i, v_i]
        if s_new < 0:
            return None
        s_i, v_i = int(s_new), int(v_new)
    path_s[t0] = s_i
    return path_s[t0:], hours[t0:]


def _appearance_rate(z: np.ndarray, dp: np.ndarray, hours: np.ndarray,
                     h0: float, h1: float, d0: float, d1: float) -> tuple[float, float]:
    """Cross-check: the hour each bin's enhancement peaks, against its diameter."""
    keep = (dp >= d0 * 0.9) & (dp <= d1 * 1.1)
    if keep.sum() < 4 or len(hours) < 8:
        return np.nan, np.nan
    lz = np.log10(np.maximum(z[:, keep], 1e-3))
    night = hours <= 5
    with np.errstate(invalid="ignore"):
        base = (np.nanmedian(lz[night], axis=0) if night.sum() >= 3
                else np.nanquantile(lz, 0.15, axis=0))
    an = lz - base
    win = np.flatnonzero((hours >= h0) & (hours <= h1 + 3))
    if len(win) < 5:
        return np.nan, np.nan

    peak_hour = np.full(keep.sum(), np.nan)
    for j in range(keep.sum()):
        v = an[win, j]
        if np.isfinite(v).sum() >= 5:
            peak_hour[j] = hours[win][int(np.nanargmax(v))]
    ok = np.isfinite(peak_hour)
    if ok.sum() < 4 or len(np.unique(peak_hour[ok])) < 3:
        return np.nan, np.nan

    slope, intercept = np.polyfit(peak_hour[ok], dp[keep][ok], 1)
    pred = slope * peak_hour[ok] + intercept
    resid = dp[keep][ok] - pred
    ss_tot = np.sum((dp[keep][ok] - dp[keep][ok].mean()) ** 2)
    r2 = 1 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else np.nan
    return float(slope), float(r2)


def track_day(day_df: pd.DataFrame, diams: np.ndarray,
              settings: TrackerSettings | None = None) -> dict:
    """Find the growing mode on one day.

    Returns the growth rate and everything needed to judge it: the standard
    error and R² of the fit, the independent appearance-time estimate, the hours
    and diameters the track spanned, and the mean ridge and anomaly along it.
    A day that cannot be tracked returns the same keys with ``note`` set to the
    reason, so a failure can be reported rather than guessed at.

    ``day_df`` is one day of dN/dlogDp with a datetime index; it is averaged to
    hours internally. Any bins outside the tracking window are ignored.
    """
    s = settings or TrackerSettings()
    diams = np.asarray(diams, dtype=float)
    fail = {c: np.nan for c in TRACK_COLUMNS}

    keep = (diams >= s.dp_lo) & (diams <= s.dp_hi)
    if keep.sum() < s.min_bins:
        return {**fail, "note": "too few size bins in the tracking window"}

    hourly = day_df.iloc[:, keep].resample("h").mean()
    hourly = hourly.dropna(how="all")
    if len(hourly) < s.min_track_hours:
        return {**fail, "note": "too few hours of data"}

    dp = diams[keep]
    hours = hourly.index.hour.to_numpy().astype(float)
    z = np.array(hourly.to_numpy(dtype=float), copy=True)
    z[z <= 0] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        lz = np.log10(z)

    R, A = _reward_surface(lz, np.log10(dp), hours, s)
    path = _best_path(R, dp, hours, s)
    if path is None:
        return {**fail, "note": "no path found"}
    path_s, path_hours = path

    rows = np.searchsorted(hours, path_hours)
    ridge_v = R[rows, path_s]
    anom_v = A[rows, path_s]
    log_dp = np.log10(dp)
    dp_fit = np.array([10 ** _peak_subbin(log_dp, _smooth_size(lz[r], log_dp, s.smooth_sig), i)
                       for r, i in zip(rows, path_s)])

    # Continuity: a growth event is present at every hour in between. Two plumes
    # are not, and neither is a pre-existing mode the search bridged across a
    # dead stretch to reach.
    ok = (ridge_v > 0.15) & (anom_v >= s.anom_floor)
    step = np.concatenate([[0.0], np.diff(dp_fit) / np.maximum(np.diff(path_hours), 1)])
    gap = np.concatenate([[True], np.diff(path_hours) != 1])
    brk = (~ok) | (step > s.gr_jump) | gap
    run = np.cumsum(brk)
    if not ok.any():
        return {**fail, "note": "no hour is clearly above its own night"}
    best_run = pd.Series(run[ok]).value_counts().idxmax()
    sel = (run == best_run) & ok
    if sel.sum() < s.min_hours:
        return {**fail, "note": "growth is not continuous for long enough"}

    idx = np.flatnonzero(sel)
    # Trim to the rising segment: a flat lead-in and a flat tail both drag a
    # straight-line slope towards zero, and a person fitting by hand would
    # include neither.
    i_max = int(np.argmax(dp_fit[idx]))
    i_min = int(np.argmin(dp_fit[idx][: i_max + 1]))
    idx = idx[i_min:i_max + 1]
    if len(idx) < s.min_hours:
        return {**fail, "note": "rising segment too short"}

    hrs, fit, w = path_hours[idx], dp_fit[idx], np.maximum(ridge_v[idx] + 1, 0.05)
    slope, intercept = np.polyfit(hrs, fit, 1, w=np.sqrt(w))
    pred = slope * hrs + intercept
    ss_res = np.sum(w * (fit - pred) ** 2)
    ss_tot = np.sum(w * (fit - np.average(fit, weights=w)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    dof = max(len(hrs) - 2, 1)
    se = float(np.sqrt(ss_res / dof / np.sum(w * (hrs - np.average(hrs, weights=w)) ** 2)))

    gr_app, r2_app = _appearance_rate(z, dp, hours, hrs[0], hrs[-1], fit[0], fit[-1])

    return {"gr": float(slope), "se": se, "r2": float(r2), "gr_app": gr_app, "r2_app": r2_app,
            "from": float(hrs[0]), "to": float(hrs[-1]),
            "dp_from": float(fit[0]), "dp_to": float(fit[-1]),
            "ridge": float(np.mean(ridge_v[idx])), "anom": float(np.mean(anom_v[idx])),
            "note": "", "track_hours": hrs, "track_dp": fit}


def track_days(df: pd.DataFrame, diams: np.ndarray,
               settings: TrackerSettings | None = None, progress=None) -> pd.DataFrame:
    """Track every day in the frame, one row per day indexed by date.

    Two summary flags are added. ``agree`` marks days where the path fit and the
    appearance-time estimate differ by less than half the larger of them, and
    ``plausible`` marks a growth rate between 0.2 and 20 nm/h with R² above 0.5
    on a mode clearly above its own night. Neither is a criterion in itself;
    they are there so a day can be set aside for inspection.

    ``progress(done, total, message)`` is called per day and may raise to cancel.
    The fitted tracks themselves are kept on ``.attrs["tracks"]`` for plotting.
    """
    settings = settings or TrackerSettings()
    groups = list(df.groupby(df.index.date))
    rows, tracks = {}, {}
    for i, (day, day_df) in enumerate(groups):
        if progress is not None:
            progress(i, len(groups), f"Tracking {day}")
        result = track_day(day_df, diams, settings)
        tracks[pd.Timestamp(day)] = (result.pop("track_hours", None), result.pop("track_dp", None))
        rows[pd.Timestamp(day)] = result

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "day"
    out.attrs["tracks"] = tracks
    if not out.empty:
        out["agree"] = (np.isfinite(out["gr"]) & np.isfinite(out["gr_app"])
                        & (np.abs(out["gr"] - out["gr_app"])
                           < 0.5 * np.maximum(out["gr"].abs(), out["gr_app"].abs())))
        out["plausible"] = (np.isfinite(out["gr"]) & (out["gr"] > 0.2) & (out["gr"] < 20)
                            & (out["r2"] > 0.5) & (out["anom"] > 0.15))
    return out

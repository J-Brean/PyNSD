"""
utils/npf_render.py
-------------------
Daily contour plots in the style the Kecorius et al. (2024) CNN was trained on.

The model
---------
The classifier is a convolutional network trained on 4,819 daily contour plots
of particle number size distributions, labelled by hand into three classes: NPF,
NO (an ordinary day) and BAD (unusable data). It works from a picture rather
than from the measurements. That suits the problem, since a person recognises a
growth event the same way, but it also means the model tends to agree with
whatever its labellers counted as an event. Clear banana shapes are recognised
well, unusual days less so.

Why the style is copied
-----------------------
A network shown a rendering unlike the ones it was trained on is being asked
about a kind of picture it has not seen, and its probabilities shift
accordingly. This turned out to matter a great deal, and not in the obvious
places. Rendering the same day with the correct values but matplotlib's default
tick label size, or with the plot panel a few tens of pixels away from where R
puts it, dropped p(NPF) on a clear event from 0.97 to below 0.01. The model sees
the whole 1100 x 600 frame and resizes it, so the size and position of
everything in that frame is part of the input.

The settings below were therefore measured from R renders that the model is
known to score correctly, rather than chosen. Images are 1100 x 600 px with the
plot panel at x 118-918 and y 118-453, the colour bar at x 963-989, and tick
labels drawn at the size R produces at pointsize 24. Axis labels are the strings
"x" and "y", which is what base R prints when ``fields::image.plot`` is called
without labels.

Axes are hour of day, 1 to 24, against log10 Dp on whatever size grid the
instrument used that day, as the training images were not on a fixed range
either. Cells that are missing or outside 0 to 5 are painted black.

Palette
-------
Two are available. ``turbo`` is the default and is what the working R pipeline
uses. ``tim`` is ``fields::tim.colors``, read from the colour bar of the
original training images, and is what the model was actually trained on. Turbo
scores clear events higher in every test run here, so it is preferred despite
not being the training palette.

Validation
----------
Thirty Marylebone days spanning the full range of p(NPF) were decoded from the R
renders, re-rendered by this module and scored again. All thirty gave the same
call at a threshold of 0.10, with a correlation of 0.94 and a mean absolute
difference in p(NPF) of 0.055. Some of that difference is decoding loss rather
than rendering, since the data had to be read back out of a PNG.

Threshold
---------
Taking the most likely class is not the best operating point. Against the 1,500
labelled images available for tuning, counting a day as NPF whenever p(NPF)
reaches 0.10 raises recall from 0.80 to 0.90, and at a base rate of 5% real
events precision falls only from 0.99 to 0.93. Below 0.10 precision drops
sharply, reaching 0.80 at a threshold of 0.05. The panel therefore defaults to
0.10 and allows it to be changed.

Mirroring
---------
A growth event is still a growth event read right to left, but the network was
not trained on flipped images and does not score them identically. Rendering
each day twice, once mirrored, and averaging the two probabilities settles
borderline days, at twice the run time. This is optional and on by default.

Reference
---------
Kecorius, S. et al. (2024), a CNN identifier for new particle formation events.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# fields::tim.colors(64), read off the colour bar of the training images.
TIM_COLORS = [
    (0, 0, 143), (0, 0, 152), (0, 0, 170), (0, 0, 179),
    (0, 0, 197), (0, 0, 214), (0, 0, 224), (0, 0, 241),
    (0, 0, 251), (0, 16, 255), (0, 27, 255), (0, 47, 255),
    (0, 66, 255), (0, 78, 255), (0, 99, 255), (0, 109, 255),
    (0, 130, 255), (0, 147, 255), (0, 160, 255), (0, 178, 255),
    (0, 191, 255), (0, 212, 255), (0, 222, 255), (0, 243, 255),
    (6, 254, 249), (19, 255, 235), (39, 255, 215), (50, 255, 204),
    (70, 255, 184), (86, 255, 168), (101, 255, 153), (117, 255, 137),
    (132, 255, 122), (153, 255, 101), (163, 255, 91), (184, 255, 70),
    (198, 255, 56), (215, 255, 39), (235, 255, 19), (245, 255, 9),
    (255, 243, 0), (255, 229, 0), (255, 212, 0), (255, 199, 0),
    (255, 181, 0), (255, 160, 0), (255, 150, 0), (255, 130, 0),
    (255, 117, 0), (255, 99, 0), (255, 78, 0), (255, 68, 0),
    (255, 47, 0), (255, 35, 0), (255, 16, 0), (254, 5, 0),
    (240, 0, 0), (220, 0, 0), (210, 0, 0), (189, 0, 0),
    (179, 0, 0), (158, 0, 0), (148, 0, 0), (128, 0, 0),
]

IMAGE_SIZE = (1100, 600)
Z_LIMITS = (0.0, 5.0)
DEFAULT_PALETTE = "turbo"

# R draws these at pointsize 24, which gives much larger tick labels than
# matplotlib's default. The size of the furniture turns out to matter: at the
# default size p(NPF) on a clear event came out near zero.
TICK_SIZE = 15

# Panel and colour bar in pixels, measured from R renders that the model scores
# correctly. The geometry matters more than it might seem: rendering the same
# values into a panel of a different size and position drops p(NPF) on a clear
# event from 0.97 to below 0.01, because the network sees the whole 1100x600
# frame and resizes it, so the data ends up at a different scale and place.
PANEL_PX = (118, 918, 118, 453)                   # x0, x1, y0, y1
BAR_PX = (963, 989, 118, 453)


def _axes_rect(px) -> tuple[float, float, float, float]:
    x0, x1, y0, y1 = px
    return (x0 / IMAGE_SIZE[0], 1 - y1 / IMAGE_SIZE[1],
            (x1 - x0) / IMAGE_SIZE[0], (y1 - y0) / IMAGE_SIZE[1])


def palette_colormap(name: str = DEFAULT_PALETTE) -> ListedColormap:
    """The 64-level colour map to render with.

    ``turbo`` matches the renders the model is known to score correctly and is
    the default. ``tim`` is the palette of the original training images, kept
    because it is what the model was trained on.
    """
    if name == "tim":
        colours = np.array(TIM_COLORS) / 255.0
    else:
        colours = matplotlib.colormaps["turbo"](np.linspace(0, 1, 64))[:, :3]
    cmap = ListedColormap(colours, name=name)
    cmap.set_bad("black")
    cmap.set_under("black")
    cmap.set_over("black")
    return cmap


def render_day(day_df: pd.DataFrame, diams: np.ndarray, path: str,
               mirror: bool = False, palette: str = DEFAULT_PALETTE) -> str:
    """Write one day's contour plot to ``path`` in the training style.

    The day is averaged to 24 hours and reindexed onto a full day, so missing
    hours stay missing and are painted black rather than interpolated over.
    Concentrations are taken as log10 dN/dlogDp and clipped to 0 to 5 by the
    colour limits, as in the training set.

    ``mirror`` flips the day left to right. The network was not trained on
    flipped images and does not score them identically, so averaging the two
    predictions steadies borderline days.
    """
    diams = np.asarray(diams, dtype=float)
    hourly = day_df.resample("h").mean().reindex(
        pd.date_range(day_df.index.normalize()[0], periods=24, freq="h", tz=day_df.index.tz))

    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.log10(hourly.to_numpy(dtype=float))
    z[~np.isfinite(z)] = np.nan
    if mirror:
        z = z[::-1]

    fig = Figure(figsize=(IMAGE_SIZE[0] / 100, IMAGE_SIZE[1] / 100), dpi=100)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes(_axes_rect(PANEL_PX))
    ax.set_facecolor("black")                       # missing hours read as black

    cmap = palette_colormap(palette)
    masked = np.ma.masked_invalid(z.T)
    ax.pcolormesh(np.arange(1, 25), np.log10(diams), masked,
                  cmap=cmap, vmin=Z_LIMITS[0], vmax=Z_LIMITS[1],
                  shading="nearest")
    ax.set_xlim(0.5, 24.5)
    ax.set_ylim(np.log10(diams).min(), np.log10(diams).max())
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.tick_params(direction="out", labelsize=TICK_SIZE)

    # The colour bar the training images carry, in the same place.
    cax = fig.add_axes(_axes_rect(BAR_PX))
    gradient = np.linspace(Z_LIMITS[1], Z_LIMITS[0], 256).reshape(-1, 1)
    cax.imshow(gradient, aspect="auto", cmap=cmap,
               vmin=Z_LIMITS[0], vmax=Z_LIMITS[1],
               extent=(0, 1, Z_LIMITS[0], Z_LIMITS[1]))
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.tick_params(direction="out", labelsize=TICK_SIZE)

    fig.savefig(path, dpi=100, facecolor="white")
    return path

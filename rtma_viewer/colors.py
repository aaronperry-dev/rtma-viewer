"""Colour ramps.

Design rules
------------
* **Fixed physical levels.**  Every hour uses identical levels, so colours mean the
  same thing in every frame, for both products, and at every interval.
* **Interpolated in OKLab**, so the ramps are perceptually even (no muddy mid-tones).
* **Fine discrete bins** (1 °F / 0.5 mph): smooth-looking shading, an exact legend, and
  PNGs that still quantise losslessly to an indexed palette.
* Sequential ramps (wind, gust, dew point) are monotone in lightness.  Temperature
  uses the classic multi-hue operational ramp (-40..120 °F) with a deliberate blue/green break at 32 °F.
* Change maps use ColorBrewer diverging ramps (colour-blind-safe) with a neutral
  middle bin straddling zero.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, from_levels_and_colors, to_rgb

# --------------------------------------------------------------------------- OKLab
_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])
_M1I, _M2I = np.linalg.inv(_M1), np.linalg.inv(_M2)


def srgb_to_oklab(rgb) -> np.ndarray:
    c = np.asarray(rgb, float)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return np.cbrt(lin @ _M1.T) @ _M2.T


def oklab_to_srgb(lab) -> np.ndarray:
    lin = (np.asarray(lab, float) @ _M2I.T) ** 3 @ _M1I.T
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055).clip(0, 1)


class Ramp:
    """Piece-wise ramp defined by (value, hex) anchors, interpolated in OKLab."""

    def __init__(self, stops: list[tuple[float, str]]):
        stops = sorted(stops)
        self.values = np.array([v for v, _ in stops], float)
        self.lab = np.array([srgb_to_oklab(to_rgb(h)) for _, h in stops])

    def __call__(self, v) -> np.ndarray:
        v = np.atleast_1d(np.asarray(v, float))
        lab = np.stack([np.interp(v, self.values, self.lab[:, i]) for i in range(3)], axis=-1)
        return oklab_to_srgb(lab)

    def lightness(self, n: int = 200) -> np.ndarray:
        return np.stack([np.interp(np.linspace(self.values[0], self.values[-1], n), self.values, self.lab[:, 0])])[0]


# --------------------------------------------------------------------------- ramps (physical units)
TEMP_F = Ramp([  # classic -40..120 F operational ramp: magenta -> purple -> blue | green (32 F) -> yellow -> orange -> red -> pink
    (-40, "#ffa4f0"), (-35, "#fd81f0"), (-30, "#fc53f1"), (-25, "#fa30f1"), (-20, "#e31eea"), (-15, "#ce14e3"),
    (-10, "#b208da"), (-5, "#9e0bdc"), (0, "#8517e3"), (5, "#711fe8"), (10, "#5339e7"), (15, "#3957e1"),
    (20, "#197ed8"), (25, "#1d91dc"), (30, "#23a9e1"),
    (31.9, "#24aee2"),                       # <32 F: blue ...
    (32.1, "#25b84a"),                       # ... >=32 F: green - the freezing line is unmistakable
    (35, "#2ac045"), (40, "#64d318"), (45, "#8ee100"), (50, "#baef00"), (55, "#ddfb00"), (60, "#e5e002"),
    (65, "#e0be04"), (70, "#da9306"), (75, "#d56e08"), (80, "#cf3e0a"), (85, "#ca190c"), (90, "#d10f24"),
    (95, "#e12146"), (100, "#f53871"), (105, "#fa3e88"), (110, "#ea288b"), (115, "#dd188e"), (120, "#d20991"),
])

DEWPOINT_F = Ramp([  # dry sand -> green -> teal -> blue -> deep violet (monotone lightness)
    (-20, "#f4ead0"), (10, "#e6dcae"), (25, "#c8d79a"), (35, "#9ccf9c"), (45, "#6dbda5"),
    (55, "#3fa3ad"), (62, "#2a82b3"), (68, "#2f60ab"), (73, "#3d44a0"), (78, "#4d2c85"), (86, "#2f0f52"),
])

WIND_MPH = Ramp([  # calm -> yellow -> orange -> red -> purple (monotone lightness)
    (0, "#f6f7f2"), (5, "#fbe8a6"), (12, "#f9c25c"), (20, "#f28a3a"), (30, "#dc4a30"),
    (40, "#b3203f"), (50, "#7c1d6f"), (60, "#471a73"), (70, "#1f1448"),
])

# ColorBrewer 11-class diverging (low -> high), https://colorbrewer2.org
_BREWER = {
    "RdBu_r": ["#053061", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7",
               "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"],   # blue = cooler, red = warmer
    "BrBG":   ["#543005", "#8c510a", "#bf812d", "#dfc27d", "#f6e8c3", "#f5f5f5",
               "#c7eae5", "#80cdc1", "#35978f", "#01665e", "#003c30"],   # brown = drier, teal = moister
    "PuOr":   ["#7f3b08", "#b35806", "#e08214", "#fdb863", "#fee0b6", "#f7f7f7",
               "#d8daeb", "#b2abd2", "#8073ac", "#542788", "#2d004b"],   # orange = weaker, purple = stronger
}


def _brewer_ramp(name: str) -> Ramp:
    cols = _BREWER[name]
    return Ramp([(i / (len(cols) - 1), c) for i, c in enumerate(cols)])


# --------------------------------------------------------------------------- styles
@dataclass
class Style:
    levels: np.ndarray
    colors: np.ndarray            # (n_bins + n_extend, 3)
    extend: str
    ticks: np.ndarray
    signed: bool = False

    def cmap_norm(self):
        return from_levels_and_colors(self.levels, [tuple(c) for c in self.colors], extend=self.extend)


def _n_extend(extend: str) -> int:
    return {"neither": 0, "min": 1, "max": 1, "both": 2}[extend]


def sequential_style(ramp: Ramp, vmin: float, vmax: float, step: float, tick: float, extend: str) -> Style:
    levels = np.arange(vmin, vmax + step / 2, step)
    centers = levels[:-1] + step / 2
    cols = ramp(centers)
    if extend in ("min", "both"):
        cols = np.vstack([ramp(vmin - 1e-6), cols])
    if extend in ("max", "both"):
        cols = np.vstack([cols, ramp(vmax + 1e-6)])
    return Style(levels, cols, extend, np.arange(vmin, vmax + tick / 2, tick))


def diverging_style(brewer: str, step: float, k: int = 20) -> Style:
    """2k+1 equal bins; the centre bin straddles zero and takes the neutral colour."""
    ramp = _brewer_ramp(brewer)
    levels = (np.arange(-k, k + 2) - 0.5) * step
    centers = np.arange(2 * k + 1) + 0.5
    cols = ramp(centers / (2 * k + 1))
    cols = np.vstack([ramp(0.0), cols, ramp(1.0)])
    return Style(levels, cols, "both", np.arange(-k, k + 1, k // 2) * step, signed=True)


_ABS = {
    "temp": lambda: sequential_style(TEMP_F, -40, 120, 1, 10, "both"),
    "dewpoint": lambda: sequential_style(DEWPOINT_F, -20, 86, 1, 10, "both"),
    "wind": lambda: sequential_style(WIND_MPH, 0, 70, 0.5, 10, "max"),
    "gust": lambda: sequential_style(WIND_MPH, 0, 70, 0.5, 10, "max"),
}

# bin width (in display units) of the change maps; +/- 20 bins either side of zero
_CHANGE_STEP = {
    "temp":     {1: 0.5, 3: 0.75, 6: 1.0, 12: 1.5, 24: 2.0},     # -> +/-10, 15, 20, 30, 40 degF
    "dewpoint": {1: 0.5, 3: 0.75, 6: 1.0, 12: 1.5, 24: 2.0},
    "wind":     {1: 0.5, 3: 0.75, 6: 1.0, 12: 1.25, 24: 1.5},    # -> +/-10, 15, 20, 25, 30 mph
    "gust":     {1: 1.0, 3: 1.25, 6: 1.5, 12: 2.0, 24: 2.0},     # -> +/-20, 25, 30, 40, 40 mph
}
_CHANGE_RAMP = {"temp": "RdBu_r", "dewpoint": "BrBG", "wind": "PuOr", "gust": "PuOr"}


def style_for(var_key: str, interval: int) -> Style:
    if interval == 0:
        return _ABS[var_key]()
    return diverging_style(_CHANGE_RAMP[var_key], _CHANGE_STEP[var_key][interval])

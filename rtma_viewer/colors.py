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
* Change maps use the WPC-style ramp: uneven whole-number levels (±1, 3, 5, 8, 12, 16, 20, 25, 30, 35, 40, 50) with a
  white band for -1..+1, blues for cooling, oranges/reds/magenta for warming.  Identical for every interval
  (1/3/6/12/24 h) and every variable, so a colour always means the same amount of change.
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

# --------------------------------------------------------------------------- styles
@dataclass
class Style:
    levels: np.ndarray
    colors: np.ndarray            # (n_bins + n_extend, 3)
    extend: str
    ticks: np.ndarray
    signed: bool = False
    uniform: bool = False          # legend with equal-width blocks (uneven levels)
    tick_fontsize: float = 12.5

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


# WPC-style change ramp, same for every interval (display units, e.g. °F or mph): 24 boundaries -> 23 bins + under/over colours.
_WPC_LEVELS = np.array([-50, -40, -35, -30, -25, -20, -16, -12, -8, -5, -3, -1,
                        1, 3, 5, 8, 12, 16, 20, 25, 30, 35, 40, 50], float)
_WPC_COLORS = [
    "#fa96c8",                                                       # below -50
    "#c670a8", "#934b89", "#60256a", "#1e1e78", "#162f99", "#0f41bb",  # -50 .. -16   (pink-violet -> navy -> blue)
    "#0752dd", "#0064ff", "#3f8aff", "#7fb1ff", "#bfd8ff",            # -16 .. -1    (blue -> pale blue)
    "#ffffff",                                                        # -1 .. +1     (no meaningful change)
    "#fdbb84", "#fc8d59", "#ef6548", "#d7301f", "#b30000", "#67001f",  # +1 .. +20    (peach -> orange -> red -> maroon)
    "#980043", "#ce1256", "#e7298a", "#df65b0", "#c994c7",            # +20 .. +50   (crimson -> magenta -> orchid)
    "#d4b9da",                                                        # above +50
]

def change_style() -> Style:
    cols = np.array([to_rgb(c) for c in _WPC_COLORS])
    return Style(_WPC_LEVELS.copy(), cols, "both", _WPC_LEVELS.copy(), signed=True, uniform=True, tick_fontsize=10.5)


_ABS = {
    "temp": lambda: sequential_style(TEMP_F, -40, 120, 1, 10, "both"),
    "dewpoint": lambda: sequential_style(DEWPOINT_F, -20, 86, 1, 10, "both"),
    "wind": lambda: sequential_style(WIND_MPH, 0, 70, 0.5, 10, "max"),
    "gust": lambda: sequential_style(WIND_MPH, 0, 70, 0.5, 10, "max"),
}

def style_for(var_key: str, interval: int) -> Style:
    if interval == 0:
        return _ABS[var_key]()
    return change_style()

"""Colour ramps.

Design rules
------------
* **Fixed physical levels.**  Every hour uses identical levels, so colours mean the
  same thing in every frame, for both products, and at every interval.
* **Interpolated in OKLab**, so the ramps are perceptually even (no muddy mid-tones).
* **Fine discrete bins** (1 °F for temperature and dew point): smooth-looking shading, an exact legend, and
  PNGs that still quantise losslessly to an indexed palette.
* Dew point is a sequential ramp, monotone in lightness.  Temperature uses the classic multi-hue operational ramp
  (-40..120 °F) with a deliberate blue/green break at 32 °F.  Wind speed/gust and apparent temperature use the NWS
  legends (sampled from NWS Gray/Portland maps): whole-number bins, exact legend colours.
* Change maps use the WPC-style ramp: uneven whole-number levels (±1, 3, 5, 8, 12, 16, 20, 25, 30, 35, 40, 50) with a
  white band for -1..+1, blues for cooling, oranges/reds/magenta for warming.  Identical for every interval
  (1/3/6/12/24 h) and every variable, so a colour always means the same amount of change.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.colors import from_levels_and_colors, to_hex, to_rgb

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

# Station (surface) pressure, mb: mostly a map of elevation, so a plain sequential ramp (monotone lightness): dark = high ground.
PRESSURE_MB = Ramp([
    (800, "#2b1a5a"), (860, "#33408f"), (920, "#2d6fb0"), (980, "#2d9bb0"), (1010, "#62c29a"),
    (1030, "#b4df8d"), (1050, "#f3f0a0"),
])

# Mean sea-level pressure, mb: diverging about ~1013 (violet/blue lows -> pale -> amber/red highs)
MSLP_MB = Ramp([
    (940, "#4a1470"), (960, "#7a2d9c"), (980, "#3a4fc0"), (996, "#4a90d9"), (1006, "#a6d3ea"),
    (1013, "#f1f1ee"), (1020, "#f8e29a"), (1030, "#f0a24e"), (1042, "#d4582a"), (1060, "#7e1b1b"),
])

# Relative humidity, %: dry brown -> tan -> pale -> green -> blue (saturated)
RH_PCT = Ramp([
    (0, "#7f3b08"), (20, "#d29a52"), (40, "#f0e0ae"), (55, "#cfe6bf"), (70, "#8dccb0"), (85, "#3a9fb8"), (100, "#1f4e9c"),
])

# Dew point depression (T - Td), °F: saturated = deep blue so fog-prone air stands out; dry = warm tones
DPDEP_F = Ramp([
    (0, "#0d3b8c"), (3, "#2e86c1"), (6, "#7fc8d8"), (10, "#d9ecc8"), (15, "#f3e3a6"), (20, "#eeb872"), (30, "#d9734a"),
    (40, "#8a2d1d"),
])

# Total cloud cover, %: clear = deep blue, broken = pale, overcast = dark grey
CLOUD_PCT = Ramp([(0, "#0b4f9c"), (25, "#5b9bd5"), (50, "#cfd8e0"), (75, "#8f9aa6"), (100, "#4b5560")])

# Terrain height, ft
TERRAIN_FT = Ramp([
    (0, "#9ccf9a"), (1000, "#e6e3a0"), (2000, "#d9b073"), (3000, "#b0804e"), (4000, "#8a6a56"), (5000, "#cfc7c2"),
    (6000, "#f4f1ee"),
])
_SEA = "#bcd9ea"

# --------------------------------------------------------------------------- NWS legends (sampled from NWS Gray/Portland maps)
# Wind speed / gust (mph): NWS "Maximum Wind Gust" legend.  Edges are the whole-number bin starts of the legend labels
# (0-5, 6-10, 11-15, ... 81-100, 101-120, 121-140, 141-150).  Same colours for sustained wind and gusts.
_NWS_WIND_LEVELS = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 120, 140, 150], float)
_NWS_WIND_COLORS = ["#103f78", "#225ea8", "#1d91c0", "#41b6c4", "#7fcdbb", "#b4d79e", "#dfff9e", "#ffffa6",
                    "#ffe873", "#ffc400", "#ffaa00", "#ff5900", "#ff0000", "#a80000", "#6e0000", "#ffbee8", "#ff73df"]

# Apparent temperature (°F): NWS "24-Hour Minimum Wind Chill Index Temperature" legend (<-60 .. 35+, 5 °F bins) joined to the
# NWS "24-Hour Maximum Heat Index Temperature" legend (75-79 .. 120+, 5 °F bins).  Neither legend covers 35-75 °F (wind chill is
# flat 35+, heat index is a single grey below 75), so that span is a smooth OKLab bridge from the wind-chill teal to the
# first heat-index yellow.
_WC_COLORS = ["#a51453",                                                      # < -60
              "#e2266a", "#fb3d9e", "#f379c4", "#ff87f3", "#ffd2fc",          # -60 .. -35
              "#ffffff", "#eeeeff", "#d0d1f0", "#b2aedc", "#897fc5", "#683ba3", "#211491",   # -35 .. 0
              "#2151b0", "#147ad6", "#3db2ff", "#5edbff", "#87ebff", "#c1ffff",              # 0 .. 30
              "#44e3d6"]                                                      # 30 .. 35
_WC_TEAL = "#14adaa"                                                          # 35+
_HI_COLORS = ["#ffffb4", "#ffed8a", "#ffc23e", "#ffa150", "#ff623e", "#f72e30", "#c5143a", "#94143a", "#6d1438"]  # 75 .. 120
_HI_TOP = "#3c143c"                                                           # 120+


def _apparent_colors() -> tuple[np.ndarray, list[str]]:
    levels = np.arange(-60, 121, 5, dtype=float)                              # 36 bins
    bridge = Ramp([(37.5, _WC_TEAL), (77.5, _HI_COLORS[0])])
    mid = [to_hex(c) for c in bridge(np.arange(40, 75, 5) + 2.5)]            # 40-45 .. 70-75
    cols = _WC_COLORS + [_WC_TEAL] + mid + _HI_COLORS + [_HI_TOP]             # under + 36 bins + over
    assert len(cols) == len(levels) + 1
    return levels, cols


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
    tick_labels: dict | None = None   # {tick value: text} instead of the number (e.g. compass points)

    def cmap_norm(self):
        cmap, norm = from_levels_and_colors(self.levels, [tuple(c) for c in self.colors], extend=self.extend)
        # matplotlib treats a value *equal to the top level* as out of range, and with extend="neither" that is
        # transparent: RH / cloud cover capped at exactly 100 %, wind direction of exactly 360 deg would punch holes.
        # Out-of-range values take the nearest end colour instead.
        extremes = {}
        if self.extend in ("neither", "min"):
            extremes["over"] = tuple(self.colors[-1])
        if self.extend in ("neither", "max"):
            extremes["under"] = tuple(self.colors[0])
        return (cmap.with_extremes(**extremes) if extremes else cmap), norm


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

# Pressure changes are an order of magnitude smaller than temperature changes (a 3-hour fall of 4 mb is a fast one), so
# pressure / MSLP use the same 25 colours on finer levels (same symmetric shape and white centre).
_PRESSURE_CHANGE_LEVELS = np.array([-30, -24, -20, -16, -12, -9, -6, -4, -3, -2, -1, -0.5,
                                    0.5, 1, 2, 3, 4, 6, 9, 12, 16, 20, 24, 30], float)
_PRESSURE_KEYS = ("pressure", "mslp")


def change_style(var_key: str | None = None) -> Style:
    cols = np.array([to_rgb(c) for c in _WPC_COLORS])
    levels = _PRESSURE_CHANGE_LEVELS if var_key in _PRESSURE_KEYS else _WPC_LEVELS
    return Style(levels.copy(), cols, "both", levels.copy(), signed=True, uniform=True, tick_fontsize=10.5)


def wind_style() -> Style:
    cols = np.array([to_rgb(c) for c in _NWS_WIND_COLORS] + [to_rgb(_NWS_WIND_COLORS[-1])])   # over = top colour
    return Style(_NWS_WIND_LEVELS.copy(), cols, "max", _NWS_WIND_LEVELS.copy(), uniform=True, tick_fontsize=10.5)


def apparent_style() -> Style:
    levels, cols = _apparent_colors()
    return Style(levels, np.array([to_rgb(c) for c in cols]), "both", np.arange(-60, 121, 10, dtype=float),
                 tick_fontsize=11)


def _custom(levels, hex_colors, extend, ticks=None, tick_fontsize=10.5, uniform=True, tick_labels=None) -> Style:
    levels = np.asarray(levels, float)
    n = len(levels) - 1 + (1 if extend in ("min", "max") else 2 if extend == "both" else 0)
    assert len(hex_colors) == n, (len(hex_colors), n)
    return Style(levels, np.array([to_rgb(c) for c in hex_colors]), extend,
                 levels if ticks is None else np.asarray(ticks, float), uniform=uniform,
                 tick_fontsize=tick_fontsize, tick_labels=tick_labels)


def visibility_style() -> Style:
    """Statute miles: <1/4, 1/2, 1, 2, 3, 5, 7, 9.5 and unrestricted (RTMA caps near 10 mi)."""
    return _custom([0, 0.25, 0.5, 1, 2, 3, 5, 7, 9.5],
                   ["#8e0152", "#c51b7d", "#e34a33", "#fc8d59", "#fdbb84", "#fee08b", "#d9ef8b", "#a6d96a", "#e8f1f5"], "max")


def ceiling_style() -> Style:
    """Feet AGL, aviation-flavoured: <200, 500 (LIFR), 1000 (IFR), 2000-3000 (MVFR), 5000, 8000, 12000+ (clear)."""
    return _custom([0, 200, 500, 1000, 2000, 3000, 5000, 8000, 12000],
                   ["#7a0045", "#c2185b", "#e53935", "#fb8c00", "#fdd835", "#aed581", "#4fc3f7", "#9fa8da", "#eceff1"], "max")


def terrain_style() -> Style:
    levels = [0, 20] + list(range(250, 6001, 250))                       # first bin is sea level
    centers = (np.array(levels[1:-1]) + np.array(levels[2:])) / 2
    cols = [_SEA] + [to_hex(c) for c in TERRAIN_FT(centers)] + [to_hex(TERRAIN_FT(6000.0)[0])]
    return _custom(levels, cols, "max", ticks=np.arange(0, 6001, 1000), tick_fontsize=11, uniform=False)


_COMPASS = {0: "N", 45: "NE", 90: "E", 135: "SE", 180: "S", 225: "SW", 270: "W", 315: "NW", 360: "N"}
_DIR_RAMP = Ramp([(0, "#3b6fd8"), (45, "#35b6c9"), (90, "#3fbf6b"), (135, "#c8d948"), (180, "#f0a03a"),
                  (225, "#e5533d"), (270, "#c0449a"), (315, "#7a4fcf"), (360, "#3b6fd8")])


def direction_style() -> Style:
    """Wind direction in 16 compass sectors (N is split across 0/360 so every bin is a true sector), cyclic colours."""
    levels = [0.0] + [11.25 + 22.5 * i for i in range(16)] + [360.0]              # 0, 11.25, 33.75 ... 348.75, 360
    centers = np.array([0.0] + [22.5 * i for i in range(1, 16)] + [360.0])        # N | NNE ... NNW | N
    return _custom(levels, [to_hex(c) for c in _DIR_RAMP(centers)], "neither", ticks=sorted(_COMPASS), tick_fontsize=11,
                   uniform=False, tick_labels=_COMPASS)


_ABS = {
    "temp": lambda: sequential_style(TEMP_F, -40, 120, 1, 10, "both"),
    "dewpoint": lambda: sequential_style(DEWPOINT_F, -20, 86, 1, 10, "both"),
    "wind": wind_style,
    "gust": wind_style,
    "apparent": apparent_style,
    "wetbulb": lambda: sequential_style(TEMP_F, -40, 120, 1, 10, "both"),    # same legend as 2 m temperature: directly comparable
    "rh": lambda: sequential_style(RH_PCT, 0, 100, 2, 10, "neither"),
    "dpdep": lambda: sequential_style(DPDEP_F, 0, 40, 1, 5, "max"),
    "wdir": direction_style,
    "pressure": lambda: sequential_style(PRESSURE_MB, 800, 1050, 2, 25, "both"),
    "mslp": lambda: sequential_style(MSLP_MB, 940, 1060, 2, 10, "both"),
    "vis": visibility_style,
    "ceiling": ceiling_style,
    "cloud": lambda: sequential_style(CLOUD_PCT, 0, 100, 5, 10, "neither"),
    "terrain": terrain_style,
}

def style_for(var_key: str, interval: int) -> Style:
    if interval == 0:
        return _ABS[var_key]()
    return change_style(var_key)

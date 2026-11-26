"""Colour ramps.

Design rules
------------
* **Fixed physical levels.**  Every hour uses identical levels, so colours mean the
  same thing in every frame, for both products, and at every interval.
* **Interpolated in OKLab**, so the ramps are perceptually even (no muddy mid-tones).
* **Discrete bins with exact legends**, and PNGs that still quantise losslessly to an indexed palette.
* Temperature, dew point, wet bulb, sea-level temperature and the 24 h max / min temperature all share one legend: the WPC
  max/min temperature forecast ramp (-40..100 °F in 5 °F bins, with colours for below -40 and above 100), sampled from the
  WPC Day 3 max / Day 4 min temperature maps.  Wind speed/gust and apparent temperature use the NWS legends (sampled from NWS
  Gray/Portland maps): whole-number bins, exact legend colours.
* Temperature-type change maps (temperature, dew point, wet bulb, apparent temperature, dew point depression; 1/3/6/12/24 h)
  use the NWS NDFD "departure from climatology" ramp: uneven whole-number levels (±1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 22, 26,
  30) with a white band for -1..+1, yellows/oranges/reds/magentas for warming and greens/cyans/blues/violet for cooling.
* Other change maps (wind, gust, pressure, humidity ...) use the WPC-style ramp: uneven whole-number levels (±1, 3, 5, 8,
  12, 16, 20, 25, 30, 35, 40, 50) with a white band for -1..+1, blues for cooling, oranges/reds/magenta for warming.
  Each ramp is identical for every interval, so a colour always means the same amount of change.
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

# 24-hour temperature range, °F: grey-green (little change) -> straw -> orange -> brick (big radiational swing)
RANGE_F = Ramp([(0, "#f3f1e8"), (10, "#cfe3c0"), (20, "#e8d78a"), (30, "#e3a24f"), (40, "#a8412d")])

# Hours at or below freezing in 24 h: warm cream (none) -> pale blue -> blue -> deep indigo (frozen all day)
FREEZE_HOURS = Ramp([(0, "#f6f1e4"), (6, "#bfe0ea"), (12, "#58a6d6"), (18, "#2d5fb0"), (24, "#2b1d70")])

# Number of freezing-point crossings in 24 h
CROSSINGS = Ramp([(0, "#f6f1e4"), (1, "#f4dc9a"), (2, "#ec9a4f"), (4, "#c4412f"), (8, "#6e1a4e")])

# Relative humidity with respect to ice, %: the RH ramp up to 100, then a hard break to violet for ice-supersaturated air
RH_ICE_PCT = Ramp([(0, "#7f3b08"), (20, "#d29a52"), (40, "#f0e0ae"), (55, "#cfe6bf"), (70, "#8dccb0"), (85, "#3a9fb8"),
                   (99.9, "#1f4e9c"), (100.1, "#c46adb"), (120, "#8a2fb0"), (140, "#3d0a63")])

# Water-vapour mixing ratio, g/kg: dry sand -> green -> teal -> blue -> violet (same family as the dew point ramp)
MIXING_G_KG = Ramp([(0, "#f4ead0"), (1, "#d7dca0"), (2, "#9ccf9c"), (4, "#4fb0a8"), (7, "#2a82b3"), (10, "#3d44a0"),
                    (16, "#2f0f52")])

# Pressure-gradient magnitude, mb / 100 km: near-white (slack) -> blue -> violet -> red (tight)
PGRAD_MB = Ramp([(0, "#f7f7f7"), (1, "#a8d3e6"), (2, "#4a90c9"), (3, "#7b4fb0"), (4, "#d6403a"), (5, "#6e0d1a")])

# Signed diagnostics (advection, convergence, vorticity, frontogenesis, terrain lift): blue negative, white about zero, warm positive
_DIV_RAMP = Ramp([(-1, "#0a2f8c"), (-0.75, "#2f6fe0"), (-0.4, "#8db7ff"), (-0.12, "#dbe8ff"), (0, "#ffffff"),
                  (0.12, "#ffe6cf"), (0.4, "#fdae6b"), (0.75, "#e0442c"), (1, "#7a0a2a")])

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

# WPC max / min temperature forecast legend (°F): 5 °F bins from -40 to 100 plus an under (< -40) and over (> 100) colour.
# Sampled from the WPC Day 3 maximum / Day 4 minimum temperature forecast maps.  A bin's colour is for values from its lower
# level up to the next one (30..35 °F is the first teal).
_TEMP_LEVELS = np.arange(-40, 101, 5, dtype=float)                                     # 29 edges -> 28 bins
_TEMP_COLORS = [
    "#ff71de",                                                                          # below -40
    "#ffbee7", "#ffffff", "#d6dbe7", "#bdbede", "#9c9ac6", "#7369ad", "#52288c", "#10007b",   # -40 .. 0
    "#103c9c", "#0065c6", "#299eff", "#4ac7ff", "#73d7ff", "#adffff",                  # 0 .. 30
    "#31cfc6", "#009a94", "#10555a", "#086d29", "#31a252", "#73c373", "#a5d79c", "#d6ffbd",   # 30 .. 70
    "#ffffb5", "#ffeb9c", "#ffcf73", "#ffae29", "#ff8e39", "#ff4d29",                  # 70 .. 100
    "#e71818",                                                                          # above 100
]

# NWS NDFD "departure from climatology" legend (°F), used for temperature differences / changes: 25 bins on 26 edges plus
# an under (< -30) and over (> 30) colour.  White for -1..+1; warm colours for positive, green -> blue -> violet for negative.
_DEPARTURE_LEVELS = np.array([-30, -26, -22, -18, -15, -12, -10, -8, -6, -4, -3, -2, -1,
                              1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 22, 26, 30], float)
_DEPARTURE_COLORS = [
    "#dc2ece",                                                                          # below -30
    "#8001fd", "#0432fd", "#035afa", "#047ffa", "#05befd", "#07fbfd", "#06e5ce", "#01cb80",   # -30 .. -6
    "#067a04", "#02a103", "#02c402", "#04dd04",                                         # -6 .. -1
    "#ffffff",                                                                          # -1 .. +1
    "#cbe607", "#fffe08", "#fedd06", "#fcbf05", "#fda005", "#fb8406", "#ff5506",        # +1 .. +12
    "#f90201", "#da0205", "#de0166", "#dd0396", "#e55ad8",                              # +12 .. +30
    "#db2ecf",                                                                          # above +30
]
# variables (°F) whose change maps use the departure ramp
_TEMP_CHANGE_KEYS = ("temp", "dewpoint", "wetbulb", "apparent", "dpdep")

# Pressure changes are an order of magnitude smaller than temperature changes (a 3-hour fall of 4 mb is a fast one), so
# pressure / MSLP use the same 25 colours on finer levels (same symmetric shape and white centre).
_PRESSURE_CHANGE_LEVELS = np.array([-30, -24, -20, -16, -12, -9, -6, -4, -3, -2, -1, -0.5,
                                    0.5, 1, 2, 3, 4, 6, 9, 12, 16, 20, 24, 30], float)
_PRESSURE_KEYS = ("pressure", "mslp")


def temperature_style() -> Style:
    return _custom(_TEMP_LEVELS, _TEMP_COLORS, "both", ticks=np.arange(-40, 101, 10), tick_fontsize=11, uniform=False)


def change_style(var_key: str | None = None) -> Style:
    if var_key in _TEMP_CHANGE_KEYS:
        levels, hexes = _DEPARTURE_LEVELS, _DEPARTURE_COLORS
    else:
        levels, hexes = _PRESSURE_CHANGE_LEVELS if var_key in _PRESSURE_KEYS else _WPC_LEVELS, _WPC_COLORS
    cols = np.array([to_rgb(c) for c in hexes])
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


def diverging_style(levels) -> Style:
    """Signed field on uneven, symmetric levels (an even number of them: the bin straddling zero is white).  Colours step
    evenly along the diverging ramp, one per bin, so the legend is a row of equal blocks."""
    levels = np.asarray(levels, float)
    assert len(levels) % 2 == 0 and np.allclose(levels, -levels[::-1]), "need symmetric levels with an even count"
    cols = _DIV_RAMP(np.linspace(-1, 1, len(levels) + 1))
    return Style(levels.copy(), cols, "both", levels.copy(), signed=True, uniform=True, tick_fontsize=10.5)


def categorical_style(hex_colors, labels) -> Style:
    """Integer-coded classes 0..n-1: one block per class, labelled in the legend."""
    n = len(hex_colors)
    return _custom(np.arange(n + 1) - 0.5, hex_colors, "neither", ticks=np.arange(n), tick_fontsize=11,
                   tick_labels={i: t for i, t in enumerate(labels)})


def flight_category_style() -> Style:
    return categorical_style(["#c43bc4", "#e63946", "#3a6fe0", "#4cb86b"], ["LIFR", "IFR", "MVFR", "VFR"])


def frostbite_style() -> Style:
    return categorical_style(["#e7edf2", "#9ecbea", "#3c7fcf", "#6a1b9a"], ["None", "30 min", "10 min", "5 min"])


def crossings_style() -> Style:
    """Integer counts 0..8+ with the tick under the middle of each block."""
    st = sequential_style(CROSSINGS, -0.5, 8.5, 1, 1, "max")
    st.ticks = np.arange(0, 9, dtype=float)
    return st


def mixing_ratio_style() -> Style:
    levels = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16]
    centers = np.array(levels[:-1]) + np.diff(levels) / 2
    cols = [to_hex(c) for c in MIXING_G_KG(centers)] + [to_hex(MIXING_G_KG(16.0)[0])]
    return _custom(levels, cols, "max", ticks=[0, 0.5, 1, 2, 3, 4, 6, 8, 12, 16], tick_fontsize=10.5)


_ABS = {
    "temp": temperature_style,
    "dewpoint": temperature_style,
    "wind": wind_style,
    "gust": wind_style,
    "apparent": apparent_style,
    "wetbulb": temperature_style,
    "rh": lambda: sequential_style(RH_PCT, 0, 100, 2, 10, "neither"),
    "dpdep": lambda: sequential_style(DPDEP_F, 0, 40, 1, 5, "max"),
    "wdir": direction_style,
    "pressure": lambda: sequential_style(PRESSURE_MB, 800, 1050, 2, 25, "both"),
    "mslp": lambda: sequential_style(MSLP_MB, 940, 1060, 2, 10, "both"),
    "vis": visibility_style,
    "ceiling": ceiling_style,
    "cloud": lambda: sequential_style(CLOUD_PCT, 0, 100, 5, 10, "neither"),
    "terrain": terrain_style,
    # derived winter diagnostics
    "slt": temperature_style,
    "rhi": lambda: sequential_style(RH_ICE_PCT, 0, 140, 2, 20, "max"),
    "mixr": mixing_ratio_style,
    "tmax24": temperature_style,
    "tmin24": temperature_style,
    "trange24": lambda: sequential_style(RANGE_F, 0, 45, 1, 5, "max"),
    "frzhrs24": lambda: sequential_style(FREEZE_HOURS, 0, 24, 1, 3, "neither"),
    "ftx24": crossings_style,
    "wcmin24": apparent_style,                                              # NWS "24-Hour Minimum Wind Chill" legend
    "gustmax24": wind_style,
    "frostbite": frostbite_style,
    "pgrad": lambda: sequential_style(PGRAD_MB, 0, 5, 0.25, 1, "max"),
    "fltcat": flight_category_style,
    "tadv": lambda: diverging_style([-30, -20, -12, -8, -5, -3, -1, 1, 3, 5, 8, 12, 20, 30]),
    "dadv": lambda: diverging_style([-30, -20, -12, -8, -5, -3, -1, 1, 3, 5, 8, 12, 20, 30]),
    "conv": lambda: diverging_style([-80, -50, -30, -20, -12, -6, -3, 3, 6, 12, 20, 30, 50, 80]),
    "vort": lambda: diverging_style([-80, -50, -30, -20, -12, -6, -3, 3, 6, 12, 20, 30, 50, 80]),
    "fgen": lambda: diverging_style([-40, -25, -15, -8, -4, -2, 2, 4, 8, 15, 25, 40]),
    "upslope": lambda: diverging_style([-150, -100, -60, -40, -25, -15, -8, -3, 3, 8, 15, 25, 40, 60, 100, 150]),
}

def style_for(var_key: str, interval: int) -> Style:
    if interval == 0:
        return _ABS[var_key]()
    return change_style(var_key)

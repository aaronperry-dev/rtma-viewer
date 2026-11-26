"""Central configuration: products, variables, region, cities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .derived import (apparent_temperature, dewpoint_depression, mslp, relative_humidity_field, wet_bulb,
                      wind_direction)


# --------------------------------------------------------------------------- products
@dataclass(frozen=True)
class Product:
    key: str
    label: str
    long_name: str
    bucket: str          # public NOAA Open Data bucket (anonymous access)
    prefix: str          # e.g. "rtma2p5" -> rtma2p5.YYYYMMDD/rtma2p5.tHHz.2dvaranl_ndfd.grb2*
    lookback_hours: int  # how far back to search for the newest published cycle
    blurb: str


PRODUCTS: dict[str, Product] = {
    "rtma": Product(
        "rtma", "RTMA", "Real-Time Mesoscale Analysis", "noaa-rtma-pds", "rtma2p5", 6,
        "NOAA/NCEP 2.5 km hourly surface analysis. Available ~35-45 min after the valid hour.",
    ),
    "urma": Product(
        "urma", "URMA", "UnRestricted Mesoscale Analysis", "noaa-urma-pds", "urma2p5", 12,
        "Same 2.5 km grid as RTMA, but run later with more observations. Lags RTMA by a few hours.",
    ),
}

# --------------------------------------------------------------------------- unit conversions (GRIB SI -> display)
FT_PER_M = 3.280839895
CEIL_CAP_FT = 12000.0        # RTMA reports "no ceiling" as 16 000 m; everything >= 12 000 ft is shown as clear / unlimited


def _k_to_f(a):
    return (a - 273.15) * 1.8 + 32.0


def _ms_to_mph(a):
    return a * 2.2369362921


def _pa_to_mb(a):
    return a / 100.0


def _m_to_ft(a):
    return a * FT_PER_M


def _m_to_mi(a):
    return a / 1609.344


def _ceil_m_to_ft(a):
    return np.minimum(a * FT_PER_M, CEIL_CAP_FT)


def _ceil_label(v: float) -> str:
    return "CLR" if v >= CEIL_CAP_FT - 1 else str(int(round(v / 100.0)) * 100)


# --------------------------------------------------------------------------- variables
G_TEMP, G_WIND, G_PRESS, G_SKY, G_TERRAIN = "Temperature & moisture", "Wind", "Pressure", "Sky & visibility", "Terrain"


@dataclass(frozen=True)
class Variable:
    key: str
    label: str            # short label for tabs
    long_name: str        # plot title
    unit: str             # display unit
    grib_var: str         # abbreviation in the .idx file
    grib_level: str       # level string in the .idx file
    convert: Callable = field(compare=False, default=lambda a: a)
    components: tuple[str, ...] = ()   # fallback: hypot of these 10 m components if grib_var is absent
    requires: tuple[str, ...] = ()     # derived variable: keys of the source variables it is computed from ...
    derive: Callable | None = field(compare=False, default=None)   # ... as derive(*fields) in display units
    # ---- presentation / pipeline behaviour
    group: str = G_TEMP                # section of the variable picker
    changes: bool = True               # also publish the 1/3/6/12/24 h change maps (False: current field only)
    static: bool = False               # time-invariant (terrain): fetched once per product, never pruned
    pack: tuple[float, float] | None = None   # cache as uint16 = round((x - offset) * scale); default float16
    decimals: int = 0                  # digits on the station labels of current-field maps ...
    chg_decimals: int | None = None    # ... and of change maps (None: same as ``decimals``)
    labeler: Callable | None = field(compare=False, default=None)   # custom station-label text for current fields
    circular: bool = False             # angle in degrees: no min/max, vector-mean average
    isobar_step: float | None = None   # draw smoothed contour lines (mb) over the current-field map
    hidden: bool = False               # fetched / cached as an input only: no map of its own, not listed on the page
    label_max_terrain_m: float | None = None   # no station value where the analysis terrain is higher (reduction not meaningful)
    note: str = field(compare=False, default="")                    # one-paragraph "how to read this" for the page


VARIABLES: dict[str, Variable] = {
    # ------------------------------------------------------------------ temperature & moisture
    "temp": Variable("temp", "Temperature", "2m Temperature", "°F", "TMP", "2 m above ground", _k_to_f),
    "dewpoint": Variable("dewpoint", "Dew point", "2m Dew Point", "°F", "DPT", "2 m above ground", _k_to_f),
    # derived (no GRIB message of its own): NDFD apparent temperature = wind chill / ambient / heat index
    "apparent": Variable("apparent", "Apparent temp", "Apparent Temperature", "°F", "", "",
                         requires=("temp", "dewpoint", "wind"), derive=apparent_temperature,
                         note="NWS wind chill at or below 50 °F (wind above 3 mph), heat index above 80 °F, otherwise the air temperature."),
    "wetbulb": Variable("wetbulb", "Wet bulb", "2m Wet-Bulb Temperature", "°F", "", "",
                        requires=("temp", "dewpoint", "pressure"), derive=wet_bulb,
                        note="The temperature falling precipitation can cool the air toward as it evaporates; computed from 2 m "
                             "temperature, dew point and the real surface pressure (with respect to liquid water). Rule of thumb: "
                             "precipitation falling into a surface wet bulb at or below 32 °F tends to stay frozen, well above "
                             "34-35 °F favours rain, and in between is marginal or mixed. It is not a precipitation forecast."),
    "rh": Variable("rh", "Rel. humidity", "2m Relative Humidity", "%", "", "",
                   requires=("temp", "dewpoint"), derive=relative_humidity_field,
                   note="Computed from 2 m temperature and dew point, with respect to liquid water."),
    "dpdep": Variable("dpdep", "Dew pt. depression", "2m Dew Point Depression (T - Td)", "°F", "", "",
                      requires=("temp", "dewpoint"), derive=dewpoint_depression,
                      note="Temperature minus dew point. Near 0 the air is saturated (fog, low cloud, rime); large values mean dry "
                           "air, where falling precipitation can evaporate and cool the surface layer."),
    # ------------------------------------------------------------------ wind
    "wind": Variable("wind", "Wind speed", "10m Wind Speed", "mph", "WIND", "10 m above ground", _ms_to_mph,
                     components=("UGRD", "VGRD"), group=G_WIND),
    "gust": Variable("gust", "Wind gust", "10m Wind Gust", "mph", "GUST", "10 m above ground", _ms_to_mph, group=G_WIND),
    "wdir_raw": Variable("wdir_raw", "Wind direction (raw)", "10m Wind Direction (raw)", "°", "WDIR", "10 m above ground",
                         group=G_WIND, changes=False, hidden=True),
    "wdir": Variable("wdir", "Wind direction", "10m Wind Direction", "°", "", "", requires=("wdir_raw", "wind"),
                     derive=wind_direction, group=G_WIND, changes=False, circular=True,
                     note="Direction the wind blows from (meteorological convention: 90° = east wind). Calm air (under 2 mph) has "
                          "no direction and is left white. The average is a vector mean."),
    # ------------------------------------------------------------------ pressure
    "pressure": Variable("pressure", "Surface pressure", "Surface Pressure", "mb", "PRES", "surface", _pa_to_mb,
                         group=G_PRESS, pack=(700.0, 100.0), chg_decimals=1,
                         note="Actual pressure at the analysis grid's terrain height (not reduced to sea level), so it mostly maps "
                              "elevation. Use it for tendencies (change maps); use MSLP to see weather systems."),
    "mslp": Variable("mslp", "MSLP", "Mean Sea-Level Pressure", "mb", "", "",
                     requires=("pressure", "terrain", "temp", "dewpoint"), derive=mslp, group=G_PRESS, chg_decimals=1,
                     isobar_step=2.0, label_max_terrain_m=1000.0,
                     note="RTMA publishes no sea-level pressure, so this is reduced from surface pressure with the hypsometric "
                          "equation (virtual temperature, 6.5 K/km column below the surface). Against METAR sea-level pressure at "
                          "New England airports it agrees to well under 1 mb. Over high terrain and cold pools the reduction is "
                          "less certain, so isobars (every 2 mb, labelled every 4) are drawn from a lightly smoothed field and "
                          "summit stations above 1,000 m are not labelled. "
                          "Tight isobar packing means strong gradient winds; change maps show falling or rising pressure."),
    # ------------------------------------------------------------------ sky & visibility
    "vis": Variable("vis", "Visibility", "Surface Visibility", "mi", "VIS", "surface", _m_to_mi, group=G_SKY,
                    changes=False, decimals=1,
                    note="Statute miles. The analysis tops out near 10 mi (16 km), so 9.5+ means unrestricted. Low values mark fog, "
                         "snow squalls and blowing snow."),
    "ceiling": Variable("ceiling", "Cloud ceiling", "Cloud Ceiling (AGL)", "ft", "CEIL", "cloud ceiling", _ceil_m_to_ft,
                        group=G_SKY, changes=False, labeler=_ceil_label,
                        note="Height of the lowest ceiling above ground, in feet; 12,000 ft or higher (or no ceiling) is shown as CLR. "
                             "Flight categories: LIFR below 500 ft, IFR below 1,000 ft, MVFR below 3,000 ft."),
    "cloud": Variable("cloud", "Cloud cover", "Total Cloud Cover", "%", "TCDC",
                      "entire atmosphere (considered as a single layer)", group=G_SKY, changes=False,
                      note="Total sky cover. Clear skies with light wind (especially over snow cover) are the setup for the coldest "
                           "nights and frost hollows; overcast keeps nights milder."),
    # ------------------------------------------------------------------ terrain (static)
    "terrain": Variable("terrain", "Terrain height", "Analysis Terrain Height", "ft", "HGT", "surface", _m_to_ft,
                        group=G_TERRAIN, changes=False, static=True,
                        note="The 2.5 km analysis grid's smoothed terrain, which is what the analysis (and MSLP reduction) "
                             "uses. Real summits are higher and valleys lower. Useful for reading elevation effects, upslope "
                             "flow and cold-air pooling."),
}
SOURCE_VARIABLES: dict[str, Variable] = {k: v for k, v in VARIABLES.items() if v.derive is None}
PANEL_VARIABLES: dict[str, Variable] = {k: v for k, v in VARIABLES.items() if not v.hidden}     # what gets maps / a page entry

# Current field + change versus N hours earlier
INTERVALS: tuple[int, ...] = (0, 1, 3, 6, 12, 24)


def panel_hours(var: Variable) -> tuple[int, ...]:
    """Intervals a variable gets a map for: all of them, or just the current field."""
    return INTERVALS if var.changes else (0,)


def hours_needed() -> dict[str, tuple[int, ...]]:
    """For every *source* variable, the hours-before-newest-cycle it must be loaded for (own panels + derived ones)."""
    need: dict[str, set[int]] = {}
    for var in VARIABLES.values():
        for key in (var.requires if var.derive is not None else (var.key,)):
            hrs = (0,) if SOURCE_VARIABLES[key].static else panel_hours(var)
            need.setdefault(key, set()).update(hrs)
    return {k: tuple(sorted(v)) for k, v in need.items()}


# --------------------------------------------------------------------------- region
# lon_min, lon_max, lat_min, lat_max that bounds the (landscape) map frame defined in render.py
# (centred on New England, reaching west over New York and east over the Gulf of Maine / Maritimes).  render.py asserts the frame fits inside.
DISPLAY_BBOX = (-80.7, -62.2, 39.8, 48.3)
# Extra degrees kept around the display box when sub-setting the national grid
DATA_PAD_DEG = 0.6
NEW_ENGLAND_STATES = {"CT", "ME", "MA", "NH", "RI", "VT"}

DISPLAY_TZ = "America/New_York"

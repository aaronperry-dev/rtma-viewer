"""Central configuration: products, variables, region, cities."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from . import kinematics as kin
from .derived import (FLIGHT_CATEGORIES, apparent_temperature, dewpoint_depression, flight_category, frostbite_category,
                      hours_at_or_below, min_wind_chill, mixing_ratio, mslp, relative_humidity_field, relative_humidity_ice,
                      sea_level_temperature, series_max, series_min, series_range, threshold_crossings, wet_bulb,
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
G_TEMP, G_24H, G_WIND, G_PRESS, G_SKY, G_KIN, G_TERRAIN = ("Temperature & moisture", "Past 24 hours", "Wind", "Pressure",
                                                            "Sky & visibility", "Surface kinematics", "Terrain")
FROSTBITE_CATEGORIES = ("None", "30 min", "10 min", "5 min")


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
    series: int = 0                    # derived from the last N hours: ``derive`` gets (N+1, ny, nx) stacks, index 0 = valid time
    needs_geom: bool = False           # ``derive`` also takes ``geom=`` (kinematics.Geometry of the grid)
    diverging: bool = False            # a current field that is signed (advection, convergence ...): "+" on the labels
    categories: tuple[str, ...] = ()   # integer-coded classes 0..n-1 (flight category ...): shown as area shares, not min / max
    stat_decimals: int = 1             # digits of the low / mean / high statistics on the page
    lines: tuple[tuple[float, str], ...] = ()   # (value, colour) reference contours drawn over the current-field map


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
    "slt": Variable("slt", "Sea-level temp", "2m Temperature Adjusted to Sea Level", "°F", "", "",
                    requires=("temp", "terrain"), derive=sea_level_temperature, changes=False,
                    note="2 m temperature plus 3.6 °F per 1,000 ft of analysis terrain height (a 6.5 K/km standard lapse rate), "
                         "drawn with the 2 m temperature colours. It takes the elevation out of the temperature map, so air-mass "
                         "boundaries, cold pools and valley inversions stand out: a uniform air mass looks flat, a cold pool "
                         "reads colder than the slopes around it. This is also the field the advection and frontogenesis maps use."),
    "rhi": Variable("rhi", "RH w.r.t. ice", "2m Relative Humidity with respect to Ice", "%", "", "",
                    requires=("temp", "dewpoint"), derive=relative_humidity_ice, changes=False,
                    note="Relative humidity over ice below freezing (over liquid water above it). The published dew point is "
                         "with respect to liquid water, so air that is exactly saturated over water reads about 110 % at 15 °F and "
                         "140 % at -30 °F. Values above 100 % mean the air is supersaturated with respect to ice: frost, rime "
                         "and ice-crystal growth are favoured, and they are normal in cold, humid air rather than an error.", stat_decimals=0),
    "mixr": Variable("mixr", "Mixing ratio", "2m Water-Vapor Mixing Ratio", "g/kg", "", "",
                     requires=("dewpoint", "pressure"), derive=mixing_ratio, changes=False, decimals=1,
                     note="Grams of water vapour per kilogram of dry air, from the dew point and the real surface pressure. "
                          "Unlike relative humidity it does not change when the air temperature does, so it follows the moisture "
                          "of the air mass itself: under about 1 g/kg is bitter, dry Arctic air; near 4 g/kg is moist air at the "
                          "freezing point; air that is saturated at 10 °F holds only about 1.5 g/kg.", stat_decimals=2),
    # ------------------------------------------------------------------ past 24 hours (from the cached hourly analyses)
    "tmax24": Variable("tmax24", "24 h max temp", "24-Hour Maximum 2m Temperature", "°F", "", "", requires=("temp",),
                       derive=series_max, series=24, group=G_24H, changes=False,
                       note="Highest 2 m temperature among the hourly analyses in the 24 hours ending at the valid time (not a "
                            "calendar-day maximum, and a short peak between two hourly analyses is missed)."),
    "tmin24": Variable("tmin24", "24 h min temp", "24-Hour Minimum 2m Temperature", "°F", "", "", requires=("temp",),
                       derive=series_min, series=24, group=G_24H, changes=False,
                       note="Lowest 2 m temperature among the hourly analyses in the 24 hours ending at the valid time. Compare "
                            "with the forecast low; radiational cooling shows up as the coldest valleys."),
    "trange24": Variable("trange24", "24 h temp range", "24-Hour 2m Temperature Range", "°F", "", "", requires=("temp",),
                         derive=series_range, series=24, group=G_24H, changes=False,
                         note="24-hour maximum minus minimum. Large ranges (30 °F or more) mean clear, dry, calm air and strong "
                              "radiational cooling; small ranges mean cloud, wind, a marine influence or an air mass that did not "
                              "change."),
    "frzhrs24": Variable("frzhrs24", "Hours ≤ 32 °F", "Hours at or Below 32°F, Past 24 Hours", "h", "", "",
                         requires=("temp",), derive=hours_at_or_below, series=24, group=G_24H, changes=False,
                         note="Hours the 2 m temperature spent at or below freezing in the last 24, interpolating between hourly "
                              "analyses. 0 = no freezing; 24 = frozen the whole day (persistent cold, slow melt, ground freezing)."),
    "ftx24": Variable("ftx24", "Freeze-thaw crossings", "Freezing-Point Crossings, Past 24 Hours", "count", "", "",
                      requires=("temp",), derive=threshold_crossings, series=24, group=G_24H, changes=False,
                      note="How many times the 2 m temperature crossed 32 °F, in either direction, in the last 24 hours. Two "
                           "or more is a freeze-thaw cycle: refreezing of wet roads and surfaces, flash freezes, crust and "
                           "ice formation.", stat_decimals=0),
    "wcmin24": Variable("wcmin24", "24 h min wind chill", "24-Hour Minimum Wind Chill", "°F", "", "", requires=("temp", "wind"),
                        derive=min_wind_chill, series=24, group=G_24H, changes=False,
                        note="Lowest hourly NWS wind chill in the last 24 hours (air temperature where wind chill is not "
                             "defined: above 50 °F or wind of 3 mph or less). Same colour legend as the NWS 24-hour minimum wind "
                             "chill map."),
    "gustmax24": Variable("gustmax24", "24 h max gust", "24-Hour Maximum Wind Gust", "mph", "", "", requires=("gust",),
                          derive=series_max, series=24, group=G_24H, changes=False,
                          note="Strongest 10 m gust among the hourly analyses in the last 24 hours (NWS legend). The analysis "
                               "gust is a diagnosed value, so treat peaks between analyses and at exposed summits with caution."),
    # ------------------------------------------------------------------ wind
    "wind": Variable("wind", "Wind speed", "10m Wind Speed", "mph", "WIND", "10 m above ground", _ms_to_mph,
                     components=("UGRD", "VGRD"), group=G_WIND),
    "gust": Variable("gust", "Wind gust", "10m Wind Gust", "mph", "GUST", "10 m above ground", _ms_to_mph, group=G_WIND),
    "wdir_raw": Variable("wdir_raw", "Wind direction (raw)", "10m Wind Direction (raw)", "°", "WDIR", "10 m above ground",
                         group=G_WIND, changes=False, hidden=True),
    "wdir": Variable("wdir", "Wind direction", "10m Wind Direction", "°", "", "", requires=("wdir_raw", "wind"),
                     derive=wind_direction, group=G_WIND, changes=False, circular=True,
                     note="Direction the wind blows from (meteorological convention: 90° = east wind). Calm air (under 2 mph) has "
                          "no direction and is left white. The average is a vector mean.", stat_decimals=0),
    "frostbite": Variable("frostbite", "Frostbite time", "Frostbite Time on Exposed Skin (from Wind Chill)", "min", "", "",
                          requires=("temp", "wind"), derive=frostbite_category, group=G_WIND, changes=False,
                          categories=FROSTBITE_CATEGORIES, labeler=lambda v: ("", "30", "10", "5")[int(round(v))],
                          note="Approximate time for frostbite to develop on exposed skin, from the NWS wind chill chart: about 30 "
                               "minutes at a wind chill of -18 °F or colder, 10 minutes at -33 °F, 5 minutes at -48 °F. The chart's "
                               "zone edges move a few degrees with wind speed, so these are round mid-chart values. Uses the NWS "
                               "wind chill formula with 2 m temperature and 10 m wind; it is blank (none) at 50 °F and above or in "
                               "light wind."),
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
    "pgrad": Variable("pgrad", "Pressure gradient", "MSLP Gradient Magnitude", "mb/100 km", "", "", requires=("mslp", "terrain"),
                      needs_geom=True, derive=kin.pressure_gradient, group=G_PRESS, changes=False, decimals=1,
                      note="Strength of the broad-scale sea-level pressure gradient, from MSLP smoothed over about 60 km (the 2.5 km "
                           "reduction is too noisy to differentiate raw, so narrow features such as a coastal front are smoothed away). At this latitude "
                           "1 mb per 100 km corresponds to roughly a 17 mph geostrophic wind (surface winds are weaker because of "
                           "friction); 2 or more is a tight gradient and a windy setup. Left blank above 1,000 m, where the "
                           "sea-level reduction is not meaningful.", stat_decimals=2),
    # ------------------------------------------------------------------ sky & visibility
    "vis": Variable("vis", "Visibility", "Surface Visibility", "mi", "VIS", "surface", _m_to_mi, group=G_SKY,
                    changes=False, decimals=1,
                    note="Statute miles. The analysis tops out near 10 mi (16 km), so 9.5+ means unrestricted. Low values mark fog, "
                         "snow squalls and blowing snow."),
    "ceiling": Variable("ceiling", "Cloud ceiling", "Cloud Ceiling (AGL)", "ft", "CEIL", "cloud ceiling", _ceil_m_to_ft,
                        group=G_SKY, changes=False, labeler=_ceil_label,
                        note="Height of the lowest ceiling above ground, in feet; 12,000 ft or higher (or no ceiling) is shown as CLR. "
                             "See Flight category for the combined visibility and ceiling classes.", stat_decimals=0),
    "fltcat": Variable("fltcat", "Flight category", "Flight Category (Visibility and Ceiling)", "", "", "",
                       requires=("vis", "ceiling"), derive=flight_category, group=G_SKY, changes=False,
                       categories=FLIGHT_CATEGORIES, labeler=lambda v: FLIGHT_CATEGORIES[int(round(v))],
                       note="The worse of the analysed visibility and ceiling: VFR (ceiling above 3,000 ft and visibility over 5 mi), "
                            "MVFR (1,000-3,000 ft and/or 3-5 mi), IFR (500-999 ft and/or 1 to under 3 mi), LIFR (under 500 ft and/or "
                            "under 1 mi). Handy for fog, low stratus and snow showers, but the analysed visibility and ceiling are "
                            "coarse, so treat it as guidance rather than a METAR."),
    "cloud": Variable("cloud", "Cloud cover", "Total Cloud Cover", "%", "TCDC",
                      "entire atmosphere (considered as a single layer)", group=G_SKY, changes=False,
                      note="Total sky cover. Clear skies with light wind (especially over snow cover) are the setup for the coldest "
                           "nights and frost hollows; overcast keeps nights milder."),
    # ------------------------------------------------------------------ surface kinematics (10 m wind components, grid-relative)
    "u10": Variable("u10", "10 m wind u (grid)", "10m Wind U Component (grid-relative)", "m/s", "UGRD", "10 m above ground",
                    group=G_KIN, changes=False, hidden=True, pack=(-65.0, 500.0)),
    "v10": Variable("v10", "10 m wind v (grid)", "10m Wind V Component (grid-relative)", "m/s", "VGRD", "10 m above ground",
                    group=G_KIN, changes=False, hidden=True, pack=(-65.0, 500.0)),
    "tadv": Variable("tadv", "Temp. advection", "Surface Temperature Advection (sea-level adjusted)", "°F/h", "", "",
                     requires=("slt", "u10", "v10"), needs_geom=True, derive=kin.advection, group=G_KIN, changes=False,
                     diverging=True,
                     note="Temperature advection by the 10 m wind, -V·∇T, in °F per hour: warm advection positive (orange), cold "
                          "advection negative (blue). It uses the sea-level-adjusted temperature, so wind blowing up or down a slope "
                          "is not mistaken for warm or cold advection. Fields are smoothed over about 8 km before differencing. "
                          "Surface advection only: it does not see the low-level jet or anything above the surface layer."),
    "dadv": Variable("dadv", "Dew point advection", "Surface Dew Point Advection (sea-level adjusted)", "°F/h", "", "",
                     requires=("dewpoint", "terrain", "u10", "v10"), needs_geom=True, derive=kin.dewpoint_advection,
                     group=G_KIN, changes=False, diverging=True,
                     note="Advection of the dew point (adjusted to sea level with a 1.8 K/km lapse rate), °F per hour: moister air "
                          "arriving is positive, drier (often colder, Arctic) air arriving is negative. Same smoothing and caveats as "
                          "temperature advection."),
    "conv": Variable("conv", "Convergence", "10m Wind Convergence", "10⁻⁵ s⁻¹", "", "", requires=("u10", "v10"),
                     needs_geom=True, derive=kin.convergence, group=G_KIN, changes=False, diverging=True,
                     note="Convergence of the 10 m wind: positive (warm colours) is air piling up, as along coastal fronts, "
                          "sea-breeze and cold-front boundaries, where lift and bands of precipitation tend to focus; negative is "
                          "divergence. Derivatives of a 2.5 km wind analysis are noisy, so the wind is smoothed over about 8 km "
                          "and the pattern matters more than any single value. Coastlines and lake shores light up because land and water "
                          "really do differ, and terrain adds its own convergence lines."),
    "vort": Variable("vort", "Vorticity", "10m Wind Relative Vorticity", "10⁻⁵ s⁻¹", "", "", requires=("u10", "v10"),
                     needs_geom=True, derive=kin.vorticity, group=G_KIN, changes=False, diverging=True,
                     note="Spin in the 10 m wind: positive is cyclonic (counter-clockwise), found in surface troughs, along fronts "
                          "and in and around lows; negative is anticyclonic. Smoothed over about 8 km like the other kinematic "
                          "maps. Coastlines and lake shores stand out for the same reason as in the convergence map."),
    "fgen": Variable("fgen", "Frontogenesis", "Surface Frontogenesis (sea-level adjusted temperature)", "°F/100 km/3 h", "", "",
                     requires=("slt", "u10", "v10"), needs_geom=True, derive=kin.frontogenesis, group=G_KIN, changes=False,
                     diverging=True,
                     note="Petterssen kinematic frontogenesis of the sea-level-adjusted temperature: positive where the 10 m wind "
                          "(convergence and deformation) is sharpening a temperature contrast, as at a coastal or arctic front or "
                          "the edge of damming; negative where it is weakening one. Horizontal kinematic terms only, smoothed "
                          "over about 10 km; treat it as a pointer to where fronts are forming rather than a precise value."),
    "upslope": Variable("upslope", "Upslope lift", "Terrain-Forced Vertical Motion (V·∇z)", "cm/s", "", "",
                        requires=("u10", "v10", "terrain"), needs_geom=True, derive=kin.terrain_lift, group=G_KIN,
                        changes=False, diverging=True,
                        note="Vertical motion forced by the terrain alone, the 10 m wind dotted with the terrain slope: positive "
                             "where the wind blows uphill (lift, and in moist flow orographic cloud and snow), negative downslope "
                             "(sinking, drying). The terrain is the smoothed 2.5 km analysis grid, and the diagnostic ignores "
                             "stability, so blocked low-level flow will not actually climb the terrain."),
    # ------------------------------------------------------------------ terrain (static)
    "terrain": Variable("terrain", "Terrain height", "Analysis Terrain Height", "ft", "HGT", "surface", _m_to_ft,
                        group=G_TERRAIN, changes=False, static=True,
                        note="The 2.5 km analysis grid's smoothed terrain, which is what the analysis (and MSLP reduction) "
                             "uses. Real summits are higher and valleys lower. Useful for reading elevation effects, upslope "
                             "flow and cold-air pooling.", stat_decimals=0),
}
# Reference lines on every absolute temperature-type map (°F): the freezing line in red and the 0 °F line in blue.  Drawn only
# where the field actually crosses the value, never on change maps.
TEMP_LINES = ((32.0, "#e00000"), (0.0, "#0a3cff"))
_TEMP_LINE_KEYS = ("temp", "dewpoint", "wetbulb", "apparent", "slt", "tmax24", "tmin24", "wcmin24")
VARIABLES = {k: replace(v, lines=TEMP_LINES) if k in _TEMP_LINE_KEYS else v for k, v in VARIABLES.items()}
SOURCE_VARIABLES: dict[str, Variable] = {k: v for k, v in VARIABLES.items() if v.derive is None}
PANEL_VARIABLES: dict[str, Variable] = {k: v for k, v in VARIABLES.items() if not v.hidden}     # what gets maps / a page entry

# Current field + change versus N hours earlier
INTERVALS: tuple[int, ...] = (0, 1, 3, 6, 12, 24)


def panel_hours(var: Variable) -> tuple[int, ...]:
    """Intervals a variable gets a map for: all of them, or just the current field."""
    return INTERVALS if var.changes else (0,)


def hours_needed() -> dict[str, tuple[int, ...]]:
    """For every *source* variable, the hours-before-newest-cycle it must be loaded for.

    Walks the derivation graph: a derived variable drawn at hours H needs its inputs at every h + i (i = 0..series), where
    ``series`` is how many past hours it summarises (0 for ordinary variables), and so on down to the GRIB messages."""
    need: dict[str, set[int]] = {}

    def add(key: str, hours: set[int]):
        var = VARIABLES[key]
        if var.derive is None:
            need.setdefault(key, set()).update({0} if var.static else hours)
            return
        spread = {h + i for h in hours for i in range(var.series + 1)}
        for dep in var.requires:
            add(dep, spread)

    for var in VARIABLES.values():
        add(var.key, set(panel_hours(var)))
    return {k: tuple(sorted(v)) for k, v in need.items()}


# --------------------------------------------------------------------------- region
# lon_min, lon_max, lat_min, lat_max that bounds the (landscape) map frame defined in render.py
# (centred on New England, reaching west over New York and east over the Gulf of Maine / Maritimes).  render.py asserts the frame fits inside.
DISPLAY_BBOX = (-80.7, -62.2, 39.8, 48.3)
# Extra degrees kept around the display box when sub-setting the national grid
DATA_PAD_DEG = 0.6
NEW_ENGLAND_STATES = {"CT", "ME", "MA", "NH", "RI", "VT"}

DISPLAY_TZ = "America/New_York"

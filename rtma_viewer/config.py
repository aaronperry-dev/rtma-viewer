"""Central configuration: products, variables, region, cities."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from . import kinematics as kin
from .derived import (FLIGHT_CATEGORIES, apparent_temperature, blizzard_hours, cold_pool, degree_hours, dewpoint_depression,
                      flash_freeze, flight_category, freezing_fog, frostbite_category, frozen_precipitation, hours_at_or_above,
                      hours_at_or_below, min_wind_chill, mixing_ratio, mslp, precip_on_cold_surface, radiational_cooling,
                      relative_humidity_field, relative_humidity_ice, sea_level_temperature, series_max, series_min, series_range,
                      series_sum, snowmaking_hours, threshold_crossings, wet_bulb, wet_snow_load, wind_chill_hours, wind_direction)


# --------------------------------------------------------------------------- products
@dataclass(frozen=True)
class Product:
    key: str
    label: str
    long_name: str
    map_title: str       # the full name printed at the top of every map
    bucket: str          # public NOAA Open Data bucket (anonymous access)
    prefix: str          # e.g. "rtma2p5" -> rtma2p5.YYYYMMDD/rtma2p5.tHHz.2dvaranl_ndfd.grb2*
    lookback_hours: int  # how far back to search for the newest published cycle
    blurb: str
    pcp_suffix: str = ""        # hourly precipitation analysis file: {prefix}.YYYYMMDD/{prefix}.YYYYMMDDHH.{pcp_suffix}
    pcp_refresh_hours: int = 0  # cached hourly precipitation younger than this is re-downloaded each run (later re-makes)


PRODUCTS: dict[str, Product] = {
    "rtma": Product(
        "rtma", "RTMA", "Real-Time Mesoscale Analysis", "NOAA Real-Time Mesoscale Analysis 2.5km (RTMA)", "noaa-rtma-pds", "rtma2p5", 6,
        "NOAA/NCEP 2.5 km hourly surface analysis. Available ~35-45 min after the valid hour.",
        pcp_suffix="pcp.184.grb2", pcp_refresh_hours=3,
    ),
    "urma": Product(
        "urma", "URMA", "Unrestricted Mesoscale Analysis", "NOAA Unrestricted Mesoscale Analysis 2.5km (URMA)", "noaa-urma-pds", "urma2p5", 12,
        "Same 2.5 km grid as RTMA, but run later with more observations. Lags RTMA by a few hours.",
        pcp_suffix="pcp_01h.wexp.grb2", pcp_refresh_hours=24,
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


def _mm_to_in(a):
    return np.maximum(a, 0.0) / 25.4          # kg m-2 of water = mm; tiny negative analysis values are noise


def _pcp_label(v: float) -> str:
    return "" if v < 0.005 else f"{v:.2f}"    # dry stations stay unlabelled: a map full of 0.00 says nothing


def _m_to_ft(a):
    return a * FT_PER_M


def _m_to_mi(a):
    return a / 1609.344


def _ceil_m_to_ft(a):
    return np.minimum(a * FT_PER_M, CEIL_CAP_FT)


def _ceil_label(v: float) -> str:
    return "CLR" if v >= CEIL_CAP_FT - 1 else str(int(round(v / 100.0)) * 100)


# --------------------------------------------------------------------------- variables
G_TEMP, G_24H, G_HAZ, G_PCP, G_WIND, G_PRESS, G_FRONT, G_SKY, G_KIN, G_TERRAIN = (
    "Temperature & moisture", "Past 24 hours", "Winter hazard diagnostics", "Precipitation", "Wind", "Pressure",
    "Surface analysis", "Sky & visibility", "Surface kinematics", "Terrain")
FROSTBITE_CATEGORIES = ("None", "30 min", "10 min", "5 min")
BLIZZARD_LABELS = ("0 h", "1 h", "2 h", "3 h", "4 h", "5 h", "6 h")
FLASH_FREEZE_CATEGORIES = ("None", "Fell below 32 °F", "Flash freeze")
FREEZING_FOG_CATEGORIES = ("None", "Favorable", "Dense")
WET_SNOW_CATEGORIES = ("< 0.10 in", "≥ 0.10 in", "≥ 0.25 in", "+ gusts")
RAD_COOL_CATEGORIES = ("None", "Clear or calm", "Clear and calm", "Ideal")


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
    lines: tuple[tuple[float, str], ...] = ()   # (value, color) reference contours drawn over the current-field map
    source: str = "idx"                # "idx": a message of the 2dvaranl file (byte range); "pcp": the hourly precipitation file
    partial_series: bool = False       # use whatever hourly inputs exist: each input's missing hours are skipped on their own, and the
                                       # build records how many hours of ``period_hours`` were really available
    period_hours: int = 0              # length of the accumulation period (precipitation totals), for the "N of M h available" note
    show_period: bool = True           # header says "(past N h)" for series products (off when the title already names the period)
    overlay: str = ""                  # "wpc" or "objective": draw fronts, troughs and H / L symbols over the map (see fronts.py / objective.py)


_PCP_NOTE = ("Liquid-equivalent precipitation analysed on the 2.5 km grid. RTMA uses the radar-only MRMS estimate (about 15 min after "
             "the hour, no gauge correction - radar is least reliable in snow, in the mountains and at long range); URMA uses the "
             "quality-controlled, gauge-corrected Stage IV analysis and is re-made as River Forecast Centre data arrive. It is water, "
             "not snow depth, and it is an analysis of what fell, not a forecast. If some of the hourly analyses are missing the total "
             "adds up the ones that exist, says how many that was, and may therefore read low. ")

_IND = ("An indicator of where surface conditions favour this, built from the analysed 2 m and 10 m fields: it cannot see the air "
        "aloft, so use it with radar, soundings and the observations, not instead of them. ")

_PCP_AVAIL = ("If some hourly precipitation analyses are missing it uses the ones that exist and says how many on the map. ")

_FROZEN_NOTE = ("Liquid-equivalent frozen precipitation (snow, sleet and freezing rain together): the precipitation that fell while "
                "the surface wet bulb was at or below 32 °F (average of the wet bulb at the start and end of each hour). It is water, not "
                "snow depth - no snow-to-liquid ratio is applied. It cannot tell the frozen types apart, ignores the air aloft (rain can "
                "fall through a cold surface layer; snow can melt before it lands) and treats an hour as all-or-nothing, so it is a rough "
                "guide to where precipitation fell cold, not a forecast of accumulation. Same bins as the precipitation maps, in cool blue-purple. " + _PCP_NOTE)

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
                         "drawn with the 2 m temperature colors. It takes the elevation out of the temperature map, so air-mass "
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
                             "defined: above 50 °F or wind of 3 mph or less). Same color legend as the NWS 24-hour minimum wind "
                             "chill map."),
    "gustmax24": Variable("gustmax24", "24 h max gust", "24-Hour Maximum Wind Gust", "mph", "", "", requires=("gust",),
                          derive=series_max, series=24, group=G_24H, changes=False,
                          note="Strongest 10 m gust among the hourly analyses in the last 24 hours (NWS legend). The analysis "
                               "gust is a diagnosed value, so treat peaks between analyses and at exposed summits with caution."),
    # ---- counts and totals over the last 24-48 hours (from the cached hourly analyses)
    "snowmk28": Variable("snowmk28", "Snowmaking hours", "Snowmaking Hours (Wet Bulb ≤ 28°F), Past 24 Hours", "h", "", "",
                         requires=("temp", "dewpoint", "pressure"), derive=snowmaking_hours, series=24, group=G_24H, changes=False,
                         show_period=False,
                         note="Hours in the last 24 with the 2 m wet-bulb temperature at or below 28 °F, interpolating between the "
                              "hourly analyses: the usual limit for ski-area snowmaking (colder and drier air makes snow more "
                              "efficiently; near 28 °F it is marginal). The wet bulb accounts for humidity, which is why a dry 30 °F "
                              "night can still be good. 24 = the whole day."),
    "wchrs15": Variable("wchrs15", "Hours WC ≤ -15 °F", "Hours with Wind Chill ≤ -15°F, Past 24 Hours", "h", "", "",
                        requires=("temp", "wind"), derive=lambda t, w: wind_chill_hours(t, w, -15.0), series=24, group=G_24H,
                        changes=False, show_period=False,
                        note="Hours in the last 24 with the NWS wind chill at or below -15 °F (the air temperature where wind chill "
                             "is not defined). Offices use different wind chill advisory and warning thresholds, so treat -15 °F "
                             "and -25 °F as reference levels, not your office's criteria."),
    "wchrs25": Variable("wchrs25", "Hours WC ≤ -25 °F", "Hours with Wind Chill ≤ -25°F, Past 24 Hours", "h", "", "",
                        requires=("temp", "wind"), derive=lambda t, w: wind_chill_hours(t, w, -25.0), series=24, group=G_24H,
                        changes=False, show_period=False,
                        note="Hours in the last 24 with the NWS wind chill at or below -25 °F. See the -15 °F map for the caveat "
                             "about office-specific criteria."),
    "gst40hrs": Variable("gst40hrs", "Hours gust ≥ 40 mph", "Hours with Wind Gust ≥ 40 mph, Past 24 Hours", "h", "", "",
                         requires=("gust",), derive=lambda g: hours_at_or_above(g, 40.0), series=24, group=G_24H, changes=False,
                         show_period=False,
                         note="Hours in the last 24 with the analysed 10 m gust at or above 40 mph (wind advisory territory in "
                              "much of the region; also a threshold for scattered tree and power-line damage). The analysis gust "
                              "is diagnosed, so exposed summits and coasts are the least reliable."),
    "gst58hrs": Variable("gst58hrs", "Hours gust ≥ 58 mph", "Hours with Wind Gust ≥ 58 mph, Past 24 Hours", "h", "", "",
                         requires=("gust",), derive=lambda g: hours_at_or_above(g, 58.0), series=24, group=G_24H, changes=False,
                         show_period=False,
                         note="Hours in the last 24 with the analysed 10 m gust at or above 58 mph, the usual high wind warning "
                              "gust level. Rare in the analysis; see the 40 mph map for the caveats."),
    "pdh24": Variable("pdh24", "Melt degree-hours 24 h", "Positive Degree-Hours Above 32°F, Past 24 Hours", "°F·h", "", "",
                      requires=("temp",), derive=degree_hours, series=24, group=G_24H, changes=False, show_period=False,
                      note="Sum over the last 24 hours of how many °F the 2 m temperature was above 32 °F (a 40 °F day is about "
                           "190 °F·h). A snowmelt index: the more degree-hours, the more melt, and so the more runoff and the faster "
                           "snow-cover loss. It ignores sun, rain and wind, which add melt, and does not know whether snow is on "
                           "the ground.", stat_decimals=0),
    "pdh48": Variable("pdh48", "Melt degree-hours 48 h", "Positive Degree-Hours Above 32°F, Past 48 Hours", "°F·h", "", "",
                      requires=("temp",), derive=degree_hours, series=48, group=G_24H, changes=False, show_period=False,
                      note="Same as the 24-hour map over 48 hours: a snowmelt index for runoff and ice-jam awareness.",
                      stat_decimals=0),
    "fdh48": Variable("fdh48", "Freezing degree-hours 48 h", "Freezing Degree-Hours Below 32°F, Past 48 Hours", "°F·h", "", "",
                      requires=("temp",), derive=lambda s: degree_hours(s, above=False), series=48, group=G_24H, changes=False,
                      show_period=False,
                      note="Sum over the last 48 hours of how many °F the 2 m temperature was below 32 °F (a day at 20 °F is "
                           "about 290 °F·h). Drives river and lake ice growth and frost depth: large values mean persistent cold. "
                           "Ice thickness also depends on snow cover and water movement, which this does not know about.",
                      stat_decimals=0),
    # ------------------------------------------------------------------ winter hazard diagnostics
    "blizz": Variable("blizz", "Blizzard criteria", "Blizzard Criteria: Consecutive Hours Met, Past 6 Hours", "h", "", "",
                      requires=("vis", "wind", "gust"), derive=blizzard_hours, series=5, group=G_HAZ, changes=False,
                      show_period=False, categories=BLIZZARD_LABELS, labeler=lambda v: ("", "1", "2", "3", "4", "5", "6")[int(round(v))],
                      note="Longest run of consecutive hourly analyses, in the last 6 hours, with visibility of 1/4 mile or less and "
                           "sustained wind or gusts of 35 mph or more. The NWS blizzard definition needs this for 3 hours or more, so "
                           "the red classes (3-6) meet it. Analysed visibility is coarse and often misses blowing snow in the open, "
                           "so a miss here does not rule a blizzard out; check observations."),
    "flashfrz": Variable("flashfrz", "Flash freeze", "Flash-Freeze Potential, Past 3 Hours", "", "", "",
                         requires=("pcp1", "temp"), derive=flash_freeze, series=3, period_hours=3, partial_series=True,
                         group=G_HAZ, changes=False, show_period=False, categories=FLASH_FREEZE_CATEGORIES,
                         labeler=lambda v: ("", "F", "FF")[int(round(v))],
                         note="Where the 2 m temperature fell from above to at or below 32 °F in the last 3 hours with liquid "
                              "precipitation in that time (F), and where it fell 10 °F or more (FF): wet roads and surfaces "
                              "freezing. " + _IND + _PCP_AVAIL),
    "frzfog": Variable("frzfog", "Freezing fog / drizzle", "Freezing Fog or Drizzle Favorable", "", "", "",
                       requires=("temp", "dewpoint", "vis", "ceiling"), derive=freezing_fog, group=G_HAZ, changes=False,
                       categories=FREEZING_FOG_CATEGORIES, labeler=lambda v: ("", "ZF", "ZF+")[int(round(v))],
                       note="2 m temperature at or below 32 °F, dew point depression of 2 °F or less, and visibility under 3 mi or a "
                            "ceiling under 1,000 ft (Favorable); visibility under 1/2 mi is flagged Dense. Saturated, sub-freezing "
                            "air with reduced visibility also describes falling snow, so read it next to radar. " + _IND),
    "wetsnow": Variable("wetsnow", "Wet-snow load", "Wet-Snow Load Risk, Past 6 Hours", "", "", "",
                        requires=("pcp1", "temp", "dewpoint", "pressure", "gust"), derive=wet_snow_load, series=6, period_hours=6,
                        partial_series=True, group=G_HAZ, changes=False, show_period=False, categories=WET_SNOW_CATEGORIES,
                        labeler=lambda v: ("", "1", "2", "3")[int(round(v))],
                        note="Heavy, sticky snow that loads trees and power lines: liquid precipitation in hours with a wet bulb at "
                             "or below 32 °F while the air was 31 °F or warmer, over 6 hours. Classes: under 0.10 in (none), at least "
                             "0.10 in, at least 0.25 in, and at least 0.25 in with gusts of 35 mph or more. The thresholds are "
                             "rules of thumb, not damage criteria. " + _IND + _PCP_AVAIL),
    "radcool": Variable("radcool", "Radiational cooling", "Radiational Cooling Setup", "", "", "",
                        requires=("cloud", "wind"), derive=radiational_cooling, group=G_HAZ, changes=False,
                        categories=RAD_COOL_CATEGORIES, labeler=lambda v: ("", "1", "2", "3")[int(round(v))],
                        note="Where the sky is clear and the wind light, the setup for the coldest nights, valley cold pools and "
                             "frost hollows: one of clear (cloud 25 % or less) or calm (5 mph or less); both; and both with very "
                             "clear and nearly calm air (10 % or less, 2 mph or less). Meaningful at night: in daytime clear, calm "
                             "air heats up. Snow cover on the ground makes it more effective. " + _IND),
    "coldpool": Variable("coldpool", "Cold pool / inversion", "Cold Pool (-) and Inversion (+): Temperature Minus Local Mean", "°F",
                         "", "", requires=("slt", "terrain"), derive=cold_pool, group=G_HAZ, changes=False, diverging=True,
                         decimals=0,
                         note="Sea-level-adjusted temperature minus its average over the land within about 25 km. Negative (blue) "
                              "means colder than the surroundings after the elevation effect is removed: valley cold pools, "
                              "radiational cooling, cold-air damming; positive (red) means warmer, as on slopes above an "
                              "inversion. Water is left out. Adjustment uses a standard lapse rate, so in a deep inversion the "
                              "adjusted temperature of high ground reads warm. " + _IND),
    # ------------------------------------------------------------------ precipitation (hourly analysis files, summed)
    "pcp1": Variable("pcp1", "1 h precip", "1-Hour Precipitation Analysis", "in", "", "", _mm_to_in, source="pcp", group=G_PCP,
                     changes=False, pack=(0.0, 2000.0), decimals=2, stat_decimals=2, labeler=_pcp_label, show_period=False,
                     note=_PCP_NOTE + "Water falling in the hour ending at the valid time."),
    "pcp3": Variable("pcp3", "3 h precip", "3-Hour Precipitation Analysis", "in", "", "", requires=("pcp1",),
                      derive=series_sum, series=2, period_hours=3, partial_series=True, group=G_PCP, changes=False,
                      decimals=2, stat_decimals=2, labeler=_pcp_label, show_period=False,
                      note=_PCP_NOTE + "Sum of the 3 hourly analyses ending at the valid time."),
    "pcp6": Variable("pcp6", "6 h precip", "6-Hour Precipitation Analysis", "in", "", "", requires=("pcp1",),
                      derive=series_sum, series=5, period_hours=6, partial_series=True, group=G_PCP, changes=False,
                      decimals=2, stat_decimals=2, labeler=_pcp_label, show_period=False,
                      note=_PCP_NOTE + "Sum of the 6 hourly analyses ending at the valid time."),
    "pcp12": Variable("pcp12", "12 h precip", "12-Hour Precipitation Analysis", "in", "", "", requires=("pcp1",),
                      derive=series_sum, series=11, period_hours=12, partial_series=True, group=G_PCP, changes=False,
                      decimals=2, stat_decimals=2, labeler=_pcp_label, show_period=False,
                      note=_PCP_NOTE + "Sum of the 12 hourly analyses ending at the valid time."),
    "pcp24": Variable("pcp24", "24 h precip", "24-Hour Precipitation Analysis", "in", "", "", requires=("pcp1",),
                      derive=series_sum, series=23, period_hours=24, partial_series=True, group=G_PCP, changes=False,
                      decimals=2, stat_decimals=2, labeler=_pcp_label, show_period=False,
                      note=_PCP_NOTE + "Sum of the 24 hourly analyses ending at the valid time."),
    "pcp48": Variable("pcp48", "48 h precip", "48-Hour Precipitation Analysis", "in", "", "", requires=("pcp1",),
                      derive=series_sum, series=47, period_hours=48, partial_series=True, group=G_PCP, changes=False,
                      decimals=2, stat_decimals=2, labeler=_pcp_label, show_period=False,
                      note=_PCP_NOTE + "Sum of the 48 hourly analyses ending at the valid time."),
    "fpcp1": Variable("fpcp1", "1 h frozen precip", "1-Hour Frozen Precipitation", "in", "", "",
                       requires=("pcp1", "temp", "dewpoint", "pressure"), derive=frozen_precipitation, series=1, period_hours=1,
                       partial_series=True, group=G_PCP, changes=False, decimals=2, stat_decimals=2, labeler=_pcp_label,
                       show_period=False, note=_FROZEN_NOTE),
    "fpcp3": Variable("fpcp3", "3 h frozen precip", "3-Hour Frozen Precipitation", "in", "", "",
                       requires=("pcp1", "temp", "dewpoint", "pressure"), derive=frozen_precipitation, series=3, period_hours=3,
                       partial_series=True, group=G_PCP, changes=False, decimals=2, stat_decimals=2, labeler=_pcp_label,
                       show_period=False, note=_FROZEN_NOTE),
    "fpcp6": Variable("fpcp6", "6 h frozen precip", "6-Hour Frozen Precipitation", "in", "", "",
                       requires=("pcp1", "temp", "dewpoint", "pressure"), derive=frozen_precipitation, series=6, period_hours=6,
                       partial_series=True, group=G_PCP, changes=False, decimals=2, stat_decimals=2, labeler=_pcp_label,
                       show_period=False, note=_FROZEN_NOTE),
    "fpcp12": Variable("fpcp12", "12 h frozen precip", "12-Hour Frozen Precipitation", "in", "", "",
                       requires=("pcp1", "temp", "dewpoint", "pressure"), derive=frozen_precipitation, series=12, period_hours=12,
                       partial_series=True, group=G_PCP, changes=False, decimals=2, stat_decimals=2, labeler=_pcp_label,
                       show_period=False, note=_FROZEN_NOTE),
    "fpcp24": Variable("fpcp24", "24 h frozen precip", "24-Hour Frozen Precipitation", "in", "", "",
                       requires=("pcp1", "temp", "dewpoint", "pressure"), derive=frozen_precipitation, series=24, period_hours=24,
                       partial_series=True, group=G_PCP, changes=False, decimals=2, stat_decimals=2, labeler=_pcp_label,
                       show_period=False, note=_FROZEN_NOTE),
    "fpcp48": Variable("fpcp48", "48 h frozen precip", "48-Hour Frozen Precipitation", "in", "", "",
                       requires=("pcp1", "temp", "dewpoint", "pressure"), derive=frozen_precipitation, series=48, period_hours=48,
                       partial_series=True, group=G_PCP, changes=False, decimals=2, stat_decimals=2, labeler=_pcp_label,
                       show_period=False, note=_FROZEN_NOTE),
    "pcpcold24": Variable("pcpcold24", "24 h precip at T ≤ 32 °F", "24-Hour Precipitation at 2m Temperature ≤ 32°F", "in", "", "",
                          requires=("pcp1", "temp"), derive=precip_on_cold_surface, series=24, period_hours=24,
                          partial_series=True, group=G_PCP, changes=False, decimals=2, stat_decimals=2, labeler=_pcp_label,
                          show_period=False,
                          note="Liquid-equivalent precipitation that fell in hours when the 2 m air temperature (mean of the "
                               "hour's two ends) was at or below 32 °F: water landing on a frozen surface. Where the air aloft was "
                               "warm this is the footprint of freezing rain and freezing drizzle, but it counts snow and sleet too, "
                               "and it cannot tell them apart. Compare with the frozen-precipitation maps, which use the wet bulb: "
                               "where this is larger the air was sub-freezing but the wet bulb was not, meaning drier air. " + _PCP_AVAIL),
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
    # ------------------------------------------------------------------ surface analysis (fronts, highs and lows)
    "sfcwpc": Variable("sfcwpc", "Fronts (WPC)", "WPC Surface Analysis over MSLP", "mb", "", "", requires=("mslp",),
                       derive=lambda m: m, group=G_FRONT, changes=False, isobar_step=2.0, label_max_terrain_m=1000.0,
                       labeler=lambda v: "", overlay="wpc",
                       note="The National Weather Service Weather Prediction Center's hand analysis of fronts, troughs and pressure "
                            "centres (its coded surface bulletin, issued every 3 hours), drawn over RTMA / URMA sea-level pressure. "
                            "The positions are WPC's, only joined smoothly. The bulletin gives each front as a line without a direction "
                            "of motion, so the triangles and semicircles are put on the side the front is moving toward, found from "
                            "the analysed equivalent potential temperature either side of the line. Troughs (dashed) also cover "
                            "outflow boundaries, squall lines and dry lines. Positions are given to 0.1 degree and the analysis is "
                            "valid only at the 3-hourly synoptic times, and WPC publishes it about 90 minutes late. The map uses the "
                            "analysis nearest the map time, or failing that the latest one up to 3 hours earlier (the label says which). "
                            "Only the newest bulletin is kept online, so the page saves each one when it first sees it and earlier times "
                            "fill in only going forward."),
    "sfcobj": Variable("sfcobj", "Fronts (objective)", "Experimental Objective Fronts and Pressure Centers over MSLP", "mb", "", "",
                       requires=("mslp", "temp", "dewpoint", "pressure", "u10", "v10", "terrain"),
                       derive=lambda m, *rest: m, group=G_FRONT, changes=False, isobar_step=2.0, label_max_terrain_m=1000.0,
                       labeler=lambda v: "", overlay="objective",
                       note="EXPERIMENTAL, machine-drawn from this analysis alone - not equivalent to WPC's analysis and no substitute "
                            "for it. Fronts are placed where the smoothed surface equivalent potential temperature changes fastest "
                            "(a thermal front parameter), kept where that change exceeds 10 K per 100 km over at least 300 km and "
                            "away from high terrain and from coasts and the Great Lakes (where land-water contrast, not an air mass, drives it), and typed by the wind across the line: blowing toward the warm side is a cold "
                            "front, toward the cold side a warm front, and nearly parallel is stationary. Occluded fronts, troughs "
                            "and dry lines are not identified, strong coast or lake contrasts can show as fronts, and a front may "
                            "break into pieces. H and L mark closed highs and lows of the smoothed sea-level pressure."),
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
                     note="Convergence of the 10 m wind: positive (warm colors) is air piling up, as along coastal fronts, "
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

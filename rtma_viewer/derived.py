"""Derived fields (display units: °F, mph, %), computed on the analysis grid.

Apparent temperature follows the NWS National Digital Forecast Database (NDFD) definition:

* T <= 50 °F  -> NWS Wind Chill Temperature (only defined for wind > 3 mph, otherwise ambient T)
* T >  80 °F  -> Heat Index (WPC/Rothfusz algorithm, needs relative humidity)
* otherwise   -> ambient air temperature

Sources
-------
* Wind chill:  https://www.weather.gov/safety/cold-wind-chill-chart
               WC = 35.74 + 0.6215 T - 35.75 V^0.16 + 0.4275 T V^0.16     (T in °F, V = 10 m wind in mph)
* Heat index:  https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml   (Steadman + Rothfusz + adjustments)
* NDFD apparent temperature definition (switch points 50 / 80 °F): NWS NDFD definitions page, and
  https://ams.confex.com/ams/89annual/webprogram/Paper149894.html (Apparent temperature climatology)
"""
from __future__ import annotations

import numpy as np

WC_MAX_T_F = 50.0      # wind chill used at or below this temperature ...
WC_MIN_WIND_MPH = 3.0  # ... and above this wind speed
HI_MIN_T_F = 80.0      # heat index used above this temperature

_HI_COEF = (-42.379, 2.04901523, 10.14333127, -0.22475541, -6.83783e-3,
            -5.481717e-2, 1.22874e-3, 8.5282e-4, -1.99e-6)


def relative_humidity(t_f, td_f):
    """Relative humidity (%) from temperature and dew point (°F), Magnus formula, clipped to 0..100."""
    t = (np.asarray(t_f, float) - 32.0) / 1.8
    td = (np.asarray(td_f, float) - 32.0) / 1.8
    a, b = 17.625, 243.04
    rh = 100.0 * np.exp(a * td / (b + td) - a * t / (b + t))
    return np.clip(rh, 0.0, 100.0)


def wind_chill(t_f, wind_mph):
    """NWS wind chill temperature (°F).  Meaningful only for T <= 50 °F and wind > 3 mph."""
    t = np.asarray(t_f, float)
    v016 = np.maximum(np.asarray(wind_mph, float), 0.0) ** 0.16
    return 35.74 + 0.6215 * t - 35.75 * v016 + 0.4275 * t * v016


def heat_index(t_f, rh):
    """NWS/WPC heat index (°F) from temperature (°F) and relative humidity (%).

    Steadman's simple formula is averaged with the temperature; where that average is >= 80 °F the
    Rothfusz regression is used instead, with the low-humidity (RH < 13 %, 80-112 °F) and high-humidity
    (RH > 85 %, 80-87 °F) adjustments.
    """
    t = np.asarray(t_f, float)
    rh = np.asarray(rh, float)
    simple = 0.5 * (t + 61.0 + 1.2 * (t - 68.0) + 0.094 * rh)
    c = _HI_COEF
    full = (c[0] + c[1] * t + c[2] * rh + c[3] * t * rh + c[4] * t ** 2 + c[5] * rh ** 2
            + c[6] * t ** 2 * rh + c[7] * t * rh ** 2 + c[8] * t ** 2 * rh ** 2)
    with np.errstate(invalid="ignore"):
        low = np.where((rh < 13) & (t >= 80) & (t <= 112),
                       -((13 - rh) / 4) * np.sqrt(np.maximum(17 - np.abs(t - 95), 0) / 17), 0.0)
        high = np.where((rh > 85) & (t >= 80) & (t <= 87), ((rh - 85) / 10) * ((87 - t) / 5), 0.0)
    return np.where((simple + t) / 2 >= 80.0, full + low + high, simple)


def apparent_temperature(t_f, td_f, wind_mph):
    """NDFD-style apparent temperature (°F) from temperature, dew point (°F) and 10 m wind speed (mph)."""
    t = np.asarray(t_f, np.float64)
    td = np.asarray(td_f, np.float64)
    v = np.asarray(wind_mph, np.float64)
    out = t.copy()
    cold = (t <= WC_MAX_T_F) & (v > WC_MIN_WIND_MPH)
    hot = t > HI_MIN_T_F
    out = np.where(cold, wind_chill(t, v), out)
    out = np.where(hot, heat_index(t, relative_humidity(t, td)), out)
    return np.where(np.isfinite(t) & np.isfinite(td) & np.isfinite(v), out, np.nan).astype(np.float32)


# ---------------------------------------------------------------------------------------------------- moisture
def _f_to_c(f):
    return (np.asarray(f, np.float64) - 32.0) / 1.8


def _c_to_f(c):
    return np.asarray(c, np.float64) * 1.8 + 32.0


def sat_vapor_pressure(t_c):
    """Saturation vapour pressure over liquid water (hPa) - Magnus / Alduchov-Eskridge, the same constants as
    ``relative_humidity`` so the two always agree."""
    t = np.asarray(t_c, np.float64)
    return 6.1094 * np.exp(17.625 * t / (243.04 + t))


def dewpoint_depression(t_f, td_f):
    """Temperature minus dew point (°F), never negative.  ~0 means saturated (fog / low cloud / rime)."""
    t = np.asarray(t_f, np.float64)
    td = np.asarray(td_f, np.float64)
    return np.maximum(t - td, 0.0).astype(np.float32)


def relative_humidity_field(t_f, td_f):
    """Relative humidity (%) as a float32 display field."""
    return relative_humidity(t_f, td_f).astype(np.float32)


_EPS = 0.62198              # Rd / Rv
_CPD, _CPV = 1005.7, 1875.0  # J kg-1 K-1: dry air, water vapour


def sat_vapor_pressure_ice(t_c):
    """Saturation vapour pressure over ice (hPa), Alduchov-Eskridge constants (1.032 hPa at -20 C, 0.128 hPa at -40 C)."""
    t = np.asarray(t_c, np.float64)
    return 6.1121 * np.exp(22.587 * t / (273.86 + t))


def relative_humidity_ice(t_f, td_f):
    """Relative humidity (%) with respect to *ice* below freezing and with respect to liquid water at or above 0 C.

    The vapour pressure comes from the (liquid-referenced) dew point, as published, so air that is exactly saturated
    over water reads ~110 % at 15 F and ~140 % at -30 F: supersaturated with respect to ice, which is where frost, rime
    and ice-crystal growth happen.  Not clipped at 100 %.
    """
    t = _f_to_c(t_f)
    td = np.minimum(_f_to_c(td_f), t)
    e = sat_vapor_pressure(td)
    es = np.where(t < 0.0, sat_vapor_pressure_ice(t), sat_vapor_pressure(t))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(t) & np.isfinite(td), 100.0 * e / es, np.nan).astype(np.float32)


def mixing_ratio(td_f, p_mb):
    """Water-vapour mixing ratio (g/kg) from the 2 m dew point (°F) and station pressure (mb)."""
    e = sat_vapor_pressure(_f_to_c(td_f))
    p = np.asarray(p_mb, np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(e) & np.isfinite(p), 1000.0 * _EPS * e / (p - e), np.nan).astype(np.float32)


def wet_bulb(t_f, td_f, p_mb, iterations: int = 26):
    """Thermodynamic wet-bulb temperature (°F) from 2 m temperature, 2 m dew point (°F) and *station* pressure (mb).

    The temperature the air reaches when water evaporates into it at constant pressure until saturated (isobaric
    adiabatic saturation), i.e. the root Tw of

        (cp_d + r cp_v) (T - Tw) = Lv(Tw) (r_s(Tw) - r),     Lv = 2.501e6 - 2370 Tw  (Tw in °C)

    with the ambient mixing ratio r from the dew point and r_s the saturation mixing ratio over liquid water.  The
    residual is monotone in Tw and the root is bracketed by Td and T, so a fixed number of bisection steps is robust
    and vectorised (26 steps -> < 1e-6 °C).  Using the real surface pressure matters at elevation: for the same
    T / Td the wet bulb is lower at 850 mb than at 1013 mb (a few tenths of a °F in moist air, up to ~1 °F in dry air).

    Checked against MetPy's wet-bulb temperature on 500 random states (T -25..30 °C, RH 20-100 %, 820-1040 mb): 0.08 °C
    rms overall and 0.02 °C rms (0.06 °C worst) below 2 °C.  It is the wet bulb with respect to liquid water (the
    NWS / METAR convention), not the ice-bulb temperature.
    """
    t = _f_to_c(t_f)
    td = np.minimum(_f_to_c(td_f), t)             # an analysed dew point a hair above T would break the bracket
    p = np.asarray(p_mb, np.float64)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        e = sat_vapor_pressure(td)
        r = _EPS * e / (p - e)
        cp = _CPD + r * _CPV
        lo, hi = td.copy(), t.copy()
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            es = sat_vapor_pressure(mid)
            f = (2.501e6 - 2370.0 * mid) * (_EPS * es / (p - es) - r) - cp * (t - mid)
            hot = f > 0.0
            hi = np.where(hot, mid, hi)
            lo = np.where(hot, lo, mid)
        tw = _c_to_f(0.5 * (lo + hi))
    return np.where(np.isfinite(t) & np.isfinite(td) & np.isfinite(p), tw, np.nan).astype(np.float32)


# ---------------------------------------------------------------------------------------------------- pressure
G0 = 9.80665                # m s-2
RD = 287.058                # J kg-1 K-1, dry air
STD_LAPSE = 0.0065          # K m-1, ICAO standard atmosphere
FT_PER_M = 3.280839895


def virtual_temperature_k(t_f, td_f, p_mb):
    """Virtual temperature (K) from 2 m temperature / dew point (°F) and pressure (mb)."""
    t_c = _f_to_c(t_f)
    e = sat_vapor_pressure(np.minimum(_f_to_c(td_f), t_c))
    return (t_c + 273.15) / (1.0 - 0.378 * e / np.asarray(p_mb, np.float64))


def mslp(p_mb, terrain_ft, t_f, td_f):
    """Mean sea-level pressure (mb) reduced from RTMA/URMA *surface* pressure.

    Hypsometric reduction through a fictitious column below the surface (the standard textbook / MetPy / WRF-post
    approach):  MSLP = P * exp( g z / (Rd * Tv_mean) ),  Tv_mean = Tv_2m + 0.5 * 6.5 K/km * z,
    with the *virtual* temperature (so humidity is accounted for) and z the analysis-grid terrain height.
    RTMA publishes no sea-level pressure, so this is a diagnostic: it reproduces a standard atmosphere exactly, is
    smooth over water and flat land, and carries the usual uncertainty over high terrain (z ~ 1 km: a 5 K error in
    the assumed column temperature is ~1.5 mb), especially under cold pools / inversions.
    """
    p = np.asarray(p_mb, np.float64)
    z = np.maximum(np.asarray(terrain_ft, np.float64) / FT_PER_M, 0.0)      # m
    with np.errstate(invalid="ignore", over="ignore"):
        tv_mean = virtual_temperature_k(t_f, td_f, p) + 0.5 * STD_LAPSE * z
        out = p * np.exp(G0 * z / (RD * tv_mean))
    return np.where(np.isfinite(out), out, np.nan).astype(np.float32)


# ---------------------------------------------------------------------------------------------------- wind direction
CALM_MPH = 2.0      # METAR calls <= 2 kt calm; RTMA encodes calm as direction 0, which would read as a north wind


def wind_direction(dir_deg, wind_mph):
    """10 m wind direction (° the wind blows *from*), NaN where the air is calm and the direction undefined."""
    d = np.asarray(dir_deg, np.float64)
    v = np.asarray(wind_mph, np.float64)
    return np.where(np.isfinite(d) & np.isfinite(v) & (v >= CALM_MPH), d, np.nan).astype(np.float32)


# ---------------------------------------------------------------------------------------------------- terrain-adjusted
SLT_LAPSE_F_PER_FT = 1.8 * STD_LAPSE / FT_PER_M         # 6.5 K/km = 3.57 F per 1000 ft
DEWPOINT_LAPSE_F_PER_FT = 1.8 * 0.0018 / FT_PER_M       # dew point falls ~1.8 K/km in well-mixed air = 0.99 F per 1000 ft


def sea_level_temperature(t_f, terrain_ft):
    """2 m temperature (°F) reduced to sea level with a 6.5 K/km lapse rate: removes the terrain signal so air-mass
    contrasts, cold pools and inversions stand out."""
    z = np.maximum(np.asarray(terrain_ft, np.float64), 0.0)
    return (np.asarray(t_f, np.float64) + SLT_LAPSE_F_PER_FT * z).astype(np.float32)


def sea_level_dewpoint(td_f, terrain_ft):
    """2 m dew point (°F) reduced to sea level with a 1.8 K/km lapse rate (used for moisture advection)."""
    z = np.maximum(np.asarray(terrain_ft, np.float64), 0.0)
    return (np.asarray(td_f, np.float64) + DEWPOINT_LAPSE_F_PER_FT * z).astype(np.float32)


# ---------------------------------------------------------------------------------------------------- winter hazards
# Approximate NWS wind-chill-chart frostbite zones (wind chill, °F).  The chart's shading is not a single wind-chill
# contour - its edges move a few degrees with wind speed - so these are round mid-chart values.
FROSTBITE_WC_F = (-18.0, -33.0, -48.0)       # 30 min, 10 min, 5 min


def frostbite_category(t_f, wind_mph):
    """0 = no frostbite concern, 1 = frostbite in ~30 min, 2 = ~10 min, 3 = ~5 min (exposed skin), from NWS wind chill."""
    t = np.asarray(t_f, np.float64)
    v = np.asarray(wind_mph, np.float64)
    wc = np.where((t <= WC_MAX_T_F) & (v > WC_MIN_WIND_MPH), wind_chill(t, v), t)
    cat = (wc <= FROSTBITE_WC_F[0]).astype(np.float32) + (wc <= FROSTBITE_WC_F[1]) + (wc <= FROSTBITE_WC_F[2])
    return np.where(np.isfinite(wc), cat, np.nan).astype(np.float32)


FLIGHT_CATEGORIES = ("LIFR", "IFR", "MVFR", "VFR")


def flight_category(vis_mi, ceiling_ft):
    """FAA / NWS flight category from visibility (statute miles) and ceiling (ft AGL): the worse of the two.

    0 LIFR  ceiling < 500 ft and/or visibility < 1 mi        2 MVFR  ceiling 1000-3000 ft and/or visibility 3-5 mi
    1 IFR   ceiling 500-999 ft and/or visibility 1 - <3 mi    3 VFR   ceiling > 3000 ft and visibility > 5 mi
    """
    vis = np.asarray(vis_mi, np.float64)
    ceil = np.asarray(ceiling_ft, np.float64)
    cat = np.full(vis.shape, 3.0)
    cat = np.where((ceil <= 3000.0) | (vis <= 5.0), 2.0, cat)
    cat = np.where((ceil < 1000.0) | (vis < 3.0), 1.0, cat)
    cat = np.where((ceil < 500.0) | (vis < 1.0), 0.0, cat)
    return np.where(np.isfinite(vis) & np.isfinite(ceil), cat, np.nan).astype(np.float32)


# ---------------------------------------------------------------------------------------------------- 24-hour summaries
# Inputs are stacks (n_hours + 1, ny, nx) of hourly analyses, index 0 = the valid time, NaN planes for missing hours.
def series_max(stack):
    return np.fmax.reduce(np.asarray(stack, np.float32), axis=0)


def series_min(stack):
    return np.fmin.reduce(np.asarray(stack, np.float32), axis=0)


def series_range(stack):
    return (series_max(stack) - series_min(stack)).astype(np.float32)


def series_sum(stack):
    """Total over an hourly stack of accumulations, using whatever hours exist: a NaN is an hour (or cell) with no data and is
    skipped, so a cell with no data in any hour stays NaN while an incomplete total reads low rather than disappearing."""
    st = np.asarray(stack, np.float32)
    out = np.nansum(st, axis=0, dtype=np.float32)
    out[~np.isfinite(st).any(axis=0)] = np.nan
    return out




def frozen_precipitation(pcp, t_f, td_f, p_mb):
    """Liquid-equivalent precipitation that fell while the surface wet bulb was at or below 32 °F.

    ``pcp[k]`` is the accumulation of the hour ending ``k`` hours before the valid time (k = 0 .. n-1); the temperature, dew
    point and pressure stacks have n + 1 planes (index k = k hours before the valid time), so each hour has the wet bulb at
    both of its ends.  An hour counts as frozen when the mean of those two wet bulbs is at or below 32 °F.

    Missing data (NaN planes) is skipped: an hour with no precipitation analysis adds nothing, and an hour with only one
    wet-bulb end uses that one; an hour with neither cannot be classified and adds nothing.  A cell with no usable hour at all is NaN."""
    pcp = np.asarray(pcp, np.float32)
    n = pcp.shape[0] - 1                                    # the stacks share one length: n hourly accumulations, n + 1 instants

    def wb_at(k):                                           # one plane at a time: a whole 49-hour stack of float64 temporaries
        return wet_bulb(np.asarray(t_f[k], np.float64),     # needs gigabytes
                        np.asarray(td_f[k], np.float64), np.asarray(p_mb[k], np.float64))

    total = np.zeros(pcp.shape[1:], np.float32)
    seen = np.zeros(pcp.shape[1:], bool)
    b = wb_at(0)
    for k in range(n):
        a, b = b, wb_at(k + 1)
        with np.errstate(invalid="ignore"):
            mid = np.where(np.isfinite(a) & np.isfinite(b), 0.5 * (a + b), np.where(np.isfinite(a), a, b))
            usable = np.isfinite(mid) & np.isfinite(pcp[k])
            total += np.where(usable & (mid <= 32.0), pcp[k], np.float32(0.0))
        seen |= usable
    total[~seen] = np.nan
    return total


def _below_and_crossing(stack, threshold):
    a = np.asarray(stack[:-1], np.float64) - threshold
    b = np.asarray(stack[1:], np.float64) - threshold
    valid = np.isfinite(a) & np.isfinite(b)
    cross = ((a <= 0.0) & (b > 0.0)) | ((a > 0.0) & (b <= 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        part = np.where(a <= 0.0, -a / (b - a), -b / (a - b))        # share of the hour spent at or below the threshold
    below = np.where(cross, part, ((a <= 0.0) & (b <= 0.0)).astype(float))
    return below, cross, valid


def hours_at_or_below(stack, threshold: float = 32.0):
    """Hours at or below ``threshold`` over the series, interpolating linearly between hourly analyses.  If some
    hourly analyses are missing the total is scaled to the full period."""
    below, _, valid = _below_and_crossing(stack, threshold)
    n = valid.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(n > 0, (below * valid).sum(0) * (len(stack) - 1) / n, np.nan)
    return out.astype(np.float32)


def threshold_crossings(stack, threshold: float = 32.0):
    """How many times the series crossed ``threshold`` (in either direction)."""
    _, cross, valid = _below_and_crossing(stack, threshold)
    n = valid.sum(0)
    return np.where(n > 0, (cross & valid).sum(0), np.nan).astype(np.float32)


def min_wind_chill(t_stack, wind_stack):
    """Lowest hourly wind chill (°F) in the series; the air temperature where wind chill is not defined."""
    t = np.asarray(t_stack, np.float64)
    v = np.asarray(wind_stack, np.float64)
    wc = np.where((t <= WC_MAX_T_F) & (v > WC_MIN_WIND_MPH), wind_chill(t, v), t)
    return np.fmin.reduce(wc, axis=0).astype(np.float32)


# ------------------------------------------------------------------------------------------------ winter hazard diagnostics
# All are surface diagnostics from the analysed 2 m / 10 m fields: indicators of where conditions favour a hazard, not forecasts
# and not substitutes for the observations or the air aloft.
def wet_bulb_plane(t_f, td_f, p_mb, k: int):
    """Wet bulb (°F) of hour-plane ``k`` of the stacks, one plane at a time (a whole stack of float64 temporaries is gigabytes)."""
    return wet_bulb(np.asarray(t_f[k], np.float64), np.asarray(td_f[k], np.float64), np.asarray(p_mb[k], np.float64))


def _mid_instants(a, b):
    """Mean of an hour's two end values, or the one that exists; NaN if neither."""
    both = np.isfinite(a) & np.isfinite(b)
    return np.where(both, 0.5 * (a + b), np.where(np.isfinite(a), a, b))


BLIZZARD_VIS_MI = 0.25         # NWS blizzard: visibility 1/4 mile or less ...
BLIZZARD_WIND_MPH = 35.0       # ... with sustained wind or frequent gusts of 35 mph or more ...
BLIZZARD_MIN_HOURS = 3         # ... lasting 3 hours or longer


def blizzard_hours(vis_mi, wind_mph, gust_mph):
    """Longest run of consecutive hourly analyses meeting the NWS blizzard conditions (visibility <= 1/4 mi and sustained wind or
    gust >= 35 mph).  Each analysis stands for an hour, so 3 or more is the 3-hour blizzard criterion.  NaN where the visibility
    is missing in every hour."""
    vis = np.asarray(vis_mi, np.float64)
    wind = np.fmax(np.asarray(wind_mph, np.float64), np.asarray(gust_mph, np.float64))
    met = (vis <= BLIZZARD_VIS_MI) & (wind >= BLIZZARD_WIND_MPH)
    run = np.zeros(vis.shape[1:], np.float32)
    best = run.copy()
    for k in range(vis.shape[0] - 1, -1, -1):                   # oldest hour first; runs may end at any hour
        run = np.where(met[k], run + 1.0, 0.0).astype(np.float32)
        best = np.maximum(best, run)
    best[~np.isfinite(vis).any(axis=0)] = np.nan
    return best


def _box_sum(a, radius: int):
    """Sum over the (2 radius + 1)-square window around every cell (truncated at the edges), by integral image."""
    size = 2 * radius + 1
    c = np.pad(np.asarray(a, np.float64), ((radius + 1, radius), (radius + 1, radius))).cumsum(0).cumsum(1)
    return c[size:, size:] - c[:-size, size:] - c[size:, :-size] + c[:-size, :-size]


COLD_POOL_RADIUS_CELLS = 10         # 21 x 21 cells = about 52 km


def cold_pool(slt_f, terrain_ft, radius: int = COLD_POOL_RADIUS_CELLS):
    """Sea-level-reduced temperature minus its mean over the land cells within about 25 km: negative where the air is colder than
    the surroundings once elevation is accounted for (valley cold pools, radiational cooling, cold-air damming), positive where
    it is warmer (inversions on slopes and ridges, warm air spilling downslope).  Water cells are left out of the mean and of
    the map."""
    slt = np.asarray(slt_f, np.float64)
    land = (np.asarray(terrain_ft, np.float64) > 3.3) & np.isfinite(slt)
    num = _box_sum(np.where(land, slt, 0.0), radius)
    den = _box_sum(land.astype(np.float64), radius)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(land & (den > 0), slt - num / den, np.nan)
    return out.astype(np.float32)


FLASH_FREEZE_FALL_F = 10.0
FLASH_FREEZE_MIN_LIQUID_IN = 0.01


def flash_freeze(pcp, t_f):
    """Flash-freeze indicator over the series (3 h): 0 none; 1 the temperature fell from above to at or below 32 °F with liquid
    precipitation in the period (a wet surface freezing); 2 the same with a fall of 10 °F or more."""
    pcp = np.asarray(pcp, np.float32)
    t = np.asarray(t_f, np.float64)
    n = pcp.shape[0] - 1
    now, then = t[0], t[-1]
    wet = np.nansum(pcp[:n], axis=0) >= FLASH_FREEZE_MIN_LIQUID_IN
    crossed = (then > 32.0) & (now <= 32.0) & wet
    cat = np.where(crossed, np.where(then - now >= FLASH_FREEZE_FALL_F, 2.0, 1.0), 0.0)
    return np.where(np.isfinite(now) & np.isfinite(then), cat, np.nan).astype(np.float32)


def freezing_fog(t_f, td_f, vis_mi, ceiling_ft):
    """Freezing fog / drizzle favourable: 2 m temperature at or below 32 °F, dew point depression 2 °F or less, and visibility
    under 3 mi or a ceiling under 1000 ft (1); dense when visibility is under 1/2 mi (2)."""
    t, td = np.asarray(t_f, np.float64), np.asarray(td_f, np.float64)
    vis, ceil = np.asarray(vis_mi, np.float64), np.asarray(ceiling_ft, np.float64)
    fav = (t <= 32.0) & ((t - td) <= 2.0) & ((vis < 3.0) | (ceil < 1000.0))
    cat = np.where(fav, np.where(vis < 0.5, 2.0, 1.0), 0.0)
    return np.where(np.isfinite(t) & np.isfinite(td) & np.isfinite(vis) & np.isfinite(ceil), cat, np.nan).astype(np.float32)


WET_SNOW_MIN_T_F = 31.0
WET_SNOW_GUST_MPH = 35.0


def wet_snow_load(pcp, t_f, td_f, p_mb, gust_mph):
    """Heavy, sticky snow that loads trees and lines.  Counts the liquid precipitation of the hours in which the wet bulb was at
    or below 32 °F while the air itself was 31 °F or warmer (marginal temperatures, where snow is densest), over the series:
    0 less than 0.10 in; 1 at least 0.10 in; 2 at least 0.25 in; 3 at least 0.25 in with a gust of 35 mph or more in the period."""
    pcp = np.asarray(pcp, np.float32)
    n = pcp.shape[0] - 1
    liquid = np.zeros(pcp.shape[1:], np.float32)
    seen = np.zeros(pcp.shape[1:], bool)
    wb_b, t_b = wet_bulb_plane(t_f, td_f, p_mb, 0), np.asarray(t_f[0], np.float64)
    for k in range(n):
        wb_a, t_a = wb_b, t_b
        wb_b, t_b = wet_bulb_plane(t_f, td_f, p_mb, k + 1), np.asarray(t_f[k + 1], np.float64)
        wb, t = _mid_instants(wb_a, wb_b), _mid_instants(t_a, t_b)
        usable = np.isfinite(wb) & np.isfinite(t) & np.isfinite(pcp[k])
        liquid += np.where(usable & (wb <= 32.0) & (t >= WET_SNOW_MIN_T_F), pcp[k], np.float32(0.0))
        seen |= usable
    gust = np.fmax.reduce(np.asarray(gust_mph, np.float64), axis=0)
    cat = np.where(liquid >= 0.25, np.where(gust >= WET_SNOW_GUST_MPH, 3.0, 2.0), np.where(liquid >= 0.10, 1.0, 0.0))
    cat[~seen] = np.nan
    return cat.astype(np.float32)


SNOWMAKING_WET_BULB_F = 28.0


def snowmaking_hours(t_f, td_f, p_mb, threshold: float = SNOWMAKING_WET_BULB_F):
    """Hours with the wet bulb at or below ``threshold`` (28 °F is the usual ski-area snowmaking limit), interpolating between
    the hourly analyses; scaled to the full period if hours are missing."""
    n = np.asarray(t_f).shape[0]
    wb = np.stack([wet_bulb_plane(t_f, td_f, p_mb, k) for k in range(n)]).astype(np.float32)
    return hours_at_or_below(wb, threshold)


def radiational_cooling(cloud_pct, wind_mph):
    """Radiational cooling setup (2 m, current analysis): 0 neither clear nor calm; 1 clear (<= 25 % cloud) or calm (<= 5 mph);
    2 both; 3 both and very favourable (<= 10 % cloud and <= 2 mph)."""
    cloud, wind = np.asarray(cloud_pct, np.float64), np.asarray(wind_mph, np.float64)
    cat = (cloud <= 25.0).astype(float) + (wind <= 5.0)
    cat = np.where((cat == 2.0) & (cloud <= 10.0) & (wind <= 2.0), 3.0, cat)
    return np.where(np.isfinite(cloud) & np.isfinite(wind), cat, np.nan).astype(np.float32)


def hours_at_or_above(stack, threshold: float):
    """Hours at or above ``threshold`` (interpolated between hourly analyses; scaled if hours are missing)."""
    return hours_at_or_below(-np.asarray(stack, np.float32), -threshold)


def wind_chill_hours(t_stack, wind_stack, threshold: float):
    """Hours with the NWS wind chill at or below ``threshold`` (the air temperature where wind chill is not defined)."""
    t, v = np.asarray(t_stack, np.float64), np.asarray(wind_stack, np.float64)
    wc = np.where((t <= WC_MAX_T_F) & (v > WC_MIN_WIND_MPH), wind_chill(t, v), t)
    return hours_at_or_below(wc.astype(np.float32), threshold)


def degree_hours(stack, above: bool = True, base: float = 32.0):
    """Degree-hours (°F·h) above (melt) or below (freezing) ``base`` over the series, by the trapezoid rule on the hourly
    temperatures; scaled to the full period if hours are missing."""
    s = np.asarray(stack, np.float64)
    e = np.maximum(s - base, 0.0) if above else np.maximum(base - s, 0.0)
    seg = 0.5 * (e[:-1] + e[1:])
    ok = np.isfinite(seg)
    n = ok.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(n > 0, np.where(ok, seg, 0.0).sum(0) * (len(s) - 1) / n, np.nan)
    return out.astype(np.float32)


def precip_on_cold_surface(pcp, t_f):
    """Liquid precipitation (in) that fell in hours when the 2 m air temperature (mean of the hour's two ends) was at or below
    32 °F: water arriving on a frozen surface.  Where the air aloft is warm, this is freezing rain's footprint; it also counts
    snow.  NaN where no hour is usable."""
    pcp = np.asarray(pcp, np.float32)
    t = np.asarray(t_f, np.float64)
    n = pcp.shape[0] - 1
    total = np.zeros(pcp.shape[1:], np.float32)
    seen = np.zeros(pcp.shape[1:], bool)
    for k in range(n):
        mid = _mid_instants(t[k], t[k + 1])
        usable = np.isfinite(mid) & np.isfinite(pcp[k])
        total += np.where(usable & (mid <= 32.0), pcp[k], np.float32(0.0))
        seen |= usable
    total[~seen] = np.nan
    return total

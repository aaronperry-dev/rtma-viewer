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

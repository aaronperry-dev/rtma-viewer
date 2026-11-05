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

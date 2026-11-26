import numpy as np
import pytest

from rtma_viewer import derived as d
from rtma_viewer.config import SOURCE_VARIABLES, VARIABLES


def test_wind_chill_matches_nws_examples():
    assert d.wind_chill(0, 15) == pytest.approx(-19.4, abs=0.1)       # NWS chart page: 0 F, 15 mph -> -19 F
    assert d.wind_chill(30, 10) == pytest.approx(21.2, abs=0.1)       # 30 F, 10 mph -> ~21 F
    assert d.wind_chill(-5, 25) == pytest.approx(-31.0, abs=1.0)


def test_heat_index_matches_nws_table():
    assert d.heat_index(90, 70) == pytest.approx(105.9, abs=0.6)      # 90 F / 70 % RH -> ~106 F
    assert d.heat_index(96, 65) == pytest.approx(121, abs=1.5)        # 96 F / 65 % RH -> ~121 F
    assert d.heat_index(100, 40) == pytest.approx(109.5, abs=1.0)     # 100 F / 40 % RH -> ~109-110 F
    assert d.heat_index(80, 40) == pytest.approx(80, abs=1.0)


def _regression(t, rh):
    c = d._HI_COEF
    return (c[0] + c[1] * t + c[2] * rh + c[3] * t * rh + c[4] * t ** 2 + c[5] * rh ** 2
            + c[6] * t ** 2 * rh + c[7] * t * rh ** 2 + c[8] * t ** 2 * rh ** 2)


def test_heat_index_adjustments():
    # low humidity (RH < 13 %, 80-112 F): subtract ((13-RH)/4) * sqrt((17-|T-95|)/17)  -> exactly 2 F at 95 F / 5 %
    assert d.heat_index(95, 5) == pytest.approx(_regression(95, 5) - 2.0)
    # high humidity (RH > 85 %, 80-87 F): add ((RH-85)/10) * ((87-T)/5)  -> 0.4 F at 85 F / 95 %
    assert d.heat_index(85, 95) == pytest.approx(_regression(85, 95) + 0.4)
    # outside both windows there is no adjustment
    assert d.heat_index(95, 50) == pytest.approx(_regression(95, 50))
    # cool/dry: the Steadman simple formula (averaged with T < 80) is returned, not the regression
    assert d.heat_index(60, 50) == pytest.approx(0.5 * (60 + 61 + 1.2 * (60 - 68) + 0.094 * 50))


def test_relative_humidity():
    assert d.relative_humidity(50, 50) == pytest.approx(100.0)
    assert d.relative_humidity(86, 68) == pytest.approx(54.0, abs=1.5)     # 30 C / 20 C -> ~55 %
    assert d.relative_humidity(40, 45) == pytest.approx(100.0)             # dew point above T is clipped
    assert d.relative_humidity(20, -20) < 25


def test_apparent_temperature_regimes():
    t = np.array([30.0, 30.0, 50.0, 50.0, 65.0, 80.0, 80.1, 95.0, 20.0])
    td = np.array([20.0, 20.0, 40.0, 40.0, 55.0, 70.0, 70.0, 75.0, 10.0])
    v = np.array([15.0, 2.0, 10.0, 3.0, 30.0, 20.0, 5.0, 5.0, 3.0001])
    a = d.apparent_temperature(t, td, v)
    assert a[0] == pytest.approx(d.wind_chill(30, 15), abs=1e-4) and a[0] < 30
    assert a[1] == 30.0                    # calm: wind chill undefined, ambient used
    assert a[2] == pytest.approx(d.wind_chill(50, 10), abs=1e-4)     # 50 F is included in the wind chill regime
    assert a[3] == 50.0                    # wind must exceed 3 mph
    assert a[4] == 65.0                    # mild: ambient, however windy
    assert a[5] == 80.0                    # heat index only *above* 80 F
    assert a[6] > 80.1 - 1e-3 and a[6] == pytest.approx(d.heat_index(80.1, d.relative_humidity(80.1, 70)), abs=1e-3)
    assert a[7] > 95.0                     # humid heat feels hotter
    assert a[8] < 20.0


def test_apparent_nan_propagates_and_dtype():
    a = d.apparent_temperature(np.array([30.0, np.nan]), np.array([20.0, 20.0]), np.array([10.0, 10.0]))
    assert a.dtype == np.float32 and np.isfinite(a[0]) and np.isnan(a[1])


def test_apparent_is_a_derived_variable():
    v = VARIABLES["apparent"]
    assert v.derive is d.apparent_temperature and set(v.requires) <= set(SOURCE_VARIABLES)
    assert "apparent" not in SOURCE_VARIABLES


# ------------------------------------------------------------------------------------------ moisture / pressure fields
C2F = lambda c: c * 1.8 + 32.0


def test_dewpoint_depression():
    a = d.dewpoint_depression(np.array([30.0, 30.0, np.nan]), np.array([20.0, 31.0, 10.0]))
    assert a.dtype == np.float32 and a[0] == pytest.approx(10.0) and a[1] == 0.0 and np.isnan(a[2])   # never negative


def test_relative_humidity_field_is_float32_percent():
    a = d.relative_humidity_field(np.array([50.0]), np.array([50.0]))
    assert a.dtype == np.float32 and a[0] == pytest.approx(100.0)


@pytest.mark.parametrize("t_c, td_c, p_mb, tw_c", [          # MetPy thermodynamic wet-bulb temperature (the reference)
    (20.0, 9.27, 1013.25, 13.672), (0.0, -9.3, 1013.25, -2.937), (-5.0, -12.0, 1000.0, -6.882),
    (2.0, 0.0, 1000.0, 1.163), (-12.0, -20.0, 850.0, -13.614), (10.0, 5.0, 900.0, 7.323),
])
def test_wet_bulb_matches_metpy(t_c, td_c, p_mb, tw_c):
    tw = (float(d.wet_bulb(C2F(t_c), C2F(td_c), p_mb)) - 32) / 1.8
    assert tw == pytest.approx(tw_c, abs=0.13 if t_c > 5 else 0.04)       # warm/humid: <=0.12 C; sub-freezing/cold: <=0.03 C


def test_wet_bulb_properties():
    rng = np.random.default_rng(1)
    t = rng.uniform(-30, 100, 5000)
    td = t - rng.uniform(0, 40, 5000)
    p = rng.uniform(800, 1050, 5000)
    tw = d.wet_bulb(t, td, p)
    assert np.all(tw <= t + 1e-3) and np.all(tw >= td - 1e-3)               # always between dew point and temperature
    assert float(d.wet_bulb(40.0, 40.0, 1000.0)) == pytest.approx(40.0, abs=1e-3)      # saturated: Tw = T = Td
    assert float(d.wet_bulb(40.0, 45.0, 1000.0)) == pytest.approx(40.0, abs=1e-3)      # Td above T is clipped, not NaN
    # for the same T / Td, lower pressure (higher ground) means a lower wet bulb
    assert float(d.wet_bulb(30.0, 10.0, 850.0)) < float(d.wet_bulb(30.0, 10.0, 1013.0)) - 0.4
    # monotone in dew point
    dps = np.linspace(-10, 30, 20)
    assert np.all(np.diff(d.wet_bulb(np.full(20, 30.0), dps, np.full(20, 1000.0))) > 0)


def test_wet_bulb_nan_propagates_and_dtype():
    a = d.wet_bulb(np.array([30.0, np.nan, 30.0]), np.array([20.0, 20.0, 20.0]), np.array([1000.0, 1000.0, np.nan]))
    assert a.dtype == np.float32 and np.isfinite(a[0]) and np.isnan(a[1]) and np.isnan(a[2])


def test_mslp_reduction():
    # sea level: identity
    assert float(d.mslp(1013.0, 0.0, 50.0, 40.0)) == pytest.approx(1013.0)
    # ICAO standard atmosphere: 898.746 mb at 1000 m with T = 8.5 C -> 1013.25 mb (dry air, so Tv = T)
    t = C2F(8.5)
    dry = float(d.mslp(898.746, 1000 * d.FT_PER_M, t, t - 120))
    assert dry == pytest.approx(1013.25, abs=0.08)
    # moist air is less dense (virtual temperature is higher), so the same station pressure reduces to a lower MSLP
    assert float(d.mslp(898.746, 1000 * d.FT_PER_M, t, t - 1)) < dry
    # colder column -> larger reduction; higher ground -> larger reduction
    assert float(d.mslp(900.0, 3000.0, 10.0, 0.0)) > float(d.mslp(900.0, 3000.0, 60.0, 50.0))
    z = np.array([0.0, 500.0, 1000.0, 3000.0])
    assert np.all(np.diff(d.mslp(np.full(4, 950.0), z, np.full(4, 30.0), np.full(4, 20.0))) > 0)
    # below-sea-level terrain is treated as sea level
    assert float(d.mslp(1013.0, -20.0, 50.0, 40.0)) == pytest.approx(1013.0)
    a = d.mslp(np.array([1000.0, np.nan]), np.array([100.0, 100.0]), np.array([40.0, 40.0]), np.array([30.0, 30.0]))
    assert a.dtype == np.float32 and np.isfinite(a[0]) and np.isnan(a[1])


# Real RTMA cell values (terrain m, 2 m T °F, 2 m Td °F, surface pressure mb) at New England airports from the 2026-11-26 02Z
# analysis, with the observed METAR sea-level pressure (mb) from the 02:00Z reports (aviationweather.gov).
_RTMA_VS_METAR = [
    ("KBOS", 1, 39.29, 31.77, 1041.23, 1041.8), ("KPWM", 0, 33.15, 26.56, 1042.14, 1042.6),
    ("KBGR", 52, 24.08, 19.31, 1036.18, 1042.8), ("KBTV", 71, 24.48, 21.36, 1033.25, 1042.0),
    ("KPVD", 25, 37.27, 32.83, 1037.84, 1041.4), ("KCON", 109, 26.46, 23.00, 1028.40, 1042.5),
    ("KBDL", 52, 32.94, 25.79, 1034.57, 1041.4),
]


def test_mslp_matches_observed_metar_slp():
    """Independent check of the whole reduction: derived MSLP vs what the airports actually reported."""
    err = []
    for icao, z_m, t, td, p, obs in _RTMA_VS_METAR:
        m = float(d.mslp(p, z_m * d.FT_PER_M, t, td))
        err.append(m - obs)
        assert abs(m - obs) < 1.0, (icao, m, obs)
    err = np.array(err)
    assert abs(err.mean()) < 0.3 and np.sqrt((err ** 2).mean()) < 0.6        # measured: bias -0.06 mb, rms 0.37 mb


def test_mslp_is_sensitive_to_terrain_height_as_expected():
    # ~1.2 mb per 10 m of height near sea level: why MSLP over smoothed 2.5 km terrain needs the caveat in the notes
    icao, z_m, t, td, p, obs = _RTMA_VS_METAR[5]
    lo, hi = (float(d.mslp(p, (z_m + dz) * d.FT_PER_M, t, td)) for dz in (0, 10))
    assert hi - lo == pytest.approx(1.2, abs=0.2)


def test_new_derived_variables_are_registered_and_ordered_correctly():
    for key, req in {"wetbulb": ("temp", "dewpoint", "pressure"), "rh": ("temp", "dewpoint"),
                     "dpdep": ("temp", "dewpoint"), "mslp": ("pressure", "terrain", "temp", "dewpoint")}.items():
        v = VARIABLES[key]
        assert v.requires == req and set(req) <= set(SOURCE_VARIABLES) and key not in SOURCE_VARIABLES
    assert VARIABLES["wetbulb"].derive is d.wet_bulb and VARIABLES["mslp"].derive is d.mslp
    assert SOURCE_VARIABLES["terrain"].static and not SOURCE_VARIABLES["terrain"].changes
    assert SOURCE_VARIABLES["pressure"].pack == (700.0, 100.0)


def test_wind_direction_masks_calm_air():
    a = d.wind_direction(np.array([0.0, 0.0, 270.0, 359.0, 90.0, np.nan]), np.array([0.0, 5.0, 1.9, 2.0, 10.0, 10.0]))
    assert a.dtype == np.float32
    assert np.isnan(a[0]) and np.isnan(a[2]) and np.isnan(a[5])               # calm (< 2 mph) or missing -> undefined
    assert a[1] == 0.0 and a[3] == 359.0 and a[4] == 90.0                       # a real north wind keeps its direction
    assert VARIABLES["wdir"].derive is d.wind_direction

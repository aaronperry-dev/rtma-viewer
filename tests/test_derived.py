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


# ------------------------------------------------------------------------------------------ winter diagnostics
def test_rh_ice_matches_saturation_pressures_and_water_above_freezing():
    # exactly saturated over liquid water: RH_ice = e_w(T) / e_i(T)  (1.2540 / 1.0320 hPa at -20 C is 121.5 %)
    for t_c, expect in ((-20.0, 121.5), (-40.0, 148.0), (-9.44, 109.8)):
        assert float(d.relative_humidity_ice(C2F(t_c), C2F(t_c))) == pytest.approx(expect, abs=1.0), t_c
    assert float(d.sat_vapor_pressure_ice(-20.0)) == pytest.approx(1.032, abs=0.003)      # standard tables
    assert float(d.sat_vapor_pressure_ice(-40.0)) == pytest.approx(0.1283, abs=0.001)
    # at or above freezing ice does not exist: identical to ordinary RH (continuous across 0 C within 0.1 %)
    for t, td in ((50.0, 40.0), (33.0, 30.0), (80.0, 60.0)):
        assert float(d.relative_humidity_ice(t, td)) == pytest.approx(float(d.relative_humidity(t, td)), abs=1e-3)
    assert float(d.relative_humidity_ice(32.0, 32.0)) == pytest.approx(100.0, abs=0.2)
    # supersaturated over ice whenever saturated over water below freezing, and not clipped at 100
    t = np.linspace(-40, 31, 60)
    assert np.all(d.relative_humidity_ice(t, t) > 100.0) and d.relative_humidity_ice(t, t).dtype == np.float32
    assert np.isnan(d.relative_humidity_ice(np.nan, 10.0))


def test_mixing_ratio_matches_metpy_and_is_independent_of_temperature():
    mpcalc = pytest.importorskip("metpy.calc")
    units = pytest.importorskip("metpy.units").units
    for td_f, p in ((32.0, 1000.0), (-10.0, 1000.0), (50.0, 900.0), (20.0, 850.0)):
        td_c = (td_f - 32) / 1.8
        e = mpcalc.saturation_vapor_pressure(td_c * units.degC)
        ref = float(mpcalc.mixing_ratio(e, p * units.hPa).to("g/kg").m)
        assert float(d.mixing_ratio(td_f, p)) == pytest.approx(ref, rel=0.01), (td_f, p)
    assert float(d.mixing_ratio(32.0, 1000.0)) == pytest.approx(3.82, abs=0.05)
    assert float(d.mixing_ratio(-10.0, 1000.0)) < 1.0                                    # "bitter, dry Arctic air"
    assert float(d.mixing_ratio(10.0, 1000.0)) == pytest.approx(1.5, abs=0.1)            # the number quoted on the page


def test_sea_level_temperature_adds_standard_lapse():
    assert float(d.sea_level_temperature(10.0, 3000.0)) == pytest.approx(10.0 + 3 * 3.57, abs=0.05)
    assert float(d.sea_level_temperature(10.0, 1000.0)) == pytest.approx(10.0 + 11.7 / 3.28084, abs=0.01)   # 6.5 K/km
    assert float(d.sea_level_temperature(10.0, -50.0)) == 10.0                            # below sea level: no adjustment
    assert float(d.sea_level_dewpoint(10.0, 3281.0)) == pytest.approx(10.0 + 3.24, abs=0.05)   # 1.8 K/km = 3.24 F/km


@pytest.mark.parametrize("vis, ceil, cat", [
    (10, 12000, 3), (5.01, 3001, 3),                    # VFR: ceiling > 3000 and visibility > 5
    (5.0, 12000, 2), (10, 3000, 2), (3.0, 12000, 2), (10, 1000, 2),   # MVFR edges (inclusive)
    (2.9, 12000, 1), (10, 999, 1), (1.0, 12000, 1), (10, 500, 1),     # IFR
    (0.9, 12000, 0), (10, 499, 0), (0.0, 0, 0),         # LIFR
    (0.5, 2000, 0), (2, 400, 0), (4, 800, 1),           # the worse of the two wins
])
def test_flight_category_boundaries(vis, ceil, cat):
    assert float(d.flight_category(vis, ceil)) == cat


def test_flight_category_missing_is_nan():
    assert np.isnan(d.flight_category(np.nan, 5000.0)) and np.isnan(d.flight_category(5.0, np.nan))
    assert d.FLIGHT_CATEGORIES == ("LIFR", "IFR", "MVFR", "VFR")


@pytest.mark.parametrize("t, v, cat", [
    (0, 15, 1),          # NWS: 0 F / 15 mph -> wind chill -19 F -> frostbite in 30 minutes
    (-15, 20, 2),        # -42 F -> ~10 minutes (the commonly quoted NWS example)
    (-20, 25, 3),        # -51 F -> 5 minutes
    (10, 10, 0), (30, 20, 0), (60, 30, 0),      # no frostbite concern
    (-30, 2, 1),         # light wind: wind chill is not defined, so the air temperature itself is compared (-30 F: 30 min)
    (-10, 2, 0), (-50, 0, 3),                   # ... which only registers in extreme cold
])
def test_frostbite_zones_follow_the_nws_chart(t, v, cat):
    assert float(d.frostbite_category(t, v)) == cat


def test_series_extremes_and_range_ignore_missing_hours():
    s = np.array([[[10.0, 5.0]], [[np.nan, 7.0]], [[30.0, 6.0]], [[20.0, np.nan]]], np.float32)   # (4, 1, 2)
    assert d.series_max(s).tolist() == [[30.0, 7.0]] and d.series_min(s).tolist() == [[10.0, 5.0]]
    assert d.series_range(s).tolist() == [[20.0, 2.0]]
    blank = np.full((3, 1, 1), np.nan, np.float32)
    assert np.isnan(d.series_max(blank)).all() and np.isnan(d.series_range(blank)).all()


def test_hours_at_or_below_freezing_interpolates_linearly():
    i = np.arange(25, dtype=np.float32)
    ramp = (20.0 + i)[:, None, None]                                   # index 0 = now = 20 F ... 24 h ago = 44 F
    assert float(d.hours_at_or_below(ramp)[0, 0]) == pytest.approx(12.0, abs=1e-4)       # 20..32 F: exactly 12 h at/below
    assert float(d.threshold_crossings(ramp)[0, 0]) == 1.0
    assert float(d.hours_at_or_below(ramp + 20)[0, 0]) == 0.0 and float(d.hours_at_or_below(ramp - 40)[0, 0]) == 24.0
    # a crossing inside an interval counts its fraction: -1 F then +3 F spends 1/4 of the hour at or below 0
    assert float(d.hours_at_or_below(np.array([-1.0, 3.0], np.float32)[:, None, None], 0.0)[0, 0]) == pytest.approx(0.25)
    # freeze, thaw, freeze, thaw: three crossings; the last one is the boundary at exactly 32 (<= counts as frozen)
    zig = np.array([30, 34, 30, 34, 34], np.float32)[:, None, None]
    assert float(d.threshold_crossings(zig)[0, 0]) == 3.0
    # a missing hour costs two intervals: the total is rescaled to the full period (approximate by design)
    gap = ramp.copy()
    gap[5] = np.nan
    assert float(d.hours_at_or_below(gap)[0, 0]) == pytest.approx(12.0, abs=1.2)
    assert np.isnan(d.hours_at_or_below(np.full((5, 1, 1), np.nan, np.float32))[0, 0])


def test_min_wind_chill_matches_hourly_minimum():
    t = np.array([[[30.0]], [[10.0]], [[-5.0]], [[60.0]]], np.float32)
    v = np.array([[[10.0]], [[2.0]], [[25.0]], [[40.0]]], np.float32)
    expect = min(float(d.wind_chill(30, 10)), 10.0, float(d.wind_chill(-5, 25)), 60.0)       # calm hour: air temperature
    assert float(d.min_wind_chill(t, v)[0, 0]) == pytest.approx(expect, abs=1e-3)
    assert d.min_wind_chill(t, v).dtype == np.float32


def test_winter_variables_are_registered():
    for key, req, series in (("slt", ("temp", "terrain"), 0), ("rhi", ("temp", "dewpoint"), 0), ("mixr", ("dewpoint", "pressure"), 0),
                             ("tmax24", ("temp",), 24), ("tmin24", ("temp",), 24), ("trange24", ("temp",), 24),
                             ("frzhrs24", ("temp",), 24), ("ftx24", ("temp",), 24), ("wcmin24", ("temp", "wind"), 24),
                             ("gustmax24", ("gust",), 24), ("frostbite", ("temp", "wind"), 0), ("fltcat", ("vis", "ceiling"), 0)):
        v = VARIABLES[key]
        assert v.requires == req and v.series == series and not v.changes and key not in SOURCE_VARIABLES, key
    assert VARIABLES["fltcat"].categories == d.FLIGHT_CATEGORIES and len(VARIABLES["frostbite"].categories) == 4


# --------------------------------------------------------------------------- precipitation sums
def test_series_sum_uses_whatever_hours_exist():
    from rtma_viewer.derived import series_sum
    st = np.array([[[0.1, np.nan, np.nan]], [[0.2, 0.5, np.nan]], [[np.nan, 0.5, np.nan]]], np.float32)
    out = series_sum(st)
    assert np.isclose(out[0, 0], 0.3) and np.isclose(out[0, 1], 1.0)       # missing hours are skipped, the rest add up
    assert np.isnan(out[0, 2])                                              # no data in any hour: still missing, not 0
    assert np.nansum(np.full((3, 1, 1), np.nan)) == 0 and np.isnan(series_sum(np.full((3, 1, 1), np.nan, np.float32))[0, 0])


def _stack(n, value):
    return np.full((n + 1, 1, 3), value, np.float64)


def test_frozen_precipitation_counts_only_hours_with_a_freezing_wet_bulb():
    from rtma_viewer.derived import frozen_precipitation, wet_bulb
    n = 4
    pcp = np.full((n + 1, 1, 3), 0.10, np.float32)                       # 0.10 in in every hour (the last plane is unused)
    t, td, p = _stack(n, 40.0), _stack(n, 38.0), _stack(n, 1000.0)
    t[:, 0, 0], td[:, 0, 0] = 25.0, 23.0                                  # column 0: cold all day  -> wet bulb well below 32
    t[:, 0, 1], td[:, 0, 1] = 35.0, 34.0                                  # column 1: wet bulb above 32 all day -> rain
    t[2:, 0, 2], td[2:, 0, 2] = 26.0, 24.0                                # column 2: mild at instants 0 and 1, cold at 2..4
    assert wet_bulb(t[:, 0, 0], td[:, 0, 0], p[:, 0, 0]).max() < 32 < wet_bulb(t[:, 0, 1], td[:, 0, 1], p[:, 0, 1]).min()
    out = frozen_precipitation(pcp, t, td, p)
    assert np.isclose(out[0, 0], 0.40) and np.isclose(out[0, 1], 0.0)
    assert 0.20 - 1e-6 <= out[0, 2] <= 0.30 + 1e-6                        # hours 2, 3 cold; hour 1 straddles; hour 0 rain


def test_frozen_precipitation_skips_missing_data():
    from rtma_viewer.derived import frozen_precipitation
    n = 3
    pcp = np.full((n + 1, 1, 2), 0.1, np.float32)
    t, td, p = _stack(n, 25.0)[:, :, :2], _stack(n, 23.0)[:, :, :2], _stack(n, 1000.0)[:, :, :2]
    pcp[1, 0, 1] = np.nan                                                 # column 1 has no precipitation analysis for hour 1
    out = frozen_precipitation(pcp, t, td, p)
    assert np.isclose(out[0, 0], 0.3) and np.isclose(out[0, 1], 0.2)      # the hour is skipped, the others still count
    # a missing wet-bulb instant: the hours on either side fall back to their other end
    t2, td2, p2 = t.copy(), td.copy(), p.copy()
    t2[1], td2[1], p2[1] = np.nan, np.nan, np.nan
    out2 = frozen_precipitation(np.full((n + 1, 1, 2), 0.1, np.float32), t2, td2, p2)
    assert np.allclose(out2, 0.3)                                         # all three hours still classified (cold at the other end)
    # nothing usable at all in a cell: missing, not zero
    out3 = frozen_precipitation(np.full((n + 1, 1, 2), np.nan, np.float32), t, td, p)
    assert np.isnan(out3).all()
    # liquid equivalent: no snow-to-liquid ratio is applied
    assert np.allclose(frozen_precipitation(np.full((n + 1, 1, 2), 0.1, np.float32), t, td, p), 0.3)

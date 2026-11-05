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

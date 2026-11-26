"""WPC coded surface bulletin parsing / archive, front symbols, and the objective front and pressure-centre detector."""
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from rtma_viewer import fronts, objective, symbols
from rtma_viewer.config import PANEL_VARIABLES, VARIABLES, hours_needed
from rtma_viewer.kinematics import Geometry

DATA = Path(__file__).parent / "data" / "codsus_20261126_12z.txt"
UTC = timezone.utc


@pytest.fixture(scope="module")
def real():
    return fronts.parse(DATA.read_text())


def test_parse_real_bulletin(real):
    assert real.valid == datetime(2026, 11, 26, 12, tzinfo=UTC)
    assert len(real.highs()) == 19 and len(real.lows()) == 14
    assert Counter(f.kind for f in real.fronts) == {"trough": 34, "cold": 8, "stationary": 7, "warm": 5, "occluded": 3}
    h = real.highs()[0]
    assert (h.lat, h.lon, h.pressure) == (44.8, -94.6, 1033.0)            # "1033 4480946": 44.8 N, 94.6 W
    w = next(f for f in real.fronts if f.kind == "warm")
    assert w.points[0] == (35.3, -77.7) and len(w.points) == 5


def test_parse_wrapped_lists_and_strength_words():
    text = ("ASUS02 KWBC 010300\nCODSUS\nCODED SURFACE FRONTAL POSITIONS\n1000 PM EST TUE DEC 31 2024\nVALID 010103Z\n"
            "HIGHS 1030 4000800 1022 3500900\nLOWS 995 4500700\nCOLD WK 4000800 4100810\n 4200820 4300830\nSTNRY 3000700 3100710 $$\n")
    an = fronts.parse(text)
    assert an.valid == datetime(2025, 1, 1, 3, tzinfo=UTC)               # issued 31 Dec, valid 1 Jan: the year rolls over
    assert [(c.kind, c.pressure) for c in an.centers] == [("H", 1030.0), ("H", 1022.0), ("L", 995.0)]
    cold = an.fronts[0]
    assert cold.kind == "cold" and len(cold.points) == 4 and cold.points[-1] == (43.0, -83.0)
    assert an.fronts[1].kind == "stationary"


def test_parse_needs_valid_line():
    with pytest.raises(ValueError):
        fronts.parse("nothing here")


def test_archive_saves_each_bulletin_and_finds_nearest(tmp_path):
    arch = fronts.Archive(tmp_path)
    assert arch.update(lambda: None) is None
    an = arch.update(lambda: DATA.read_text())
    assert an.valid == datetime(2026, 11, 26, 12, tzinfo=UTC)
    assert (tmp_path / "2026112612.txt").exists()
    t = an.valid
    assert arch.nearest(t + timedelta(minutes=80)).valid == t               # within 90 minutes
    assert arch.nearest(t + timedelta(hours=3)) is None                      # a full synoptic interval later: not current
    assert arch.nearest(t + timedelta(hours=3), max_back=fronts.MAX_BACK).valid == t     # ... unless falling back to the last one
    assert arch.nearest(t - timedelta(hours=1), max_back=fronts.MAX_BACK) is not None    # nearest also wins over "older"
    assert arch.nearest(t + timedelta(hours=5), max_back=fronts.MAX_BACK) is None
    assert fronts.Archive(tmp_path / "empty").nearest(t) is None


def test_archive_survives_junk(tmp_path):
    assert fronts.Archive(tmp_path).update(lambda: "VALID but no date") is None


# --------------------------------------------------------------------------------------------------------------- symbols
def _flat(lon, lat):
    """A trivial 'projection': 1 degree = 100 km."""
    return np.column_stack([np.asarray(lon, float) * 100e3, np.asarray(lat, float) * 100e3, np.zeros(np.size(lon))])


def test_catmull_rom_passes_through_the_points():
    pts = np.array([[0, 0], [100e3, 50e3], [200e3, 0], [300e3, 80e3]], float)
    d = symbols.catmull_rom(pts)
    for p in pts:
        assert np.min(np.hypot(*(d - p).T)) < 1.0
    assert np.all(np.hypot(*np.diff(d, axis=0).T) < 10e3)


def test_symbols_go_on_the_warm_side_for_a_cold_front():
    # an east-west line running west to east: left of travel = north.  Warmer air to the south => triangles point south (to the right)
    xy_front = fronts.Front("cold", ((40.0, 10.0), (40.0, 11.0), (40.0, 12.0), (40.0, 13.0), (40.0, 14.0)))
    theta = lambda xy: np.where(np.asarray(xy)[:, 1] < 4.0e6, 300.0, 280.0)          # warm south of y = 40 deg (4000 km)
    line, polys = symbols.front_symbols(xy_front, _flat, theta)
    assert len(polys) >= 5
    for poly, colour in polys:
        assert colour == symbols.COLORS["cold"]
        assert poly[2, 1] < poly[:2, 1].mean()                               # apex south of the base
    colder_south = lambda xy: np.where(np.asarray(xy)[:, 1] < 4.0e6, 280.0, 300.0)
    _, polys = symbols.front_symbols(xy_front, _flat, colder_south)
    assert all(poly[2, 1] > poly[:2, 1].mean() for poly, _ in polys)


def test_warm_front_semicircles_on_the_cold_side_and_alternation():
    warm = fronts.Front("warm", ((40.0, 10.0), (40.0, 11.0), (40.0, 12.0), (40.0, 13.0)))
    theta = lambda xy: np.where(np.asarray(xy)[:, 1] < 4.0e6, 300.0, 280.0)
    _, polys = symbols.front_symbols(warm, _flat, theta)
    assert polys and all(c == symbols.COLORS["warm"] for _, c in polys)
    assert all(np.all(p[:, 1] >= 4.0e6 - 1) for p, _ in polys)                # all on the north (cold) side
    occ = fronts.Front("occluded", warm.points)
    _, polys = symbols.front_symbols(occ, _flat, theta)
    assert len({len(p) for p, _ in polys}) == 2                              # triangles and semicircles alternate
    _, polys = symbols.front_symbols(fronts.Front("trough", warm.points), _flat, theta)
    assert polys == []


def test_draw_analysis_runs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    an = fronts.Analysis(datetime(2026, 1, 1, tzinfo=UTC), "WPC", (fronts.Center("L", 40.5, 12.0, 996.0), fronts.Center("H", 90.0, 12.0, 1030.0)),
                         (fronts.Front("cold", ((40.0, 10.0), (40.0, 14.0))), fronts.Front("trough", ((41.0, 10.0), (41.0, 14.0)))))
    fig, ax = plt.subplots()
    symbols.draw_analysis(ax, an, _flat, (9e5, 15e5, 3.5e6, 4.5e6), None, "x")
    texts = [t.get_text() for t in ax.texts] + [t.get_text() for t in ax.get_children() if hasattr(t, "get_text")]
    assert "L" in texts and "996" in texts and "H" not in texts               # the high at 90 N is off the map
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------- objective
def _grid(n=120, cell=2500.0):
    y, x = np.mgrid[0:n, 0:n] * cell
    lat = 40.0 + y / 111e3
    lon = -75.0 + x / (111e3 * np.cos(np.radians(42.0)))
    return x, y, lat, lon, Geometry.from_latlon(lat, lon)


def test_equivalent_potential_temperature_physics():
    th = objective.equivalent_potential_temperature
    cold_dry, warm_moist = th(10.0, 0.0, 1000.0), th(70.0, 65.0, 1000.0)
    assert cold_dry < 270 and 325 < warm_moist < 340
    assert th(50.0, 50.0, 1000.0) > th(50.0, 30.0, 1000.0)                    # moisture raises it
    assert th(50.0, 40.0, 850.0) > th(50.0, 40.0, 1000.0)                      # same surface T at a lower pressure is warmer air (hence the terrain mask)


def test_front_detection_on_a_synthetic_cold_front():
    x, y, lat, lon, geom = _grid()
    tau = 285.0 + 12.0 * np.tanh((y - 150e3) / 20e3)                          # warm (south ... north?) : higher tau to the north of y = 150 km
    u = np.zeros_like(x)
    v = np.full_like(x, 6.0)                                                  # wind toward +y = toward the higher tau: cold front
    ff = objective.front_field(tau, u, v, geom)
    found = objective.find_fronts(ff, lat, lon, np.zeros_like(x), g_min=5.0, min_km=100.0)
    assert found and all(f.kind == "cold" for f in found)
    lats = np.concatenate([[p[0] for p in f.points] for f in found])
    assert np.all(np.abs(lats - (40.0 + 150e3 / 111e3)) < 0.1)                # on the gradient maximum, within ~10 km
    v = np.full_like(x, -6.0)
    warm = objective.find_fronts(objective.front_field(tau, u, v, geom), lat, lon, np.zeros_like(x), g_min=5.0, min_km=100.0)
    assert warm and all(f.kind == "warm" for f in warm)
    still = objective.find_fronts(objective.front_field(tau, u, np.zeros_like(x), geom), lat, lon, np.zeros_like(x), g_min=5.0, min_km=100.0)
    assert still and all(f.kind == "stationary" for f in still)


def test_weak_gradients_and_high_terrain_make_no_fronts():
    x, y, lat, lon, geom = _grid()
    weak = 285.0 + 1.0 * np.tanh((y - 150e3) / 20e3)
    ff = objective.front_field(weak, np.zeros_like(x), np.zeros_like(x), geom)
    assert objective.find_fronts(ff, lat, lon, np.zeros_like(x), g_min=10.0, min_km=100.0) == []
    strong = 285.0 + 12.0 * np.tanh((y - 150e3) / 20e3)
    ff = objective.front_field(strong, np.zeros_like(x), np.zeros_like(x), geom)
    assert objective.find_fronts(ff, lat, lon, np.full_like(x, 6000.0), g_min=5.0, min_km=100.0) == []


def test_coastal_fronts_are_dropped():
    import shapely
    f_near = fronts.Front("cold", tuple((40.0, 10.0 + i * 0.2) for i in range(8)))
    f_far = fronts.Front("cold", tuple((45.0, 10.0 + i * 0.2) for i in range(8)))
    coast = shapely.LineString([(9e5, 4.0e6 + 10e3), (13e5, 4.0e6 + 10e3)])
    barrier = objective.coastal_barrier([coast], [])
    kept = objective.drop_coastal([f_near, f_far], _flat, barrier)
    assert kept == [f_far]
    assert objective.drop_coastal([f_near], _flat, None) == [f_near]


def test_pressure_centres():
    x, y, lat, lon, geom = _grid(160)
    cx, cy = 200e3, 200e3
    r2 = ((x - cx) ** 2 + (y - cy) ** 2) / (80e3) ** 2
    mslp = 1015.0 - 20.0 * np.exp(-r2)                                       # one low of 995 mb
    found = objective.find_centers(mslp, np.zeros_like(x), lat, lon, geom)
    assert [(c.kind, c.pressure) for c in found] == [("L", 995.0)]
    assert abs(found[0].lat - (40.0 + cy / 111e3)) < 0.1
    high = objective.find_centers(2030.0 - mslp, np.zeros_like(x), lat, lon, geom)
    assert [c.kind for c in high] == ["H"]
    assert objective.find_centers(np.full_like(x, 1013.0), np.zeros_like(x), lat, lon, geom) == []
    assert objective.find_centers(mslp, np.full_like(x, 6000.0), lat, lon, geom) == []     # over mountains: not meaningful


# -------------------------------------------------------------------------------------------------------------- wiring
def test_surface_analysis_variables_are_wired():
    for key, overlay in (("sfcwpc", "wpc"), ("sfcobj", "objective")):
        var = PANEL_VARIABLES[key]
        assert var.overlay == overlay and not var.changes
        assert var.labeler(1013.0) == ""                                     # no station values over the fronts map
    need = hours_needed()
    for key in ("u10", "v10", "temp", "dewpoint", "pressure", "terrain"):
        assert 0 in need[key]
    assert VARIABLES["sfcwpc"].group == VARIABLES["sfcobj"].group

import numpy as np

from rtma_viewer import colors as c
from rtma_viewer.config import INTERVALS, VARIABLES


def lightness(style):
    return c.srgb_to_oklab(style.colors)[:, 0]


def test_sequential_ramps_are_monotone_in_lightness():
    for key in ("dewpoint",):
        L = lightness(c.style_for(key, 0))
        assert np.all(np.diff(L) <= 2e-3), key


def test_temperature_has_freezing_break():
    s = c.style_for("temp", 0)
    assert 32 in s.levels                                   # a bin edge sits exactly on 32 F
    i = int(np.where(s.levels == 32)[0][0]) + 1             # +1: leading 'under' colour
    below, above = s.colors[i - 1], s.colors[i]
    assert above[1] > above[2] and below[2] >= below[1]     # greener above 32, bluer below


def test_change_ramp_is_symmetric_with_white_centre():
    for var in VARIABLES:
        for h in INTERVALS[1:]:
            s = c.style_for(var, h)
            assert np.isclose(s.levels[0], -s.levels[-1])
            mid = len(s.colors) // 2
            assert s.levels[len(s.levels) // 2 - 1] < 0 < s.levels[len(s.levels) // 2]   # zero inside the centre bin
            assert np.ptp(s.colors[mid]) < 0.02                                         # ~neutral grey/white
            cmap, norm = s.cmap_norm()                                                   # matplotlib accepts it
            assert cmap.N == len(s.levels) - 1                                           # bins
            assert np.allclose(cmap.get_under()[:3], s.colors[0]) and np.allclose(cmap.get_over()[:3], s.colors[-1])


def test_levels_are_fixed():
    # identical call -> identical levels (colour meaning never changes between runs)
    assert np.array_equal(c.style_for("wind", 0).levels, c.style_for("gust", 0).levels)


def test_change_ramp_matches_wpc_levels():
    s = c.style_for("temp", 24)
    assert list(s.levels) == [-50, -40, -35, -30, -25, -20, -16, -12, -8, -5, -3, -1,
                              1, 3, 5, 8, 12, 16, 20, 25, 30, 35, 40, 50]
    assert len(s.colors) == len(s.levels) + 1          # 23 bins + under + over
    assert np.allclose(s.colors[12], (1, 1, 1))        # -1..+1 is white
    # warming is warm-coloured, cooling is blue
    assert s.colors[13][0] > s.colors[13][2] and s.colors[10][2] > s.colors[10][0]
    # whole numbers, and identical for every interval and variable (no per-interval scaling)
    assert all(float(v).is_integer() for v in s.levels)
    for var in ("temp", "dewpoint", "wind", "gust", "apparent"):
        for h in (1, 3, 6, 12, 24):
            t = c.style_for(var, h)
            assert np.array_equal(t.levels, s.levels) and np.allclose(t.colors, s.colors)


def hexes(style):
    return ["#%02x%02x%02x" % tuple(np.round(c * 255).astype(int)) for c in style.colors]


def test_wind_and_gust_use_the_nws_gust_legend():
    s = c.style_for("gust", 0)
    assert list(s.levels) == [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 120, 140, 150]
    assert all(float(v).is_integer() for v in s.levels)
    h = hexes(s)
    assert h[0] == "#103f78" and h[10] == "#ffaa00" and h[16] == "#ff73df" and h[17] == h[16]
    assert len(h) == len(s.levels)                       # 17 bins + over
    assert np.array_equal(s.levels, c.style_for("wind", 0).levels)
    assert np.allclose(s.colors, c.style_for("wind", 0).colors)


def test_apparent_ramp_joins_wind_chill_and_heat_index_legends():
    s = c.style_for("apparent", 0)
    assert s.levels[0] == -60 and s.levels[-1] == 120 and np.all(np.diff(s.levels) == 5)
    assert len(s.colors) == len(s.levels) + 1             # 36 bins + under + over
    h = hexes(s)
    at = lambda lo: h[int((lo + 60) // 5) + 1]            # colour of the bin starting at `lo`
    assert h[0] == "#a51453" and at(-60) == "#e2266a" and at(-35) == "#ffffff"      # wind chill legend
    assert at(0) == "#2151b0" and at(30) == "#44e3d6" and at(35) == "#14adaa"
    assert at(75) == "#ffffb4" and at(85) == "#ffc23e" and at(115) == "#6d1438" and h[-1] == "#3c143c"   # heat index legend
    # the 35-75 bridge is smooth: neighbouring bins differ by a small OKLab step
    lab = c.srgb_to_oklab(s.colors[np.arange(len(s.colors))])
    bridge = lab[(35 + 60) // 5 + 1:(75 + 60) // 5 + 2]
    assert np.all(np.linalg.norm(np.diff(bridge, axis=0), axis=1) < 0.09)

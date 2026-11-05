import numpy as np

from rtma_viewer import colors as c
from rtma_viewer.config import INTERVALS, VARIABLES


def lightness(style):
    return c.srgb_to_oklab(style.colors)[:, 0]


def test_sequential_ramps_are_monotone_in_lightness():
    for key in ("dewpoint", "wind", "gust"):
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
    for var in ("temp", "dewpoint", "wind", "gust"):
        for h in (1, 3, 6, 12, 24):
            t = c.style_for(var, h)
            assert np.array_equal(t.levels, s.levels) and np.allclose(t.colors, s.colors)

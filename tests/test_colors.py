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


def test_diverging_is_symmetric_with_neutral_centre():
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

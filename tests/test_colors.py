import numpy as np

from rtma_viewer import colors as c
from rtma_viewer.config import INTERVALS, PANEL_VARIABLES as VARIABLES


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


# ------------------------------------------------------------------------------------------ styles for the added variables
def test_every_style_is_opaque_at_and_beyond_its_ends():
    """Regression: with extend="neither", a value exactly at the top level (RH / cloud cover = 100, wind direction = 360)
    used to map to a transparent pixel."""
    for var in VARIABLES:
        for h in (0, 6):
            s = c.style_for(var, h)
            cmap, norm = s.cmap_norm()
            for v in (s.levels[0], s.levels[-1], s.levels[0] - 1e-3, s.levels[-1] + 1e-3, s.levels[0] - 500, s.levels[-1] + 500):
                assert cmap(norm(np.array([v])))[0][3] == 1.0, (var, h, v)
    for key in ("rh", "cloud", "wdir"):                                           # the exact-top value takes the top colour
        s = c.style_for(key, 0)
        cmap, norm = s.cmap_norm()
        assert np.allclose(cmap(norm(np.array([s.levels[-1]])))[0][:3], s.colors[-1])


def test_every_variable_has_a_style_with_consistent_shape():
    for key in VARIABLES:
        s = c.style_for(key, 0)
        n_bins = len(s.levels) - 1
        extra = {"neither": 0, "min": 1, "max": 1, "both": 2}[s.extend]
        assert len(s.colors) == n_bins + extra, key
        assert np.all(np.diff(s.levels) > 0), key
        assert np.all((s.colors >= 0) & (s.colors <= 1)), key
        cmap, _ = s.cmap_norm()                                                    # matplotlib accepts it
        assert cmap.N == n_bins


def test_wet_bulb_shares_the_temperature_legend():
    a, b = c.style_for("temp", 0), c.style_for("wetbulb", 0)
    assert np.array_equal(a.levels, b.levels) and np.allclose(a.colors, b.colors) and 32 in b.levels


def test_pressure_ramps():
    L = lightness(c.style_for("pressure", 0))
    assert np.all(np.diff(L) >= -2e-3)                                            # sequential: monotone lightness
    s = c.style_for("mslp", 0)
    assert s.levels[0] == 940 and s.levels[-1] == 1060 and 1012 in s.levels and 1014 in s.levels      # 2 mb bins about ~1013


def test_pressure_changes_use_finer_levels_with_the_same_shape():
    for key in ("pressure", "mslp"):
        s = c.style_for(key, 3)
        assert list(s.levels) == [-30, -24, -20, -16, -12, -9, -6, -4, -3, -2, -1, -0.5,
                                  0.5, 1, 2, 3, 4, 6, 9, 12, 16, 20, 24, 30]
        assert np.allclose(s.colors, c.style_for("temp", 3).colors)                # same 25 colours, white centre
        assert np.array_equal(s.levels, c.style_for(key, 24).levels)               # still identical across intervals


def test_wind_direction_style_is_a_cyclic_16_point_compass():
    s = c.style_for("wdir", 0)
    assert s.levels[0] == 0 and s.levels[-1] == 360 and len(s.colors) == 17
    assert np.allclose(s.colors[0], s.colors[-1])                                  # north at both ends of the bar
    assert not s.uniform and s.tick_labels[0] == "N" and s.tick_labels[90] == "E" and s.tick_labels[270] == "W"
    cmap, norm = s.cmap_norm()
    col = lambda deg: tuple(np.round(cmap(norm(np.array([deg])))[0][:3], 3))
    assert col(0.0) == col(5.0) == col(355.0) == col(360.0)                        # 348.75..11.25 is all "N"
    assert col(90.0) != col(180.0) != col(270.0)


def test_visibility_and_ceiling_use_aviation_style_breaks():
    v, ce = c.style_for("vis", 0), c.style_for("ceiling", 0)
    assert list(v.levels) == [0, 0.25, 0.5, 1, 2, 3, 5, 7, 9.5] and v.extend == "max"
    assert list(ce.levels) == [0, 200, 500, 1000, 2000, 3000, 5000, 8000, 12000]
    assert 500 in ce.levels and 1000 in ce.levels and 3000 in ce.levels          # LIFR / IFR / MVFR boundaries are bin edges


def test_terrain_style_has_a_sea_level_bin():
    s = c.style_for("terrain", 0)
    assert s.levels[0] == 0 and s.levels[1] == 20 and s.levels[-1] == 6000
    assert s.colors[0][2] > s.colors[0][0]                                         # sea: bluish, not the lowland green

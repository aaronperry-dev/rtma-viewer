"""Render helpers that do not need the map layers: station-label text, smoothing, circular statistics."""
from types import SimpleNamespace

import numpy as np
import pytest

from rtma_viewer import render
from rtma_viewer.config import VARIABLES


def test_label_text_defaults_to_whole_numbers():
    assert render.label_text(None, 41.6, False) == "42"
    assert render.label_text(None, -0.4, False) == "0" and render.label_text(None, 3.0, True) == "+3"
    assert render.label_text(VARIABLES["temp"], -7.5, False) in ("-8", "-7")             # banker's rounding either way, whole number
    assert render.label_text(VARIABLES["temp"], 0.2, True) == "0"


def test_label_text_per_variable_decimals():
    assert render.label_text(VARIABLES["vis"], 9.94, False) == "9.9"                    # visibility: tenths of a mile
    assert render.label_text(VARIABLES["mslp"], 1013.4, False) == "1013"                # absolute pressure: whole mb ...
    assert render.label_text(VARIABLES["mslp"], 1.24, True) == "+1.2"                   # ... tendency: tenths
    assert render.label_text(VARIABLES["mslp"], -3.06, True) == "-3.1"
    assert render.label_text(VARIABLES["mslp"], 0.02, True) == "0.0"
    assert render.label_text(VARIABLES["pressure"], -0.5, True) == "-0.5"
    assert render.label_text(VARIABLES["wdir"], 269.6, False) == "270"


def test_ceiling_labels():
    f = VARIABLES["ceiling"]
    assert render.label_text(f, 12000.0, False) == "CLR"
    assert render.label_text(f, 849.0, False) == "800" and render.label_text(f, 2549.0, False) == "2500"
    assert render.label_text(f, 12.0, False) == "0"


def test_smooth_preserves_constants_ignores_nan_and_damps_noise():
    flat = np.full((40, 50), 1013.0)
    assert np.allclose(render.smooth(flat, 5.0), 1013.0)
    a = flat.copy()
    a[10:14, 10:14] = np.nan
    s = render.smooth(a, 5.0)
    assert np.isnan(s[11, 11]) and np.allclose(s[np.isfinite(s)], 1013.0)              # NaN stays NaN, neighbours unbiased
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((80, 90))
    assert render.smooth(noise, 5.0).std() < 0.15 * noise.std()
    ramp = np.tile(np.linspace(1000, 1020, 90), (80, 1))
    assert np.allclose(render.smooth(ramp, 5.0)[:, 20:70], ramp[:, 20:70], atol=1e-3)  # a linear gradient survives smoothing


def test_region_stats_normal_and_circular():
    land = np.ones((4, 4), bool)
    ctx = SimpleNamespace(land=land)
    f = np.arange(16, dtype=float).reshape(4, 4)
    assert render.region_stats(ctx, f) == {"min": 0.0, "mean": 7.5, "max": 15.0}
    ang = np.array([[350.0, 10.0], [350.0, 10.0]])                                      # straddles north: arithmetic mean would be 180
    st = render.region_stats(SimpleNamespace(land=np.ones((2, 2), bool)), ang, circular=True)
    assert st["min"] is None and st["max"] is None and (st["mean"] % 360 < 0.5 or st["mean"] % 360 > 359.5)
    south = np.full((2, 2), 180.0)
    assert render.region_stats(SimpleNamespace(land=np.ones((2, 2), bool)), south, circular=True)["mean"] == pytest.approx(180.0)
    assert render.region_stats(SimpleNamespace(land=np.zeros((2, 2), bool)), south) == {"min": None, "mean": None, "max": None}

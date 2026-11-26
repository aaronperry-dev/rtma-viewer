"""Surface kinematics: grid geometry, the wind-component convention of the real RTMA, and the diagnostics themselves
(checked against analytic fields and against MetPy)."""
import json
from pathlib import Path

import numpy as np
import pytest

from rtma_viewer import kinematics as kin
from rtma_viewer.config import VARIABLES
from rtma_viewer.testing import DemoGrid

DX = 2539.703


def cartesian(ny=60, nx=70, dx=DX):
    """Uniform-spacing geometry plus metre coordinates (x east-ish, y north-ish), origin at the centre."""
    geom = kin.Geometry(np.full((ny, nx), dx), np.full((ny, nx), dx))
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    return geom, (x - nx / 2) * dx, (y - ny / 2) * dx


# ------------------------------------------------------------------------------------------------- geometry / convention
@pytest.fixture(scope="module")
def demo():
    return DemoGrid(nx=240, ny=200)


def test_geometry_spacing_matches_the_projection_scale_factor(demo):
    """Ground distance per cell = grid spacing / map-scale factor (pyproj's, independent of our lat/lon differences)."""
    g = kin.Geometry.from_latlon(demo.lat, demo.lon)
    for i, j in [(10, 10), (100, 120), (190, 230)]:
        k = demo.proj.get_factors(demo.lon[i, j], demo.lat[i, j]).parallel_scale
        assert g.dx[i, j] == pytest.approx(DX / k, rel=2e-3), (i, j)
        assert g.dy[i, j] == pytest.approx(DX / k, rel=2e-3)
    assert g.dx.shape == demo.lat.shape


def test_demo_rotation_is_the_lambert_convergence_angle(demo):
    """East of the central meridian the grid's x axis points a little clockwise of east: -n * (lon - lon0), n = sin(25 deg)."""
    u, v = demo.to_grid_relative(np.ones_like(demo.lat), np.zeros_like(demo.lat))        # an east wind, in grid components
    angle = np.degrees(np.arctan2(-v, u))                                                # = angle of +x ccw from east
    expect = -np.sin(np.radians(25.0)) * (demo.lon + 95.0)
    assert np.abs(angle - expect).max() < 0.15


SAMPLE = json.loads((Path(__file__).parent / "data" / "rtma_uv_sample.json").read_text())


def test_real_rtma_wind_components_are_grid_relative():
    """60 cells of real RTMA (2026-11-26 02z): the direction from (UGRD, VGRD) differs from the published earth-relative
    WDIR by exactly the grid's convergence angle, -n (lon - lon0).  If NOAA ever switched to east/north components this
    would fail - and every kinematic map would silently be rotated ~10 degrees."""
    lat, lon, u, v, wdir = np.array(SAMPLE["rows"]).T
    d_uv = np.degrees(np.arctan2(-u, -v)) % 360.0
    diff = (d_uv - wdir + 180.0) % 360.0 - 180.0
    pred = -np.sin(np.radians(25.0)) * (lon + 95.0)
    assert np.abs(diff - pred).max() < 0.5
    assert abs(diff.mean()) > 8.0                              # i.e. clearly NOT earth-relative here (about -10.6 deg)


# ------------------------------------------------------------------------------------------------- analytic checks
def test_solid_body_rotation_and_pure_divergence():
    geom, x, y = cartesian()
    omega, div = 5e-5, 3e-5
    u, v = -omega * y, omega * x                                       # solid-body rotation: vorticity 2 omega, no divergence
    assert np.allclose(kin.vorticity(u, v, geom, sigma=0.01), 2 * omega * 1e5, atol=1e-3)
    assert np.allclose(kin.convergence(u, v, geom, sigma=0.01), 0.0, atol=1e-3)
    u, v = 0.5 * div * x, 0.5 * div * y                                # uniform expansion: divergence = div, no vorticity
    assert np.allclose(kin.convergence(u, v, geom, sigma=0.01), -div * 1e5, atol=1e-3)     # (convergence is minus divergence)
    assert np.allclose(kin.vorticity(u, v, geom, sigma=0.01), 0.0, atol=1e-3)


def test_advection_of_a_linear_field_and_signs():
    geom, x, y = cartesian()
    t = 40.0 + 0.002 * x                                               # warmer toward +x: 2 F per km
    east = np.full_like(t, 5.0)
    assert np.allclose(kin.advection(t, east, 0 * east, geom, sigma=0.01), -5.0 * 0.002 * 3600, rtol=1e-4)   # wind from the warm side: cold... -36 F/h
    west = -east                                                       # wind blowing toward -x carries the warm air from +x
    assert np.allclose(kin.advection(t, west, 0 * east, geom, sigma=0.01), +36.0, rtol=1e-4)                # warm advection is positive
    assert kin.advection(t, east, east, geom).dtype == np.float32


def test_terrain_lift_is_wind_dot_slope():
    geom, x, y = cartesian()
    z_ft = 0.05 * x * 3.280839895 + 2000.0                             # 50 m rise per km toward +x
    u = np.full_like(x, 10.0)
    assert np.allclose(kin.terrain_lift(u, 0 * u, z_ft, geom, sigma=0.01), 0.05 * 10 * 100.0, rtol=1e-3)   # 50 cm/s up the slope
    assert np.allclose(kin.terrain_lift(-u, 0 * u, z_ft, geom, sigma=0.01), -50.0, rtol=1e-3)             # downslope: sinking
    assert np.allclose(kin.terrain_lift(0 * u, u, z_ft, geom, sigma=0.01), 0.0, atol=1e-3)                 # along the contours: none


def test_pressure_gradient_units_and_terrain_mask():
    geom, x, y = cartesian(80, 90)
    p = 1010.0 + x / 1e5                                               # 1 mb per 100 km
    z_ft = np.zeros_like(x)
    z_ft[:, :10] = 4000.0                                              # 1,200 m: above the reduction limit
    g = kin.pressure_gradient(p, z_ft, geom, sigma=2.0)
    assert np.allclose(g[10:-10, 30:60], 1.0, atol=1e-3)
    assert np.isnan(g[40, 3])                                          # blanked over the high terrain
    assert g.dtype == np.float32


def test_diagnostics_propagate_missing_data():
    geom, x, y = cartesian(40, 40)
    u = np.full_like(x, 5.0)
    u[10:14, 10:14] = np.nan
    out = kin.convergence(u, u, geom)
    assert np.isnan(out[11, 11]) and np.isfinite(out[30, 30])


# ------------------------------------------------------------------------------------------------- against MetPy
def _metpy_case():
    mc = pytest.importorskip("metpy.calc")
    units = pytest.importorskip("metpy.units").units
    geom, x, y = cartesian(50, 60)
    rng = np.random.default_rng(3)
    smooth = lambda a: kin.smooth(a, 4.0)
    u = smooth(rng.standard_normal(x.shape)) * 20                      # m/s, smooth random fields
    v = smooth(rng.standard_normal(x.shape)) * 20
    theta = 280.0 + 60 * smooth(rng.standard_normal(x.shape)) + 0.0003 * x   # K (so |grad| is never ~0)
    return mc, units, geom, u, v, theta


def test_frontogenesis_matches_metpy():
    mc, units, geom, u, v, theta = _metpy_case()
    ref = mc.frontogenesis(theta * units.K, u * units("m/s"), v * units("m/s"), dx=DX * units.m, dy=DX * units.m)
    ref = ref.to("K / m / s").m * 1.8 * kin.FGEN_UNITS                 # K/m/s -> F/100 km/3 h
    ours = kin.frontogenesis(theta * 1.8, u, v, geom, sigma=0.01)      # (offset between K and F does not matter for gradients)
    inner = (slice(3, -3), slice(3, -3))
    assert np.corrcoef(ours[inner].ravel(), ref[inner].ravel())[0, 1] > 0.999
    assert np.allclose(ours[inner], ref[inner], rtol=2e-2, atol=2e-2 * np.abs(ref[inner]).max())


def test_divergence_vorticity_advection_match_metpy():
    mc, units, geom, u, v, theta = _metpy_case()
    kw = dict(dx=DX * units.m, dy=DX * units.m)
    q = lambda a, un: a * units(un)
    inner = (slice(2, -2), slice(2, -2))
    div = mc.divergence(q(u, "m/s"), q(v, "m/s"), **kw).to("1/s").m * 1e5
    vor = mc.vorticity(q(u, "m/s"), q(v, "m/s"), **kw).to("1/s").m * 1e5
    adv = mc.advection(q(theta, "K"), q(u, "m/s"), q(v, "m/s"), **kw).to("K/s").m * 3600.0 * 1.8
    assert np.allclose(-kin.convergence(u, v, geom, sigma=0.01)[inner], div[inner], rtol=1e-3, atol=1e-3)
    assert np.allclose(kin.vorticity(u, v, geom, sigma=0.01)[inner], vor[inner], rtol=1e-3, atol=1e-3)
    assert np.allclose(kin.advection(theta * 1.8, u, v, geom, sigma=0.01)[inner], adv[inner], rtol=1e-3, atol=1e-3)


# ------------------------------------------------------------------------------------------------- registry
def test_kinematic_variables_are_registered():
    for key in ("tadv", "dadv", "conv", "vort", "fgen", "upslope"):
        v = VARIABLES[key]
        assert v.needs_geom and v.diverging and not v.changes and v.derive is not None, key
        assert {"u10", "v10"} <= set(v.requires) or key in ("tadv", "fgen", "dadv"), key
    assert VARIABLES["tadv"].requires == ("slt", "u10", "v10") and VARIABLES["fgen"].requires == ("slt", "u10", "v10")
    assert VARIABLES["pgrad"].requires == ("mslp", "terrain") and VARIABLES["pgrad"].needs_geom
    assert VARIABLES["u10"].hidden and VARIABLES["u10"].pack == (-65.0, 500.0)

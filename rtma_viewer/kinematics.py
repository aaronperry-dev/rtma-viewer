"""Grid geometry and surface kinematics (advection, divergence, vorticity, frontogenesis, terrain lift, gradients).

Conventions
-----------
* Everything works in *grid* coordinates: x = increasing column, y = increasing row.  NCEP's RTMA / URMA publish the 10 m
  wind components relative to that grid (GRIB2 ``uvRelativeToGrid = 1``), not east / north - on this Lambert-conformal grid
  the two differ by ~10 degrees over New England - so ``UGRD`` / ``VGRD`` can be differentiated directly.  (The test-suite
  pins this against ``WDIR`` from real data.)
* Grid spacing is the true ground distance between neighbouring cells, taken from the cells' latitude / longitude, so the
  map-scale variation of the projection (~1 % across New England) is included.
* Derivatives are centred finite differences (one-sided at the edge) of a NaN-aware Gaussian-smoothed field.  At 2.5 km the
  analysed 10 m wind and 2 m temperature carry terrain-scale and analysis noise that finite differences amplify, so every
  diagnostic smooths first (``SIGMA_CELLS``; sigma = 3 cells ~ 7.6 km, an effective resolution of roughly 20 km).
* Display units: advection °F/h, divergence / vorticity 1e-5 s-1, frontogenesis °F / 100 km / 3 h, terrain lift cm/s.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .derived import sea_level_dewpoint

R_EARTH = 6371229.0            # m, the sphere NCEP uses for the NDFD grids
FT_PER_M = 3.280839895
SIGMA_CELLS = 3.0              # smoothing of wind / scalar fields before differencing
FGEN_SIGMA_CELLS = 4.0         # frontogenesis involves products of gradients: smooth a little more
LIFT_SIGMA_CELLS = 1.0         # terrain and wind are already smooth: just take the grid-scale noise off
GRADIENT_SIGMA_CELLS = 25.0    # MSLP gradient: ~60 km.  The 2.5 km sea-level reduction is noisy (10 m of terrain error is ~1 mb),
                               # so its raw gradient is mostly noise: on a day with a broad 0.7 mb / 100 km gradient, sigma = 5 cells
                               # reads 1.6, sigma = 12 reads 1.1, sigma = 25 reads ~0.85 (measured on live RTMA)
FGEN_UNITS = 1e5 * 3 * 3600.0  # (unit / m / s) -> (unit / 100 km / 3 h)


def smooth(a: np.ndarray, sigma: float) -> np.ndarray:
    """NaN-aware separable Gaussian smoothing (sigma in grid cells), no SciPy needed."""
    r = max(1, int(round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    w = np.isfinite(a).astype(np.float64)
    v = np.where(w > 0, a, 0.0).astype(np.float64)

    def conv(m, axis):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        m = np.pad(m, pad, mode="edge")
        return np.apply_along_axis(lambda x: np.convolve(x, k, mode="valid"), axis, m)

    num, den = conv(conv(v, 0), 1), conv(conv(w, 0), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(w > 0, out, np.nan)


@dataclass(frozen=True)
class Geometry:
    """Ground distance (m) between neighbouring cells along the grid's x (columns) and y (rows) axes."""
    dx: np.ndarray
    dy: np.ndarray

    @classmethod
    def from_latlon(cls, lat: np.ndarray, lon: np.ndarray) -> "Geometry":
        phi, lam = np.radians(np.asarray(lat, np.float64)), np.radians(np.asarray(lon, np.float64))
        cos_phi = np.cos(phi)

        def step(axis):          # centred difference: half the distance between cell i-1 and i+1
            return R_EARTH * np.hypot(np.gradient(phi, axis=axis), cos_phi * np.gradient(lam, axis=axis))

        return cls(step(1), step(0))

    # first derivatives per metre
    def ddx(self, a):
        return np.gradient(a, axis=1) / self.dx

    def ddy(self, a):
        return np.gradient(a, axis=0) / self.dy


# ---------------------------------------------------------------------------------------------------- diagnostics
def advection(scalar, u_ms, v_ms, geom: Geometry, sigma: float = SIGMA_CELLS):
    """-V . grad(scalar), in (scalar unit) per hour.  Positive = the wind carries higher values of the scalar in."""
    s, u, v = smooth(scalar, sigma), smooth(u_ms, sigma), smooth(v_ms, sigma)
    return (-(u * geom.ddx(s) + v * geom.ddy(s)) * 3600.0).astype(np.float32)


def convergence(u_ms, v_ms, geom: Geometry, sigma: float = SIGMA_CELLS):
    """Convergence of the horizontal wind, -(du/dx + dv/dy), in 1e-5 s-1 (positive = air piling up)."""
    u, v = smooth(u_ms, sigma), smooth(v_ms, sigma)
    return (-(geom.ddx(u) + geom.ddy(v)) * 1e5).astype(np.float32)


def vorticity(u_ms, v_ms, geom: Geometry, sigma: float = SIGMA_CELLS):
    """Relative vorticity dv/dx - du/dy in 1e-5 s-1 (positive = cyclonic: counter-clockwise in the Northern Hemisphere)."""
    u, v = smooth(u_ms, sigma), smooth(v_ms, sigma)
    return ((geom.ddx(v) - geom.ddy(u)) * 1e5).astype(np.float32)


def frontogenesis(scalar, u_ms, v_ms, geom: Geometry, sigma: float = FGEN_SIGMA_CELLS):
    """Petterssen 2-D kinematic frontogenesis of ``scalar`` (°F), in °F per 100 km per 3 h.

        F = d|grad T|/dt = -(1/|grad T|) [ Tx^2 ux + Ty^2 vy + Tx Ty (vx + uy) ]

    Positive where convergence and deformation in the wind are strengthening the temperature gradient.  Only the horizontal
    kinematic terms (no vertical motion, no diabatic heating), as in the standard surface-analysis diagnostic; it is the
    same quantity as MetPy's ``frontogenesis`` (checked in the tests).
    """
    t, u, v = smooth(scalar, sigma), smooth(u_ms, sigma), smooth(v_ms, sigma)
    tx, ty = geom.ddx(t), geom.ddy(t)
    ux, uy, vx, vy = geom.ddx(u), geom.ddy(u), geom.ddx(v), geom.ddy(v)
    mag = np.hypot(tx, ty)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = -(tx * tx * ux + ty * ty * vy + tx * ty * (vx + uy)) / mag
    f = np.where(mag > 1e-7, f, 0.0)                    # < 0.1 °F per 1000 km: no gradient to sharpen
    return np.where(np.isfinite(t) & np.isfinite(f), f * FGEN_UNITS, np.nan).astype(np.float32)


def terrain_lift(u_ms, v_ms, terrain_ft, geom: Geometry, sigma: float = LIFT_SIGMA_CELLS):
    """Vertical motion forced by the terrain, w = V . grad(z), in cm/s (positive = wind blowing uphill)."""
    z = np.asarray(terrain_ft, np.float64) / FT_PER_M
    u, v, zs = smooth(u_ms, sigma), smooth(v_ms, sigma), smooth(z, sigma)
    return ((u * geom.ddx(zs) + v * geom.ddy(zs)) * 100.0).astype(np.float32)


def gradient_magnitude(field, geom: Geometry, sigma: float = GRADIENT_SIGMA_CELLS):
    """|grad field| in (field unit) per 100 km."""
    s = smooth(field, sigma)
    return (np.hypot(geom.ddx(s), geom.ddy(s)) * 1e5).astype(np.float32)


def dewpoint_advection(td_f, terrain_ft, u_ms, v_ms, geom: Geometry, sigma: float = SIGMA_CELLS):
    """Advection of the sea-level-adjusted dew point (°F/h): positive = moister air arriving, negative = drying."""
    return advection(sea_level_dewpoint(td_f, terrain_ft), u_ms, v_ms, geom, sigma)


PRESSURE_GRADIENT_MAX_TERRAIN_M = 1000.0      # the sea-level reduction is not meaningful above this


def pressure_gradient(mslp_mb, terrain_ft, geom: Geometry, sigma: float = GRADIENT_SIGMA_CELLS,
                      max_terrain_m: float = PRESSURE_GRADIENT_MAX_TERRAIN_M):
    """|grad MSLP| in mb per 100 km of an MSLP field smoothed over ~60 km, left blank over high terrain.  At 45 N,
    1 mb / 100 km is about a 17 mph geostrophic wind (surface winds are weaker, friction)."""
    m = np.where(np.asarray(terrain_ft, np.float64) > max_terrain_m * FT_PER_M, np.nan, mslp_mb)
    return gradient_magnitude(m, geom, sigma)

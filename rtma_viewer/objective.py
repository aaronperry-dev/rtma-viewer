"""Objective surface fronts and pressure centres from the analysed fields (an experimental, machine-drawn companion to WPC's analysis).

Fronts - after Hewson (1998) "Objective fronts" (Meteorol. Appl. 5, 37-65), applied to the surface analysis:

1. thermal field tau = surface equivalent potential temperature (Bolton 1980), smoothed; unlike temperature it is nearly independent
   of elevation in a mixed boundary layer, so terrain does not draw fronts, and it follows the air mass (moisture counts);
2. G = |grad tau|; the front line is where G is a maximum along the direction of grad tau: the zero contour of
   F = grad(G) . grad(tau) / |grad tau|, kept where F is falling through zero (a maximum, not a minimum) and G is big enough;
3. each line is split into pieces by the wind component across it, v_n (positive = wind blowing toward the warm side):
   cold front (cold air advancing, v_n > 0), warm front (v_n < 0), stationary (|v_n| small).

It has no memory of previous analyses and sees only the surface, so it finds frontal zones, not analysts' fronts: occlusions,
troughs and dry lines are not identified, strong land-sea or lake contrasts can show as fronts, and a front may appear in
fragments.  Pressure centres are closed minima / maxima of smoothed MSLP with enough depth, away from high terrain.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import shapely
from contourpy import contour_generator

from .derived import sat_vapor_pressure
from .fronts import Analysis, Center, Front
from .kinematics import FT_PER_M, Geometry, smooth

# ---- tuning (see tests / README): all lengths in grid cells (2.5 km) or km
SIGMA_TAU = 8.0                 # smoothing of the thermal field: ~20 km, an effective resolution of about 60 km
SIGMA_G = 5.0                   # smoothing of the gradient magnitude before its own gradient is taken
SIGMA_WIND = 8.0
G_MIN_K_PER_100KM = 10.0        # weaker thermal gradients are not fronts
MIN_FRONT_KM = 300.0            # shorter pieces are dropped
MIN_PIECE_KM = 100.0            # a run of one type shorter than this is merged into its neighbour
V_N_STATIONARY = 2.0            # m/s: wind across the front weaker than this -> stationary
MAX_TERRAIN_M = 1000.0          # thermal gradients over high terrain mostly reflect elevation, not air masses
EDGE_CELLS = 8                  # smoothed gradients are unreliable this close to the edge of the data window
COAST_KM = 40.0                 # a front lying mostly this close to a coast or a Great Lake is the land-water contrast, not an air-mass boundary
COAST_FRACTION = 0.5
BIG_LAKE_M2 = 5e9
CENTER_SIGMA = 10.0
CENTER_RADIUS_KM = 150.0
CENTER_DEPTH_MB = 1.5


def equivalent_potential_temperature(t_f, td_f, p_mb):
    """Bolton (1980) equivalent potential temperature (K) from 2 m temperature and dew point (°F) and surface pressure (mb)."""
    t = (np.asarray(t_f, np.float64) - 32.0) / 1.8 + 273.15
    td = np.minimum((np.asarray(td_f, np.float64) - 32.0) / 1.8 + 273.15, t)
    p = np.asarray(p_mb, np.float64)
    e = np.minimum(sat_vapor_pressure(td - 273.15), 0.95 * p)
    r = 0.62197 * e / (p - e)                                       # kg/kg
    tl = np.where(td - t > -1e-6, t, 1.0 / (1.0 / (td - 56.0) + np.log(t / td) / 800.0) + 56.0)
    th_dl = t * (1000.0 / (p - e)) ** 0.2854 * (t / tl) ** (0.28 * r)
    return th_dl * np.exp((3036.0 / tl - 1.78) * r * (1.0 + 0.448 * r))


def _bilinear(a, cr):
    """Values of ``a`` at fractional (col, row) positions ``cr`` (n x 2)."""
    c, r = cr[:, 0], cr[:, 1]
    c0 = np.clip(np.floor(c).astype(int), 0, a.shape[1] - 2)
    r0 = np.clip(np.floor(r).astype(int), 0, a.shape[0] - 2)
    fc, fr = np.clip(c - c0, 0, 1), np.clip(r - r0, 0, 1)
    return (a[r0, c0] * (1 - fc) * (1 - fr) + a[r0, c0 + 1] * fc * (1 - fr) + a[r0 + 1, c0] * (1 - fc) * fr +
            a[r0 + 1, c0 + 1] * fc * fr)


def _km(lat, lon):
    """Length (km) of each step of a lat/lon polyline."""
    la, lo = np.radians(lat), np.radians(lon)
    d = 6371.0 * np.hypot(np.diff(la), np.cos(0.5 * (la[1:] + la[:-1])) * np.diff(lo))
    return d


def _moving_mean(a, w):
    if len(a) < 3 or w < 2:
        return a
    w = min(w, len(a) | 1)
    pad = np.pad(a, (w // 2, w // 2), mode="edge")
    return np.convolve(pad, np.ones(w) / w, mode="valid")[: len(a)]


def coastal_barrier(coast_lines, lakes):
    """One line geometry (map coordinates) of the coast and the shores of the large lakes, for ``drop_coastal``."""
    parts = list(coast_lines) + [g.boundary for g in lakes if g.area > BIG_LAKE_M2]
    return shapely.union_all(parts) if parts else None


def drop_coastal(fronts: list[Front], to_xy, barrier, km=COAST_KM, fraction=COAST_FRACTION) -> list[Front]:
    """Remove fronts that hug a coast / big-lake shore (``barrier`` from ``coastal_barrier``; ``to_xy(lon, lat)`` -> map metres)."""
    if barrier is None:
        return list(fronts)
    out = []
    for f in fronts:
        ll = np.array(f.points)
        xy = np.asarray(to_xy(ll[:, 1], ll[:, 0]), float)[:, :2]
        near = shapely.distance(shapely.points(xy), barrier) < km * 1e3
        if near.mean() < fraction:
            out.append(f)
    return out


@dataclass
class FrontField:
    """The gradient quantities of the thermal field, in grid coordinates (per metre / per second)."""
    tau: np.ndarray             # smoothed thermal field (K)
    g: np.ndarray               # |grad tau| (K/m)
    f: np.ndarray               # d|grad tau| / dn
    dfdn: np.ndarray            # derivative of f along n: negative where |grad tau| is a maximum
    nx: np.ndarray
    ny: np.ndarray
    vn: np.ndarray              # wind component along n (m/s)


def front_field(tau, u, v, geom: Geometry) -> FrontField:
    s = smooth(tau, SIGMA_TAU)
    tx, ty = geom.ddx(s), geom.ddy(s)
    g = np.hypot(tx, ty)
    with np.errstate(invalid="ignore", divide="ignore"):
        nx, ny = tx / g, ty / g
    gs = smooth(g, SIGMA_G)
    f = geom.ddx(gs) * nx + geom.ddy(gs) * ny
    dfdn = geom.ddx(f) * nx + geom.ddy(f) * ny
    vn = smooth(u, SIGMA_WIND) * nx + smooth(v, SIGMA_WIND) * ny
    return FrontField(s, g, f, dfdn, nx, ny, vn)


def _runs(mask):
    """(start, stop) index pairs of consecutive True values."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def find_fronts(ff: FrontField, lat, lon, terrain_ft, *, g_min=G_MIN_K_PER_100KM, min_km=MIN_FRONT_KM,
                min_piece_km=MIN_PIECE_KM, vn_stationary=V_N_STATIONARY) -> list[Front]:
    ny_, nx_ = ff.f.shape
    f = np.ma.masked_invalid(ff.f)
    lines = contour_generator(z=f).lines(0.0)
    g_thr = g_min / 1e5
    high = np.asarray(terrain_ft, np.float64) > MAX_TERRAIN_M * FT_PER_M
    fronts: list[Front] = []
    for line in lines:
        if len(line) < 8:
            continue
        g = _bilinear(np.nan_to_num(ff.g), line)
        d = _bilinear(np.nan_to_num(ff.dfdn, nan=1.0), line)
        hi = _bilinear(high.astype(float), line) > 0.5
        edge = ((line[:, 0] < EDGE_CELLS) | (line[:, 0] > nx_ - 1 - EDGE_CELLS) | (line[:, 1] < EDGE_CELLS) |
                (line[:, 1] > ny_ - 1 - EDGE_CELLS))
        ok = (g >= g_thr) & (d < 0) & ~hi & ~edge
        # bridge short gaps (a few cells) so one front is not cut into dashes by a noisy vertex
        for a, b in _runs(~ok):
            if b - a <= 4 and a > 0 and b < len(ok):
                ok[a:b] = True
        for a, b in _runs(ok):
            seg = line[a:b]
            if len(seg) < 8:
                continue
            lats, lons = _bilinear(lat, seg), _bilinear(lon, seg)
            if _km(lats, lons).sum() < min_km:
                continue
            vn = _moving_mean(_bilinear(np.nan_to_num(ff.vn), seg), 41)
            kind = np.where(vn > vn_stationary, 1, np.where(vn < -vn_stationary, -1, 0))
            seglen = np.concatenate([[0.0], np.cumsum(_km(lats, lons))])
            pieces = []                                              # (start, stop, class), merged below
            for cls in (1, 0, -1):
                for i, j in _runs(kind == cls):
                    pieces.append([i, j, cls])
            pieces.sort()
            merged: list[list[int]] = []
            for p in pieces:
                if merged and (seglen[min(p[1], len(seglen) - 1)] - seglen[p[0]] < min_piece_km or merged[-1][2] == p[2]):
                    merged[-1][1] = p[1]
                else:
                    merged.append(p)
            if merged and len(merged) > 1 and seglen[min(merged[0][1], len(seglen) - 1)] - seglen[merged[0][0]] < min_piece_km:
                merged[1][0] = merged[0][0]
                merged.pop(0)
            for i, j, cls in merged:
                j = min(j + 1, len(seg))                              # pieces share their end vertex so the line stays joined
                if _km(lats[i:j], lons[i:j]).sum() < min_km * 0.5:
                    continue
                step = 4
                la, lo = _moving_mean(lats[i:j], 9)[::step], _moving_mean(lons[i:j], 9)[::step]
                la[-1], lo[-1] = _moving_mean(lats[i:j], 9)[-1], _moving_mean(lons[i:j], 9)[-1]
                kindname = "cold" if cls > 0 else "warm" if cls < 0 else "stationary"
                fronts.append(Front(kindname, tuple(zip(la.tolist(), lo.tolist()))))
    return fronts


def find_centers(mslp, terrain_ft, lat, lon, geom: Geometry, *, radius_km=CENTER_RADIUS_KM, depth_mb=CENTER_DEPTH_MB) -> list[Center]:
    """Closed lows and highs of the smoothed MSLP: the extreme within ``radius_km`` and at least ``depth_mb`` away from the
    mean of a ring at 1.35 x that radius (needs most of the ring inside the data), not over high terrain."""
    m = np.where(np.asarray(terrain_ft) > MAX_TERRAIN_M * FT_PER_M, np.nan, mslp)
    sm = smooth(m, CENTER_SIGMA)
    fine = smooth(m, 3.0)
    ny, nx = sm.shape
    cell_km = float(np.nanmean(geom.dx)) / 1e3
    st = 3                                                        # the smoothed field is smooth: search a coarse copy of it
    rad = max(2, int(round(radius_km / cell_km / st)))           # radius and ring in coarse cells
    ring_r = rad * 1.35
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    disk = (yy ** 2 + xx ** 2) <= rad ** 2
    found: list[Center] = []
    for kind, sign in (("L", 1.0), ("H", -1.0)):                  # work on sign * p so both are searches for a minimum
        a = (sign * sm)[::st, ::st]
        cny, cnx = a.shape
        for r in range(rad, cny - rad):
            for c in range(rad, cnx - rad):
                v = a[r, c]
                if not np.isfinite(v):
                    continue
                win = a[r - rad:r + rad + 1, c - rad:c + rad + 1]
                if np.nanmin(np.where(disk, win, np.nan)) < v - 1e-9:
                    continue
                ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
                rr, cc = np.rint(r + ring_r * np.sin(ang)).astype(int), np.rint(c + ring_r * np.cos(ang)).astype(int)
                inside = (rr >= 0) & (rr < cny) & (cc >= 0) & (cc < cnx)
                if inside.sum() < 12:
                    continue
                ring = a[rr[inside], cc[inside]]
                ring = ring[np.isfinite(ring)]
                if len(ring) < 10 or ring.mean() - v < depth_mb:
                    continue
                R, C = r * st, c * st                             # refine to the exact extreme cell nearby on the lightly smoothed field
                r0, r1, c0, c1 = max(R - 2 * st, 0), min(R + 2 * st + 1, ny), max(C - 2 * st, 0), min(C + 2 * st + 1, nx)
                sub = sign * fine[r0:r1, c0:c1]
                if not np.isfinite(sub).any():
                    continue
                k = np.unravel_index(np.nanargmin(sub), sub.shape)
                rr0, cc0 = r0 + k[0], c0 + k[1]
                found.append(Center(kind, float(lat[rr0, cc0]), float(lon[rr0, cc0]), float(round(sign * sub[k]))))
    # one centre per neighbourhood: keep the deepest (lowest low / highest high)
    out: list[Center] = []
    for cen in sorted(found, key=lambda x: x.pressure if x.kind == "L" else -x.pressure):
        if all(not (o.kind == cen.kind and _km(np.array([o.lat, cen.lat]), np.array([o.lon, cen.lon]))[0] < 1.6 * radius_km)
               for o in out):
            out.append(cen)
    return out


def analyze(valid: datetime, t_f, td_f, p_mb, mslp, u, v, terrain_ft, lat, lon, geom: Geometry, to_xy=None, barrier=None) -> Analysis:
    """Objective fronts and pressure centres of one analysis time (all fields on the analysis window grid).  With ``to_xy`` and
    ``barrier`` (see ``drop_coastal``) fronts that follow a coast or big-lake shore are dropped."""
    tau = equivalent_potential_temperature(t_f, td_f, p_mb)
    ff = front_field(tau, u, v, geom)
    fronts = find_fronts(ff, lat, lon, terrain_ft)
    if to_xy is not None:
        fronts = drop_coastal(fronts, to_xy, barrier)
    centers = find_centers(mslp, terrain_ft, lat, lon, geom)
    return Analysis(valid, "RTMA", tuple(centers), tuple(fronts))

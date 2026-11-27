"""Draw a surface analysis (fronts, troughs, highs, lows) on a map axes with the standard NWS / WPC symbols.

* cold front: blue line with triangles; warm front: red line with semicircles; stationary: alternating blue triangles and red
  semicircles on opposite sides; occluded: purple line with alternating triangles and semicircles on the same side; trough:
  dashed brown line (WPC's "TROF" also covers outflow boundaries, squall lines and dry lines).
* Symbols point in the direction the front moves.  The bulletin gives only the line, so the side is taken from the air itself:
  a cold front advances into warmer air (triangles on the warmer side), a warm front into colder air (semicircles on the
  colder side), and an occluded front like a cold front.  ``warmer`` samples a thermal field (equivalent potential
  temperature) on either side of the line; without it the side falls back to east (cold, occluded) and north (warm).
* The lines pass through the bulletin's points as a smooth Catmull-Rom curve: nothing is moved, only joined.
"""
from __future__ import annotations

from typing import Callable

import matplotlib.patheffects as pe
import numpy as np
from matplotlib.collections import PolyCollection

from .fronts import Analysis, Front

COLORS = {"cold": "#1f4fe0", "warm": "#e01f1f", "occluded": "#8a2be2", "stationary": "#1f4fe0", "trough": "#9a5b1f"}
WARM_RED = "#e01f1f"
H_COLOR, L_COLOR = "#1f3fd6", "#d11f1f"

BASE_M = 24e3             # symbol width along the line
HEIGHT_M = 15e3           # triangle height / semicircle radius is BASE / 2
SPACING_M = 64e3          # start-to-start distance of symbols
DENSE_STEP_M = 3e3
SIDE_PROBE_M = 60e3       # how far either side of the line the thermal field is sampled
LINE_W = 2.3


def catmull_rom(xy: np.ndarray, step: float = DENSE_STEP_M) -> np.ndarray:
    """Smooth curve through ``xy`` (n x 2, metres), sampled about every ``step`` metres."""
    xy = np.asarray(xy, float)
    if len(xy) < 3:
        n = max(2, int(np.hypot(*(xy[-1] - xy[0])) / step) + 1)
        return np.column_stack([np.linspace(xy[0, 0], xy[-1, 0], n), np.linspace(xy[0, 1], xy[-1, 1], n)])
    p = np.vstack([2 * xy[0] - xy[1], xy, 2 * xy[-1] - xy[-2]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        n = max(2, int(np.hypot(*(p2 - p1)) / step))
        u = np.linspace(0, 1, n, endpoint=False)[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
    out.append(xy[-1:])
    return np.vstack(out)


class FieldSampler:
    """Nearest-cell lookup of a gridded field at map (projected) coordinates.  The analysis grid is close to affine in the map
    projection (a few km of error over the domain), which is plenty for a probe tens of km either side of a front."""

    def __init__(self, x: np.ndarray, y: np.ndarray, field: np.ndarray):
        ny, nx = x.shape
        rr, cc = np.mgrid[0:ny:4, 0:nx:4]
        A = np.column_stack([cc.ravel(), rr.ravel(), np.ones(cc.size)])
        cx = np.linalg.lstsq(A, x[::4, ::4].ravel(), rcond=None)[0]
        cy = np.linalg.lstsq(A, y[::4, ::4].ravel(), rcond=None)[0]
        self._inv = np.linalg.inv(np.array([[cx[0], cx[1]], [cy[0], cy[1]]]))
        self._off = np.array([cx[2], cy[2]])
        self.field = field

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        col_row = (np.asarray(xy, float) - self._off) @ self._inv.T
        c, r = np.rint(col_row[:, 0]).astype(int), np.rint(col_row[:, 1]).astype(int)
        ny, nx = self.field.shape
        ok = (c >= 0) & (c < nx) & (r >= 0) & (r < ny)
        out = np.full(len(c), np.nan)
        out[ok] = self.field[r[ok], c[ok]]
        return out


def _frame(dense: np.ndarray):
    """Cumulative distance, unit tangents and left-hand unit normals along a dense polyline."""
    d = np.diff(dense, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.gradient(dense, axis=0)
    t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
    n = np.column_stack([-t[:, 1], t[:, 0]])
    return s, t, n


def warm_side_sign(dense: np.ndarray, n: np.ndarray, warmer: Callable | None, kind: str) -> float:
    """+1 if the warmer air is on the left of the line's direction, -1 if on the right.  Falls back to a fixed convention
    (cold / occluded fronts advance east, warm fronts north, so their warm side is the east / north one...) when the thermal
    field says nothing."""
    if warmer is not None and len(dense) > 3:
        idx = np.arange(0, len(dense), max(1, len(dense) // 25))
        left = warmer(dense[idx] + n[idx] * SIDE_PROBE_M)
        right = warmer(dense[idx] - n[idx] * SIDE_PROBE_M)
        diff = left - right
        diff = diff[np.isfinite(diff)]
        if len(diff) >= 2 and abs(diff.mean()) > 0.5:
            return 1.0 if diff.mean() > 0 else -1.0
    axis = 1 if kind == "warm" else 0                                 # warm front: poleward (north); the others: eastward
    return 1.0 if n[:, axis].sum() * (1 if kind != "warm" else -1) >= 0 else -1.0


def _triangle(p, t, n, side):
    a, b = p[0], p[1]
    return np.array([a, b, (a + b) / 2 + side * n * HEIGHT_M])


def _semicircle(p, t, n, side):
    a, b = p[0], p[1]
    mid, r = (a + b) / 2, np.hypot(*(b - a)) / 2
    ang = np.linspace(0, np.pi, 14)
    arc = mid + r * np.cos(ang)[:, None] * (t) + side * r * np.sin(ang)[:, None] * n
    return arc                                                        # from b round to a; the polygon closes along the base


def _along(dense, s, dist):
    return np.array([np.interp(dist, s, dense[:, 0]), np.interp(dist, s, dense[:, 1])])


def front_symbols(front: Front, to_xy, warmer: Callable | None = None):
    """(dense line xy, [(polygon, colour), ...]) of one front in map coordinates."""
    ll = np.array(front.points)
    xy = np.asarray(to_xy(ll[:, 1], ll[:, 0]), float)[:, :2]
    dense = catmull_rom(xy)
    polys: list = []
    if front.kind == "trough" or len(dense) < 3:
        return dense, polys
    s, t, n = _frame(dense)
    warm_sign = warm_side_sign(dense, n, warmer, front.kind)
    k, dist = 0, SPACING_M / 2
    while dist + BASE_M < s[-1]:
        p = np.array([_along(dense, s, dist), _along(dense, s, dist + BASE_M)])
        i = int(np.searchsorted(s, dist + BASE_M / 2))
        i = min(i, len(s) - 1)
        tt, nn = t[i], n[i]
        if front.kind == "cold":
            polys.append((_triangle(p, tt, nn, warm_sign), COLORS["cold"]))
        elif front.kind == "warm":
            polys.append((_semicircle(p, tt, nn, -warm_sign), COLORS["warm"]))
        elif front.kind == "occluded":
            shape = _triangle if k % 2 == 0 else _semicircle
            polys.append((shape(p, tt, nn, warm_sign), COLORS["occluded"]))
        elif front.kind == "stationary":
            if k % 2 == 0:
                polys.append((_triangle(p, tt, nn, warm_sign), COLORS["cold"]))
            else:
                polys.append((_semicircle(p, tt, nn, -warm_sign), COLORS["warm"]))
        k += 1
        dist += SPACING_M
    return dense, polys


def _near(xy: np.ndarray, extent, pad: float) -> bool:
    x0, x1, y0, y1 = extent
    return bool((xy[:, 0] > x0 - pad).any() and (xy[:, 0] < x1 + pad).any() and (xy[:, 1] > y0 - pad).any()
                and (xy[:, 1] < y1 + pad).any() and
                ((xy[:, 0] > x0 - pad) & (xy[:, 0] < x1 + pad) & (xy[:, 1] > y0 - pad) & (xy[:, 1] < y1 + pad)).any())


def draw_analysis(ax, analysis: Analysis, to_xy, extent, warmer: Callable | None = None, label: str = "", zorder: float = 6.0):
    """Draw fronts, troughs and pressure centres of ``analysis`` that fall in ``extent`` = (x0, x1, y0, y1) map metres."""
    halo = [pe.withStroke(linewidth=LINE_W + 1.3, foreground="white")]
    for front in analysis.fronts:
        ll = np.array(front.points)
        xy = np.asarray(to_xy(ll[:, 1], ll[:, 0]), float)[:, :2]
        if not _near(xy, extent, 150e3):
            continue
        dense, polys = front_symbols(front, to_xy, warmer)
        if front.kind == "trough":
            ax.plot(dense[:, 0], dense[:, 1], color="white", lw=3.4, zorder=zorder, alpha=0.9, solid_capstyle="butt")
            ax.plot(dense[:, 0], dense[:, 1], color=COLORS["trough"], lw=2.0, ls=(0, (5, 3)), zorder=zorder + 0.05,
                    dash_capstyle="butt")
            continue
        ax.plot(dense[:, 0], dense[:, 1], color=COLORS[front.kind if front.kind != "stationary" else "cold"], lw=LINE_W,
                zorder=zorder, path_effects=halo, solid_capstyle="butt")
        if polys:
            ax.add_collection(PolyCollection([p for p, _ in polys], facecolors=[c for _, c in polys], edgecolors="white",
                                             linewidths=0.6, zorder=zorder + 0.1))
    x0, x1, y0, y1 = extent
    keep = 30e3
    for c in analysis.centers:
        px, py = np.asarray(to_xy(np.array([c.lon]), np.array([c.lat])), float)[0, :2]
        if not (x0 + keep < px < x1 - keep and y0 + keep < py < y1 - keep):
            continue
        colour = H_COLOR if c.kind == "H" else L_COLOR
        ax.annotate(c.kind, (px, py), ha="center", va="center", fontsize=26, fontweight="bold", color=colour, zorder=zorder + 0.3,
                    path_effects=[pe.withStroke(linewidth=2.0, foreground="white")], annotation_clip=False)
        ax.annotate(f"{c.pressure:.0f}", (px, py), xytext=(0, -19), textcoords="offset points", ha="center", va="center", fontsize=10,
                    fontweight="bold", color=colour, zorder=zorder + 0.3,
                    path_effects=[pe.withStroke(linewidth=1.7, foreground="white")], annotation_clip=False)
    if label:
        ax.text(0.008, 0.012, label, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="left", va="bottom",
                zorder=zorder + 0.5, path_effects=[pe.withStroke(linewidth=1.9, foreground="white")])

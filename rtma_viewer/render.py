"""Cartopy/matplotlib map panels -> small, palette-quantised PNGs.

Look (after the Penn State e-Wall style): white page with a thin-framed map, a one-line header
("RTMA: 2m Temperature (°F)" left, "Valid: 17:00z Thu Nov 5 2026" right), a wide segmented colour bar
bottom-left, thick state lines over thin county lines, and bold haloed values at METAR stations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import shapely
from cartopy.feature import ShapelyFeature
from PIL import Image

from . import boundaries as bnd
from . import stations as stn
from .colors import Style, style_for
from .config import DISPLAY_BBOX, Product, Variable
from .grib import Grid

log = logging.getLogger(__name__)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]      # bundled with matplotlib -> identical everywhere
plt.rcParams["axes.unicode_minus"] = True

PC = ccrs.PlateCarree()
PROJ = ccrs.LambertConformal(central_longitude=-70.6, central_latitude=44.2, standard_parallels=(42.0, 46.0))

BRANDING = ("aaronperry-dev.github.io", " (Aaron Perry)")
SHOW_STATION_IDS = False         # True adds a small ICAO id under each value

# page layout, inches (12 x 8.76 at 100 dpi = 1200 x 876 px)
FIG_W = 12.0
MARGIN, HEADER_H, FOOTER_H = 0.25, 0.52, 0.82
MAP_W = FIG_W - 2 * MARGIN
MAP_H = MAP_W / 1.55
FIG_H = HEADER_H + MAP_H + FOOTER_H

# map frame: centred on New England, 880 km tall; width follows from the aspect ratio
MAP_CENTER, HALF_H_M = (-71.4, 44.2), 440e3
MIN_SEP_M, EDGE_M = 40e3, 22e3        # station label spacing / keep-out near the frame edge


def _extent():
    cx, cy = PROJ.transform_point(*MAP_CENTER, PC)
    hw = HALF_H_M * MAP_W / MAP_H
    return cx - hw, cx + hw, cy - HALF_H_M, cy + HALF_H_M


EXTENT = _extent()


def _check_frame_inside_bbox():
    x0, x1, y0, y1 = EXTENT
    t = np.linspace(0, 1, 100)
    xs = np.concatenate([x0 + (x1 - x0) * t, np.full(100, x1), x1 - (x1 - x0) * t, np.full(100, x0)])
    ys = np.concatenate([np.full(100, y0), y0 + (y1 - y0) * t, np.full(100, y1), y1 - (y1 - y0) * t])
    ll = PC.transform_points(PROJ, xs, ys)
    lon0, lon1, lat0, lat1 = DISPLAY_BBOX
    if ll[:, 0].min() < lon0 or ll[:, 0].max() > lon1 or ll[:, 1].min() < lat0 or ll[:, 1].max() > lat1:
        raise ValueError(f"map frame spills outside DISPLAY_BBOX: lon {ll[:,0].min():.2f}..{ll[:,0].max():.2f}, "
                         f"lat {ll[:,1].min():.2f}..{ll[:,1].max():.2f}")


_check_frame_inside_bbox()


@dataclass
class MapContext:
    grid: Grid
    x: np.ndarray
    y: np.ndarray
    land: np.ndarray                                   # bool mask: inside the six New England states
    labels: list[tuple[stn.Station, tuple[int, int], float, float]]   # station, nearest (row, col), x, y
    layers: bnd.Boundaries

    @classmethod
    def build(cls, grid: Grid, layers: bnd.Boundaries, stations: list[stn.Station] | None = None) -> "MapContext":
        xy = PROJ.transform_points(PC, grid.lon, grid.lat)
        x, y = xy[..., 0], xy[..., 1]
        land = shapely.contains_xy(layers.ne_land, grid.lon, grid.lat)
        stations = stn.load() if stations is None else stations
        sxy = PROJ.transform_points(PC, np.array([s.lon for s in stations]), np.array([s.lat for s in stations]))[:, :2]
        labels = []
        for i in stn.declutter(stations, sxy, EXTENT, MIN_SEP_M, EDGE_M):
            px, py = sxy[i]
            d = (x - px) ** 2 + (y - py) ** 2
            k = np.unravel_index(np.argmin(d), d.shape)
            if d[k] < (6000.0) ** 2:                                   # nearest analysis cell must really be there
                labels.append((stations[i], k, px, py))
        return cls(grid, x, y, land, labels, layers)


def region_stats(ctx: MapContext, field: np.ndarray) -> dict:
    v = field[ctx.land & np.isfinite(field)]
    if v.size == 0:
        return {"min": None, "mean": None, "max": None}
    return {"min": round(float(v.min()), 1), "mean": round(float(v.mean()), 1), "max": round(float(v.max()), 1)}


def station_values(ctx: MapContext, field: np.ndarray) -> dict[str, float | None]:
    return {s.icao: (round(float(field[k]), 1) if np.isfinite(field[k]) else None) for s, k, _, _ in ctx.labels}


def _bold_regular(fig, x, y, bold, regular, size, right=False):
    """One line that starts with a bold word followed by regular text (x in figure fraction)."""
    r = fig.canvas.get_renderer()
    wpx = fig.get_figwidth() * fig.dpi
    kw = dict(fontsize=size, color="black", va="baseline")
    if not right:
        tb = fig.text(x, y, bold, fontweight="bold", **kw)
        fig.text(x + tb.get_window_extent(r).width / wpx, y, regular, **kw)
    else:
        tr = fig.text(x, y, regular, ha="right", **kw)
        fig.text(x - tr.get_window_extent(r).width / wpx, y, bold, ha="right", fontweight="bold", **kw)


def render_panel(ctx: MapContext, field: np.ndarray, style: Style, *, texts: tuple[str, str, str, str], signed: bool,
                 out: Path, dpi: int = 100, colors: int = 256) -> int:
    cmap, norm = style.cmap_norm()
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=dpi, facecolor="white")
    ax = fig.add_axes([MARGIN / FIG_W, FOOTER_H / FIG_H, MAP_W / FIG_W, MAP_H / FIG_H], projection=PROJ)
    ax.set_extent(list(EXTENT), crs=PROJ)
    ax.spines["geo"].set_linewidth(1.0)
    ax.spines["geo"].set_edgecolor("black")

    mesh = ax.pcolormesh(ctx.x, ctx.y, np.ma.masked_invalid(field), cmap=cmap, norm=norm, shading="nearest", zorder=1)

    L = ctx.layers
    def add(geoms, **kw):
        if geoms:
            ax.add_feature(ShapelyFeature(geoms, PROJ, facecolor="none", **kw))
    add(L.counties, edgecolor="black", linewidth=0.45, alpha=0.85, zorder=2)
    add(L.lakes, edgecolor="black", linewidth=1.0, zorder=3)
    add(L.states, edgecolor="black", linewidth=1.8, zorder=4)
    add(L.coast, edgecolor="black", linewidth=1.2, zorder=4)

    # values at METAR stations (nearest analysis cell), bold with a white halo
    halo = lambda w: [pe.withStroke(linewidth=w, foreground="white")]
    for st, (r, c), px, py in ctx.labels:
        v = field[r, c]
        if not np.isfinite(v):
            continue
        n = int(round(float(v)))
        txt = "0" if n == 0 else (f"{n:+d}" if signed else f"{n:d}")
        ax.annotate(txt, (px, py), xytext=(0, 1.5 if SHOW_STATION_IDS else 0), textcoords="offset points",
                    fontsize=7.4, fontweight="bold", color="black", ha="center", va="center", zorder=8,
                    path_effects=halo(2.4), annotation_clip=False)
        if SHOW_STATION_IDS:
            ax.annotate(st.icao, (px, py), xytext=(0, -6.6), textcoords="offset points", fontsize=5.8,
                        color="#222222", ha="center", va="center", zorder=8, path_effects=halo(1.8),
                        annotation_clip=False)

    # ------------------------------------------------------------------ header (title left, valid time right)
    base = 1 - 0.35 / FIG_H
    left_bold, left_rest, right_bold, right_rest = texts
    _bold_regular(fig, MARGIN / FIG_W, base, left_bold, left_rest, 12.5)
    _bold_regular(fig, 1 - MARGIN / FIG_W, base, right_bold, right_rest, 12.5, right=True)

    # ------------------------------------------------------------------ colour bar (bottom left) + branding
    bar_h, bar_y = 0.28, FOOTER_H - 0.13 - 0.28
    cax = fig.add_axes([0.48 / FIG_W, bar_y / FIG_H, 7.35 / FIG_W, bar_h / FIG_H])
    fmt = (lambda v, _: "0" if abs(v) < 1e-9 else f"{v:+g}") if signed else (lambda v, _: f"{v:g}")
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="neither", ticks=style.ticks,
                      format=mticker.FuncFormatter(fmt), spacing="proportional")
    cb.ax.tick_params(labelsize=12.5, length=3, width=0.8, pad=2, colors="black")
    cb.outline.set_linewidth(0.9)
    cy = (bar_y + bar_h / 2) / FIG_H
    _bold_regular(fig, 1 - MARGIN / FIG_W, cy - 0.006, BRANDING[0], BRANDING[1], 10.5, right=True)

    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    exact = np.vstack([np.round(style.colors * 255).astype(np.uint8),
                       np.array([[255, 255, 255], [0, 0, 0]], np.uint8)])
    return save_png(rgb, out, colors, exact)


def save_png(rgb: np.ndarray, out: Path, colors: int = 256, exact: np.ndarray | None = None) -> int:
    """Indexed PNG with a *fixed* palette: the exact colour-bin colours (so legend swatches and flat bins are
    reproduced exactly, however little area they cover) plus a median-cut palette for everything else
    (anti-aliased text/lines).  No dithering.  Then lossless recompression if ``oxipng`` is available."""
    flat = rgb.reshape(-1, 3)
    exact = np.zeros((0, 3), np.uint8) if exact is None else np.unique(exact, axis=0)
    pack = lambda a: (a[:, 0].astype(np.uint32) << 16) | (a[:, 1].astype(np.uint32) << 8) | a[:, 2].astype(np.uint32)
    rest = flat[~np.isin(pack(flat), pack(exact))]
    n_rest = max(8, colors - len(exact))
    pal = list(map(tuple, exact))
    if rest.size:
        q = Image.fromarray(rest.reshape(1, -1, 3)).quantize(colors=n_rest, method=Image.Quantize.MEDIANCUT,
                                                              dither=Image.Dither.NONE)
        raw = q.getpalette()[: 3 * n_rest]
        pal += [tuple(raw[i:i + 3]) for i in range(0, len(raw), 3)]
    pal = list(dict.fromkeys(pal))[:256]
    palette_img = Image.new("P", (1, 1))
    flat_pal = [c for col in pal for c in col]
    palette_img.putpalette(flat_pal + [0] * (768 - len(flat_pal)))
    img = Image.fromarray(rgb).quantize(palette=palette_img, dither=Image.Dither.NONE)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, format="PNG", optimize=True, compress_level=9)
    try:
        import oxipng
        oxipng.optimize(str(out), level=4)
    except Exception:
        pass
    return out.stat().st_size


# ------------------------------------------------------------------------- header text
def _z(t: datetime) -> str:
    return f"{t:%H:%Mz %a %b} {t.day}"


def panel_text(product: Product, var: Variable, interval: int, valid: datetime, ref: datetime | None):
    """(left bold, left regular, right bold, right regular) - e.g. 'RTMA' ': 2m Temperature (°F)' 'Valid' ': 17:00z Thu Nov 5 2026'."""
    chg = f" {interval}-hr Change" if interval else ""
    right = f": {_z(valid)} {valid.year}" + (f"  (vs {_z(ref)})" if ref else "")
    return product.label, f": {var.long_name}{chg} ({var.unit})", "Valid", right


def style_and_flags(var: Variable, interval: int):
    return style_for(var.key, interval), interval > 0

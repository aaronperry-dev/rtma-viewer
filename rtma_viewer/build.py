"""Build the static site: fetch -> render -> data.json + index.html.

    python -m rtma_viewer.build --out public --cache .cache/fields
    python -m rtma_viewer.build --demo            # offline, synthetic data
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import boundaries, fronts, objective, render, symbols
from .config import FT_PER_M, INTERVALS, PANEL_VARIABLES, PRODUCTS, SOURCE_VARIABLES, VARIABLES, hours_needed, panel_hours
from .fields import FieldStore
from . import kinematics as kin
from .kinematics import Geometry

log = logging.getLogger("rtma_viewer")
SITE_DIR = Path(__file__).resolve().parent.parent / "site"
iso = lambda t: t.strftime("%Y-%m-%dT%H:%M:%SZ")
MIN_SERIES_FRACTION = 0.75       # a 24-hour summary needs at least this share of its hourly analyses (19 of 25)


def series_inputs(fields: dict, var, valid: datetime):
    """Stacks (hours + 1, ny, nx) of each input over the ``var.series`` hours up to ``valid`` (index 0 = valid time), with NaN
    planes for hours that are missing.  None if there is too little to work with.

    Ordinary summaries (24-hour temperature ...) blank an hour for *all* inputs when any input lacks it, and need 75 % of the
    hours.  ``partial_series`` products (precipitation totals) use whatever exists: each input's missing hours are blanked on their
    own, and one usable accumulation hour is enough (``available_hours`` counts them, and the map says how many there were)."""
    n = var.series + 1
    cols = [[fields.get((valid - timedelta(hours=i), k)) for i in range(n)] for k in var.requires]
    if var.partial_series:
        if available_hours(fields, var, valid) == 0:
            return None
        blank = np.full(next(p for c in cols for p in c if p is not None).shape, np.nan, np.float32)
        return [np.stack([p if p is not None else blank for p in c]) for c in cols]
    ok = [all(c[i] is not None for c in cols) for i in range(n)]
    if sum(ok) < MIN_SERIES_FRACTION * n or not ok[0]:
        return None
    blank = np.full(next(c[0] for c in cols).shape, np.nan, np.float32)
    return [np.stack([c[i] if ok[i] else blank for i in range(len(ok))]) for c in cols]


def available_hours(fields: dict, var, valid: datetime) -> int:
    """How many of a precipitation total's ``var.period_hours`` hourly accumulations can actually be used: the hour needs its
    precipitation analysis and, for frozen precipitation, a wet bulb at one end of the hour at least."""
    got = lambda key, i: fields.get((valid - timedelta(hours=i), key)) is not None
    wet = [k for k in var.requires if k != "pcp1"]
    count = 0
    for i in range(var.period_hours):
        if got("pcp1", i) and (not wet or all(got(k, i) for k in wet) or all(got(k, i + 1) for k in wet)):
            count += 1
    return count


def surface_overlay(var, product, cycle: datetime, fields: dict, ctx, geom, archive) -> tuple | None:
    """(callable for render_panel's ``overlay``, short label) drawing fronts and pressure centres for ``var.overlay`` at ``cycle``,
    or None when there is nothing to draw (no bulletin near that time, missing inputs)."""
    to_xy = lambda lon, lat: render.PROJ.transform_points(render.PC, np.asarray(lon, float), np.asarray(lat, float))
    if var.overlay == "wpc":
        an = archive.nearest(cycle, max_back=fronts.MAX_BACK) if archive is not None else None
        if an is None:
            log.warning("%s %s: no WPC analysis within %d min of %s", product.key, var.key, int(fronts.MAX_GAP.total_seconds() // 60), cycle)
            return None
        label = f"WPC analysis valid {an.valid:%H}Z" + ("" if an.valid == cycle else f" - {abs((cycle - an.valid).total_seconds()) / 3600:.0f} h {'before' if an.valid < cycle else 'after'} this map")
    else:
        need = ("temp", "dewpoint", "pressure", "mslp", "u10", "v10", "terrain")
        f = {k: fields.get((cycle, k)) for k in need}
        if any(v is None for v in f.values()) or geom is None:
            return None
        an = objective.analyze(cycle, f["temp"], f["dewpoint"], f["pressure"], f["mslp"], f["u10"], f["v10"], f["terrain"],
                               ctx.grid.lat, ctx.grid.lon, geom, to_xy, objective.coastal_barrier(ctx.layers.coast, ctx.layers.lakes))
        label = "Objective (experimental)"
    t = fields.get((cycle, "temp")); td = fields.get((cycle, "dewpoint")); p = fields.get((cycle, "pressure"))
    warmer = None
    if t is not None and td is not None and p is not None:
        tau = kin.smooth(objective.equivalent_potential_temperature(t, td, p), objective.SIGMA_TAU)
        warmer = symbols.FieldSampler(ctx.x, ctx.y, tau)
    log.info("%s %s: %d fronts, %d pressure centres (%s)", product.key, var.key, len(an.fronts), len(an.centers), label)
    return (lambda ax, extent: symbols.draw_analysis(ax, an, to_xy, extent, warmer, label)), an


def build_product(product, client, now, out: Path, cache: Path, layers, dpi: int, colors: int, wpc: bool = False) -> dict:
    t0 = time.time()
    store = FieldStore(product, client, cache)
    cycle = store.latest_cycle(now)
    rec = {"label": product.label, "long_name": product.long_name, "blurb": product.blurb,
           "status": "unavailable", "cycle": None, "panels": {}}
    if cycle is None:
        log.error("%s: no cycle found in the last %d h", product.key, product.lookback_hours)
        return rec
    log.info("%s: newest cycle %s", product.key, cycle)

    need = hours_needed()                                 # only the cycles each variable's panels (and derived inputs) use
    jobs = sorted(((cycle - timedelta(hours=h), SOURCE_VARIABLES[k]) for k, hrs in need.items() for h in hrs),
                  key=lambda j: j[0], reverse=True)       # newest cycle first: one GRIB inventory read serves all its variables
    fields = store.load(jobs, now)
    for var in SOURCE_VARIABLES.values():                 # static fields (terrain) are valid at every time
        if var.static:
            for h in INTERVALS:
                fields[(cycle - timedelta(hours=h), var.key)] = fields.get((cycle, var.key))
    hours: dict[tuple[datetime, str], int] = {}               # usable hourly inputs of each precipitation total
    geom = Geometry.from_latlon(store.grid.lat, store.grid.lon) if store.grid is not None else None
    for var in VARIABLES.values():                        # derived variables (in dependency order), from the cached fields
        if var.derive is None:
            continue
        for h in panel_hours(var):
            t = cycle - timedelta(hours=h)
            parts = series_inputs(fields, var, t) if var.series else [fields.get((t, k)) for k in var.requires]
            if parts is None or any(a is None for a in parts):
                fields[(t, var.key)] = None
                continue
            res = var.derive(*parts, geom=geom) if var.needs_geom else var.derive(*parts)
            if var.partial_series:
                hours[(t, var.key)] = available_hours(fields, var, t)
                if not np.isfinite(res).any():            # nothing usable anywhere (e.g. no wet bulb for any hour)
                    res = None
            fields[(t, var.key)] = res
    ctx = render.MapContext.build(store.grid, layers)
    archive = fronts.Archive(cache / "wpc") if any(v.overlay == "wpc" for v in PANEL_VARIABLES.values()) else None
    if archive is not None and wpc:
        archive.update()                                  # file the newest WPC bulletin (the server keeps only that one)

    total_png = 0
    terrain_ft = fields.get((cycle, "terrain"))
    for var in PANEL_VARIABLES.values():
        rec["panels"][var.key] = {}
        skip = None                                       # stations where a value would be misleading (MSLP on summits)
        if var.label_max_terrain_m and terrain_ft is not None:
            skip = terrain_ft > var.label_max_terrain_m * FT_PER_M
        cur = fields.get((cycle, var.key))
        for h in INTERVALS:
            ref_t = cycle - timedelta(hours=h)
            if h not in panel_hours(var):                 # e.g. visibility / ceiling / wind direction: current field only
                rec["panels"][var.key][str(h)] = None
                continue
            prev = fields.get((ref_t, var.key))
            if cur is None or prev is None:
                rec["panels"][var.key][str(h)] = None
                log.warning("%s %s +%dh unavailable", product.key, var.key, h)
                continue
            data = cur if h == 0 else cur - prev
            style, signed = render.style_and_flags(var, h)
            have = hours.get((cycle - timedelta(hours=h), var.key)) if var.period_hours else None
            ov = None
            if var.overlay:
                got = surface_overlay(var, product, cycle, fields, ctx, geom, archive)
                if got is None:
                    rec["panels"][var.key][str(h)] = None
                    continue
                ov = got[0]
            texts = render.panel_text(product, var, h, cycle, ref_t if h else None, hours=have)
            rel = f"img/{product.key}_{var.key}_{h}h.png"
            stats = render.region_stats(ctx, data, circular=var.circular, categories=var.categories)
            size = render.render_panel(ctx, data, style, texts=texts, signed=signed, out=out / rel,
                                       dpi=dpi, colors=colors, var=var, skip_labels=skip, overlay=ov)
            total_png += size
            rec["panels"][var.key][str(h)] = {"file": rel, "ref": iso(ref_t) if h else None, **stats,
                                              "stations": render.station_values(ctx, data, skip)}
            if have is not None:                          # incomplete totals say so, on the map and on the page
                rec["panels"][var.key][str(h)].update(hours=have, period=var.period_hours)
                if have < var.period_hours:
                    log.warning("%s %s: only %d of %d hourly analyses available", product.key, var.key, have, var.period_hours)
    rec.update(status="ok", cycle=iso(cycle), lag_minutes=int((now - cycle).total_seconds() // 60))
    rec["bytes"] = {"png": total_png, "downloaded": store.src.bytes_downloaded,
                    "s3_requests": store.src.requests, **dict(store.stats)}
    keep = {k: (max(hrs) + 4 if max(hrs) else 0) for k, hrs in need.items()}   # current-only variables keep just the newest cycle
    store.prune(keep_hours=max(INTERVALS) + 4, now=cycle, keep_by_var=keep)
    log.info("%s: %d PNG, %.2f MB; downloaded %.2f MB in %d requests; %s; %.0fs", product.key,
             sum(1 for v in rec["panels"].values() for p in v.values() if p), total_png / 1e6,
             store.src.bytes_downloaded / 1e6, store.src.requests, dict(store.stats), time.time() - t0)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="public")
    ap.add_argument("--cache", default=".cache/fields", help="small float16 field cache (never committed)")
    ap.add_argument("--products", default="rtma,urma")
    ap.add_argument("--demo", action="store_true", help="offline synthetic data via a fake S3")
    ap.add_argument("--now", help="override 'now' (ISO UTC), mostly for demos/tests")
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--colors", type=int, default=256, help="PNG palette size (exact colour bins + anti-aliasing colours)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("botocore").setLevel(logging.WARNING)

    now = datetime.fromisoformat(args.now.replace("Z", "+00:00")) if args.now else datetime.now(timezone.utc)
    now = now.astimezone(timezone.utc)
    if args.demo:
        from .testing import FakeS3
        client = FakeS3(now)
        cache = Path(args.cache) / "demo"
    else:
        from .s3 import make_client
        client, cache = make_client(), Path(args.cache)

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "img").mkdir(parents=True)

    layers = boundaries.load(render.PROJ)
    manifest = {"generated": iso(now), "version": int(now.timestamp()), "demo": args.demo,
                "intervals": list(INTERVALS),
                "variables": [{"key": v.key, "label": v.label, "long_name": v.long_name, "unit": v.unit,
                               "group": v.group, "note": v.note, "circular": v.circular, "diverging": v.diverging,
                               "categories": list(v.categories), "digits": v.stat_decimals,
                               "lines": [{"value": lv, "color": col} for lv, col in v.lines]}
                              for v in PANEL_VARIABLES.values()],
                "products": {}}
    for key in args.products.split(","):
        manifest["products"][key] = build_product(PRODUCTS[key], client, now, out, cache, layers, args.dpi, args.colors, wpc=not args.demo)

    (out / "data.json").write_text(json.dumps(manifest, separators=(",", ":")))
    shutil.copy(SITE_DIR / "index.html", out / "index.html")
    (out / ".nojekyll").touch()

    ok = [k for k, v in manifest["products"].items() if v["status"] == "ok"]
    if not ok:
        log.error("no product produced data")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

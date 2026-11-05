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

from . import boundaries, render
from .config import INTERVALS, PRODUCTS, VARIABLES
from .fields import FieldStore

log = logging.getLogger("rtma_viewer")
SITE_DIR = Path(__file__).resolve().parent.parent / "site"
iso = lambda t: t.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_product(product, client, now, out: Path, cache: Path, layers, dpi: int, colors: int) -> dict:
    t0 = time.time()
    store = FieldStore(product, client, cache)
    cycle = store.latest_cycle(now)
    rec = {"label": product.label, "long_name": product.long_name, "blurb": product.blurb,
           "status": "unavailable", "cycle": None, "panels": {}}
    if cycle is None:
        log.error("%s: no cycle found in the last %d h", product.key, product.lookback_hours)
        return rec
    log.info("%s: newest cycle %s", product.key, cycle)

    jobs = [(cycle - timedelta(hours=h), var) for var in VARIABLES.values() for h in INTERVALS]
    fields = store.load(jobs)
    ctx = render.MapContext.build(store.grid, layers)

    total_png = 0
    for var in VARIABLES.values():
        rec["panels"][var.key] = {}
        cur = fields[(cycle, var.key)]
        for h in INTERVALS:
            ref_t = cycle - timedelta(hours=h)
            prev = fields.get((ref_t, var.key))
            if cur is None or prev is None:
                rec["panels"][var.key][str(h)] = None
                log.warning("%s %s +%dh unavailable", product.key, var.key, h)
                continue
            data = cur if h == 0 else cur - prev
            style, signed = render.style_and_flags(var, h)
            texts = render.panel_text(product, var, h, cycle, ref_t if h else None)
            rel = f"img/{product.key}_{var.key}_{h}h.png"
            stats = render.region_stats(ctx, data)
            size = render.render_panel(ctx, data, style, texts=texts, signed=signed, out=out / rel,
                                       dpi=dpi, colors=colors)
            total_png += size
            rec["panels"][var.key][str(h)] = {"file": rel, "ref": iso(ref_t) if h else None, **stats,
                                              "stations": render.station_values(ctx, data)}
    rec.update(status="ok", cycle=iso(cycle), lag_minutes=int((now - cycle).total_seconds() // 60))
    rec["bytes"] = {"png": total_png, "downloaded": store.src.bytes_downloaded,
                    "s3_requests": store.src.requests, **dict(store.stats)}
    store.prune(keep_hours=max(INTERVALS) + 4, now=cycle)
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
                "variables": [{"key": v.key, "label": v.label, "long_name": v.long_name, "unit": v.unit}
                              for v in VARIABLES.values()],
                "products": {}}
    for key in args.products.split(","):
        manifest["products"][key] = build_product(PRODUCTS[key], client, now, out, cache, layers, args.dpi, args.colors)

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

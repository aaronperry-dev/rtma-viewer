"""Field store: S3 byte-range fetch -> GRIB decode -> New England window -> small on-disk cache.

Re-runs only download what is new: each hour the previous run's fields are found in the
cache (restored by GitHub Actions), so a run normally fetches just the newest cycle.
"""
from __future__ import annotations

import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import grib
from .config import Product, Variable
from .s3 import S3Source, find_entry

log = logging.getLogger(__name__)


class FieldStore:
    def __init__(self, product: Product, client, cache_dir: str | Path, workers: int = 8):
        self.product = product
        self.src = S3Source(client, product)
        self.dir = Path(cache_dir) / product.key
        self.grid: grib.Grid | None = grib.Grid.load(self.dir / "grid.npz")
        self.pcp_grid: grib.Grid | None = grib.Grid.load(self.dir / "pcp_grid.npz")   # same lattice, but NOAA crops it differently
        self.workers = workers
        self.stats = Counter()

    # ---------------------------------------------------------------- cycles
    def latest_cycle(self, now: datetime) -> datetime | None:
        now = now.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
        for back in range(self.product.lookback_hours + 1):
            cyc = now - timedelta(hours=back)
            if self.src.locate(cyc):
                return cyc
        return None

    # ---------------------------------------------------------------- cache
    def _path(self, cycle: datetime, var: Variable) -> Path:
        if var.static:                                       # time-invariant: one file, never pruned
            return self.dir / "static" / f"{var.key}.npy"
        return self.dir / f"{cycle:%Y%m%d%H}_{var.key}.npy"

    @staticmethod
    def _encode(var: Variable, disp: np.ndarray) -> np.ndarray:
        """Display-unit field -> compact cache dtype.  float16 (<=0.06 units) unless the variable needs more
        precision than that: pressure is packed as uint16 at 0.01 mb (float16 would be 0.5-1 mb at ~1000 mb)."""
        if var.static:
            return disp.astype(np.float32)
        if var.pack is None:
            return disp.astype(np.float16)
        offset, scale = var.pack
        with np.errstate(invalid="ignore"):
            q = np.round((disp.astype(np.float64) - offset) * scale)
        bad = ~np.isfinite(q) | (q < 0) | (q > 65534)
        return np.where(bad, 65535, q).astype(np.uint16)       # 65535 = missing

    @staticmethod
    def _decode(var: Variable, arr: np.ndarray) -> np.ndarray:
        if var.pack is None or arr.dtype != np.uint16:
            return arr.astype(np.float32)
        offset, scale = var.pack
        out = arr.astype(np.float32) / np.float32(scale) + np.float32(offset)
        out[arr == 65535] = np.nan
        return out

    def prune(self, keep_hours: int, now: datetime, keep_by_var: dict[str, int] | None = None):
        """Delete cached cycles older than ``keep_hours`` (per variable key if given in ``keep_by_var``, so variables
        that only ever need the newest cycle do not pile up 28 hours of files)."""
        for p in self.dir.glob("??????????_*.npy"):
            try:
                key = p.stem.split("_", 1)[1]
                cutoff = now - timedelta(hours=(keep_by_var or {}).get(key, keep_hours))
                if datetime.strptime(p.name[:10], "%Y%m%d%H").replace(tzinfo=timezone.utc) < cutoff:
                    p.unlink()
            except ValueError:
                pass

    # ---------------------------------------------------------------- loading
    def load(self, jobs: list[tuple[datetime, Variable]], now: datetime | None = None
             ) -> dict[tuple[datetime, str], np.ndarray | None]:
        """Return display-unit fields (float32, window-shaped) for each (cycle, variable); None if unavailable.

        ``now`` (UTC) enables the re-download of recent cached precipitation (``Product.pcp_refresh_hours``): the hourly
        precipitation analyses are re-made for a while after they are first published."""
        out: dict[tuple[datetime, str], np.ndarray | None] = {}
        todo: list[tuple[datetime, Variable]] = []
        stale: dict[tuple[datetime, str], np.ndarray] = {}        # cached values to fall back on if a refresh finds nothing
        for cyc, var in dict.fromkeys(jobs):
            p = self._path(cyc, var)
            if self.grid is not None and p.exists():
                try:
                    cached = self._decode(var, np.load(p))
                except Exception:
                    p.unlink(missing_ok=True)
                else:
                    if (var.source == "pcp" and now is not None
                            and now - cyc < timedelta(hours=self.product.pcp_refresh_hours)):
                        stale[(cyc, var.key)] = cached
                        todo.append((cyc, var))
                        continue
                    out[(cyc, var.key)] = cached
                    self.stats["cache_hits"] += 1
                    continue
            todo.append((cyc, var))
        pcp_todo = [(c, v) for c, v in todo if v.source == "pcp"]
        todo = [(c, v) for c, v in todo if v.source != "pcp"]

        # 1) which messages do we need?  (cycle, grib var, level)
        wanted: dict[tuple[datetime, str, str], None] = {}
        plan: dict[tuple[datetime, str], list[tuple[datetime, str, str]]] = {}
        for cyc, var in todo:
            loc = self.src.locate(cyc)
            if loc is None:
                out[(cyc, var.key)] = None
                continue
            if find_entry(loc.entries, var.grib_var, var.grib_level):
                parts = [(cyc, var.grib_var, var.grib_level)]
            elif var.components and all(find_entry(loc.entries, c, var.grib_level) for c in var.components):
                parts = [(cyc, c, var.grib_level) for c in var.components]
            else:
                log.warning("%s %s: %s %s not in idx", self.product.key, cyc, var.grib_var, var.grib_level)
                out[(cyc, var.key)] = None
                continue
            plan[(cyc, var.key)] = parts
            wanted.update(dict.fromkeys(parts))

        # 2) download the byte ranges concurrently
        def fetch(item):
            cyc, v, lvl = item
            return item, self.src.fetch_message(self.src.locate(cyc), v, lvl)

        blobs: dict = {}
        if wanted:
            with ThreadPoolExecutor(self.workers) as pool:
                for item, data in pool.map(fetch, list(wanted)):
                    blobs[item] = data
            self.stats["messages_fetched"] += len(blobs)

        # 3) decode (single thread: eccodes is not thread-safe), window, convert, cache
        raw: dict = {}
        for item in list(blobs):
            data = blobs.pop(item)                              # release each download once decoded
            if data is None:
                raw[item] = None
                continue
            vals, geo = grib.decode(data, want_geometry=self.grid is None)
            if self.grid is None and geo is not None:
                self.grid = grib.make_window(*geo)
                self.grid.save(self.dir / "grid.npz")
            raw[item] = vals[self.grid.slices].copy()           # a copy: a view would keep every full CONUS array alive

        for cyc, var in todo:
            parts = plan.get((cyc, var.key))
            if parts is None:
                continue
            arrays = [raw.get(p) for p in parts]
            if any(a is None for a in arrays):
                out[(cyc, var.key)] = None
                continue
            si = arrays[0] if len(arrays) == 1 else np.hypot(*arrays)
            stored = self._encode(var, var.convert(si))        # compact on disk (float16 / packed uint16 / static float32)
            path = self._path(cyc, var)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, stored)
            out[(cyc, var.key)] = self._decode(var, stored)    # same values a later cache hit will return

        self._load_precip(pcp_todo, out, stale)             # after the analysis fields: it needs their grid to check its own
        return out

    # ---------------------------------------------------------------- hourly precipitation files
    def _pcp_window(self, vals: np.ndarray, geo) -> np.ndarray | None:
        """Cut the New England window out of a precipitation grid.  The precipitation files are on the same 2.5 km lattice as
        the analysis files but cropped differently, so the window has its own indices; it is found from the coordinates once and
        checked against the analysis window - a different lattice would need resampling, which we refuse to fake."""
        if self.grid is None:
            return None
        if self.pcp_grid is None and geo is not None:
            cand = grib.make_window(*geo)
            ok = (cand.lat.shape == self.grid.lat.shape and np.abs(cand.lat - self.grid.lat).max() < 5e-4
                  and np.abs(cand.lon - self.grid.lon).max() < 5e-4)
            if not ok:
                log.error("%s: precipitation grid does not match the analysis grid; skipping precipitation", self.product.key)
                return None
            self.pcp_grid = cand
            self.pcp_grid.save(self.dir / "pcp_grid.npz")
        if self.pcp_grid is None:
            return None
        win = vals[self.pcp_grid.slices]
        return win if win.shape == self.grid.lat.shape else None

    def _load_precip(self, todo, out, stale):
        if not todo:
            return
        with ThreadPoolExecutor(self.workers) as pool:
            blobs = list(pool.map(lambda cv: self.src.fetch_pcp(cv[0]), todo))
        self.stats["pcp_files_fetched"] += sum(b is not None for b in blobs)
        for i, (cyc, var) in enumerate(todo):
            blob, blobs[i] = blobs[i], None                 # release each file once decoded
            key = (cyc, var.key)
            if blob is None:                                # not published: keep what the cache had, if anything
                out[key] = stale.get(key)
                continue
            vals, geo = grib.decode(blob, want_geometry=self.pcp_grid is None)
            win = self._pcp_window(vals, geo)
            if win is None:
                out[key] = stale.get(key)
                continue
            stored = self._encode(var, var.convert(win))
            path = self._path(cyc, var)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, stored)
            out[key] = self._decode(var, stored)

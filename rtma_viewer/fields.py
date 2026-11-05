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
        return self.dir / f"{cycle:%Y%m%d%H}_{var.key}.npy"

    def prune(self, keep_hours: int, now: datetime):
        cutoff = now - timedelta(hours=keep_hours)
        for p in self.dir.glob("??????????_*.npy"):
            try:
                if datetime.strptime(p.name[:10], "%Y%m%d%H").replace(tzinfo=timezone.utc) < cutoff:
                    p.unlink()
            except ValueError:
                pass

    # ---------------------------------------------------------------- loading
    def load(self, jobs: list[tuple[datetime, Variable]]) -> dict[tuple[datetime, str], np.ndarray | None]:
        """Return display-unit fields (float32, window-shaped) for each (cycle, variable); None if unavailable."""
        out: dict[tuple[datetime, str], np.ndarray | None] = {}
        todo: list[tuple[datetime, Variable]] = []
        for cyc, var in dict.fromkeys(jobs):
            p = self._path(cyc, var)
            if self.grid is not None and p.exists():
                try:
                    out[(cyc, var.key)] = np.load(p).astype(np.float32)
                    self.stats["cache_hits"] += 1
                    continue
                except Exception:
                    p.unlink(missing_ok=True)
            todo.append((cyc, var))

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
        for item, data in blobs.items():
            if data is None:
                raw[item] = None
                continue
            vals, geo = grib.decode(data, want_geometry=self.grid is None)
            if self.grid is None and geo is not None:
                self.grid = grib.make_window(*geo)
                self.grid.save(self.dir / "grid.npz")
            raw[item] = vals[self.grid.slices]

        for cyc, var in todo:
            parts = plan.get((cyc, var.key))
            if parts is None:
                continue
            arrays = [raw.get(p) for p in parts]
            if any(a is None for a in arrays):
                out[(cyc, var.key)] = None
                continue
            si = arrays[0] if len(arrays) == 1 else np.hypot(*arrays)
            disp = var.convert(si).astype(np.float16)          # float16 is plenty (<=0.06 units) and halves the cache
            self.dir.mkdir(parents=True, exist_ok=True)
            np.save(self._path(cyc, var), disp)
            out[(cyc, var.key)] = disp.astype(np.float32)
        return out

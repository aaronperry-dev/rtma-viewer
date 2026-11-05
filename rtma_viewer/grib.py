"""GRIB2 decoding (eccodes, straight from memory) and the New England sub-grid."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DATA_PAD_DEG, DISPLAY_BBOX


@dataclass
class Grid:
    """Geometry of the sub-grid we keep (a window of the national 2.5 km grid)."""
    lat: np.ndarray            # (ny, nx) float32
    lon: np.ndarray            # (ny, nx) float32, -180..180
    window: tuple[int, int, int, int]   # r0, r1, c0, c1 into the full grid

    @property
    def slices(self):
        r0, r1, c0, c1 = self.window
        return slice(r0, r1), slice(c0, c1)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, lat=self.lat, lon=self.lon, window=np.array(self.window))

    @classmethod
    def load(cls, path: Path) -> "Grid | None":
        if not path.exists():
            return None
        try:
            z = np.load(path)
            return cls(z["lat"], z["lon"], tuple(int(v) for v in z["window"]))
        except Exception:
            return None


def decode(msg: bytes, want_geometry: bool = False):
    """Decode one GRIB2 message.  Returns ``(values[ny, nx] float64 with NaN for missing, (lat, lon) | None)``."""
    import eccodes as ec

    gid = ec.codes_new_from_message(msg)
    try:
        try:
            nx, ny = ec.codes_get(gid, "Nx"), ec.codes_get(gid, "Ny")
        except Exception:
            nx, ny = ec.codes_get(gid, "Ni"), ec.codes_get(gid, "Nj")
        vals = ec.codes_get_array(gid, "values").astype(np.float64)
        missing = ec.codes_get(gid, "missingValue")
        vals[vals == missing] = np.nan
        geo = None
        if want_geometry:
            lat = ec.codes_get_array(gid, "latitudes").reshape(ny, nx)
            lon = ec.codes_get_array(gid, "longitudes").reshape(ny, nx)
            lon = np.where(lon > 180, lon - 360, lon)
            geo = (lat, lon)
        return vals.reshape(ny, nx), geo
    finally:
        ec.codes_release(gid)


def make_window(lat: np.ndarray, lon: np.ndarray) -> Grid:
    """Smallest rectangular window of the grid covering the display box plus padding."""
    lon0, lon1, lat0, lat1 = DISPLAY_BBOX
    p = DATA_PAD_DEG
    m = (lat >= lat0 - p) & (lat <= lat1 + p) & (lon >= lon0 - p) & (lon <= lon1 + p)
    if not m.any():
        raise ValueError("grid does not overlap New England")
    rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    return Grid(lat[r0:r1, c0:c1].astype(np.float32), lon[r0:r1, c0:c1].astype(np.float32), (r0, r1, c0, c1))

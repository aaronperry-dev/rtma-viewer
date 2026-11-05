"""Offline stand-in for the NOAA S3 buckets (``--demo`` mode and the test-suite).

It serves *real* GRIB2 messages on a Lambert-conformal grid with the same projection and
2.5 km spacing as the NDFD CONUS grid (but a smaller extent), a correct ``.idx`` sidecar, and
honours ``Range:`` requests - so the production fetch/decode/render code runs unchanged.
The meteorology is synthetic (a moving cold front + noise): it exists to exercise the pipeline.
"""
from __future__ import annotations

import io
import re
from collections import OrderedDict
from datetime import datetime, timedelta, timezone

import numpy as np

R_EARTH = 6371229.0
DX = 2539.703
EPOCH = datetime(2026, 11, 5, tzinfo=timezone.utc)

MESSAGES = [("PRES", "surface"), ("TMP", "2 m above ground"), ("DPT", "2 m above ground"),
            ("UGRD", "10 m above ground"), ("VGRD", "10 m above ground"),
            ("WIND", "10 m above ground"), ("GUST", "10 m above ground")]


def _noise(shape, seed, sigma):
    n = np.random.default_rng(seed).standard_normal(shape)
    f = np.fft.rfft2(n)
    ky, kx = np.fft.fftfreq(shape[0])[:, None], np.fft.rfftfreq(shape[1])[None, :]
    out = np.fft.irfft2(f * np.exp(-2 * (np.pi * sigma) ** 2 * (kx ** 2 + ky ** 2)), s=shape)
    return out / out.std()


class DemoGrid:
    def __init__(self, nx=820, ny=660, centre=(-71.4, 44.2)):
        import pyproj
        self.nx, self.ny = nx, ny
        self.proj = pyproj.Proj(proj="lcc", lat_1=25, lat_2=25, lat_0=25, lon_0=-95, a=R_EARTH, b=R_EARTH)
        xc, yc = self.proj(*centre)
        self.x0, self.y0 = xc - nx / 2 * DX, yc - ny / 2 * DX
        xs, ys = self.x0 + np.arange(nx) * DX, self.y0 + np.arange(ny) * DX
        X, Y = np.meshgrid(xs, ys)
        self.lon, self.lat = self.proj(X, Y, inverse=True)
        self.xkm, self.ykm = (X - xc) / 1000, (Y - yc) / 1000
        shape = (ny, nx)
        self.n = [_noise(shape, s, sg) for s, sg in ((1, 6), (2, 10), (3, 5), (4, 4))]
        self.first = self.proj(self.x0, self.y0, inverse=True)   # lon, lat of grid point (0, 0)

    def fields_si(self, cycle: datetime) -> dict[str, np.ndarray]:
        th = (cycle - EPOCH).total_seconds() / 3600
        local = (cycle.hour - 5) % 24
        diurnal = np.cos(2 * np.pi * (local - 15) / 24)
        s = (self.xkm - (-250 + 40 * (th - 12))) / 90.0           # front moving east at 40 km/h
        warm = (1 + np.tanh(s)) / 2
        t_f = 46 + 6 * diurnal - 2.0 * (self.lat - 44.2) + 2.5 * self.n[0] - 11 * (1 - warm)
        td_f = np.minimum(36 + 9 * warm + 2.5 * self.n[1] - 0.12 * (self.lat - 44.2) * 5, t_f - 0.8)
        wind = np.clip(7 + 13 * np.exp(-s ** 2 / 1.5) + 3 * self.n[2] + 2 * warm, 0.3, None)
        gust = wind * (1.3 + 0.1 * np.abs(self.n[3])) + 3
        dirn = np.radians(250 + 25 * self.n[3])
        return {
            "PRES": np.full_like(t_f, 101325.0),
            "TMP": (t_f - 32) / 1.8 + 273.15, "DPT": (td_f - 32) / 1.8 + 273.15,
            "WIND": wind / 2.2369362921, "GUST": gust / 2.2369362921,
            "UGRD": -wind / 2.2369362921 * np.sin(dirn), "VGRD": -wind / 2.2369362921 * np.cos(dirn),
        }

    def message(self, values: np.ndarray, cycle: datetime) -> bytes:
        import eccodes as ec
        g = ec.codes_grib_new_from_samples("GRIB2")
        ec.codes_set(g, "gridDefinitionTemplateNumber", 30)
        keys = dict(shapeOfTheEarth=6, Nx=self.nx, Ny=self.ny,
                    latitudeOfFirstGridPointInDegrees=self.first[1],
                    longitudeOfFirstGridPointInDegrees=self.first[0] % 360,
                    LoVInDegrees=265.0, LaDInDegrees=25.0, Latin1InDegrees=25.0, Latin2InDegrees=25.0,
                    DxInMetres=DX, DyInMetres=DX, latitudeOfSouthernPoleInDegrees=-90.0,
                    longitudeOfSouthernPoleInDegrees=0.0, iScansNegatively=0, jScansPositively=1,
                    dataDate=int(f"{cycle:%Y%m%d}"), dataTime=cycle.hour * 100,
                    packingType="grid_simple", bitsPerValue=14)
        for k, v in keys.items():
            ec.codes_set(g, k, v)
        ec.codes_set_values(g, values.ravel().astype(np.float64))
        try:
            return ec.codes_get_message(g)
        finally:
            ec.codes_release(g)


class FakeS3:
    """Implements the three boto3 calls the pipeline uses."""
    _KEY = re.compile(r"^(?P<p>[a-z0-9]+)\.(?P<d>\d{8})/(?P=p)\.t(?P<h>\d{2})z\.2dvaranl_ndfd\.grb2_wexp(?P<idx>\.idx)?$")

    def __init__(self, now: datetime, urma_lag_hours: int = 3, missing: set[tuple[str, int]] | None = None):
        self.now = now.replace(minute=0, second=0, microsecond=0)
        self.lag = urma_lag_hours
        self.missing = missing or set()     # {(product_prefix, hours_before_now)} that were "never published"
        self.grid = DemoGrid()
        self._files: OrderedDict = OrderedDict()
        self.range_requests: list[tuple[str, str | None]] = []

    def _cycle_ok(self, prefix: str, cycle: datetime) -> bool:
        lag = self.lag if prefix.startswith("urma") else 0
        hrs_before = int((self.now - cycle).total_seconds() // 3600)
        return hrs_before >= lag and (prefix, hrs_before) not in self.missing

    def _file(self, prefix: str, cycle: datetime):
        key = (prefix, cycle)
        if key not in self._files:
            si = self.grid.fields_si(cycle)
            blob, idx, off = b"", [], 0
            for i, (var, level) in enumerate(MESSAGES, 1):
                m = self.grid.message(si[var], cycle)
                idx.append(f"{i}:{off}:d={cycle:%Y%m%d%H}:{var}:{level}:anl:")
                blob += m
                off += len(m)
            self._files[key] = (blob, "\n".join(idx) + "\n")
            while len(self._files) > 3:
                self._files.popitem(last=False)
        return self._files[key]

    def list_objects_v2(self, Bucket, Prefix, MaxKeys=1000, **_):
        m = re.match(r"^([a-z0-9]+)\.(\d{8})/\1\.t(\d{2})z\.2dvaranl_ndfd$", Prefix)
        if not m:
            return {}
        cyc = datetime.strptime(m[2] + m[3], "%Y%m%d%H").replace(tzinfo=timezone.utc)
        if not self._cycle_ok(m[1], cyc):
            return {}
        return {"Contents": [{"Key": Prefix + ".grb2_wexp"}, {"Key": Prefix + ".grb2_wexp.idx"}]}

    def get_object(self, Bucket, Key, Range=None, **_):
        m = self._KEY.match(Key)
        if not m:
            raise KeyError(Key)
        cyc = datetime.strptime(m["d"] + m["h"], "%Y%m%d%H").replace(tzinfo=timezone.utc)
        if not self._cycle_ok(m["p"], cyc):
            raise KeyError(Key)
        blob, idx = self._file(m["p"], cyc)
        data = idx.encode() if m["idx"] else blob
        if not m["idx"]:
            self.range_requests.append((Key, Range))
        if Range:
            a, b = re.match(r"bytes=(\d+)-(\d*)", Range).groups()
            data = data[int(a): (int(b) + 1 if b else None)]
        return {"Body": io.BytesIO(data)}

"""Anonymous access to the NOAA RTMA/URMA open-data buckets.

Only the bytes of the GRIB2 messages we need are downloaded: every GRIB file has a
sidecar ``.idx`` listing the start byte of each message, so a single variable is a
single ``Range:`` request (~1-2 MB) instead of the whole ~100 MB file.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime

from .config import Product

log = logging.getLogger(__name__)


def make_client():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client(
        "s3",
        region_name="us-east-1",
        config=Config(signature_version=UNSIGNED, retries={"max_attempts": 6, "mode": "standard"},
                      max_pool_connections=16, connect_timeout=15, read_timeout=60),
    )


# ------------------------------------------------------------------------- .idx parsing
@dataclass(frozen=True)
class IdxEntry:
    num: int
    start: int
    end: int | None   # inclusive; None = to end of file
    date: str
    var: str
    level: str


def parse_idx(text: str) -> list[IdxEntry]:
    """Parse wgrib2-style inventory lines: ``num:start:d=YYYYMMDDHH:VAR:level:anl:``."""
    rows = []
    for line in text.splitlines():
        parts = line.strip().split(":")
        if len(parts) < 5 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        rows.append((int(parts[0]), int(parts[1]), parts[2][2:], parts[3], parts[4]))
    out = []
    for i, (num, start, date, var, level) in enumerate(rows):
        end = rows[i + 1][1] - 1 if i + 1 < len(rows) else None
        out.append(IdxEntry(num, start, end, date, var, level))
    return out


def find_entry(entries: list[IdxEntry], var: str, level: str) -> IdxEntry | None:
    for e in entries:
        if e.var == var and e.level == level:
            return e
    return None


# ------------------------------------------------------------------------- source
@dataclass
class Located:
    key: str
    entries: list[IdxEntry]


class S3Source:
    def __init__(self, client, product: Product):
        self.client = client
        self.product = product
        self.bytes_downloaded = 0
        self.requests = 0
        self._located: dict[datetime, Located | None] = {}
        self._lock = threading.Lock()

    def prefix_for(self, cycle: datetime) -> str:
        p = self.product.prefix
        return f"{p}.{cycle:%Y%m%d}/{p}.t{cycle:%H}z.2dvaranl_ndfd"

    def locate(self, cycle: datetime) -> Located | None:
        """Find the CONUS 2-D analysis file (+ its idx) for a cycle, or None if not published (yet)."""
        if cycle in self._located:
            return self._located[cycle]
        result = None
        try:
            resp = self.client.list_objects_v2(Bucket=self.product.bucket, Prefix=self.prefix_for(cycle), MaxKeys=20)
            keys = {o["Key"] for o in resp.get("Contents", [])}
            # file name varies slightly between NCEP releases (…grb2 / …grb2_wexp): try every candidate that has an idx
            candidates = sorted((k for k in keys if not k.endswith(".idx") and k + ".idx" in keys),
                                key=lambda k: (not k.endswith("_wexp"), k))
            for key in candidates:
                body = self._get(key + ".idx").decode("utf-8", "replace")
                entries = parse_idx(body)
                if entries:
                    result = Located(key, entries)
                    break
        except Exception as exc:  # network / permissions: treat as "not available" but say why
            log.warning("locate %s %s failed: %s", self.product.key, cycle, exc)
        self._located[cycle] = result
        return result

    def _get(self, key: str, byte_range: str | None = None) -> bytes:
        kwargs = {"Bucket": self.product.bucket, "Key": key}
        if byte_range:
            kwargs["Range"] = byte_range
        data = self.client.get_object(**kwargs)["Body"].read()
        with self._lock:
            self.bytes_downloaded += len(data)
            self.requests += 1
        return data

    def fetch_message(self, loc: Located, var: str, level: str) -> bytes | None:
        """Download exactly one GRIB2 message with an HTTP byte-range request."""
        entry = find_entry(loc.entries, var, level)
        if entry is None:
            return None
        rng = f"bytes={entry.start}-{entry.end}" if entry.end is not None else f"bytes={entry.start}-"
        data = self._get(loc.key, rng)
        if data[:4] != b"GRIB":
            raise ValueError(f"byte range {rng} of {loc.key} did not return a GRIB message")
        return data

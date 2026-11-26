"""Surface analysis features (fronts, troughs, highs, lows) and the WPC coded surface bulletin that supplies them.

The NWS Weather Prediction Center draws the surface analysis by hand every 3 hours (00, 03 ... 21Z) and publishes it as the
"high resolution coded surface bulletin" (CODSUS).  Format (https://www.wpc.ncep.noaa.gov/html/read_coded_bull_hrtxt.html):

    VALID 112612Z                              month, day, hour (UTC)
    HIGHS 1033 4480946 1033 4421107 ...        central pressure (mb) + 7-digit position: lat in tenths of a degree N (3 digits),
    LOWS 998 3480777 ...                       lon in tenths of a degree W (4 digits)  ->  44.8 N 94.6 W
    COLD 6050826 5870900 5781026 ...           a front or trough: a run of 7-digit points, in order
    TROF / WARM / STNRY / OCFNT                trough (also outflow boundary, squall line, dry line) / warm / stationary / occluded

Lists wrap onto following lines without repeating the keyword, so the text is read as one stream of tokens.  An older bulletin
(1-degree, 4-5 digit positions) and optional strength words (WK / MDT / STG after a front) are tolerated.

Only the newest bulletin is kept on the NWS server, so ``Archive`` saves each one the first time it is seen; a map for an earlier
valid time then finds the analysis that was current at that time.
"""
from __future__ import annotations

import logging
import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

URLS = ("https://tgftp.nws.noaa.gov/data/raw/as/asus02.kwbc.cod.sus.txt",
        "https://www.wpc.ncep.noaa.gov/discussions/codsus_hr")
MAX_BACK = timedelta(hours=3)             # the bulletin appears about 1.5 h after its valid time: for the newest maps, fall back to the last one
MAX_GAP = timedelta(minutes=90)          # an analysis this close to the map's valid time (the 3-hourly cadence's half) is "current"
KEEP = timedelta(days=7)
KINDS = {"COLD": "cold", "WARM": "warm", "STNRY": "stationary", "OCFNT": "occluded", "TROF": "trough"}
_STRENGTH = {"WK", "MDT", "STG"}


@dataclass(frozen=True)
class Center:
    kind: str                     # "H" or "L"
    lat: float
    lon: float
    pressure: float               # mb


@dataclass(frozen=True)
class Front:
    kind: str                     # cold | warm | stationary | occluded | trough
    points: tuple                 # ((lat, lon), ...) in order


@dataclass(frozen=True)
class Analysis:
    valid: datetime
    source: str                   # "WPC" or "RTMA"
    centers: tuple
    fronts: tuple

    def highs(self):
        return [c for c in self.centers if c.kind == "H"]

    def lows(self):
        return [c for c in self.centers if c.kind == "L"]


def _position(tok: str) -> tuple[float, float] | None:
    """(lat, lon) of a 7-digit (tenths) or 4/5-digit (whole degrees) position token; longitude in degrees east."""
    if not tok.isdigit():
        return None
    if len(tok) == 7:
        lat, lon_w = int(tok[:3]) / 10.0, int(tok[3:]) / 10.0
    elif len(tok) in (4, 5):
        lat, lon_w = float(int(tok[:2])), float(int(tok[2:]))
    else:
        return None
    lon = -lon_w if lon_w <= 180.0 else 360.0 - lon_w
    return lat, lon


def _valid_time(text: str) -> datetime:
    """UTC valid time of the analysis: the VALID line has month, day and hour only, so the year comes from the issue date in the
    header (a bulletin issued on 1 Jan can be valid on 31 Dec)."""
    m = re.search(r"VALID\s+(\d{2})(\d{2})(\d{2})Z", text)
    if not m:
        raise ValueError("no VALID line")
    month, day, hour = map(int, m.groups())
    d = re.search(r"\b([A-Z]{3})\s+(\d{1,2})\s+(\d{4})\b", text)
    try:
        ref = datetime.strptime(" ".join(d.groups()), "%b %d %Y").replace(tzinfo=timezone.utc) if d else None
    except ValueError:
        ref = None
    ref = ref or datetime.now(timezone.utc)
    cands = []
    for year in (ref.year - 1, ref.year, ref.year + 1):
        try:
            cands.append(datetime(year, month, day, hour, tzinfo=timezone.utc))
        except ValueError:
            pass
    if not cands:
        raise ValueError("bad VALID date")
    return min(cands, key=lambda c: abs(c - ref))


def parse(text: str) -> Analysis:
    """Decode a coded surface bulletin.  Raises ValueError if it has no VALID line."""
    valid = _valid_time(text)
    body = text.split("VALID", 1)[1]
    body = body.split("$$", 1)[0]
    tokens = body.split()[1:]                                                 # drop the "112612Z" token
    centers: list[Center] = []
    fronts: list[Front] = []
    i = 0
    while i < len(tokens):
        word = tokens[i]
        if word in ("HIGHS", "LOWS"):
            i += 1
            while i + 1 < len(tokens) and tokens[i].isdigit() and tokens[i + 1].isdigit() and tokens[i] not in KINDS:
                pos = _position(tokens[i + 1])
                if pos is None or len(tokens[i]) not in (3, 4):
                    break
                centers.append(Center("H" if word == "HIGHS" else "L", pos[0], pos[1], float(tokens[i])))
                i += 2
        elif word in KINDS:
            i += 1
            pts = []
            while i < len(tokens) and (tokens[i].isdigit() or tokens[i] in _STRENGTH):
                if tokens[i] not in _STRENGTH:
                    pos = _position(tokens[i])
                    if pos is not None:
                        pts.append(pos)
                i += 1
            if len(pts) >= 2:
                fronts.append(Front(KINDS[word], tuple(pts)))
        else:
            i += 1
    return Analysis(valid, "WPC", tuple(centers), tuple(fronts))


def fetch_text(timeout: float = 20.0) -> str | None:
    """Latest bulletin text from the NWS servers (first URL that answers), or None."""
    for url in URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "rtma-viewer (github pages)"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                text = r.read().decode("latin1")
            if "VALID" in text:
                return text
        except Exception as exc:                                             # network trouble: fronts are optional
            log.warning("WPC bulletin %s: %s", url, exc)
    return None


class Archive:
    """The bulletins seen so far, one small text file per analysis time."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)

    def update(self, fetch: Callable[[], str | None] = fetch_text) -> Analysis | None:
        text = fetch()
        if not text:
            return None
        try:
            an = parse(text)
        except ValueError as exc:
            log.warning("WPC bulletin not understood: %s", exc)
            return None
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{an.valid:%Y%m%d%H}.txt"
        if not path.exists():
            path.write_text(text)
        for old in self.dir.glob("??????????.txt"):                          # keep a week
            try:
                if datetime.strptime(old.stem, "%Y%m%d%H").replace(tzinfo=timezone.utc) < an.valid - KEEP:
                    old.unlink()
            except ValueError:
                pass
        return an

    def nearest(self, valid: datetime, max_gap: timedelta = MAX_GAP, max_back: timedelta = timedelta(0)) -> Analysis | None:
        """The stored analysis closest to ``valid`` (within ``max_gap``), or failing that the newest one valid up to ``max_back``
        before it (the newest map is made before WPC has published its time), or None."""
        best, best_gap, older, older_t = None, None, None, None
        for path in self.dir.glob("??????????.txt"):
            try:
                t = datetime.strptime(path.stem, "%Y%m%d%H").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            gap = abs(t - valid)
            if gap <= max_gap and (best_gap is None or gap < best_gap):
                best, best_gap = path, gap
            if valid - max_back <= t <= valid and (older_t is None or t > older_t):
                older, older_t = path, t
        if best is None:
            best = older
        if best is None:
            return None
        try:
            return parse(best.read_text())
        except ValueError:
            return None

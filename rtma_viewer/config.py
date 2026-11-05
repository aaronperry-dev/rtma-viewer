"""Central configuration: products, variables, region, cities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .derived import apparent_temperature


# --------------------------------------------------------------------------- products
@dataclass(frozen=True)
class Product:
    key: str
    label: str
    long_name: str
    bucket: str          # public NOAA Open Data bucket (anonymous access)
    prefix: str          # e.g. "rtma2p5" -> rtma2p5.YYYYMMDD/rtma2p5.tHHz.2dvaranl_ndfd.grb2*
    lookback_hours: int  # how far back to search for the newest published cycle
    blurb: str


PRODUCTS: dict[str, Product] = {
    "rtma": Product(
        "rtma", "RTMA", "Real-Time Mesoscale Analysis", "noaa-rtma-pds", "rtma2p5", 6,
        "NOAA/NCEP 2.5 km hourly surface analysis. Available ~35-45 min after the valid hour.",
    ),
    "urma": Product(
        "urma", "URMA", "UnRestricted Mesoscale Analysis", "noaa-urma-pds", "urma2p5", 12,
        "Same 2.5 km grid as RTMA, but run later with more observations. Lags RTMA by a few hours.",
    ),
}

# --------------------------------------------------------------------------- variables
def _k_to_f(a):
    return (a - 273.15) * 1.8 + 32.0


def _ms_to_mph(a):
    return a * 2.2369362921


@dataclass(frozen=True)
class Variable:
    key: str
    label: str            # short label for tabs
    long_name: str        # plot title
    unit: str             # display unit
    grib_var: str         # abbreviation in the .idx file
    grib_level: str       # level string in the .idx file
    convert: Callable = field(compare=False, default=lambda a: a)
    components: tuple[str, ...] = ()   # fallback: hypot of these 10 m components if grib_var is absent
    requires: tuple[str, ...] = ()     # derived variable: keys of the source variables it is computed from ...
    derive: Callable | None = field(compare=False, default=None)   # ... as derive(*fields) in display units


VARIABLES: dict[str, Variable] = {
    "temp": Variable("temp", "Temperature", "2m Temperature", "°F", "TMP", "2 m above ground", _k_to_f),
    "dewpoint": Variable("dewpoint", "Dew point", "2m Dew Point", "°F", "DPT", "2 m above ground", _k_to_f),
    "wind": Variable("wind", "Wind speed", "10m Wind Speed", "mph", "WIND", "10 m above ground", _ms_to_mph,
                     components=("UGRD", "VGRD")),
    "gust": Variable("gust", "Wind gust", "10m Wind Gust", "mph", "GUST", "10 m above ground", _ms_to_mph),
    # derived (no GRIB message of its own): NDFD apparent temperature = wind chill / ambient / heat index
    "apparent": Variable("apparent", "Apparent temp", "Apparent Temperature", "°F", "", "",
                         requires=("temp", "dewpoint", "wind"), derive=apparent_temperature),
}
SOURCE_VARIABLES: dict[str, Variable] = {k: v for k, v in VARIABLES.items() if v.derive is None}

# Current field + change versus N hours earlier
INTERVALS: tuple[int, ...] = (0, 1, 3, 6, 12, 24)

# --------------------------------------------------------------------------- region
# lon_min, lon_max, lat_min, lat_max that bounds the (landscape) map frame defined in render.py
# (centred on New England, reaching west over New York and east over the Gulf of Maine / Maritimes).  render.py asserts the frame fits inside.
DISPLAY_BBOX = (-80.7, -62.2, 39.8, 48.3)
# Extra degrees kept around the display box when sub-setting the national grid
DATA_PAD_DEG = 0.6
NEW_ENGLAND_STATES = {"CT", "ME", "MA", "NH", "RI", "VT"}

DISPLAY_TZ = "America/New_York"

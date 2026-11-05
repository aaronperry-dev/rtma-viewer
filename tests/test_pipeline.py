"""Exercise the real fetch/decode/cache code against a fake S3 that serves genuine GRIB2 + idx + Range."""
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from rtma_viewer.config import PRODUCTS, VARIABLES
from rtma_viewer.fields import FieldStore
from rtma_viewer.testing import FakeS3

NOW = datetime(2026, 11, 5, 17, 40, tzinfo=timezone.utc)
HOUR = datetime(2026, 11, 5, 17, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def s3():
    return FakeS3(NOW)


def test_latest_cycle_respects_lag(s3, tmp_path):
    assert FieldStore(PRODUCTS["rtma"], s3, tmp_path).latest_cycle(NOW) == HOUR
    assert FieldStore(PRODUCTS["urma"], s3, tmp_path).latest_cycle(NOW) == HOUR - timedelta(hours=3)


def test_no_cycle_means_none(tmp_path):
    s3 = FakeS3(NOW, missing={("rtma2p5", h) for h in range(0, 20)})
    assert FieldStore(PRODUCTS["rtma"], s3, tmp_path).latest_cycle(NOW) is None


def test_byte_range_fetch_values_and_cache(s3, tmp_path):
    st = FieldStore(PRODUCTS["rtma"], s3, tmp_path)
    jobs = [(HOUR, v) for v in VARIABLES.values()]
    s3.range_requests.clear()
    out = st.load(jobs)

    # one ranged GET per variable - never the whole file
    assert len(s3.range_requests) == 4
    # (the last message in an idx has no known end, so it is an open-ended "bytes=N-" request)
    assert all(re.fullmatch(r"bytes=\d+-\d*", r) for _, r in s3.range_requests)
    blob, _ = s3._file("rtma2p5", HOUR)
    assert st.src.bytes_downloaded < 0.75 * len(blob)

    # decoded values match the truth to within GRIB packing precision
    truth = s3.grid.fields_si(HOUR)
    g = st.grid.slices
    assert np.allclose(out[(HOUR, "temp")], ((truth["TMP"][g] - 273.15) * 1.8 + 32), atol=0.1)
    assert np.allclose(out[(HOUR, "gust")], truth["GUST"][g] * 2.2369362921, atol=0.1)

    # second store instance: everything comes from the on-disk cache, zero S3 traffic
    st2 = FieldStore(PRODUCTS["rtma"], s3, tmp_path)
    s3.range_requests.clear()
    again = st2.load(jobs)
    assert st2.stats["cache_hits"] == 4 and not s3.range_requests
    assert np.array_equal(again[(HOUR, "temp")], out[(HOUR, "temp")])


def test_wind_falls_back_to_components(tmp_path):
    class NoWind(FakeS3):
        def _file(self, prefix, cycle):
            blob, idx = super()._file(prefix, cycle)
            return blob, "\n".join(l for l in idx.splitlines() if ":WIND:" not in l) + "\n"

    st = FieldStore(PRODUCTS["rtma"], NoWind(NOW), tmp_path)
    out = st.load([(HOUR, VARIABLES["wind"])])
    truth = st.src.client.grid.fields_si(HOUR)["WIND"][st.grid.slices] * 2.2369362921
    assert np.allclose(out[(HOUR, "wind")], truth, atol=0.2)


def test_missing_hour_is_none(tmp_path):
    s3 = FakeS3(NOW, missing={("rtma2p5", 6)})
    out = FieldStore(PRODUCTS["rtma"], s3, tmp_path).load([(HOUR - timedelta(hours=6), VARIABLES["temp"])])
    assert out[(HOUR - timedelta(hours=6), "temp")] is None

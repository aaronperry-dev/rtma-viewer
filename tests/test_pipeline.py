"""Exercise the real fetch/decode/cache code against a fake S3 that serves genuine GRIB2 + idx + Range."""
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from rtma_viewer.config import INTERVALS, PANEL_VARIABLES, PRODUCTS, SOURCE_VARIABLES, VARIABLES, hours_needed, panel_hours
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
    jobs = [(HOUR, v) for v in SOURCE_VARIABLES.values()]
    n = len(SOURCE_VARIABLES)
    s3.range_requests.clear()
    out = st.load(jobs)

    # one ranged GET per variable - never the whole file (and never the UGRD / VGRD / SPFH messages we do not use)
    assert len(s3.range_requests) == n
    # (the last message in an idx has no known end, so it is an open-ended "bytes=N-" request)
    assert all(re.fullmatch(r"bytes=\d+-\d*", r) for _, r in s3.range_requests)
    blob, _ = s3._file("rtma2p5", HOUR)
    assert st.src.bytes_downloaded < 0.85 * len(blob)

    # decoded values match the truth to within GRIB packing precision
    truth = s3.grid.fields_si(HOUR)
    g = st.grid.slices
    assert np.allclose(out[(HOUR, "temp")], ((truth["TMP"][g] - 273.15) * 1.8 + 32), atol=0.1)
    assert np.allclose(out[(HOUR, "gust")], truth["GUST"][g] * 2.2369362921, atol=0.1)
    assert np.allclose(out[(HOUR, "pressure")], truth["PRES"][g] / 100.0, atol=0.05)       # mb, 0.01 mb packing
    assert np.allclose(out[(HOUR, "terrain")], truth["HGT"][g] * 3.280839895, atol=0.5)    # ft
    assert np.allclose(out[(HOUR, "vis")], truth["VIS"][g] / 1609.344, atol=0.01)          # statute miles
    assert np.allclose(out[(HOUR, "wdir_raw")], truth["WDIR"][g], atol=0.13)               # float16: 0.25 deg steps above 256
    assert out[(HOUR, "ceiling")].max() <= 12000.0 + 1e-3                                   # "no ceiling" is capped

    # second store instance: everything comes from the on-disk cache, zero S3 traffic
    st2 = FieldStore(PRODUCTS["rtma"], s3, tmp_path)
    s3.range_requests.clear()
    again = st2.load(jobs)
    assert st2.stats["cache_hits"] == n and not s3.range_requests
    for key in SOURCE_VARIABLES:
        assert np.array_equal(again[(HOUR, key)], out[(HOUR, key)], equal_nan=True), key


def test_pressure_cache_is_precise_enough_for_tendencies(s3, tmp_path):
    """float16 would quantise ~1000 mb to 0.5-1 mb; pressure is cached as packed uint16 at 0.01 mb instead."""
    st = FieldStore(PRODUCTS["rtma"], s3, tmp_path)
    var = SOURCE_VARIABLES["pressure"]
    out = st.load([(HOUR, var)])[(HOUR, "pressure")]
    on_disk = np.load(st._path(HOUR, var))
    assert on_disk.dtype == np.uint16                                   # same 2 bytes/cell as float16, 50x finer
    truth = s3.grid.fields_si(HOUR)["PRES"][st.grid.slices] / 100.0
    assert np.abs(out - truth).max() < 0.02
    assert np.abs(out.astype(np.float16).astype(np.float32) - truth).max() > 0.1      # what float16 would have done
    # a 0.3 mb change survives the round trip (the whole point of the tendency maps)
    a = FieldStore._decode(var, FieldStore._encode(var, np.full((2, 2), 1013.20, np.float32)))
    b = FieldStore._decode(var, FieldStore._encode(var, np.full((2, 2), 1013.50, np.float32)))
    assert np.allclose(b - a, 0.30, atol=0.011)


def test_pack_encodes_missing_and_out_of_range_as_nan():
    var = SOURCE_VARIABLES["pressure"]
    enc = FieldStore._encode(var, np.array([np.nan, 650.0, 1013.25, 1400.0], np.float32))
    dec = FieldStore._decode(var, enc)
    assert np.isnan(dec[0]) and np.isnan(dec[1]) and np.isnan(dec[3]) and dec[2] == pytest.approx(1013.25, abs=0.006)


def test_static_terrain_is_cached_once_and_reused_for_any_cycle(s3, tmp_path):
    st = FieldStore(PRODUCTS["rtma"], s3, tmp_path)
    ter = SOURCE_VARIABLES["terrain"]
    st.load([(HOUR, ter)])
    assert (tmp_path / "rtma" / "static" / "terrain.npy").exists()
    s3.range_requests.clear()
    other = HOUR - timedelta(hours=5)                                   # a different cycle: still a cache hit
    out = FieldStore(PRODUCTS["rtma"], s3, tmp_path).load([(other, ter)])
    assert out[(other, "terrain")] is not None and not s3.range_requests
    st.prune(keep_hours=0, now=HOUR + timedelta(days=30))               # static files are never pruned
    assert (tmp_path / "rtma" / "static" / "terrain.npy").exists()


def test_prune_keeps_only_newest_for_current_only_variables(s3, tmp_path):
    st = FieldStore(PRODUCTS["rtma"], s3, tmp_path)
    hrs = [HOUR - timedelta(hours=h) for h in (0, 1, 2)]
    st.load([(c, SOURCE_VARIABLES[k]) for c in hrs for k in ("temp", "vis")])
    st.prune(keep_hours=28, now=HOUR, keep_by_var={"vis": 0})
    left = sorted(p.name for p in (tmp_path / "rtma").glob("??????????_*.npy"))
    assert [n for n in left if n.endswith("_vis.npy")] == [f"{HOUR:%Y%m%d%H}_vis.npy"]
    assert len([n for n in left if n.endswith("_temp.npy")]) == 3


def test_fetch_plan_only_asks_for_what_the_panels_need():
    need = hours_needed()
    assert set(need) == set(SOURCE_VARIABLES)
    assert need["terrain"] == (0,)                                     # static: newest cycle only
    for k in ("vis", "ceiling", "cloud", "wdir_raw"):
        assert need[k] == (0,), k                                       # current-field-only variables
    for k in ("temp", "dewpoint", "wind", "gust", "pressure"):
        assert need[k] == INTERVALS, k                                  # feed change maps (directly or via derived fields)
    # every derived variable's inputs are loaded for every hour it is drawn at
    for v in VARIABLES.values():
        if v.derive is not None:
            for k in v.requires:
                assert set(panel_hours(v)) <= set(need[k]) or SOURCE_VARIABLES[k].static, (v.key, k)


def test_wind_falls_back_to_components(tmp_path):
    class NoWind(FakeS3):
        def _file(self, prefix, cycle):
            blob, idx = super()._file(prefix, cycle)
            return blob, "\n".join(l for l in idx.splitlines() if ":WIND:" not in l) + "\n"

    st = FieldStore(PRODUCTS["rtma"], NoWind(NOW), tmp_path)
    out = st.load([(HOUR, SOURCE_VARIABLES["wind"])])
    truth = st.src.client.grid.fields_si(HOUR)["WIND"][st.grid.slices] * 2.2369362921
    assert np.allclose(out[(HOUR, "wind")], truth, atol=0.2)


def test_missing_hour_is_none(tmp_path):
    s3 = FakeS3(NOW, missing={("rtma2p5", 6)})
    out = FieldStore(PRODUCTS["rtma"], s3, tmp_path).load([(HOUR - timedelta(hours=6), SOURCE_VARIABLES["temp"])])
    assert out[(HOUR - timedelta(hours=6), "temp")] is None


def test_mslp_round_trips_through_grib_cache_and_derivation(s3, tmp_path):
    """Fake GRIB (synthetic station pressure built from a known sea-level field) -> byte-range fetch -> decode ->
    packed cache -> derive MSLP: the known sea-level field must come back, to within cache precision."""
    from rtma_viewer.config import VARIABLES as V

    st = FieldStore(PRODUCTS["rtma"], s3, tmp_path)
    keys = V["mslp"].requires
    f = st.load([(HOUR, SOURCE_VARIABLES[k]) for k in keys])
    m = V["mslp"].derive(*[f[(HOUR, k)] for k in keys])
    truth = s3.grid.slp_mb(HOUR)[st.grid.slices]
    assert np.abs(m - truth).max() < 0.12                      # float16 temperature (0.03 F) + 0.01 mb pressure packing
    tw = V["wetbulb"].derive(*[f[(HOUR, k)] for k in V["wetbulb"].requires])
    t, td = f[(HOUR, "temp")], f[(HOUR, "dewpoint")]
    assert np.all(tw <= t + 1e-3) and np.all(tw >= np.minimum(td, t) - 1e-3)


def test_hidden_inputs_are_fetched_but_have_no_page_entry():
    hidden = [k for k, v in VARIABLES.items() if v.hidden]
    assert hidden == ["wdir_raw"] and "wdir_raw" in SOURCE_VARIABLES and "wdir_raw" not in PANEL_VARIABLES
    assert "wdir" in PANEL_VARIABLES and VARIABLES["wdir"].requires == ("wdir_raw", "wind")
    assert hours_needed()["wind"] == INTERVALS                      # still needed at every hour for the other panels

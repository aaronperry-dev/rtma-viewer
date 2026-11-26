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
    n_idx = sum(v.source == "idx" for v in SOURCE_VARIABLES.values())
    s3.range_requests.clear()
    out = st.load([j for j in jobs if j[1].source == "idx"])

    # one ranged GET per analysis variable - never the whole file (the specific-humidity message is the only one we do not use);
    # the hourly precipitation file is its own small object, fetched whole
    assert len(s3.range_requests) == n_idx
    # (the last message in an idx has no known end, so it is an open-ended "bytes=N-" request)
    assert all(re.fullmatch(r"bytes=\d+-\d*", r) for _, r in s3.range_requests)
    blob, _ = s3._file("rtma2p5", HOUR)
    assert st.src.bytes_downloaded < 0.95 * len(blob)
    out.update(st.load([j for j in jobs if j[1].source == "pcp"]))            # the precipitation file comes on top

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
    s3.pcp_requests.clear()
    assert st2.stats["cache_hits"] == 0
    again = st2.load(jobs)
    assert st2.stats["cache_hits"] == n and not s3.range_requests and not s3.pcp_requests
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
    for k in ("vis", "ceiling", "cloud", "wdir_raw", "u10", "v10"):
        assert need[k] == (0,), k                                       # current-field-only variables
    for k in ("dewpoint", "pressure"):
        assert need[k] == tuple(range(25)), k                           # change maps, and a day of wet bulbs for frozen precipitation
    for k in ("temp", "wind", "gust"):
        assert need[k] == tuple(range(25)), k                           # ... and the 24-hour summaries use every hourly analysis
    # every derived variable's inputs exist for every hour it is drawn at (series variables: and the hours before it)
    for v in VARIABLES.values():
        if v.derive is None:
            continue
        hours = {h + i for h in panel_hours(v) for i in range(v.series + 1)}
        for k in v.requires:
            dep = VARIABLES[k]
            have = set(need[k]) if dep.derive is None else set(panel_hours(dep))
            assert hours <= have or dep.static, (v.key, k)
    # derived-from-derived variables come after what they use (the build computes in dictionary order)
    order = list(VARIABLES)
    for v in VARIABLES.values():
        for k in v.requires:
            if VARIABLES[k].derive is not None:
                assert order.index(k) < order.index(v.key), (v.key, k)


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
    assert hidden == ["wdir_raw", "u10", "v10"] and all(k in SOURCE_VARIABLES and k not in PANEL_VARIABLES for k in hidden)
    assert "wdir" in PANEL_VARIABLES and VARIABLES["wdir"].requires == ("wdir_raw", "wind")
    assert set(INTERVALS) <= set(hours_needed()["wind"])           # still needed at every change interval for the other panels


def test_wind_components_round_trip_at_millimetre_per_second_precision(s3, tmp_path):
    """u / v feed derivatives, so they are packed as uint16 at 0.002 m/s instead of float16 (0.03 m/s at 20 m/s)."""
    st = FieldStore(PRODUCTS["rtma"], s3, tmp_path)
    out = st.load([(HOUR, SOURCE_VARIABLES["u10"]), (HOUR, SOURCE_VARIABLES["v10"])])
    truth = s3.grid.fields_si(HOUR)
    g = st.grid.slices
    assert np.abs(out[(HOUR, "u10")] - truth["UGRD"][g]).max() < 0.005 and np.abs(out[(HOUR, "v10")] - truth["VGRD"][g]).max() < 0.005
    assert np.load(st._path(HOUR, SOURCE_VARIABLES["u10"])).dtype == np.uint16
    # grid-relative magnitude is the speed
    assert np.allclose(np.hypot(out[(HOUR, "u10")], out[(HOUR, "v10")]), truth["WIND"][g], atol=0.01)


def test_series_inputs_stack_hours_newest_first_and_tolerate_gaps():
    from rtma_viewer.build import series_inputs
    var = VARIABLES["wcmin24"]                                            # needs temp + wind over 25 hourly analyses
    f = {(HOUR - timedelta(hours=i), k): np.full((2, 3), float(i + (100 if k == "wind" else 0)), np.float32)
         for i in range(25) for k in ("temp", "wind")}
    t, w = series_inputs(f, var, HOUR)
    assert t.shape == (25, 2, 3) and t[0, 0, 0] == 0 and t[24, 0, 0] == 24 and w[3, 0, 0] == 103
    for i in (5, 6, 7):                                                   # a missing hour in either input blanks that plane in both
        f[(HOUR - timedelta(hours=i), "temp" if i != 6 else "wind")] = None
    t, w = series_inputs(f, var, HOUR)
    assert np.isnan(t[5]).all() and np.isnan(t[6]).all() and np.isnan(w[7]).all() and t[8, 0, 0] == 8
    for i in range(8, 20):                                                # more than 25 % missing: no summary at all
        f[(HOUR - timedelta(hours=i), "temp")] = None
    assert series_inputs(f, var, HOUR) is None
    f2 = {k: v for k, v in f.items() if k[0] != HOUR}                     # the valid hour itself missing: nothing to show
    assert series_inputs(f2, var, HOUR) is None


# --------------------------------------------------------------------------- hourly precipitation analyses
PCP = SOURCE_VARIABLES["pcp1"]


def _load_pcp(store, cycles, now=None):
    return store.load([(c, PCP) for c in cycles] + [(cycles[0], SOURCE_VARIABLES["temp"])], now)


def test_precip_comes_from_its_own_cropped_file_and_matches_the_truth(s3, tmp_path):
    st = FieldStore(PRODUCTS["rtma"], s3, tmp_path)
    s3.pcp_requests.clear()
    out = _load_pcp(st, [HOUR])
    assert s3.pcp_requests == [f"rtma2p5.{HOUR:%Y%m%d}/rtma2p5.{HOUR:%Y%m%d%H}.pcp.184.grb2"]
    # NOAA crops the precipitation grid differently: same lattice, different indices -> its own window, equal shape
    assert st.pcp_grid.window != st.grid.window and st.pcp_grid.lat.shape == st.grid.lat.shape
    assert np.abs(st.pcp_grid.lat - st.grid.lat).max() < 5e-4 and np.abs(st.pcp_grid.lon - st.grid.lon).max() < 5e-4
    truth_in = s3.grid.precip_mm(HOUR)[st.grid.slices] / 25.4
    assert out[(HOUR, "pcp1")].shape == st.grid.lat.shape
    assert np.nanmax(truth_in) > 0.1                                                  # the demo band is in the window
    assert np.allclose(out[(HOUR, "pcp1")], truth_in, atol=0.002)                      # inches, 0.0005 in packing + GRIB packing
    assert np.load(st._path(HOUR, PCP)).dtype == np.uint16                              # cached packed, 2 bytes per cell


def test_urma_precip_keeps_its_missing_cells(s3, tmp_path):
    st = FieldStore(PRODUCTS["urma"], s3, tmp_path)
    cyc = HOUR - timedelta(hours=3)
    out = _load_pcp(st, [cyc])[(cyc, "pcp1")]
    truth = s3.grid.precip_mm(cyc, urma=True)[st.grid.slices]
    assert np.isnan(truth).any() and np.array_equal(np.isnan(out), np.isnan(truth))    # offshore NaN survives the uint16 cache
    assert s3.pcp_requests[-1].endswith(".pcp_01h.wexp.grb2")


def test_recent_precip_is_refetched_old_precip_is_not(s3, tmp_path):
    st = FieldStore(PRODUCTS["rtma"], s3, tmp_path)               # refresh window: 3 h for RTMA
    old = HOUR - timedelta(hours=10)
    _load_pcp(st, [HOUR, old])
    s3.pcp_requests.clear()
    st2 = FieldStore(PRODUCTS["rtma"], s3, tmp_path)
    _load_pcp(st2, [HOUR, old], NOW)
    assert [k.split("/")[-1] for k in s3.pcp_requests] == [f"rtma2p5.{HOUR:%Y%m%d%H}.pcp.184.grb2"]      # only the recent hour was re-downloaded
    s3.pcp_requests.clear()
    _load_pcp(FieldStore(PRODUCTS["rtma"], s3, tmp_path), [HOUR, old])                  # no ``now``: pure cache, as before
    assert not s3.pcp_requests


def test_refresh_falls_back_to_the_cache_when_the_file_vanishes(tmp_path):
    full = FakeS3(NOW)
    first = FieldStore(PRODUCTS["rtma"], full, tmp_path)
    kept = _load_pcp(first, [HOUR])[(HOUR, "pcp1")]
    gone = FakeS3(NOW, missing={("pcp:rtma2p5", 0)})
    again = _load_pcp(FieldStore(PRODUCTS["rtma"], gone, tmp_path), [HOUR], NOW)[(HOUR, "pcp1")]
    assert np.array_equal(again, kept, equal_nan=True)
    # and with nothing cached a missing file is simply unavailable
    nothing = _load_pcp(FieldStore(PRODUCTS["rtma"], gone, tmp_path / "empty"), [HOUR])[(HOUR, "pcp1")]
    assert nothing is None


def test_precip_hours_needed_and_sums():
    need = hours_needed()
    assert need["pcp1"] == tuple(range(48))                       # 48 hourly files cover every period up to 48 h
    for k in ("temp", "dewpoint", "pressure"):
        assert need[k][-1] >= 24                                   # the frozen-precipitation product needs a day of wet bulbs
    for n in (3, 6, 12, 24, 48):
        v = VARIABLES[f"pcp{n}"]
        assert v.series == n - 1 and v.requires == ("pcp1",) and panel_hours(v) == (0,) and v.min_series_fraction == 1.0


def test_series_inputs_are_strict_for_precipitation():
    from rtma_viewer.build import series_inputs
    v = VARIABLES["pcp6"]
    base = np.ones((4, 5), np.float32)
    fields = {(HOUR - timedelta(hours=i), "pcp1"): base for i in range(6)}
    assert series_inputs(fields, v, HOUR)[0].shape == (6, 4, 5)
    fields[(HOUR - timedelta(hours=3), "pcp1")] = None            # one hour missing: a 6 h total would silently read low
    assert series_inputs(fields, v, HOUR) is None
    t = VARIABLES["tmax24"]                                        # temperature summaries keep their 75 % tolerance
    tf = {(HOUR - timedelta(hours=i), "temp"): base for i in range(25) if i != 7}
    assert series_inputs(tf, t, HOUR) is not None

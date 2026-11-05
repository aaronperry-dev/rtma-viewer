import numpy as np

from rtma_viewer import stations as stn

NE_BOX = (40.9, 47.6, -73.8, -66.8)   # lat_min, lat_max, lon_min, lon_max (with a little slack)


def test_station_list_sane():
    st = stn.load()
    ids = [s.icao for s in st]
    assert len(ids) == len(set(ids)) >= 60
    assert "KMWN" in ids and next(s for s in st if s.icao == "KMWN").priority == 0
    for s in st:
        assert NE_BOX[0] - 0.2 <= s.lat <= NE_BOX[1] + 0.2 and NE_BOX[2] - 0.2 <= s.lon <= NE_BOX[3] + 0.2, s
        assert s.state in {"CT", "MA", "ME", "NH", "RI", "VT"}
    # every state is represented, with the big airports present
    assert {s.state for s in st} == {"CT", "MA", "ME", "NH", "RI", "VT"}
    assert {"KBOS", "KPVD", "KBDL", "KBTV", "KPWM", "KBGR", "KCON", "KMHT"} <= set(ids)


def test_declutter_keeps_priority_and_spacing():
    mk = lambda i, p: stn.Station(f"K{i:03d}", "", "MA", 0, 0, 0, p)
    st = [mk(0, 0), mk(1, 1), mk(2, 1), mk(3, 2), mk(4, 1)]
    xy = np.array([[0, 0], [10, 0], [100, 0], [105, 0], [500, 500]], float)
    kept = stn.declutter(st, xy, (-50, 600, -50, 600), min_sep=40, edge=0)
    assert kept == [0, 2, 4]          # #1 is 10 m from priority-0 #0; #3 is 5 m from #2; #4 is far away
    # priority 0 is never dropped, even when crowded
    assert stn.declutter([mk(0, 0), mk(1, 0)], np.array([[0, 0], [1, 0]], float), (-9, 9, -9, 9), 40, 0) == [0, 1]
    # outside the frame (minus the edge keep-out) -> not labelled
    assert stn.declutter([mk(0, 1)], np.array([[595, 0]], float), (0, 600, -50, 50), 40, 10) == []

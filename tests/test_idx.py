from rtma_viewer.s3 import find_entry, parse_idx

IDX = """\
1:0:d=2026110512:PRES:surface:anl:
2:612345:d=2026110512:TMP:2 m above ground:anl:
3:1201234:d=2026110512:DPT:2 m above ground:anl:
4:1800000:d=2026110512:WIND:10 m above ground:anl:
5:2400000:d=2026110512:GUST:10 m above ground:anl:
"""


def test_parse_and_ranges():
    e = parse_idx(IDX)
    assert [x.var for x in e] == ["PRES", "TMP", "DPT", "WIND", "GUST"]
    tmp = find_entry(e, "TMP", "2 m above ground")
    assert (tmp.start, tmp.end) == (612345, 1201233)     # inclusive end = next start - 1
    assert find_entry(e, "GUST", "10 m above ground").end is None   # last message: open-ended range
    assert find_entry(e, "TMP", "10 m above ground") is None


def test_parse_ignores_junk():
    assert parse_idx("\n<html>nope</html>\n") == []

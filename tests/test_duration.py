import pytest
from ocp.duration import parse_duration, format_duration, DurationError


@pytest.mark.parametrize(
    "s,expected",
    [
        ("5s", 5), ("90s", 90), ("5m", 300), ("1h", 3600),
        ("1h30m", 5400), ("2h30m15s", 9015), ("10m", 600),
    ],
)
def test_parse_ok(s, expected):
    assert parse_duration(s) == expected


@pytest.mark.parametrize(
    "s", ["", " ", "0s", "abc", "5", "1d", "5m m", "-3m", "1.5m", "m5"],
)
def test_parse_rejects(s):
    with pytest.raises(DurationError):
        parse_duration(s)


def test_format():
    assert format_duration(0) == "0s"
    assert format_duration(45) == "45s"
    assert format_duration(60) == "1m"
    assert format_duration(90) == "1m 30s"
    assert format_duration(3600) == "1h"
    assert format_duration(5400) == "1h 30m"
    assert format_duration(7215) == "2h 15s"

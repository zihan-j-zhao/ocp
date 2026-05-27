import pytest
from ocp.gpuspec import parse_gpu_spec, GpuSpecError


def test_single():
    assert parse_gpu_spec("7") == [7]


def test_list():
    assert parse_gpu_spec("0,2,3") == [0, 2, 3]


def test_whitespace_tolerated():
    assert parse_gpu_spec(" 1 , 2 ") == [1, 2]


def test_order_preserved():
    assert parse_gpu_spec("3,1,2") == [3, 1, 2]


@pytest.mark.parametrize("s", ["", " ", "1,,2", "abc", "-1", "1,1", ","])
def test_rejects(s):
    with pytest.raises(GpuSpecError):
        parse_gpu_spec(s)

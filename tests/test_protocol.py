import pytest
from ocp.ipc.protocol import Request, Response, ProtocolError


def test_request_roundtrip():
    r = Request(cmd="PAUSE", args={"gpus": [7], "duration_s": 600})
    r2 = Request.from_bytes(r.to_bytes())
    assert r2.cmd == "PAUSE"
    assert r2.args == {"gpus": [7], "duration_s": 600}
    assert r2.v == 1


def test_response_success_roundtrip():
    r = Response.success("abc", data={"leases": []})
    r2 = Response.from_bytes(r.to_bytes())
    assert r2.ok is True
    assert r2.data == {"leases": []}


def test_response_failure_roundtrip():
    r = Response.failure("abc", "E_PAUSE_HELD", "held", {"conflicts": []})
    r2 = Response.from_bytes(r.to_bytes())
    assert r2.ok is False
    assert r2.error["code"] == "E_PAUSE_HELD"
    assert r2.error["data"] == {"conflicts": []}


def test_bad_request_missing_cmd():
    with pytest.raises(ProtocolError):
        Request.from_bytes(b"{}")


def test_bad_request_non_dict_args():
    with pytest.raises(ProtocolError):
        Request.from_bytes(b'{"cmd": "X", "args": 1}')


def test_bad_protocol_version():
    with pytest.raises(ProtocolError):
        Request.from_bytes(b'{"cmd": "X", "v": 99}')

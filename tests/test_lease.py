import time
import pytest

from ocp.lease import LeaseStore, PauseHeld


def test_acquire_release(tmp_path):
    s = LeaseStore(tmp_path / "state.json")
    leases = s.acquire(uid=1001, user="alice", gpus=[0], duration_s=60)
    assert len(leases) == 1
    assert s.covers(0)
    released, errors = s.release(uid=1001, gpus=[0], is_root=False)
    assert released == [0] and not errors
    assert not s.covers(0)


def test_pause_held_atomic(tmp_path):
    s = LeaseStore(tmp_path / "state.json")
    s.acquire(uid=1001, user="alice", gpus=[0], duration_s=60)
    with pytest.raises(PauseHeld) as ei:
        s.acquire(uid=2002, user="bob", gpus=[0, 1], duration_s=60)
    assert ei.value.conflicts[0]["gpu"] == 0
    # Nothing was installed on gpu 1.
    assert not s.covers(1)


def test_extend_never_shrinks(tmp_path):
    s = LeaseStore(tmp_path / "state.json")
    now = time.time()
    s.acquire(uid=1001, user="alice", gpus=[0], duration_s=600, now=now)
    long_exp = s.get(0).expires_at
    s.acquire(uid=1001, user="alice", gpus=[0], duration_s=60, now=now)
    assert s.get(0).expires_at == long_exp


def test_extend_grows(tmp_path):
    s = LeaseStore(tmp_path / "state.json")
    now = time.time()
    s.acquire(uid=1001, user="alice", gpus=[0], duration_s=60, now=now)
    s.acquire(uid=1001, user="alice", gpus=[0], duration_s=600, now=now)
    assert s.get(0).expires_at - now == pytest.approx(600, abs=1)


def test_release_only_holder(tmp_path):
    s = LeaseStore(tmp_path / "state.json")
    s.acquire(uid=1001, user="alice", gpus=[0], duration_s=60)
    released, errors = s.release(uid=2002, gpus=[0], is_root=False)
    assert released == []
    assert errors[0]["code"] == "E_NOT_LEASE_HOLDER"
    released, errors = s.release(uid=0, gpus=[0], is_root=True)
    assert released == [0]


def test_release_partial(tmp_path):
    s = LeaseStore(tmp_path / "state.json")
    s.acquire(uid=1001, user="alice", gpus=[0, 1], duration_s=60)
    released, errors = s.release(uid=1001, gpus=[0, 1, 2], is_root=False)
    assert sorted(released) == [0, 1]
    assert errors == [{"gpu": 2, "code": "E_NO_PAUSE"}]


def test_persistence_roundtrip(tmp_path):
    p = tmp_path / "state.json"
    s = LeaseStore(p)
    now = time.time()
    s.acquire(uid=1001, user="alice", gpus=[3], duration_s=600, now=now)
    s2 = LeaseStore(p)
    s2.load(now=now)
    assert s2.covers(3)
    assert s2.get(3).user == "alice"


def test_persistence_drops_expired(tmp_path):
    p = tmp_path / "state.json"
    s = LeaseStore(p)
    s.acquire(uid=1001, user="alice", gpus=[3], duration_s=1, now=1000.0)
    s2 = LeaseStore(p)
    loaded = s2.load(now=2000.0)
    assert loaded == [] and not s2.covers(3)


def test_sweep_expired(tmp_path):
    s = LeaseStore(tmp_path / "state.json")
    s.acquire(uid=1001, user="alice", gpus=[0], duration_s=10, now=1000.0)
    s.acquire(uid=2002, user="bob", gpus=[1], duration_s=100, now=1000.0)
    expired = s.sweep_expired(now=1020.0)
    assert [e.gpu for e in expired] == [0]
    assert s.covers(1) and not s.covers(0)


def test_concurrent_disjoint_holders(tmp_path):
    """Different users may hold different GPUs simultaneously."""
    s = LeaseStore(tmp_path / "state.json")
    s.acquire(uid=1001, user="alice", gpus=[0], duration_s=60)
    s.acquire(uid=2002, user="bob",   gpus=[1], duration_s=60)
    assert s.get(0).user == "alice"
    assert s.get(1).user == "bob"

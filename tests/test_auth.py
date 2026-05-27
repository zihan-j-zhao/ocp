from ocp.auth import Caller, is_root, is_whitelisted


def test_root_is_root():
    c = Caller(pid=1, uid=0, gid=0, user="root")
    assert is_root(c)
    assert is_whitelisted(c, [])


def test_whitelist_by_name():
    c = Caller(pid=1, uid=1001, gid=100, user="alice")
    assert is_whitelisted(c, ["alice"])


def test_whitelist_by_uid():
    c = Caller(pid=1, uid=1042, gid=100, user="carol")
    assert is_whitelisted(c, [1042])


def test_not_whitelisted():
    c = Caller(pid=1, uid=2002, gid=100, user="bob")
    assert not is_whitelisted(c, ["alice", 1042])


def test_bool_in_whitelist_is_ignored():
    # True == 1 in Python; make sure we don't accidentally grant uid=1.
    c = Caller(pid=1, uid=1, gid=100, user="bin")
    assert not is_whitelisted(c, [True])

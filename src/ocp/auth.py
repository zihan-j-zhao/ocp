"""Whitelist + root authorization based on SO_PEERCRED uid/name."""
from __future__ import annotations

import pwd
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Caller:
    pid: int
    uid: int
    gid: int
    user: str


def caller_from_creds(pid: int, uid: int, gid: int) -> Caller:
    try:
        user = pwd.getpwuid(uid).pw_name
    except KeyError:
        user = f"uid={uid}"
    return Caller(pid=pid, uid=uid, gid=gid, user=user)


def is_root(caller: Caller) -> bool:
    return caller.uid == 0


def is_whitelisted(caller: Caller, whitelist: Iterable) -> bool:
    if is_root(caller):
        return True
    for entry in whitelist:
        if isinstance(entry, bool):  # bool is a subclass of int; skip silently
            continue
        if isinstance(entry, int) and entry == caller.uid:
            return True
        if isinstance(entry, str) and entry == caller.user:
            return True
    return False

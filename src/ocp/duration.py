"""Parse and format human duration strings like '5m', '90s', '1h30m'."""
from __future__ import annotations

import re


_UNIT_S = {"s": 1, "m": 60, "h": 3600}
_TOKEN_RE = re.compile(r"(\d+)([smh])")


class DurationError(ValueError):
    pass


def parse_duration(s: str) -> int:
    """Return total seconds (> 0). Rejects empty, zero, and unknown forms."""
    if not isinstance(s, str):
        raise DurationError("duration must be a string")
    s = s.strip().lower()
    if not s:
        raise DurationError("empty duration")
    tokens = _TOKEN_RE.findall(s)
    rebuilt = "".join(f"{n}{u}" for n, u in tokens)
    if rebuilt != s:
        raise DurationError(f"unrecognized duration: {s!r}")
    total = sum(int(n) * _UNIT_S[u] for n, u in tokens)
    if total <= 0:
        raise DurationError(f"duration must be > 0: {s!r}")
    return total


def format_duration(seconds: int) -> str:
    """Human-readable: 90 -> '1m 30s', 3600 -> '1h'."""
    if seconds <= 0:
        return "0s"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    parts: list[str] = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    if s or not parts:
        parts.append(f"{s}s")
    return " ".join(parts)

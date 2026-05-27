"""Parse GPU specs like '7' or '0,2,3' from the CLI."""
from __future__ import annotations


class GpuSpecError(ValueError):
    pass


def parse_gpu_spec(s: str) -> list[int]:
    """Return a list of GPU indices. Rejects empty, negative, and duplicates."""
    if not isinstance(s, str):
        raise GpuSpecError("gpu spec must be a string")
    raw = s.split(",")
    # Reject mid-list empties like '1,,2' or trailing/leading commas.
    parts = [p.strip() for p in raw]
    if any(p == "" for p in parts):
        if all(p == "" for p in parts):
            raise GpuSpecError(f"empty gpu spec: {s!r}")
        raise GpuSpecError(f"empty index in gpu spec: {s!r}")
    out: list[int] = []
    seen: set[int] = set()
    for p in parts:
        try:
            idx = int(p)
        except ValueError:
            raise GpuSpecError(f"not a gpu index: {p!r}")
        if idx < 0:
            raise GpuSpecError(f"negative gpu index: {idx}")
        if idx in seen:
            raise GpuSpecError(f"duplicate gpu index: {idx}")
        seen.add(idx)
        out.append(idx)
    return out

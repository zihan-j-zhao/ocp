"""Per-GPU pause leases with atomic multi-GPU acquisition."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path


class LeaseError(Exception):
    pass


class PauseHeld(LeaseError):
    def __init__(self, conflicts: list[dict]):
        super().__init__("pause held")
        self.conflicts = conflicts


@dataclass
class Lease:
    uid: int
    user: str
    gpu: int
    acquired_at: float  # wall-clock
    expires_at: float   # wall-clock

    def remaining_s(self, now: float | None = None) -> int:
        now = now if now is not None else time.time()
        return max(0, int(self.expires_at - now))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Lease":
        return cls(
            uid=int(d["uid"]),
            user=str(d["user"]),
            gpu=int(d["gpu"]),
            acquired_at=float(d["acquired_at"]),
            expires_at=float(d["expires_at"]),
        )


class LeaseStore:
    """In-memory per-GPU lease map with atomic file persistence."""

    def __init__(self, state_path: Path):
        self._path = Path(state_path)
        self._leases: dict[int, Lease] = {}

    # --- query -------------------------------------------------------------

    def get(self, gpu: int) -> Lease | None:
        return self._leases.get(gpu)

    def all(self) -> list[Lease]:
        return list(self._leases.values())

    def covers(self, gpu: int) -> bool:
        return gpu in self._leases

    # --- mutate ------------------------------------------------------------

    def acquire(
        self,
        *,
        uid: int,
        user: str,
        gpus: list[int],
        duration_s: int,
        now: float | None = None,
    ) -> list[Lease]:
        """Acquire-or-extend leases on `gpus`. Atomic across the set.

        Raises PauseHeld if any requested GPU is held by another uid.
        For GPUs already held by `uid`, the deadline is extended to
        `max(current, now + duration_s)` and never shrinks.
        """
        now = now if now is not None else time.time()
        conflicts = []
        for g in gpus:
            existing = self._leases.get(g)
            if existing is not None and existing.uid != uid:
                conflicts.append({
                    "gpu": g,
                    "holder": existing.user,
                    "remaining_s": existing.remaining_s(now),
                })
        if conflicts:
            raise PauseHeld(conflicts)

        new_leases: list[Lease] = []
        for g in gpus:
            existing = self._leases.get(g)
            new_expires = max(
                existing.expires_at if existing else 0.0,
                now + duration_s,
            )
            acquired_at = existing.acquired_at if existing else now
            lease = Lease(
                uid=uid, user=user, gpu=g,
                acquired_at=acquired_at, expires_at=new_expires,
            )
            self._leases[g] = lease
            new_leases.append(lease)
        self.persist()
        return new_leases

    def release(
        self,
        *,
        uid: int,
        gpus: list[int],
        is_root: bool,
        now: float | None = None,
    ) -> tuple[list[int], list[dict]]:
        """Per-GPU release. Returns (released_gpus, errors)."""
        released: list[int] = []
        errors: list[dict] = []
        for g in gpus:
            lease = self._leases.get(g)
            if lease is None:
                errors.append({"gpu": g, "code": "E_NO_PAUSE"})
                continue
            if lease.uid != uid and not is_root:
                errors.append({
                    "gpu": g,
                    "code": "E_NOT_LEASE_HOLDER",
                    "holder": lease.user,
                })
                continue
            del self._leases[g]
            released.append(g)
        if released:
            self.persist()
        return released, errors

    def expire(self, gpu: int) -> Lease | None:
        lease = self._leases.pop(gpu, None)
        if lease is not None:
            self.persist()
        return lease

    def sweep_expired(self, now: float | None = None) -> list[Lease]:
        now = now if now is not None else time.time()
        expired = [g for g, l in self._leases.items() if l.expires_at <= now]
        out = [self._leases.pop(g) for g in expired]
        if out:
            self.persist()
        return out

    # --- persistence -------------------------------------------------------

    def load(self, *, now: float | None = None) -> list[Lease]:
        now = now if now is not None else time.time()
        if not self._path.exists():
            return []
        try:
            raw = json.loads(self._path.read_text() or "{}")
        except (OSError, json.JSONDecodeError):
            return []
        loaded: list[Lease] = []
        for d in raw.get("leases") or []:
            try:
                lease = Lease.from_dict(d)
            except (KeyError, ValueError, TypeError):
                continue
            if lease.expires_at > now:
                self._leases[lease.gpu] = lease
                loaded.append(lease)
        return loaded

    def persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        data = {"leases": [l.to_dict() for l in self._leases.values()]}
        tmp.write_text(json.dumps(data, separators=(",", ":")))
        os.replace(tmp, self._path)

"""Stderr-only logging. systemd captures it into the journal."""
from __future__ import annotations

import logging
import sys


def setup(level: str = "INFO") -> None:
    root = logging.getLogger()
    try:
        root.setLevel(level)
    except (ValueError, TypeError):
        root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(handler)

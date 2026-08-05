"""Verifier helpers. Each returns a plain function State -> float in [0, 1]."""
from __future__ import annotations

from typing import Callable

from cyberl.state import State


def flag(expected: str) -> Callable[[State], float]:
    """Reward 1.0 when the agent's final answer contains the flag string."""
    return lambda s: 1.0 if expected in (s.answer or "") else 0.0


def contains(text: str) -> Callable[[State], float]:
    """Reward 1.0 when text appears anywhere in the transcript's text content."""
    def check(s: State) -> float:
        blob = " ".join(str(m.get("content") or "") for m in s.transcript)
        return 1.0 if text in blob else 0.0
    return check


def file_exists(path: str, host: str | None = None) -> Callable[[State], float]:
    """Reward 1.0 when the file exists in the world."""
    return lambda s: 1.0 if s.file(path, host) is not None else 0.0

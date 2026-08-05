"""Models: any callable that maps (messages, tools) -> an assistant message dict."""
from __future__ import annotations

from typing import Protocol


class Model(Protocol):
    def __call__(self, messages: list[dict], tools: list[dict]) -> dict:
        """Return an OpenAI-shaped assistant message dict."""
        ...


class ScriptedModel:
    """A deterministic model that replays a fixed list of assistant messages."""

    def __init__(self, steps: list[dict]):
        self._steps = list(steps)
        self._i = 0

    def __call__(self, messages: list[dict], tools: list[dict]) -> dict:
        step = self._steps[self._i]   # IndexError when exhausted (intended)
        self._i += 1
        return step

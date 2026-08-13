"""Models: any callable that maps (messages, tools) -> an assistant message dict."""
from __future__ import annotations

from typing import Protocol

try:
    from openai import OpenAI
except ImportError:  # optional extra (`pip install opencrl[openai]`)
    OpenAI = None  # type: ignore[assignment]


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
        if self._i >= len(self._steps):
            raise StopIteration(
                f"opencrl: ScriptedModel exhausted after {len(self._steps)} step(s)")
        step = self._steps[self._i]
        self._i += 1
        return step


class OpenAIModel:
    """Wraps the OpenAI-compatible chat API. Requires the `openai` extra."""

    def __init__(self, model: str, base_url: str | None = None,
                 api_key: str | None = None, max_retries: int = 3):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key,
                             max_retries=max_retries)

    def __call__(self, messages: list[dict], tools: list[dict]) -> dict:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, tools=tools or None)
        msg = resp.choices[0].message
        return {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in (msg.tool_calls or [])
            ] or None,
        }

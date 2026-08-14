"""The permissive Gym space for opencrl message transcripts.

Imported lazily (it needs gymnasium) so the core never depends on gym. Kept a
real top-level class — not a closure-local one — so it is picklable by
qualified name, which `AsyncVectorEnv` requires when it ships spaces across a
process pipe.
"""
from __future__ import annotations

import gymnasium as gym


class MessageSpace(gym.spaces.Space):
    """Permissive space for OpenAI-style message dicts (or, when sequence=True,
    a transcript list of them). LLM tool-call messages don't fit numeric Gym
    spaces, so this validates shape without pretending to be a Box/Text.

    Equality is by value (the sequence flag) so gym's vector envs — which
    compare every sub-env's spaces to the first — accept a batch of them.
    """

    def __init__(self, sequence: bool = False):
        super().__init__(shape=None, dtype=None)
        self._sequence = sequence

    def contains(self, x):
        if self._sequence:
            return isinstance(x, list) and all(isinstance(m, dict) for m in x)
        return isinstance(x, dict)

    def sample(self, mask=None):
        if self._sequence:
            return []
        return {"role": "assistant", "content": "", "tool_calls": None}

    def __eq__(self, other):
        return isinstance(other, MessageSpace) and other._sequence == self._sequence

    def __hash__(self):
        return hash(("MessageSpace", self._sequence))

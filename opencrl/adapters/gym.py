"""Gym adapter: expose a Task as a gymnasium.Env, reusing the Episode primitive.

gymnasium is imported lazily so the core never depends on it.
"""
from __future__ import annotations

from opencrl.backend import resolve_backend
from opencrl.rollout import Episode
from opencrl.task import Task, load_world


def to_gym(task: Task, backend=None):
    import gymnasium as gym

    class _MessageSpace(gym.spaces.Space):
        """Permissive space for OpenAI-style message dicts (or, if sequence=True,
        a transcript list of them). LLM tool-call messages don't fit numeric Gym
        spaces, so this validates shape without pretending to be a Box/Text."""
        def __init__(self, sequence=False):
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

    class OpencrlEnv(gym.Env):
        def __init__(self):
            self._task = task
            self._backend = resolve_backend(backend or task.backend)
            self._episode = None
            self._world = None
            self._steps = 0
            # Actions are OpenAI-style assistant-message dicts and observations are
            # message transcripts — neither fits a standard numeric Gym space, so these
            # are permissive placeholders provided so the env satisfies the Gym API.
            self.action_space = _MessageSpace()
            self.observation_space = _MessageSpace(sequence=True)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            if self._world is not None:      # tear down a prior episode's world
                self._backend.down(self._world)
            self._world = self._backend.up(load_world(self._task), self._task.caps)
            self._episode = Episode(self._task, self._world)
            self._steps = 0
            self._episode.start()
            # A copy: the caller may retain this observation while step() goes
            # on to mutate self._episode.transcript in place.
            return list(self._episode.transcript), {}

        def step(self, action):
            self._steps += 1
            self._episode.transcript.append(action)
            if action.get("tool_calls"):
                self._episode.transcript.extend(
                    self._episode.run_tool_calls(action["tool_calls"]))
                if self._steps >= self._task.max_steps:
                    # match rollout(): score the verifier at the step budget
                    reward = float(self._task.reward(self._episode.state("")))
                    return list(self._episode.transcript), reward, False, True, {}
                return list(self._episode.transcript), 0.0, False, False, {}
            answer = action.get("content") or ""
            reward = float(self._task.reward(self._episode.state(answer)))
            return list(self._episode.transcript), reward, True, False, {}

        def close(self):
            if self._world is not None:
                self._backend.down(self._world)
                self._world = None

    return OpencrlEnv()

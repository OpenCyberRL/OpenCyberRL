"""Gym adapter: expose a Task as a gymnasium.Env, reusing the Episode primitive.

gymnasium is imported lazily so the core never depends on it.
"""
from __future__ import annotations

from cyberl.backend import resolve_backend
from cyberl.rollout import Episode
from cyberl.task import Task, load_world


def to_gym(task: Task, backend=None):
    import gymnasium as gym

    class CyberlEnv(gym.Env):
        def __init__(self):
            self._task = task
            self._backend = resolve_backend(backend or task.backend)
            self._episode = None
            self._world = None
            self._steps = 0
            # Actions are OpenAI-style assistant-message dicts and observations are
            # message transcripts — neither fits a standard numeric Gym space, so these
            # are permissive placeholders provided so the env satisfies the Gym API.
            self.action_space = gym.spaces.Text(max_length=1_000_000)
            self.observation_space = gym.spaces.Sequence(gym.spaces.Text(max_length=1_000_000))

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            if self._world is not None:      # tear down a prior episode's world
                self._backend.down(self._world)
            self._world = self._backend.up(load_world(self._task), self._task.caps)
            self._episode = Episode(self._task, self._world)
            self._steps = 0
            return self._episode.start(), {}

        def step(self, action):
            self._steps += 1
            self._episode.transcript.append(action)
            if action.get("tool_calls"):
                self._episode.transcript.extend(
                    self._episode.run_tool_calls(action["tool_calls"]))
                if self._steps >= self._task.max_steps:
                    # match rollout(): score the verifier at the step budget
                    reward = float(self._task.reward(self._episode.state("")))
                    return self._episode.transcript, reward, False, True, {}
                return self._episode.transcript, 0.0, False, False, {}
            answer = action.get("content") or ""
            reward = float(self._task.reward(self._episode.state(answer)))
            return self._episode.transcript, reward, True, False, {}

        def close(self):
            if self._world is not None:
                self._backend.down(self._world)
                self._world = None

    return CyberlEnv()

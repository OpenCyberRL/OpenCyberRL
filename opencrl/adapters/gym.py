"""Gym adapter: expose a Task as a gymnasium.Env, reusing the Episode primitive.

gymnasium is imported lazily so the core never depends on it.
"""
from __future__ import annotations

from opencrl.backend import resolve_backend
from opencrl.reward import resolve_reward
from opencrl.rollout import Episode
from opencrl.task import Task, load_world


def to_gym(task: Task, backend=None, _resolved_backend=None):
    import gymnasium as gym
    from opencrl.adapters._gym_space import MessageSpace

    class OpencrlEnv(gym.Env):
        def __init__(self):
            self._task = task
            self._backend = _resolved_backend or resolve_backend(backend or task.backend)
            self._episode = None
            self._world = None
            self._steps = 0
            # Actions are OpenAI-style assistant-message dicts and observations are
            # message transcripts — neither fits a standard numeric Gym space, so these
            # are permissive placeholders provided so the env satisfies the Gym API.
            self.action_space = MessageSpace()
            self.observation_space = MessageSpace(sequence=True)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            if self._world is not None:      # tear down a prior episode's world
                self._backend.down(self._world)
                self._world = None
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
                    reward, _ = resolve_reward(self._task.reward(self._episode.state("")))
                    return list(self._episode.transcript), reward, False, True, {}
                return list(self._episode.transcript), 0.0, False, False, {}
            answer = action.get("content") or ""
            reward, _ = resolve_reward(self._task.reward(self._episode.state(answer)))
            return list(self._episode.transcript), reward, True, False, {}

        def close(self):
            if self._world is not None:
                self._backend.down(self._world)
                self._world = None

    return OpencrlEnv()


def to_gym_vector(task: Task, num_envs: int, backend=None, async_mode: bool = False):
    """A Gymnasium vector env of `num_envs` OpencrlEnvs sharing ONE resolved
    backend, so build-once applies across the batch (no per-env herd build)."""
    import gymnasium as gym
    resolved = resolve_backend(backend or task.backend)
    prebuild = getattr(resolved, "prebuild", None)
    if prebuild is not None:
        prebuild(load_world(task), task.caps)
    make = lambda: to_gym(task, _resolved_backend=resolved)
    if async_mode:
        # observations are Python message objects, not arrays — gym has no
        # shared-memory handler for this custom space, so disable it.
        return gym.vector.AsyncVectorEnv([make for _ in range(num_envs)],
                                         shared_memory=False)
    return gym.vector.SyncVectorEnv([make for _ in range(num_envs)])

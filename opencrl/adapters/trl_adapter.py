"""TRL on-policy adapter: wrap an OpenCyberRL Task as a TRL GRPO environment."""
from __future__ import annotations

from opencrl.adapters._common import (
    _score_with_stages, _extract_answer, _SYSTEM_PROMPT,
)
from opencrl.backend import resolve_backend
from opencrl.rollout import Episode
from opencrl.state import State
from opencrl.task import Task, load_world


def to_trl(task: Task, backend=None):
    """Return (environment_factory, reward_func) for TRL GRPOTrainer.

    The environment manages sandbox lifecycle: backend.up() on reset(),
    backend.down() on close(). get_reward() scores the episode via the
    task's reward function with full State (world + transcript + answer).

    The reward_func is for transcript-only rewards (flag, contains) where
    world access isn't needed. For world-dependent rewards (file_exists,
    exec), use the environment's get_reward() path.

    Requires the `trl` extra: pip install opencrl[trl]
    """
    resolved_backend = resolve_backend(backend or task.backend)

    class OpenCRLREnvironment:
        """TRL environment backed by an OpenCyberRL sandboxed world."""

        def __init__(self):
            self._world = None
            self._episode = None

        def reset(self):
            """Stand up a fresh world and return the task goal as the prompt."""
            self._world = resolved_backend.up(load_world(task), task.caps)
            self._episode = Episode(task, self._world)
            self._episode.start()
            return task.goal

        def get_reward(self):
            """Score the completed episode. Must be called before close()."""
            answer = _extract_answer(self._episode.transcript)
            reward, _ = _score_with_stages(task, self._episode.state(answer))
            return reward

        def close(self):
            """Tear down the sandbox world."""
            if self._world is not None:
                resolved_backend.down(self._world)
                self._world = None

    def reward_func(prompts, completions, completion_ids=None, **kwargs):
        """TRL GRPO reward function for transcript-only rewards.

        Logs stage breakdown via log_extra if available.
        """
        results = []
        log_extra = kwargs.get("log_extra", None)
        for prompt, completion in zip(prompts, completions):
            transcript = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            if isinstance(completion, list):
                transcript.extend(completion)
            else:
                transcript.append(
                    {"role": "assistant", "content": str(completion),
                     "tool_calls": None})
            answer = _extract_answer(transcript)
            state = State(world=None, transcript=transcript, answer=answer)
            reward, stages = _score_with_stages(task, state)
            if log_extra and stages:
                log_extra({"stage_breakdown": stages})
            results.append(reward)
        return results

    return OpenCRLREnvironment, reward_func

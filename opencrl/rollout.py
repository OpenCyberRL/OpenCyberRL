"""The rollout loop: drive a model through tool calls until it answers, then score."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from opencrl.backend import resolve_backend
from opencrl.reward import resolve_reward
from opencrl.state import State
from opencrl.task import Caps, Task, load_world


@dataclass
class Rollout:
    """The one standard artifact every adapter consumes."""
    task: str
    transcript: list[dict]
    reward: float
    caps: Caps
    stages: dict[str, float] | None = None

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "transcript": self.transcript,
            "reward": self.reward,
            "caps": asdict(self.caps),
            "stages": self.stages,
        }


class Episode:
    """Holds the running world + transcript; knows how to run one tool call."""

    def __init__(self, task: Task, world):
        self.task = task
        self.world = world
        self.tools = {t.name: t for t in task.tools}
        self.transcript: list[dict] = []

    def start(self) -> list[dict]:
        system = (
            "You are a security agent working inside an isolated sandbox. "
            "Use the tools to accomplish the goal, then state your final answer."
        )
        self.transcript = [
            {"role": "system", "content": system},
            {"role": "user", "content": self.task.goal},
        ]
        return self.transcript

    def run_tool_calls(self, tool_calls: list[dict]) -> list[dict]:
        results = []
        for call in tool_calls:
            try:
                fn = call["function"]
                tool = self.tools[fn["name"]]
                args = json.loads(fn.get("arguments") or "{}")
                output = tool.run(self.world, **args)
            except Exception as e:
                output = f"[opencrl: tool call failed: {e}]"
            results.append(
                {"role": "tool", "tool_call_id": call.get("id", ""), "content": output}
            )
        return results

    def state(self, answer: str) -> State:
        return State(world=self.world, transcript=self.transcript, answer=answer)


def rollout(task: Task, model, backend=None) -> Rollout:
    backend = resolve_backend(backend or task.backend)
    world = backend.up(load_world(task), task.caps)
    try:
        episode = Episode(task, world)
        messages = episode.start()
        tool_schemas = [t.openai_schema() for t in task.tools]
        answer = ""
        for _ in range(task.max_steps):
            reply = model(messages, tool_schemas)
            messages.append(reply)
            if reply.get("tool_calls"):
                messages.extend(episode.run_tool_calls(reply["tool_calls"]))
                continue
            answer = reply.get("content") or ""
            break
        reward, stages = resolve_reward(task.reward(episode.state(answer)))
        return Rollout(task.name, episode.transcript, reward, task.caps, stages)
    finally:
        backend.down(world)

"""`opencrl run <group>`: score a whole task group through the persistent pool.

Resolves a group expression against the discovered task index (reusing
opencrl.warm.build_index's derivation), then runs `episodes` rollouts per
task over one shared GroupPool: warm idle worlds are reset and reused, cold
tasks are brought up under the pool's LRU eviction policy. Every successful
episode emits one standard Rollout — with its per-stage score breakdown —
to a JSONL file, one Rollout.to_dict() per line, in submission order. A task
that fails anywhere (resolution, hook validation, build, bring-up, rollout,
scoring) is reported with its error and skipped; the rest of the group still
runs, matching opencrl.warm's failure posture.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from opencrl.groups import GroupExpr, TaskIndex, resolve_group
from opencrl.rollout import Rollout
from opencrl.runner import run_group
from opencrl.task import get_task
from opencrl.warm import _reason, build_index


@dataclass(frozen=True)
class TaskOutcome:
    """One task's slice of a group run: its episodes, or why it never ran."""
    task: str
    rollouts: list[Rollout]      # successful episodes, in submission order
    error: str | None = None     # first failure; None when every episode ran


@dataclass(frozen=True)
class GroupEvalResult:
    """A whole group run: one TaskOutcome per task, in submission order."""
    group: str                   # the group expression as given
    outcomes: list[TaskOutcome]

    @property
    def rollouts(self) -> list[Rollout]:
        """Every successful episode's Rollout, in submission order."""
        return [r for o in self.outcomes for r in o.rollouts]

    def write_jsonl(self, path: "str | Path") -> int:
        """Write one Rollout.to_dict() per line; return the line count."""
        lines = [json.dumps(r.to_dict()) for r in self.rollouts]
        Path(path).write_text("".join(line + "\n" for line in lines),
                              encoding="utf-8")
        return len(lines)

    def summarize(self) -> str:
        """One-line tally, e.g. ``"3 tasks, 6 episodes, 1 failed"``."""
        episodes = sum(len(o.rollouts) for o in self.outcomes)
        failed = sum(1 for o in self.outcomes if o.error)
        parts = [f"{len(self.outcomes)} tasks, {episodes} episodes"]
        if failed:
            parts.append(f"{failed} failed")
        return ", ".join(parts)


def run_group_eval(expr: GroupExpr, model=None, *, episodes: int = 1,
                   index: TaskIndex | None = None, path: str | None = None,
                   backend=None, concurrency: int | None = None,
                   capacity: int | None = None,
                   output: "str | Path | None" = None) -> GroupEvalResult:
    """Run a whole group through one shared persistent pool (module docstring).

    `expr` resolves against `index`, defaulting to warm.build_index(path)'s
    discovered-task derivation. Each resolved task runs `episodes` rollouts
    on the pool; `backend` defaults to the tasks' own (pass one to override).
    Pass `output` to also write the JSONL artifact. Returns a GroupEvalResult
    in submission order; a task that fails is reported in its TaskOutcome,
    never by raising.
    """
    if episodes < 1:
        raise ValueError(f"opencrl: episodes must be >= 1, got {episodes}")
    idx = build_index(path) if index is None else index
    names = resolve_group(expr, idx)
    outcomes: dict[str, TaskOutcome] = {}
    loaded: list = []
    for name in names:                     # a stale index name is a report, not a crash
        try:
            loaded.append(get_task(name))
        except Exception as e:
            outcomes[name] = TaskOutcome(name, [], _reason(e))
    errors: dict[str, str] = {}

    def on_error(task, exc: BaseException) -> None:
        errors.setdefault(task.name, _reason(exc))

    results = run_group(loaded, model, episodes=episodes,
                        concurrency=concurrency, backend=backend,
                        on_error=on_error, capacity=capacity)
    by_task: dict[str, list[Rollout]] = {}
    episodes_of = [t for t in loaded for _ in range(episodes)]
    for task, r in zip(episodes_of, results):      # results are submission-ordered
        if r is not None:
            by_task.setdefault(task.name, []).append(r)
    result = GroupEvalResult(_render(expr), [
        outcomes.get(name) or TaskOutcome(name, by_task.get(name, []),
                                          errors.get(name))
        for name in names
    ])
    if output is not None:
        result.write_jsonl(output)
    return result


def _render(expr: GroupExpr) -> str:
    """The expression as a printable group label."""
    return expr if isinstance(expr, str) else ", ".join(str(e) for e in expr)

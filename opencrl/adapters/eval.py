"""Eval adapter: run N rollouts (optionally concurrent), write JSONL, report stats."""
from __future__ import annotations

import json
from collections import defaultdict

from opencrl.runner import run_batch
from opencrl.task import Task


def evaluate(task: Task, model=None, *, model_factory=None, n: int = 1,
             out: str | None = None, backend=None,
             concurrency: int | None = None) -> dict:
    rewards: list[float] = [0.0] * n
    stage_sums: dict[str, float] = defaultdict(float)
    stage_counts: dict[str, int] = defaultdict(int)
    handle = open(out, "w") if out else None

    def on_result(i: int, r) -> None:        # called under the runner's lock
        rewards[i] = r.reward
        for name, s in (r.stages or {}).items():
            stage_sums[name] += s
            stage_counts[name] += 1
        if handle:
            handle.write(json.dumps(r.to_dict()) + "\n")
            handle.flush()

    try:
        run_batch(task, model, model_factory=model_factory, n=n,
                  concurrency=concurrency, backend=backend, on_result=on_result)
    finally:
        if handle:
            handle.close()

    mean = sum(rewards) / len(rewards) if rewards else 0.0
    stage_means = {name: stage_sums[name] / stage_counts[name] for name in stage_sums}
    return {"n": n, "mean_reward": mean, "rewards": rewards, "stage_means": stage_means}

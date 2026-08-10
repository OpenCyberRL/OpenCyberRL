"""Eval adapter: run N rollouts, write JSONL, report reward stats."""
from __future__ import annotations

import json
from collections import defaultdict

from opencrl.rollout import rollout
from opencrl.task import Task


def evaluate(task: Task, model, n: int = 1, out: str | None = None,
             backend=None) -> dict:
    rewards = []
    stage_sums: dict[str, float] = defaultdict(float)
    stage_counts: dict[str, int] = defaultdict(int)
    handle = open(out, "w") if out else None
    try:
        for _ in range(n):
            r = rollout(task, model, backend=backend)
            rewards.append(r.reward)
            for name, s in (r.stages or {}).items():
                stage_sums[name] += s
                stage_counts[name] += 1
            if handle:
                handle.write(json.dumps(r.to_dict()) + "\n")
    finally:
        if handle:
            handle.close()
    mean = sum(rewards) / len(rewards) if rewards else 0.0
    stage_means = {name: stage_sums[name] / stage_counts[name]
                   for name in stage_sums}
    return {"n": n, "mean_reward": mean, "rewards": rewards,
            "stage_means": stage_means}

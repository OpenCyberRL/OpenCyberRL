"""Offline export: convert OpenCyberRL Rollouts to HuggingFace Datasets."""
from __future__ import annotations

from opencrl.adapters._common import _extract_prompt, _extract_completion
from opencrl.rollout import Rollout, rollout
from opencrl.task import Task


def export_rollouts(task: Task, model, n: int = 16, backend=None,
                   fmt: str = "dpo") -> "Dataset":
    """Run N rollouts and export as a HuggingFace Dataset.

    fmt: "dpo"  -> {prompt, chosen, rejected} preference pairs
          "sft"  -> {prompt, completion} high-reward trajectories
          "kto"  -> {prompt, completion, label} unpaired preferences
    """
    rollouts = [rollout(task, model, backend=backend) for _ in range(n)]
    if fmt == "dpo":
        return _to_dpo(task, rollouts)
    elif fmt == "sft":
        return _to_sft(task, rollouts)
    elif fmt == "kto":
        return _to_kto(task, rollouts)
    else:
        raise ValueError(f"opencrl: unknown format {fmt!r}; use dpo/sft/kto")


def _to_dpo(task: Task, rollouts: list[Rollout]):
    """Pair high-reward rollouts (chosen) with low-reward (rejected)."""
    from datasets import Dataset
    ranked = sorted(rollouts, key=lambda r: r.reward, reverse=True)
    pairs = []
    for i in range(0, len(ranked) - 1, 2):
        chosen_r, rejected_r = ranked[i], ranked[i + 1]
        pairs.append({
            "prompt": _extract_prompt(task),
            "chosen": _extract_completion(chosen_r.transcript),
            "rejected": _extract_completion(rejected_r.transcript),
        })
    return Dataset.from_list(pairs)


def _to_sft(task: Task, rollouts: list[Rollout]):
    """Keep only rollouts with reward >= 1.0."""
    from datasets import Dataset
    good = [r for r in rollouts if r.reward >= 1.0]
    rows = [{
        "prompt": _extract_prompt(task),
        "completion": _extract_completion(r.transcript),
    } for r in good]
    return Dataset.from_list(rows)


def _to_kto(task: Task, rollouts: list[Rollout]):
    """Each rollout becomes {prompt, completion, label} where label = reward >= 0.5."""
    from datasets import Dataset
    rows = [{
        "prompt": _extract_prompt(task),
        "completion": _extract_completion(r.transcript),
        "label": r.reward >= 0.5,
    } for r in rollouts]
    return Dataset.from_list(rows)

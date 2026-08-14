"""Offline export: convert OpenCyberRL Rollouts to HuggingFace Datasets."""
from __future__ import annotations

from opencrl.adapters._common import _extract_prompt, _extract_completion, _extract_assistant_only
from opencrl.rollout import Rollout
from opencrl.runner import run_batch
from opencrl.task import Task


def export_rollouts(task: Task, model=None, n: int = 16, backend=None,
                    fmt: str = "dpo", *, model_factory=None,
                    concurrency: int | None = None) -> "Dataset":
    """Run N rollouts and export as a HuggingFace Dataset.

    fmt: "dpo"  -> {prompt, chosen, rejected} preference pairs
          "sft"  -> {prompt, completion} high-reward trajectories
          "kto"  -> {prompt, completion, label} unpaired preferences
    """
    rollouts = run_batch(task, model, model_factory=model_factory, n=n,
                         concurrency=concurrency, backend=backend)
    if fmt == "dpo":
        return _to_dpo(task, rollouts)
    elif fmt == "sft":
        return _to_sft(task, rollouts)
    elif fmt == "kto":
        return _to_kto(task, rollouts)
    else:
        raise ValueError(f"opencrl: unknown format {fmt!r}; use dpo/sft/kto")


def _to_dpo(task: Task, rollouts: list[Rollout]):
    """Pair high-reward rollouts (chosen) with low-reward (rejected).

    Pairs highest with lowest for maximum preference contrast.
    Filters out pairs where both rollouts have equal reward.
    """
    from datasets import Dataset
    ranked = sorted(rollouts, key=lambda r: r.reward, reverse=True)
    pairs = []
    for i in range(len(ranked) // 2):
        chosen_r = ranked[i]
        rejected_r = ranked[-(i + 1)]
        if chosen_r.reward == rejected_r.reward:
            continue  # skip equal-reward pairs (no preference signal)
        pairs.append({
            "prompt": _extract_prompt(task),
            "chosen": _extract_assistant_only(chosen_r.transcript),
            "rejected": _extract_assistant_only(rejected_r.transcript),
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

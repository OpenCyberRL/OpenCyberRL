"""Verifier helpers. Each returns a plain function State -> float in [0, 1].

Staged rewards (chain/goals) instead return a Score{value, stages}: an
aggregate in [0, 1] plus the per-stage breakdown of what each tier earned.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from opencrl.state import State


def flag(expected: str) -> Callable[[State], float]:
    """Reward 1.0 when the agent's final answer contains the flag string."""
    return lambda s: 1.0 if expected in (s.answer or "") else 0.0


def contains(text: str) -> Callable[[State], float]:
    """Reward 1.0 when text appears anywhere in the transcript's text content."""
    def check(s: State) -> float:
        blob = " ".join(str(m.get("content") or "") for m in s.transcript)
        return 1.0 if text in blob else 0.0
    return check


def file_exists(path: str, host: str | None = None) -> Callable[[State], float]:
    """Reward 1.0 when the file exists in the world."""
    return lambda s: 1.0 if s.file(path, host) is not None else 0.0


@dataclass(frozen=True)
class Score:
    """A staged reward: the aggregate in [0, 1] plus each stage's own score."""
    value: float
    stages: dict[str, float]        # ordered {stage name: score in [0, 1]}


@dataclass(frozen=True)
class Stage:
    """One named sub-task. Built with stage(); consumed by chain()/goals()."""
    name: str
    check: Callable[[State], float]
    weight: float


def stage(name: str, check: Callable[[State], float], weight: float = 1.0) -> Stage:
    """Declare a named sub-task scored by `check` (a State -> float in [0, 1]).

    `weight` is relative; chain()/goals() normalize weights against their sum,
    so equal weights (the default) split the reward evenly.
    """
    if not name or not name.strip():
        raise ValueError("opencrl: stage name must be a non-empty string")
    if not callable(check):
        raise TypeError("opencrl: stage check must be callable (State -> float)")
    if weight <= 0:
        raise ValueError(f"opencrl: stage {name!r} weight must be > 0, got {weight}")
    return Stage(name, check, float(weight))


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else float(x)


def _validate(stages: tuple[Stage, ...]) -> None:
    if not stages:
        raise ValueError("opencrl: chain()/goals() need at least one stage")
    names = [st.name for st in stages]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"opencrl: duplicate stage names: {dupes}")


def _evaluate(stages: tuple[Stage, ...], state: State, *, gated: bool) -> Score:
    total = sum(st.weight for st in stages)     # > 0: validated at build time
    value = 0.0
    breakdown: dict[str, float] = {}
    open_gate = True
    for st in stages:
        if gated and not open_gate:
            breakdown[st.name] = 0.0            # locked: not even evaluated
            continue
        s = _clamp01(st.check(state))
        breakdown[st.name] = s
        value += (st.weight / total) * s
        if gated and s < 1.0:
            open_gate = False                   # a chain breaks at its weakest link
    return Score(value, breakdown)


def chain(*stages: Stage) -> Callable[[State], Score]:
    """Ordered, gated kill-chain: credit stops at the first tier not fully cleared."""
    _validate(stages)
    return lambda state: _evaluate(stages, state, gated=True)


def goals(*stages: Stage) -> Callable[[State], Score]:
    """Independent sub-goals: every stage scores on its own; shares are summed."""
    _validate(stages)
    return lambda state: _evaluate(stages, state, gated=False)


def resolve_reward(scored: float | Score) -> tuple[float, dict[str, float] | None]:
    """Split a reward function's result into (aggregate float, optional breakdown).

    A Score yields (its value, a copy of its stage breakdown); a plain float
    yields (float(scored), None). Every reward consumer — the rollout loop and
    the gym adapter — routes through here so the float|Score contract lives in
    one place.
    """
    if isinstance(scored, Score):
        return scored.value, dict(scored.stages)
    return float(scored), None

"""The Task contract: what a contributor writes."""
from __future__ import annotations

import importlib.util
import inspect
import yaml
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from cyberl.backend import Backend
    from cyberl.state import State
    from cyberl.tools import Tool


@dataclass(frozen=True)
class Caps:
    """Capability metadata used for safety filtering and network policy."""
    offensive: bool = False
    needs_internet: bool = False


@dataclass(frozen=True)
class Task:
    """A self-contained cyber task. Frozen: no methods, no lifecycle."""
    goal: str
    reward: "Callable[[State], float]"
    world: "str | dict | None" = None
    backend: "str | Backend" = "docker"
    tools: tuple = ()
    caps: Caps = field(default_factory=Caps)
    max_steps: int = 30
    agent: str = ""          # which service the agent's shell runs on ("" = sole/first)
    name: str = ""           # filled by @task
    dir: str = ""            # filled by @task (dir of the defining file)


_REGISTRY: dict[str, Callable[[], Task]] = {}


def task(fn=None, *, name=None):
    """Register a task factory. Fills Task.name and Task.dir automatically."""
    def wrap(factory):
        task_name = name or factory.__name__
        src_dir = str(Path(inspect.getfile(factory)).resolve().parent)

        def build() -> Task:
            return replace(factory(), name=task_name, dir=src_dir)

        _REGISTRY[task_name] = build
        return build

    return wrap(fn) if fn is not None else wrap


def get_task(name: str) -> Task:
    if name not in _REGISTRY:
        raise KeyError(f"unknown task {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def list_tasks() -> list[str]:
    return sorted(_REGISTRY)


def discover(path="tasks") -> None:
    """Import every tasks/<name>/task.py so their @task decorators register."""
    root = Path(path)
    for task_py in sorted(root.glob("*/task.py")):
        spec = importlib.util.spec_from_file_location(
            f"cyberl_tasks.{task_py.parent.name}", task_py
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


def load_world(task: "Task") -> dict:
    """Resolve a Task's world spec to a dict.

    A dict is returned unchanged; a string is read as YAML relative to the
    task's source directory (Task.dir); None becomes an empty spec.
    """
    world = task.world
    if world is None:
        return {}
    if isinstance(world, dict):
        return world
    with open(Path(task.dir) / world) as f:
        return yaml.safe_load(f) or {}

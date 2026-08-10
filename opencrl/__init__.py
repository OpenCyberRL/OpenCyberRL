"""OpenCyberRL — plug-and-play cybersecurity RL task-building framework.

The entire public API lives here. `import opencrl` and read this file to
understand the whole surface.
"""
from opencrl.task import Caps, Task, task, get_task, list_tasks, discover
from opencrl.tools import Tool, shell
from opencrl.state import State
from opencrl.reward import flag, contains, file_exists, stage, chain, goals, Score
from opencrl.backend import register_backend, resolve_backend
from opencrl.backends.mock import MockBackend
from opencrl.backends.docker import Docker
from opencrl.backends.qemu import Qemu
from opencrl.backends.sandbox import Sandbox
from opencrl.models import ScriptedModel, OpenAIModel
from opencrl.rollout import rollout, Rollout
from opencrl.adapters.eval import evaluate
from opencrl.adapters.gym import to_gym

__all__ = [
    "task", "Task", "Caps", "Tool", "shell",
    "flag", "contains", "file_exists", "stage", "chain", "goals", "Score",
    "rollout", "Rollout", "State", "evaluate", "to_gym",
    "Docker", "Qemu", "Sandbox", "MockBackend", "ScriptedModel", "OpenAIModel",
    "register_backend", "resolve_backend",
    "get_task", "list_tasks", "discover",
]

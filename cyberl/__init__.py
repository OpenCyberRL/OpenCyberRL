"""OpenCyberRL — plug-and-play cybersecurity RL task-building framework.

The entire public API lives here. `import cyberl` and read this file to
understand the whole surface.
"""
from cyberl.task import Caps, Task, task, get_task, list_tasks, discover
from cyberl.tools import Tool, shell
from cyberl.state import State
from cyberl.reward import flag, contains, file_exists
from cyberl.backend import register_backend, resolve_backend
from cyberl.backends.mock import MockBackend
from cyberl.backends.docker import Docker
from cyberl.models import ScriptedModel, OpenAIModel
from cyberl.rollout import rollout, Rollout
from cyberl.adapters.eval import evaluate
from cyberl.adapters.gym import to_gym

__all__ = [
    "task", "Task", "Caps", "Tool", "shell",
    "flag", "contains", "file_exists",
    "rollout", "Rollout", "State", "evaluate", "to_gym",
    "Docker", "MockBackend", "ScriptedModel", "OpenAIModel",
    "register_backend", "resolve_backend",
    "get_task", "list_tasks", "discover",
]

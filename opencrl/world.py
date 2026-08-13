"""World: a handle to the running, sandboxed environment."""
from __future__ import annotations

from typing import Protocol


class World(Protocol):
    """What every backend returns from up(). Duck-typed; kept tiny on purpose."""
    agent: str  # name of the host the agent's shell runs on

    def exec(self, command: str, host: str | None = None) -> str:
        """Run a command (on the agent host by default) and return combined
        stdout and stderr as a single string."""
        ...

    def read_file(self, path: str, host: str | None = None) -> str | None:
        """Return file contents as text, or None if it does not exist."""
        ...

"""State: the read-only view of a finished episode handed to the verifier."""
from __future__ import annotations

from dataclasses import dataclass

from cyberl.world import World


@dataclass
class State:
    """Verifiers read this. They cannot mutate the world."""
    world: World
    transcript: list[dict]
    answer: str               # the agent's final answer (last assistant content)

    def file(self, path: str, host: str | None = None) -> str | None:
        return self.world.read_file(path, host)

    def exec(self, command: str, host: str | None = None) -> str:
        return self.world.exec(command, host)

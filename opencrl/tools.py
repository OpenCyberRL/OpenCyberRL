"""Tools an agent can call. A Tool is a name + JSON schema + a run() function."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    schema: dict                      # JSON schema for the tool's parameters
    run: Callable[..., str]           # run(world, **args) -> str

    def openai_schema(self) -> dict:
        """The tool as the model's OpenAI-style tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema,
            },
        }


shell = Tool(
    name="shell",
    description="Run a shell command on your host and return its output "
               "(stdout and stderr combined).",
    schema={
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
    run=lambda world, command: world.exec(command),
)

# Write a task

A task is a frozen dataclass, `Task`. The `@task` decorator registers a factory
that returns one. Registration lets the CLI and the test suite find the task by
name.

```python
from opencrl import task, Task, shell, flag, Caps


@task
def read_flag() -> Task:
    return Task(
        goal="Read the flag and state it as your final answer.",
        tools=[shell],
        reward=flag("CTF{demo}"),
        world="world.yml",
        backend="docker",
        caps=Caps(offensive=True),
    )
```

## Fields

Set every field when you build the `Task`. The dataclass is frozen.

| Field | Type | Meaning |
|---|---|---|
| `goal` | `str` | The objective. The agent receives it as the first user message. |
| `reward` | `Callable[[State], float \| Score]` | The verifier. It scores the finished episode. See [Rewards and scoring](guide-rewards.md). |
| `world` | `str \| dict \| None` | The environment. A string is a path to a YAML file, resolved against the task's own directory. A dict is used as is. `None` is an empty world. |
| `backend` | `str \| Backend` | The backend that starts the world. Defaults to `"docker"`. |
| `tools` | `tuple` | The tools the agent can call. Usually `[shell]`. |
| `caps` | `Caps` | Capability metadata: `offensive` and `needs_internet`. It sets network policy and travels with every `Rollout`. |
| `max_steps` | `int` | The maximum number of agent turns. Defaults to 30. |
| `name` | `str` | Set by `@task`. Do not set it yourself. |
| `dir` | `str` | Set by `@task`. Do not set it yourself. |

## The world

The world is the sandbox the agent works in. Give it in one of two ways:

- **A file path**, `world="world.yml"`. The framework reads the file relative to
  the task's directory. Use this for build steps or multiple hosts. See
  [Docker](backend-docker.md).
- **An inline dict**, `world={...}`. The framework uses the dict as is. Use this
  for short, self-contained tasks.

## Tools

A tool is a name, a description, a JSON schema, and a `run` function. The
built-in `shell` tool runs a command in the world and returns its output.

To add a tool, build a `Tool` and pass it in `tools`:

```python
from opencrl import Tool

read = Tool(
    name="read_file",
    description="Read a file from the host.",
    schema={"type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]},
    run=lambda world, path: world.read_file(path) or "",
)
```

`run` receives the world first, then the tool's parameters as keyword
arguments. It returns a string.

## Capabilities

`Caps` records two facts about a task:

- `offensive`: the task involves offensive security.
- `needs_internet`: the world needs external network access.

`caps` sets network policy. The Docker backend blocks all external egress unless
`needs_internet` is `True`. `caps` also rides along in every `Rollout`, so
downstream consumers can filter or audit what ran.

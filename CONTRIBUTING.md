# Contributing

## Write your first task

```bash
uv run cyberl new mytask
```

This scaffolds `tasks/mytask/task.py` and `tasks/mytask/world.yml`:

```python
# tasks/mytask/task.py
from cyberl import task, Task, shell, flag, Caps


@task
def mytask() -> Task:
    return Task(
        world="world.yml",
        backend="docker",
        tools=[shell],
        goal="Read the flag and state it as your final answer.",
        reward=flag("CTF{change-me}"),
        caps=Caps(offensive=True, needs_internet=False),
    )
```

```yaml
# tasks/mytask/world.yml
x-cyberl:
  agent: box
services:
  box:
    image: alpine:3.20
    command: sh -c "echo CTF{change-me} > /root/flag.txt && sleep 600"
```

Edit the goal, the flag, and the world, then wire up a real verifier. That's
the whole authoring surface — a `Task` is a plain frozen dataclass, and
`@task` registers the factory function so the CLI and test suite can find it
by name.

## The `Task` fields

`Task` (`cyberl/task.py`) is frozen — set every field at construction time:

| Field | Type | Meaning |
|---|---|---|
| `goal` | `str` | The objective, given to the agent as the initial user message. |
| `reward` | `Callable[[State], float]` | Verifier: scores the finished episode. Build one with `flag`, `contains`, `file_exists`, or write your own — it just needs to take a `State` and return a float. |
| `world` | `str \| dict \| None` | The environment. A string is a path to a YAML file resolved relative to the task's own directory; a dict is used as-is; `None` is an empty world. |
| `backend` | `str \| Backend` | Which backend stands the world up — `"docker"` by default, or an already-configured backend instance (e.g. `Docker(cpus=1.0)`). |
| `tools` | `tuple` | The `Tool` objects the agent may call — usually `[shell]`. |
| `caps` | `Caps` | Capability metadata: `offensive` (bool) and `needs_internet` (bool). Drives network policy and rides along in every `Rollout`. |
| `max_steps` | `int` | Max agent turns before the episode ends unanswered. Defaults to 30; the reference tasks use 8. |
| `name` | `str` | Filled in automatically by `@task` (the function name, or an explicit `name=` passed to the decorator). Don't set it yourself. |
| `dir` | `str` | Filled in automatically by `@task`: the directory of the file that defined the task, used to resolve a string `world:` path. Don't set it yourself. |

## The `world.yml` shape

`world.yml` is a Docker Compose file with one extra top-level key:

```yaml
x-cyberl:
  agent: attacker      # the service world.exec()/read_file() target by default
services:
  attacker:
    build: build/attacker   # built from source, not pulled from a registry
  web:
    build: build/web
```

Everything under `services:` is regular Compose — `build:`, `command:`,
`environment:`, etc. Build contexts (`build/...`) are resolved relative to
the task's directory, so `docker compose` can be invoked from anywhere.

To segment hosts from each other, declare `networks:` and list each
service's memberships explicitly:

```yaml
x-cyberl:
  agent: attacker
networks:
  edge:
  backend:
services:
  attacker:
    build: build/attacker
    networks: [edge]
  web:
    build: build/web
    networks: [edge, backend]      # bridges both — the only path from edge to backend
  internal:
    build: build/internal
    networks: [backend]
```

The docker backend forces `internal: true` on every network — declared or
the implicit default one services get if they list none — unless the task's
`caps.needs_internet` is true. So `internal` above is unreachable from
`attacker` except through whatever `web` exposes.

## Conformance: every task ships a reference solution

Every task needs `tasks/mytask/test_task.py` with a `ScriptedModel` that
plays out a known-good solution and must score reward `1.0`:

```python
import json
from pathlib import Path
import pytest
from cyberl import rollout, get_task
from cyberl.task import discover
from cyberl.models import ScriptedModel

pytestmark = pytest.mark.docker

def test_reference_solution_scores_one():
    discover(Path(__file__).resolve().parent.parent)
    model = ScriptedModel([
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "1", "type": "function",
                         "function": {"name": "shell",
                                      "arguments": json.dumps({"command": "..."})}}]},
        {"role": "assistant", "content": "The flag is CTF{change-me}",
         "tool_calls": None},
    ])
    r = rollout(get_task("mytask"), model)
    assert r.reward == 1.0
```

Run it (Docker tests are deselected by default — pass `-m docker` explicitly):

```bash
uv run pytest -m docker tasks/mytask/test_task.py
```

This is what proves a task is playable and correctly scored before it ships
— no task is "done" without a passing `test_task.py`.

## Worked examples — copy one of these

- **`tasks/web_sqli`** — single host pair (`attacker` + `web`), SQL
  injection via `curl`. The simplest realistic shape to copy for a
  single-vulnerability web task.
- **`tasks/privesc`** — one host, SUID-based privilege escalation. The
  simplest possible `world.yml` (a single `build:` service, no networking).
- **`tasks/lateral`** — three hosts across two segmented networks, pivoting
  through a web host that fetches URLs server-side to reach an
  internal-only target. Copy this for anything that needs network
  segmentation.

Read each task's `task.py`, `world.yml`, and `test_task.py` together — they're
short, and reading all three is the fastest way to understand the contract.

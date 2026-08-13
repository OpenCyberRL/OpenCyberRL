# CLI

The `opencrl` command manages task modules and scaffolds new tasks. Run it
with `uv run opencrl`.

## Commands

### `install`: install task modules

```bash
uv run opencrl install              # clone modules repo, list available modules
uv run opencrl install examples     # activate a module
```

With no argument, clones (or updates) the community modules repo and lists
all available modules. With a module name, activates that module so its tasks
are discoverable by `opencrl list` and the Python API.

Modules are cloned to `~/.opencrl/modules/opencyberrl-modules/`. The list of
active modules is stored in `~/.opencrl/active`.

### `uninstall`: deactivate a module

```bash
uv run opencrl uninstall examples
```

Removes the module from the active list. The cloned repo stays on disk.

### `update`: pull latest modules

```bash
uv run opencrl update
```

Runs `git pull` on the modules repo to fetch new tasks and updates.

### `info`: show task details

```bash
uv run opencrl info web_sqli
```

Shows the task's goal, backend, tools, capabilities, max steps, and world
spec in a detailed panel.
Runs `git pull` on the modules repo to fetch new tasks and updates.

### `list`: list available tasks

```bash
uv run opencrl list
```

Discovers and lists all tasks from active modules and local `tasks/`
directory. Shows each task's name and capabilities.

### `new`: scaffold a task

```bash
uv run opencrl new mytask
```

Creates `tasks/mytask/task.py` and `tasks/mytask/world.yml` in the local
`tasks/` directory. Refuses to overwrite an existing task.

## Options

| Option | Commands | Meaning |
|---|---|---|
| `--path` | `list` | Override task discovery path. Defaults to auto-discover (local `tasks/` + active modules). |

## Running tasks

The CLI handles discovery and module management. To run a task, use the
Python API:

```python
from opencrl import discover, get_task, rollout, OpenAIModel

discover()
r = rollout(get_task("web_sqli"), OpenAIModel("gpt-4o-mini"))
print(r.reward)
```

See [Run and evaluate](guide-run-eval.md) for batch evaluation and JSONL output.

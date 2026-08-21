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

### `run`: run a task group through the persistent pool

```bash
uv run opencrl run cybergym/level1 -n 4 -o rollouts.jsonl -m gpt-4o-mini
```

Resolves a group expression (or several) — same syntax as `opencrl warm` —
and runs every resolved task through one shared world pool: warm worlds are
reset and reused between episodes, cold tasks are brought up under the
pool's eviction policy. Writes one `Rollout` per episode to the JSONL file
(including the per-stage score breakdown). A task that fails is reported and
skipped; the rest of the group still runs. Exit codes match `warm`: 0 all
episodes ran, 1 a task failed, 2 the group expression could not be resolved.

The model id comes from `--model` or `$OPENCRL_MODEL` and is passed to the
OpenAI-compatible chat API (requires the `openai` extra). The equivalent
Python entry point is `run_group_eval()`:

```python
from opencrl import discover, run_group_eval, OpenAIModel

discover()
result = run_group_eval("cybergym/level1", OpenAIModel("gpt-4o-mini"),
                        episodes=4, output="rollouts.jsonl")
for outcome in result.outcomes:
    print(outcome.task, len(outcome.rollouts), outcome.error)
```

## Options

| Option | Commands | Meaning |
|---|---|---|
| `--path` | `list`, `warm`, `run` | Override task discovery path. Defaults to auto-discover (local `tasks/` + active modules). |

## Running single tasks

The CLI handles discovery and module management. To run a task, use the
Python API:

```python
from opencrl import discover, get_task, rollout, OpenAIModel

discover()
r = rollout(get_task("web_sqli"), OpenAIModel("gpt-4o-mini"))
print(r.reward)
```

See [Run and evaluate](guide-run-eval.md) for batch evaluation and JSONL output.

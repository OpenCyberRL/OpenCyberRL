# Install

OpenCyberRL needs Python 3.11 or later. The project uses
[uv](https://docs.astral.sh/uv/) for environment and package management.

## From source

Clone the repository and sync the environment:

```bash
git clone https://github.com/OpenCyberRL/OpenCyberRL.git
cd OpenCyberRL
uv sync
```

`uv sync` installs the package, PyYAML, and rich.

## Optional extras

Four features live behind optional extras:

| Extra | Adds | Needed for |
|---|---|---|
| `openai` | the `openai` client | `OpenAIModel` (for `rollout` and `evaluate` in Python) |
| `gym` | `gymnasium` | the Gym adapter, `to_gym()` |
| `trl` | `trl` | TRL on-policy GRPO adapter, `to_trl()` |
| `verl` | `verl`, `pandas` | Verl on-policy adapter, `to_verl()` |

Install any combination:

uv sync --extra openai --extra gym --extra trl --extra verl
```

## Backend requirements

A backend starts the task world. Install the backend you use:

- **Docker backend** (the default): needs Docker with the `docker compose` plugin.
- **QEMU backend**: needs `qemu-system-x86_64` on the host.
- **Mock backend**: needs nothing. It runs in memory, for tests.

## Verify the install

Run the fast test suite. It uses the mock backend, so it needs no Docker or
QEMU:

```bash
uv run pytest
```

## Install task modules

Tasks live in a separate community repo. Install them with the CLI:

```bash
opencrl install              # clone the modules repo and list available modules
opencrl install examples     # activate the examples module
opencrl list                 # list all available tasks
```

Modules are cloned to `~/.opencrl/modules/`. See [Run and evaluate](guide-run-eval.md)
for running tasks programmatically via the Python API.

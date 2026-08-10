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

`uv sync` installs the package and its one runtime dependency, PyYAML.

## Optional extras

Two features live behind optional extras:

| Extra | Adds | Needed for |
|---|---|---|
| `openai` | the `openai` client | `OpenAIModel`, and the `run` and `eval` CLI commands |
| `gym` | `gymnasium` | the Gym adapter, `to_gym()` |

Install one or both:

```bash
uv sync --extra openai --extra gym
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

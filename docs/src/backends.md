# Backends

A backend starts and stops the world. It implements two methods:

```python
class Backend:
    def up(self, spec: dict, caps: Caps) -> World: ...
    def down(self, world) -> None: ...
```

`up()` builds the world from the task's spec and returns a `World`. A `World`
runs commands and reads files:

```python
class World:
    agent: str                                       # the default target host
    def exec(self, command: str, host=None) -> str: ...
    def read_file(self, path: str, host=None) -> str | None: ...
```

## Built-in backends

| Backend | Select with | Use for |
|---|---|---|
| Docker | `"docker"` (default) | Containers: web, service, and multi-host tasks. See [Docker](backend-docker.md). |
| QEMU | `"qemu"` | Kernel targets that need a private kernel. See [QEMU](backend-qemu.md). |
| Mock | `"mock"` | In-memory tests. It runs no container. |

Select a backend by name, or pass a configured instance:

```python
from opencrl import Task, Docker

Task(..., backend="docker")          # by name
Task(..., backend=Docker(cpus=1.0))  # a configured instance
```

## The mock backend

`MockBackend` answers commands and file reads from dictionaries you supply. It
runs no container, so tests stay fast:

```python
from opencrl import MockBackend, rollout

backend = MockBackend(exec_map={"id": "uid=0(root)"}, files={"/flag": "CTF{win}"})
r = rollout(task, model, backend=backend)
```

## Register a custom backend

Register a factory under a name, then select that name from a task:

```python
from opencrl import register_backend

register_backend("myvm", lambda: MyBackend())
```

```python
Task(..., backend="myvm")
```

A custom backend implements `up()` and `down()` and returns a `World` with
`exec()` and `read_file()`. The framework needs nothing else.

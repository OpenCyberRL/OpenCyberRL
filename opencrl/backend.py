"""Backend protocol + registry. A backend stands up a World from a world spec."""
from __future__ import annotations

import os
from typing import Callable, Protocol

from opencrl.task import Caps
from opencrl.world import World


class Backend(Protocol):
    def up(self, spec: dict, caps: Caps) -> World: ...
    def down(self, world: World) -> None: ...


_BACKENDS: dict[str, Callable[[], "Backend"]] = {}


def register_backend(name: str, factory: Callable[[], "Backend"]) -> None:
    _BACKENDS[name] = factory


def resolve_backend(backend) -> "Backend":
    """Accept a registry name (default config) or an already-configured instance."""
    if isinstance(backend, str):
        if backend not in _BACKENDS:
            raise KeyError(f"unknown backend {backend!r}; known: {sorted(_BACKENDS)}")
        return _BACKENDS[backend]()
    return backend


def basedir_of(spec: dict) -> str:
    """The world file's own directory, as stamped by task.load_world (or "")."""
    return (spec.get("x-opencrl") or {}).get("basedir", "")


def resolve_path(path: str, basedir: str, what: str) -> str:
    """Resolve a world-spec path: absolute as-is, relative against `basedir`.

    Raises if `path` is relative and there's no basedir (an inline world=
    dict has no backing file to resolve against).
    """
    if os.path.isabs(path):
        return path
    if not basedir:
        raise ValueError(
            f"opencrl: relative '{what}' path {path!r} requires a world.yml "
            f"file (no basedir); use an absolute path in an inline world")
    return os.path.join(basedir, path)

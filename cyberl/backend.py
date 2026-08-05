"""Backend protocol + registry. A backend stands up a World from a world spec."""
from __future__ import annotations

from typing import Callable, Protocol

from cyberl.task import Caps
from cyberl.world import World


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

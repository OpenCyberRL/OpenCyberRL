"""In-memory backend for fast, Docker-free tests."""
from __future__ import annotations

from cyberl.backend import register_backend
from cyberl.task import Caps


class MockWorld:
    agent = "agent"

    def __init__(self, exec_map=None, files=None):
        self._exec = exec_map or {}
        self._files = files or {}

    def exec(self, command: str, host: str | None = None) -> str:
        return self._exec.get(command, "")

    def read_file(self, path: str, host: str | None = None) -> str | None:
        return self._files.get(path)


class MockBackend:
    def __init__(self, exec_map=None, files=None):
        self._exec = exec_map
        self._files = files

    def up(self, spec: dict, caps: Caps) -> MockWorld:
        return MockWorld(self._exec, self._files)

    def down(self, world) -> None:
        pass


register_backend("mock", lambda: MockBackend())

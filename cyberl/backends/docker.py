"""Docker/Compose backend. Shells out to the `docker` CLI — no SDK dependency."""
from __future__ import annotations

import copy
import os
import subprocess
import tempfile
import uuid

import yaml

from cyberl.backend import register_backend
from cyberl.task import Caps

_NET = "cyberl_net"


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True)


class DockerWorld:
    def __init__(self, project: str, compose_file: str, agent: str):
        self.project = project
        self.compose_file = compose_file
        self.agent = agent

    def _compose(self, *args: str) -> subprocess.CompletedProcess:
        return _run(["docker", "compose", "-p", self.project,
                     "-f", self.compose_file, *args])

    def exec(self, command: str, host: str | None = None) -> str:
        svc = host or self.agent
        cp = self._compose("exec", "-T", svc, "sh", "-lc", command)
        return (cp.stdout or "") + (cp.stderr or "")

    def read_file(self, path: str, host: str | None = None) -> str | None:
        svc = host or self.agent
        cp = self._compose("exec", "-T", svc, "sh", "-lc", f"cat {path}")
        if cp.returncode != 0:
            return None
        return cp.stdout


class Docker:
    def __init__(self, cpus: float | None = None, memory: str | None = None):
        self.cpus = cpus
        self.memory = memory

    def _render(self, spec: dict, caps: Caps) -> tuple[dict, str]:
        doc = copy.deepcopy(spec)
        agent = (doc.pop("x-cyberl", {}) or {}).get("agent", "")
        services = doc.setdefault("services", {})
        if not agent:
            agent = next(iter(services), "")
        # Inject an internal network so hosts reach each other but not the internet.
        doc["networks"] = {_NET: {"internal": not caps.needs_internet}}
        for name, svc in services.items():
            svc.setdefault("networks", [_NET])
            if self.cpus is not None:
                svc["cpus"] = self.cpus
            if self.memory is not None:
                svc["mem_limit"] = self.memory
        return doc, agent

    def up(self, spec: dict, caps: Caps) -> DockerWorld:
        doc, agent = self._render(spec or {}, caps)
        project = f"cyberl-{uuid.uuid4().hex[:8]}"
        fd, path = tempfile.mkstemp(prefix="cyberl-", suffix=".yml")
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(doc, f)
        cp = _run(["docker", "compose", "-p", project, "-f", path,
                   "up", "-d", "--build"])
        if cp.returncode != 0:
            raise RuntimeError(f"docker compose up failed:\n{cp.stderr}")
        return DockerWorld(project, path, agent)

    def down(self, world: DockerWorld) -> None:
        world._compose("down", "-v", "--remove-orphans")
        try:
            os.unlink(world.compose_file)
        except OSError:
            pass


register_backend("docker", lambda: Docker())

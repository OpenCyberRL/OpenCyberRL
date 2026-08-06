"""Docker/Compose backend. Shells out to the `docker` CLI — no SDK dependency."""
from __future__ import annotations

import copy
import os
import shlex
import subprocess
import sys
import tempfile
import uuid

import yaml

from cyberl.backend import register_backend
from cyberl.task import Caps

_NET = "cyberl_net"


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True)


class DockerWorld:
    def __init__(self, project: str, compose_file: str, agent: str, basedir: str = ""):
        self.project = project
        self.compose_file = compose_file
        self.agent = agent
        self.basedir = basedir

    def _compose(self, *args: str) -> subprocess.CompletedProcess:
        base = ["docker", "compose", "-p", self.project, "-f", self.compose_file]
        if self.basedir:
            base += ["--project-directory", self.basedir]
        return _run(base + list(args))

    def exec(self, command: str, host: str | None = None) -> str:
        svc = host or self.agent
        cp = self._compose("exec", "-T", svc, "sh", "-lc", command)
        return (cp.stdout or "") + (cp.stderr or "")

    def read_file(self, path: str, host: str | None = None) -> str | None:
        svc = host or self.agent
        cp = self._compose("exec", "-T", svc, "sh", "-lc", f"cat -- {shlex.quote(path)}")
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

        internal = not caps.needs_internet
        declared = doc.get("networks") or {}
        for svc in services.values():
            svc.setdefault("networks", [_NET])   # services with none join the default internal net
            if self.cpus is not None:
                svc["cpus"] = self.cpus
            if self.memory is not None:
                svc["mem_limit"] = self.memory
        # Define and lock down EVERY network any service references (incl. an
        # implicit `default` and any name not declared at top level), so no
        # service can sit on an unconfigured, externally-routable network.
        used = {n for svc in services.values() for n in (svc.get("networks") or [])}
        doc["networks"] = {name: {**(declared.get(name) or {}), "internal": internal}
                           for name in (set(declared) | used)}
        return doc, agent

    def up(self, spec: dict, caps: Caps) -> DockerWorld:
        spec = spec or {}
        # Read before _render pops `x-cyberl` off the spec.
        basedir = (spec.get("x-cyberl") or {}).get("basedir", "")
        doc, agent = self._render(spec, caps)
        project = f"cyberl-{uuid.uuid4().hex[:8]}"
        fd, path = tempfile.mkstemp(prefix="cyberl-", suffix=".yml")
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(doc, f)
        base = ["docker", "compose", "-p", project, "-f", path]
        if basedir:
            # Compose file lives in a tempdir; without this, relative
            # env_file/bind-mount/configs paths resolve against the tempdir
            # instead of the task directory they were written against.
            base += ["--project-directory", basedir]
        cp = _run(base + ["up", "-d", "--build"])
        if cp.returncode != 0:
            # Best-effort cleanup of anything that started before the failure,
            # so a failed up() never leaks containers/networks or the temp file.
            _run(base + ["down", "-v", "--remove-orphans"])
            try:
                os.unlink(path)
            except OSError:
                pass
            raise RuntimeError(f"docker compose up failed:\n{cp.stderr}")
        return DockerWorld(project, path, agent, basedir)

    def down(self, world: DockerWorld) -> None:
        cp = world._compose("down", "-v", "--remove-orphans")
        if cp.returncode != 0:
            print(f"cyberl: warning: teardown failed for project {world.project}; "
                  f"containers/networks may remain. Compose file retained at "
                  f"{world.compose_file}.\n{cp.stderr}", file=sys.stderr)
            return
        try:
            os.unlink(world.compose_file)
        except OSError:
            pass


register_backend("docker", lambda: Docker())

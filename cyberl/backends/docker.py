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

# Compose's own `default` network: services that declare no `networks:` are
# implicitly placed on it, so reusing that name (rather than a custom one)
# keeps them co-located with any service that explicitly lists `default`.
_NET = "default"

# Cap on any single exec/read_file's returned output, so a hostile or buggy
# command (e.g. `yes`) can't balloon host memory just because it stayed
# within the exec_timeout window.
_MAX_OUTPUT = 100_000


def _run(args: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True,
                          errors="replace", timeout=timeout)


class DockerWorld:
    def __init__(self, project: str, compose_file: str, agent: str, basedir: str = "",
                 exec_timeout: float = 120.0):
        self.project = project
        self.compose_file = compose_file
        self.agent = agent
        self.basedir = basedir
        self.exec_timeout = exec_timeout

    def _compose(self, *args: str, timeout: float | None = None) -> subprocess.CompletedProcess:
        base = ["docker", "compose", "-p", self.project, "-f", self.compose_file]
        if self.basedir:
            base += ["--project-directory", self.basedir]
        return _run(base + list(args), timeout=timeout)

    def exec(self, command: str, host: str | None = None) -> str:
        svc = host or self.agent
        try:
            cp = self._compose("exec", "-T", svc, "sh", "-lc", command,
                               timeout=self.exec_timeout)
        except subprocess.TimeoutExpired:
            return f"[cyberl: command timed out after {self.exec_timeout}s]"
        out = (cp.stdout or "") + (cp.stderr or "")
        if len(out) > _MAX_OUTPUT:
            out = out[:_MAX_OUTPUT] + "\n[cyberl: output truncated]"
        return out

    def read_file(self, path: str, host: str | None = None) -> str | None:
        svc = host or self.agent
        try:
            cp = self._compose("exec", "-T", svc, "sh", "-lc", f"cat -- {shlex.quote(path)}",
                               timeout=self.exec_timeout)
        except subprocess.TimeoutExpired:
            return None
        if cp.returncode != 0:
            return None
        content = cp.stdout
        if len(content) > _MAX_OUTPUT:
            content = content[:_MAX_OUTPUT] + "\n[cyberl: output truncated]"
        return content


class Docker:
    def __init__(self, cpus: float | None = None, memory: str | None = None,
                 exec_timeout: float = 120.0):
        self.cpus = cpus
        self.memory = memory
        self.exec_timeout = exec_timeout

    def _render(self, spec: dict, caps: Caps) -> tuple[dict, str]:
        doc = copy.deepcopy(spec)
        if "include" in doc:
            # `include:` merges extra services/networks in AFTER our policy
            # runs, so an included service could sit on a non-internal
            # `default` net regardless of Caps(needs_internet=False).
            raise ValueError(
                "cyberl: top-level Compose 'include' is not supported "
                "(it bypasses network isolation)")
        agent = (doc.pop("x-cyberl", {}) or {}).get("agent", "")
        services = doc.setdefault("services", {})
        if not agent:
            agent = next(iter(services), "")

        internal = not caps.needs_internet
        declared = doc.get("networks") or {}
        if not caps.needs_internet:
            # A network declared `external: true` is a pre-existing Docker
            # network; setting `internal` on our doc is a no-op for it, so a
            # service on it would keep that network's routes regardless of
            # needs_internet=False.
            for name, cfg in declared.items():
                if (cfg or {}).get("external"):
                    raise ValueError(
                        f"cyberl: external network {name!r} is not allowed "
                        f"when needs_internet=False")
        for svc in services.values():
            if not svc.get("networks"):        # None, missing, [], or {} -> default internal net
                svc["networks"] = [_NET]
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
        # No timeout: image builds are slow and legitimately open-ended.
        cp = _run(base + ["up", "-d", "--build"], timeout=None)
        if cp.returncode != 0:
            # Best-effort cleanup of anything that started before the failure,
            # so a failed up() never leaks containers/networks or the temp file.
            _run(base + ["down", "-v", "--remove-orphans"], timeout=None)
            try:
                os.unlink(path)
            except OSError:
                pass
            raise RuntimeError(f"docker compose up failed:\n{cp.stderr}")
        return DockerWorld(project, path, agent, basedir, exec_timeout=self.exec_timeout)

    def down(self, world: DockerWorld) -> None:
        # No timeout: teardown should be allowed to run to completion.
        cp = world._compose("down", "-v", "--remove-orphans", timeout=None)
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

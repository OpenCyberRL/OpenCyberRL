"""Docker/Compose backend. Shells out to the `docker` CLI — no SDK dependency."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import uuid

import yaml

from opencrl.backend import basedir_of, register_backend
from opencrl.task import Caps

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


def _pin_build_images(doc: dict) -> list[tuple[str, str]]:
    """Assign every `build:` service a deterministic image tag (only when it
    declares none) and return (image_ref, identity) pairs for the build cache.

    `identity` digests the full effective build config plus the service-level
    `platform` (outside `build:` but selects the architecture). The cache keys
    on the emitted image reference and remembers which identity last built it,
    so a service pinning an explicit `image:` shared with a different build
    config is rebuilt rather than silently reused, and services differing only
    by target / dockerfile_inline / additional_contexts / args / platform stay
    distinct.
    """
    pins: list[tuple[str, str]] = []
    for svc in (doc.get("services") or {}).values():
        build = svc.get("build")
        if not build:
            continue
        platform = str(svc.get("platform", ""))
        if isinstance(build, str):
            key = f"{platform}|{build}"
        else:
            key = platform + "|" + json.dumps(build, sort_keys=True, default=str)
        digest = hashlib.sha256(key.encode()).hexdigest()[:12]
        svc.setdefault("image", f"opencrl-build-{digest}")
        pins.append((svc["image"], digest))
    return pins


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
            return f"[opencrl: command timed out after {self.exec_timeout}s]"
        out = (cp.stdout or "") + (cp.stderr or "")
        if len(out) > _MAX_OUTPUT:
            out = out[:_MAX_OUTPUT] + "\n[opencrl: output truncated]"
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
            content = content[:_MAX_OUTPUT] + "\n[opencrl: output truncated]"
        return content


class Docker:
    def __init__(self, cpus: float | None = None, memory: str | None = None,
                 exec_timeout: float = 120.0):
        self.cpus = cpus
        self.memory = memory
        self.exec_timeout = exec_timeout
        self._built: dict[str, str] = {}       # image ref -> digest that last built it
        self._built_lock = threading.Lock()

    def __getstate__(self):
        # threading.Lock isn't picklable; drop it so a Docker instance survives
        # cloudpickling into spawn-based multiprocessing (async gym vectors).
        state = self.__dict__.copy()
        state.pop("_built_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._built_lock = threading.Lock()

    def _render(self, spec: dict, caps: Caps) -> tuple[dict, str]:
        doc = copy.deepcopy(spec)
        if "include" in doc:
            # `include:` merges extra services/networks in AFTER our policy
            # runs, so an included service could sit on a non-internal
            # `default` net regardless of Caps(needs_internet=False).
            raise ValueError(
                "opencrl: top-level Compose 'include' is not supported "
                "(it bypasses network isolation)")
        agent = (doc.pop("x-opencrl", {}) or {}).get("agent", "")
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
                        f"opencrl: external network {name!r} is not allowed "
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

    def _write_compose(self, doc: dict) -> str:
        fd, path = tempfile.mkstemp(prefix="opencrl-", suffix=".yml")
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(doc, f)
        return path

    def prebuild(self, spec: dict, caps: Caps) -> None:
        """Build a task's images ONCE (into stable tags) before fan-out, so N
        concurrent up()s don't each rebuild. No-op if the world has no build:."""
        spec = spec or {}
        basedir = basedir_of(spec)
        doc, _agent = self._render(spec, caps)
        pins = _pin_build_images(doc)
        if not pins:
            return
        path = self._write_compose(doc)
        project = f"opencrl-prebuild-{uuid.uuid4().hex[:8]}"
        base = ["docker", "compose", "-p", project, "-f", path]
        if basedir:
            base += ["--project-directory", basedir]
        try:
            cp = _run(base + ["build"], timeout=None)
            if cp.returncode != 0:
                raise RuntimeError(f"docker compose build failed:\n{cp.stderr}")
            with self._built_lock:
                for img, dig in pins:
                    self._built[img] = dig
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def up(self, spec: dict, caps: Caps) -> DockerWorld:
        spec = spec or {}
        # Read before _render pops `x-opencrl` off the spec.
        basedir = basedir_of(spec)
        doc, agent = self._render(spec, caps)
        pins = _pin_build_images(doc)
        project = f"opencrl-{uuid.uuid4().hex[:8]}"
        path = self._write_compose(doc)
        base = ["docker", "compose", "-p", project, "-f", path]
        if basedir:
            # Compose file lives in a tempdir; without this, relative
            # env_file/bind-mount/configs paths resolve against the tempdir
            # instead of the task directory they were written against.
            base += ["--project-directory", basedir]
        with self._built_lock:
            need_build = [img for img, dig in pins if self._built.get(img) != dig]
        # Build only images whose current owner isn't this exact build config; a
        # world with no build: still gets --build (a no-op) to preserve prior behavior.
        build_flag = ["--build"] if (need_build or not pins) else []
        # No timeout: image builds are slow and legitimately open-ended.
        cp = _run(base + ["up", "-d"] + build_flag, timeout=None)
        if cp.returncode != 0:
            # Best-effort cleanup of anything that started before the failure,
            # so a failed up() never leaks containers/networks or the temp file.
            _run(base + ["down", "-v", "--remove-orphans"], timeout=None)
            try:
                os.unlink(path)
            except OSError:
                pass
            raise RuntimeError(f"docker compose up failed:\n{cp.stderr}")
        if pins:
            with self._built_lock:
                for img, dig in pins:
                    self._built[img] = dig
        return DockerWorld(project, path, agent, basedir, exec_timeout=self.exec_timeout)

    def down(self, world: DockerWorld) -> None:
        # No timeout: teardown should be allowed to run to completion.
        cp = world._compose("down", "-v", "--remove-orphans", timeout=None)
        if cp.returncode != 0:
            print(f"opencrl: warning: teardown failed for project {world.project}; "
                  f"containers/networks may remain. Compose file retained at "
                  f"{world.compose_file}.\n{cp.stderr}", file=sys.stderr)
            return
        try:
            os.unlink(world.compose_file)
        except OSError:
            pass


register_backend("docker", lambda: Docker())

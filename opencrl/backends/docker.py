"""Docker/Compose backend. Shells out to the `docker` CLI — no SDK dependency."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
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


_INTERP_BRACE = re.compile(r'\$\{([^}]*)\}')


def _expand_vars(text: str) -> str:
    """Resolve Compose-style interpolation (${VAR}, ${VAR:-default},
    ${VAR-default}, ${VAR:?err}, $VAR) to effective values for cache identity
    hashing. Compose applies these at runtime; hashing the raw placeholder
    would alias builds that differ only in resolved arg values."""
    text = text.replace("$$", "\x00")   # preserve Compose's literal-$$
    def repl_brace(m):
        inner = m.group(1)
        for sep in (":-", "-", ":?", "?"):
            idx = inner.find(sep)
            if idx >= 0:
                name = inner[:idx].strip()
                default = inner[idx + len(sep):]
                val = os.environ.get(name)
                return val if val else default
        return os.environ.get(inner.strip(), "")
    text = _INTERP_BRACE.sub(repl_brace, text)
    text = re.sub(r'\$([A-Za-z_][A-Za-z0-9_]*)',
                  lambda m: os.environ.get(m.group(1), ""), text)
    return text.replace("\x00", "$")


def _pin_build_images(doc: dict, basedir: str = "") -> list[str]:
    """Override every `build:` service's image with a deterministic,
    content-addressed tag and return the per-service build identities (digests).

    The digest covers `basedir` (Compose resolves an omitted build context to
    the project directory), the service `platform` (outside `build:`, selects
    the architecture), and the full effective build config — with Compose
    interpolation (${VAR}, ${VAR:-default}) resolved against the environment —
    so distinct or re-parameterized build inputs never collide on a tag.
    Overriding (not defaulting) the image means an explicit `image:` shared
    across different build configs can't alias, which removes any need for
    ownership bookkeeping or shared-tag locking. Any non-build service that
    references an overridden image is rewritten to the new tag, preserving
    Compose image-sharing semantics. Services with no `build:` key are
    untouched.
    """
    services = doc.get("services") or {}
    rewrites: dict[str, str] = {}   # original image -> new content-addressed tag
    identities: list[str] = []
    for svc in services.values():
        if "build" not in svc:
            continue
        build = svc["build"]
        platform = str(svc.get("platform", ""))
        if isinstance(build, str):
            build_key = build
        else:
            build_key = json.dumps(build or {}, sort_keys=True, default=str)
        build_key = _expand_vars(build_key)
        digest = hashlib.sha256(f"{basedir}|{platform}|{build_key}".encode()).hexdigest()[:12]
        new_tag = f"opencrl-build-{digest}"
        original = svc.get("image")
        if original:
            rewrites[original] = new_tag
        svc["image"] = new_tag
        identities.append(digest)
    # Rewrite consumers of an overridden image so they run the built image,
    # not a stale pull of the original tag.
    if rewrites:
        for svc in services.values():
            img = svc.get("image")
            if img in rewrites:
                svc["image"] = rewrites[img]
    return identities


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
        self._built: set[str] = set()          # build identities (digests) built this instance
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
        identities = _pin_build_images(doc, basedir)
        if not identities:
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
                self._built.update(identities)
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
        identities = _pin_build_images(doc, basedir)
        project = f"opencrl-{uuid.uuid4().hex[:8]}"
        path = self._write_compose(doc)
        base = ["docker", "compose", "-p", project, "-f", path]
        if basedir:
            # Compose file lives in a tempdir; without this, relative
            # env_file/bind-mount/configs paths resolve against the tempdir
            # instead of the task directory they were written against.
            base += ["--project-directory", basedir]
        with self._built_lock:
            need_build = [d for d in identities if d not in self._built]
        # Build only identities not yet built by this instance; a world with no
        # build: still gets --build (a no-op) to preserve prior behavior.
        build_flag = ["--build"] if (need_build or not identities) else []
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
        if identities:
            with self._built_lock:
                self._built.update(identities)
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

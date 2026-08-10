"""Docker Sandboxes backend. Shells out to the `sbx` CLI — no SDK dependency.

Runs a single-host world inside a Docker Sandbox (a microVM): stronger
isolation than a shared-kernel container, far less setup than the qemu
backend. No external egress unless the task sets caps.needs_internet.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid

import yaml

from opencrl.backend import basedir_of, register_backend, resolve_path
from opencrl.task import Caps

# Cap on any single exec/read_file's returned output (mirrors the docker backend).
_MAX_OUTPUT = 100_000


def _run(args: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True,
                          errors="replace", timeout=timeout)


def _cap(text: str) -> str:
    if len(text) > _MAX_OUTPUT:
        return text[:_MAX_OUTPUT] + "\n[opencrl: output truncated]"
    return text


class SandboxWorld:
    agent = "sandbox"                # single microVM; the `host` arg is ignored

    def __init__(self, name: str, workdir: str, sbx_bin: str = "sbx",
                 exec_timeout: float = 120.0):
        self.name = name
        self._workdir = workdir      # temp dir holding the kit (removed in down())
        self.sbx_bin = sbx_bin
        self.exec_timeout = exec_timeout

    def exec(self, command: str, host: str | None = None) -> str:
        try:
            cp = _run([self.sbx_bin, "exec", self.name, "sh", "-c", command],
                      timeout=self.exec_timeout)
        except subprocess.TimeoutExpired:
            return f"[opencrl: command timed out after {self.exec_timeout}s]"
        return _cap((cp.stdout or "") + (cp.stderr or ""))

    def read_file(self, path: str, host: str | None = None) -> str | None:
        try:
            cp = _run([self.sbx_bin, "cp", f"{self.name}:{path}", "-"],
                      timeout=self.exec_timeout)
        except subprocess.TimeoutExpired:
            return None
        if cp.returncode != 0:
            return None
        return _cap(cp.stdout or "")


class Sandbox:
    def __init__(self, cpus: int = 2, memory: str = "2g", exec_timeout: float = 120.0,
                 sbx_bin: str = "sbx"):
        self.cpus = cpus
        self.memory = memory
        self.exec_timeout = exec_timeout
        self.sbx_bin = sbx_bin

    def _kit(self, spec: dict, caps: Caps) -> dict:
        image = spec.get("image")
        if not image:
            raise ValueError("opencrl: sandbox world requires 'image'")
        return {"sandbox": {
            "image": image,
            "cpus": spec.get("cpus", self.cpus),
            "memory": spec.get("memory", self.memory),
            "network": "allow" if caps.needs_internet else "deny",
        }}

    def up(self, spec: dict, caps: Caps) -> SandboxWorld:
        spec = spec or {}
        basedir = basedir_of(spec)
        kit = self._kit(spec, caps)                  # raises on missing image first
        name = f"opencrl-{uuid.uuid4().hex[:8]}"
        workdir = tempfile.mkdtemp(prefix="opencrl-sbx-")
        try:
            kit_path = os.path.join(workdir, "kit.yaml")
            with open(kit_path, "w") as f:
                yaml.safe_dump(kit, f)
            argv = [self.sbx_bin, "create", "--name", name, "--kit", kit_path]
            if spec.get("files"):
                argv.append(resolve_path(spec["files"], basedir, "files"))
            cp = _run(argv, timeout=None)
            if cp.returncode != 0:
                raise RuntimeError(f"sbx create failed:\n{cp.stderr}")
            for cmd in spec.get("setup", []):
                scp = _run([self.sbx_bin, "exec", name, "sh", "-c", cmd], timeout=None)
                if scp.returncode != 0:
                    raise RuntimeError(f"sandbox setup failed ({cmd!r}):\n{scp.stderr}")
            return SandboxWorld(name, workdir, self.sbx_bin, self.exec_timeout)
        except Exception:
            _teardown(name, workdir, self.sbx_bin)
            raise

    def down(self, world: SandboxWorld) -> None:
        _teardown(world.name, world._workdir, world.sbx_bin)


def _teardown(name: str, workdir: str, sbx_bin: str) -> None:
    if name:
        try:
            _run([sbx_bin, "rm", "--force", name], timeout=None)
        except OSError:
            pass
    if workdir:
        shutil.rmtree(workdir, ignore_errors=True)


register_backend("sandbox", lambda: Sandbox())

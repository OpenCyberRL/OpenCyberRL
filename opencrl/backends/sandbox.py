"""Docker Sandboxes backend. Shells out to the `sbx` CLI — no SDK dependency.

Runs a single-host world inside a Docker Sandbox (a microVM): stronger
isolation than a shared-kernel container, far less setup than the qemu
backend. No external egress unless the task sets caps.needs_internet.
"""
from __future__ import annotations

import subprocess

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

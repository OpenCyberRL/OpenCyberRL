"""Docker Sandboxes backend. Shells out to the `docker sandbox` CLI — no SDK.

Runs a single-host world inside a Docker Sandbox (a microVM): stronger
isolation than a shared-kernel container, far less setup than the qemu
backend. No external egress unless the task sets caps.needs_internet.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid

from opencrl.backend import basedir_of, register_backend, resolve_path
from opencrl.task import Caps

# The Docker Sandboxes CLI is `docker sandbox <command>`.
_CLI = ["docker", "sandbox"]
# Cap on any single exec/read_file's returned output (mirrors the docker backend).
_MAX_OUTPUT = 100_000


def _run(args: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True,
                          errors="replace", timeout=timeout)


def _cap(text: str) -> str:
    if len(text) > _MAX_OUTPUT:
        return text[:_MAX_OUTPUT] + "\n[opencrl: output truncated]"
    return text


# `--policy deny` blocks only NON-allowlisted egress. The sandbox proxy's config
# also carries a built-in allowlist (github, pypi, npm, AI APIs, …) and bypass
# rules, all of which survive a deny. To airgap a no-egress guest we block every
# allowlisted domain and refuse to run when bypass rules exist or the config
# can't be read (fail closed). sandboxd may be pointed at a non-default config
# via SANDBOXD_PROXY_CONFIG, so honor that path too.
_DEFAULT_PROXY_CONFIG = os.path.expanduser("~/.sandboxd/proxy-config.json")


def _proxy_config_path() -> str:
    return os.environ.get("SANDBOXD_PROXY_CONFIG") or _DEFAULT_PROXY_CONFIG


def _egress_block_args() -> list[str]:
    """`--block-host` args that airgap the guest, from the daemon's proxy config.

    Fail closed (raise) when the config can't be read or carries bypass rules
    that skip the proxy and can't be blocked — better to error than run with
    egress a no-egress task expected to be denied.
    """
    path = _proxy_config_path()
    try:
        with open(path) as f:
            net = json.load(f).get("network", {})
    except (OSError, ValueError) as e:
        raise RuntimeError(
            f"opencrl: cannot read the Docker Sandbox proxy config ({path}) to "
            "enforce no egress. Set caps(needs_internet=True) to allow network, "
            "or use the docker/qemu backend for a guaranteed airgap.") from e
    if net.get("bypassDomains") or net.get("bypassCIDRs"):
        raise RuntimeError(
            "opencrl: the Docker Sandbox proxy has bypass rules that skip the "
            "proxy and can't be blocked; a no-egress guest can't be guaranteed. "
            "Clear the proxy bypass config, set caps(needs_internet=True), or use "
            "the docker/qemu backend.")
    return [a for d in net.get("allowedDomains", []) for a in ("--block-host", d)]


class SandboxWorld:
    agent = "sandbox"                # single microVM; the `host` arg is ignored

    def __init__(self, name: str, workspace: str, exec_timeout: float = 120.0):
        self.name = name
        self._workspace = workspace  # temp workspace dir (removed in down())
        self.exec_timeout = exec_timeout

    def exec(self, command: str, host: str | None = None) -> str:
        try:
            cp = _run([*_CLI, "exec", self.name, "sh", "-c", command],
                      timeout=self.exec_timeout)
        except subprocess.TimeoutExpired:
            return f"[opencrl: command timed out after {self.exec_timeout}s]"
        return _cap((cp.stdout or "") + (cp.stderr or ""))

    def read_file(self, path: str, host: str | None = None) -> str | None:
        # No `docker sandbox cp`, so read via `exec cat`. exec propagates the
        # inner exit code, so a missing file returns non-zero -> None.
        try:
            cp = _run([*_CLI, "exec", self.name, "cat", "--", path],
                      timeout=self.exec_timeout)
        except subprocess.TimeoutExpired:
            return None
        if cp.returncode != 0:
            return None
        return _cap(cp.stdout or "")


class Sandbox:
    def __init__(self, agent: str = "codex", exec_timeout: float = 120.0):
        self.agent = agent          # the create-agent; we exec, never `run` it
        self.exec_timeout = exec_timeout

    def up(self, spec: dict, caps: Caps) -> SandboxWorld:
        spec = spec or {}
        image = spec.get("image")
        if not image:
            raise ValueError("opencrl: sandbox world requires 'image'")
        basedir = basedir_of(spec)
        name = f"opencrl-{uuid.uuid4().hex[:8]}"
        # A throwaway workspace: `docker sandbox create` requires one and mounts
        # it read-write, so copy the task's `files` into a temp dir rather than
        # exposing (and letting the guest mutate) the author's own directory.
        workspace = tempfile.mkdtemp(prefix="opencrl-sbx-")
        try:
            if spec.get("files"):
                # symlinks=True copies links verbatim; without it copytree would
                # dereference a link in `files` and copy a host file's *contents*
                # into the guest-readable workspace.
                shutil.copytree(resolve_path(spec["files"], basedir, "files"),
                                workspace, dirs_exist_ok=True, symlinks=True)
            create = _run([*_CLI, "create", "--name", name, "--template", image,
                           self.agent, workspace], timeout=None)
            if create.returncode != 0:
                raise RuntimeError(f"docker sandbox create failed:\n{create.stderr}")
            if not caps.needs_internet:
                # Deny the default policy AND block the proxy's built-in allowlist
                # (fail closed if it can't be read or has bypass rules).
                block = _egress_block_args()
                deny = _run([*_CLI, "network", "proxy", name, "--policy", "deny",
                             *block], timeout=None)
                if deny.returncode != 0:
                    raise RuntimeError(
                        f"docker sandbox network proxy (deny) failed:\n{deny.stderr}")
            for cmd in spec.get("setup", []):
                setup = _run([*_CLI, "exec", name, "sh", "-c", cmd], timeout=None)
                if setup.returncode != 0:
                    raise RuntimeError(f"sandbox setup failed ({cmd!r}):\n{setup.stderr}")
            return SandboxWorld(name, workspace, self.exec_timeout)
        except Exception:
            _teardown(name, workspace)
            raise

    def down(self, world: SandboxWorld) -> None:
        _teardown(world.name, world._workspace)


def _teardown(name: str, workspace: str) -> None:
    if name:
        try:
            cp = _run([*_CLI, "rm", name])
        except OSError:
            cp = None
        if cp is not None and cp.returncode != 0:
            # A non-zero rm means the microVM may still be running — surface it
            # rather than reporting a clean teardown.
            print(f"opencrl: warning: 'docker sandbox rm {name}' failed; the "
                  f"sandbox may still be running.\n{cp.stderr}", file=sys.stderr)
    if workspace:
        shutil.rmtree(workspace, ignore_errors=True)


register_backend("sandbox", lambda: Sandbox())

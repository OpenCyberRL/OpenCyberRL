import shutil
import subprocess

import pytest

from opencrl import Caps
from opencrl.backends.sandbox import Sandbox

pytestmark = pytest.mark.sandbox


def _has_docker_sandbox() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "sandbox", "version"],
                              capture_output=True).returncode == 0
    except OSError:
        return False


_skip = pytest.mark.skipif(
    not _has_docker_sandbox(),
    reason="the `docker sandbox` CLI (Docker Sandboxes) is required",
)


@_skip
def test_sandbox_boots_and_execs():
    backend = Sandbox()
    world = backend.up(
        {"image": "ubuntu:24.04", "setup": ["echo CTF{sbx} > /root/flag"]},
        Caps(),
    )
    try:
        assert "uid=0(root)" in world.exec("id")             # the microVM shell
        assert world.read_file("/root/flag").strip() == "CTF{sbx}"
        assert world.read_file("/does/not/exist") is None    # exit-code -> None
    finally:
        backend.down(world)

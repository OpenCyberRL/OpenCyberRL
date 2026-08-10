import shutil

import pytest

from opencrl import Caps
from opencrl.backends.sandbox import Sandbox

pytestmark = pytest.mark.sandbox

_skip = pytest.mark.skipif(
    shutil.which("sbx") is None,
    reason="the sbx CLI (Docker Sandboxes) is required",
)


@_skip
def test_sandbox_boots_and_execs():
    backend = Sandbox()
    world = backend.up(
        {"image": "ubuntu:24.04", "setup": ["echo CTF{sbx} > /root/flag"]},
        Caps(),
    )
    try:
        assert "root" in world.exec("id")
        assert world.read_file("/root/flag").strip() == "CTF{sbx}"
        assert world.read_file("/does/not/exist") is None
    finally:
        backend.down(world)

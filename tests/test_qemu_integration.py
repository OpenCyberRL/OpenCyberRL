# tests/test_qemu_integration.py
import shutil
from pathlib import Path

import pytest

from opencrl import Caps
from opencrl.backends.qemu import Qemu

pytestmark = pytest.mark.qemu

_FIX = Path(__file__).parent / "fixtures" / "qemu"
_KERNEL = _FIX / "bzImage"
_INITRD = _FIX / "rootfs.cpio.gz"

_skip = pytest.mark.skipif(
    shutil.which("qemu-system-x86_64") is None
    or not (_KERNEL.exists() and _INITRD.exists()),
    reason="qemu-system-x86_64 and a built fixture (run tests/fixtures/qemu/build.sh) are required",
)


def _world():
    return {"kernel": str(_KERNEL), "initrd": str(_INITRD),
            "cpus": 1, "memory": "256m"}


@_skip
def test_qemu_boots_and_execs():
    backend = Qemu()
    world = backend.up(_world(), Caps())
    try:
        assert "uid=1000" in world.exec("id")           # unprivileged shell
        assert world.exec("uname -r").strip()            # kernel version present
        assert world.read_file("/etc/nope-does-not-exist") is None
        # /root/flag exists but is root-only (mode 600); the unprivileged
        # shell cannot read it -> None, proving the isolation is real.
        assert world.read_file("/root/flag") is None
    finally:
        backend.down(world)

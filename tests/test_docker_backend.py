import pytest
from cyberl.task import Caps
from cyberl.backends.docker import Docker

pytestmark = pytest.mark.docker

SPEC = {
    "x-cyberl": {"agent": "box"},
    "services": {"box": {"image": "alpine:3.20", "command": "sleep 600"}},
}

def test_docker_up_exec_readfile_down():
    backend = Docker()
    world = backend.up(SPEC, Caps())
    try:
        assert world.agent == "box"
        assert "root" in world.exec("id")
        world.exec("sh -c 'echo hello > /tmp/f.txt'")
        assert world.read_file("/tmp/f.txt").strip() == "hello"
        assert world.read_file("/does/not/exist") is None
    finally:
        backend.down(world)

def test_docker_no_egress_by_default():
    backend = Docker()
    world = backend.up(SPEC, Caps(needs_internet=False))
    try:
        # With an internal network there is no route off-box; ping/curl fails fast.
        out = world.exec("sh -c 'wget -T 2 -q -O- http://example.com || echo BLOCKED'")
        assert "BLOCKED" in out
    finally:
        backend.down(world)

def test_docker_exec_survives_non_utf8_output():
    # A command emitting bytes that aren't valid UTF-8 must not raise
    # UnicodeDecodeError — _run decodes with errors="replace" instead of
    # the strict default.
    backend = Docker()
    world = backend.up(SPEC, Caps())
    try:
        out = world.exec("printf '\\xff\\xfe\\xfd'")
        assert isinstance(out, str)
    finally:
        backend.down(world)

def test_docker_exec_bounded_by_timeout():
    # A hanging command (e.g. `sleep infinity`) must not hang the rollout or
    # teardown: exec() bounds it at exec_timeout and returns a sentinel
    # instead of raising or blocking forever.
    backend = Docker(exec_timeout=1.0)
    world = backend.up(SPEC, Caps())
    try:
        out = world.exec("sleep 5")
        assert out == "[cyberl: command timed out after 1.0s]"
    finally:
        backend.down(world)

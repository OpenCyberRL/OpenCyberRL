import pytest
from cyberl.task import Caps
from cyberl.backend import register_backend, resolve_backend
from cyberl.backends.mock import MockBackend, MockWorld

def test_mock_world_exec_and_files():
    w = MockWorld(exec_map={"id": "uid=0(root)"}, files={"/f": "hi"})
    assert w.exec("id") == "uid=0(root)"
    assert w.exec("unknown") == ""
    assert w.read_file("/f") == "hi"
    assert w.read_file("/nope") is None

def test_mock_backend_up_down_roundtrip():
    b = MockBackend(exec_map={"whoami": "agent"})
    world = b.up({"services": {}}, Caps())
    assert world.exec("whoami") == "agent"
    b.down(world)  # no error

def test_resolve_backend_by_name_and_instance():
    inst = MockBackend()
    register_backend("mock", lambda: MockBackend())
    assert isinstance(resolve_backend("mock"), MockBackend)
    assert resolve_backend(inst) is inst

def test_resolve_unknown_backend_raises():
    with pytest.raises(KeyError):
        resolve_backend("does-not-exist")

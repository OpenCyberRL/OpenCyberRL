import threading
import pytest
from opencrl.task import Task, Caps
from opencrl.tools import shell
from opencrl.reward import flag
from opencrl.backends.mock import MockBackend, MockWorld
from opencrl.models import ScriptedModel
from opencrl.runner import run_batch, ThreadRunner


def make_task():
    return Task(goal="g", reward=flag("CTF{win}"), tools=(shell,), name="demo",
                caps=Caps(), max_steps=3, backend=MockBackend())


def _answer_model():
    return lambda m, t: {"role": "assistant", "content": "CTF{win}", "tool_calls": None}


def test_run_batch_returns_index_aligned_results():
    rs = run_batch(make_task(), _answer_model(), n=5, backend=MockBackend())
    assert len(rs) == 5
    assert all(r.reward == 1.0 for r in rs)


def test_run_batch_requires_exactly_one_of_model_or_factory():
    with pytest.raises(ValueError, match="exactly one"):
        run_batch(make_task(), n=1, backend=MockBackend())
    with pytest.raises(ValueError, match="exactly one"):
        run_batch(make_task(), _answer_model(),
                  model_factory=lambda: _answer_model(), n=1, backend=MockBackend())


def test_run_batch_uses_model_factory_per_rollout():
    def factory():
        return ScriptedModel([{"role": "assistant", "content": "CTF{win}",
                               "tool_calls": None}])
    rs = run_batch(make_task(), model_factory=factory, n=4, backend=MockBackend())
    assert [r.reward for r in rs] == [1.0, 1.0, 1.0, 1.0]


def test_run_batch_actually_parallel():
    barrier = threading.Barrier(3, timeout=5)
    def model(m, t):
        barrier.wait()
        return {"role": "assistant", "content": "CTF{win}", "tool_calls": None}
    rs = run_batch(make_task(), model, n=3, concurrency=3, backend=MockBackend())
    assert len(rs) == 3


def test_run_batch_no_world_leak_on_exception():
    class RecordingBackend(MockBackend):
        def __init__(self):
            super().__init__(); self.ups = 0; self.downs = 0; self._lock = threading.Lock()
        def up(self, spec, caps):
            with self._lock: self.ups += 1
            return MockWorld()
        def down(self, world):
            with self._lock: self.downs += 1
    be = RecordingBackend()
    def boom(m, t): raise RuntimeError("model crashed")
    with pytest.raises(RuntimeError):
        run_batch(make_task(), boom, n=4, concurrency=2, backend=be)
    assert be.ups == be.downs


def test_run_batch_calls_prebuild_once():
    calls = []
    class PrebuildBackend(MockBackend):
        def prebuild(self, spec, caps): calls.append(1)
    run_batch(make_task(), _answer_model(), n=3, backend=PrebuildBackend())
    assert calls == [1]


def test_run_batch_aborts_when_on_result_raises():
    ups = []

    class BE(MockBackend):
        def up(self, spec, caps):
            ups.append(1)
            return MockWorld()

    def bad(i, r):
        raise RuntimeError("callback boom")

    with pytest.raises(RuntimeError, match="callback boom"):
        run_batch(make_task(), _answer_model(), n=5, concurrency=1,
                  backend=BE(), on_result=bad)
    assert len(ups) == 1   # aborted after the first; remaining rollouts skip up()


def test_run_batch_empty_skips_backend():
    class BoomBackend(MockBackend):
        def up(self, spec, caps):
            raise AssertionError("empty batch must not touch the backend")
        def prebuild(self, spec, caps):
            raise AssertionError("empty batch must not prebuild")
    assert run_batch(make_task(), _answer_model(), n=0, backend=BoomBackend()) == []


def test_run_batch_instantiates_models_lazily():
    live = 0
    max_live = 0
    lock = threading.Lock()

    def factory():
        nonlocal live, max_live
        with lock:
            live += 1
            max_live = max(max_live, live)
        return _answer_model()

    def on_result(i, r):
        nonlocal live
        with lock:
            live -= 1

    run_batch(make_task(), model_factory=factory, n=8, concurrency=2,
              backend=MockBackend(), on_result=on_result)
    assert max_live <= 2   # eager pre-instantiation would reach 8

import threading
import pytest
from opencrl.task import Task, Caps
from opencrl.tools import shell
from opencrl.reward import flag
from opencrl.backends.mock import MockBackend, MockWorld
from opencrl.runner import run_batch


def _model():
    return lambda m, t: {"role": "assistant", "content": "CTF{win}", "tool_calls": None}


def test_pool_reuses_worlds_and_runs_reset():
    reset_calls = []
    barrier = threading.Barrier(2, timeout=5)   # forces the 2 workers to run lockstep

    class W(MockWorld):
        def exec(self, command, host=None):
            if command == "RESET":
                reset_calls.append(command)
            return ""

    class BE(MockBackend):
        def __init__(self):
            super().__init__()
            self.ups = 0
            self.downs = 0
            self._lock = threading.Lock()

        def up(self, spec, caps):
            with self._lock:
                self.ups += 1
            return W()

        def down(self, world):
            with self._lock:
                self.downs += 1

    be = BE()

    def model(m, t):
        barrier.wait()   # each worker consumes exactly one job per barrier cycle
        return {"role": "assistant", "content": "CTF{win}", "tool_calls": None}

    task = Task(goal="g", reward=flag("CTF{win}"), tools=(shell,), name="p",
                caps=Caps(), max_steps=2, backend=be, reset="RESET")
    rs = run_batch(task, model, n=6, concurrency=2, backend=be, pool=True)

    assert len(rs) == 6 and all(r.reward == 1.0 for r in rs)
    assert be.ups == 2 and be.downs == 2          # pool of 2 worlds, reused and torn down
    assert len(reset_calls) == 6 - 2              # reset once per reuse (all but each worker's first)


def test_pool_requires_reset_hook():
    task = Task(goal="g", reward=flag("CTF{win}"), tools=(shell,), name="p",
                caps=Caps(), max_steps=2, backend=MockBackend())   # no reset
    with pytest.raises(ValueError, match="reset"):
        run_batch(task, _model(), n=3, backend=MockBackend(), pool=True)


def test_pool_reads_reset_from_world_metadata():
    reset_calls = []

    class W(MockWorld):
        def exec(self, command, host=None):
            if command == "RESET":
                reset_calls.append(command)
            return ""

    class BE(MockBackend):
        def up(self, spec, caps):
            return W()

    # reset declared only in world.yml metadata (x-opencrl.reset), not Task.reset
    task = Task(goal="g", reward=flag("CTF{win}"), tools=(shell,), name="p",
                caps=Caps(), max_steps=2, backend=BE(),
                world={"x-opencrl": {"agent": "box", "reset": "RESET"},
                       "services": {"box": {"image": "x"}}})
    rs = run_batch(task, _model(), n=3, concurrency=1, backend=BE(), pool=True)
    assert len(rs) == 3 and all(r.reward == 1.0 for r in rs)
    assert len(reset_calls) == 2   # single pooled world reset between the 3 episodes


def test_pool_propagates_worker_exception_and_tears_down():
    ups, downs = [], []

    class W(MockWorld):
        def exec(self, command, host=None):
            return ""

    class BE(MockBackend):
        def up(self, spec, caps):
            ups.append(1)
            return W()

        def down(self, world):
            downs.append(1)

    def boom(m, t):
        raise RuntimeError("model boom")

    task = Task(goal="g", reward=flag("CTF{win}"), tools=(shell,), name="p",
                caps=Caps(), max_steps=2, backend=BE(), reset="RESET")
    with pytest.raises(RuntimeError, match="model boom"):
        run_batch(task, boom, n=6, concurrency=2, backend=BE(), pool=True)
    assert downs and len(downs) == len(ups)   # every world that came up was torn down


def test_pool_tears_down_all_worlds_even_if_one_down_fails():
    downs = []

    class W(MockWorld):
        def exec(self, command, host=None):
            return ""

    class BE(MockBackend):
        def up(self, spec, caps):
            return W()

        def down(self, world):
            downs.append(1)
            if len(downs) == 1:
                raise RuntimeError("down boom")

    task = Task(goal="g", reward=flag("CTF{win}"), tools=(shell,), name="p",
                caps=Caps(), max_steps=2, backend=BE(), reset="RESET")
    model = lambda m, t: {"role": "assistant", "content": "CTF{win}", "tool_calls": None}
    with pytest.raises(RuntimeError, match="down boom"):
        run_batch(task, model, n=6, concurrency=2, backend=BE(), pool=True)
    assert len(downs) == 2   # both pooled worlds torn down despite the first failing

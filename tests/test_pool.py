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

"""GroupPool: persistent worlds shared across a task group's tasks."""
import json
import threading

import pytest
from opencrl.backends.mock import MockBackend, MockWorld
from opencrl.reward import flag
from opencrl.runner import GroupPool, run_group
from opencrl.task import Caps, Task
from opencrl.tools import shell


def _task(name, reset="RESET"):
    """A minimal winnable task; reset=None leaves the pool hook unset."""
    return Task(goal="g", reward=flag("CTF{win}"), tools=(shell,), name=name,
                caps=Caps(), max_steps=3, reset=reset)


def _win(m, t):
    return {"role": "assistant", "content": "CTF{win}", "tool_calls": None}


def _tool_call(command):
    """An OpenAI-style tool call running `command` through the shell tool."""
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": "c", "function": {
                "name": "shell",
                "arguments": json.dumps({"command": command})}}]}


class CountingBackend(MockBackend):
    """MockBackend recording the identity of every world upped/downed."""

    def __init__(self):
        super().__init__()
        self.ups = []
        self.downs = []

    def up(self, spec, caps):
        w = MockWorld()
        self.ups.append(w)
        return w

    def down(self, world):
        self.downs.append(world)


def _torn_down(be):
    """The set of downed world identities (objects lack __eq__)."""
    return set(map(id, be.downs))


def _brought_up(be):
    return set(map(id, be.ups))


def test_group_pool_reuses_worlds_across_tasks():
    resets = []

    class W(MockWorld):
        def exec(self, command, host=None):
            if command == "RESET":
                resets.append(command)
            return ""

    class BE(CountingBackend):
        def up(self, spec, caps):
            w = W()
            self.ups.append(w)
            return w

    be = BE()
    tasks = [_task(f"t{i}") for i in range(3)]
    seen = []
    rs = run_group(tasks, _win, episodes=3, concurrency=1, capacity=2,
                   backend=be, on_result=lambda i, r: seen.append(i))

    assert len(rs) == 9 and all(r.reward == 1.0 for r in rs)
    assert [r.task for r in rs] == ["t0"] * 3 + ["t1"] * 3 + ["t2"] * 3
    assert seen == list(range(9))       # on_result fires in submission order
    assert len(be.ups) == 3             # one bring-up per task, not per episode
    assert len(resets) == 6             # 9 episodes minus the 3 cold firsts
    assert _torn_down(be) == _brought_up(be) and len(be.downs) == 3


def test_cold_bringup_evicts_lru_idle_when_full():
    be = CountingBackend()
    pool = GroupPool(be, capacity=2)
    t1, t2, t3 = _task("t1"), _task("t2"), _task("t3")

    w1 = pool.draw(t1)
    pool.release(w1)                    # idle: [w1]
    w2 = pool.draw(t2)
    pool.release(w2)                    # idle: [w1, w2]
    w3 = pool.draw(t3)                  # full: evict the LRU idle world (w1)
    assert be.downs == [w1]
    assert be.ups == [w1, w2, w3]
    pool.release(w3)                    # idle: [w2, w3]

    assert pool.draw(t2) is w2          # w2 survived the eviction: warm reuse
    assert len(be.ups) == 3
    pool.release(w2)
    pool.close()
    assert _torn_down(be) == _brought_up(be)


def test_lru_tracks_most_recent_use():
    be = CountingBackend()
    pool = GroupPool(be, capacity=2)
    t1, t2, t3 = _task("t1"), _task("t2"), _task("t3")

    w1 = pool.draw(t1)
    pool.release(w1)                    # idle: [w1]
    w2 = pool.draw(t2)
    pool.release(w2)                    # idle: [w1, w2]
    pool.draw(t1)                       # touch w1: it becomes most recently used
    pool.release(w1)                    # idle: [w2, w1]
    pool.draw(t3)                       # evicts w2, not the freshly used w1
    assert be.downs == [w2]
    pool.close()


def test_inflight_world_is_never_evicted():
    be = CountingBackend()
    pool = GroupPool(be, capacity=2)
    tx, ty, tz = _task("tx"), _task("ty"), _task("tz")
    barrier = threading.Barrier(2, timeout=5)

    def holder():
        w = pool.draw(tx)               # in flight for the whole churn below
        barrier.wait()                  # main: start churning
        barrier.wait()                  # main: done churning
        pool.release(w)

    a = threading.Thread(target=holder)
    a.start()
    barrier.wait()                      # tx's world is now in flight
    held = be.ups[0]

    wy = pool.draw(ty)
    pool.release(wy)                    # pool full: one in flight + [wy]
    wz = pool.draw(tz)                  # must evict wy — never the held world
    assert be.downs == [wy]
    pool.release(wz)
    wy2 = pool.draw(ty)                 # another eviction, same rule
    assert be.downs == [wy, wz]
    pool.release(wy2)

    barrier.wait()
    a.join()
    pool.close()
    assert len(be.ups) == 4 and held not in be.downs[:2]   # downed only at teardown
    assert _torn_down(be) == _brought_up(be)


def test_reset_scrubs_agent_writes_between_episodes():
    class DirtyWorld(MockWorld):
        """World an agent can contaminate; RESET restores initial state."""
        def __init__(self):
            super().__init__()
            self.dirty = False

        def exec(self, command, host=None):
            if command == "WRITE":
                self.dirty = True
            elif command == "RESET":
                self.dirty = False
            elif command == "STATUS":
                return "DIRTY" if self.dirty else "CLEAN"
            return ""

    class BE(CountingBackend):
        def up(self, spec, caps):
            w = DirtyWorld()
            self.ups.append(w)
            return w

    be = BE()
    probes = []

    def model(messages, tools):
        # every episode probes cleanliness, then contaminates, then answers
        if len(messages) == 2:
            return _tool_call("STATUS")
        if len(messages) == 4:
            probes.append(messages[-1]["content"])
            return _tool_call("WRITE")
        return {"role": "assistant", "content": "CTF{win}", "tool_calls": None}

    rs = run_group([_task("t")], model, episodes=3, concurrency=1,
                   capacity=1, backend=be)
    assert len(rs) == 3 and all(r.reward == 1.0 for r in rs)
    assert probes == ["CLEAN", "CLEAN", "CLEAN"]   # each episode starts pristine
    assert len(be.ups) == 1                        # one world, reset and reused


def test_group_pool_requires_reset_hook_before_prebuild():
    prebuilds = []

    class BE(CountingBackend):
        def prebuild(self, spec, caps):
            prebuilds.append(spec)

    be = BE()
    tasks = [_task("t1"), _task("t2", reset=None)]
    with pytest.raises(ValueError, match="reset"):
        run_group(tasks, _win, episodes=2, backend=be)
    assert prebuilds == []             # validation fires before any build work
    assert be.ups == []                # and before any world comes up


def test_worker_failure_tears_down_every_live_world():
    be = CountingBackend()
    calls = []

    def model(m, t):
        calls.append(1)
        if len(calls) == 4:            # fails during the last episode
            raise RuntimeError("model boom")
        return {"role": "assistant", "content": "CTF{win}", "tool_calls": None}

    tasks = [_task("t1"), _task("t2")]
    with pytest.raises(RuntimeError, match="model boom"):
        run_group(tasks, model, episodes=2, concurrency=1, capacity=2, backend=be)
    assert be.ups                       # two tasks ran before the failure
    assert _torn_down(be) == _brought_up(be)


def test_teardown_continues_when_one_down_fails():
    downs = []

    class BE(MockBackend):
        def up(self, spec, caps):
            return MockWorld()

        def down(self, world):
            downs.append(1)
            if len(downs) == 1:
                raise RuntimeError("down boom")

    tasks = [_task("t1"), _task("t2")]
    with pytest.raises(RuntimeError, match="down boom"):
        run_group(tasks, _win, episodes=1, concurrency=1, capacity=2, backend=BE())
    assert len(downs) == 2              # both worlds torn down despite the failure


def test_draw_waits_when_all_live_worlds_are_inflight():
    resets = []

    class W(MockWorld):
        def exec(self, command, host=None):
            if command == "RESET":
                resets.append(command)
            return ""

    class BE(CountingBackend):
        def up(self, spec, caps):
            w = W()
            self.ups.append(w)
            return w

    be = BE()
    task = _task("t")
    playing = threading.Event()         # first rollout holds the only world
    release = threading.Event()
    rollouts = []

    def factory():
        n = len(rollouts)
        rollouts.append(1)

        def model(m, t):
            if n == 0:                  # holder: block mid-episode on command
                playing.set()
                assert release.wait(timeout=5)
            return {"role": "assistant", "content": "CTF{win}", "tool_calls": None}

        return model

    box = {}
    done = threading.Event()

    def run():
        try:
            box["rs"] = run_group([task], None, model_factory=factory,
                                  episodes=2, concurrency=2, capacity=1, backend=be)
        except BaseException as e:
            box["err"] = e
        finally:
            done.set()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    assert playing.wait(timeout=5)      # one worker is mid-episode; the other blocks
    release.set()                       # let it finish; the blocked draw now reuses
    assert done.wait(timeout=5)
    t.join()

    assert "err" not in box
    rs = box["rs"]
    assert len(rs) == 2 and all(r.reward == 1.0 for r in rs)
    assert len(be.ups) == 1             # the second worker waited, then reused
    assert len(resets) == 1             # warm draw reset before the second episode


def test_run_group_empty_group_never_touches_backend():
    be = CountingBackend()
    assert run_group([], _win, backend=be) == []
    assert run_group([_task("t")], _win, episodes=0, backend=be) == []
    assert be.ups == [] and be.downs == []


def test_group_pool_rejects_nonpositive_capacity():
    with pytest.raises(ValueError, match="capacity"):
        GroupPool(CountingBackend(), capacity=0)


def test_release_unknown_world_is_rejected():
    pool = GroupPool(CountingBackend(), capacity=1)
    with pytest.raises(ValueError, match="checked out"):
        pool.release(MockWorld())

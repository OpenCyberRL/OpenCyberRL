"""Run many rollouts concurrently. The Runner seam keeps rollout() plain-sync.

ThreadRunner is the only implementation today; rollouts are I/O-bound (subprocess
to `docker`, HTTP to the model) so threads give real parallelism with no async
contagion into the core. AsyncRunner is a documented, unbuilt seam for a future
remote-sandbox backend (it would implement the same Runner protocol).
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from typing import Callable, Protocol

from opencrl.backend import resolve_backend
from opencrl.rollout import Rollout, _play, rollout
from opencrl.task import Task, load_world


def _model_provider(model, model_factory) -> "Callable[[int], object]":
    """Return a provider(i) -> model. Exactly one of `model` (shared across
    rollouts; must be safe for concurrent calls, e.g. OpenAIModel) or
    `model_factory` (a fresh instance per rollout, e.g. ScriptedModel).

    A factory is called lazily, as each rollout starts, so at most the active
    workers' models are ever live at once — n=100, concurrency=4 builds 4 at a
    time, not 100 upfront.
    """
    if (model is None) == (model_factory is None):
        raise ValueError(
            "opencrl: pass exactly one of model= (shared across rollouts; must be "
            "safe for concurrent calls, e.g. OpenAIModel) or model_factory= "
            "(called once per rollout for a fresh instance, e.g. ScriptedModel)")
    if model_factory is not None:
        return lambda i: model_factory()
    return lambda i: model


class Runner(Protocol):
    def run(self, task: Task, *, provider: "Callable[[int], object]", n: int,
            backend, on_result: "Callable[[int, Rollout], None] | None" = None
            ) -> list[Rollout]: ...


class ThreadRunner:
    """Bounded thread pool. Default workers = min(n, os.cpu_count() or 1)."""

    def __init__(self, concurrency: int | None = None):
        self.concurrency = concurrency

    def run(self, task, *, provider, n, backend, on_result=None) -> list[Rollout]:
        workers = self.concurrency or min(n, os.cpu_count() or 1)
        workers = max(1, min(workers, n))
        results: list[Rollout | None] = [None] * n
        lock = threading.Lock()
        abort = threading.Event()  # first exception stops pending rollouts

        def one(i: int) -> None:
            if abort.is_set():  # a prior rollout raised; skip (serial stop-on-first)
                return
            try:
                r = rollout(task, provider(i), backend=backend)
                if on_result is not None:
                    with lock:
                        on_result(i, r)
                results[i] = r
            except BaseException:  # rollout OR on_result failure stops pending work
                abort.set()
                raise

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(one, i) for i in range(n)]
            for f in futures:
                f.result()
        return results  # type: ignore[return-value]

def _pool_reset_hook(task: "Task", world_spec: dict) -> "str | None":
    """A task's pool reset hook: Task.reset, else x-opencrl.reset metadata."""
    return task.reset or (world_spec.get("x-opencrl") or {}).get("reset")


def _require_pool_reset(task: "Task", world_spec: dict) -> str:
    """Return the reset hook or raise — pooling without one silently corrupts scoring."""
    reset_cmd = _pool_reset_hook(task, world_spec)
    if not reset_cmd:
        raise ValueError(
            "opencrl: pool=True requires a reset hook — set Task.reset or "
            "x-opencrl.reset in world.yml (a shell command that restores the "
            "world to its initial state); none is set")
    return reset_cmd


class PoolRunner:
    """Keep `concurrency` live worlds; reset (not down/up) between episodes.

    Trusts the reset hook to fully restore initial state. Off by default and
    requires a reset hook — a bad reset silently corrupts scoring.
    """

    def __init__(self, concurrency: int | None = None, reset: str | None = None):
        self.concurrency = concurrency
        self.reset = reset

    def run(self, task, *, provider, n, backend, on_result=None) -> list[Rollout]:
        size = self.concurrency or min(n, os.cpu_count() or 1)
        size = max(1, min(size, n))
        results: list[Rollout | None] = [None] * n
        jobs: "Queue[int]" = Queue()
        for i in range(n):
            jobs.put(i)
        lock = threading.Lock()
        worlds: list = []
        worlds_lock = threading.Lock()
        abort = threading.Event()            # first failure stops pending jobs
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                world = backend.up(load_world(task), task.caps)
            except BaseException as e:       # never came up: nothing to tear down
                with lock:
                    errors.append(e)
                abort.set()
                return
            with worlds_lock:
                worlds.append(world)
            first = True
            while not abort.is_set():
                try:
                    i = jobs.get_nowait()
                except Empty:
                    return
                try:
                    if not first:
                        world.exec(self.reset)   # restore initial state between episodes
                    first = False
                    r = _play(task, provider(i), world)
                    if on_result is not None:
                        with lock:
                            on_result(i, r)
                    results[i] = r
                except BaseException as e:    # reset / _play / on_result failure
                    with lock:
                        errors.append(e)
                    abort.set()
                    return

        threads = [threading.Thread(target=worker) for _ in range(size)]
        started: list = []
        try:
            for t in threads:
                t.start()
                started.append(t)
            for t in started:
                t.join()
        finally:
            # Even on interruption (Ctrl-C during join) or a failed start, stop
            # workers pulling new jobs and let every started worker finish before
            # tearing worlds down — never down() a world under an active rollout.
            abort.set()
            for t in started:
                t.join()
            for w in worlds:                 # tear down every world, even if one fails
                try:
                    backend.down(w)
                except BaseException as e:
                    errors.append(e)
        if errors:                           # match ThreadRunner: re-raise the first failure
            raise errors[0]
        return results  # type: ignore[return-value]


def run_batch(task: Task, model=None, *, model_factory=None, n: int,
              concurrency: int | None = None, backend=None,
              on_result: "Callable[[int, Rollout], None] | None" = None,
              pool: bool = False) -> list[Rollout]:
    """Run n rollouts concurrently; return Rollouts in submission order.

    Resolves the backend ONCE (so a shared Docker instance can reuse built
    images) and prebuilds once before fan-out to avoid a thundering herd of
    identical image builds. With pool=True (Task.reset or x-opencrl.reset), keeps
    a small pool of live worlds and resets between episodes instead of down/up.
    """
    provider = _model_provider(model, model_factory)   # validates model xor factory
    if n <= 0:                                          # no work: never touch the backend
        return []
    resolved = resolve_backend(backend or task.backend)
    world_spec = load_world(task)
    # Validate the pool reset hook BEFORE prebuilding — prebuild can trigger
    # an expensive, side-effecting Docker build, and pool=True without a reset
    # hook is a user error that should fail fast before any build runs.
    reset_cmd = None
    if pool:
        reset_cmd = _require_pool_reset(task, world_spec)
    prebuild = getattr(resolved, "prebuild", None)
    if prebuild is not None:
        prebuild(world_spec, task.caps)
    if pool:
        runner: Runner = PoolRunner(concurrency, reset=reset_cmd)
    else:
        runner = ThreadRunner(concurrency)
    return runner.run(task, provider=provider, n=n, backend=resolved, on_result=on_result)


class GroupPool:
    """Persistent worlds shared across a task group; LRU eviction of idle worlds.

    draw(task) checks a world out for one episode: a warm idle world for the
    task runs its reset hook and is reused; a cold task is brought up via
    backend.up, evicting the least-recently-used idle world once `capacity`
    worlds are live. A checked-out world is in flight and is NEVER evicted —
    the hard invariant (note: close() is not eviction; it tears down every
    live world, checked-out ones included, and owns all teardown afterwards).
    release() returns it as most-recently-used idle; discard() tears it down
    instead (a failed episode's world must not be reused). Every task needs
    a reset hook (Task.reset or x-opencrl.reset).
    """

    def __init__(self, backend, capacity: int):
        if capacity < 1:
            raise ValueError(
                f"opencrl: group pool capacity must be >= 1, got {capacity}")
        self._backend = backend
        self._capacity = capacity
        self._cond = threading.Condition()
        # world -> (task id, reset hook); idle is LRU-ordered (oldest first).
        self._idle: "OrderedDict[object, tuple[int, str]]" = OrderedDict()
        self._inflight: dict[object, tuple[int, str]] = {}
        self._reserved = 0                  # bring-ups in progress, counted as live
        self._entries: dict[int, tuple] = {}  # task id -> (task, world spec, reset hook)
        self._closed = False

    def draw(self, task):
        """Check out a world for one episode on `task` (see class docstring)."""
        task_obj, spec, reset_cmd = self.entry(task)
        world = victim = None
        with self._cond:
            while True:
                self._check_open()
                world = self._warm(id(task))
                if world is not None:
                    break
                live = len(self._idle) + len(self._inflight) + self._reserved
                if live < self._capacity:
                    self._reserved += 1
                    break
                if self._idle:             # full, but an idle world can go
                    victim, _ = self._idle.popitem(last=False)
                    self._reserved += 1
                    break
                self._cond.wait()          # every live world is in flight
        if world is not None:              # warm: restore initial state, reuse
            try:
                world.exec(reset_cmd)
            except BaseException:
                self.discard(world)        # drop the poisoned world; report the reset failure
                raise
            return world
        if victim is not None:             # cold: make room, then bring up
            try:
                self._backend.down(victim)
            except BaseException:
                self._unreserve()
                raise
        try:
            world = self._backend.up(spec, task_obj.caps)
        except BaseException:
            self._unreserve()
            raise
        stale = False
        with self._cond:
            self._reserved -= 1
            stale = self._closed           # close() raced this bring-up
            if not stale:
                self._inflight[world] = (id(task_obj), reset_cmd)
        if stale:
            self._backend.down(world)
            raise RuntimeError("opencrl: group pool closed during bring-up")
        return world

    def release(self, world) -> None:
        """Return a checked-out world to the pool as most-recently-used idle.

        A no-op once the pool is closed: close() owns teardown from then on
        and has already downed every live world — downing again would be a
        double teardown.
        """
        with self._cond:
            if self._closed:            # close() already tore this world down
                return
            lease = self._inflight.pop(world, None)
            if lease is None:
                raise ValueError(
                    "opencrl: release() on a world that is not checked out")
            self._idle[world] = lease
            self._cond.notify_all()     # a blocked draw may now proceed

    def discard(self, world) -> None:
        """Drop a checked-out world and tear it down (best effort).

        For worlds whose episode failed: unlike release() the world never
        returns to idle (a contaminated world must not be reused), and a
        failing down() is swallowed so it cannot mask the episode error.
        A no-op once the pool is closed: close() owns teardown from then on.
        """
        with self._cond:
            self._inflight.pop(world, None)
            closed = self._closed
            self._cond.notify_all()     # live count dropped: wake blocked draws
        if closed:                      # close() already tore this world down
            return
        try:
            self._backend.down(world)
        except BaseException:
            pass

    def close(self) -> None:
        """Tear down every live world, idle or in flight; raise the first failure.

        Idempotent. Does NOT wait for in-flight episodes: call it only after
        every draw/release/discard has returned (run_group guarantees this by
        joining its workers before closing). Afterwards every live world has
        been downed exactly once and release()/discard() are no-ops.
        """
        with self._cond:
            if self._closed:
                return
            self._closed = True
            worlds = list(self._idle) + list(self._inflight)
            self._idle.clear()
            self._inflight.clear()
            self._cond.notify_all()     # wake blocked draws (they will raise)
        errors: list[BaseException] = []
        for w in worlds:                # tear down every world, even if one fails
            try:
                self._backend.down(w)
            except BaseException as e:
                errors.append(e)
        if errors:
            raise errors[0]

    def entry(self, task):
        """(task, world spec, reset hook) for a task, cached; validates the hook.

        Public so callers like run_group can validate hooks and prebuild
        before any world comes up. Concurrent first calls may race the cache;
        the recomputed entry is identical, so the benign overwrite needs no
        lock (load_world is file I/O and stays off the pool lock).
        """
        cached = self._entries.get(id(task))
        if cached is None:
            spec = load_world(task)
            cached = (task, spec, _require_pool_reset(task, spec))
            self._entries[id(task)] = cached
        return cached

    def _warm(self, key: int):
        """Pop an idle world for task `key`, marking it in flight."""
        found = None
        for world, lease in self._idle.items():
            if lease[0] == key:
                found = (world, lease)
                break
        if found is None:
            return None
        del self._idle[found[0]]        # mutate only after the scan ends
        self._inflight[found[0]] = found[1]
        return found[0]

    def _unreserve(self) -> None:
        """Release a failed bring-up's reservation and wake blocked draws."""
        with self._cond:
            self._reserved -= 1
            self._cond.notify_all()

    def _check_open(self) -> None:
        """Raise if the pool has been closed."""
        if self._closed:
            raise RuntimeError("opencrl: group pool is closed")


def run_group(tasks, model=None, *, model_factory=None, episodes: int = 1,
              concurrency: int | None = None, backend=None,
              on_result: "Callable[[int, Rollout], None] | None" = None,
              capacity: int | None = None) -> list[Rollout]:
    """Run `episodes` rollouts per task over a task group on one shared pool.

    Worlds persist across the group's tasks: a warm idle world for a task is
    reset and reused; a cold task is brought up, evicting the LRU idle world
    once `capacity` worlds are live (default: one per worker, so draws never
    wait). The group shares one backend (the `backend` arg, else the first
    task's). Returns Rollouts in submission order — task order, then episode
    order. Every task needs a reset hook (Task.reset or x-opencrl.reset),
    validated before any prebuild runs.
    """
    tasks = list(tasks)
    provider = _model_provider(model, model_factory)   # validates model xor factory
    if capacity is not None and capacity < 1:
        raise ValueError(
            f"opencrl: group pool capacity must be >= 1, got {capacity}")
    jobs = [(t, e) for t in tasks for e in range(episodes)]
    if not jobs:                                       # no work: never touch the backend
        return []
    resolved = resolve_backend(backend or tasks[0].backend)
    size = concurrency or min(len(jobs), os.cpu_count() or 1)
    size = max(1, min(size, len(jobs)))
    pool = GroupPool(resolved, capacity or size)
    # Validate reset hooks + load specs before prebuild; dict dedupes tasks.
    entries = {id(t): pool.entry(t) for t in tasks}
    prebuild = getattr(resolved, "prebuild", None)
    if prebuild is not None:           # prebuild each distinct task once, not per episode
        for task_obj, spec, _reset in entries.values():
            prebuild(spec, task_obj.caps)
    results: list[Rollout | None] = [None] * len(jobs)
    jobs_q: "Queue[int]" = Queue()
    for i in range(len(jobs)):
        jobs_q.put(i)
    lock = threading.Lock()
    abort = threading.Event()          # first failure stops pending jobs
    errors: list[BaseException] = []

    def worker() -> None:
        while not abort.is_set():
            try:
                i = jobs_q.get_nowait()
            except Empty:
                return
            task = jobs[i][0]
            world = None
            try:
                world = pool.draw(task)
                r = _play(task, provider(i), world)
                pool.release(world)    # idle again: evictable, reusable
                world = None
                if on_result is not None:
                    with lock:
                        on_result(i, r)
                results[i] = r
            except BaseException as e:  # draw / _play / on_result failure
                if world is not None:
                    pool.discard(world)  # tear the failed world down; wake blocked draws
                with lock:
                    errors.append(e)
                abort.set()
                return

    threads = [threading.Thread(target=worker) for _ in range(size)]
    started: list = []
    try:
        for t in threads:
            t.start()
            started.append(t)
        for t in started:
            t.join()
    finally:
        # Even on interruption (Ctrl-C during join) or a failed start, stop
        # workers pulling new jobs and let every started worker finish before
        # tearing worlds down — never down() a world under an active rollout.
        abort.set()
        for t in started:
            t.join()
        try:
            pool.close()               # every live world downed, even after failure
        except BaseException as e:
            errors.append(e)
    if errors:                         # match run_batch: re-raise the first failure
        raise errors[0]
    return results  # type: ignore[return-value]

"""Run many rollouts concurrently. The Runner seam keeps rollout() plain-sync.

ThreadRunner is the only implementation today; rollouts are I/O-bound (subprocess
to `docker`, HTTP to the model) so threads give real parallelism with no async
contagion into the core. AsyncRunner is a documented, unbuilt seam for a future
remote-sandbox backend (it would implement the same Runner protocol).
"""
from __future__ import annotations

import os
import threading
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
        reset_cmd = task.reset or (world_spec.get("x-opencrl") or {}).get("reset")
        if not reset_cmd:
            raise ValueError(
                "opencrl: pool=True requires a reset hook — set Task.reset or "
                "x-opencrl.reset in world.yml (a shell command that restores the "
                "world to its initial state); none is set")
    prebuild = getattr(resolved, "prebuild", None)
    if prebuild is not None:
        prebuild(world_spec, task.caps)
    if pool:
        runner: Runner = PoolRunner(concurrency, reset=reset_cmd)
    else:
        runner = ThreadRunner(concurrency)
    return runner.run(task, provider=provider, n=n, backend=resolved, on_result=on_result)

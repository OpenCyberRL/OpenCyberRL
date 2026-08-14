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
from typing import Callable, Protocol

from opencrl.backend import resolve_backend
from opencrl.rollout import Rollout, rollout
from opencrl.task import Task, load_world


def _resolve_models(model, model_factory, n: int) -> list:
    """Exactly one of model (shared, must be concurrency-safe) or model_factory."""
    if (model is None) == (model_factory is None):
        raise ValueError(
            "opencrl: pass exactly one of model= (shared across rollouts; must be "
            "safe for concurrent calls, e.g. OpenAIModel) or model_factory= "
            "(called once per rollout for a fresh instance, e.g. ScriptedModel)")
    if model_factory is not None:
        return [model_factory() for _ in range(n)]
    return [model] * n


class Runner(Protocol):
    def run(self, task: Task, *, models: list, backend,
            on_result: "Callable[[int, Rollout], None] | None" = None) -> list[Rollout]: ...


class ThreadRunner:
    """Bounded thread pool. Default workers = min(n, os.cpu_count() or 1)."""

    def __init__(self, concurrency: int | None = None):
        self.concurrency = concurrency

    def run(self, task, *, models, backend, on_result=None) -> list[Rollout]:
        n = len(models)
        workers = self.concurrency or min(n, os.cpu_count() or 1)
        workers = max(1, min(workers, n)) if n else 1
        results: list[Rollout | None] = [None] * n
        lock = threading.Lock()
        abort = threading.Event()  # first exception stops pending rollouts

        def one(i: int) -> None:
            if abort.is_set():  # a prior rollout raised; skip (serial stop-on-first)
                return
            try:
                r = rollout(task, models[i], backend=backend)
            except BaseException:
                abort.set()
                raise
            if on_result is not None:
                with lock:
                    on_result(i, r)
            results[i] = r

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(one, i) for i in range(n)]
            for f in futures:
                f.result()
        return results  # type: ignore[return-value]


def run_batch(task: Task, model=None, *, model_factory=None, n: int,
              concurrency: int | None = None, backend=None,
              on_result: "Callable[[int, Rollout], None] | None" = None) -> list[Rollout]:
    """Run n rollouts concurrently; return Rollouts in submission order.

    Resolves the backend ONCE (so a shared Docker instance can reuse built
    images) and prebuilds once before fan-out to avoid a thundering herd of
    identical image builds.
    """
    models = _resolve_models(model, model_factory, n)
    resolved = resolve_backend(backend or task.backend)
    prebuild = getattr(resolved, "prebuild", None)
    if prebuild is not None:
        prebuild(load_world(task), task.caps)
    return ThreadRunner(concurrency).run(task, models=models, backend=resolved,
                                         on_result=on_result)

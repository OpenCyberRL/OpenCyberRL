"""`opencrl warm <group>`: eagerly prebuild a task group's images.

Resolves a group expression against a discovered task index (via
`opencrl.groups.resolve_group`), then runs the Docker backend's prebuild
machinery for every resolved task so training starts immediately instead of
paying first-use builds. A failed build is recorded and reported — it never
aborts the remaining tasks.
"""
from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from opencrl.backend import resolve_backend
from opencrl.backends import docker as docker_backend
from opencrl.groups import GroupExpr, TaskIndex, resolve_group
from opencrl.task import discover, get_task, list_tasks, load_world

# Terminal status order used by summarize() and CLI coloring.
_STATUSES = ("built", "cached", "no-build", "skipped", "failed")


@dataclass(frozen=True)
class WarmResult:
    """Outcome of warming one task."""
    name: str
    status: str            # one of _STATUSES
    error: str | None = None


def build_index(path: str | None = None) -> list[dict]:
    """Discover tasks and derive a group-spec index (name/module/project/level).

    The module is inferred from the task's dir path relative to the modules
    repo (first path component), or ``"local"`` for tasks outside it (./tasks/
    or an explicit ``--path``). ``project`` defaults to ``None`` and ``level``
    to ``0`` — cybergym's richer module-side index (``index.yaml`` + loader)
    will layer richer metadata on top of this derivation later.
    """
    discover(path)
    return [_entry(get_task(name)) for name in list_tasks()]


def _entry(task) -> dict:
    """One index row for a task."""
    return {"name": task.name, "module": _module_of(task),
            "project": None, "level": 0}


def _module_of(task) -> str:
    """Task's module: dir relative to the modules repo, else ``"local"``."""
    from opencrl import modules
    try:
        rel = Path(task.dir).resolve().relative_to(modules.modules_dir().resolve())
    except ValueError:
        return "local"
    return rel.parts[0] if rel.parts else "local"


def warm_group(expr: GroupExpr, index: TaskIndex, backend,
               report: Callable[[WarmResult], None] | None = None) -> list[WarmResult]:
    """Resolve ``expr`` against ``index`` and prebuild every task's images.

    Raises ``ValueError`` (from ``resolve_group``) for an unresolvable
    expression; per-task build failures become ``failed`` results instead.
    """
    names = resolve_group(expr, index)
    return [_warm_one(name, backend, report) for name in names]


def _warm_one(name: str, backend, report) -> WarmResult:
    """Warm one task; any error becomes a ``failed`` result, never a raise."""
    try:
        result = _classify_and_build(name, backend)
    except Exception as exc:
        result = WarmResult(name, "failed", _reason(exc))
    if report is not None:
        report(result)
    return result


def _classify_and_build(name: str, backend) -> WarmResult:
    """Prebuild one task, classifying the outcome against the pinned tags."""
    task = get_task(name)
    if not _is_docker_task(task):
        return WarmResult(name, "skipped",
                          f"backend {task.backend!r} has no images to prebuild")
    spec = load_world(task)
    tags = _pinned_tags(backend, spec, task.caps)
    if tags == []:
        return WarmResult(name, "no-build")
    before = {t: _layer_chain(t) for t in tags} if tags else None
    backend.prebuild(spec, task.caps)
    if before is None:                # unknowable backend: prebuild ran, that's all
        return WarmResult(name, "built")
    after = {t: _layer_chain(t) for t in tags}
    # A pinned tag may exist yet hold stale layers (its digest covers the
    # build config, not file contents), and BuildKit rewrites provenance
    # metadata (hence the image ID) on every build — so "cached" requires
    # the layer chain to be unchanged across the prebuild.
    cached = all(before[t] is not None and before[t] == after[t] for t in tags)
    return WarmResult(name, "cached" if cached else "built")


def _pinned_tags(backend, spec: dict, caps) -> list[str] | None:
    """Content-addressed tags prebuild will pin for this world spec.

    ``None`` when the backend doesn't expose the ``pinned_tags`` seam
    (e.g. a test double) — the outcome then can't be pre-classified.
    """
    pinned = getattr(backend, "pinned_tags", None)
    if pinned is None:
        return None
    return pinned(spec, caps)


def _is_docker_task(task) -> bool:
    """Whether the task's backend resolves to the Docker backend."""
    try:
        be = resolve_backend(task.backend)
    except KeyError:                   # unknown backend name
        return False
    # Resolved late (not imported at module load) so tests that swap
    # docker.Docker for a fake still identify fakes as the docker backend.
    return isinstance(be, docker_backend.Docker)


def _layer_chain(tag: str) -> str | None:
    """The image's layer-chain fingerprint, or ``None`` when the tag is absent.

    A transient docker-inspect failure also reads as absent, so the caller
    classifies such a task as ``built`` rather than ``cached``.
    """
    try:
        cp = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{json .RootFS.Layers}}", tag],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return cp.stdout.strip() if cp.returncode == 0 else None


def _reason(exc: Exception, limit: int = 200) -> str:
    """Collapse an exception's message to one bounded line."""
    text = " ".join(str(exc).split())
    return text if len(text) <= limit else text[:limit] + "…"


def summarize(results: list[WarmResult]) -> str:
    """One-line status tally, e.g. ``"1 built, 1 cached, 1 failed"``."""
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    parts = [f"{counts[s]} {s}" for s in _STATUSES if s in counts]
    parts += [f"{counts[s]} {s}" for s in sorted(set(counts) - set(_STATUSES))]
    return ", ".join(parts) if parts else "no tasks warmed"

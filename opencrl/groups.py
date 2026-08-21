"""Group expressions: resolve a group spec to an ordered task-name list.

A *task index* is an abstract, already-materialized list of dicts, each shaped
``{"name": str, "module": str, "project": str | None, "level": int (default 0)}``.
The index format itself is owned by the modules that produce it; this module
only consumes it. Everything here is pure: no file I/O, docker, or network.

Group expressions:

- ``"module/level<N>"``   — all tasks of that module at level N (e.g. ``"cybergym/level1"``)
- ``"module/project=<name>"`` — all tasks of that module for one project
- an explicit list of task names (a ``str``/``list``/``tuple`` of names)

Output order is deterministic: filter results are sorted by task name (the
index order carries no meaning); explicit lists preserve the caller's order
(it is meaningful — the user wrote it that way). A bare string that is neither
a valid filter expression nor a known task name is an error.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

_LEVEL_RE = re.compile(r"^level(\d+)$")
_PROJECT_RE = re.compile(r"^project=(.+)$")

TaskIndex = Sequence[Mapping[str, object]]
GroupExpr = str | Sequence[str]


def validate_index(index: TaskIndex) -> None:
    """Validate the shape of a task index.

    Raises ``ValueError`` naming the offending entry when an entry is not a
    mapping, lacks a non-empty string ``name``/``module``, has a ``project``
    that is neither a string nor ``None``, a ``level`` that is not a
    non-negative int, or when task names are duplicated.
    """
    seen: set[str] = set()
    for i, entry in enumerate(index):
        if not isinstance(entry, Mapping):
            raise ValueError(f"index entry {i} is not a mapping: {entry!r}")
        name = entry.get("name")
        module = entry.get("module")
        for key, value in (("name", name), ("module", module)):
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"index entry {i} ({name!r}) needs a non-empty string {key!r}"
                )
        project = entry.get("project")
        if project is not None and not isinstance(project, str):
            raise ValueError(
                f"index entry {i} ({name!r}) has a non-string 'project': {project!r}"
            )
        level = entry.get("level", 0)
        if isinstance(level, bool) or not isinstance(level, int) or level < 0:
            raise ValueError(
                f"index entry {i} ({name!r}) has an invalid 'level': {level!r}"
            )
        if name in seen:
            raise ValueError(f"duplicate task name in index: {name!r}")
        seen.add(name)


def resolve_group(expr: GroupExpr, index: TaskIndex) -> list[str]:
    """Resolve a group expression to an ordered list of task names.

    Filters (``module/level<N>``, ``module/project=<name>``) return names
    sorted alphabetically; an explicit list of names preserves input order.
    Unknown modules, levels, projects, or task names — and any expression
    that resolves to nothing — raise ``ValueError`` naming what was tried
    and what exists.
    """
    validate_index(index)
    names = {entry["name"] for entry in index}
    if isinstance(expr, str):
        return _resolve_str(expr, index, names)
    if isinstance(expr, Sequence):
        return _resolve_explicit(list(expr), names)
    raise TypeError(f"group expression must be a string or sequence of strings, got {type(expr)!r}")


def _resolve_str(expr: str, index: TaskIndex, names: set[str]) -> list[str]:
    """Resolve a single string expression: filter form, else task name."""
    module, slash, suffix = expr.partition("/")
    level_m = _LEVEL_RE.match(suffix) if slash else None
    project_m = _PROJECT_RE.match(suffix) if slash else None
    if level_m is not None or project_m is not None:
        known_modules = sorted({entry["module"] for entry in index})
        if not _module_names(index, module):
            raise ValueError(
                f"unknown module {module!r} in group expression {expr!r}; "
                f"known modules: {_join(known_modules)}"
            )
        if level_m is not None:
            return _filter_level(index, module, int(level_m.group(1)))
        return _filter_project(index, module, project_m.group(1))
    # Not a filter form: the whole string is a task name.
    return _resolve_explicit([expr], names)


def _resolve_explicit(requested: list[str], names: set[str]) -> list[str]:
    """Resolve an explicit list of task names, preserving input order.

    Duplicates pass through verbatim: the caller's list is the caller's
    decision (e.g. running multiple replicas of one task), so deduplication
    is left to consumers.
    """
    if not requested:
        raise ValueError("empty group expression: an explicit task list must name at least one task")
    missing = [name for name in requested if name not in names]
    if missing:
        raise ValueError(
            f"unknown task(s) {missing!r} in group expression; "
            f"known tasks: {_join(sorted(names))}"
        )
    return list(requested)


def _filter_level(index: TaskIndex, module: str, level: int) -> list[str]:
    """All task names of ``module`` at ``level``, sorted by name."""
    matched = sorted(
        entry["name"]
        for entry in index
        if entry["module"] == module and entry.get("level", 0) == level
    )
    if not matched:
        levels = sorted({entry.get("level", 0) for entry in index if entry["module"] == module})
        raise ValueError(
            f"module {module!r} has no tasks at level {level}; "
            f"available levels: {_join(levels)}"
        )
    return matched


def _filter_project(index: TaskIndex, module: str, project: str) -> list[str]:
    """All task names of ``module`` for ``project``, sorted by name."""
    matched = sorted(
        entry["name"]
        for entry in index
        if entry["module"] == module and entry.get("project") == project
    )
    if not matched:
        projects = sorted(
            {
                entry.get("project")
                for entry in index
                if entry["module"] == module and entry.get("project") is not None
            }
        )
        raise ValueError(
            f"module {module!r} has no tasks for project {project!r}; "
            f"available projects: {_join(projects)}"
        )
    return matched


def _module_names(index: TaskIndex, module: str) -> set[str]:
    """Task names belonging to ``module`` (empty when the module is unknown)."""
    return {entry["name"] for entry in index if entry["module"] == module}


def _join(items: Sequence[object]) -> str:
    """Render a sorted collection for error messages; '(none)' when empty."""
    rendered = ", ".join(str(item) for item in items)
    return rendered if rendered else "(none)"

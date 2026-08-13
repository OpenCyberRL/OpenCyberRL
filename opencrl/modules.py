"""Module repo management — clone, activate, and discover community modules.

The community-contributed task modules live in a separate git repository
(``opencyberrl-modules``).  This module provides the plumbing to clone/update
that repo and track which modules the user has activated, all without pulling
in heavy optional dependencies.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

_MODULES_URL = "https://github.com/OpenCyberRL/opencyberrl-modules.git"


def _home() -> Path:
    """Return the opencrl home directory (``$OPENCRL_HOME`` or ``~/.opencrl``)."""
    return Path(os.environ.get("OPENCRL_HOME", os.path.expanduser("~/.opencrl")))


def modules_dir() -> Path:
    """Return the path where the community modules repo is cloned."""
    return _home() / "modules" / "opencyberrl-modules"


def active_file() -> Path:
    """Return the path to the file tracking activated module names."""
    return _home() / "active"


def active_modules() -> list[str]:
    """Return the list of activated module names (empty if file is absent)."""
    f = active_file()
    if not f.exists():
        return []
    return [line.strip() for line in f.read_text().splitlines() if line.strip()]


def _write_active(modules: list[str]) -> None:
    """Write the active-module list, creating parent directories as needed."""
    f = active_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("".join(name + "\n" for name in modules))


def activate_module(name: str) -> None:
    """Add ``name`` to the active-modules list (no-op if already present)."""
    active = active_modules()
    if name not in active:
        active.append(name)
        _write_active(active)


def deactivate_module(name: str) -> None:
    """Remove ``name`` from the active-modules list (no-op if not present)."""
    active = active_modules()
    if name in active:
        active.remove(name)
        _write_active(active)


def is_cloned() -> bool:
    """Return ``True`` when the modules repo has been cloned locally."""
    return (modules_dir() / ".git").exists()


def clone_modules_repo() -> None:
    """Shallow-clone the community modules repo into ``modules_dir()``."""
    target = modules_dir()
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", _MODULES_URL, str(target)],
        check=True,
        capture_output=True,
    )


def update_modules_repo() -> None:
    """Fast-forward pull the community modules repo."""
    subprocess.run(
        ["git", "-C", str(modules_dir()), "pull", "--ff-only"],
        check=True,
        capture_output=True,
    )


def list_available_modules() -> list[str]:
    """Return a sorted list of available module directory names.

    A directory is a *module* when it contains at least one ``*/task.py``
    (i.e. a ``task.py`` one level below the module root).  Directories whose
    name starts with ``"."`` are ignored.
    """
    md = modules_dir()
    if not md.exists():
        return []
    names = []
    for entry in md.iterdir():
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if any((entry / sub / "task.py").exists() for sub in os.listdir(entry)):
            names.append(entry.name)
    return sorted(names)


def active_module_paths() -> list[Path]:
    """Return paths for activated modules that actually exist on disk."""
    md = modules_dir()
    return [md / name for name in active_modules() if (md / name).exists()]

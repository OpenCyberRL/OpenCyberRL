"""opencrl CLI: install | uninstall | update | list | info | new."""
from __future__ import annotations

import argparse
import keyword
import sys
from pathlib import Path

from opencrl.task import discover, get_task, list_tasks

_TEMPLATE = Path(__file__).parent / "_template"

# ── rich helpers ───────────────────────────────────────────────────────────

def _console():
    from rich.console import Console
    return Console()

def _error(console, msg: str, hint: str | None = None):
    from rich.panel import Panel
    body = f"[red]✗ {msg}[/red]"
    if hint:
        body += f"\n\n[dim]{hint}[/dim]"
    console.print(Panel(body, border_style="red", title="[red]Error[/red]",
                        title_align="left"))

def _success(console, msg: str, hint: str | None = None):
    from rich.panel import Panel
    body = f"[green]✓ {msg}[/green]"
    if hint:
        body += f"\n\n[dim]{hint}[/dim]"
    console.print(Panel(body, border_style="green", title="[green]Success[/green]",
                        title_align="left"))

def _task_description(task) -> str:
    """Extract a short description from the task goal (first sentence or 60 chars)."""
    goal = task.goal
    # Truncate at first period if short enough, else at 60 chars
    if len(goal) <= 60:
        return goal
    cut = goal[:60]
    # Try to cut at a space boundary
    if " " in goal[55:]:
        cut = goal[:goal.index(" ", 55)]
    return cut + "…"


# ── commands ──────────────────────────────────────────────────────────────

def _cmd_install(args) -> int:
    from opencrl import modules
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.table import Table

    console = _console()

    if not modules.is_cloned():
        with Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[cyan]{task.description}"),
            BarColumn(bar_width=30, complete_style="cyan", finished_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            transient=True, console=console,
        ) as progress:
            task = progress.add_task(
                description="Cloning opencyberrl-modules", total=100)
            try:
                modules.clone_modules_repo()
                progress.update(task, completed=100)
            except Exception as e:
                _error(console, f"Failed to clone modules repo: {e}",
                       hint="Check your network connection and try again.")
                return 1
    else:
        # Silently update existing clone
        try:
            modules.update_modules_repo()
        except Exception:
            pass  # offline is fine

    available = modules.list_available_modules()
    if not available:
        _error(console, "No modules found in the modules repo.",
               hint="The repo might be empty or not yet cloned.")
        return 0

    active = modules.active_modules()

    if args.module is None:
        # List all available modules in a rich table
        table = Table(
            title="[bold cyan]Available Modules[/bold cyan]",
            show_header=True, header_style="bold",
            border_style="cyan",
        )
        table.add_column("Module", style="cyan", no_wrap=True)
        table.add_column("Tasks", justify="right", style="yellow")
        table.add_column("Status", justify="center")

        for name in available:
            mod_path = modules.modules_dir() / name
            task_count = sum(1 for _ in mod_path.glob("*/task.py"))
            if name in active:
                status = "[green]✓ active[/green]"
            else:
                status = "[dim]—[/dim]"
            table.add_row(name, str(task_count), status)

        console.print(table)
        console.print(
            f"\n[dim]Activate with:[/dim] [bold]opencrl install <module>[/bold]")
        return 0

    # Install specific module
    if args.module not in available:
        _error(console, f"Module '{args.module}' not found.",
               hint=f"Available modules: {', '.join(available)}")
        return 1

    modules.activate_module(args.module)
    mod_path = modules.modules_dir() / args.module

    # List individual tasks in the activated module
    task_dirs = sorted(mod_path.glob("*/task.py"))
    task_count = len(task_dirs)

    # Try to discover and show task descriptions
    discover()
    all_tasks = list_tasks()

    task_lines = []
    for task_py in task_dirs:
        task_name = task_py.parent.name
        if task_name in all_tasks:
            try:
                t = get_task(task_name)
                desc = _task_description(t)
                task_lines.append(f"  [green]✓[/green] {task_name:<16} [dim]{desc}[/dim]")
            except Exception:
                task_lines.append(f"  [green]✓[/green] {task_name}")
        else:
            task_lines.append(f"  [green]✓[/green] {task_name}")

    tasks_str = "\n".join(task_lines)
    _success(console,
             f"Activated module: [cyan]{args.module}[/cyan] ({task_count} tasks)",
             hint=tasks_str)
    console.print(
        f"\n[dim]Run[/dim] [bold]opencrl list[/bold] [dim]to see all tasks[/dim]")
    return 0


def _cmd_uninstall(args) -> int:
    from opencrl import modules
    from rich.panel import Panel

    console = _console()

    active = modules.active_modules()
    if args.module not in active:
        _error(console, f"Module '{args.module}' is not active.",
               hint=f"Active modules: {', '.join(active) or 'none'}")
        return 1

    modules.deactivate_module(args.module)
    _success(console, f"Deactivated module: [cyan]{args.module}[/cyan]")
    return 0


def _cmd_update(args) -> int:
    from opencrl import modules
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.table import Table

    console = _console()

    if not modules.is_cloned():
        _error(console, "No modules repo found.",
               hint="Run [bold]opencrl install[/bold] first.")
        return 1

    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[cyan]{task.description}"),
        BarColumn(bar_width=30, complete_style="cyan", finished_style="green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        transient=True, console=console,
    ) as progress:
        task = progress.add_task(description="Pulling latest modules", total=100)
        try:
            modules.update_modules_repo()
            progress.update(task, completed=100)
        except Exception as e:
            _error(console, f"Failed to update: {e}",
                   hint="Check your network connection and try again.")
            return 1

    available = modules.list_available_modules()
    active = modules.active_modules()

    table = Table(
        title="[bold cyan]Modules Updated[/bold cyan]",
        show_header=True, header_style="bold",
        border_style="green",
    )
    table.add_column("Module", style="cyan", no_wrap=True)
    table.add_column("Tasks", justify="right", style="yellow")
    table.add_column("Status", justify="center")
    for name in available:
        mod_path = modules.modules_dir() / name
        task_count = sum(1 for _ in mod_path.glob("*/task.py"))
        if name in active:
            status = "[green]✓ active[/green]"
        else:
            status = "[dim]—[/dim]"
        table.add_row(name, str(task_count), status)
    console.print(table)
    return 0


def _cmd_list(args) -> int:
    from rich.tree import Tree
    from rich.console import Group

    console = _console()
    discover(args.path)
    tasks = list_tasks()

    # Group tasks by source (local vs module)
    from opencrl import modules
    active = modules.active_modules() if modules.is_cloned() else []

    # Optional module filter: show only that module's tasks
    selected = getattr(args, "module", None)
    if selected is not None and selected not in ["local"] + active:
        _error(console, f"Unknown module '{selected}'.",
               hint="Available: " + ", ".join(["local"] + active))
        return 1

    if not tasks:
        _error(console, "No tasks found.",
               hint="Install community modules with: [bold]opencrl install[/bold]")
        return 0

    # Determine source for each task
    local_tasks = []
    module_tasks: dict[str, list[str]] = {}

    for name in sorted(tasks):
        # Check if it's a local task
        local_path = Path("tasks") / name / "task.py"
        if local_path.exists():
            local_tasks.append(name)
            continue
        # Check which module it belongs to
        found = False
        if modules.is_cloned():
            for mod_name in active:
                mod_task_path = modules.modules_dir() / mod_name / name / "task.py"
                if mod_task_path.exists():
                    module_tasks.setdefault(mod_name, []).append(name)
                    found = True
                    break
        if not found:
            # Unknown source — put in a misc bucket
            module_tasks.setdefault("other", []).append(name)

    # Apply the module filter to the grouped tasks
    if selected == "local":
        module_tasks = {}
    elif selected is not None:
        local_tasks = []
        module_tasks = {selected: module_tasks.get(selected, [])}

    total = len(local_tasks) + sum(len(names) for names in module_tasks.values())
    if not total:
        _error(console, f"No tasks found in module '{selected}'.")
        return 0

    # Build a tree view
    tree = Tree("[bold cyan]Tasks[/bold cyan]", guide_style="dim")

    if local_tasks:
        local_branch = tree.add("[yellow]local[/yellow]")
        for name in local_tasks:
            t = get_task(name)
            desc = _task_description(t)
            caps = _caps_badge(t)
            local_branch.add(f"[green]{name}[/green] {caps} [dim]— {desc}[/dim]")

    for mod_name in sorted(module_tasks):
        if mod_name == "other":
            label = "[dim]other[/dim]"
        else:
            label = f"[cyan]{mod_name}[/cyan]"
        mod_branch = tree.add(label)
        for name in module_tasks[mod_name]:
            t = get_task(name)
            desc = _task_description(t)
            caps = _caps_badge(t)
            mod_branch.add(f"[green]{name}[/green] {caps} [dim]— {desc}[/dim]")

    console.print(tree)
    console.print(f"\n[dim]{total} task(s) found[/dim]")
    return 0


def _caps_badge(task) -> str:
    """Return a colored capability badge string."""
    badges = []
    if task.caps.offensive:
        badges.append("[red]offensive[/red]")
    if task.caps.needs_internet:
        badges.append("[blue]internet[/blue]")
    if not badges:
        return ""
    return f"[{' | '.join(badges)}]"


def _cmd_info(args) -> int:
    from rich.table import Table
    from rich.panel import Panel

    console = _console()
    discover(args.path)
    if args.name not in list_tasks():
        _error(console, f"Task '{args.name}' not found.",
               hint="Run [bold]opencrl list[/bold] to see available tasks.")
        return 1

    t = get_task(args.name)

    table = Table(show_header=False, box=None, border_style="cyan")
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()
    table.add_row("Name", t.name)
    table.add_row("Goal", t.goal)
    table.add_row("Backend", str(t.backend))
    table.add_row("Tools", ", ".join(tool.name for tool in t.tools) or "—")
    table.add_row("Offensive", "✓" if t.caps.offensive else "—")
    table.add_row("Needs internet", "✓" if t.caps.needs_internet else "—")
    table.add_row("Max steps", str(t.max_steps))
    world = t.world if isinstance(t.world, str) else ("inline dict" if t.world else "—")
    table.add_row("World", world)

    console.print(Panel(table, title=f"[bold cyan]{t.name}[/bold cyan]",
                        border_style="cyan", title_align="left"))
    return 0


def _cmd_new(args) -> int:
    from rich.panel import Panel

    console = _console()

    if not args.name.isidentifier() or keyword.iskeyword(args.name):
        _error(console, f"'{args.name}' is not a valid task name.",
               hint="Must be a Python identifier (letters, digits, underscores), not a keyword.")
        return 1

    dest = Path("tasks") / args.name
    if dest.exists():
        _error(console, f"{dest} already exists.",
               hint="Remove the directory or choose a different name.")
        return 1

    dest.mkdir(parents=True)
    (dest / "task.py").write_text(
        (_TEMPLATE / "task.py").read_text().replace("NAME", args.name))
    (dest / "world.yml").write_text((_TEMPLATE / "world.yml").read_text())

    _success(console, f"Created [cyan]{args.name}[/cyan]",
             hint=f"  [dim]tasks/{args.name}/task.py[/dim]\n  [dim]tasks/{args.name}/world.yml[/dim]\n\n"
                  f"Edit the goal, reward, and world, then run:\n"
                  f"  [bold]python -c \"from opencrl import discover, get_task, rollout; discover(); print(rollout(get_task('{args.name}'), model))\"[/bold]")
    return 0


# ── entrypoint ────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="opencrl",
        description="OpenCyberRL — sandboxed cybersecurity RL task framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Commands:\n"
            "  install              Clone modules repo, list available modules\n"
            "  install <module>     Activate a module\n"
            "  uninstall <module>   Deactivate a module\n"
            "  update               Pull latest modules\n"
            "  list [module]        List tasks (all, or one module's / local)\n"
            "  info <task>          Show task details\n"
            "  new <name>           Scaffold a new task\n"
        ),
    )
    p.add_argument("--path", default=None, help="tasks directory (default: auto-discover)")
    p.add_argument("--version", action="store_true", help="show version and exit")
    sub = p.add_subparsers(dest="cmd", required=True)

    inst = sub.add_parser("install", help="clone/list/activate task modules")
    inst.add_argument("module", nargs="?", default=None,
                      help="module name to activate (omit to list available)")
    inst.set_defaults(fn=_cmd_install)

    un = sub.add_parser("uninstall", help="deactivate a module")
    un.add_argument("module", help="module name to deactivate")
    un.set_defaults(fn=_cmd_uninstall)

    sub.add_parser("update", help="pull latest modules").set_defaults(fn=_cmd_update)
    lst = sub.add_parser("list", help="list discovered tasks")
    lst.add_argument("module", nargs="?", default=None,
                     help="only show this module's tasks ('local' for ./tasks/)")
    lst.set_defaults(fn=_cmd_list)

    info = sub.add_parser("info", help="show task details")
    info.add_argument("name", help="task name")
    info.set_defaults(fn=_cmd_info)

    n = sub.add_parser("new", help="scaffold a new task")
    n.add_argument("name", help="task name (Python identifier)")
    n.set_defaults(fn=_cmd_new)

    # Handle --version before subparser check (required subparsers would
    # otherwise reject a bare --version)
    raw = sys.argv[1:] if argv is None else argv
    if "--version" in raw:
        from importlib.metadata import version
        console = _console()
        console.print(f"[bold cyan]opencrl[/bold cyan] v{version('opencrl')}")
        return 0

    args = p.parse_args(argv)
    return args.fn(args)

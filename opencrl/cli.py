"""opencrl CLI: install | uninstall | update | list | new."""
from __future__ import annotations

import argparse
import keyword
import sys
from pathlib import Path

from opencrl.task import discover, get_task, list_tasks

_TEMPLATE = Path(__file__).parent / "_template"


def _cmd_install(args) -> int:
    from opencrl import modules
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    console = Console()

    if not modules.is_cloned():
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      transient=True, console=console) as progress:
            progress.add_task(description="Cloning opencyberrl-modules...", total=None)
            try:
                modules.clone_modules_repo()
            except Exception as e:
                console.print(f"[red]Error cloning modules repo:[/red] {e}")
                return 1
    else:
        try:
            modules.update_modules_repo()
        except Exception:
            pass  # offline is fine for listing

    available = modules.list_available_modules()
    if not available:
        console.print("[yellow]No modules found in the modules repo.[/yellow]")
        return 0

    if args.module is None:
        table = Table(title="Available modules", show_header=True)
        table.add_column("Module", style="cyan")
        table.add_column("Tasks", justify="right")
        table.add_column("Status", style="green")
        active = modules.active_modules()
        for name in available:
            mod_path = modules.modules_dir() / name
            task_count = sum(1 for _ in mod_path.glob("*/task.py"))
            status = "✓ active" if name in active else ""
            table.add_row(name, str(task_count), status)
        console.print(table)
        console.print("\nInstall with: [bold]opencrl install <module>[/bold]")
        return 0

    if args.module not in available:
        console.print(f"[red]Error:[/red] module '{args.module}' not found.")
        console.print(f"Available: {', '.join(available)}")
        return 1

    modules.activate_module(args.module)
    mod_path = modules.modules_dir() / args.module
    task_count = sum(1 for _ in mod_path.glob("*/task.py"))
    console.print(f"[green]✓[/green] Activated module: [cyan]{args.module}[/cyan] "
                  f"({task_count} tasks)")
    console.print(f"Run with: [bold]opencrl list[/bold] to see available tasks")
    return 0


def _cmd_uninstall(args) -> int:
    from opencrl import modules
    from rich.console import Console
    console = Console()
    modules.deactivate_module(args.module)
    console.print(f"[green]✓[/green] Deactivated module: [cyan]{args.module}[/cyan]")
    return 0


def _cmd_update(args) -> int:
    from opencrl import modules
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    console = Console()

    if not modules.is_cloned():
        console.print("[red]Error:[/red] No modules repo found. Run [bold]opencrl install[/bold] first.")
        return 1

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  transient=True, console=console) as progress:
        progress.add_task(description="Updating modules repo...", total=None)
        try:
            modules.update_modules_repo()
        except Exception as e:
            console.print(f"[red]Error updating:[/red] {e}")
            return 1

    available = modules.list_available_modules()
    console.print(f"[green]✓[/green] Updated. {len(available)} modules available.")
    return 0


def _cmd_list(args) -> int:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    discover(args.path)
    tasks = list_tasks()
    if not tasks:
        console.print("[yellow]No tasks found.[/yellow] "
                      "Install modules with: [bold]opencrl install[/bold]")
        return 0

    table = Table(title="Available tasks", show_header=True)
    table.add_column("Task", style="green")
    table.add_column("Offensive", justify="center")
    table.add_column("Internet", justify="center")
    for name in sorted(tasks):
        t = get_task(name)
        off = "✓" if t.caps.offensive else ""
        net = "✓" if t.caps.needs_internet else ""
        table.add_row(name, off, net)
    console.print(table)
    return 0


def _cmd_new(args) -> int:
    if not args.name.isidentifier() or keyword.iskeyword(args.name):
        print(f"opencrl: error: {args.name!r} is not a valid task name "
              f"(must be a Python identifier, not a keyword)", file=sys.stderr)
        return 1
    dest = Path("tasks") / args.name
    if dest.exists():
        print(f"opencrl: error: {dest} already exists — refusing to overwrite",
              file=sys.stderr)
        return 1
    dest.mkdir(parents=True)
    (dest / "task.py").write_text(
        (_TEMPLATE / "task.py").read_text().replace("NAME", args.name))
    (dest / "world.yml").write_text((_TEMPLATE / "world.yml").read_text())
    print(f"created {dest}/task.py and {dest}/world.yml")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="opencrl")
    p.add_argument("--path", default=None, help="tasks directory (default: auto-discover)")
    sub = p.add_subparsers(dest="cmd", required=True)

    inst = sub.add_parser("install")
    inst.add_argument("module", nargs="?", default=None)
    inst.set_defaults(fn=_cmd_install)

    un = sub.add_parser("uninstall")
    un.add_argument("module")
    un.set_defaults(fn=_cmd_uninstall)

    sub.add_parser("update").set_defaults(fn=_cmd_update)
    sub.add_parser("list").set_defaults(fn=_cmd_list)

    n = sub.add_parser("new")
    n.add_argument("name")
    n.set_defaults(fn=_cmd_new)

    args = p.parse_args(argv)
    return args.fn(args)

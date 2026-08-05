"""cyberl CLI: new | list | run | eval. A thin wrapper over the library."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from cyberl.task import discover, get_task, list_tasks

_TEMPLATE = Path(__file__).parent / "_template"


def _cmd_new(args) -> int:
    dest = Path("tasks") / args.name
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "task.py").write_text(
        (_TEMPLATE / "task.py").read_text().replace("NAME", args.name))
    (dest / "world.yml").write_text((_TEMPLATE / "world.yml").read_text())
    print(f"created {dest}/task.py and {dest}/world.yml")
    return 0


def _cmd_list(args) -> int:
    discover(args.path)
    for name in list_tasks():
        t = get_task(name)
        print(f"{name}\toffensive={t.caps.offensive}\tinternet={t.caps.needs_internet}")
    return 0


def _build_model(args):
    from cyberl.models import OpenAIModel
    return OpenAIModel(model=args.model, base_url=args.base_url,
                       api_key=os.environ.get("OPENAI_API_KEY"))


def _cmd_run(args) -> int:
    from cyberl.rollout import rollout
    discover(args.path)
    r = rollout(get_task(args.name), _build_model(args))
    for m in r.transcript:
        print(f"[{m['role']}] {m.get('content') or m.get('tool_calls')}")
    print(f"reward={r.reward}")
    return 0


def _cmd_eval(args) -> int:
    from cyberl.adapters.eval import evaluate
    discover(args.path)
    stats = evaluate(get_task(args.name), _build_model(args),
                     n=args.n, out=args.out)
    print(f"n={stats['n']} mean_reward={stats['mean_reward']}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="cyberl")
    p.add_argument("--path", default="tasks", help="tasks directory")
    sub = p.add_subparsers(dest="cmd", required=True)

    n = sub.add_parser("new"); n.add_argument("name"); n.set_defaults(fn=_cmd_new)
    sub.add_parser("list").set_defaults(fn=_cmd_list)

    for name in ("run", "eval"):
        sp = sub.add_parser(name)
        sp.add_argument("name")
        sp.add_argument("--model", default="gpt-4o-mini")
        sp.add_argument("--base-url", default=None)
        if name == "eval":
            sp.add_argument("-n", type=int, default=8)
            sp.add_argument("--out", default="rollouts.jsonl")
        sp.set_defaults(fn=_cmd_run if name == "run" else _cmd_eval)

    args = p.parse_args(argv)
    return args.fn(args)

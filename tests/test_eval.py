import json
from cyberl.task import Task, Caps
from cyberl.tools import shell
from cyberl.reward import flag
from cyberl.models import ScriptedModel
from cyberl.backends.mock import MockBackend
from cyberl.adapters.eval import evaluate

def make_task():
    return Task(goal="g", reward=flag("CTF{win}"), tools=(shell,), name="demo",
                caps=Caps(offensive=True), max_steps=3)

def solved():
    return ScriptedModel([{"role": "assistant", "content": "CTF{win}", "tool_calls": None}])

def test_evaluate_writes_jsonl_and_reports(tmp_path):
    out = tmp_path / "log.jsonl"
    # fresh scripted model per rollout via a factory-less single-step model:
    stats = evaluate(make_task(), lambda m, t: {"role": "assistant",
                     "content": "CTF{win}", "tool_calls": None},
                     n=3, out=str(out), backend=MockBackend())
    assert stats["n"] == 3
    assert stats["mean_reward"] == 1.0
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["reward"] == 1.0

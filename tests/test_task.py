import dataclasses
from pathlib import Path
import pytest
from cyberl.task import Caps, Task, task, get_task, list_tasks

def test_task_is_frozen():
    t = Task(goal="g", reward=lambda s: 1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.goal = "x"

def test_caps_defaults():
    c = Caps()
    assert c.offensive is False and c.needs_internet is False

def test_task_decorator_registers_and_fills_name_and_dir():
    @task
    def demo() -> Task:
        return Task(goal="read flag", reward=lambda s: 1.0)

    assert "demo" in list_tasks()
    built = get_task("demo")
    assert built.name == "demo"
    assert Path(built.dir) == Path(__file__).parent

def test_task_decorator_custom_name():
    @task(name="renamed")
    def other() -> Task:
        return Task(goal="g", reward=lambda s: 1.0)
    assert get_task("renamed").name == "renamed"

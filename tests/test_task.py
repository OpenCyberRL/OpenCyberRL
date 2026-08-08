import dataclasses
from pathlib import Path
import pytest
from cyberl.task import Caps, Task, task, get_task, list_tasks, discover

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

def test_discover_registers_module_using_future_annotations_dataclass(tmp_path):
    # A discovered task.py that combines `from __future__ import annotations`
    # with a module-level @dataclass fails unless discover() registers the
    # module in sys.modules before exec_module: dataclasses resolves
    # stringified annotations via sys.modules[cls.__module__], which raises
    # AttributeError on a module that was never registered.
    task_dir = tmp_path / "future_annotations_dataclass_demo"
    task_dir.mkdir()
    (task_dir / "task.py").write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "from cyberl.task import Task, task\n"
        "\n"
        "@dataclass\n"
        "class Config:\n"
        "    value: int = 1\n"
        "\n"
        "@task\n"
        "def future_annotations_dataclass_demo() -> Task:\n"
        "    return Task(goal='g', reward=lambda s: 1.0)\n"
    )

    discover(tmp_path)

    assert "future_annotations_dataclass_demo" in list_tasks()

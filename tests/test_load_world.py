from cyberl.task import Task, load_world

def test_load_world_passthrough_dict():
    t = Task(goal="g", reward=lambda s: 1.0, world={"services": {"box": {}}})
    assert load_world(t) == {"services": {"box": {}}}

def test_load_world_none_is_empty():
    t = Task(goal="g", reward=lambda s: 1.0, world=None)
    assert load_world(t) == {}

def test_load_world_reads_yaml_relative_to_dir(tmp_path):
    (tmp_path / "world.yml").write_text("services:\n  box:\n    image: alpine\n")
    t = Task(goal="g", reward=lambda s: 1.0, world="world.yml", dir=str(tmp_path))
    assert load_world(t) == {"services": {"box": {"image": "alpine"}}}

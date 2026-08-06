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
    # x-cyberl.basedir is injected for file-backed worlds (see Fix B test below);
    # this test is about the YAML content itself.
    assert load_world(t)["services"] == {"box": {"image": "alpine"}}

def test_load_world_resolves_string_build_context(tmp_path):
    (tmp_path / "world.yml").write_text(
        "services:\n  web:\n    build: build/web\n"
    )
    t = Task(goal="g", reward=lambda s: 1.0, world="world.yml", dir=str(tmp_path))
    doc = load_world(t)
    assert doc["services"]["web"]["build"] == str((tmp_path / "build/web").resolve())

def test_load_world_resolves_dict_build_context(tmp_path):
    (tmp_path / "world.yml").write_text(
        "services:\n  web:\n    build:\n      context: build/web\n"
    )
    t = Task(goal="g", reward=lambda s: 1.0, world="world.yml", dir=str(tmp_path))
    doc = load_world(t)
    assert doc["services"]["web"]["build"]["context"] == str(
        (tmp_path / "build/web").resolve()
    )

def test_load_world_leaves_absolute_build_context_unchanged(tmp_path):
    (tmp_path / "world.yml").write_text(
        "services:\n  web:\n    build: /already/abs\n"
    )
    t = Task(goal="g", reward=lambda s: 1.0, world="world.yml", dir=str(tmp_path))
    doc = load_world(t)
    assert doc["services"]["web"]["build"] == "/already/abs"

def test_load_world_leaves_image_only_service_untouched(tmp_path):
    (tmp_path / "world.yml").write_text(
        "services:\n  web:\n    image: alpine:3.20\n"
    )
    t = Task(goal="g", reward=lambda s: 1.0, world="world.yml", dir=str(tmp_path))
    doc = load_world(t)
    assert doc["services"]["web"] == {"image": "alpine:3.20"}
    assert "build" not in doc["services"]["web"]

def test_load_world_injects_basedir_for_file_backed_world(tmp_path):
    (tmp_path / "world.yml").write_text("services:\n  box:\n    image: alpine\n")
    t = Task(goal="g", reward=lambda s: 1.0, world="world.yml", dir=str(tmp_path))
    doc = load_world(t)
    assert doc["x-cyberl"]["basedir"] == str(tmp_path.resolve())

def test_load_world_basedir_anchored_at_nested_world_file_dir(tmp_path):
    # A nested world="sub/world.yml": build contexts and basedir must both
    # resolve relative to sub/, not task.dir (tmp_path) — they need to agree
    # so env_file/bind-mounts and build: share a base.
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "world.yml").write_text(
        "services:\n  web:\n    build: build/web\n"
    )
    t = Task(goal="g", reward=lambda s: 1.0, world="sub/world.yml", dir=str(tmp_path))
    doc = load_world(t)
    assert doc["x-cyberl"]["basedir"] == str(sub.resolve())
    assert doc["services"]["web"]["build"] == str((sub / "build/web").resolve())

def test_load_world_dict_world_has_no_basedir():
    t = Task(goal="g", reward=lambda s: 1.0, world={"services": {"box": {}}})
    doc = load_world(t)
    assert "x-cyberl" not in doc

def test_load_world_none_world_has_no_basedir():
    t = Task(goal="g", reward=lambda s: 1.0, world=None)
    doc = load_world(t)
    assert "x-cyberl" not in doc

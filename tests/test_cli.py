from pathlib import Path
from opencrl.cli import main

def test_new_scaffolds_task(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = main(["new", "myctf"])
    assert rc == 0
    assert (tmp_path / "tasks" / "myctf" / "task.py").exists()
    assert (tmp_path / "tasks" / "myctf" / "world.yml").exists()

def test_list_shows_discovered_task(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["new", "myctf"])
    rc = main(["list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "myctf" in out

def test_new_refuses_to_overwrite_existing_task(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main(["new", "myctf"])
    task_py = (tmp_path / "tasks" / "myctf" / "task.py").read_text()
    world_yml = (tmp_path / "tasks" / "myctf" / "world.yml").read_text()
    rc = main(["new", "myctf"])
    assert rc == 1
    assert (tmp_path / "tasks" / "myctf" / "task.py").read_text() == task_py
    assert (tmp_path / "tasks" / "myctf" / "world.yml").read_text() == world_yml

def test_new_rejects_non_identifier_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = main(["new", "web-sqli"])
    assert rc == 1
    assert not (tmp_path / "tasks" / "web-sqli").exists()
    assert not (tmp_path / "tasks").exists()

def test_run_prints_stage_breakdown(monkeypatch, capsys):
    import sys
    import opencrl.cli as cli
    from opencrl.rollout import Rollout
    from opencrl.task import Caps

    fake = Rollout(task="t", transcript=[{"role": "assistant", "content": "done"}],
                   reward=0.5, caps=Caps(), stages={"root": 1.0, "flag": 0.0})
    monkeypatch.setattr(cli, "discover", lambda path: None)
    monkeypatch.setattr(cli, "get_task", lambda name: object())
    monkeypatch.setattr(cli, "_build_model", lambda args: object())
    rollout_mod = sys.modules["opencrl.rollout"]
    monkeypatch.setattr(rollout_mod, "rollout", lambda task, model: fake)

    rc = cli.main(["run", "whatever"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "reward=0.5" in out
    assert "root: 1.0" in out
    assert "flag: 0.0" in out

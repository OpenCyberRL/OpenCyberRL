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

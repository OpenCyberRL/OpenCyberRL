from pathlib import Path
from cyberl.cli import main

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

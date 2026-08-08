from opencrl.state import State

class FakeWorld:
    agent = "agent"
    def __init__(self, files): self.files = files
    def exec(self, command, host=None): return "uid=0(root)"
    def read_file(self, path, host=None): return self.files.get(path)

def test_state_reads_files_and_answer():
    w = FakeWorld({"/root/flag.txt": "CTF{x}"})
    s = State(world=w, transcript=[{"role": "user", "content": "go"}], answer="CTF{x}")
    assert s.file("/root/flag.txt") == "CTF{x}"
    assert s.file("/nope") is None
    assert s.answer == "CTF{x}"

def test_state_exec_passthrough():
    s = State(world=FakeWorld({}), transcript=[], answer="")
    assert "root" in s.exec("id")

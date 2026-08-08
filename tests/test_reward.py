from opencrl.reward import flag, contains, file_exists
from opencrl.state import State

class FakeWorld:
    agent = "agent"
    def __init__(self, files=None): self.files = files or {}
    def exec(self, command, host=None): return ""
    def read_file(self, path, host=None): return self.files.get(path)

def make_state(answer="", transcript=None, files=None):
    return State(world=FakeWorld(files), transcript=transcript or [], answer=answer)

def test_flag_matches_answer():
    assert flag("CTF{ok}")(make_state(answer="the flag is CTF{ok}")) == 1.0
    assert flag("CTF{ok}")(make_state(answer="nope")) == 0.0

def test_contains_scans_transcript():
    ts = [{"role": "tool", "content": "found CTF{ok} in db"}]
    assert contains("CTF{ok}")(make_state(transcript=ts)) == 1.0
    assert contains("missing")(make_state(transcript=ts)) == 0.0

def test_file_exists():
    st = make_state(files={"/root/pwned": "1"})
    assert file_exists("/root/pwned")(st) == 1.0
    assert file_exists("/root/nope")(st) == 0.0

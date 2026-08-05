from cyberl.tools import Tool, shell

class FakeWorld:
    def exec(self, command, host=None):
        return f"ran: {command}"

def test_shell_runs_on_world():
    assert shell.run(FakeWorld(), command="whoami") == "ran: whoami"

def test_openai_schema_shape():
    s = shell.openai_schema()
    assert s["type"] == "function"
    assert s["function"]["name"] == "shell"
    assert "command" in s["function"]["parameters"]["properties"]

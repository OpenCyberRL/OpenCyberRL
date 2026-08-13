# tests/test_models.py
import pytest
from opencrl.models import ScriptedModel

def test_scripted_model_returns_steps_in_order():
    m = ScriptedModel([
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "1", "type": "function",
                         "function": {"name": "shell", "arguments": '{"command": "id"}'}}]},
        {"role": "assistant", "content": "done", "tool_calls": None},
    ])
    first = m([], [])
    assert first["tool_calls"][0]["function"]["name"] == "shell"
    second = m([], [])
    assert second["content"] == "done"

def test_scripted_model_raises_when_exhausted():
    m = ScriptedModel([{"role": "assistant", "content": "x", "tool_calls": None}])
    m([], [])
    with pytest.raises(StopIteration):
        m([], [])

def test_scripted_model_raises_stopiteration_when_exhausted():
    """Exhausting a ScriptedModel should raise StopIteration with a message,
    not a bare IndexError."""
    model = ScriptedModel([
        {"role": "assistant", "content": "only step", "tool_calls": None},
    ])
    model([], [])  # consume the one step
    with pytest.raises(StopIteration, match="exhausted"):
        model([], [])  # no more steps

def test_openai_model_passes_max_retries_to_client():
    """OpenAIModel must wire max_retries into the OpenAI client constructor."""
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import opencrl.models as models
    original = models.OpenAI
    models.OpenAI = FakeOpenAI
    try:
        models.OpenAIModel("gpt-4o", max_retries=5)
    finally:
        models.OpenAI = original
    assert captured.get("max_retries") == 5

def test_openai_model_defaults_max_retries_to_three():
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import opencrl.models as models
    original = models.OpenAI
    models.OpenAI = FakeOpenAI
    try:
        models.OpenAIModel("gpt-4o")
    finally:
        models.OpenAI = original
    assert captured.get("max_retries") == 3

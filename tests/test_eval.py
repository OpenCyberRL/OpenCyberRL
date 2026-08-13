import json
from opencrl.task import Task, Caps
from opencrl.tools import shell
from opencrl.reward import flag, contains, stage, chain
from opencrl.backends.mock import MockBackend
from opencrl.adapters.eval import evaluate

def make_task():
    return Task(goal="g", reward=flag("CTF{win}"), tools=(shell,), name="demo",
                caps=Caps(offensive=True), max_steps=3)

def test_evaluate_writes_jsonl_and_reports(tmp_path):
    out = tmp_path / "log.jsonl"
    # fresh scripted model per rollout via a factory-less single-step model:
    stats = evaluate(make_task(), lambda m, t: {"role": "assistant",
                     "content": "CTF{win}", "tool_calls": None},
                     n=3, out=str(out), backend=MockBackend())
    assert stats["n"] == 3
    assert stats["mean_reward"] == 1.0
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["reward"] == 1.0


def staged_task():
    # both stages are satisfied by scanning the transcript/answer, so a
    # one-step model (no tool call) works and can be reused across rollouts.
    return Task(goal="g", name="staged", max_steps=3, tools=(shell,),
                reward=chain(stage("saw_root", contains("uid=0")),
                             stage("got_flag", flag("CTF{win}"))))


def test_evaluate_reports_stage_means(tmp_path):
    model = lambda m, t: {"role": "assistant",
                          "content": "ran id -> uid=0; the flag is CTF{win}",
                          "tool_calls": None}
    stats = evaluate(staged_task(), model, n=3, out=str(tmp_path / "l.jsonl"),
                     backend=MockBackend())
    assert stats["mean_reward"] == 1.0
    assert stats["stage_means"] == {"saw_root": 1.0, "got_flag": 1.0}


def test_evaluate_scalar_task_has_empty_stage_means(tmp_path):
    model = lambda m, t: {"role": "assistant", "content": "CTF{win}", "tool_calls": None}
    stats = evaluate(make_task(), model, n=2, out=str(tmp_path / "l.jsonl"),
                     backend=MockBackend())
    assert stats["stage_means"] == {}

def test_evaluate_flushes_each_rollout_before_exception(tmp_path):
    """If the model raises mid-eval, already-completed rollouts must be on disk."""
    out = tmp_path / "log.jsonl"
    call_count = [0]

    def flaky_model(m, t):
        call_count[0] += 1
        if call_count[0] == 2:
            raise RuntimeError("model crashed")
        return {"role": "assistant", "content": "CTF{win}", "tool_calls": None}

    try:
        evaluate(make_task(), flaky_model, n=3, out=str(out), backend=MockBackend())
    except RuntimeError:
        pass  # expected

    # The first rollout must have been flushed before the crash
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["reward"] == 1.0

def test_evaluate_flushes_each_rollout_to_disk_during_eval(tmp_path):
    """Each completed rollout must reach disk immediately, not only at close.

    The 2nd model call inspects the output file: rollout 1 must already be
    on disk at that point (proving per-rollout flush, not just close-flush).
    """
    out = tmp_path / "log.jsonl"
    call_count = [0]
    seen_during_call_2 = [None]

    def inspect_model(m, t):
        call_count[0] += 1
        if call_count[0] == 2 and out.exists():
            seen_during_call_2[0] = out.read_text().strip().splitlines()
        return {"role": "assistant", "content": "CTF{win}", "tool_calls": None}

    evaluate(make_task(), inspect_model, n=2, out=str(out), backend=MockBackend())

    assert seen_during_call_2[0] is not None, "output file did not exist during 2nd rollout"
    assert len(seen_during_call_2[0]) == 1
    assert json.loads(seen_during_call_2[0][0])["reward"] == 1.0

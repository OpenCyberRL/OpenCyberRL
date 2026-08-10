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


import pytest
from opencrl.reward import stage, chain, goals, Score


def test_chain_full_pass_scores_one():
    r = chain(stage("a", lambda s: 1.0), stage("b", lambda s: 1.0))(make_state())
    assert isinstance(r, Score)
    assert r.value == 1.0
    assert r.stages == {"a": 1.0, "b": 1.0}


def test_chain_stops_at_first_gap_and_short_circuits():
    called = []

    def spy(name, val):
        def check(s):
            called.append(name)
            return val
        return check

    r = chain(
        stage("a", spy("a", 1.0)),
        stage("b", spy("b", 0.0)),   # gate closes here
        stage("c", spy("c", 1.0)),   # must NOT be evaluated
    )(make_state())
    assert round(r.value, 6) == round(1 / 3, 6)     # (1/3)*1 + (1/3)*0, c locked
    assert r.stages == {"a": 1.0, "b": 0.0, "c": 0.0}
    assert "c" not in called                        # short-circuited past the gate


def test_chain_partial_credit_on_stalled_tier():
    r = chain(
        stage("a", lambda s: 1.0),
        stage("b", lambda s: 0.5),   # partial, then the gate closes
        stage("c", lambda s: 1.0),
    )(make_state())
    assert round(r.value, 6) == 0.5                  # (1/3)*1 + (1/3)*0.5
    assert r.stages == {"a": 1.0, "b": 0.5, "c": 0.0}


def test_goals_are_independent_and_order_free():
    r = goals(
        stage("x", lambda s: 1.0),
        stage("y", lambda s: 0.0),
        stage("z", lambda s: 1.0),
    )(make_state())
    assert round(r.value, 6) == round(2 / 3, 6)
    assert r.stages == {"x": 1.0, "y": 0.0, "z": 1.0}


def test_equal_weight_default_sums_to_one():
    r = goals(*[stage(f"g{i}", lambda s: 1.0) for i in range(4)])(make_state())
    assert r.value == 1.0


def test_relative_weights_are_normalized():
    r = goals(
        stage("big", lambda s: 1.0, weight=3),
        stage("small", lambda s: 0.0, weight=1),
    )(make_state())
    assert r.value == 0.75


def test_scores_are_clamped_to_unit_interval():
    r = goals(stage("over", lambda s: 1.5), stage("under", lambda s: -0.2))(make_state())
    assert r.stages == {"over": 1.0, "under": 0.0}
    assert r.value == 0.5


def test_invariant_value_equals_weighted_breakdown():
    stgs = (
        stage("a", lambda s: 1.0, weight=1),
        stage("b", lambda s: 0.4, weight=2),
        stage("c", lambda s: 0.0, weight=1),
    )
    r = goals(*stgs)(make_state())
    total = 1 + 2 + 1
    recon = sum((st.weight / total) * r.stages[st.name] for st in stgs)
    assert round(r.value, 9) == round(recon, 9)


def test_empty_stages_raise():
    with pytest.raises(ValueError):
        chain()
    with pytest.raises(ValueError):
        goals()


def test_duplicate_stage_names_raise():
    with pytest.raises(ValueError):
        chain(stage("dup", lambda s: 1.0), stage("dup", lambda s: 1.0))


def test_bad_stage_args_raise():
    with pytest.raises(ValueError):
        stage("w", lambda s: 1.0, weight=0)
    with pytest.raises(ValueError):
        stage("", lambda s: 1.0)
    with pytest.raises(TypeError):
        stage("nc", "not-callable")

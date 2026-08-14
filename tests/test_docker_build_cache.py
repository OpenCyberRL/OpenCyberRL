import opencrl.backends.docker as dk
from opencrl.backends.docker import Docker, _pin_build_images
from opencrl.task import Caps


class FakeRun:
    """Records argv of every docker invocation; returns success."""
    def __init__(self):
        self.calls = []
    def __call__(self, args, timeout=None):
        self.calls.append(args)
        class CP:
            returncode = 0; stdout = ""; stderr = ""
        return CP()


def _spec():
    return {"x-opencrl": {"agent": "a", "basedir": "/abs/task"},
            "services": {"a": {"build": "build/a"}, "b": {"image": "alpine"}}}


def test_pin_build_images_is_stable_and_only_touches_build_services():
    doc = {"services": {"a": {"build": "/abs/task/build/a"}, "b": {"image": "alpine"}}}
    _pin_build_images(doc)
    tag = doc["services"]["a"]["image"]
    assert tag.startswith("opencrl-build-")
    assert doc["services"]["b"]["image"] == "alpine"
    doc2 = {"services": {"a": {"build": "/abs/task/build/a"}}}
    _pin_build_images(doc2)
    assert doc2["services"]["a"]["image"] == tag


def test_prebuild_runs_build_and_populates_cache(monkeypatch):
    fake = FakeRun(); monkeypatch.setattr(dk, "_run", fake)
    be = Docker()
    be.prebuild(_spec(), Caps())
    assert any("build" in c for c in fake.calls)
    assert be._built


def test_up_skips_build_when_cached(monkeypatch):
    fake = FakeRun(); monkeypatch.setattr(dk, "_run", fake)
    be = Docker()
    be.prebuild(_spec(), Caps())
    fake.calls.clear()
    be.up(_spec(), Caps())
    up_calls = [c for c in fake.calls if "up" in c]
    assert up_calls and all("--build" not in c for c in up_calls)


def test_up_builds_when_not_cached(monkeypatch):
    fake = FakeRun(); monkeypatch.setattr(dk, "_run", fake)
    be = Docker()
    be.up(_spec(), Caps())
    up_calls = [c for c in fake.calls if "up" in c]
    assert any("--build" in c for c in up_calls)


def test_pin_build_images_distinguishes_by_target():
    a = {"services": {"x": {"build": {"context": "/c", "target": "prod"}}}}
    b = {"services": {"x": {"build": {"context": "/c", "target": "dev"}}}}
    _pin_build_images(a)
    _pin_build_images(b)
    assert a["services"]["x"]["image"] != b["services"]["x"]["image"]


def test_docker_is_picklable():
    import pickle
    be = Docker()
    be._built.add("opencrl-build-abc")
    be2 = pickle.loads(pickle.dumps(be))
    with be2._built_lock:            # recreated lock is usable
        pass
    assert be2._built == {"opencrl-build-abc"}

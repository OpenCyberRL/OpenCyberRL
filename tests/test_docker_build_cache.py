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
    be._built["shared:latest"] = "abc"
    be2 = pickle.loads(pickle.dumps(be))
    with be2._built_lock:            # recreated lock is usable
        pass
    assert be2._built == {"shared:latest": "abc"}


def test_pin_build_images_distinguishes_by_platform():
    a = {"services": {"x": {"build": "/c", "platform": "linux/amd64"}}}
    b = {"services": {"x": {"build": "/c", "platform": "linux/arm64"}}}
    _pin_build_images(a)
    _pin_build_images(b)
    assert a["services"]["x"]["image"] != b["services"]["x"]["image"]


def test_build_cache_keyed_by_config_not_explicit_image(monkeypatch):
    fake = FakeRun(); monkeypatch.setattr(dk, "_run", fake)
    be = Docker()
    s1 = {"services": {"a": {"build": {"context": "/c", "target": "one"},
                             "image": "shared:latest"}}}
    s2 = {"services": {"a": {"build": {"context": "/c", "target": "two"},
                             "image": "shared:latest"}}}
    be.up(s1, Caps())
    be.up(s2, Caps())
    ups = [c for c in fake.calls if "up" in c]
    # same explicit image but different build config -> distinct identities -> both build
    assert len(ups) == 2 and all("--build" in c for c in ups)


def test_build_cache_invalidates_shared_explicit_tag(monkeypatch):
    fake = FakeRun(); monkeypatch.setattr(dk, "_run", fake)
    be = Docker()
    A = {"services": {"a": {"build": {"context": "/c", "target": "A"},
                            "image": "shared:latest"}}}
    B = {"services": {"a": {"build": {"context": "/c", "target": "B"},
                            "image": "shared:latest"}}}
    be.up(A, Caps())
    be.up(B, Caps())          # overwrites shared:latest with B's build
    fake.calls.clear()
    be.up(A, Caps())          # tag now owned by B -> A must rebuild, not reuse B
    ups = [c for c in fake.calls if "up" in c]
    assert ups and all("--build" in c for c in ups)

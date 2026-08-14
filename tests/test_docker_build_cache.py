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
    be._built.add("abc")
    be2 = pickle.loads(pickle.dumps(be))
    with be2._built_lock:            # recreated lock is usable
        pass
    assert be2._built == {"abc"}


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
    # explicit image overridden with unique content tags -> different configs both build
    assert len(ups) == 2 and all("--build" in c for c in ups)


def test_build_cache_reuses_unique_tag_across_specs(monkeypatch):
    fake = FakeRun(); monkeypatch.setattr(dk, "_run", fake)
    be = Docker()
    A = {"services": {"a": {"build": {"context": "/c", "target": "A"}}}}
    B = {"services": {"a": {"build": {"context": "/c", "target": "B"}}}}
    be.up(A, Caps())
    be.up(B, Caps())          # different config -> its own unique tag, doesn't touch A's
    fake.calls.clear()
    be.up(A, Caps())          # A's unique content tag is intact -> reuse, no rebuild
    ups = [c for c in fake.calls if "up" in c]
    assert ups and all("--build" not in c for c in ups)


def test_pin_build_images_includes_basedir_for_omitted_context():
    a = {"services": {"x": {"build": {"dockerfile": "Dockerfile"}}}}
    b = {"services": {"x": {"build": {"dockerfile": "Dockerfile"}}}}
    _pin_build_images(a, "/task/a")
    _pin_build_images(b, "/task/b")
    # omitted context resolves to the project dir -> different dirs must not alias
    assert a["services"]["x"]["image"] != b["services"]["x"]["image"]


def test_pin_build_images_treats_empty_mapping_as_build():
    d = {"services": {"x": {"build": {}}}}
    ids = _pin_build_images(d, "/task")
    assert ids and d["services"]["x"]["image"].startswith("opencrl-build-")


def test_pin_build_images_rewrites_shared_image_consumers():
    # Service 'builder' builds into image: shared:latest; service 'consumer'
    # references the same tag. The override must rewrite the consumer too,
    # or it would pull a stale/missing image instead of the built one.
    doc = {"services": {
        "builder": {"build": "/ctx", "image": "shared:latest"},
        "consumer": {"image": "shared:latest"},
    }}
    _pin_build_images(doc)
    builder_img = doc["services"]["builder"]["image"]
    consumer_img = doc["services"]["consumer"]["image"]
    assert builder_img.startswith("opencrl-build-")
    assert consumer_img == builder_img   # rewritten to match the built tag


def test_pin_build_images_resolves_interpolation_in_digest(monkeypatch):
    monkeypatch.setenv("VERSION", "1.0")
    a = {"services": {"x": {"build": {"context": "/c", "args": {"V": "${VERSION}"}}}}}
    _pin_build_images(a, "/task")
    monkeypatch.setenv("VERSION", "2.0")
    b = {"services": {"x": {"build": {"context": "/c", "args": {"V": "${VERSION}"}}}}}
    _pin_build_images(b, "/task")
    assert a["services"]["x"]["image"] != b["services"]["x"]["image"]


def test_pin_build_images_resolves_default_interpolation(monkeypatch):
    monkeypatch.delenv("MISSING_VAR", raising=False)
    a = {"services": {"x": {"build": {"context": "/c", "args": {"V": "${MISSING_VAR:-fallback}"}}}}}
    _pin_build_images(a, "/task")
    assert a["services"]["x"]["image"].startswith("opencrl-build-")

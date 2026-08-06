import pytest
from cyberl.task import Caps
from cyberl.backends.docker import Docker

def _render(spec, caps=Caps()):
    return Docker()._render(spec, caps)

def test_render_default_network_when_none_declared():
    doc, agent = _render({"x-cyberl": {"agent": "box"},
                          "services": {"box": {"image": "alpine"}}})
    assert agent == "box"
    assert doc["networks"] == {"default": {"internal": True}}
    assert doc["services"]["box"]["networks"] == ["default"]

def test_render_honors_declared_networks():
    spec = {"x-cyberl": {"agent": "a"},
            "networks": {"edge": None, "backend": None},
            "services": {
                "a": {"image": "x", "networks": ["edge"]},
                "w": {"image": "x", "networks": ["edge", "backend"]},
                "i": {"image": "x", "networks": ["backend"]},
            }}
    doc, agent = _render(spec)
    assert set(doc["networks"]) == {"edge", "backend"}     # no default injected
    assert doc["networks"]["edge"] == {"internal": True}
    assert doc["networks"]["backend"] == {"internal": True}
    assert doc["services"]["a"]["networks"] == ["edge"]     # kept as declared
    assert doc["services"]["i"]["networks"] == ["backend"]

def test_render_egress_when_needs_internet():
    doc, _ = _render({"services": {"box": {"image": "x"}}}, Caps(needs_internet=True))
    assert doc["networks"]["default"] == {"internal": False}

def test_render_mixed_declared_and_default():
    spec = {"networks": {"edge": None},
            "services": {"a": {"image": "x", "networks": ["edge"]},
                         "b": {"image": "x"}}}   # b declares none -> default net
    doc, _ = _render(spec)
    assert "default" in doc["networks"]
    assert doc["services"]["b"]["networks"] == ["default"]
    assert doc["services"]["a"]["networks"] == ["edge"]

def test_render_locks_down_undeclared_network():
    # A service names a network ("default") that is absent from the top-level
    # `networks:` block. Compose would otherwise create it as an externally
    # routable network regardless of Caps(needs_internet=False) — the implicit
    # egress hole Fix A closes.
    spec = {"services": {"box": {"image": "x", "networks": ["default"]}}}
    doc, _ = _render(spec)
    assert doc["networks"]["default"] == {"internal": True}

def test_render_rejects_external_network_when_no_internet():
    spec = {"networks": {"ext": {"external": True}},
            "services": {"box": {"image": "x", "networks": ["ext"]}}}
    with pytest.raises(ValueError, match="external network"):
        _render(spec, Caps(needs_internet=False))

def test_render_allows_external_network_when_internet_needed():
    spec = {"networks": {"ext": {"external": True}},
            "services": {"box": {"image": "x", "networks": ["ext"]}}}
    doc, _ = _render(spec, Caps(needs_internet=True))   # must not raise
    assert doc["services"]["box"]["networks"] == ["ext"]

def test_render_rejects_top_level_include():
    spec = {"include": ["../other/compose.yml"],
            "services": {"box": {"image": "x"}}}
    with pytest.raises(ValueError, match="include"):
        _render(spec)

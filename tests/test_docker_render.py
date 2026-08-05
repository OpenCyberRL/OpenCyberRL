from cyberl.task import Caps
from cyberl.backends.docker import Docker

def _render(spec, caps=Caps()):
    return Docker()._render(spec, caps)

def test_render_default_network_when_none_declared():
    doc, agent = _render({"x-cyberl": {"agent": "box"},
                          "services": {"box": {"image": "alpine"}}})
    assert agent == "box"
    assert doc["networks"] == {"cyberl_net": {"internal": True}}
    assert doc["services"]["box"]["networks"] == ["cyberl_net"]

def test_render_honors_declared_networks():
    spec = {"x-cyberl": {"agent": "a"},
            "networks": {"edge": None, "backend": None},
            "services": {
                "a": {"image": "x", "networks": ["edge"]},
                "w": {"image": "x", "networks": ["edge", "backend"]},
                "i": {"image": "x", "networks": ["backend"]},
            }}
    doc, agent = _render(spec)
    assert set(doc["networks"]) == {"edge", "backend"}     # no cyberl_net injected
    assert doc["networks"]["edge"] == {"internal": True}
    assert doc["networks"]["backend"] == {"internal": True}
    assert doc["services"]["a"]["networks"] == ["edge"]     # kept as declared
    assert doc["services"]["i"]["networks"] == ["backend"]

def test_render_egress_when_needs_internet():
    doc, _ = _render({"services": {"box": {"image": "x"}}}, Caps(needs_internet=True))
    assert doc["networks"]["cyberl_net"] == {"internal": False}

def test_render_mixed_declared_and_default():
    spec = {"networks": {"edge": None},
            "services": {"a": {"image": "x", "networks": ["edge"]},
                         "b": {"image": "x"}}}   # b declares none -> default net
    doc, _ = _render(spec)
    assert "cyberl_net" in doc["networks"]
    assert doc["services"]["b"]["networks"] == ["cyberl_net"]
    assert doc["services"]["a"]["networks"] == ["edge"]

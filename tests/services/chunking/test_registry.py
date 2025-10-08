from Medical_KG_rev.services.chunking import registry


def test_register_defaults_handles_optional_dependencies(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(registry.simple, "register", lambda: calls.append("simple"))
    monkeypatch.setattr(registry.langchain_splitter, "register", lambda: (_ for _ in ()).throw(RuntimeError("missing")))
    monkeypatch.setattr(registry.llamaindex_parser, "register", lambda: (_ for _ in ()).throw(RuntimeError("missing")))
    monkeypatch.setattr(registry.profile_chunkers, "register", lambda: calls.append("profiles"))

    registry.register_defaults()

    assert calls == ["simple", "profiles"]

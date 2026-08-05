def test_public_surface_imports():
    import cyberl
    for name in [
        "task", "Task", "Caps", "Tool", "shell",
        "flag", "contains", "file_exists",
        "rollout", "Rollout", "State", "evaluate", "to_gym",
        "Docker", "MockBackend", "ScriptedModel", "OpenAIModel",
        "register_backend", "resolve_backend",
        "get_task", "list_tasks", "discover",
    ]:
        assert hasattr(cyberl, name), f"missing public export: {name}"

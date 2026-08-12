def test_public_surface_imports():
    import opencrl
    for name in [
        "task", "Task", "Caps", "Tool", "shell",
        "flag", "contains", "file_exists",
        "stage", "chain", "goals", "Score",
        "rollout", "Rollout", "State", "evaluate", "to_gym",
        "Docker", "Qemu", "MockBackend", "ScriptedModel", "OpenAIModel",
        "register_backend", "resolve_backend",
        "get_task", "list_tasks", "discover",
    ]:
        assert hasattr(opencrl, name), f"missing public export: {name}"

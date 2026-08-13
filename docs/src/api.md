# Python API

`import opencrl` exposes the whole public surface. This page lists it by group.

## Tasks

- `@task`: register a task factory. It fills in `name` and `dir`.
- `Task(goal, reward, world=None, backend="docker", tools=(), caps=Caps(), max_steps=30)`: the task contract, a frozen dataclass. See [Write a task](guide-task.md).
- `Caps(offensive=False, needs_internet=False)`: capability metadata.
- `Tool(name, description, schema, run)`: an action the agent can call.
- `shell`: the built-in shell tool.
- `get_task(name)`: return a registered task.
- `list_tasks()`: list registered task names.
- `discover(path="tasks")`: import every `tasks/*/task.py` so the tasks register.

## Rewards

- `flag(expected)`: 1.0 when the final answer contains `expected`.
- `contains(text)`: 1.0 when `text` appears in the transcript.
- `file_exists(path, host=None)`: 1.0 when the file exists.
- `stage(name, check, weight=1.0)`: one named sub-task.
- `chain(*stages)`: an ordered, gated staged reward.
- `goals(*stages)`: an independent staged reward.
- `Score(value, stages)`: the result of a staged reward.

See [Rewards and scoring](guide-rewards.md).

## Rollouts

- `rollout(task, model, backend=None)`: run one episode; return a `Rollout`.
- `Rollout(task, transcript, reward, caps, stages=None)`: the standard artifact. `to_dict()` returns a JSON-serializable form.
- `State`: the read-only view a verifier scores, with `exec()`, `file()`, `answer`, and `transcript`.

## Models

- `OpenAIModel(model, base_url=None, api_key=None)`: an OpenAI-compatible client. Needs the `openai` extra.
- `ScriptedModel(steps)`: replay a fixed message list, for tests.

## Adapters

- `evaluate(task, model, n=1, out=None, backend=None)`: run `n` rollouts; return summary statistics; write JSONL when `out` is set.
- `to_trl(task, backend=None)`: return `(environment_factory, reward_func)` for TRL GRPO training. Needs the `trl` extra.
- `to_verl(task, backend=None, out_path=".", decode_callback=None)`: return `(parquet_path, reward_func)` for Verl training. Needs the `verl` extra.
- `export_rollouts(task, model, n=16, backend=None, fmt="dpo")`: run `n` rollouts and export as a HuggingFace `Dataset` in DPO, SFT, or KTO format. Needs `datasets` installed.

## Backends

- `Docker(...)`, `Qemu(...)`, `MockBackend(...)`: the built-in backends.
- `register_backend(name, factory)`: register a backend factory under a name.
- `resolve_backend(name_or_instance)`: return a backend from a name or an instance.

See [Backends](backends.md).

# Quickstart

This guide runs one agent against one task. It needs Docker running, the
`openai` extra, and an API key:

```bash
uv sync --extra openai
export OPENAI_API_KEY=sk-...
```

## Write a task

A task can inline its world as a dictionary, so it needs no separate files.
Save this as `hello.py`:

```python
import opencrl
from opencrl import task, Task, shell, flag, Caps, rollout


@task
def hello() -> Task:
    return Task(
        goal="Read /flag and state it.",
        tools=[shell],
        reward=flag("CTF{hi}"),
        backend="docker",
        world={
            "x-opencrl": {"agent": "box"},
            "services": {
                "box": {
                    "image": "alpine:3.20",
                    "command": "sh -c 'echo CTF{hi}>/flag; sleep 600'",
                }
            },
        },
        caps=Caps(),
    )


r = rollout(hello(), opencrl.OpenAIModel("gpt-4o-mini"))
print(r.reward, r.transcript[-1])
```

Run it:

```bash
uv run python hello.py
```

`rollout()` starts the world, drives the model through tool calls until it
gives a final answer, scores that answer, and stops the world. The score is
`1.0` when the answer contains `CTF{hi}`.

## Scaffold a task from the CLI

The CLI writes a task file and a world file for you:

```bash
uv run opencrl new mytask
```

This creates `tasks/mytask/task.py` and `tasks/mytask/world.yml`. Edit the
goal, the reward, and the world, then run it:

```bash
uv run opencrl run mytask --model gpt-4o-mini
```

## Next steps

- [Write a task](guide-task.md): every field, explained.
- [Rewards and scoring](guide-rewards.md): from a single flag to staged rewards.
- [Run and evaluate](guide-run-eval.md): batch runs and JSONL output.

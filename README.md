<p align="center"><img src="logo.png" alt="OpenCyberRL" width="360"></p>

# OpenCyberRL

A plug-and-play framework for authoring sandboxed cybersecurity RL tasks —
for training and evaluating LLM tool-calling agents. Task authors bring the
environment (a Docker Compose world) and the verifier (a plain function); the
framework runs the agent loop and emits one standard `Rollout` artifact you
can score, log, or feed into RL. Three reference tasks ship under `tasks/` as
worked examples.

## Install

```bash
uv sync
```

For the OpenAI-compatible model client and the Gymnasium adapter:

```bash
uv sync --extra openai --extra gym
```

## 30-second quickstart

A task is a goal, a set of tools, a reward function, and a world. This one
inlines the world as a plain dict, so it needs no separate files:

```python
import cyberl
from cyberl import task, Task, shell, flag, Caps, rollout

@task
def hello() -> Task:
    return Task(
        goal="Read /flag and state it.",
        tools=[shell],
        reward=flag("CTF{hi}"),
        backend="docker",
        world={
            "x-cyberl": {"agent": "box"},
            "services": {
                "box": {
                    "image": "alpine:3.20",
                    "command": "sh -c 'echo CTF{hi}>/flag; sleep 600'",
                }
            },
        },
        caps=Caps(),
    )

r = rollout(hello(), cyberl.OpenAIModel("gpt-4o-mini"))
print(r.reward, r.transcript[-1])
```

Running this for real needs the `openai` extra and an `OPENAI_API_KEY` in
the environment; the snippet above is illustrative of the shape.

`rollout()` stands up the world, drives the model through tool calls until it
gives a final answer, scores that answer with `reward`, and tears the world
down — win or lose.

## CLI

```bash
cyberl new <name>                          # scaffold tasks/<name>/{task.py,world.yml}
cyberl list                                 # list discovered tasks and their caps
cyberl run <name> --model gpt-4o-mini       # run one rollout, print the transcript + reward
cyberl eval <name> -n 16 --out rollouts.jsonl   # run N rollouts, write Rollout JSONL, report mean reward
```

## The `Rollout` artifact

Every rollout — from the CLI, `rollout()`, or `evaluate()` — produces the
same JSON-serializable shape (`Rollout.to_dict()`):

```json
{
  "task": "web_sqli",
  "transcript": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "goal..."},
    {"role": "assistant", "content": null, "tool_calls": [{"id": "1", "type": "function", "function": {"name": "shell", "arguments": "{\"command\": \"...\"}"}}]},
    {"role": "tool", "tool_call_id": "1", "content": "command output"},
    {"role": "assistant", "content": "final answer", "tool_calls": null}
  ],
  "reward": 1.0,
  "caps": {"offensive": true, "needs_internet": false}
}
```

`cyberl eval` writes one of these per line to a JSONL file, ready for
downstream scoring pipelines or RL training.

## Safety

Worlds are sandboxed by default: every Docker network a task uses is brought
up with `internal: true`, so containers get no external egress unless the
task sets `caps=Caps(needs_internet=True)`. Every image is built from source
under the task's `build/` directory rather than pulled from a registry.
`caps` (`offensive`, `needs_internet`) travels with every rollout, so
downstream consumers can filter or audit what ran. Tasks can also declare
their own `networks:` in `world.yml` to segment hosts from each other — the
`lateral` reference task puts its attacker on one subnet and its internal
target on another, reachable only through a web host that bridges both.

## Reference tasks

`tasks/web_sqli`, `tasks/privesc`, and `tasks/lateral` are complete,
passing examples — see [CONTRIBUTING.md](CONTRIBUTING.md) to write your own.

## License

MIT — see [LICENSE](LICENSE).

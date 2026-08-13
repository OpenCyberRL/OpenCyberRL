# Run and evaluate

## One rollout

`rollout()` runs one episode:

```python
from opencrl import rollout, OpenAIModel

r = rollout(task, OpenAIModel("gpt-4o-mini"))
```

It returns a `Rollout`:

| Field | Type | Meaning |
|---|---|---|
| `task` | `str` | The task name. |
| `transcript` | `list[dict]` | Every message, in order. |
| `reward` | `float` | The aggregate score. |
| `caps` | `Caps` | The task's capabilities. |
| `stages` | `dict[str, float] \| None` | The per-stage breakdown, or `None` for a scalar reward. |

`Rollout.to_dict()` returns a JSON-serializable form of the same data.

## The Rollout artifact

Every run, from `rollout()`, `evaluate()`, or the CLI, produces the same shape:

```json
{
  "task": "read_flag",
  "transcript": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "goal..."},
    {"role": "assistant", "content": null, "tool_calls": [ ... ]},
    {"role": "tool", "tool_call_id": "1", "content": "command output"},
    {"role": "assistant", "content": "final answer", "tool_calls": null}
  ],
  "reward": 1.0,
  "caps": {"offensive": true, "needs_internet": false},
  "stages": null
}
```

## Evaluate over many runs

`evaluate()` runs a task `n` times and reports the results. Pass `out` to write
one `Rollout` per line to a JSONL file:

```python
from opencrl import evaluate, OpenAIModel

stats = evaluate(task, OpenAIModel("gpt-4o-mini"), n=16, out="rollouts.jsonl")
print(stats["mean_reward"])   # the mean over 16 runs
print(stats["stage_means"])   # the mean of each stage, for a staged reward
```

`evaluate()` returns a dictionary:

| Key | Meaning |
|---|---|
| `n` | The number of runs. |
| `mean_reward` | The mean aggregate reward. |
| `rewards` | The reward from each run. |
| `stage_means` | The mean of each stage. Empty for a scalar reward. |

`stage_means` shows which sub-task agents reach and where they stop.

## Models

A model is any callable that maps messages and tools to an assistant message.
Two ship with the framework:

- **`OpenAIModel(model, base_url=None, api_key=None)`**: an OpenAI-compatible
  client. It needs the `openai` extra. It reads the API key from the argument or
  from `OPENAI_API_KEY`. Set `base_url` to reach a compatible server.
- **`ScriptedModel(steps)`**: replays a fixed list of assistant messages. Use it
  in tests to play out a known solution.

```python
from opencrl import ScriptedModel

model = ScriptedModel([
    {"role": "assistant", "content": "The flag is CTF{win}", "tool_calls": None},
])
```

## Export rollouts for offline training

`export_rollouts()` runs `n` rollouts and packages them as a HuggingFace
`Dataset` for offline preference fine-tuning. It supports three formats:

```python
from opencrl import export_rollouts, OpenAIModel

# DPO preference pairs: highest-reward vs lowest-reward
ds = export_rollouts(task, OpenAIModel("gpt-4o-mini"), n=16, fmt="dpo")

# SFT: only successful trajectories (reward >= 1.0)
ds = export_rollouts(task, OpenAIModel("gpt-4o-mini"), n=16, fmt="sft")

# KTO: unpaired, each rollout labeled good/bad
ds = export_rollouts(task, OpenAIModel("gpt-4o-mini"), n=16, fmt="kto")
```

For on-policy training with TRL or Verl, see
[RL training adapters](guide-rl-adapters.md).

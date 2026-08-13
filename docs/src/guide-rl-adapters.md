# RL training adapters

OpenCyberRL ships three adapters that connect tasks to reinforcement-learning
training frameworks. Each one takes a `Task` and returns the pieces the
framework needs: an environment, a reward function, or a dataset.

| Adapter | Framework | Mode | Extra |
|---|---|---|---|
| `to_trl` | [TRL](https://github.com/huggingface/trl) | On-policy (GRPO) | `trl` |
| `to_verl` | [Verl](https://github.com/volcengine/verl) | On-policy | `verl` |
| `export_rollouts` | — | Offline (DPO/SFT/KTO) | `datasets` |

## TRL: on-policy GRPO

`to_trl()` returns `(environment_factory, reward_func)` for TRL's
`GRPOTrainer`. The environment manages the sandbox lifecycle:
`backend.up()` on `reset()`, `backend.down()` on `_close()`.

```python
from opencrl import to_trl, task, Task, shell, flag, Caps

@task
def read_flag() -> Task:
    return Task(
        goal="Read /flag and state it.",
        tools=[shell],
        reward=flag("CTF{win}"),
        world={"x-opencrl": {"agent": "box"},
               "services": {"box": {"image": "alpine:3.20",
                                    "command": "sh -c 'echo CTF{win}>/flag; sleep 600'"}}},
        caps=Caps(offensive=True),
    )

env_cls, reward_func = to_trl(read_flag(), backend="docker")
```

### The environment

`env_cls` is a factory you pass to `GRPOTrainer`. Each instance:

- `reset(**kwargs)` — stands up a fresh world, starts an episode, returns the
  task goal as the prompt. Accepts arbitrary keyword arguments from TRL's
  `reset_kwargs`.
- `shell(command)` — runs a shell command in the world, returns its output.
  Exposed to the model as a tool.
- `read_file(path)` — reads a file from the world, returns its contents (or
  empty string if absent). Exposed to the model as a tool.
- `get_reward()` — scores the completed episode with full `State` (world +
  transcript + answer). Use this for world-dependent rewards (`file_exists`,
  custom verifiers that call `state.exec`).
- `_close()` — tears down the sandbox world. Prefixed with `_` so TRL doesn't
  expose it to the model as a callable tool.

### The reward function

`reward_func(prompts, completions, completion_ids=None, **kwargs)` is TRL's
batch reward function. It builds a transcript from each prompt/completion pair
and scores it. This path uses transcript-only rewards (`flag`, `contains`) —
the world is not available. For world-dependent rewards, use the environment's
`get_reward()` method instead.

When the model generates conversational (list-of-messages) prompts, the reward
function uses them directly. When prompts are strings, it wraps them with the
system prompt and user goal.

## Verl: on-policy training

`to_verl()` returns `(parquet_path, reward_func)` for Verl's training pipeline.
It writes a single-row Parquet dataset and builds a reward function matching
Verl's `compute_score` API.

```python
from opencrl import to_verl

parquet_path, reward_func = to_verl(read_flag(), backend="docker",
                                    out_path="data/")
```

### The Parquet dataset

The Parquet file contains one row with:

- `data_source`: the task name.
- `prompt`: the system and user messages (as a list of message dicts).
- `ability`: `"cyber"`.
- `reward_model`: `{"style": "rule", "ground_truth": "..."}`.
- `extra_info`: `{"split": "train", "index": 0}`.
- `agent_name`: `"tool_agent_loop"` (only when the task has tools).

### The reward function

The reward function matches Verl's `compute_score` signature:

```python
fn(data_source, solution_str, ground_truth, extra_info=None) -> float
```

It builds a transcript from the solution string and scores it with the task's
reward function. Only transcript-only rewards (`flag`, `contains`) work here —
Verl's API does not pass the running world. For world-dependent rewards, write
a custom reward function that accesses the world through Verl's
`tool_agent_loop`.

### Ground truth extraction

When the task uses `flag(expected)`, the expected flag string is extracted as
the `ground_truth` field in the Parquet. For `file_exists`, `contains`, `chain`,
`goals`, and custom reward functions, `ground_truth` is empty — the reward
function does all the scoring.

### Decoding token IDs

Pass a `decode_callback` to integrate with a tokenizer:

```python
def decode(response_ids, response_mask):
    return tokenizer.decode(response_ids)

_, reward_func = to_verl(task, decode_callback=decode)
```

The callback is used when the caller wraps the reward function and has access
to raw token IDs. In the standard `compute_score` API, `solution_str` is
already decoded text.

## Offline export: DPO, SFT, KTO

`export_rollouts()` runs `n` rollouts and converts them to a HuggingFace
`Dataset` in one of three formats:

```python
from opencrl import export_rollouts, OpenAIModel

ds = export_rollouts(read_flag(), OpenAIModel("gpt-4o-mini"),
                     n=16, fmt="dpo")
```

| Format | Columns | Pairing |
|---|---|---|
| `dpo` | `prompt`, `chosen`, `rejected` | Highest-reward vs lowest-reward (max contrast). Equal-reward pairs are skipped. Completions contain only assistant messages. |
| `sft` | `prompt`, `completion` | Only rollouts with reward >= 1.0. Completion includes assistant and tool messages (full trajectory). |
| `kto` | `prompt`, `completion`, `label` | Every rollout. `label` is `True` when reward >= 0.5. Completion includes assistant and tool messages. |

DPO pairs maximize preference contrast: the highest-reward rollout is paired
with the lowest, the second-highest with the second-lowest, and so on. Pairs
where both rollouts have equal reward are dropped (no preference signal). DPO
completions contain only assistant messages — tool results are
environment-generated, not model-generated, so they don't belong in a
preference pair.

SFT and KTO completions include the full trajectory (assistant + tool
messages), since the model learns from the complete interaction.

## Staged rewards through adapters

Staged rewards (`chain`, `goals`) work through all three adapters. The TRL
environment's `get_reward()` and the Verl reward function both return the
aggregate float. Stage breakdowns are available through the TRL reward
function's `log_extra` callback when present.

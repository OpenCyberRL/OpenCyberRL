# Rewards and scoring

A reward function scores a finished episode. It takes a `State` and returns a
float from 0.0 to 1.0. The framework scores the episode once, at the end.

## Built-in verifiers

Three helpers cover common checks:

```python
from opencrl import flag, contains, file_exists

flag("CTF{win}")            # 1.0 if the agent's final answer contains the flag
contains("uid=0")           # 1.0 if the text appears anywhere in the transcript
file_exists("/root/loot")   # 1.0 if the file exists in the world
```

Each helper returns a function. Pass that function as the task's `reward`:

```python
reward=flag("CTF{win}")
```

## A custom verifier

A verifier is any function from `State` to a float. `State` gives read-only
access to the world and the transcript:

```python
def rooted(state) -> float:
    return 1.0 if "uid=0" in state.exec("id") else 0.0

reward=rooted
```

`State` provides:

- `state.exec(command, host=None)`: run a command in the world.
- `state.file(path, host=None)`: read a file, or `None` if it is absent.
- `state.answer`: the agent's final answer.
- `state.transcript`: the full message list.

## Staged rewards

A single flag gives a pass-or-fail score. A staged reward gives partial credit
for the sub-tasks the agent completes. Build one with `stage` and either `chain`
or `goals`.

A `stage` is a name, a check, and an optional weight.

### `chain`: an ordered kill-chain

`chain` scores stages in order. Credit stops at the first stage the agent does
not fully complete. Use it when each step depends on the one before it:

```python
from opencrl import chain, stage, flag, contains, file_exists

reward = chain(
    stage("recon",    contains("uid=")),
    stage("foothold", file_exists("/tmp/foothold")),
    stage("root",     contains("uid=0")),
    stage("flag",     flag("CTF{win}")),
)
```

Say the agent reaches `foothold` but fails `root`. The reward is 0.5. `recon`
and `foothold` score in full, and `flag` stays out of reach.

### `goals`: independent sub-tasks

`goals` scores every stage on its own. The agent earns each stage's share in any
order:

```python
from opencrl import goals, stage, flag, file_exists

reward = goals(
    stage("read_config", file_exists("/tmp/loot")),
    stage("get_flag",    flag("CTF{win}")),
)
```

### Weights

Stages share the reward equally by default. Set a weight to change a stage's
share. Weights are relative, and the framework normalizes them:

```python
reward = chain(
    stage("recon", contains("uid=")),
    stage("flag",  flag("CTF{win}"), weight=3),   # worth three times a recon
)
```

### The breakdown

A staged reward returns a `Score`: the aggregate value and the per-stage
breakdown. The rollout keeps both:

```python
r = rollout(task, model)
print(r.reward)     # 0.5
print(r.stages)     # {"recon": 1.0, "foothold": 1.0, "root": 0.0, "flag": 0.0}
```

`r.reward` stays a float, so every consumer that reads a scalar reward keeps
working. A task that returns a plain float leaves `r.stages` as `None`.

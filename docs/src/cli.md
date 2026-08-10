# CLI

The `opencrl` command wraps the library. Run it with `uv run opencrl`.

The `run` and `eval` commands build an `OpenAIModel`, so they need the `openai`
extra and an `OPENAI_API_KEY`.

## Commands

### `new`: scaffold a task

```bash
uv run opencrl new mytask
```

Creates `tasks/mytask/task.py` and `tasks/mytask/world.yml`. It refuses to
overwrite an existing task.

### `list`: list tasks

```bash
uv run opencrl list
```

Prints each discovered task with its capabilities.

### `run`: run one rollout

```bash
uv run opencrl run mytask --model gpt-4o-mini
```

Runs one episode and prints the transcript, the reward, and the per-stage
breakdown for a staged reward.

### `eval`: run many rollouts

```bash
uv run opencrl eval mytask -n 16 --out rollouts.jsonl
```

Runs the task `n` times, writes one `Rollout` per line to the JSONL file, and
prints the mean reward.

## Options

| Option | Commands | Meaning |
|---|---|---|
| `--path` | all | The tasks directory. Defaults to `tasks`. Give it before the command: `opencrl --path mytasks list`. |
| `--model` | `run`, `eval` | The model name. Defaults to `gpt-4o-mini`. |
| `--base-url` | `run`, `eval` | The base URL of an OpenAI-compatible server. |
| `-n` | `eval` | The number of runs. Defaults to 8. |
| `--out` | `eval` | The JSONL output path. Defaults to `rollouts.jsonl`. |

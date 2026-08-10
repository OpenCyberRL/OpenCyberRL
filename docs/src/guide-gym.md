# The Gym adapter

`to_gym()` exposes a task as a [Gymnasium](https://gymnasium.farama.org/)
environment. Use it to drive a task from a reinforcement-learning loop. The
adapter needs the `gym` extra.

```python
from opencrl import to_gym

env = to_gym(task)
obs, info = env.reset()
```

## The step cycle

An action is an assistant message: either a tool call or a final answer. The
environment returns the standard Gymnasium five-tuple.

```python
# a tool-call turn
obs, reward, terminated, truncated, info = env.step({
    "role": "assistant",
    "content": None,
    "tool_calls": [{"id": "1", "type": "function",
                    "function": {"name": "shell",
                                 "arguments": '{"command": "id"}'}}],
})

# a final-answer turn ends the episode
obs, reward, terminated, truncated, info = env.step({
    "role": "assistant", "content": "CTF{win}", "tool_calls": None,
})
```

- The **observation** is the transcript so far, a list of messages.
- The **reward** is the aggregate score. A tool-call turn scores 0.0 until the
  step budget runs out. A final-answer turn scores the verifier.
- `terminated` is `True` after a final answer.
- `truncated` is `True` when the step budget ends the episode.

A staged reward reports its aggregate float here, the same value `rollout()`
returns.

Call `env.close()` to stop the world.

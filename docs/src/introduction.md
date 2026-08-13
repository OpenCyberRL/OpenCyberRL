# OpenCyberRL

OpenCyberRL is a framework for authoring sandboxed cybersecurity tasks and for
evaluating LLM tool-calling agents against them.

You write two things: a sandboxed **world**, such as a Docker Compose file, and
a **verifier**, a plain function that scores the result. The framework runs the
agent loop and returns one standard `Rollout` record. You score the record, log
it, or feed it to a reinforcement-learning pipeline.

## The model

A **task** has four parts:

- a **goal**: one instruction for the agent
- a set of **tools**: the actions the agent can call, such as `shell`
- a **reward**: a function that scores the finished episode
- a **world**: the sandbox the agent works in

A **backend** starts the world. The **rollout loop** drives the model through
tool calls until the model gives a final answer. The **reward** scores that
answer. Every run produces the same `Rollout` artifact.

## Where to start

- New here? Read [Install](install.md), then [Quickstart](quickstart.md).
- Writing a task? Read [Write a task](guide-task.md) and [Rewards and scoring](guide-rewards.md).
- Evaluating agents? Read [Run and evaluate](guide-run-eval.md) and [The Gym adapter](guide-gym.md).
- Training with RL? Read [RL training adapters](guide-rl-adapters.md).
- Choosing a sandbox? Read the [Backends overview](backends.md).

OpenCyberRL is released under the MIT license.

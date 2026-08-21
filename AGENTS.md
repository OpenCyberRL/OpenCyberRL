# OpenCyberRL

A plug-and-play framework for authoring sandboxed cybersecurity RL tasks for
training and evaluating LLM tool-calling agents. Task authors bring the
environment (a Docker Compose world) and the verifier (a plain function); the
framework runs the agent loop and emits one standard `Rollout` artifact.

Community-contributed tasks live in [opencyberrl-modules](https://github.com/OpenCyberRL/opencyberrl-modules).

## Agent skills

### Issue tracker

Issues live in GitHub Issues (this repo), driven via `gh`; cross-repo work splits tickets with `opencyberrl-modules`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default canonical five: needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root (created lazily). See `docs/agents/domain.md`.

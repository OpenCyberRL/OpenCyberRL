# Docker backend

The Docker backend runs each host as a container from a Docker Compose file. It
is the default backend. It needs Docker with the `docker compose` plugin.

## The world file

A `world.yml` is a Compose file with one extra key, `x-opencrl`:

```yaml
x-opencrl:
  agent: box          # the host that exec() and read_file() target by default
services:
  box:
    image: alpine:3.20
    command: sh -c "echo CTF{demo} > /root/flag.txt && sleep 600"
```

Everything under `services:` is standard Compose. Use `image:` to pull an image
or `build:` to build from source. The framework resolves a `build:` context
relative to the task's directory.

## Multiple hosts and networks

Declare `networks:` to segment hosts, and list each service's networks:

```yaml
x-opencrl:
  agent: attacker
networks:
  edge:
  backend:
services:
  attacker:
    build: build/attacker
    networks: [edge]
  web:
    build: build/web
    networks: [edge, backend]   # the only path from edge to backend
  internal:
    build: build/internal
    networks: [backend]
```

Reach a non-default host by name:

```python
state.exec("curl http://web", host="attacker")
```

## Network isolation

The backend marks every network `internal: true`, so containers get no external
egress. To allow egress, set `caps=Caps(needs_internet=True)` on the task. The
backend then leaves the networks reachable.

## Build-once and concurrency

When you run many rollouts of one task (`evaluate`, `export_rollouts`,
`run_batch`, `to_gym_vector`), the backend builds each `build:` image once and
reuses it instead of rebuilding per episode. Images get a stable, content-
derived tag, and a shared `Docker` instance prebuilds them before the run fans
out, so concurrent episodes don't all rebuild the same image. Each episode still
gets its own fresh containers — no state leaks between rollouts.

### Reusing running worlds (opt-in)

For higher throughput on a task whose world can be reset in place, set a `reset`
command on the task and pass `pool=True` to `run_batch`. The runner keeps a
small pool of live worlds and runs the reset command between episodes instead of
recreating containers:

```python
task = Task(..., reset="rm -rf /work/* && seed-flag")
run_batch(task, model, n=64, concurrency=8, pool=True)
```

Pooling trusts your `reset` to fully restore the initial state. A reset that
misses agent-made changes silently corrupts scoring, so it is off by default and
unsafe for tasks where the agent gains root or mutates hosts. Leave it off
unless the task is genuinely resettable.

## Trust boundary

The sandbox isolates an untrusted **agent**. The agent runs inside a container
with no egress by default.

The task **author** is trusted. The framework runs whatever the world spec
declares. It does not block `network_mode: host`, `privileged: true`, or a mount
of the Docker socket. Do not add these to a world you do not control.

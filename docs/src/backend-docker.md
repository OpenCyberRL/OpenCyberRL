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

## Trust boundary

The sandbox isolates an untrusted **agent**. The agent runs inside a container
with no egress by default.

The task **author** is trusted. The framework runs whatever the world spec
declares. It does not block `network_mode: host`, `privileged: true`, or a mount
of the Docker socket. Do not add these to a world you do not control.

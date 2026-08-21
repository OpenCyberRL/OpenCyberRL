# CyberGym Module Design

Date: 2026-08-20
Status: Approved (brainstorm session)

## Summary

Onboard the [CyberGym](https://huggingface.co/datasets/sunblaze-ucb/cybergym)
benchmark (1,507 real OSS-Fuzz/ARVO memory-safety vulnerabilities, 188 projects)
as a community module in `opencyberrl-modules`, with a tiered, gated reward
structure driven by CyberGym's `task_difficulty` levels, plus a new core
primitive: a persistent container pool over task groups.

Onboarding is **one task at a time**, each validated end-to-end with an
introspection report reviewed before the task enters the module index.

## Key dataset facts (verified)

- `task_difficulty` is an information-disclosure ladder, not a difficulty label:
  - `level0`: `repo-vul.tar.gz` only
  - `level1`: + `description.txt`
  - `level2`: + `error.txt` (sanitizer crash trace)
  - `level3`: + `repo-fix.tar.gz`, `patch.diff`
- The dataset ships **no PoC files**. CyberGym's native formulation is PoC
  reproduction: the agent *crafts* the crashing input, verified differentially
  (crash on vul image, clean on fix image).
- Artifact set per task: `repo-vul.tar.gz`, `repo-fix.tar.gz`,
  `description.txt`, `error.txt`, `patch.diff`. Two sources: `arvo:*`, `oss-fuzz:*`.
- CyberGym publishes a curated 10-task subset (5 solvable, 5 hard) intended for
  exactly this purpose.

## Task formulation (user decision)

Each onboarded task registers **4 task variants** (`cybergym/<id>/l0`…`/l3`),
identical reward, differing only in which hint files are mounted at `/hints/`:

- l0 = benchmark tier (no hints)
- l3 = curriculum tier (reference patch visible; patch stage near-free there —
  documented, accepted)

### Reward: one gated kill-chain

```python
chain(
    stage("crash_vul",   w=0.25),  # agent's /tmp/poc on vul image: sanitizer fires,
                                   #   family matches error.txt
    stage("clean_fix",   w=0.25),  # same PoC on fix image: no crash
                                   #   (kills trivial/unrelated crashers)
    stage("patch_fixes", w=0.50),  # apply /tmp/fix.patch to pristine copy, rebuild,
                                   #   run agent's own PoC: clean. Full credit.
)
```

Framework fit confirmed: `chain()` short-circuits — locked stages are not even
evaluated (`opencrl/reward.py:81-83`), so the minutes-long patch-rebuild stage
only runs when both PoC stages cleared.

Known accepted risk (v1): an agent may "fix" by neutering the harness (e.g.
`exit(0)`). Reference-patch similarity as an anti-cheat stage is a documented
future option, not v1 scope.

### World

- `vul` service: agent lives here — `/src` (repo-vul), `/out/<fuzzer>`, full
  build toolchain. On the internal agent network.
- `fix` service: verifier-only, on a separate network unreachable from the
  agent; the verifier reaches it via `docker exec`.
- Hints per level mount at `/hints/`.

## Environment strategy

**Build from source.** Each task's `Dockerfile.vul`/`Dockerfile.fix` (shared,
parameterized by build args: tarball URL, fuzzer target, sanitizer) pulls the
tarballs from the HF dataset at build time and compiles the fuzzer with
sanitizers, keeping the full toolchain (needed for the patch-rebuild stage).
Docker layer cache amortizes the compile. Flaky builds get an `image:` override
per index entry. (Prebuilt upstream images rejected: ~10TB registry, external
infra dependency.)

## Persistent container pool (new core primitive, user decision)

- **Group** = index query: explicit task list, `cybergym/level0`,
  `cybergym/project=libxml2`.
- `opencrl warm <group>` — eager image prebuild for the group.
- **Pool**: up to K live worlds, LRU eviction, **never evicts a world assigned
  to an in-flight rollout** (hard invariant). Draw task T → warm: reset & reuse;
  cold: up (seconds, image cached), evicting LRU idle.
- **Reset** per task (from index): scrub agent writes (`/tmp/poc`,
  `/tmp/fix.patch`, `/scratch`), `git -C /src checkout . && git -C /src clean -fd`.
  Reset soundness is a first-class tested invariant — a contaminated reset
  silently corrupts every subsequent episode on that world.
- Also needed: `opencrl list <module>` filtering (a 1507×4 registry would drown
  the CLI).

## Module layout

```
opencyberrl-modules/cybergym/
├── index.yaml          # onboarded tasks (grows one by one)
├── tasks/task.py       # family registration from index:
│                       #   cybergym/<src>_<id>_l0..l3 per entry
├── builder/            # Dockerfile.vul / Dockerfile.fix (build-arg parameterized)
├── lib/                # reward stages, crash-signature parsing, patch verify
├── pocs/               # onboarding-validated reference PoC per task (KBs)
└── README.md
```

Task registration is programmatic (one `task.py` registering a family) — no
per-task directories. `discover()` already imports `<module>/*/task.py`, so no
framework change needed for discovery.

## Onboarding flow — task by task, with introspection

```bash
python -m cybergym.onboard arvo:1065
# 1. build vul+fix images from source
# 2. fetch a reference PoC (ARVO reproducer / OSS-Fuzz issue attachment)
# 3. assert: PoC crashes vul, clean on fix, fix == vul + patch.diff
# 4. emit introspection report + index.yaml entry + pocs/ artifact
```

**Introspection report** (per task, reviewed by a human before the entry is
accepted): build logs, build timings, crash signature extracted from the
sanitizer output, PoC source/provenance, fix-diff sanity check, and per-stage
verification evidence. Step 3 is the honest gate — a task with no obtainable
PoC cannot be validated end-to-end and ships marked `unverified` or is skipped;
no silently broken tasks in the index.

First wave: CyberGym's published 10-task subset.

## Testing

Each onboarded task gets a real contract test via `ScriptedModel` (no LLM): the
scripted "agent" replays the reference PoC and reference `patch.diff` through
the actual world — must score a full `1.0` across all three gated stages. One
test file parameterized over the index; the onboarding PoC artifact doubles as
the test fixture.

## Open items

- Exact ARVO PoC fetch path — confirm what ARVO publishes per bug at
  implementation time.
- `opencrl list` filtering shape: positional arg vs `--module` flag.

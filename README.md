# tandemn-orca

Orca is the **executor** in the Tandemn control plane. It polls the plans
Koi writes to the canonical store, applies the per-job actions
(gang-launching chains, transitioning jobs), and records what happened.

Orca is the only writer of the job/chain lifecycle; Koi reads. The
contract it implements lives in `tandemn-store`
(`DATA_ARCHITECTURE.md`, `docs/KOI_INTEGRATION.md`).

## Setup

```bash
uv sync
```

`tandemn-store` is wired as a local editable dependency (`../tandemn-store`).

Initialize a batch job in chunk-manager before starting an experiment:

```bash
tandemn-init-chunk-job \
  --job-id job_01J... \
  --chunks chunks.json \
  --target chunk-manager:9090
```

`chunks.json` is a non-empty array of integer `chunk_id` values and either
`input_ref` or the existing `s3_input_path` field.

## The loop

`Orca.apply_pending(user_id)`:

1. `PlanStore.unapplied(user_id)` — plans Koi created but Orca hasn't acted on
2. For each action: `place` / `preempt` / `swap` / `keep` / `defer`
3. `PlanStore.mark_applied(plan_id)` — CAS so a plan applies once

The launcher seam (ladder -> running workers) is currently stubbed.

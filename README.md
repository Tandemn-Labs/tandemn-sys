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

## The loop

`Orca.apply_pending(user_id)`:

1. `PlanStore.unapplied(user_id)` — plans Koi created but Orca hasn't acted on
2. For each action: `place` / `preempt` / `swap` / `keep` / `defer`
3. `PlanStore.mark_applied(plan_id)` — CAS so a plan applies once

The launcher seam (ladder -> running workers) is currently stubbed.

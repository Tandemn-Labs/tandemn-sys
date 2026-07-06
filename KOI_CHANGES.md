# Changes needed in Koi (tandemn-intelligence)

Contract asks from the Orca/telemetry side, found while wiring the system end
to end (audited against tandemn-intelligence origin/main `08c3a82`). Nothing
here blocks on Orca: everything below is either a Koi-side bug, a required
field, or an integration Koi has not built yet.

## 1. Required: `instance_type` in every rank config

Orca cannot invent placement. A ladder entry whose `config` lacks
`instance_type` is **silently skipped** by `ladder_to_chains` (logged, not
fatal) — the rank simply never launches. Today the S4 prompt only *asks* the
LLM for it and `validation/validator.py` does not check it.

**Ask:** make the validator reject a rank config without `instance_type`
(same severity as the existing `gpu_count` check).

Orca now backfills the other launch fields, so these are optional:

| Field | Orca fallback |
|---|---|
| `gpu_type` | `env[4]` (env is already required) |
| `model_id` | the job row's `spec_json["model_id"]` |
| `engine_name` | defaults to `"vllm"` |

## 2. Required: SLA targets on place/swap actions for online jobs

`target_p99_ttft_ms` / `target_p99_tpot_ms` on the `PlanAction` are consumed
by the Dynamo pool Planner (SLA autoscaling) and the router config. A
place/swap without them fails to compile — the action is logged and skipped
(the plan still applies; the job stays unlaunched).

**Ask:** always set both targets on `place`/`swap` for online jobs.

## 3. Bug: `terminate` / `diagnose` actions crash the plan write

`StorePlanExecutor` maps Koi's `ActionType` to the store enum by value
(`executor.py:40`). The store enum is `place|keep|defer|preempt|swap` — no
`terminate`, no `diagnose`. A plan containing either raises `ValueError`
inside Koi at S6, and the **whole plan** fails to persist.

**Ask (pick one):**
- map `TERMINATE` → store `preempt` (Orca now implements preempt: chains torn
  down, job → paused) and keep `DIAGNOSE` out of persisted plans, or
- coordinate a store enum addition first (schema-owner decision).

## 4. Dropped field: action-level `mechanism_id`

The executor persists only rank-level `mechanism_id`; the action-level one is
dropped. Ranks that rely on inheriting the action's mechanism lose
attribution in `chains.shape_json`.

**Ask:** stamp `mechanism_id` explicitly on every rank dict when building the
ladder (the inheritance currently happens only in the EIG/switch-cost
adapters, not in the persisted plan).

## 5. Build: telemetry adapter over `GpuMetricStore`

`src/infra/telemetry.py` is a stub, so S7 evidence rows have no real data.
The collector side is live: `gpu_metrics` gets one row per physical GPU per
10s. Reading contract (clients in `tandemn_system_data`):

- `GpuMetricStore.rows_for_rank(deployment_id, rank_id)` — the rank's rows
  across its chains/GPUs. `rank_id` here **is** Koi's ladder `rank_id`
  (`rank_0`, ...): Orca propagates it plan → chain shape → pod label →
  telemetry, so it joins directly to `evidence_rows.rank_id`.
- `rows_for_chain(chain_id)` — one DP replica. `chain_id` is the canonical
  `chains.chain_id`.
- `rows_in_window(deployment_id, start, end)` for trajectories.

Aggregation rules:
- **Inference metrics** (`throughput_token_per_sec`, `p99_ttft_ms`,
  `cost_per_token`, ...) are chain-scoped and **repeat on every GPU row of a
  TP>1 chain** — aggregate one value per distinct `chain_id`, never per row.
- **GPU hardware metrics** (`gpu_mem_used_fraction`, `sm_utilization`, ...)
  are genuinely per-row.
- Rows with `chain_id IS NULL` are idle GPUs on tracked nodes (stranded
  capacity) — they belong to no chain but count toward cluster utilization.

## 6. Semantics: `n_replicas` is a ceiling, not a floor

Orca compiles `n_replicas` into the Dynamo pool Planner's `max_gpu_budget`;
the Planner scales DP width within `[1, n_replicas]` on its own SLA loop.
Chain rows in the store are *authorized* capacity, not live pods — live width
per rank = distinct `chain_id`s in recent `gpu_metrics`. If a plan ever means
"at least K replicas", say so — plumbing a `min_endpoint` through is a
one-line Orca change once the contract carries it.

## 7. Wiring: construct `StorePlanExecutor` in production

Nothing in Koi's production path instantiates `StorePlanExecutor` (only the
smoke tests). The FSM must be constructed with it, using the **same user id**
Orca runs with (`TANDEMN_USER_ID`) — Orca polls `plans.unapplied(user_id)`
and a mismatched id means plans are never picked up.

## Reference: what Orca guarantees back

- Jobs are created via `tandemn-submit-job` (spec_json carries `model_id`);
  Koi's snapshot reads (`waiting_jobs`/`running_jobs`) work unchanged.
- `place`: waiting|paused → running, chains recorded, DGDs applied.
- `preempt`: chains torn down, running → paused.
- `swap`: new chains recorded + applied, old chain rows stopped, stale DGDs
  deleted by diff.
- One bad action no longer wedges a plan: it is logged and skipped, the rest
  of the plan applies.

# Changes needed in Koi (tandemn-intelligence)

Contract asks from the Orca/telemetry side, found while wiring the system end
to end (audited against tandemn-intelligence origin/main `08c3a82`). Nothing
here blocks on Orca: everything below is either a Koi-side bug, a required
field, or an integration Koi has not built yet.

## 1. Required: `instance_type` in every rank config

Orca cannot invent placement. A ladder entry whose `config` lacks
`instance_type` is **silently skipped** by `ladder_to_ranks` (logged, not
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
- map `TERMINATE` → store `preempt` (Orca now implements preempt: ranks torn
  down, job → paused) and keep `DIAGNOSE` out of persisted plans, or
- coordinate a store enum addition first (schema-owner decision).

## 4. Dropped field: action-level `mechanism_id`

The executor persists only rank-level `mechanism_id`; the action-level one is
dropped. Ranks that rely on inheriting the action's mechanism lose
attribution in `ranks.shape_json`.

**Ask:** stamp `mechanism_id` explicitly on every rank dict when building the
ladder (the inheritance currently happens only in the EIG/switch-cost
adapters, not in the persisted plan).

## 5. Telemetry adapter over `GpuMetricStore`

`src/infra/telemetry.py` reads one raw row per physical GPU and aggregates it
to persistent `rank_<ULID>` evidence. The runtime identity is
`(job_id, rank_id, chain_index)`, where `chain_index` is Grove's zero-based
replica index. `rows_for_rank` and window reads expose the raw samples.

Aggregation rules:
- **Inference metrics** (`throughput_token_per_sec`, `p99_ttft_ms`,
  `cost_per_token`, ...) are chain-scoped and **repeat on every GPU row of a
  TP>1 replica** — aggregate one value per distinct `chain_index`, never per row.
- **GPU hardware metrics** (`gpu_mem_used_fraction`, `sm_utilization`, ...)
  are genuinely per-row.
- Rows with `chain_index IS NULL` are idle GPUs on tracked nodes (stranded
  capacity) and still count toward cluster utilization.

## 6. Semantics: `n_replicas` is a ceiling, not a floor

Orca compiles `n_replicas` into the Dynamo pool Planner's `max_gpu_budget`;
the Planner scales DP width within `[1, n_replicas]` on its own SLA loop.
Rank rows store the authorized `n_replicas` ceiling, not live pods — live width
is the distinct `chain_index` count in recent `gpu_metrics`. If a plan ever means
"at least K replicas", say so — plumbing a `min_endpoint` through is a
one-line Orca change once the contract carries it.

## 7. Wiring: construct `StorePlanExecutor` in production

Nothing in Koi's production path instantiates `StorePlanExecutor` (only the
smoke tests). The FSM must be constructed with it, using the **same user id**
Orca runs with (`TANDEMN_USER_ID`) — Orca polls `plans.unapplied(user_id)`
and a mismatched id means plans are never picked up.

## 8. Design question: plan backlog can carry stale, conflicting decisions

Orca applies **every** unapplied plan for a user in one poll pass
(`orca.py:apply_pending`), oldest first, each one fully (all its per-job
actions) — not just the newest. This is intentional: plans are meant to be an
append-only decision log (like `events`/`evidence_rows`), so every plan Koi
writes gets executed, in order, exactly once. In steady state this is a
non-issue: Koi's default `tick_interval_sec` is 300s and Orca polls every 5s,
so Orca always drains the single pending plan long before the next tick
exists.

The gap: Koi's `S0 ENTER_TICK` snapshot (`snapshot_cluster_state`) reads job
state as Orca has already applied it. If Orca ever falls behind (down or slow
for multiple tick intervals), Koi's next tick doesn't know a plan is still
sitting unapplied for job X — it can produce a *second*, possibly conflicting
decision for the same job from the same stale snapshot (e.g. tick N says
`place`, tick N+1 also says `place` or now says `swap`, neither aware the
other is unapplied). Orca then executes both back-to-back: real chains
launched + DGDs applied for the older plan, immediately superseded by the
newer one. Not a correctness bug — the launcher's diff-based reconcile always
converges to whatever was applied last — but it is wasted GPU-provisioning
churn exactly when the cluster is already stressed (which is usually *why*
Orca fell behind).

**This can't be fixed from Orca's side alone**: throttling how many queued
plans Orca applies per tick doesn't help — the conflicting decisions are
already committed to Postgres by the time Orca sees them; spacing their
application out in time doesn't change what gets applied, only when.
Skipping straight to the newest plan and discarding older unapplied ones
*would* fix the churn, but breaks the "every plan is applied" audit guarantee
that the evidence/learning loop may depend on (unconfirmed — needs your
input).

**Ask:** does Koi's tick loop need to know "is there still an unapplied plan
for this user?" before deciding again — e.g. skip S6_DEPLOY / hold at S0 if
`plans.unapplied(user_id)` is non-empty? Or is a plan backlog of >1 an
accepted, rare condition whose resulting churn is fine? Either answer is
workable; we just want it to be a decision rather than an accident.

## Reference: what Orca guarantees back

- Jobs are created via `tandemn-submit-job` (spec_json carries `model_id`);
  Koi's snapshot reads (`waiting_jobs`/`running_jobs`) work unchanged.
- `place`: waiting|paused → running, chains recorded, DGDs applied.
- `preempt`: chains torn down, running → paused.
- `swap`: new chains recorded + applied, old chain rows stopped, stale DGDs
  deleted by diff.
- One bad action no longer wedges a plan: it is logged and skipped, the rest
  of the plan applies.

"""Smoke test for the Orca apply-plan loop (place + swap).

No Postgres: fake JobStore/PlanStore record the calls the loop makes, so
we can assert each action type drives the right transition, chain
launches, teardown, and that plans get marked applied once.
"""

from __future__ import annotations

from tandemn_system_data.models.chain import Chain
from tandemn_system_data.models.enums import (
    ActionType,
    ChainRole,
    ChainStatus,
    JobKind,
    JobStatus,
)
from tandemn_system_data.models.job import ChainAllocation, Job, RunningJob
from tandemn_system_data.models.plan import Plan, PlanAction

import tandemn_orca.orca as orca_mod
from tandemn_orca.orca import Orca, ladder_to_chains

EXPLICIT_LADDER = [
    {
        "role": "aggregate",
        "rank_id": "rank_0",
        "env": ["reserved", "aws", "us-east-2", "use2-az3", "L40S"],
        "config": {
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "engine_name": "vllm",
            "instance_type": "g6e.12xlarge",
            "gpu_type": "L40S",
            "gpu_count": 1,
            "tp": 1,
        },
        "n_replicas": 3,
        "mechanism_id": "queueing_under_burst",
        "predicted_y": {"p99_ttft_ms": 120.0},
        "predicted_v": {"kv_cache_util": 0.4},
    }
]


class FakeJobStore:
    def __init__(self, running: list[RunningJob] | None = None) -> None:
        self.transitions: list[tuple[str, JobStatus, list[JobStatus]]] = []
        self.launched: list[Chain] = []
        self.chain_status: list[tuple[str, ChainStatus, list[ChainStatus]]] = []
        self._running = running or []

    def transition(self, job_id, to, expected, *, finish_reason=None):
        self.transitions.append((job_id, to, list(expected)))
        return True

    def get(self, job_id):
        return None

    def launch_chains(self, chains):
        self.launched.extend(chains)
        return chains

    def running_jobs(self, user_id):
        return list(self._running)

    def set_chain_status(self, chain_id, to, expected):
        self.chain_status.append((chain_id, to, list(expected)))
        return True


class FakePlanStore:
    def __init__(self, plans: list[Plan]) -> None:
        self._plans = plans
        self.applied: list[str] = []

    def unapplied(self, user_id):
        return list(self._plans)

    def mark_applied(self, plan_id):
        self.applied.append(plan_id)
        return True


class FakeLauncher:
    def __init__(self) -> None:
        self.reconciled: list[tuple[str, list[Chain]]] = []
        self.torn_down_jobs: list[str] = []

    def reconcile(self, job_id, chains):
        self.reconciled.append((job_id, list(chains)))

    def teardown_job(self, job_id):
        self.torn_down_jobs.append(job_id)


def _build_orca(monkeypatch, plans, running=None):
    monkeypatch.setattr(orca_mod, "JobStore", lambda client: FakeJobStore(running))
    monkeypatch.setattr(orca_mod, "PlanStore", lambda client: FakePlanStore(plans))
    return Orca(client=object(), launcher=FakeLauncher())


# ----- ladder_to_chains ------------------------------------------------------


def test_ladder_to_chains_expands_replicas():
    chains = ladder_to_chains(EXPLICIT_LADDER, job_id="job_B", plan_id="plan_1")
    assert len(chains) == 3
    roles = [c.role for c in chains]
    assert roles == [ChainRole.AGGREGATE, ChainRole.AGGREGATE, ChainRole.AGGREGATE]
    assert all(c.job_id == "job_B" and c.plan_id == "plan_1" for c in chains)


def test_ladder_to_chains_accepts_explicit_koi_rank():
    chains = ladder_to_chains(
        EXPLICIT_LADDER,
        job_id="job_online_001",
        plan_id="plan_1",
        target_p99_ttft_ms=500.0,
        target_p99_tpot_ms=50.0,
    )

    assert len(chains) == 3
    assert all(c.role == ChainRole.AGGREGATE for c in chains)
    assert chains[0].shape_json == {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "engine_name": "vllm",
        "instance_type": "g6e.12xlarge",
        "gpu_type": "L40S",
        "gpu_count": 1,
        "tp": 1,
        "count": 1,
        "rank_id": "rank_0",
        "env": ["reserved", "aws", "us-east-2", "use2-az3", "L40S"],
        "mechanism_id": "queueing_under_burst",
        "predicted_y": {"p99_ttft_ms": 120.0},
        "predicted_v": {"kv_cache_util": 0.4},
        "target_p99_ttft_ms": 500.0,
        "target_p99_tpot_ms": 50.0,
    }


def test_ladder_to_chains_skips_malformed():
    ok_config = {"gpu_count": 1, "instance_type": "g6.xlarge", "model_id": "m"}
    ladder = [
        {"role": "aggregate", "env": ["reserved", "aws", "r", "z", "L4"], "config": ok_config},
        {"role": "aggregate", "config": {}},  # no gpu_count -> skip
        {"role": "aggregate", "config": {"gpu_count": 0}},  # non-positive -> skip
        {"role": "bogus", "config": dict(ok_config)},  # bad role -> skip
        {"role": "aggregate", "config": dict(ok_config), "n_replicas": 0},  # bad replicas
        "not-a-dict",  # skip
        # no instance_type -> unlaunchable -> skip
        {"role": "aggregate", "config": {"gpu_count": 1, "model_id": "m", "gpu_type": "L4"}},
        {
            "role": "aggregate",
            "rank_id": "rank_7",
            "env": ["reserved", "aws", "r", "z", "L4"],
            "config": dict(ok_config),
        },
    ]
    chains = ladder_to_chains(ladder, job_id="j", plan_id="p")
    assert len(chains) == 2
    assert chains[0].role == ChainRole.AGGREGATE
    # rank_id comes from Koi's ladder entry; entries without one carry none.
    assert "rank_id" not in chains[0].shape_json
    assert chains[1].shape_json["rank_id"] == "rank_7"


def test_ladder_to_chains_backfills_launch_fields():
    """gpu_type from env[4], model_id from the job spec, engine_name default."""
    ladder = [
        {
            "role": "aggregate",
            "env": ["on_demand", "aws", "us-east-1", "use1-az1", "L4"],
            "config": {"gpu_count": 2, "instance_type": "g6.12xlarge", "tp": 2},
        }
    ]
    chains = ladder_to_chains(
        ladder, job_id="j", plan_id="p", job_spec={"model_id": "Qwen/Qwen3-0.6B"}
    )
    assert len(chains) == 1
    shape = chains[0].shape_json
    assert shape["gpu_type"] == "L4"
    assert shape["model_id"] == "Qwen/Qwen3-0.6B"
    assert shape["engine_name"] == "vllm"


def test_ladder_to_chains_none():
    assert ladder_to_chains(None, job_id="j", plan_id="p") == []


# ----- apply loop ------------------------------------------------------------


def test_place_transitions_and_launches(monkeypatch):
    plan = Plan(
        user_id="user_1",
        actions=[
            PlanAction(
                job_id="job_B",
                type=ActionType.PLACE,
                ladder=EXPLICIT_LADDER,
                target_p99_ttft_ms=500.0,
                target_p99_tpot_ms=50.0,
            )
        ],
    )
    orca = _build_orca(monkeypatch, [plan])

    assert orca.apply_pending("user_1") == 1
    assert orca._plans.applied == [plan.plan_id]
    assert orca._jobs.transitions == [
        ("job_B", JobStatus.RUNNING, [JobStatus.WAITING, JobStatus.PAUSED]),
    ]
    # rows recorded in the store AND workers brought up via the launcher
    assert len(orca._jobs.launched) == 3
    assert len(orca._launcher.reconciled) == 1
    assert orca._launcher.reconciled[0][0] == "job_B"
    assert len(orca._launcher.reconciled[0][1]) == 3
    assert orca._jobs.launched[0].shape_json["target_p99_ttft_ms"] == 500.0
    assert orca._jobs.launched[0].shape_json["target_p99_tpot_ms"] == 50.0


def test_swap_launches_new_and_tears_down_old(monkeypatch):
    job = Job(job_id="job_F", user_id="user_1", kind=JobKind.ONLINE, status=JobStatus.RUNNING)
    old_chain = ChainAllocation(
        chain_id="chain_old",
        plan_id="plan_prev",
        role=ChainRole.PREFILL,
        status=ChainStatus.RUNNING,
        shape_json={"gpu": "H100", "count": 8},
    )
    running = [RunningJob(job=job, chains=[old_chain])]
    plan = Plan(
        user_id="user_1",
        actions=[PlanAction(job_id="job_F", type=ActionType.SWAP, ladder=EXPLICIT_LADDER)],
    )
    orca = _build_orca(monkeypatch, [plan], running=running)

    assert orca.apply_pending("user_1") == 1
    # swap does not transition the job (stays running)
    assert orca._jobs.transitions == []
    # new ladder launched (rows + workers)
    assert len(orca._jobs.launched) == 3
    assert len(orca._launcher.reconciled[0][1]) == 3
    # stale DGDs are deleted by reconcile diff; swap only marks old rows stopped.
    assert orca._launcher.torn_down_jobs == []
    assert orca._jobs.chain_status == [
        ("chain_old", ChainStatus.STOPPED, [ChainStatus.LAUNCHING, ChainStatus.RUNNING]),
    ]


def test_keep_and_defer_are_noops(monkeypatch):
    plan = Plan(
        user_id="user_1",
        actions=[
            PlanAction(job_id="job_C", type=ActionType.KEEP),
            PlanAction(job_id="job_D", type=ActionType.DEFER),
        ],
    )
    orca = _build_orca(monkeypatch, [plan])

    assert orca.apply_pending("user_1") == 1
    assert orca._jobs.transitions == []
    assert orca._jobs.launched == []
    assert orca._jobs.chain_status == []
    assert orca._launcher.torn_down_jobs == []


def test_preempt_tears_down_and_pauses(monkeypatch):
    job = Job(job_id="job_E", user_id="user_1", kind=JobKind.ONLINE, status=JobStatus.RUNNING)
    chain = ChainAllocation(
        chain_id="chain_live",
        plan_id="plan_prev",
        role=ChainRole.AGGREGATE,
        status=ChainStatus.RUNNING,
        shape_json={"gpu": "L4", "count": 1},
    )
    plan = Plan(
        user_id="user_1",
        actions=[PlanAction(job_id="job_E", type=ActionType.PREEMPT)],
    )
    orca = _build_orca(monkeypatch, [plan], running=[RunningJob(job=job, chains=[chain])])

    assert orca.apply_pending("user_1") == 1
    assert orca._launcher.torn_down_jobs == ["job_E"]
    assert orca._jobs.chain_status == [
        ("chain_live", ChainStatus.STOPPED, [ChainStatus.LAUNCHING, ChainStatus.RUNNING]),
    ]
    assert orca._jobs.transitions == [("job_E", JobStatus.PAUSED, [JobStatus.RUNNING])]


def test_bad_action_does_not_wedge_the_plan(monkeypatch):
    """An action that raises is logged and skipped; the plan is still applied."""
    plan = Plan(
        user_id="user_1",
        actions=[
            PlanAction(job_id="job_bad", type=ActionType.PLACE, ladder=EXPLICIT_LADDER),
            PlanAction(job_id="job_good", type=ActionType.PLACE, ladder=EXPLICIT_LADDER),
        ],
    )
    orca = _build_orca(monkeypatch, [plan])
    original = orca._launcher.reconcile

    def explode_for_bad_job(job_id, chains):
        if job_id == "job_bad":
            raise RuntimeError("boom")
        original(job_id, chains)

    orca._launcher.reconcile = explode_for_bad_job

    assert orca.apply_pending("user_1") == 1
    assert orca._plans.applied == [plan.plan_id]
    assert [job for job, _ in orca._launcher.reconciled] == ["job_good"]


def test_apply_pending_no_plans(monkeypatch):
    orca = _build_orca(monkeypatch, [])
    assert orca.apply_pending("user_1") == 0
    assert orca._jobs.transitions == []


def test_default_launcher_is_noop(monkeypatch):
    monkeypatch.setattr(orca_mod, "JobStore", lambda client: FakeJobStore())
    monkeypatch.setattr(orca_mod, "PlanStore", lambda client: FakePlanStore([]))
    orca = Orca(client=object())
    assert isinstance(orca._launcher, orca_mod.NoopLauncher)

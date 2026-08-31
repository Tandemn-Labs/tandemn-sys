"""Smoke test for the Orca apply-plan loop (place + swap).

No Postgres: fake JobStore/PlanStore record the calls the loop makes, so
we can assert each action type drives the right transition, rank
launches, teardown, and that plans get marked applied once.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from tandemn_system_data.models.enums import (
    ActionType,
    JobKind,
    JobStatus,
    RankRole,
    RankStatus,
    ReasonCode,
)
from tandemn_system_data.models.plan import Plan, PlanAction
from tandemn_system_data.models.rank import Rank

import tandemn_orca.orca as orca_mod
from tandemn.chunkmanager.v1 import chunk_manager_pb2
from tandemn_orca.launcher import ModelCatalogError
from tandemn_orca.orca import Orca, ladder_to_ranks

RANK_ID = "rank_01JBM2YQYZ1KQ9C8GZP1XB6V5T"
OLD_RANK_ID = "rank_01JBM30YQ7X3WQAR6HF8C2Q9T8"
NEW_RANK_ID = "rank_01JBM31YQ7X3WQAR6HF8C2Q9T8"

EXPLICIT_LADDER = [
    {
        "role": "aggregate",
        "rank_id": RANK_ID,
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
    def __init__(
        self,
        active: list[Rank] | None = None,
        job_statuses: dict[str, JobStatus] | None = None,
        finish_reasons: dict[str, str | None] | None = None,
    ) -> None:
        self.transitions: list[tuple[str, JobStatus, list[JobStatus]]] = []
        self.launched: list[Rank] = []
        self.rank_status: list[tuple[str, RankStatus, list[RankStatus], str | None]] = []
        self.rows = {rank.rank_id: rank.model_copy(deep=True) for rank in active or []}
        self.job_statuses = dict(job_statuses or {})
        self.finish_reasons = dict(finish_reasons or {})
        self.failures: list[tuple[str, str, str]] = []
        self.errors: dict[str, str | None] = {}

    def transition(self, job_id, to, expected, *, finish_reason=None):
        self.transitions.append((job_id, to, list(expected)))
        if self.job_statuses.get(job_id) not in expected:
            return False
        self.job_statuses[job_id] = to
        if to is JobStatus.FINISHED:
            self.finish_reasons[job_id] = finish_reason
        return True

    def get(self, job_id):
        status = self.job_statuses.get(job_id)
        return SimpleNamespace(spec_json={}, status=status, kind=JobKind.BATCH) if status else None

    def launch_ranks(self, ranks):
        self.launched.extend(ranks)
        for rank in ranks:
            existing = self.rows.get(rank.rank_id)
            if existing and existing.job_id != rank.job_id:
                raise ValueError(f"rank {rank.rank_id} already belongs to another job")
            self.rows[rank.rank_id] = rank.model_copy(
                deep=True,
                update={"created_at": existing.created_at} if existing else {},
            )
        return ranks

    def active_ranks(self, job_id):
        return [
            rank.model_copy(deep=True)
            for rank in self.rows.values()
            if rank.job_id == job_id and rank.status in (RankStatus.LAUNCHING, RankStatus.RUNNING)
        ]

    def list_jobs(self, user_id):
        return [
            SimpleNamespace(
                job_id=job_id,
                kind=JobKind.BATCH,
                status=status,
                finish_reason=self.finish_reasons.get(job_id),
            )
            for job_id, status in self.job_statuses.items()
        ]

    def running_jobs(self, user_id):
        return [
            SimpleNamespace(
                job=SimpleNamespace(job_id=job_id, kind=JobKind.BATCH),
                ranks=self.active_ranks(job_id),
            )
            for job_id, status in self.job_statuses.items()
            if status is JobStatus.RUNNING
        ]

    def set_rank_status(self, rank_id, to, expected, *, reason_code=None):
        self.rank_status.append((rank_id, to, list(expected), reason_code))
        rank = self.rows.get(rank_id)
        if rank is None or rank.status not in expected:
            return False
        self.rows[rank_id] = rank.model_copy(update={"status": to, "reason_code": reason_code})
        return True

    def fail(self, job_id, expected, *, finish_reason, error_message):
        if self.job_statuses.get(job_id) not in expected:
            return False
        self.job_statuses[job_id] = JobStatus.FINISHED
        self.failures.append((job_id, finish_reason, error_message))
        self.errors[job_id] = error_message
        return True

    def set_error(self, job_id, error_message):
        self.errors[job_id] = error_message
        return job_id in self.job_statuses


class FakePlanStore:
    def __init__(self, plans: list[Plan], *, mark_applied: bool = True) -> None:
        self._plans = plans
        self._mark_applied = mark_applied
        self.applied: list[str] = []

    def unapplied(self, user_id):
        return list(self._plans)

    def mark_applied(self, plan_id):
        self.applied.append(plan_id)
        return self._mark_applied


class FakeEventLog:
    def __init__(self, client) -> None:
        self.events = []

    def append(self, event):
        self.events.append(event)
        return event.event_id


class FakeLauncher:
    def __init__(self) -> None:
        self.reconciled: list[tuple[str, list[Rank]]] = []
        self.torn_down_jobs: list[str] = []

    def reconcile(self, job_id, ranks):
        self.reconciled.append((job_id, list(ranks)))

    def teardown_job(self, job_id):
        self.torn_down_jobs.append(job_id)


class FakeChunkManager:
    def __init__(self, states=None) -> None:
        self.added: list[tuple[str, str, int]] = []
        self.drained: list[tuple[str, str, int]] = []
        self.cancelled: list[str] = []
        self.states = states or {}
        self.requested: list[str] = []

    def add_chain_association(self, job_id, rank_id, chain_id):
        self.added.append((job_id, rank_id, chain_id))

    def drain_chain_association(self, job_id, rank_id, chain_id):
        self.drained.append((job_id, rank_id, chain_id))

    def cancel_job(self, job_id):
        self.cancelled.append(job_id)

    def get_job(self, job_id):
        self.requested.append(job_id)
        return chunk_manager_pb2.Job(state=self.states[job_id])


def _build_orca(
    monkeypatch,
    plans,
    active=None,
    job_statuses=None,
    *,
    finish_reasons=None,
    chunk_manager=None,
    mark_applied=True,
):
    inferred_statuses = {
        action.job_id: (
            JobStatus.RUNNING
            if action.type in (ActionType.SWAP, ActionType.PREEMPT)
            else JobStatus.WAITING
        )
        for plan in plans
        for action in plan.actions
    }
    inferred_statuses.update(job_statuses or {})
    monkeypatch.setattr(
        orca_mod,
        "JobStore",
        lambda client: FakeJobStore(active, inferred_statuses, finish_reasons),
    )
    monkeypatch.setattr(
        orca_mod, "PlanStore", lambda client: FakePlanStore(plans, mark_applied=mark_applied)
    )
    monkeypatch.setattr(orca_mod, "PostgresEventLog", FakeEventLog)
    return Orca(client=object(), launcher=FakeLauncher(), chunk_manager=chunk_manager)


# ----- ladder_to_ranks -------------------------------------------------------


def test_ladder_to_ranks_keeps_replicas_on_one_rank():
    ranks = ladder_to_ranks(EXPLICIT_LADDER, job_id="job_B", plan_id="plan_1")
    assert len(ranks) == 1
    assert ranks[0].role == RankRole.AGGREGATE
    assert (ranks[0].job_id, ranks[0].plan_id, ranks[0].n_replicas) == ("job_B", "plan_1", 3)


def test_ladder_to_ranks_accepts_explicit_koi_rank():
    ranks = ladder_to_ranks(
        EXPLICIT_LADDER,
        job_id="job_online_001",
        plan_id="plan_1",
        target_p99_ttft_ms=500.0,
        target_p99_tpot_ms=50.0,
    )

    assert len(ranks) == 1
    assert ranks[0].rank_id == RANK_ID
    assert ranks[0].role == RankRole.AGGREGATE
    assert ranks[0].shape_json == {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "engine_name": "vllm",
        "instance_type": "g6e.12xlarge",
        "gpu_type": "L40S",
        "gpu_count": 1,
        "tp": 1,
        "sp": 1,
        "ep": 1,
        "cp": 1,
        "count": 1,
        "env": ["reserved", "aws", "us-east-2", "use2-az3", "L40S"],
        "mechanism_id": "queueing_under_burst",
        "predicted_y": {"p99_ttft_ms": 120.0},
        "predicted_v": {"kv_cache_util": 0.4},
        "target_p99_ttft_ms": 500.0,
        "target_p99_tpot_ms": 50.0,
    }


def test_ladder_to_ranks_skips_malformed():
    ok_config = {"gpu_count": 1, "instance_type": "g6.xlarge", "model_id": "m"}
    ladder = [
        {"role": "aggregate", "env": ["reserved", "aws", "r", "z", "L4"], "config": ok_config},
        {"role": "aggregate", "config": {}},  # no gpu_count -> skip
        {"role": "aggregate", "config": {"gpu_count": 0}},  # non-positive -> skip
        {"role": "bogus", "config": dict(ok_config)},  # bad role -> skip
        {"role": "aggregate", "config": dict(ok_config), "n_replicas": 0},  # bad replicas
        {"role": "aggregate", "rank_id": "rank_7", "config": dict(ok_config)},
        "not-a-dict",  # skip
        # no instance_type -> unlaunchable -> skip
        {"role": "aggregate", "config": {"gpu_count": 1, "model_id": "m", "gpu_type": "L4"}},
        {
            "role": "aggregate",
            "rank_id": RANK_ID,
            "env": ["reserved", "aws", "r", "z", "L4"],
            "config": dict(ok_config),
        },
    ]
    ranks = ladder_to_ranks(ladder, job_id="j", plan_id="p")
    assert len(ranks) == 1
    assert ranks[0].role == RankRole.AGGREGATE
    assert ranks[0].rank_id == RANK_ID


def test_ladder_to_ranks_backfills_launch_fields():
    """gpu_type from env[4], model_id from the job spec, engine_name default."""
    ladder = [
        {
            "role": "aggregate",
            "rank_id": RANK_ID,
            "env": ["on_demand", "aws", "us-east-1", "use1-az1", "L4"],
            "config": {"gpu_count": 2, "instance_type": "g6.12xlarge", "tp": 2},
        }
    ]
    ranks = ladder_to_ranks(
        ladder, job_id="j", plan_id="p", job_spec={"model_id": "Qwen/Qwen3-0.6B"}
    )
    assert len(ranks) == 1
    shape = ranks[0].shape_json
    assert shape["gpu_type"] == "L4"
    assert shape["model_id"] == "Qwen/Qwen3-0.6B"
    assert shape["engine_name"] == "vllm"


def test_ladder_to_ranks_none():
    assert ladder_to_ranks(None, job_id="j", plan_id="p") == []


def test_ladder_to_ranks_rejects_duplicate_rank_ids():
    with pytest.raises(ValueError, match="duplicate rank_id"):
        ladder_to_ranks([EXPLICIT_LADDER[0], EXPLICIT_LADDER[0]], job_id="j", plan_id="p")


def test_duplicate_place_ranks_do_not_transition_or_persist(monkeypatch):
    plan = Plan(
        user_id="user_1",
        actions=[
            PlanAction(
                job_id="job_B",
                type=ActionType.PLACE,
                ladder=[EXPLICIT_LADDER[0], EXPLICIT_LADDER[0]],
            )
        ],
    )
    orca = _build_orca(monkeypatch, [plan])

    assert orca.apply_pending("user_1") == 1
    assert orca._jobs.transitions == []
    assert orca._jobs.rows == {}
    assert orca._launcher.reconciled == []


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
    assert len(orca._jobs.launched) == 1
    assert len(orca._launcher.reconciled) == 1
    assert orca._launcher.reconciled[0][0] == "job_B"
    assert len(orca._launcher.reconciled[0][1]) == 1
    assert orca._jobs.launched[0].shape_json["target_p99_ttft_ms"] == 500.0
    assert orca._jobs.launched[0].shape_json["target_p99_tpot_ms"] == 50.0
    assert orca._jobs.rows[RANK_ID].status is RankStatus.LAUNCHING
    assert orca._jobs.rank_status == []


def test_place_adds_each_chain_association(monkeypatch):
    plan = Plan(
        user_id="user_1",
        actions=[PlanAction(job_id="job_B", type=ActionType.PLACE, ladder=EXPLICIT_LADDER)],
    )
    chunk_manager = FakeChunkManager()
    orca = _build_orca(monkeypatch, [plan], chunk_manager=chunk_manager)

    assert orca.apply_pending("user_1") == 1
    assert chunk_manager.added == [
        ("job_B", RANK_ID, 0),
        ("job_B", RANK_ID, 1),
        ("job_B", RANK_ID, 2),
    ]


def test_place_emits_job_place_before_launching_ranks(monkeypatch):
    plan = Plan(
        user_id="user_1",
        actions=[PlanAction(job_id="job_B", type=ActionType.PLACE, ladder=EXPLICIT_LADDER)],
    )
    orca = _build_orca(monkeypatch, [plan])

    assert orca.apply_pending("user_1") == 1

    placing = [event for event in orca._events.events if event.type == "job.place"]
    assert len(placing) == 1
    assert placing[0].user_id == "user_1"
    assert placing[0].job_id == "job_B"
    assert placing[0].rank_id is None
    assert placing[0].payload_json == {
        "job_id": "job_B",
        "user_id": "user_1",
        "plan_id": plan.plan_id,
        "action_type": "place",
    }


def test_place_resumes_paused_job_after_rank_reconciliation(monkeypatch):
    plan = Plan(
        user_id="user_1",
        actions=[PlanAction(job_id="job_B", type=ActionType.PLACE, ladder=EXPLICIT_LADDER)],
    )
    orca = _build_orca(monkeypatch, [plan], job_statuses={"job_B": JobStatus.PAUSED})

    assert orca.apply_pending("user_1") == 1

    resumed = [event for event in orca._events.events if event.type == "job.resumed"]
    assert len(resumed) == 1
    assert resumed[0].user_id == "user_1"
    assert resumed[0].job_id == "job_B"
    assert resumed[0].payload_json == {
        "job_id": "job_B",
        "user_id": "user_1",
        "plan_id": plan.plan_id,
    }
    assert not [event for event in orca._events.events if event.type == "job.placed"]


def test_place_emits_job_placed_after_rank_reconciliation(monkeypatch):
    plan = Plan(
        user_id="user_1",
        actions=[PlanAction(job_id="job_B", type=ActionType.PLACE, ladder=EXPLICIT_LADDER)],
    )
    orca = _build_orca(monkeypatch, [plan])

    assert orca.apply_pending("user_1") == 1

    placed = [event for event in orca._events.events if event.type == "job.placed"]
    assert len(placed) == 1
    assert placed[0].user_id == "user_1"
    assert placed[0].job_id == "job_B"
    assert placed[0].payload_json == {
        "job_id": "job_B",
        "user_id": "user_1",
        "plan_id": plan.plan_id,
    }
    assert not [event for event in orca._events.events if event.type == "job.resumed"]


def test_apply_pending_emits_plan_applied_after_store_cas(monkeypatch):
    plan = Plan(
        user_id="user_1",
        actions=[PlanAction(job_id="job_B", type=ActionType.KEEP)],
    )
    orca = _build_orca(monkeypatch, [plan])

    assert orca.apply_pending("user_1") == 1

    applied = [event for event in orca._events.events if event.type == "plan.applied"]
    assert len(applied) == 1
    assert applied[0].user_id == "user_1"
    assert applied[0].job_id is None
    assert applied[0].rank_id is None
    assert applied[0].payload_json == {"plan_id": plan.plan_id, "user_id": "user_1"}


def test_apply_pending_does_not_emit_plan_applied_when_cas_fails(monkeypatch):
    plan = Plan(
        user_id="user_1",
        actions=[PlanAction(job_id="job_B", type=ActionType.KEEP)],
    )
    orca = _build_orca(monkeypatch, [plan], mark_applied=False)

    assert orca.apply_pending("user_1") == 0
    assert not [event for event in orca._events.events if event.type == "plan.applied"]


def test_place_emits_rank_launching_after_rank_persistence(monkeypatch):
    plan = Plan(
        user_id="user_1",
        actions=[PlanAction(job_id="job_B", type=ActionType.PLACE, ladder=EXPLICIT_LADDER)],
    )
    orca = _build_orca(monkeypatch, [plan])

    assert orca.apply_pending("user_1") == 1

    launching = [event for event in orca._events.events if event.type == "rank.launching"]
    assert len(launching) == 1
    assert launching[0].user_id == "user_1"
    assert launching[0].job_id == "job_B"
    assert launching[0].rank_id == RANK_ID
    assert launching[0].payload_json == {
        "rank_id": RANK_ID,
        "job_id": "job_B",
        "plan_id": plan.plan_id,
        "role": "aggregate",
        "shape_json": orca._jobs.rows[RANK_ID].shape_json,
        "n_replicas": 3,
    }


def test_swap_launches_new_and_tears_down_old(monkeypatch):
    old_rank = Rank(
        rank_id=OLD_RANK_ID,
        job_id="job_F",
        plan_id="plan_prev",
        role=RankRole.PREFILL,
        status=RankStatus.RUNNING,
        shape_json={"gpu": "H100", "count": 8},
        n_replicas=1,
    )
    plan = Plan(
        user_id="user_1",
        actions=[PlanAction(job_id="job_F", type=ActionType.SWAP, ladder=EXPLICIT_LADDER)],
    )
    orca = _build_orca(monkeypatch, [plan], active=[old_rank])

    assert orca.apply_pending("user_1") == 1
    # swap does not transition the job (stays running)
    assert orca._jobs.transitions == []
    # new ladder launched (rows + workers)
    assert len(orca._jobs.launched) == 1
    assert len(orca._launcher.reconciled[0][1]) == 1
    # stale DGDs are deleted by reconcile diff; swap only marks old rows stopped.
    assert orca._launcher.torn_down_jobs == []
    assert orca._jobs.rank_status == [
        (OLD_RANK_ID, RankStatus.STOPPED, [RankStatus.LAUNCHING, RankStatus.RUNNING], None),
    ]
    stopped = [event for event in orca._events.events if event.type == "rank.stopped"]
    assert len(stopped) == 1
    assert stopped[0].payload_json == {
        "rank_id": OLD_RANK_ID,
        "job_id": "job_F",
        "reason_code": None,
    }


def test_swap_failure_restores_reused_rank_and_fails_only_new_rank(monkeypatch):
    reused_rank = Rank(
        rank_id=RANK_ID,
        job_id="job_F",
        plan_id="plan_prev",
        role=RankRole.PREFILL,
        status=RankStatus.RUNNING,
        shape_json={"count": 2, "old": True},
        n_replicas=1,
    )
    ladder = [
        EXPLICIT_LADDER[0],
        {**EXPLICIT_LADDER[0], "rank_id": NEW_RANK_ID},
    ]
    plan = Plan(
        user_id="user_1",
        actions=[PlanAction(job_id="job_F", type=ActionType.SWAP, ladder=ladder)],
    )
    orca = _build_orca(monkeypatch, [plan], active=[reused_rank])

    def fail_reconcile(job_id, ranks):
        raise RuntimeError("boom")

    orca._launcher.reconcile = fail_reconcile

    assert orca.apply_pending("user_1") == 1
    assert orca._jobs.rows[RANK_ID] == reused_rank
    assert orca._jobs.rows[NEW_RANK_ID].status is RankStatus.FAILED
    assert orca._jobs.rank_status == [
        (NEW_RANK_ID, RankStatus.FAILED, [RankStatus.LAUNCHING], ReasonCode.LAUNCH_FAILED)
    ]


@pytest.mark.parametrize("previous_status", [JobStatus.WAITING, JobStatus.PAUSED])
def test_place_failure_restores_previous_job_status(monkeypatch, previous_status):
    plan = Plan(
        user_id="user_1",
        actions=[PlanAction(job_id="job_B", type=ActionType.PLACE, ladder=EXPLICIT_LADDER)],
    )
    orca = _build_orca(monkeypatch, [plan], job_statuses={"job_B": previous_status})

    def fail_reconcile(job_id, ranks):
        raise RuntimeError("boom")

    orca._launcher.reconcile = fail_reconcile

    assert orca.apply_pending("user_1") == 1
    assert orca._jobs.job_statuses["job_B"] is previous_status
    assert orca._jobs.transitions == [
        ("job_B", JobStatus.RUNNING, [JobStatus.WAITING, JobStatus.PAUSED]),
        ("job_B", previous_status, [JobStatus.RUNNING]),
    ]
    assert orca._jobs.rows[RANK_ID].status is RankStatus.FAILED


def test_launch_failure_emits_rank_failed(monkeypatch):
    plan = Plan(
        user_id="user_1",
        actions=[PlanAction(job_id="job_B", type=ActionType.PLACE, ladder=EXPLICIT_LADDER)],
    )
    orca = _build_orca(monkeypatch, [plan])

    def fail_reconcile(job_id, ranks):
        raise RuntimeError("launcher unavailable")

    orca._launcher.reconcile = fail_reconcile

    assert orca.apply_pending("user_1") == 1

    failed = [event for event in orca._events.events if event.type == "rank.failed"]
    assert len(failed) == 1
    assert failed[0].user_id == "user_1"
    assert failed[0].job_id == "job_B"
    assert failed[0].rank_id == RANK_ID
    assert failed[0].payload_json == {
        "rank_id": RANK_ID,
        "job_id": "job_B",
        "reason_code": ReasonCode.LAUNCH_FAILED,
        "detail": "launcher unavailable",
    }


def test_place_catalog_failure_finishes_job_with_visible_error(monkeypatch):
    plan = Plan(
        user_id="user_1",
        actions=[PlanAction(job_id="job_B", type=ActionType.PLACE, ladder=EXPLICIT_LADDER)],
    )
    orca = _build_orca(monkeypatch, [plan])
    message = "ModelCatalog 'model' field 'max_num_seq' is missing for gpu_type 'L40S'"

    def fail_catalog(*_):
        raise ModelCatalogError(message)

    orca._launcher.reconcile = fail_catalog

    assert orca.apply_pending("user_1") == 1

    assert orca._jobs.job_statuses["job_B"] is JobStatus.FINISHED
    assert orca._jobs.failures == [("job_B", ReasonCode.MODEL_CATALOG_INVALID, message)]
    assert orca._jobs.rows[RANK_ID].reason_code == ReasonCode.MODEL_CATALOG_INVALID
    finished = [event for event in orca._events.events if event.type == "job.finished"]
    assert len(finished) == 1
    assert finished[0].user_id == "user_1"
    assert finished[0].job_id == "job_B"
    assert finished[0].payload_json == {
        "job_id": "job_B",
        "user_id": "user_1",
        "finish_reason": ReasonCode.MODEL_CATALOG_INVALID,
        "detail": message,
    }


def test_swap_catalog_failure_keeps_running_job_and_records_error(monkeypatch):
    old_rank = Rank(
        rank_id=OLD_RANK_ID,
        job_id="job_F",
        plan_id="plan_prev",
        role=RankRole.AGGREGATE,
        status=RankStatus.RUNNING,
        shape_json={"count": 1},
        n_replicas=1,
    )
    plan = Plan(
        user_id="user_1",
        actions=[PlanAction(job_id="job_F", type=ActionType.SWAP, ladder=EXPLICIT_LADDER)],
    )
    orca = _build_orca(monkeypatch, [plan], active=[old_rank])
    message = "ModelCatalog 'model' is missing"

    def fail_catalog(*_):
        raise ModelCatalogError(message)

    orca._launcher.reconcile = fail_catalog

    assert orca.apply_pending("user_1") == 1

    assert orca._jobs.job_statuses["job_F"] is JobStatus.RUNNING
    assert orca._jobs.errors["job_F"] == message
    assert orca._jobs.rows[OLD_RANK_ID] == old_rank
    assert orca._jobs.rows[RANK_ID].reason_code == ReasonCode.MODEL_CATALOG_INVALID


def test_swap_reuses_rank_with_new_replica_count_and_stops_removed_rank(monkeypatch):
    reused_rank = Rank(
        rank_id=RANK_ID,
        job_id="job_F",
        plan_id="plan_prev",
        role=RankRole.AGGREGATE,
        status=RankStatus.RUNNING,
        shape_json={"count": 1},
        n_replicas=1,
    )
    removed_rank = Rank(
        rank_id=OLD_RANK_ID,
        job_id="job_F",
        plan_id="plan_prev",
        role=RankRole.PREFILL,
        status=RankStatus.RUNNING,
        shape_json={"count": 2},
        n_replicas=2,
    )
    plan = Plan(
        user_id="user_1",
        actions=[PlanAction(job_id="job_F", type=ActionType.SWAP, ladder=EXPLICIT_LADDER)],
    )
    orca = _build_orca(monkeypatch, [plan], active=[reused_rank, removed_rank])

    assert orca.apply_pending("user_1") == 1
    assert orca._jobs.launched[0].rank_id == RANK_ID
    assert orca._jobs.launched[0].n_replicas == 3
    assert orca._jobs.rows[RANK_ID].n_replicas == 3
    assert orca._jobs.rows[RANK_ID].status is RankStatus.RUNNING
    assert orca._jobs.rank_status == [
        (OLD_RANK_ID, RankStatus.STOPPED, [RankStatus.LAUNCHING, RankStatus.RUNNING], None),
    ]


def test_swap_drains_removed_and_excess_chains(monkeypatch):
    reused_rank = Rank(
        rank_id=RANK_ID,
        job_id="job_F",
        plan_id="plan_prev",
        role=RankRole.AGGREGATE,
        status=RankStatus.RUNNING,
        shape_json={"count": 1},
        n_replicas=3,
    )
    removed_rank = Rank(
        rank_id=OLD_RANK_ID,
        job_id="job_F",
        plan_id="plan_prev",
        role=RankRole.PREFILL,
        status=RankStatus.RUNNING,
        shape_json={"count": 1},
        n_replicas=2,
    )
    ladder = [{**EXPLICIT_LADDER[0], "n_replicas": 1}]
    plan = Plan(
        user_id="user_1",
        actions=[PlanAction(job_id="job_F", type=ActionType.SWAP, ladder=ladder)],
    )
    chunk_manager = FakeChunkManager()
    orca = _build_orca(
        monkeypatch,
        [plan],
        active=[reused_rank, removed_rank],
        chunk_manager=chunk_manager,
    )

    assert orca.apply_pending("user_1") == 1
    assert chunk_manager.drained == [
        ("job_F", OLD_RANK_ID, 0),
        ("job_F", OLD_RANK_ID, 1),
        ("job_F", RANK_ID, 1),
        ("job_F", RANK_ID, 2),
    ]


def test_swap_without_launchable_rank_keeps_old_rank(monkeypatch):
    old_rank = Rank(
        rank_id=OLD_RANK_ID,
        job_id="job_F",
        plan_id="plan_prev",
        role=RankRole.AGGREGATE,
        status=RankStatus.RUNNING,
        shape_json={"count": 1},
        n_replicas=1,
    )
    plan = Plan(
        user_id="user_1",
        actions=[PlanAction(job_id="job_F", type=ActionType.SWAP, ladder=[])],
    )
    orca = _build_orca(monkeypatch, [plan], active=[old_rank])

    assert orca.apply_pending("user_1") == 1
    assert orca._jobs.rank_status == []


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
    assert orca._jobs.rank_status == []
    assert orca._launcher.torn_down_jobs == []


def test_reconcile_running_restores_active_rank(monkeypatch):
    rank = Rank(
        rank_id=RANK_ID,
        job_id="job_B",
        plan_id="plan_1",
        role=RankRole.AGGREGATE,
        status=RankStatus.RUNNING,
        shape_json={"count": 1},
        n_replicas=1,
    )
    orca = _build_orca(
        monkeypatch,
        [],
        active=[rank],
        job_statuses={"job_B": JobStatus.RUNNING},
    )

    assert orca.reconcile_running("user_1") == 1
    assert orca._launcher.reconciled[0][0] == "job_B"
    assert [restored.rank_id for restored in orca._launcher.reconciled[0][1]] == [RANK_ID]


@pytest.mark.parametrize(
    ("chunk_state", "finish_reason"),
    [
        (chunk_manager_pb2.JOB_STATE_SUCCEEDED, None),
        (chunk_manager_pb2.JOB_STATE_FAILED, ReasonCode.FAILED),
        (chunk_manager_pb2.JOB_STATE_CANCELLED, ReasonCode.CANCELLED),
    ],
)
def test_reconcile_chunk_jobs_finishes_terminal_batch_job(monkeypatch, chunk_state, finish_reason):
    chunk_manager = FakeChunkManager({"job_B": chunk_state})
    orca = _build_orca(
        monkeypatch,
        [],
        job_statuses={"job_B": JobStatus.RUNNING},
        chunk_manager=chunk_manager,
    )

    assert orca.reconcile_chunk_jobs("user_1") == 1
    assert chunk_manager.requested == ["job_B"]
    assert orca._jobs.job_statuses["job_B"] is JobStatus.FINISHED
    assert orca._jobs.finish_reasons["job_B"] == finish_reason


def test_reconcile_chunk_jobs_leaves_running_batch_job_unchanged(monkeypatch):
    chunk_manager = FakeChunkManager({"job_B": chunk_manager_pb2.JOB_STATE_RUNNING})
    orca = _build_orca(
        monkeypatch,
        [],
        job_statuses={"job_B": JobStatus.RUNNING},
        chunk_manager=chunk_manager,
    )

    assert orca.reconcile_chunk_jobs("user_1") == 0
    assert orca._jobs.job_statuses["job_B"] is JobStatus.RUNNING


def test_reconcile_finished_tears_down_active_ranks_once(monkeypatch):
    rank = Rank(
        rank_id=RANK_ID,
        job_id="job_B",
        plan_id="plan_1",
        role=RankRole.AGGREGATE,
        status=RankStatus.RUNNING,
        shape_json={"count": 1},
        n_replicas=1,
    )
    orca = _build_orca(
        monkeypatch,
        [],
        active=[rank],
        job_statuses={"job_B": JobStatus.FINISHED},
    )

    assert orca.reconcile_finished("user_1") == 1
    assert orca.reconcile_finished("user_1") == 0
    assert orca._launcher.torn_down_jobs == ["job_B"]
    assert orca._jobs.rows[RANK_ID].status is RankStatus.STOPPED
    assert orca._jobs.rows[RANK_ID].reason_code is None
    assert orca._jobs.job_statuses["job_B"] is JobStatus.FINISHED
    finished = [event for event in orca._events.events if event.type == "job.finished"]
    assert len(finished) == 1
    assert finished[0].payload_json == {
        "job_id": "job_B",
        "user_id": "user_1",
        "finish_reason": None,
        "detail": None,
    }
    stopped = [event for event in orca._events.events if event.type == "rank.stopped"]
    assert len(stopped) == 1
    assert stopped[0].payload_json == {
        "rank_id": RANK_ID,
        "job_id": "job_B",
        "reason_code": None,
    }


def test_reconcile_finished_cancels_chunk_job_once(monkeypatch):
    chunk_manager = FakeChunkManager()
    orca = _build_orca(
        monkeypatch,
        [],
        job_statuses={"job_B": JobStatus.FINISHED},
        finish_reasons={"job_B": ReasonCode.CANCELLED},
        chunk_manager=chunk_manager,
    )

    assert orca.reconcile_finished("user_1") == 0
    assert orca.reconcile_finished("user_1") == 0
    assert chunk_manager.cancelled == ["job_B"]


def test_preempt_tears_down_and_pauses(monkeypatch):
    rank = Rank(
        rank_id=OLD_RANK_ID,
        job_id="job_E",
        plan_id="plan_prev",
        role=RankRole.AGGREGATE,
        status=RankStatus.RUNNING,
        shape_json={"gpu": "L4", "count": 1},
        n_replicas=1,
    )
    plan = Plan(
        user_id="user_1",
        actions=[PlanAction(job_id="job_E", type=ActionType.PREEMPT)],
    )
    orca = _build_orca(monkeypatch, [plan], active=[rank])

    assert orca.apply_pending("user_1") == 1
    assert orca._launcher.torn_down_jobs == ["job_E"]
    # Preemption records why, so a preempted rank is distinguishable from one
    # that stopped because its job finished.
    assert orca._jobs.rank_status == [
        (
            OLD_RANK_ID,
            RankStatus.STOPPED,
            [RankStatus.LAUNCHING, RankStatus.RUNNING],
            ReasonCode.PREEMPTED,
        ),
    ]
    assert orca._jobs.transitions == [("job_E", JobStatus.PAUSED, [JobStatus.RUNNING])]
    stopped = [event for event in orca._events.events if event.type == "rank.stopped"]
    assert len(stopped) == 1
    assert stopped[0].payload_json == {
        "rank_id": OLD_RANK_ID,
        "job_id": "job_E",
        "reason_code": ReasonCode.PREEMPTED,
    }
    paused = [event for event in orca._events.events if event.type == "job.paused"]
    assert len(paused) == 1
    assert paused[0].user_id == "user_1"
    assert paused[0].job_id == "job_E"
    assert paused[0].payload_json == {
        "job_id": "job_E",
        "user_id": "user_1",
        "plan_id": plan.plan_id,
    }


def test_preempt_drains_each_chain_association(monkeypatch):
    rank = Rank(
        rank_id=OLD_RANK_ID,
        job_id="job_E",
        plan_id="plan_prev",
        role=RankRole.AGGREGATE,
        status=RankStatus.RUNNING,
        shape_json={"gpu": "L4", "count": 1},
        n_replicas=2,
    )
    plan = Plan(
        user_id="user_1",
        actions=[PlanAction(job_id="job_E", type=ActionType.PREEMPT)],
    )
    chunk_manager = FakeChunkManager()
    orca = _build_orca(monkeypatch, [plan], active=[rank], chunk_manager=chunk_manager)

    assert orca.apply_pending("user_1") == 1
    assert chunk_manager.drained == [
        ("job_E", OLD_RANK_ID, 0),
        ("job_E", OLD_RANK_ID, 1),
    ]


def test_bad_action_does_not_wedge_the_plan(monkeypatch):
    """An action that raises is logged and skipped; the plan is still applied."""
    good_ladder = [{**EXPLICIT_LADDER[0], "rank_id": NEW_RANK_ID}]
    plan = Plan(
        user_id="user_1",
        actions=[
            PlanAction(job_id="job_bad", type=ActionType.PLACE, ladder=EXPLICIT_LADDER),
            PlanAction(job_id="job_good", type=ActionType.PLACE, ladder=good_ladder),
        ],
    )
    orca = _build_orca(monkeypatch, [plan])
    original = orca._launcher.reconcile

    def explode_for_bad_job(job_id, ranks):
        if job_id == "job_bad":
            raise RuntimeError("boom")
        original(job_id, ranks)

    orca._launcher.reconcile = explode_for_bad_job

    assert orca.apply_pending("user_1") == 1
    assert orca._plans.applied == [plan.plan_id]
    assert [job for job, _ in orca._launcher.reconciled] == ["job_good"]
    assert orca._jobs.rank_status == [
        (RANK_ID, RankStatus.FAILED, [RankStatus.LAUNCHING], ReasonCode.LAUNCH_FAILED),
    ]
    assert orca._jobs.rows[NEW_RANK_ID].status is RankStatus.LAUNCHING


def test_apply_pending_no_plans(monkeypatch):
    orca = _build_orca(monkeypatch, [])
    assert orca.apply_pending("user_1") == 0
    assert orca._jobs.transitions == []


def test_default_launcher_is_noop(monkeypatch):
    monkeypatch.setattr(orca_mod, "JobStore", lambda client: FakeJobStore())
    monkeypatch.setattr(orca_mod, "PlanStore", lambda client: FakePlanStore([]))
    orca = Orca(client=object())
    assert isinstance(orca._launcher, orca_mod.NoopLauncher)

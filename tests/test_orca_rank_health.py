"""Orca's DGD-status rank health reconciler.

No Kubernetes and no Postgres: a fake cluster client serves DGD documents and a
fake JobStore records every status write, so the debounce, the promotion guard,
and the failure-cause lookup can be asserted directly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from tandemn_system_data.models.enums import RankRole, RankStatus, ReasonCode
from tandemn_system_data.models.rank import Rank

import tandemn_orca.orca as orca_mod
from tandemn_orca.orca import Orca
from tandemn_orca.rank_health import Verdict

RANK_ID = "rank_01JBM2YQYZ1KQ9C8GZP1XB6V5T"
JOB_ID = "job-01JBM2YQYZ1KQ9C8GZP1XB6V5T"
USER_ID = "usr_test"


def _rank(status: RankStatus = RankStatus.RUNNING) -> Rank:
    return Rank(
        rank_id=RANK_ID,
        job_id=JOB_ID,
        plan_id="plan-1",
        role=RankRole.AGGREGATE,
        shape_json={"env": ["reserved", "aws", "us-east-1", "use1-az1", "L40S"]},
        n_replicas=2,
        status=status,
    )


def _dgd(worker: int, *, state: str = "successful", router: int = 1) -> dict:
    def service(count: int) -> dict:
        return {"componentKind": "Deployment", "replicas": count, "readyReplicas": count}

    return {
        "metadata": {
            "name": "tdm-abc",
            "generation": 1,
            "labels": {"tandemn.com/rank-id": RANK_ID},
        },
        "status": {
            "state": state,
            "observedGeneration": 1,
            "services": {"VllmDecodeWorker": service(worker), "LocalRouter": service(router)},
        },
    }


class FakeJobStore:
    def __init__(self, rank: Rank) -> None:
        self.rank = rank
        self.writes: list[tuple[str, RankStatus, list[RankStatus], str | None]] = []

    def running_jobs(self, user_id):
        return [SimpleNamespace(job=SimpleNamespace(job_id=JOB_ID), ranks=[self.rank])]

    def set_rank_status(self, rank_id, to, expected, *, reason_code=None):
        self.writes.append((rank_id, to, list(expected), reason_code))
        if self.rank.status not in expected:
            return False
        self.rank = self.rank.model_copy(update={"status": to, "reason_code": reason_code})
        return True


class FakeK8s:
    def __init__(self, dgds: list[dict], pods: list[dict] | None = None) -> None:
        self.dgds = dgds
        self.pods = pods or []
        self.pod_lookups = 0

    def job_dgds(self, job_id):
        return self.dgds

    def rank_pods(self, job_id, rank_id):
        self.pod_lookups += 1
        return self.pods


class FakeLauncher:
    def __init__(self, k8s: FakeK8s) -> None:
        self.k8s = k8s

    def k8s_for_rank(self, rank):
        return self.k8s

    def reconcile(self, job_id, ranks):  # pragma: no cover - unused here
        pass

    def teardown_job(self, job_id):  # pragma: no cover - unused here
        pass


def _orca(monkeypatch, store: FakeJobStore, k8s: FakeK8s, *, polls: int = 2) -> Orca:
    monkeypatch.setattr(orca_mod, "JobStore", lambda client: store)
    monkeypatch.setattr(orca_mod, "PlanStore", lambda client: object())
    return Orca(client=object(), launcher=FakeLauncher(k8s), down_polls_before_failed=polls)


# ----- promotion -------------------------------------------------------------


def test_serving_promotes_a_launching_rank(monkeypatch):
    store = FakeJobStore(_rank(RankStatus.LAUNCHING))
    orca = _orca(monkeypatch, store, FakeK8s([_dgd(worker=2)]))

    health = orca.reconcile_rank_health(USER_ID)

    assert [item.verdict for item in health] == [Verdict.SERVING]
    assert store.writes == [(RANK_ID, RankStatus.RUNNING, [RankStatus.LAUNCHING], None)]


def test_serving_does_not_rewrite_an_already_running_rank(monkeypatch):
    """set_rank_status writes reason_code and updated_at unconditionally, so a
    no-op CAS every poll would erase failure provenance."""
    store = FakeJobStore(_rank(RankStatus.RUNNING))
    orca = _orca(monkeypatch, store, FakeK8s([_dgd(worker=2)]))

    orca.reconcile_rank_health(USER_ID)
    orca.reconcile_rank_health(USER_ID)

    assert store.writes == []


# ----- debounce --------------------------------------------------------------


def test_one_down_poll_reports_no_opinion_and_writes_nothing(monkeypatch):
    store = FakeJobStore(_rank())
    orca = _orca(monkeypatch, store, FakeK8s([_dgd(worker=0)]))

    health = orca.reconcile_rank_health(USER_ID)

    assert health[0].verdict is Verdict.UNKNOWN
    assert "1/2" in health[0].detail
    assert store.writes == []


def test_consecutive_down_polls_fail_the_rank(monkeypatch):
    store = FakeJobStore(_rank())
    orca = _orca(monkeypatch, store, FakeK8s([_dgd(worker=0)]))

    orca.reconcile_rank_health(USER_ID)
    health = orca.reconcile_rank_health(USER_ID)

    assert health[0].verdict is Verdict.DOWN
    assert store.writes == [
        (
            RANK_ID,
            RankStatus.FAILED,
            [RankStatus.LAUNCHING, RankStatus.RUNNING],
            ReasonCode.HEARTBEAT_TIMEOUT,
        )
    ]


def test_a_recovered_rank_resets_the_streak(monkeypatch):
    store = FakeJobStore(_rank())
    k8s = FakeK8s([_dgd(worker=0)])
    orca = _orca(monkeypatch, store, k8s)

    orca.reconcile_rank_health(USER_ID)
    k8s.dgds = [_dgd(worker=2)]
    orca.reconcile_rank_health(USER_ID)
    k8s.dgds = [_dgd(worker=0)]
    orca.reconcile_rank_health(USER_ID)

    assert store.writes == []


# ----- launching vs died -----------------------------------------------------


def test_zero_replicas_while_pending_never_fails_a_new_rank(monkeypatch):
    """Karpenter plus image pull plus model load runs 8-12 minutes."""
    store = FakeJobStore(_rank(RankStatus.LAUNCHING))
    orca = _orca(monkeypatch, store, FakeK8s([_dgd(worker=0, state="pending")]))

    for _ in range(5):
        health = orca.reconcile_rank_health(USER_ID)

    assert health[0].verdict is Verdict.UNKNOWN
    assert store.writes == []


def test_zero_replicas_after_serving_fails_even_while_pending(monkeypatch):
    store = FakeJobStore(_rank())
    k8s = FakeK8s([_dgd(worker=2)])
    orca = _orca(monkeypatch, store, k8s)

    orca.reconcile_rank_health(USER_ID)
    k8s.dgds = [_dgd(worker=0, state="pending")]
    orca.reconcile_rank_health(USER_ID)
    orca.reconcile_rank_health(USER_ID)

    assert store.writes[-1][1] is RankStatus.FAILED


# ----- failure cause ---------------------------------------------------------


def test_pod_termination_reason_beats_the_generic_code(monkeypatch):
    pods = [
        {
            "metadata": {"name": "worker-0"},
            "status": {
                "containerStatuses": [
                    {"name": "main", "lastState": {"terminated": {"reason": "OOMKilled"}}}
                ]
            },
        }
    ]
    store = FakeJobStore(_rank())
    k8s = FakeK8s([_dgd(worker=0)], pods)
    orca = _orca(monkeypatch, store, k8s)

    orca.reconcile_rank_health(USER_ID)
    orca.reconcile_rank_health(USER_ID)

    assert store.writes[-1][3] == ReasonCode.OOM
    # Pods are read once, on the failing transition, not on every poll.
    assert k8s.pod_lookups == 1


def test_pod_lookup_failure_falls_back_to_the_dgd_reason(monkeypatch):
    class ExplodingK8s(FakeK8s):
        def rank_pods(self, job_id, rank_id):
            raise RuntimeError("api server unreachable")

    store = FakeJobStore(_rank())
    orca = _orca(monkeypatch, store, ExplodingK8s([_dgd(worker=0)]))

    orca.reconcile_rank_health(USER_ID)
    orca.reconcile_rank_health(USER_ID)

    assert store.writes[-1][3] == ReasonCode.HEARTBEAT_TIMEOUT


# ----- degraded environments -------------------------------------------------


def test_noop_launcher_reconciles_nothing(monkeypatch):
    store = FakeJobStore(_rank())
    monkeypatch.setattr(orca_mod, "JobStore", lambda client: store)
    monkeypatch.setattr(orca_mod, "PlanStore", lambda client: object())

    assert Orca(client=object()).reconcile_rank_health(USER_ID) == []
    assert store.writes == []


def test_a_failed_dgd_read_does_not_abort_the_pass(monkeypatch):
    class ExplodingK8s(FakeK8s):
        def job_dgds(self, job_id):
            raise RuntimeError("api server unreachable")

    store = FakeJobStore(_rank())
    orca = _orca(monkeypatch, store, ExplodingK8s([]))

    assert orca.reconcile_rank_health(USER_ID) == []
    assert store.writes == []


@pytest.mark.parametrize("polls", [0, -3])
def test_debounce_threshold_is_at_least_one_poll(monkeypatch, polls):
    store = FakeJobStore(_rank())
    orca = _orca(monkeypatch, store, FakeK8s([_dgd(worker=0)]), polls=polls)

    orca.reconcile_rank_health(USER_ID)

    assert store.writes[-1][1] is RankStatus.FAILED

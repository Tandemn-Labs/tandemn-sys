"""Parsing DGD status into a rank serving verdict.

The dispatch tests matter more than they look: the operator omits the replica
field it does not use rather than zeroing it, so reading the wrong one reports
a healthy rank as down.
"""

from __future__ import annotations

from tandemn_system_data.models.enums import ReasonCode

from tandemn_orca.rank_health import (
    Verdict,
    batch_rank_health,
    dgd_by_rank_id,
    rank_health,
    serving_replicas,
    termination_reason_code,
)

RANK_ID = "rank_01JBM2YQYZ1KQ9C8GZP1XB6V5T"
JOB_ID = "job-01JBM2YQYZ1KQ9C8GZP1XB6V5T"


def _service(kind: str, count: int) -> dict:
    field = "availableReplicas" if kind == "PodCliqueScalingGroup" else "readyReplicas"
    return {"componentKind": kind, "replicas": count, field: count}


def _dgd(
    *,
    state: str = "successful",
    worker: int | None = 2,
    worker_kind: str = "Deployment",
    router: int | None = 1,
    generation: int = 4,
    observed: int | None = 4,
    conditions: list[dict] | None = None,
    status_key: str = "services",
) -> dict:
    services = {}
    if worker is not None:
        services["VllmDecodeWorker"] = _service(worker_kind, worker)
    if router is not None:
        services["LocalRouter"] = _service("Deployment", router)
    status: dict = {"state": state, status_key: services}
    if observed is not None:
        status["observedGeneration"] = observed
    if conditions is not None:
        status["conditions"] = conditions
    return {
        "metadata": {
            "name": "tdm-abc",
            "generation": generation,
            "labels": {"tandemn.com/rank-id": RANK_ID},
        },
        "status": status,
    }


def _pod(chain: int, *, ready: bool = False, failed: bool = False) -> dict:
    return {
        "metadata": {
            "labels": {
                "tandemn.com/chain-id": str(chain),
            }
        },
        "status": {
            "phase": "Failed" if failed else "Running",
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
        },
    }


# ----- replica field dispatch ------------------------------------------------


def test_deployment_reads_ready_replicas():
    assert serving_replicas(_service("Deployment", 3)) == 3


def test_pod_clique_reads_ready_replicas():
    assert serving_replicas(_service("PodClique", 2)) == 2


def test_leader_worker_set_reads_ready_replicas():
    assert serving_replicas(_service("LeaderWorkerSet", 1)) == 1


def test_scaling_group_reads_available_replicas():
    """Multinode Grove: readyReplicas is absent, not zero."""
    status = _service("PodCliqueScalingGroup", 2)
    assert "readyReplicas" not in status
    assert serving_replicas(status) == 2


def test_scaling_group_is_not_read_as_zero_by_the_ready_field():
    health = rank_health(JOB_ID, RANK_ID, _dgd(worker=2, worker_kind="PodCliqueScalingGroup"))
    assert health.verdict is Verdict.SERVING
    assert health.serving_replicas == 2


def test_beta_components_read_ready_replicas():
    health = rank_health(JOB_ID, RANK_ID, _dgd(worker=2, status_key="components"))

    assert health.verdict is Verdict.SERVING
    assert health.serving_replicas == 2


def test_unknown_component_kind_yields_no_count():
    assert serving_replicas({"componentKind": "StatefulSet", "readyReplicas": 4}) is None


def test_absent_replica_field_yields_no_count():
    assert serving_replicas({"componentKind": "Deployment"}) is None


# ----- verdicts --------------------------------------------------------------


def test_serving_when_worker_and_router_are_up():
    health = rank_health(JOB_ID, RANK_ID, _dgd(worker=3, router=1))
    assert health.verdict is Verdict.SERVING
    assert health.serving_replicas == 3
    assert health.reason_code is None


def test_missing_deployment_is_down():
    health = rank_health(JOB_ID, RANK_ID, None)
    assert health.verdict is Verdict.DOWN
    assert health.reason_code == ReasonCode.NODE_LOST


def test_failed_state_reports_the_failing_condition():
    dgd = _dgd(
        state="failed",
        conditions=[
            {"type": "Ready", "status": "False", "reason": "ImagePullFailure", "message": "no auth"}
        ],
    )
    health = rank_health(JOB_ID, RANK_ID, dgd)
    assert health.verdict is Verdict.DOWN
    assert health.reason_code == ReasonCode.FAILED
    assert "ImagePullFailure" in health.detail


def test_unreconciled_generation_is_unknown():
    """Replica counts still describe the spec before Orca's patch."""
    health = rank_health(JOB_ID, RANK_ID, _dgd(generation=9, observed=8, worker=0))
    assert health.verdict is Verdict.UNKNOWN


def test_zero_workers_while_launching_is_unknown():
    health = rank_health(JOB_ID, RANK_ID, _dgd(state="pending", worker=0), ever_served=False)
    assert health.verdict is Verdict.UNKNOWN


def test_zero_workers_after_serving_is_down():
    health = rank_health(JOB_ID, RANK_ID, _dgd(state="pending", worker=0), ever_served=True)
    assert health.verdict is Verdict.DOWN
    assert health.reason_code == ReasonCode.HEARTBEAT_TIMEOUT


def test_zero_workers_with_successful_state_is_down_even_before_serving():
    health = rank_health(JOB_ID, RANK_ID, _dgd(state="successful", worker=0), ever_served=False)
    assert health.verdict is Verdict.DOWN


def test_missing_worker_service_while_launching_is_unknown():
    health = rank_health(JOB_ID, RANK_ID, _dgd(state="initializing", worker=None))
    assert health.verdict is Verdict.UNKNOWN


def test_missing_worker_service_after_serving_is_down():
    health = rank_health(JOB_ID, RANK_ID, _dgd(worker=None), ever_served=True)
    assert health.verdict is Verdict.DOWN


def test_local_router_still_starting_is_unknown():
    """Orca inserts the LocalRouter hop, so ready workers alone serve nothing."""
    health = rank_health(JOB_ID, RANK_ID, _dgd(worker=4, router=0))
    assert health.verdict is Verdict.UNKNOWN
    assert health.serving_replicas is None


def test_dead_local_router_is_down_after_rank_served():
    health = rank_health(JOB_ID, RANK_ID, _dgd(worker=4, router=0), ever_served=True)
    assert health.verdict is Verdict.DOWN
    assert health.reason_code == ReasonCode.PROCESS_CRASH
    assert health.serving_replicas == 4


def test_absent_local_router_entry_does_not_veto():
    health = rank_health(JOB_ID, RANK_ID, _dgd(worker=1, router=None))
    assert health.verdict is Verdict.SERVING


# ----- batch pods -------------------------------------------------------------


def test_batch_counts_only_fully_ready_chains():
    health = batch_rank_health(
        JOB_ID,
        RANK_ID,
        [_pod(0, ready=True), _pod(0, ready=True), _pod(1, ready=True), _pod(1)],
        expected_replicas=2,
        nodes_per_chain=2,
    )

    assert health.verdict is Verdict.SERVING
    assert health.serving_replicas == 1


def test_batch_keeps_chains_from_prior_plan():
    health = batch_rank_health(
        JOB_ID,
        RANK_ID,
        [_pod(0, ready=True)],
        expected_replicas=1,
        nodes_per_chain=1,
    )

    assert health.verdict is Verdict.SERVING


def test_batch_all_failed_is_down():
    health = batch_rank_health(
        JOB_ID,
        RANK_ID,
        [_pod(0, failed=True), _pod(1, failed=True)],
        expected_replicas=2,
        nodes_per_chain=1,
    )

    assert health.verdict is Verdict.DOWN
    assert health.reason_code == ReasonCode.PROCESS_CRASH


def test_batch_zero_ready_after_serving_is_down():
    health = batch_rank_health(
        JOB_ID,
        RANK_ID,
        [_pod(0)],
        expected_replicas=1,
        nodes_per_chain=1,
        ever_served=True,
    )

    assert health.verdict is Verdict.DOWN
    assert health.reason_code == ReasonCode.HEARTBEAT_TIMEOUT


# ----- helpers ---------------------------------------------------------------


def test_dgd_by_rank_id_indexes_on_the_label():
    indexed = dgd_by_rank_id([_dgd(), {"metadata": {"name": "unlabeled"}}])
    assert set(indexed) == {RANK_ID}


def test_termination_reason_maps_oom_kill():
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
    assert termination_reason_code(pods) == (
        ReasonCode.OOM,
        "worker-0/main terminated: OOMKilled",
    )


def test_termination_reason_maps_crash_loop():
    pods = [
        {
            "metadata": {"name": "worker-1"},
            "status": {
                "containerStatuses": [
                    {"name": "main", "state": {"waiting": {"reason": "CrashLoopBackOff"}}}
                ]
            },
        }
    ]
    code, detail = termination_reason_code(pods)
    assert code == ReasonCode.PROCESS_CRASH
    assert "CrashLoopBackOff" in detail


def test_termination_reason_absent_for_healthy_pods():
    assert termination_reason_code([{"metadata": {"name": "w"}, "status": {}}]) is None

"""Derive rank serving health from DGD status or batch worker pods.

Why the DGD is the source of truth for "is this rank up":

The operator configures each worker container's readiness probe as an HTTP GET
against the worker's own ``/health`` on ``DYN_SYSTEM_PORT``, with
``DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS=["generate"]``. That endpoint answers
503 until the generate endpoint is actually served and 200 afterwards, so a
replica counted ready by kubelet is one whose serving path works -- not merely a
live process. The operator aggregates those counts onto ``.status.components``,
which makes the DGD a strictly stronger signal than either the Prometheus
scrape (which lags service discovery on cold start) or the frontend's own
``/health`` (whose instance list is a lease-based etcd registration that
outlives an OOMKill).

Two traps this module exists to avoid:

1. The replica count lives in a different field per workload kind, and the
   unused field is omitted from the JSON rather than zeroed. Reading
   ``readyReplicas`` on a PodCliqueScalingGroup returns None, which a naive
   ``or 0`` turns into "zero ready" for a perfectly healthy rank.
2. A rank is only usable when both its worker *and* its LocalRouter are up --
   Orca inserts the LocalRouter hop, so workers alone serving nothing is a
   reachable state the operator reports as partially available.

Version note: Dynamo 1.4.2 serves ``nvidia.com/v1beta1`` with component status
in ``.status.components``. ``.status.services`` remains a read-only fallback
for existing alpha objects during migration.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from tandemn_system_data.models.enums import ReasonCode

logger = logging.getLogger(__name__)

# Service keys rendered by dynamo_compiler.render_rank_dgd. The operator copies
# their names verbatim into status.components.
WORKER_SERVICE = "VllmDecodeWorker"
ROUTER_SERVICE = "LocalRouter"

# Which .status.components[*] field carries the ready count, per workload kind the
# operator chose. The other field is absent from the JSON, never zero, so this
# must be a dispatch and not a `ready or available` fallback.
#   Deployment           readyReplicas    counts pods
#   PodClique            readyReplicas    counts pods
#   LeaderWorkerSet      readyReplicas    counts groups
#   PodCliqueScalingGroup availableReplicas counts gangs
SERVING_REPLICA_FIELD = {
    "Deployment": "readyReplicas",
    "PodClique": "readyReplicas",
    "LeaderWorkerSet": "readyReplicas",
    "PodCliqueScalingGroup": "availableReplicas",
}

# DGD lifecycle states.
STATE_FAILED = "failed"
STATE_SUCCESSFUL = "successful"
PENDING_STATES = frozenset({"initializing", "pending", ""})

# Container termination reasons worth translating into a rank reason code.
_TERMINATION_REASONS = {
    "OOMKilled": ReasonCode.OOM,
    "Error": ReasonCode.PROCESS_CRASH,
    "ContainerCannotRun": ReasonCode.PROCESS_CRASH,
    "DeadlineExceeded": ReasonCode.PROCESS_CRASH,
}


class Verdict(StrEnum):
    """Serving health of one rank as read from its DGD."""

    SERVING = "serving"
    DOWN = "down"
    # No conclusion: still launching, or the operator has not yet reconciled the
    # generation Orca applied. Callers must not fail a rank on UNKNOWN.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RankHealth:
    """One rank's serving verdict and the evidence behind it."""

    rank_id: str
    job_id: str
    verdict: Verdict
    serving_replicas: int | None
    reason_code: str | None
    detail: str


def rank_health(
    job_id: str,
    rank_id: str,
    dgd: Mapping[str, Any] | None,
    *,
    ever_served: bool = False,
) -> RankHealth:
    """Read one rank's serving health from its DynamoGraphDeployment.

    ``ever_served`` distinguishes the two ways a rank can report zero ready
    replicas. Before a rank has ever served, zero means "still coming up" --
    Karpenter provisioning plus image pull plus model load runs 8-12 minutes,
    and the operator holds the DGD in ``pending`` throughout. After a rank has
    served, zero means it died, and the operator recomputing ``pending`` must
    not be read as launching again.
    """
    if dgd is None:
        return RankHealth(
            rank_id,
            job_id,
            Verdict.DOWN,
            0,
            ReasonCode.NODE_LOST,
            "no DynamoGraphDeployment carries this rank's label",
        )

    metadata = dgd.get("metadata") or {}
    status = dgd.get("status") or {}
    state = str(status.get("state") or "")

    # The operator has not yet processed the spec Orca applied, so the replica
    # counts below still describe the previous generation.
    observed = status.get("observedGeneration")
    generation = metadata.get("generation")
    if isinstance(observed, int) and isinstance(generation, int) and observed < generation:
        return _unknown(job_id, rank_id, f"operator has not reconciled generation {generation}")

    if state == STATE_FAILED:
        return RankHealth(
            rank_id,
            job_id,
            Verdict.DOWN,
            0,
            ReasonCode.FAILED,
            f"DGD state=failed: {_condition_detail(status)}",
        )

    components = status.get("components")
    if not isinstance(components, dict):
        components = status.get("services")
    if not isinstance(components, dict) or WORKER_SERVICE not in components:
        if ever_served:
            return RankHealth(
                rank_id,
                job_id,
                Verdict.DOWN,
                0,
                ReasonCode.HEARTBEAT_TIMEOUT,
                "worker service vanished from DGD status",
            )
        return _unknown(job_id, rank_id, f"DGD state={state or 'unset'}, no worker status yet")

    workers = serving_replicas(components[WORKER_SERVICE])
    if workers is None:
        return _unknown(job_id, rank_id, "worker service reports no recognized replica field")
    if workers == 0:
        if not ever_served and state in PENDING_STATES:
            return _unknown(job_id, rank_id, f"no worker ready yet, DGD state={state or 'unset'}")
        return RankHealth(
            rank_id,
            job_id,
            Verdict.DOWN,
            0,
            ReasonCode.HEARTBEAT_TIMEOUT,
            f"0 worker replicas ready (DGD state={state or 'unset'})",
        )

    # Orca inserts a LocalRouter between the frontend and the workers, so ready
    # workers alone do not make the rank routable.
    router_status = components.get(ROUTER_SERVICE)
    if router_status is not None:
        routers = serving_replicas(router_status)
        if routers == 0:
            if not ever_served:
                return _unknown(
                    job_id,
                    rank_id,
                    f"{workers} worker replica(s) ready; LocalRouter is still starting",
                )
            return RankHealth(
                rank_id,
                job_id,
                Verdict.DOWN,
                workers,
                ReasonCode.PROCESS_CRASH,
                f"{workers} worker replica(s) ready but LocalRouter has none",
            )

    return RankHealth(rank_id, job_id, Verdict.SERVING, workers, None, "")


def batch_rank_health(
    job_id: str,
    rank_id: str,
    pods: list[dict[str, Any]],
    *,
    expected_replicas: int,
    nodes_per_chain: int,
    ever_served: bool = False,
) -> RankHealth:
    """Count batch chains whose pods are all ready."""
    chains: dict[str, list[dict[str, Any]]] = {}
    for pod in pods:
        metadata = pod.get("metadata") or {}
        pod_labels = metadata.get("labels") or {}
        chain_id = pod_labels.get("tandemn.com/chain-id")
        if chain_id is not None:
            chains.setdefault(str(chain_id), []).append(pod)

    ready = sum(
        len([pod for pod in members if _pod_ready(pod)]) >= nodes_per_chain
        for members in chains.values()
    )
    if ready:
        return RankHealth(rank_id, job_id, Verdict.SERVING, ready, None, "")
    if not chains:
        if ever_served:
            return RankHealth(
                rank_id,
                job_id,
                Verdict.DOWN,
                0,
                ReasonCode.NODE_LOST,
                "no batch worker pods carry this rank and plan",
            )
        return _unknown(job_id, rank_id, "no batch worker pods ready yet")

    failed = sum(any(_pod_failed(pod) for pod in members) for members in chains.values())
    if failed >= expected_replicas:
        return RankHealth(
            rank_id,
            job_id,
            Verdict.DOWN,
            0,
            ReasonCode.PROCESS_CRASH,
            f"all {failed} batch chain(s) failed",
        )
    if ever_served:
        return RankHealth(
            rank_id,
            job_id,
            Verdict.DOWN,
            0,
            ReasonCode.HEARTBEAT_TIMEOUT,
            "0 batch chains ready",
        )
    return _unknown(job_id, rank_id, "batch workers are still starting")


def serving_replicas(service_status: Mapping[str, Any]) -> int | None:
    """Ready replica count for one service, or None when it cannot be read."""
    kind = service_status.get("componentKind")
    field = SERVING_REPLICA_FIELD.get(str(kind))
    if field is None:
        logger.warning("unrecognized componentKind %r in DGD status", kind)
        return None
    value = service_status.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return max(0, value)


def dgd_by_rank_id(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index DGDs by the rank they carry, dropping unlabeled objects."""
    indexed: dict[str, dict[str, Any]] = {}
    for item in items:
        labels = (item.get("metadata") or {}).get("labels") or {}
        rank_id = labels.get("tandemn.com/rank-id")
        if rank_id:
            indexed[str(rank_id)] = item
    return indexed


def _pod_ready(pod: Mapping[str, Any]) -> bool:
    conditions = (pod.get("status") or {}).get("conditions") or []
    return any(
        condition.get("type") == "Ready" and condition.get("status") == "True"
        for condition in conditions
        if isinstance(condition, dict)
    )


def _pod_failed(pod: Mapping[str, Any]) -> bool:
    status = pod.get("status") or {}
    if status.get("phase") == "Failed":
        return True
    return any(
        ((container.get("state") or {}).get("waiting") or {}).get("reason") == "CrashLoopBackOff"
        for container in status.get("containerStatuses") or []
        if isinstance(container, dict)
    )


def termination_reason_code(pods: list[dict[str, Any]]) -> tuple[str, str] | None:
    """Translate the newest container termination into (reason_code, detail).

    Only the DGD's replica counts say a rank is down; they never say why. Pod
    termination state is the one place the real cause surfaces, so callers look
    it up on the failing transition rather than on every poll.
    """
    for pod in pods:
        statuses = ((pod.get("status") or {}).get("containerStatuses")) or []
        for container in statuses:
            terminated = (container.get("lastState") or {}).get("terminated") or {}
            reason = terminated.get("reason")
            code = _TERMINATION_REASONS.get(str(reason))
            if code is not None:
                name = (pod.get("metadata") or {}).get("name", "?")
                return code, f"{name}/{container.get('name', '?')} terminated: {reason}"
            waiting = (container.get("state") or {}).get("waiting") or {}
            if str(waiting.get("reason")) == "CrashLoopBackOff":
                name = (pod.get("metadata") or {}).get("name", "?")
                return ReasonCode.PROCESS_CRASH, f"{name} in CrashLoopBackOff"
    return None


def _unknown(job_id: str, rank_id: str, detail: str) -> RankHealth:
    return RankHealth(rank_id, job_id, Verdict.UNKNOWN, None, None, detail)


def _condition_detail(status: Mapping[str, Any]) -> str:
    """Summarize the most informative non-True condition, if any."""
    conditions = status.get("conditions")
    if not isinstance(conditions, list):
        return "no conditions reported"
    for condition in conditions:
        if not isinstance(condition, dict) or condition.get("status") == "True":
            continue
        parts = [
            str(condition.get(key)) for key in ("type", "reason", "message") if condition.get(key)
        ]
        if parts:
            return " / ".join(parts)
    return "no failing condition reported"

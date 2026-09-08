"""Orca apply-plan loop.

One pass: poll the plans Koi has created but not yet applied, apply each
action, and CAS the plan to ``applied``.

Action semantics (DATA_ARCHITECTURE.md §6):
    place    waiting|paused -> running   record the ladder's ranks + apply DGDs
    keep     running                     no change
    defer    waiting                     no change
    preempt  running -> paused           tear down the job's ranks
    swap     running                     relaunch on a new ladder

Rank rows are *authorized* capacity, not launched pods: the pool DGD's worker
replicas are owned by Dynamo's scaling adapter (DGDSA), and the in-pool Planner
scales DP width within [min_endpoint, max_gpu_budget]. Actual live width per
rank = distinct chain indexes in recent gpu_metrics rows.

Ladder shape (opaque JSONB; Koi <-> Orca contract):
    [{"role": "aggregate", "rank_id": "rank_<ULID>", "env": [...], "config": {...},
      "n_replicas": 3}]
``count`` / ``gpu_count`` is the GPU count per replica (required, positive int).
``chains`` / ``n_replicas`` is the rank's maximum runtime replica count.
``rank_id`` is the canonical Koi-supplied ``rank_<ULID>`` preserved into DGDs and pods.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import time
from dataclasses import replace
from typing import Any

from tandemn_system_data.clients import JobStore, ModelCatalogStore, PlanStore, PostgresClient
from tandemn_system_data.clients.event_log import PostgresEventLog
from tandemn_system_data.events import (
    JobFinishedPayload,
    JobPausedPayload,
    JobPlacedPayload,
    JobPlacePayload,
    JobResumedPayload,
    PlanAppliedPayload,
    RankFailedPayload,
    RankLaunchingPayload,
    RankRunningPayload,
    RankStoppedPayload,
)
from tandemn_system_data.models.enums import (
    ActionType,
    JobKind,
    JobStatus,
    RankRole,
    RankStatus,
    ReasonCode,
)
from tandemn_system_data.models.event import Event
from tandemn_system_data.models.plan import Plan, PlanAction
from tandemn_system_data.models.rank import Rank

from tandemn.chunkmanager.v1 import chunk_manager_pb2
from tandemn_orca.chunk_manager import ChunkManagerClient
from tandemn_orca.compiler_common import rank_node_count
from tandemn_orca.dynamo_kubernetes import load_kube_client
from tandemn_orca.launcher import (
    DynamoLauncher,
    Launcher,
    ModelCatalogError,
    MultiClusterLauncher,
    NoopLauncher,
)
from tandemn_orca.rank_health import (
    RankHealth,
    Verdict,
    batch_rank_health,
    dgd_by_rank_id,
    rank_health,
    termination_reason_code,
)
from tandemn_orca.router_health import RankHealthPublisher
from tandemn_orca.tunnels import PortForwardManager, RouterProcessManager

# AWS capacity refresh is disabled pending the GCP ResourceMap refresher.
logger = logging.getLogger(__name__)

ACTIVE_RANK_STATUSES = (RankStatus.LAUNCHING, RankStatus.RUNNING)
RANK_ID_RE = re.compile(r"rank_[0-7][0-9A-HJKMNP-TV-Z]{25}\Z")


def ladder_to_ranks(
    ladder: list[dict[str, Any]] | None,
    *,
    job_id: str,
    plan_id: str,
    target_p99_ttft_ms: float | None = None,
    target_p99_tpot_ms: float | None = None,
    job_spec: dict[str, Any] | None = None,
) -> list[Rank]:
    """Translate a ladder into Rank rows, skipping malformed entries.

    A missing/invalid role, config, replica count, GPU count, or instance type
    skips the entry. Koi keeps some launch fields outside the rank config, so
    gaps are backfilled here: ``gpu_type`` from ``env[4]``, ``model_id`` from
    the job's ``spec_json``, ``engine_name`` defaulting to ``vllm``.
    """
    job_spec = job_spec or {}
    ranks: list[Rank] = []
    seen_rank_ids: set[str] = set()
    for entry in ladder or []:
        if not isinstance(entry, dict):
            logger.warning("skipping non-dict ladder entry: %r", entry)
            continue

        role = _parse_role(entry.get("role"))
        config = entry.get("config")
        if role is None or not isinstance(config, dict):
            logger.warning("skipping malformed ladder entry: %r", entry)
            continue
        count = config.get("gpu_count", config.get("count"))
        replicas = entry.get("n_replicas", entry.get("chains", 1))
        if type(count) is not int or count <= 0:
            logger.warning("skipping ladder entry without positive int gpu_count: %r", entry)
            continue
        if type(replicas) is not int or replicas <= 0:
            logger.warning("skipping ladder entry without positive int n_replicas: %r", entry)
            continue
        shape_json = dict(config)
        shape_json["count"] = count
        shape_json.setdefault("sp", 1)
        shape_json.setdefault("ep", 1)
        shape_json.setdefault("cp", 1)
        rank_id = entry.get("rank_id")
        if not isinstance(rank_id, str) or RANK_ID_RE.fullmatch(rank_id) is None:
            logger.warning("skipping ladder entry without canonical rank_<ULID>: %r", entry)
            continue
        if rank_id in seen_rank_ids:
            raise ValueError(f"duplicate rank_id {rank_id}")
        seen_rank_ids.add(rank_id)
        env = entry.get("env")
        if env is not None:
            shape_json["env"] = list(env) if isinstance(env, (list, tuple)) else env
        if entry.get("mechanism_id") is not None:
            shape_json["mechanism_id"] = entry["mechanism_id"]
        for key in ("predicted_y", "predicted_v"):
            if isinstance(entry.get(key), dict):
                shape_json[key] = dict(entry[key])
        if target_p99_ttft_ms is not None:
            shape_json["target_p99_ttft_ms"] = target_p99_ttft_ms
        if target_p99_tpot_ms is not None:
            shape_json["target_p99_tpot_ms"] = target_p99_tpot_ms

        # Backfill launch fields the compiler requires but Koi keeps elsewhere.
        if not shape_json.get("gpu_type") and isinstance(env, (list, tuple)) and len(env) >= 5:
            shape_json["gpu_type"] = str(env[4])  # env = (market, cloud, region, zone, gpu_type)
        if not shape_json.get("model_id") and job_spec.get("model_id"):
            shape_json["model_id"] = str(job_spec["model_id"])
        shape_json.setdefault("engine_name", "vllm")
        missing = [
            key for key in ("instance_type", "gpu_type", "model_id") if not shape_json.get(key)
        ]
        if missing:
            logger.warning("skipping unlaunchable ladder entry (missing %s): %r", missing, entry)
            continue

        ranks.append(
            Rank(
                rank_id=rank_id,
                job_id=job_id,
                plan_id=plan_id,
                role=role,
                shape_json=shape_json,
                n_replicas=replicas,
            )
        )
    return ranks


def _parse_role(role_name: object) -> RankRole | None:
    try:
        return RankRole(role_name)
    except ValueError:
        return None


class Orca:
    """One instance per process, sharing a PostgresClient with the stores."""

    def __init__(
        self,
        client: PostgresClient,
        launcher: Launcher | None = None,
        *,
        chunk_manager: ChunkManagerClient | None = None,
        down_polls_before_failed: int = 2,
    ) -> None:
        self._client = client
        self._jobs = JobStore(client)
        self._plans = PlanStore(client)
        self._events = PostgresEventLog(client)
        self._launcher = launcher or NoopLauncher()
        self._chunk_manager = chunk_manager
        self._cancelled_chunk_jobs: set[str] = set()
        self._down_polls_before_failed = max(1, down_polls_before_failed)
        # Consecutive DOWN readings per rank. A single bad poll must not fail a
        # rank: the operator recomputes status on its own cadence and a rolling
        # update briefly reports zero ready replicas.
        self._down_streak: dict[str, int] = {}
        # Ranks observed serving at least once. Before that, zero ready replicas
        # means "still launching"; after it, the same reading means "died".
        self._served_rank_ids: set[str] = set()

    def apply_pending(self, user_id: str) -> int:
        """Apply every unapplied plan for a user. Returns plans applied."""
        applied = 0
        for plan in self._plans.unapplied(user_id):
            self._apply_plan(plan)
            if self._plans.mark_applied(plan.plan_id):
                self._events.append(
                    Event(
                        user_id=plan.user_id,
                        type="plan.applied",
                        payload_json=PlanAppliedPayload(
                            plan_id=plan.plan_id,
                            user_id=plan.user_id,
                        ).model_dump(mode="json"),
                    )
                )
                applied += 1
                logger.info("applied plan %s", plan.plan_id)
            else:
                # Another Orca worker already applied it; CAS lost the race.
                logger.info("plan %s already applied, skipping", plan.plan_id)
        return applied

    def reconcile_rank_health(self, user_id: str) -> list[RankHealth]:
        """Move ranks.status to match what each rank's DGD reports.

        This is the only path by which a RUNNING rank can reach FAILED: plan
        application sets RUNNING once the API server accepts the DGD and never
        revisits it, so without this poll a rank whose workers died stays
        RUNNING forever and keeps receiving routed traffic.

        Returns the effective health of every active rank so the caller can
        publish it to the job routers.
        """
        k8s_for_rank = getattr(self._launcher, "k8s_for_rank", None)
        if k8s_for_rank is None:
            return []  # NoopLauncher: no cluster to observe.

        results: list[RankHealth] = []
        active_rank_ids: set[str] = set()
        for running in self._jobs.running_jobs(user_id):
            job_id = running.job.job_id
            # One DGD list per cluster, not per rank: a job's ranks may span
            # kube contexts but usually share one.
            by_cluster: dict[int, tuple[Any, list[Rank]]] = {}
            for allocation in running.ranks:
                rank = Rank(
                    rank_id=allocation.rank_id,
                    job_id=job_id,
                    plan_id=allocation.plan_id,
                    role=allocation.role,
                    shape_json=dict(allocation.shape_json),
                    n_replicas=allocation.n_replicas,
                    status=allocation.status,
                    reason_code=allocation.reason_code,
                )
                try:
                    k8s = k8s_for_rank(rank, job_kind=running.job.kind)
                except ValueError:
                    logger.exception("no cluster client for rank %s", rank.rank_id)
                    continue
                by_cluster.setdefault(id(k8s), (k8s, []))[1].append(rank)

            for k8s, ranks in by_cluster.values():
                if running.job.kind is JobKind.BATCH:
                    for rank in ranks:
                        active_rank_ids.add(rank.rank_id)
                        try:
                            pods = k8s.rank_pods(job_id, rank.rank_id)
                        except Exception:
                            logger.exception(
                                "batch pod status read failed for rank %s", rank.rank_id
                            )
                            continue
                        self._apply_rank_health(
                            user_id,
                            k8s,
                            rank,
                            batch_rank_health(
                                job_id,
                                rank.rank_id,
                                pods,
                                expected_replicas=rank.n_replicas,
                                nodes_per_chain=rank_node_count(rank),
                                ever_served=(
                                    rank.status is RankStatus.RUNNING
                                    or rank.rank_id in self._served_rank_ids
                                ),
                            ),
                        )
                    continue
                try:
                    deployments = dgd_by_rank_id(k8s.job_dgds(job_id))
                except Exception:
                    logger.exception("DGD status read failed for job %s", job_id)
                    continue
                for rank in ranks:
                    active_rank_ids.add(rank.rank_id)
                    ever_served = (
                        rank.status is RankStatus.RUNNING or rank.rank_id in self._served_rank_ids
                    )
                    health = rank_health(
                        job_id,
                        rank.rank_id,
                        deployments.get(rank.rank_id),
                        ever_served=ever_served,
                    )
                    if health.verdict is Verdict.UNKNOWN and not ever_served:
                        try:
                            cause = termination_reason_code(k8s.rank_pods(job_id, rank.rank_id))
                        except Exception:
                            logger.exception(
                                "startup pod status read failed for rank %s", rank.rank_id
                            )
                            cause = None
                        if cause is not None:
                            health = RankHealth(
                                rank.rank_id,
                                job_id,
                                Verdict.DOWN,
                                0,
                                cause[0],
                                cause[1],
                            )
                    results.append(
                        self._apply_rank_health(
                            user_id,
                            k8s,
                            rank,
                            health,
                        )
                    )

        # A rank that left the active set takes its debounce state with it, so a
        # relaunched rank_id starts clean rather than inheriting a stale streak.
        self._down_streak = {
            rank_id: streak
            for rank_id, streak in self._down_streak.items()
            if rank_id in active_rank_ids
        }
        self._served_rank_ids &= active_rank_ids
        return results

    def _apply_rank_health(
        self, user_id: str, k8s: Any, rank: Rank, health: RankHealth
    ) -> RankHealth:
        """Record one rank's health, returning the debounced verdict."""
        if health.verdict is Verdict.SERVING:
            self._served_rank_ids.add(rank.rank_id)
            self._down_streak.pop(rank.rank_id, None)
            # Only promote out of LAUNCHING. set_rank_status writes
            # reason_code and updated_at unconditionally, so a no-op CAS would
            # erase failure provenance and reset every age-based rule.
            if rank.status is RankStatus.LAUNCHING:
                promoted = self._jobs.set_rank_status(
                    rank.rank_id, RankStatus.RUNNING, [RankStatus.LAUNCHING]
                )
                if promoted:
                    self._events.append(
                        Event(
                            user_id=user_id,
                            job_id=rank.job_id,
                            rank_id=rank.rank_id,
                            type="rank.running",
                            payload_json=RankRunningPayload(
                                rank_id=rank.rank_id,
                                job_id=rank.job_id,
                            ).model_dump(mode="json"),
                        )
                    )
                    logger.info(
                        "rank %s serving with %s replica(s)",
                        rank.rank_id,
                        health.serving_replicas,
                    )
            return health

        if health.verdict is Verdict.UNKNOWN:
            self._down_streak.pop(rank.rank_id, None)
            return health

        streak = self._down_streak.get(rank.rank_id, 0) + 1
        self._down_streak[rank.rank_id] = streak
        if streak < self._down_polls_before_failed:
            # Not yet conclusive. Report no opinion rather than DOWN so a
            # single bad reading cannot evict every session homed on this rank.
            return replace(
                health,
                verdict=Verdict.UNKNOWN,
                reason_code=None,
                detail=f"{health.detail} ({streak}/{self._down_polls_before_failed})",
            )

        reason_code, detail = self._failure_cause(k8s, rank, health)
        failed = self._jobs.set_rank_status(
            rank.rank_id,
            RankStatus.FAILED,
            list(ACTIVE_RANK_STATUSES),
            reason_code=reason_code,
        )
        if failed:
            self._events.append(
                Event(
                    user_id=user_id,
                    job_id=rank.job_id,
                    rank_id=rank.rank_id,
                    type="rank.failed",
                    payload_json=RankFailedPayload(
                        rank_id=rank.rank_id,
                        job_id=rank.job_id,
                        reason_code=str(reason_code),
                        detail=detail,
                    ).model_dump(mode="json"),
                )
            )
            logger.warning("rank %s failed (%s): %s", rank.rank_id, reason_code, detail)
        return replace(health, reason_code=reason_code, detail=detail)

    def _failure_cause(self, k8s: Any, rank: Rank, health: RankHealth) -> tuple[str, str]:
        """Prefer the container's own termination reason over the DGD's silence.

        The DGD reports that replicas are not ready, never why. Pod termination
        state is the only place OOMKilled and CrashLoopBackOff surface, and it
        is read here rather than every poll because this runs once per failure.
        """
        try:
            pods = k8s.rank_pods(rank.job_id, rank.rank_id)
        except Exception:
            logger.exception("pod lookup failed for rank %s", rank.rank_id)
            pods = []
        cause = termination_reason_code(pods)
        if cause is not None:
            return cause
        return health.reason_code or ReasonCode.FAILED, health.detail

    def reconcile_finished(self, user_id: str) -> int:
        """Tear down active ranks whose owning jobs are already finished."""
        reconciled = 0
        # ponytail: list_jobs currently caps this recovery scan at 200 jobs.
        for job in self._jobs.list_jobs(user_id):
            if job.status is not JobStatus.FINISHED:
                continue
            if (
                self._chunk_manager is not None
                and job.kind is JobKind.BATCH
                and job.finish_reason == ReasonCode.CANCELLED
                and job.job_id not in self._cancelled_chunk_jobs
            ):
                self._chunk_manager.cancel_job(job.job_id)
                self._cancelled_chunk_jobs.add(job.job_id)
            rank_ids = self._active_rank_ids(job.job_id)
            if not rank_ids:
                continue
            self._launcher.teardown_job(job.job_id)
            self._events.append(
                Event(
                    user_id=user_id,
                    job_id=job.job_id,
                    type="job.finished",
                    payload_json=JobFinishedPayload(
                        job_id=job.job_id,
                        user_id=user_id,
                        finish_reason=job.finish_reason,
                    ).model_dump(mode="json"),
                )
            )
            self._stop_ranks(user_id, job.job_id, rank_ids, job.finish_reason)
            reconciled += 1
        return reconciled

    def reconcile_chunk_jobs(self, user_id: str) -> int:
        """Copy terminal chunk-manager state into the canonical Store job."""
        if self._chunk_manager is None:
            return 0
        reconciled = 0
        for running in self._jobs.running_jobs(user_id):
            if running.job.kind is not JobKind.BATCH:
                continue
            try:
                chunk_job = self._chunk_manager.get_job(running.job.job_id)
            except Exception:
                logger.exception("chunk job reconciliation failed for %s", running.job.job_id)
                continue
            if chunk_job.state == chunk_manager_pb2.JOB_STATE_SUCCEEDED:
                finish_reason = None
            elif chunk_job.state == chunk_manager_pb2.JOB_STATE_FAILED:
                finish_reason = ReasonCode.FAILED
            elif chunk_job.state == chunk_manager_pb2.JOB_STATE_CANCELLED:
                finish_reason = ReasonCode.CANCELLED
            else:
                continue
            if self._jobs.transition(
                running.job.job_id,
                JobStatus.FINISHED,
                [JobStatus.RUNNING],
                finish_reason=finish_reason,
            ):
                reconciled += 1
        return reconciled

    def reconcile_running(self, user_id: str) -> int:
        """Restore infrastructure, local configs, and tunnels after Orca restarts."""
        reconciled = 0
        for running in self._jobs.running_jobs(user_id):
            ranks = [
                Rank(
                    rank_id=allocation.rank_id,
                    job_id=running.job.job_id,
                    plan_id=allocation.plan_id,
                    role=allocation.role,
                    shape_json=dict(allocation.shape_json),
                    n_replicas=allocation.n_replicas,
                    status=allocation.status,
                    reason_code=allocation.reason_code,
                )
                for allocation in running.ranks
            ]
            if ranks:
                self._add_chain_associations(ranks)
                self._launcher.reconcile(running.job.job_id, ranks, job_kind=running.job.kind)
                reconciled += 1
        return reconciled

    def _apply_plan(self, plan: Plan) -> None:
        for action in plan.actions:
            # One bad action must not wedge the plan: an exception here would
            # leave the plan unapplied and retried forever on every pass.
            try:
                self._apply_action(plan, action)
            except Exception:
                logger.exception(
                    "action %s for job %s failed (plan %s); continuing",
                    action.type,
                    action.job_id,
                    plan.plan_id,
                )

    def _apply_action(self, plan: Plan, action: PlanAction) -> None:
        match action.type:
            case ActionType.PLACE:
                self._place(plan, action)
            case ActionType.SWAP:
                self._swap(plan, action)
            case ActionType.PREEMPT:
                self._preempt(plan, action)
            case ActionType.KEEP | ActionType.DEFER:
                pass

    # ----- per-action handlers --------------------------------------------

    def _place(self, plan: Plan, action: PlanAction) -> None:
        """waiting|paused -> running: record the ladder's ranks and apply DGDs."""
        job = self._jobs.get(action.job_id)
        ranks = self._materialize_ranks(plan, action, job.spec_json if job else None)
        previous_status = (
            job.status if job and job.status in (JobStatus.WAITING, JobStatus.PAUSED) else None
        )
        moved = self._jobs.transition(
            action.job_id,
            JobStatus.RUNNING,
            [JobStatus.WAITING, JobStatus.PAUSED],
        )
        if not moved:
            raise ValueError(f"place: job {action.job_id} is not waiting or paused")
        self._events.append(
            Event(
                user_id=plan.user_id,
                job_id=action.job_id,
                type="job.place",
                payload_json=JobPlacePayload(
                    job_id=action.job_id,
                    user_id=plan.user_id,
                    plan_id=plan.plan_id,
                    action_type="place",
                ).model_dump(mode="json"),
            )
        )
        try:
            assert job is not None
            self._launch_ranks(plan, action.job_id, ranks, job.kind)
        except ModelCatalogError as exc:
            finished = self._jobs.fail(
                action.job_id,
                [JobStatus.RUNNING],
                finish_reason=ReasonCode.MODEL_CATALOG_INVALID,
                error_message=str(exc),
            )
            if finished:
                self._events.append(
                    Event(
                        user_id=plan.user_id,
                        job_id=action.job_id,
                        type="job.finished",
                        payload_json=JobFinishedPayload(
                            job_id=action.job_id,
                            user_id=plan.user_id,
                            finish_reason=ReasonCode.MODEL_CATALOG_INVALID,
                            detail=str(exc),
                        ).model_dump(mode="json"),
                    )
                )
            raise
        except Exception:
            if moved and previous_status is not None:
                self._jobs.transition(action.job_id, previous_status, [JobStatus.RUNNING])
            raise
        if previous_status is JobStatus.PAUSED:
            self._events.append(
                Event(
                    user_id=plan.user_id,
                    job_id=action.job_id,
                    type="job.resumed",
                    payload_json=JobResumedPayload(
                        job_id=action.job_id,
                        user_id=plan.user_id,
                        plan_id=plan.plan_id,
                    ).model_dump(mode="json"),
                )
            )
        else:
            self._events.append(
                Event(
                    user_id=plan.user_id,
                    job_id=action.job_id,
                    type="job.placed",
                    payload_json=JobPlacedPayload(
                        job_id=action.job_id,
                        user_id=plan.user_id,
                        plan_id=plan.plan_id,
                    ).model_dump(mode="json"),
                )
            )
        logger.info("placed job %s (plan %s)", action.job_id, plan.plan_id)

    def _preempt(self, plan: Plan, action: PlanAction) -> None:
        """running -> paused: tear down the job's ranks, keep the job row."""
        # Record why: without it a preempted rank is indistinguishable from one
        # that finished cleanly, since STOPPED is written on both paths.
        self._teardown_ranks(plan.user_id, action.job_id, ReasonCode.PREEMPTED)
        moved = self._jobs.transition(action.job_id, JobStatus.PAUSED, [JobStatus.RUNNING])
        if moved:
            self._events.append(
                Event(
                    user_id=plan.user_id,
                    job_id=action.job_id,
                    type="job.paused",
                    payload_json=JobPausedPayload(
                        job_id=action.job_id,
                        user_id=plan.user_id,
                        plan_id=plan.plan_id,
                    ).model_dump(mode="json"),
                )
            )
        else:
            logger.warning("preempt: job %s was not running", action.job_id)
        logger.info("preempted job %s (plan %s)", action.job_id, plan.plan_id)

    def _swap(self, plan: Plan, action: PlanAction) -> None:
        """running: relaunch on a new ladder.

        Dynamo reconciliation deletes stale DGDs by diff. Orca only marks the
        old Rank rows stopped after writing and reconciling the new desired rows.
        """
        old_ranks = self._jobs.active_ranks(action.job_id)
        old_by_id = {rank.rank_id: rank for rank in old_ranks}
        job = self._jobs.get(action.job_id)
        if job is None or job.status is not JobStatus.RUNNING:
            raise ValueError(f"swap: job {action.job_id} is not running")
        ranks = self._materialize_ranks(plan, action, job.spec_json if job else None)
        ranks = [
            rank.model_copy(
                update={
                    "status": old_by_id[rank.rank_id].status,
                    "reason_code": old_by_id[rank.rank_id].reason_code,
                }
            )
            if rank.rank_id in old_by_id
            else rank
            for rank in ranks
        ]
        try:
            recorded = self._launch_ranks(plan, action.job_id, ranks, job.kind, old_by_id)
        except ModelCatalogError as exc:
            self._jobs.set_error(action.job_id, str(exc))
            raise
        self._jobs.set_error(action.job_id, None)
        recorded_by_id = {rank.rank_id: rank for rank in recorded}
        removed = [rank for rank in old_ranks if rank.rank_id not in recorded_by_id]
        self._drain_chain_associations(removed)
        if self._uses_chunk_manager(action.job_id):
            assert self._chunk_manager is not None
            for old_rank in old_ranks:
                replacement = recorded_by_id.get(old_rank.rank_id)
                if replacement is None:
                    continue
                for chain_id in range(replacement.n_replicas, old_rank.n_replicas):
                    self._chunk_manager.drain_chain_association(
                        action.job_id, old_rank.rank_id, chain_id
                    )
        self._stop_ranks(
            plan.user_id,
            action.job_id,
            set(old_by_id) - {rank.rank_id for rank in recorded},
        )
        logger.info("swapped job %s (plan %s)", action.job_id, plan.plan_id)

    # ----- launcher seam ---------------------------------------------------

    def _materialize_ranks(
        self, plan: Plan, action: PlanAction, job_spec: dict[str, Any] | None
    ) -> list[Rank]:
        ranks = ladder_to_ranks(
            action.ladder,
            job_id=action.job_id,
            plan_id=plan.plan_id,
            target_p99_ttft_ms=action.target_p99_ttft_ms,
            target_p99_tpot_ms=action.target_p99_tpot_ms,
            job_spec=job_spec,
        )
        if not ranks:
            raise ValueError(f"no launchable ranks for job {action.job_id} (plan {plan.plan_id})")
        return ranks

    def _launch_ranks(
        self,
        plan: Plan,
        job_id: str,
        ranks: list[Rank],
        job_kind: JobKind,
        previous: dict[str, Rank] | None = None,
    ) -> list[Rank]:
        previous = previous or {}
        recorded = self._jobs.launch_ranks(ranks)
        for rank in recorded:
            if rank.rank_id in previous:
                continue
            self._events.append(
                Event(
                    user_id=plan.user_id,
                    job_id=job_id,
                    rank_id=rank.rank_id,
                    type="rank.launching",
                    payload_json=RankLaunchingPayload(
                        rank_id=rank.rank_id,
                        job_id=job_id,
                        plan_id=rank.plan_id,
                        role=rank.role,
                        shape_json=rank.shape_json,
                        n_replicas=rank.n_replicas,
                    ).model_dump(mode="json"),
                )
            )
        try:
            self._add_chain_associations(recorded)
            self._launcher.reconcile(job_id, recorded, job_kind=job_kind)
        except Exception as exc:
            try:
                reused = [previous[rank.rank_id] for rank in recorded if rank.rank_id in previous]
                if reused:
                    self._jobs.launch_ranks(reused)
            finally:
                reason_code = (
                    ReasonCode.MODEL_CATALOG_INVALID
                    if isinstance(exc, ModelCatalogError)
                    else ReasonCode.LAUNCH_FAILED
                )
                for rank in recorded:
                    if rank.rank_id not in previous:
                        failed = self._jobs.set_rank_status(
                            rank.rank_id,
                            RankStatus.FAILED,
                            [RankStatus.LAUNCHING],
                            reason_code=reason_code,
                        )
                        if failed:
                            self._events.append(
                                Event(
                                    user_id=plan.user_id,
                                    job_id=job_id,
                                    rank_id=rank.rank_id,
                                    type="rank.failed",
                                    payload_json=RankFailedPayload(
                                        rank_id=rank.rank_id,
                                        job_id=job_id,
                                        reason_code=str(reason_code),
                                        detail=str(exc),
                                    ).model_dump(mode="json"),
                                )
                            )
            raise
        return recorded

    def _teardown_ranks(self, user_id: str, job_id: str, reason_code: str | None = None) -> None:
        ranks = self._jobs.active_ranks(job_id)
        rank_ids = {rank.rank_id for rank in ranks}
        self._drain_chain_associations(ranks)
        self._launcher.teardown_job(job_id)
        self._stop_ranks(user_id, job_id, rank_ids, reason_code)

    def _add_chain_associations(self, ranks: list[Rank]) -> None:
        if not ranks or not self._uses_chunk_manager(ranks[0].job_id):
            return
        assert self._chunk_manager is not None
        for rank in ranks:
            for chain_id in range(rank.n_replicas):
                self._chunk_manager.add_chain_association(rank.job_id, rank.rank_id, chain_id)

    def _drain_chain_associations(self, ranks: list[Rank]) -> None:
        if not ranks or not self._uses_chunk_manager(ranks[0].job_id):
            return
        assert self._chunk_manager is not None
        for rank in ranks:
            for chain_id in range(rank.n_replicas):
                self._chunk_manager.drain_chain_association(rank.job_id, rank.rank_id, chain_id)

    def _uses_chunk_manager(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        return self._chunk_manager is not None and job is not None and job.kind is JobKind.BATCH

    def _active_rank_ids(self, job_id: str) -> set[str]:
        return {rank.rank_id for rank in self._jobs.active_ranks(job_id)}

    def _stop_ranks(
        self, user_id: str, job_id: str, rank_ids: set[str], reason_code: str | None = None
    ) -> None:
        for rank_id in rank_ids:
            stopped = self._jobs.set_rank_status(
                rank_id,
                RankStatus.STOPPED,
                list(ACTIVE_RANK_STATUSES),
                reason_code=reason_code,
            )
            if stopped:
                self._events.append(
                    Event(
                        user_id=user_id,
                        job_id=job_id,
                        rank_id=rank_id,
                        type="rank.stopped",
                        payload_json=RankStoppedPayload(
                            rank_id=rank_id,
                            job_id=job_id,
                            reason_code=reason_code,
                        ).model_dump(mode="json"),
                    )
                )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Orca against Tandemn Store plans.")
    parser.add_argument("--user-id", default=os.getenv("TANDEMN_USER_ID"))
    parser.add_argument("--namespace", default=os.getenv("TANDEMN_K8S_NAMESPACE", "dynamo-system"))
    parser.add_argument(
        "--batch-namespace",
        default=os.getenv("TANDEMN_BATCH_K8S_NAMESPACE", "tandemn-system"),
        help="Kubernetes namespace for batch workers (defaults to --namespace)",
    )
    parser.add_argument("--router-config-dir", default=os.getenv("TANDEMN_ROUTER_CONFIG_DIR"))
    parser.add_argument("--router-binary", default=os.getenv("TANDEMN_ROUTER_BINARY"))
    parser.add_argument("--cluster-contexts", default=os.getenv("TANDEMN_CLUSTER_CONTEXTS"))
    parser.add_argument(
        "--online-worker-secret",
        default=os.getenv("TANDEMN_ONLINE_WORKER_SECRET"),
        help="optional Secret exposed to online vLLM workers via envFrom",
    )
    parser.add_argument(
        "--chunk-manager-target",
        default=os.getenv("TANDEMN_CHUNK_MANAGER_TARGET"),
        help="chunk-manager gRPC target, for example chunk-manager:9090",
    )
    parser.add_argument(
        "--batch-worker-secret",
        default=os.getenv("TANDEMN_BATCH_WORKER_SECRET"),
        help="optional Secret exposed to batch worker containers via envFrom",
    )
    parser.add_argument(
        "--batch-aws-region",
        default=os.getenv("TANDEMN_BATCH_AWS_REGION"),
        help="optional AWS_DEFAULT_REGION for batch workers",
    )
    parser.add_argument(
        "--router-port-base",
        type=int,
        default=int(os.getenv("TANDEMN_ROUTER_PORT_BASE", "18000")),
    )
    parser.add_argument(
        "--router-port-span",
        type=int,
        default=int(os.getenv("TANDEMN_ROUTER_PORT_SPAN", "10000")),
    )
    # AWS capacity refresh is disabled pending the GCP ResourceMap refresher.
    # parser.add_argument(
    #     "--aws-regions",
    #     default=os.getenv("TANDEMN_AWS_REGIONS", "us-east-1,us-east-2,us-west-1,us-west-2"),
    # )
    # parser.add_argument(
    #     "--capacity-refresh-seconds",
    #     type=float,
    #     default=float(os.getenv("TANDEMN_CAPACITY_REFRESH_SECONDS", "86400")),
    # )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=float(os.getenv("TANDEMN_ORCA_POLL_SECONDS", "5")),
    )
    parser.add_argument(
        "--down-polls-before-failed",
        type=int,
        default=int(os.getenv("TANDEMN_DOWN_POLLS_BEFORE_FAILED", "2")),
        help="consecutive DGD polls reporting no ready replica before a rank is failed",
    )
    parser.add_argument(
        "--no-rank-health",
        dest="rank_health",
        action="store_false",
        help="skip the DGD-status rank health poll (leaves ranks.status at plan-apply values)",
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--skip-capacity-refresh",
        action="store_true",
        help="deprecated no-op; AWS capacity refresh is currently disabled",
    )
    return parser.parse_args(argv)


def load_cluster_contexts(path: str) -> dict[str, str]:
    with open(path) as file:
        raw = json.load(file)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("cluster context map must be a non-empty JSON object")
    contexts = {str(key): str(value) for key, value in raw.items() if key and value}
    if len(contexts) != len(raw):
        raise ValueError("cluster context keys and values must be non-empty")
    return contexts


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    if not args.user_id:
        raise SystemExit("--user-id or TANDEMN_USER_ID is required")
    batch_namespace = args.batch_namespace or args.namespace
    client = PostgresClient()
    # AWS capacity refresh is disabled pending the GCP ResourceMap refresher.
    # refresher = CapacityRefresher(
    #     client,
    #     args.user_id,
    #     parse_region_csv(args.aws_regions),
    #     refresh_seconds=args.capacity_refresh_seconds,
    # )
    # try:
    #     refresher.refresh_if_due(force=True)
    # except Exception:
    #     logger.exception("capacity refresh failed")

    if args.cluster_contexts:
        contexts = load_cluster_contexts(args.cluster_contexts)
        launchers = {}
        for key, context in contexts.items():
            online_k8s = load_kube_client(args.namespace, context=context)
            batch_k8s = (
                online_k8s
                if batch_namespace == args.namespace
                else load_kube_client(batch_namespace, context=context)
            )
            launchers[key] = DynamoLauncher(
                namespace=args.namespace,
                k8s=online_k8s,
                context=context,
                batch_chunk_manager_address=args.chunk_manager_target,
                batch_namespace=batch_namespace,
                batch_k8s=batch_k8s,
                online_worker_secret=args.online_worker_secret,
                batch_worker_secret=args.batch_worker_secret,
                batch_aws_region=args.batch_aws_region,
            )
        default_cluster = None
    else:
        launchers = {
            "default": DynamoLauncher(
                namespace=args.namespace,
                batch_chunk_manager_address=args.chunk_manager_target,
                batch_namespace=batch_namespace,
                online_worker_secret=args.online_worker_secret,
                batch_worker_secret=args.batch_worker_secret,
                batch_aws_region=args.batch_aws_region,
            )
        }
        default_cluster = "default"
    tunnel_manager = PortForwardManager() if args.router_config_dir else None
    router_manager = None
    if args.router_binary and args.router_config_dir:
        if not os.getenv("TANDEMN_ROUTER_TELEMETRY_TOKEN"):
            raise SystemExit(
                "--router-binary requires TANDEMN_ROUTER_TELEMETRY_TOKEN in the environment"
            )
        router_manager = RouterProcessManager(args.router_binary)
    launcher = MultiClusterLauncher(
        launchers,
        router_config_dir=args.router_config_dir,
        router_port_base=args.router_port_base,
        router_port_span=args.router_port_span,
        model_catalogs=ModelCatalogStore(client) if args.router_config_dir else None,
        default_cluster=default_cluster,
        tunnels=tunnel_manager,
        routers=router_manager,
    )
    chunk_manager = None
    if args.chunk_manager_target:
        chunk_manager = ChunkManagerClient(args.chunk_manager_target)
    orca = Orca(
        client,
        launcher=launcher,
        chunk_manager=chunk_manager,
        down_polls_before_failed=args.down_polls_before_failed,
    )
    telemetry_token = os.getenv("TANDEMN_ROUTER_TELEMETRY_TOKEN")
    health_publisher = (
        RankHealthPublisher(telemetry_token) if telemetry_token and args.rank_health else None
    )
    previous_sigterm = None
    if tunnel_manager is not None:

        def stop_on_sigterm(*_args: object) -> None:
            raise KeyboardInterrupt

        previous_sigterm = signal.signal(signal.SIGTERM, stop_on_sigterm)
    try:
        try:
            applied = orca.apply_pending(args.user_id)
            logger.info("applied %s pending plan(s) at startup", applied)
        except Exception:
            logger.exception("initial plan apply failed")
        try:
            reconciled = orca.reconcile_chunk_jobs(args.user_id)
            logger.info("reconciled %s terminal chunk job(s)", reconciled)
        except Exception:
            logger.exception("chunk job reconciliation failed")
        try:
            reconciled = orca.reconcile_finished(args.user_id)
            logger.info("reconciled %s finished job(s)", reconciled)
        except Exception:
            logger.exception("finished job reconciliation failed")
        try:
            reconciled = orca.reconcile_running(args.user_id)
            logger.info("reconciled %s running job(s)", reconciled)
        except Exception:
            logger.exception("running job reconciliation failed")
        if args.rank_health:
            try:
                health = orca.reconcile_rank_health(args.user_id)
                if health_publisher is not None and health:
                    health_publisher.publish(health)
            except Exception:
                logger.exception("rank health reconciliation failed")
        if args.once:
            return
        while True:
            time.sleep(args.interval_seconds)
            # AWS capacity refresh is disabled pending the GCP ResourceMap refresher.
            # try:
            #     refresher.refresh_if_due()
            # except Exception:
            #     logger.exception("capacity refresh failed")
            try:
                applied = orca.apply_pending(args.user_id)
                logger.info("applied %s plan(s) for user %s", applied, args.user_id)
            except Exception:
                logger.exception("orca apply loop failed")
            try:
                reconciled = orca.reconcile_chunk_jobs(args.user_id)
                logger.info("reconciled %s terminal chunk job(s)", reconciled)
            except Exception:
                logger.exception("chunk job reconciliation failed")
            try:
                reconciled = orca.reconcile_finished(args.user_id)
                logger.info("reconciled %s finished job(s)", reconciled)
            except Exception:
                logger.exception("finished job reconciliation failed")
            try:
                reconciled = orca.reconcile_running(args.user_id)
                logger.info("reconciled %s running job(s)", reconciled)
            except Exception:
                logger.exception("running job reconciliation failed")
            if args.rank_health:
                try:
                    health = orca.reconcile_rank_health(args.user_id)
                    if health_publisher is not None and health:
                        health_publisher.publish(health)
                except Exception:
                    logger.exception("rank health reconciliation failed")
    finally:
        if chunk_manager is not None:
            chunk_manager.close()
        if router_manager is not None:
            router_manager.close()
        if tunnel_manager is not None:
            tunnel_manager.close()
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()

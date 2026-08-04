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
import logging
import os
import re
import time
from typing import Any

from tandemn_system_data.clients import JobStore, ModelCatalogStore, PlanStore, PostgresClient
from tandemn_system_data.models.enums import ActionType, JobStatus, RankRole, RankStatus, ReasonCode
from tandemn_system_data.models.plan import Plan, PlanAction
from tandemn_system_data.models.rank import Rank

from tandemn_orca.dynamo_kubernetes import load_kube_client
from tandemn_orca.launcher import DynamoLauncher, Launcher, ModelCatalogError, NoopLauncher
from tandemn_orca.scripts.resource_map_from_aws import CapacityRefresher, parse_region_csv

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

    def __init__(self, client: PostgresClient, launcher: Launcher | None = None) -> None:
        self._client = client
        self._jobs = JobStore(client)
        self._plans = PlanStore(client)
        self._launcher = launcher or NoopLauncher()

    def apply_pending(self, user_id: str) -> int:
        """Apply every unapplied plan for a user. Returns plans applied."""
        applied = 0
        for plan in self._plans.unapplied(user_id):
            self._apply_plan(plan)
            if self._plans.mark_applied(plan.plan_id):
                applied += 1
                logger.info("applied plan %s", plan.plan_id)
            else:
                # Another Orca worker already applied it; CAS lost the race.
                logger.info("plan %s already applied, skipping", plan.plan_id)
        return applied

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
        try:
            self._launch_ranks(action.job_id, ranks)
        except ModelCatalogError as exc:
            self._jobs.fail(
                action.job_id,
                [JobStatus.RUNNING],
                finish_reason=ReasonCode.MODEL_CATALOG_INVALID,
                error_message=str(exc),
            )
            raise
        except Exception:
            if moved and previous_status is not None:
                self._jobs.transition(action.job_id, previous_status, [JobStatus.RUNNING])
            raise
        logger.info("placed job %s (plan %s)", action.job_id, plan.plan_id)

    def _preempt(self, plan: Plan, action: PlanAction) -> None:
        """running -> paused: tear down the job's ranks, keep the job row."""
        self._teardown_ranks(action.job_id)
        moved = self._jobs.transition(action.job_id, JobStatus.PAUSED, [JobStatus.RUNNING])
        if not moved:
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
            recorded = self._launch_ranks(action.job_id, ranks, old_by_id)
        except ModelCatalogError as exc:
            self._jobs.set_error(action.job_id, str(exc))
            raise
        self._jobs.set_error(action.job_id, None)
        self._stop_ranks(set(old_by_id) - {rank.rank_id for rank in recorded})
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
        job_id: str,
        ranks: list[Rank],
        previous: dict[str, Rank] | None = None,
    ) -> list[Rank]:
        previous = previous or {}
        recorded = self._jobs.launch_ranks(ranks)
        try:
            self._launcher.reconcile(job_id, recorded)
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
                        self._jobs.set_rank_status(
                            rank.rank_id,
                            RankStatus.FAILED,
                            [RankStatus.LAUNCHING],
                            reason_code=reason_code,
                        )
            raise
        for rank in recorded:
            self._jobs.set_rank_status(rank.rank_id, RankStatus.RUNNING, [rank.status])
        return recorded

    def _teardown_ranks(self, job_id: str) -> None:
        rank_ids = self._active_rank_ids(job_id)
        self._launcher.teardown_job(job_id)
        self._stop_ranks(rank_ids)

    def _active_rank_ids(self, job_id: str) -> set[str]:
        return {rank.rank_id for rank in self._jobs.active_ranks(job_id)}

    def _stop_ranks(self, rank_ids: set[str]) -> None:
        for rank_id in rank_ids:
            self._jobs.set_rank_status(rank_id, RankStatus.STOPPED, list(ACTIVE_RANK_STATUSES))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Orca against Tandemn Store plans.")
    parser.add_argument("--user-id", default=os.getenv("TANDEMN_USER_ID"))
    parser.add_argument("--namespace", default=os.getenv("TANDEMN_K8S_NAMESPACE", "default"))
    parser.add_argument("--router-namespace", default=os.getenv("TANDEMN_ROUTER_NAMESPACE"))
    parser.add_argument("--router-kubeconfig", default=os.getenv("TANDEMN_ROUTER_KUBECONFIG"))
    parser.add_argument("--router-context", default=os.getenv("TANDEMN_ROUTER_CONTEXT"))
    parser.add_argument("--router-image", default=os.getenv("TANDEMN_ROUTER_IMAGE"))
    parser.add_argument(
        "--aws-regions",
        default=os.getenv("TANDEMN_AWS_REGIONS", "us-east-1,us-east-2,us-west-1,us-west-2"),
    )
    parser.add_argument(
        "--capacity-refresh-seconds",
        type=float,
        default=float(os.getenv("TANDEMN_CAPACITY_REFRESH_SECONDS", "86400")),
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=float(os.getenv("TANDEMN_ORCA_POLL_SECONDS", "5")),
    )
    parser.add_argument("--once", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    if not args.user_id:
        raise SystemExit("--user-id or TANDEMN_USER_ID is required")
    if args.router_namespace and not args.router_image:
        raise SystemExit(
            "--router-image or TANDEMN_ROUTER_IMAGE is required with router publishing"
        )

    client = PostgresClient()
    refresher = CapacityRefresher(
        client,
        args.user_id,
        parse_region_csv(args.aws_regions),
        refresh_seconds=args.capacity_refresh_seconds,
    )
    try:
        refresher.refresh_if_due(force=True)
    except Exception:
        logger.exception("capacity refresh failed")

    router_k8s = (
        load_kube_client(args.router_namespace, args.router_kubeconfig, args.router_context)
        if args.router_namespace
        else None
    )
    orca = Orca(
        client,
        launcher=DynamoLauncher(
            namespace=args.namespace,
            router_k8s=router_k8s,
            router_image=args.router_image,
            model_catalogs=ModelCatalogStore(client) if router_k8s is not None else None,
        ),
    )
    while True:
        try:
            refresher.refresh_if_due()
        except Exception:
            logger.exception("capacity refresh failed")
        try:
            applied = orca.apply_pending(args.user_id)
            logger.info("applied %s plan(s) for user %s", applied, args.user_id)
        except Exception:
            logger.exception("orca apply loop failed")
        if args.once:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()

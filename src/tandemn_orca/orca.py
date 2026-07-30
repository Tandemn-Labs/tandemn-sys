"""Orca apply-plan loop.

One pass: poll the plans Koi has created but not yet applied, apply each
action, and CAS the plan to ``applied``.

Action semantics (DATA_ARCHITECTURE.md §6):
    place    waiting|paused -> running   record the ladder's chains + apply DGDs
    keep     running                     no change
    defer    waiting                     no change
    preempt  running -> paused           tear down the job's chains
    swap     running                     relaunch on a new ladder

Chain rows are *authorized* capacity, not launched pods: the pool DGD's worker
replicas are owned by Dynamo's scaling adapter (DGDSA), and the in-pool Planner
scales DP width within [min_endpoint, max_gpu_budget]. Actual live width per
rank = distinct chain_ids in recent gpu_metrics rows.

Ladder shape (opaque JSONB; Koi <-> Orca contract):
    [{"role": "aggregate", "rank_id": "rank_0", "env": [...], "config": {...},
      "n_replicas": 3}]
``count`` / ``gpu_count`` is the GPU count per chain (required, positive int).
``chains`` / ``n_replicas`` is how many chain rows to record for max capacity.
``rank_id`` is Koi's logical rank id (unique within the job's action; Koi
autofills ``rank_{i}``); Orca preserves it into chain shapes, DGDs and pods.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any

from tandemn_system_data.clients import JobStore, PlanStore, PostgresClient
from tandemn_system_data.models.chain import Chain
from tandemn_system_data.models.enums import ActionType, ChainRole, ChainStatus, JobStatus
from tandemn_system_data.models.plan import Plan, PlanAction

from tandemn_orca.launcher import DynamoLauncher, Launcher, NoopLauncher
from tandemn_orca.scripts.resource_map_from_aws import CapacityRefresher, parse_region_csv

logger = logging.getLogger(__name__)

ACTIVE_CHAIN_STATUSES = (ChainStatus.LAUNCHING, ChainStatus.RUNNING)


def ladder_to_chains(
    ladder: list[dict[str, Any]] | None,
    *,
    job_id: str,
    plan_id: str,
    target_p99_ttft_ms: float | None = None,
    target_p99_tpot_ms: float | None = None,
    job_spec: dict[str, Any] | None = None,
) -> list[Chain]:
    """Translate a ladder into Chain rows, skipping malformed entries.

    A missing/invalid role, config, replica count, GPU count, or instance type
    skips the entry. Koi keeps some launch fields outside the rank config, so
    gaps are backfilled here: ``gpu_type`` from ``env[4]``, ``model_id`` from
    the job's ``spec_json``, ``engine_name`` defaulting to ``vllm``.
    """
    job_spec = job_spec or {}
    chains: list[Chain] = []
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
        # Koi's logical rank id; required so every DGD and pod has canonical
        # identity matching Koi's evidence rows.
        if not entry.get("rank_id"):
            logger.warning("skipping ladder entry without rank_id: %r", entry)
            continue
        shape_json["rank_id"] = str(entry["rank_id"])
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

        for _ in range(replicas):
            chains.append(
                Chain(job_id=job_id, plan_id=plan_id, role=role, shape_json=dict(shape_json))
            )
    return chains


def _parse_role(role_name: object) -> ChainRole | None:
    try:
        return ChainRole(role_name)
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
        """waiting|paused -> running: record the ladder's chains and apply DGDs."""
        moved = self._jobs.transition(
            action.job_id,
            JobStatus.RUNNING,
            [JobStatus.WAITING, JobStatus.PAUSED],
        )
        if not moved:
            logger.warning(
                "place: job %s not in waiting|paused; launching chains anyway", action.job_id
            )
        self._launch_ladder(plan, action)
        logger.info("placed job %s (plan %s)", action.job_id, plan.plan_id)

    def _preempt(self, plan: Plan, action: PlanAction) -> None:
        """running -> paused: tear down the job's chains, keep the job row."""
        # Collect + tear down while the job still reads as running; the
        # chain lookup goes through running_jobs.
        self._teardown_chains(plan.user_id, action.job_id)
        moved = self._jobs.transition(action.job_id, JobStatus.PAUSED, [JobStatus.RUNNING])
        if not moved:
            logger.warning("preempt: job %s was not running", action.job_id)
        logger.info("preempted job %s (plan %s)", action.job_id, plan.plan_id)

    def _swap(self, plan: Plan, action: PlanAction) -> None:
        """running: relaunch on a new ladder.

        Dynamo reconciliation deletes stale DGDs by diff. Orca only marks the
        old Chain rows stopped after writing the new desired rows.
        """
        old_chain_ids = self._active_chain_ids(plan.user_id, action.job_id)
        self._launch_ladder(plan, action)
        self._stop_chains(old_chain_ids)
        logger.info("swapped job %s (plan %s)", action.job_id, plan.plan_id)

    # ----- launcher seam ---------------------------------------------------

    def _launch_ladder(self, plan: Plan, action: PlanAction) -> list[Chain]:
        job = self._jobs.get(action.job_id)
        chains = ladder_to_chains(
            action.ladder,
            job_id=action.job_id,
            plan_id=plan.plan_id,
            target_p99_ttft_ms=action.target_p99_ttft_ms,
            target_p99_tpot_ms=action.target_p99_tpot_ms,
            job_spec=job.spec_json if job else None,
        )
        if not chains:
            logger.warning("no launchable chains for job %s (plan %s)", action.job_id, plan.plan_id)
            return []
        # Record canonical rows (status=LAUNCHING) first, then bring up the
        # real workers behind the launcher seam.
        recorded = self._jobs.launch_chains(chains)
        self._launcher.reconcile(action.job_id, recorded)
        return recorded

    def _teardown_chains(self, user_id: str, job_id: str) -> None:
        chain_ids = self._active_chain_ids(user_id, job_id)
        self._launcher.teardown_job(job_id)
        self._stop_chains(chain_ids)

    def _active_chain_ids(self, user_id: str, job_id: str) -> list[str]:
        for running in self._jobs.running_jobs(user_id):
            if running.job.job_id != job_id:
                continue
            return [c.chain_id for c in running.chains]
        return []

    def _stop_chains(self, chain_ids: list[str]) -> None:
        for chain_id in chain_ids:
            self._jobs.set_chain_status(chain_id, ChainStatus.STOPPED, list(ACTIVE_CHAIN_STATUSES))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Orca against Tandemn Store plans.")
    parser.add_argument("--user-id", default=os.getenv("TANDEMN_USER_ID"))
    parser.add_argument("--namespace", default=os.getenv("TANDEMN_K8S_NAMESPACE", "default"))
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

    orca = Orca(client, launcher=DynamoLauncher(namespace=args.namespace))
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

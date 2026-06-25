"""Orca apply-plan loop.

One pass: poll the plans Koi has created but not yet applied, apply each
action, and CAS the plan to ``applied``.

Action semantics (DATA_ARCHITECTURE.md §6):
    place    waiting|paused -> running   gang-launch the ladder's chains
    keep     running                     no change
    defer    waiting                     no change
    preempt  running -> paused           tear down the job's chains
    swap     running                     relaunch on a new ladder

Ladder shape (opaque JSONB; Koi <-> Orca contract):
    [{"prefill": {"gpu": "H100", "count": 8, "chains": 2}},
     {"decode":  {"gpu": "A100", "count": 8, "chains": 1}}]
Each entry is {role: shape}. ``count`` is the GPU count per chain
(required, positive int). ``chains`` is how many chains of that role to
launch (default 1) and is not part of the persisted shape.
"""

from __future__ import annotations

import logging
from typing import Any

from tandemn_system_data.clients import JobStore, PlanStore, PostgresClient
from tandemn_system_data.models.chain import Chain
from tandemn_system_data.models.enums import ActionType, ChainRole, ChainStatus, JobStatus
from tandemn_system_data.models.plan import Plan, PlanAction

from tandemn_orca.launcher import Launcher, NoopLauncher

logger = logging.getLogger(__name__)

ACTIVE_CHAIN_STATUSES = (ChainStatus.LAUNCHING, ChainStatus.RUNNING)


def ladder_to_chains(
    ladder: list[dict[str, Any]] | None,
    *,
    job_id: str,
    plan_id: str,
) -> list[Chain]:
    """Translate a ladder into Chain rows, skipping malformed entries.

    Each entry maps one role to a shape dict. ``chains`` (how many of that
    role) is consumed here; the rest of the shape is stored on each chain.
    A missing/invalid role or a non-positive int ``count`` skips the entry.
    """
    chains: list[Chain] = []
    for entry in ladder or []:
        if not isinstance(entry, dict):
            logger.warning("skipping non-dict ladder entry: %r", entry)
            continue
        for role_name, shape in entry.items():
            role = _parse_role(role_name)
            if role is None or not isinstance(shape, dict):
                logger.warning("skipping ladder entry with bad role/shape: %r", entry)
                continue
            count = shape.get("count")
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                logger.warning("skipping ladder entry without positive int count: %r", shape)
                continue
            replicas = shape.get("chains", 1)
            shape_json = {k: v for k, v in shape.items() if k != "chains"}
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
            self._apply_action(plan, action)

    def _apply_action(self, plan: Plan, action: PlanAction) -> None:
        match action.type:
            case ActionType.PLACE:
                self._place(plan, action)
            case ActionType.SWAP:
                self._swap(plan, action)
            case ActionType.PREEMPT | ActionType.KEEP | ActionType.DEFER:
                pass

    # ----- per-action handlers --------------------------------------------

    def _place(self, plan: Plan, action: PlanAction) -> None:
        """waiting|paused -> running: gang-launch the ladder's chains."""
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

    def _swap(self, plan: Plan, action: PlanAction) -> None:
        """running: relaunch on a new ladder.

        Cold-start (launch the new ladder) and teardown (stop the old
        chains) are independent — the new launch does not wait on teardown.
        """
        self._launch_ladder(plan, action)
        self._teardown_chains(plan.user_id, action.job_id)
        logger.info("swapped job %s (plan %s)", action.job_id, plan.plan_id)

    # ----- launcher seam ---------------------------------------------------

    def _launch_ladder(self, plan: Plan, action: PlanAction) -> list[Chain]:
        chains = ladder_to_chains(action.ladder, job_id=action.job_id, plan_id=plan.plan_id)
        if not chains:
            logger.warning("no launchable chains for job %s (plan %s)", action.job_id, plan.plan_id)
            return []
        # Record canonical rows (status=LAUNCHING) first, then bring up the
        # real workers behind the launcher seam.
        recorded = self._jobs.launch_chains(chains)
        self._launcher.launch(recorded)
        return recorded

    def _teardown_chains(self, user_id: str, job_id: str) -> None:
        for running in self._jobs.running_jobs(user_id):
            if running.job.job_id != job_id:
                continue
            chain_ids = [c.chain_id for c in running.chains]
            self._launcher.teardown(chain_ids)
            for chain_id in chain_ids:
                self._jobs.set_chain_status(
                    chain_id, ChainStatus.STOPPED, list(ACTIVE_CHAIN_STATUSES)
                )
            return


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    raise SystemExit("orca.main: wire a user_id and run loop before using")


if __name__ == "__main__":
    main()

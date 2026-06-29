"""Launcher seam — make a job's desired chains real.

Orca records canonical chain rows in the store, but bringing up the
actual GPU workers (SkyPilot today; Dynamo / td_operator k8s later) is a
swappable implementation. ca and live in Orca, not the store.

A Launcher operates on canonical ``Chain`` rows Orca has produced. Recording
rows and updating status/events stays in Orca; the launcher only touches
infrastructure.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Protocol

from tandemn_system_data.models.chain import Chain

from tandemn_orca.dynamo_compiler import compile_job
from tandemn_orca.dynamo_kubernetes import (
    DynamoKubernetesClient,
    ObjectKey,
    load_kube_client,
    object_key,
)

logger = logging.getLogger(__name__)


class Launcher(Protocol):
    """Reconciles job infrastructure to match Orca's Chain rows."""

    def reconcile(self, job_id: str, chains: list[Chain]) -> None:
        """Make infrastructure match the desired chains for one job."""
        ...

    def teardown_job(self, job_id: str) -> None:
        """Delete all infrastructure for one job."""
        ...


class ReconcileError(RuntimeError):
    def __init__(
        self,
        apply_error: BaseException | None,
        delete_error: BaseException | None,
    ) -> None:
        self.apply_error = apply_error
        self.delete_error = delete_error
        parts = []
        if apply_error:
            parts.append(f"apply failed: {apply_error}")
        if delete_error:
            parts.append(f"delete failed: {delete_error}")
        super().__init__("; ".join(parts))


class NoopLauncher:
    """Records intent only — no infrastructure is touched.

    The default in the MVP: Orca persists chain rows but does not yet
    bring up real workers. Swap in a SkyPilot / Dynamo launcher later.
    """

    def reconcile(self, job_id: str, chains: list[Chain]) -> None:
        for chain in chains:
            logger.info(
                "noop reconcile: job=%s chain=%s role=%s shape=%s",
                job_id,
                chain.chain_id,
                chain.role,
                chain.shape_json,
            )

    def teardown_job(self, job_id: str) -> None:
        logger.info("noop teardown: job %s", job_id)


class DynamoLauncher:
    def __init__(
        self,
        namespace: str = "default",
        k8s: DynamoKubernetesClient | None = None,
    ) -> None:
        self.namespace = namespace
        self.k8s = k8s or load_kube_client(namespace)

    def reconcile(self, job_id: str, chains: list[Chain]) -> None:
        desired = compile_job(job_id, chains, self.namespace)
        desired_keys = {object_key(obj) for obj in desired}
        stale = self.k8s.list_job_objects(job_id) - desired_keys
        self._apply_and_delete(desired, stale)

    def teardown_job(self, job_id: str) -> None:
        self.k8s.delete_all_for_job(job_id)

    def _apply_and_delete(self, desired: list[dict], stale: set[ObjectKey]) -> None:
        with ThreadPoolExecutor(max_workers=2) as pool:
            apply_future = pool.submit(self.k8s.apply_many, desired)
            delete_future = pool.submit(self.k8s.delete_many, stale)

            apply_error = _future_error(apply_future)
            delete_error = _future_error(delete_future)

        if apply_error or delete_error:
            raise ReconcileError(apply_error, delete_error)


def _future_error(future: Future[Any]) -> BaseException | None:
    try:
        future.result()
    except BaseException as exc:
        return exc
    return None

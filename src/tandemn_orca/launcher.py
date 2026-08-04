"""Launcher seam — make a job's desired ranks real.

Orca records canonical rank rows in the store; bringing up the actual GPU
workers is a swappable implementation (DynamoLauncher applies DGDs;
NoopLauncher records intent only).

A Launcher operates on canonical ``Rank`` rows Orca has produced. Recording
rows and updating status/events stays in Orca; the launcher only touches
infrastructure.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

from tandemn_system_data.models.rank import Rank

from tandemn_orca.dynamo_compiler import (
    compile_job,
    render_router_objects,
    router_configmap_name,
    router_name,
)
from tandemn_orca.dynamo_kubernetes import (
    DynamoKubernetesClient,
    ObjectKey,
    load_kube_client,
    object_key,
)

logger = logging.getLogger(__name__)


class Launcher(Protocol):
    """Reconciles job infrastructure to match Orca's Rank rows."""

    def reconcile(self, job_id: str, ranks: list[Rank]) -> None:
        """Make infrastructure match the desired ranks for one job."""
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

    Used when Orca should persist rank rows without touching a cluster
    (tests, dry runs); production uses DynamoLauncher.
    """

    def reconcile(self, job_id: str, ranks: list[Rank]) -> None:
        for rank in ranks:
            logger.info(
                "noop reconcile: job=%s rank=%s role=%s shape=%s",
                job_id,
                rank.rank_id,
                rank.role,
                rank.shape_json,
            )

    def teardown_job(self, job_id: str) -> None:
        logger.info("noop teardown: job %s", job_id)


class DynamoLauncher:
    def __init__(
        self,
        namespace: str = "default",
        k8s: DynamoKubernetesClient | None = None,
        router_k8s: DynamoKubernetesClient | None = None,
        router_image: str | None = None,
    ) -> None:
        self.namespace = namespace
        self.k8s = k8s or load_kube_client(namespace)
        self.router_k8s = router_k8s
        self.router_image = router_image

    def reconcile(self, job_id: str, ranks: list[Rank]) -> None:
        desired = compile_job(job_id, ranks, self.namespace)
        desired_keys = {object_key(obj) for obj in desired}
        stale = self.k8s.list_job_objects(job_id) - desired_keys
        apply_error = _call(self.k8s.apply_many, desired)
        if apply_error:
            raise ReconcileError(apply_error, None)
        if self.router_k8s is not None:
            router_objects = render_router_objects(
                job_id,
                ranks,
                self.router_image or "",
                self.router_k8s.namespace,
            )
            apply_error = _call(self.router_k8s.apply_many, router_objects)
            if apply_error:
                raise ReconcileError(apply_error, None)
        self._delete_stale(stale)

    def teardown_job(self, job_id: str) -> None:
        self.k8s.delete_all_for_job(job_id)
        if self.router_k8s is not None:
            name = router_name(job_id)
            self.router_k8s.delete_many(
                {
                    ("ConfigMap", router_configmap_name(job_id)),
                    ("Deployment", name),
                    ("Service", name),
                }
            )

    def _delete_stale(self, stale: set[ObjectKey]) -> None:
        delete_error = _call(self.k8s.delete_many, stale)
        if delete_error:
            raise ReconcileError(None, delete_error)


def _call[T](fn: Callable[[T], object], arg: T) -> BaseException | None:
    try:
        fn(arg)
    except BaseException as exc:
        return exc
    return None

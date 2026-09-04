"""Launcher seam — make a job's desired ranks real.

Orca records canonical rank rows in the store; bringing up the actual GPU
workers is a swappable implementation (DynamoLauncher applies DGDs;
NoopLauncher records intent only).

A Launcher operates on canonical ``Rank`` rows Orca has produced. Recording
rows and updating status/events stays in Orca; the launcher only touches
infrastructure.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from tandemn_system_data.clients import ModelCatalogStore
from tandemn_system_data.models.enums import JobKind
from tandemn_system_data.models.rank import Rank

from tandemn_orca.batch_compiler import compile_batch_job
from tandemn_orca.dynamo_compiler import (
    compile_job,
    pool_dgd_name,
    render_router_config,
)
from tandemn_orca.dynamo_kubernetes import (
    DynamoKubernetesClient,
    ObjectKey,
    load_kube_client,
    object_key,
)
from tandemn_orca.tunnels import PortForwardManager, RouterProcessManager, TunnelSpec

logger = logging.getLogger(__name__)


class Launcher(Protocol):
    """Reconciles job infrastructure to match Orca's Rank rows."""

    def reconcile(
        self, job_id: str, ranks: list[Rank], *, job_kind: JobKind = JobKind.ONLINE
    ) -> None:
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


class ModelCatalogError(ValueError):
    """A router capacity value is missing or invalid for a selected rank."""


class NoopLauncher:
    """Records intent only — no infrastructure is touched.

    Used when Orca should persist rank rows without touching a cluster
    (tests, dry runs); production uses DynamoLauncher.
    """

    def reconcile(
        self, job_id: str, ranks: list[Rank], *, job_kind: JobKind = JobKind.ONLINE
    ) -> None:
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
        context: str | None = None,
        batch_chunk_manager_address: str | None = None,
        batch_namespace: str | None = None,
        batch_k8s: DynamoKubernetesClient | None = None,
        batch_worker_secret: str | None = None,
        batch_aws_region: str | None = None,
    ) -> None:
        self.namespace = namespace
        self.k8s = k8s or load_kube_client(namespace)
        self.context = context
        self.batch_chunk_manager_address = batch_chunk_manager_address
        self.batch_namespace = batch_namespace or namespace
        self.batch_worker_secret = batch_worker_secret
        self.batch_aws_region = batch_aws_region
        self.batch_k8s = batch_k8s or (
            self.k8s
            if self.batch_namespace == namespace
            else load_kube_client(self.batch_namespace, context=context)
        )

    def reconcile(
        self, job_id: str, ranks: list[Rank], *, job_kind: JobKind = JobKind.ONLINE
    ) -> None:
        if job_kind is JobKind.BATCH:
            desired = compile_batch_job(
                job_id,
                ranks,
                self.batch_namespace,
                self.batch_chunk_manager_address,
                self.batch_worker_secret,
                self.batch_aws_region,
            )
            k8s = self.batch_k8s
        else:
            desired = compile_job(job_id, ranks, self.namespace, self.batch_worker_secret)
            k8s = self.k8s
        desired_keys = {object_key(obj) for obj in desired}
        stale = k8s.list_job_objects(job_id) - desired_keys
        apply_error = _call(k8s.apply_many, desired)
        if apply_error:
            raise ReconcileError(apply_error, None)
        self._delete_stale(k8s, stale)

    def teardown_job(self, job_id: str) -> None:
        self.k8s.delete_all_for_job(job_id)
        if self.batch_k8s is not self.k8s:
            self.batch_k8s.delete_all_for_job(job_id)

    def k8s_for_rank(
        self, rank: Rank, *, job_kind: JobKind = JobKind.ONLINE
    ) -> DynamoKubernetesClient:
        """Cluster client that owns this rank's objects."""
        return self.batch_k8s if job_kind is JobKind.BATCH else self.k8s

    def _delete_stale(self, k8s: DynamoKubernetesClient, stale: set[ObjectKey]) -> None:
        delete_error = _call(k8s.delete_many, stale)
        if delete_error:
            raise ReconcileError(None, delete_error)


class MultiClusterLauncher:
    """Apply rank groups to kube contexts and publish one laptop router config."""

    def __init__(
        self,
        launchers: dict[str, DynamoLauncher],
        *,
        router_config_dir: str | None = None,
        router_port_base: int = 18000,
        router_port_span: int = 10000,
        model_catalogs: ModelCatalogStore | None = None,
        default_cluster: str | None = None,
        tunnels: PortForwardManager | None = None,
        routers: RouterProcessManager | None = None,
    ) -> None:
        if not launchers:
            raise ValueError("at least one cluster launcher is required")
        self.launchers = launchers
        self.router_config_dir = Path(router_config_dir) if router_config_dir else None
        self.router_port_base = router_port_base
        self.router_port_span = router_port_span
        self.model_catalogs = model_catalogs
        self.default_cluster = default_cluster
        self.tunnels = tunnels
        self.routers = routers

    def reconcile(
        self, job_id: str, ranks: list[Rank], *, job_kind: JobKind = JobKind.ONLINE
    ) -> None:
        groups: dict[str, list[Rank]] = {key: [] for key in self.launchers}
        for rank in ranks:
            key = self.default_cluster or _rank_cluster_key(rank)
            if key not in groups:
                raise ValueError(f"no kube context configured for rank environment {key!r}")
            groups[key].append(rank)

        router_config = None
        ports: dict[str, int] = {}
        if job_kind is JobKind.ONLINE and self.router_config_dir is not None:
            max_num_seq = _max_num_seq_by_rank(ranks, self.model_catalogs)
            ports = _ports_by_rank(ranks, self.router_port_base, self.router_port_span)
            router_config = render_router_config(job_id, ranks, max_num_seq, ports)

        for key, launcher in self.launchers.items():
            launcher.reconcile(job_id, groups[key], job_kind=job_kind)

        if router_config is not None:
            assert self.router_config_dir is not None
            if self.tunnels is not None:
                self.tunnels.reconcile(
                    job_id,
                    [
                        TunnelSpec(
                            job_id=job_id,
                            rank_id=rank.rank_id,
                            context=self.launchers[key].context,
                            namespace=self.launchers[key].namespace,
                            service=f"{pool_dgd_name(job_id, rank)}-frontend",
                            local_port=ports[rank.rank_id],
                        )
                        for key, grouped_ranks in groups.items()
                        for rank in grouped_ranks
                    ],
                )
            _write_router_config(self.router_config_dir, job_id, router_config)
            if self.routers is not None:
                self.routers.reconcile(job_id, str(self.router_config_dir / f"{job_id}.json"))
            for key, grouped_ranks in groups.items():
                for rank in grouped_ranks:
                    logger.info(
                        "router tunnel: kubectl --context %s --namespace %s port-forward "
                        "service/%s-frontend %s:8000",
                        self.launchers[key].context or key,
                        self.launchers[key].namespace,
                        pool_dgd_name(job_id, rank),
                        ports[rank.rank_id],
                    )

    def k8s_for_rank(
        self, rank: Rank, *, job_kind: JobKind = JobKind.ONLINE
    ) -> DynamoKubernetesClient:
        """Cluster client that owns this rank's objects."""
        key = self.default_cluster or _rank_cluster_key(rank)
        launcher = self.launchers.get(key)
        if launcher is None:
            raise ValueError(f"no kube context configured for rank environment {key!r}")
        return launcher.k8s_for_rank(rank, job_kind=job_kind)

    def teardown_job(self, job_id: str) -> None:
        for launcher in self.launchers.values():
            launcher.teardown_job(job_id)
        if self.tunnels is not None:
            self.tunnels.reconcile(job_id, [])
        if self.routers is not None:
            self.routers.reconcile(job_id, None)
        if self.router_config_dir is not None:
            self.router_config_dir.joinpath(f"{job_id}.json").unlink(missing_ok=True)


def _call[T](fn: Callable[[T], object], arg: T) -> BaseException | None:
    try:
        fn(arg)
    except BaseException as exc:
        return exc
    return None


def _max_num_seq_by_rank(
    ranks: list[Rank], model_catalogs: ModelCatalogStore | None
) -> dict[str, int]:
    if model_catalogs is None:
        raise ModelCatalogError("local router config requires a ModelCatalogStore")
    values: dict[str, int] = {}
    for rank in ranks:
        model_id = rank.shape_json.get("model_id")
        gpu_type = rank.shape_json.get("gpu_type")
        if not model_id or not gpu_type:
            raise ModelCatalogError(
                f"rank {rank.rank_id} is missing model_id or gpu_type for ModelCatalog lookup"
            )
        catalog = model_catalogs.get(str(model_id))
        if catalog is None:
            raise ModelCatalogError(f"ModelCatalog {model_id!r} is missing")
        matches = [
            entry
            for entry in catalog.max_num_seq
            if isinstance(entry, dict) and str(entry.get("gpu_type")) == str(gpu_type)
        ]
        if len(matches) != 1:
            raise ModelCatalogError(
                f"ModelCatalog {model_id!r} field 'max_num_seq' must contain exactly one "
                f"entry for gpu_type {gpu_type!r}"
            )
        value = matches[0].get("value")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ModelCatalogError(
                f"ModelCatalog {model_id!r} field 'max_num_seq' for gpu_type "
                f"{gpu_type!r} must be a positive integer"
            )
        values[rank.rank_id] = value
    return values


def _rank_cluster_key(rank: Rank) -> str:
    env = rank.shape_json.get("env")
    if not isinstance(env, (list, tuple)) or len(env) != 5 or not env[1] or not env[2]:
        raise ValueError(f"rank {rank.rank_id} requires a five-part env for cluster selection")
    return f"{env[1]}|{env[2]}"


def _ports_by_rank(ranks: list[Rank], base: int, span: int) -> dict[str, int]:
    if base < 1 or span < 1 or base + span > 65536:
        raise ValueError("router port range must fit within 1..65535")
    ports: dict[str, int] = {}
    used: dict[int, str] = {}
    for rank in ranks:
        digest = hashlib.sha256(rank.rank_id.encode()).digest()
        port = base + int.from_bytes(digest[:4], "big") % span
        if port in used and used[port] != rank.rank_id:
            raise ValueError(
                f"router port collision between ranks {used[port]!r} and {rank.rank_id!r}"
            )
        ports[rank.rank_id] = port
        used[port] = rank.rank_id
    return ports


def _write_router_config(directory: Path, job_id: str, config: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{job_id}.json"
    temporary = directory / f".{job_id}.json.tmp"
    with temporary.open("w") as file:
        json.dump(config, file, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)

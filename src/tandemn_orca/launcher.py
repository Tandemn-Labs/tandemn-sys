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
from tandemn_system_data.models.rank import Rank

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


class ModelCatalogError(ValueError):
    """A router capacity value is missing or invalid for a selected rank."""


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
        router_config_dir: str | None = None,
        router_port_base: int = 18000,
        router_port_span: int = 10000,
        model_catalogs: ModelCatalogStore | None = None,
    ) -> None:
        self.namespace = namespace
        self.k8s = k8s or load_kube_client(namespace)
        self.router_config_dir = Path(router_config_dir) if router_config_dir else None
        self.router_port_base = router_port_base
        self.router_port_span = router_port_span
        self.model_catalogs = model_catalogs

    def reconcile(self, job_id: str, ranks: list[Rank]) -> None:
        desired = compile_job(job_id, ranks, self.namespace)
        router_config = None
        ports: dict[str, int] = {}
        if self.router_config_dir is not None:
            max_num_seq = _max_num_seq_by_rank(ranks, self.model_catalogs)
            ports = _ports_by_rank(ranks, self.router_port_base, self.router_port_span)
            router_config = render_router_config(job_id, ranks, max_num_seq, ports)
        desired_keys = {object_key(obj) for obj in desired}
        stale = self.k8s.list_job_objects(job_id) - desired_keys
        apply_error = _call(self.k8s.apply_many, desired)
        if apply_error:
            raise ReconcileError(apply_error, None)
        self._delete_stale(stale)
        if router_config is not None:
            assert self.router_config_dir is not None
            _write_router_config(self.router_config_dir, job_id, router_config)
            for rank in ranks:
                logger.info(
                    "router tunnel: kubectl --namespace %s port-forward service/%s-frontend %s:8000",
                    self.namespace,
                    pool_dgd_name(job_id, rank),
                    ports[rank.rank_id],
                )

    def teardown_job(self, job_id: str) -> None:
        self.k8s.delete_all_for_job(job_id)
        if self.router_config_dir is not None:
            self.router_config_dir.joinpath(f"{job_id}.json").unlink(missing_ok=True)

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

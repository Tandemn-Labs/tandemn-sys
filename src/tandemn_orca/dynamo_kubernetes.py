"""Kubernetes SDK calls for Orca-owned workloads."""

from __future__ import annotations

import json
from typing import Any

from kubernetes import client, config
from kubernetes.client.rest import ApiException
from kubernetes.config.config_exception import ConfigException

type ObjectKey = tuple[str, str]

APPLY = "application/apply-patch+yaml"
FIELD_MANAGER = "tandemn-orca"
GROUP = "nvidia.com"
VERSION = "v1alpha1"
DGD_PLURAL = "dynamographdeployments"
LWS_GROUP = "leaderworkerset.x-k8s.io"
LWS_VERSION = "v1"
LWS_PLURAL = "leaderworkersets"

# The Python client defaults to no timeout, so a blackholed API server would
# block Orca's single-threaded poll loop forever -- stalling plan application,
# not just the call that hung.
REQUEST_TIMEOUT_SECONDS = 10


def load_kube_client(
    namespace: str = "default",
    kubeconfig: str | None = None,
    context: str | None = None,
) -> DynamoKubernetesClient:
    if kubeconfig is not None or context is not None:
        api_client = config.new_client_from_config(config_file=kubeconfig, context=context)
        return DynamoKubernetesClient(
            namespace,
            custom=client.CustomObjectsApi(api_client),
            core=client.CoreV1Api(api_client),
            batch=client.BatchV1Api(api_client),
        )
    try:
        config.load_incluster_config()
    except ConfigException:
        config.load_kube_config()
    return DynamoKubernetesClient(namespace)


def object_key(obj: dict[str, Any]) -> ObjectKey:
    return obj["kind"], obj["metadata"]["name"]


def job_selector(job_id: str) -> str:
    return f"tandemn.com/managed-by=orca,tandemn.com/job-id={job_id}"


class DynamoKubernetesClient:
    def __init__(
        self,
        namespace: str = "default",
        custom: Any = None,
        core: Any = None,
        batch: Any = None,
    ) -> None:
        self.namespace = namespace
        self.custom = custom or client.CustomObjectsApi()
        self._core = core
        self._batch = batch

    @property
    def core(self) -> Any:
        if self._core is None:
            self._core = client.CoreV1Api()
        return self._core

    @property
    def batch(self) -> Any:
        if self._batch is None:
            self._batch = client.BatchV1Api()
        return self._batch

    def list_job_objects(self, job_id: str) -> set[ObjectKey]:
        objects = {
            ("DynamoGraphDeployment", item["metadata"]["name"]) for item in self.job_dgds(job_id)
        }
        objects.update(
            ("Job", item.metadata.name)
            for item in self.batch.list_namespaced_job(
                self.namespace,
                label_selector=job_selector(job_id),
                _request_timeout=REQUEST_TIMEOUT_SECONDS,
            ).items
        )
        objects.update(
            ("LeaderWorkerSet", item["metadata"]["name"]) for item in self.job_lws(job_id)
        )
        return objects

    def job_dgds(self, job_id: str) -> list[dict[str, Any]]:
        """Every DGD Orca owns for a job, status included."""
        return self.custom.list_namespaced_custom_object(
            GROUP,
            VERSION,
            self.namespace,
            DGD_PLURAL,
            label_selector=job_selector(job_id),
            _request_timeout=REQUEST_TIMEOUT_SECONDS,
        ).get("items", [])

    def job_lws(self, job_id: str) -> list[dict[str, Any]]:
        try:
            return self.custom.list_namespaced_custom_object(
                LWS_GROUP,
                LWS_VERSION,
                self.namespace,
                LWS_PLURAL,
                label_selector=job_selector(job_id),
                _request_timeout=REQUEST_TIMEOUT_SECONDS,
            ).get("items", [])
        except ApiException as exc:
            if exc.status != 404:
                raise
            return []

    def rank_pods(self, job_id: str, rank_id: str) -> list[dict[str, Any]]:
        """Pods backing one rank, for reading why its containers died.

        Raw JSON, not the typed model: the generated client's ``to_dict()``
        renames keys to snake_case, and every other object here keeps the
        API server's camelCase.
        """
        response = self.core.list_namespaced_pod(
            self.namespace,
            label_selector=f"tandemn.com/job-id={job_id},tandemn.com/rank-id={rank_id}",
            _preload_content=False,
            _request_timeout=REQUEST_TIMEOUT_SECONDS,
        )
        return json.loads(response.data).get("items", [])

    def apply_many(self, objects: list[dict[str, Any]]) -> set[ObjectKey]:
        for obj in objects:
            self.apply(obj)
        return {object_key(obj) for obj in objects}

    def apply(self, obj: dict[str, Any]) -> None:
        kind, name = object_key(obj)
        if kind == "DynamoGraphDeployment":
            self.custom.patch_namespaced_custom_object(
                GROUP,
                VERSION,
                self.namespace,
                DGD_PLURAL,
                name,
                obj,
                field_manager=FIELD_MANAGER,
                force=True,
                _content_type=APPLY,
                _request_timeout=REQUEST_TIMEOUT_SECONDS,
            )
            return
        if kind == "Job":
            self.batch.patch_namespaced_job(
                name,
                self.namespace,
                obj,
                field_manager=FIELD_MANAGER,
                force=True,
                _content_type=APPLY,
                _request_timeout=REQUEST_TIMEOUT_SECONDS,
            )
            return
        if kind == "LeaderWorkerSet":
            self.custom.patch_namespaced_custom_object(
                LWS_GROUP,
                LWS_VERSION,
                self.namespace,
                LWS_PLURAL,
                name,
                obj,
                field_manager=FIELD_MANAGER,
                force=True,
                _content_type=APPLY,
                _request_timeout=REQUEST_TIMEOUT_SECONDS,
            )
            return
        raise ValueError(f"unsupported Kubernetes object kind: {kind}")

    def delete_many(self, keys: set[ObjectKey]) -> None:
        for kind, name in sorted(keys):
            self.delete(kind, name)

    def delete_all_for_job(self, job_id: str) -> None:
        self.delete_many(self.list_job_objects(job_id))

    def delete(self, kind: str, name: str) -> None:
        try:
            if kind == "DynamoGraphDeployment":
                self.custom.delete_namespaced_custom_object(
                    GROUP,
                    VERSION,
                    self.namespace,
                    DGD_PLURAL,
                    name,
                    _request_timeout=REQUEST_TIMEOUT_SECONDS,
                )
                return
            if kind == "Job":
                self.batch.delete_namespaced_job(
                    name,
                    self.namespace,
                    body=client.V1DeleteOptions(propagation_policy="Background"),
                    _request_timeout=REQUEST_TIMEOUT_SECONDS,
                )
                return
            if kind == "LeaderWorkerSet":
                self.custom.delete_namespaced_custom_object(
                    LWS_GROUP,
                    LWS_VERSION,
                    self.namespace,
                    LWS_PLURAL,
                    name,
                    _request_timeout=REQUEST_TIMEOUT_SECONDS,
                )
                return
            raise ValueError(f"unsupported Kubernetes object kind: {kind}")
        except ApiException as exc:
            if exc.status != 404:
                raise

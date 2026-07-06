"""Kubernetes SDK calls for Orca-owned Dynamo objects."""

from __future__ import annotations

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


def load_kube_client(namespace: str = "default") -> DynamoKubernetesClient:
    try:
        config.load_incluster_config()
    except ConfigException:
        config.load_kube_config()
    return DynamoKubernetesClient(namespace)


def object_key(obj: dict[str, Any]) -> ObjectKey:
    return obj["kind"], obj["metadata"]["name"]


def job_selector(job_id: str) -> str:
    return f"tandemn.ai/managed-by=orca,tandemn.ai/job-id={job_id}"


class DynamoKubernetesClient:
    def __init__(self, namespace: str = "default", core: Any = None, custom: Any = None) -> None:
        self.namespace = namespace
        self.core = core or client.CoreV1Api()
        self.custom = custom or client.CustomObjectsApi()

    def list_job_objects(self, job_id: str) -> set[ObjectKey]:
        selector = job_selector(job_id)
        configmaps = self.core.list_namespaced_config_map(
            self.namespace, label_selector=selector
        ).items
        dgds = self.custom.list_namespaced_custom_object(
            GROUP, VERSION, self.namespace, DGD_PLURAL, label_selector=selector
        ).get("items", [])
        return {
            *(("ConfigMap", item.metadata.name) for item in configmaps),
            *(("DynamoGraphDeployment", item["metadata"]["name"]) for item in dgds),
        }

    def apply_many(self, objects: list[dict[str, Any]]) -> set[ObjectKey]:
        for obj in objects:
            self.apply(obj)
        return {object_key(obj) for obj in objects}

    def apply(self, obj: dict[str, Any]) -> None:
        kind, name = object_key(obj)
        if kind == "ConfigMap":
            self.core.patch_namespaced_config_map(
                name,
                self.namespace,
                obj,
                field_manager=FIELD_MANAGER,
                force=True,
                _content_type=APPLY,
            )
            return
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
            if kind == "ConfigMap":
                self.core.delete_namespaced_config_map(name, self.namespace)
                return
            if kind == "DynamoGraphDeployment":
                self.custom.delete_namespaced_custom_object(
                    GROUP, VERSION, self.namespace, DGD_PLURAL, name
                )
                return
            raise ValueError(f"unsupported Kubernetes object kind: {kind}")
        except ApiException as exc:
            if exc.status != 404:
                raise

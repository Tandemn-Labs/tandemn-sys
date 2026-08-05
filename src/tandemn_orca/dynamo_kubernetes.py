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
    ) -> None:
        self.namespace = namespace
        self.custom = custom or client.CustomObjectsApi()

    def list_job_objects(self, job_id: str) -> set[ObjectKey]:
        selector = job_selector(job_id)
        dgds = self.custom.list_namespaced_custom_object(
            GROUP, VERSION, self.namespace, DGD_PLURAL, label_selector=selector
        ).get("items", [])
        return {
            *(("DynamoGraphDeployment", item["metadata"]["name"]) for item in dgds),
        }

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
                    GROUP, VERSION, self.namespace, DGD_PLURAL, name
                )
                return
            raise ValueError(f"unsupported Kubernetes object kind: {kind}")
        except ApiException as exc:
            if exc.status != 404:
                raise

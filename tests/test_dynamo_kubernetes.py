from __future__ import annotations

from kubernetes.client.rest import ApiException

from tandemn_orca.dynamo_kubernetes import (
    APPLY,
    DGD_PLURAL,
    FIELD_MANAGER,
    GROUP,
    REQUEST_TIMEOUT_SECONDS,
    VERSION,
    DynamoKubernetesClient,
    job_selector,
    load_kube_client,
    object_key,
)


def _dgd(name: str = "pool") -> dict:
    return {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": name},
    }


class FakeCustom:
    def __init__(self) -> None:
        self.applied: list[tuple] = []
        self.deleted: list[tuple] = []

    def list_namespaced_custom_object(self, *args, **kwargs):
        self.selector = (args, kwargs)
        return {"items": [{"metadata": {"name": "dgd"}}]}

    def patch_namespaced_custom_object(self, *args, **kwargs):
        self.applied.append((args, kwargs))

    def delete_namespaced_custom_object(self, *args, **kwargs):
        self.deleted.append((args, kwargs))


def test_object_key_and_selector():
    assert object_key(_dgd("x")) == ("DynamoGraphDeployment", "x")
    assert job_selector("job_1") == "tandemn.com/managed-by=orca,tandemn.com/job-id=job_1"


def test_list_job_objects_reads_dgds():
    custom = FakeCustom()
    client = DynamoKubernetesClient("ns", custom=custom)

    assert client.list_job_objects("job_1") == {("DynamoGraphDeployment", "dgd")}
    assert custom.selector == (
        (GROUP, VERSION, "ns", DGD_PLURAL),
        {
            "label_selector": "tandemn.com/managed-by=orca,tandemn.com/job-id=job_1",
            "_request_timeout": REQUEST_TIMEOUT_SECONDS,
        },
    )


def test_apply_many_uses_server_side_apply():
    custom = FakeCustom()
    client = DynamoKubernetesClient("ns", custom=custom)

    assert client.apply_many([_dgd("dgd")]) == {("DynamoGraphDeployment", "dgd")}
    assert custom.applied == [
        (
            (GROUP, VERSION, "ns", DGD_PLURAL, "dgd", _dgd("dgd")),
            {
                "field_manager": FIELD_MANAGER,
                "force": True,
                "_content_type": APPLY,
                "_request_timeout": REQUEST_TIMEOUT_SECONDS,
            },
        )
    ]


def test_delete_many_deletes_dgds():
    custom = FakeCustom()
    client = DynamoKubernetesClient("ns", custom=custom)

    client.delete_many({("DynamoGraphDeployment", "dgd")})

    assert custom.deleted == [
        (
            (GROUP, VERSION, "ns", DGD_PLURAL, "dgd"),
            {"_request_timeout": REQUEST_TIMEOUT_SECONDS},
        )
    ]


def test_delete_ignores_not_found():
    class MissingCustom(FakeCustom):
        def delete_namespaced_custom_object(self, *args, **kwargs):
            raise ApiException(status=404)

    DynamoKubernetesClient("ns", custom=MissingCustom()).delete("DynamoGraphDeployment", "dgd")


def test_load_kube_client_falls_back_to_kubeconfig(monkeypatch):
    calls: list[str] = []

    def incluster():
        from kubernetes.config.config_exception import ConfigException

        calls.append("incluster")
        raise ConfigException("no cluster")

    monkeypatch.setattr("tandemn_orca.dynamo_kubernetes.config.load_incluster_config", incluster)
    monkeypatch.setattr(
        "tandemn_orca.dynamo_kubernetes.config.load_kube_config",
        lambda: calls.append("kubeconfig"),
    )
    monkeypatch.setattr(
        "tandemn_orca.dynamo_kubernetes.client.CustomObjectsApi", lambda: FakeCustom()
    )

    loaded = load_kube_client("ns")

    assert loaded.namespace == "ns"
    assert calls == ["incluster", "kubeconfig"]


def test_load_kube_client_uses_explicit_context(monkeypatch):
    api_client = object()
    monkeypatch.setattr(
        "tandemn_orca.dynamo_kubernetes.config.new_client_from_config",
        lambda **kwargs: api_client,
    )
    monkeypatch.setattr(
        "tandemn_orca.dynamo_kubernetes.client.CustomObjectsApi",
        lambda loaded: ("custom", loaded),
    )
    monkeypatch.setattr(
        "tandemn_orca.dynamo_kubernetes.client.CoreV1Api",
        lambda loaded: ("core", loaded),
    )

    loaded = load_kube_client("serving", context="cloud-region")

    assert loaded.namespace == "serving"
    assert loaded.custom == ("custom", api_client)
    assert loaded.core == ("core", api_client)


def test_job_dgds_keeps_status():
    """list_job_objects discards everything but names; the health poll needs status."""

    class StatusCustom(FakeCustom):
        def list_namespaced_custom_object(self, *args, **kwargs):
            self.selector = (args, kwargs)
            return {
                "items": [
                    {"metadata": {"name": "dgd"}, "status": {"state": "successful"}},
                ]
            }

    client = DynamoKubernetesClient("ns", custom=StatusCustom())

    assert client.job_dgds("job_1") == [
        {"metadata": {"name": "dgd"}, "status": {"state": "successful"}}
    ]


def test_rank_pods_reads_raw_json_not_the_snake_cased_model():
    class FakeResponse:
        data = b'{"items": [{"status": {"containerStatuses": [{"name": "main"}]}}]}'

    class FakeCore:
        def list_namespaced_pod(self, *args, **kwargs):
            self.call = (args, kwargs)
            return FakeResponse()

    core = FakeCore()
    client = DynamoKubernetesClient("ns", custom=FakeCustom(), core=core)

    pods = client.rank_pods("job_1", "rank_1")

    # camelCase survives: the generated model's to_dict() would rename this.
    assert pods[0]["status"]["containerStatuses"][0]["name"] == "main"
    assert core.call[1]["label_selector"] == "tandemn.com/job-id=job_1,tandemn.com/rank-id=rank_1"
    assert core.call[1]["_preload_content"] is False

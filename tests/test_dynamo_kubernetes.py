from __future__ import annotations

from types import SimpleNamespace

from kubernetes.client.rest import ApiException

from tandemn_orca.dynamo_kubernetes import (
    APPLY,
    DGD_PLURAL,
    FIELD_MANAGER,
    GROUP,
    VERSION,
    DynamoKubernetesClient,
    job_selector,
    load_kube_client,
    object_key,
)


def _configmap(name: str = "router-config") -> dict:
    return {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": name}}


def _dgd(name: str = "pool") -> dict:
    return {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": name},
    }


def _deployment(name: str = "router") -> dict:
    return {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": name}}


def _service(name: str = "router") -> dict:
    return {"apiVersion": "v1", "kind": "Service", "metadata": {"name": name}}


class FakeCore:
    def __init__(self) -> None:
        self.applied: list[tuple] = []
        self.deleted: list[tuple] = []

    def list_namespaced_config_map(self, namespace, label_selector):
        self.selector = (namespace, label_selector)
        return SimpleNamespace(items=[SimpleNamespace(metadata=SimpleNamespace(name="cm"))])

    def patch_namespaced_config_map(self, *args, **kwargs):
        self.applied.append((args, kwargs))

    def delete_namespaced_config_map(self, *args):
        self.deleted.append(args)

    def patch_namespaced_service(self, *args, **kwargs):
        self.applied.append((args, kwargs))

    def delete_namespaced_service(self, *args):
        self.deleted.append(args)


class FakeCustom:
    def __init__(self) -> None:
        self.applied: list[tuple] = []
        self.deleted: list[tuple] = []

    def list_namespaced_custom_object(self, *args, **kwargs):
        self.selector = (args, kwargs)
        return {"items": [{"metadata": {"name": "dgd"}}]}

    def patch_namespaced_custom_object(self, *args, **kwargs):
        self.applied.append((args, kwargs))

    def delete_namespaced_custom_object(self, *args):
        self.deleted.append(args)


class FakeApps:
    def __init__(self) -> None:
        self.applied: list[tuple] = []
        self.deleted: list[tuple] = []

    def patch_namespaced_deployment(self, *args, **kwargs):
        self.applied.append((args, kwargs))

    def delete_namespaced_deployment(self, *args):
        self.deleted.append(args)


def test_object_key_and_selector():
    assert object_key(_dgd("x")) == ("DynamoGraphDeployment", "x")
    assert job_selector("job_1") == "tandemn.com/managed-by=orca,tandemn.com/job-id=job_1"


def test_list_job_objects_reads_configmaps_and_dgds():
    core = FakeCore()
    custom = FakeCustom()
    client = DynamoKubernetesClient("ns", core=core, custom=custom)

    assert client.list_job_objects("job_1") == {
        ("ConfigMap", "cm"),
        ("DynamoGraphDeployment", "dgd"),
    }
    assert core.selector == ("ns", "tandemn.com/managed-by=orca,tandemn.com/job-id=job_1")
    assert custom.selector == (
        (GROUP, VERSION, "ns", DGD_PLURAL),
        {"label_selector": "tandemn.com/managed-by=orca,tandemn.com/job-id=job_1"},
    )


def test_apply_many_uses_server_side_apply_for_supported_objects():
    core = FakeCore()
    custom = FakeCustom()
    apps = FakeApps()
    client = DynamoKubernetesClient("ns", core=core, custom=custom, apps=apps)

    assert client.apply_many(
        [_configmap("cm"), _dgd("dgd"), _deployment("router"), _service("router")]
    ) == {
        ("ConfigMap", "cm"),
        ("DynamoGraphDeployment", "dgd"),
        ("Deployment", "router"),
        ("Service", "router"),
    }

    assert core.applied == [
        (
            ("cm", "ns", _configmap("cm")),
            {"field_manager": FIELD_MANAGER, "force": True, "_content_type": APPLY},
        ),
        (
            ("router", "ns", _service("router")),
            {"field_manager": FIELD_MANAGER, "force": True, "_content_type": APPLY},
        ),
    ]
    assert custom.applied == [
        (
            (GROUP, VERSION, "ns", DGD_PLURAL, "dgd", _dgd("dgd")),
            {"field_manager": FIELD_MANAGER, "force": True, "_content_type": APPLY},
        )
    ]
    assert apps.applied == [
        (
            ("router", "ns", _deployment("router")),
            {"field_manager": FIELD_MANAGER, "force": True, "_content_type": APPLY},
        )
    ]


def test_delete_many_deletes_supported_objects():
    core = FakeCore()
    custom = FakeCustom()
    apps = FakeApps()
    client = DynamoKubernetesClient("ns", core=core, custom=custom, apps=apps)

    client.delete_many(
        {
            ("ConfigMap", "cm"),
            ("DynamoGraphDeployment", "dgd"),
            ("Deployment", "router"),
            ("Service", "router"),
        }
    )

    assert core.deleted == [("cm", "ns"), ("router", "ns")]
    assert custom.deleted == [(GROUP, VERSION, "ns", DGD_PLURAL, "dgd")]
    assert apps.deleted == [("router", "ns")]


def test_delete_ignores_not_found():
    class MissingCore(FakeCore):
        def delete_namespaced_config_map(self, *args):
            raise ApiException(status=404)

    DynamoKubernetesClient("ns", core=MissingCore(), custom=FakeCustom()).delete("ConfigMap", "cm")


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
    monkeypatch.setattr("tandemn_orca.dynamo_kubernetes.client.CoreV1Api", lambda: FakeCore())
    monkeypatch.setattr(
        "tandemn_orca.dynamo_kubernetes.client.CustomObjectsApi", lambda: FakeCustom()
    )
    monkeypatch.setattr("tandemn_orca.dynamo_kubernetes.client.AppsV1Api", lambda: FakeApps())

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
        "tandemn_orca.dynamo_kubernetes.client.CoreV1Api",
        lambda loaded: ("core", loaded),
    )
    monkeypatch.setattr(
        "tandemn_orca.dynamo_kubernetes.client.CustomObjectsApi",
        lambda loaded: ("custom", loaded),
    )
    monkeypatch.setattr(
        "tandemn_orca.dynamo_kubernetes.client.AppsV1Api",
        lambda loaded: ("apps", loaded),
    )

    loaded = load_kube_client("routing", "/config/control", "control")

    assert loaded.namespace == "routing"
    assert loaded.core == ("core", api_client)
    assert loaded.custom == ("custom", api_client)
    assert loaded.apps == ("apps", api_client)

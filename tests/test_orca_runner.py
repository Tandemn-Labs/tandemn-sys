from __future__ import annotations

from typing import ClassVar

import pytest

import tandemn_orca.orca as mod


class FakeOrca:
    instances: ClassVar[list[FakeOrca]] = []

    def __init__(self, client, launcher) -> None:
        self.client = client
        self.launcher = launcher
        self.applied_users: list[str] = []
        FakeOrca.instances.append(self)

    def apply_pending(self, user_id: str) -> int:
        self.applied_users.append(user_id)
        return 1


class FailingOnceOrca(FakeOrca):
    def apply_pending(self, user_id: str) -> int:
        self.applied_users.append(user_id)
        if len(self.applied_users) == 1:
            raise RuntimeError("boom")
        return 2


class FakeLauncher:
    instances: ClassVar[list[FakeLauncher]] = []

    def __init__(self, namespace: str, k8s=None, context=None) -> None:
        self.namespace = namespace
        self.k8s = k8s
        self.context = context
        FakeLauncher.instances.append(self)


class FakeMultiClusterLauncher:
    instances: ClassVar[list[FakeMultiClusterLauncher]] = []

    def __init__(self, launchers, **kwargs) -> None:
        self.launchers = launchers
        for key, value in kwargs.items():
            setattr(self, key, value)
        FakeMultiClusterLauncher.instances.append(self)


class FakeRefresher:
    instances: ClassVar[list[FakeRefresher]] = []

    def __init__(self, client, user_id, regions, refresh_seconds) -> None:
        self.client = client
        self.user_id = user_id
        self.regions = regions
        self.refresh_seconds = refresh_seconds
        self.calls: list[bool] = []
        FakeRefresher.instances.append(self)

    def refresh_if_due(self, *, force=False):
        self.calls.append(force)
        return True


def _patch_runner(monkeypatch, orca_cls=FakeOrca):
    FakeOrca.instances.clear()
    FakeLauncher.instances.clear()
    FakeMultiClusterLauncher.instances.clear()
    FakeRefresher.instances.clear()
    sleeps: list[float] = []
    monkeypatch.setattr(mod, "PostgresClient", lambda: "client")
    monkeypatch.setattr(mod, "DynamoLauncher", FakeLauncher)
    monkeypatch.setattr(mod, "MultiClusterLauncher", FakeMultiClusterLauncher)
    monkeypatch.setattr(mod, "load_kube_client", lambda *args, **kwargs: (args, kwargs))
    monkeypatch.setattr(mod, "ModelCatalogStore", lambda client: ("catalogs", client))
    monkeypatch.setattr(mod, "PortForwardManager", lambda: "tunnels")
    monkeypatch.setattr(mod, "CapacityRefresher", FakeRefresher)
    monkeypatch.setattr(mod, "Orca", orca_cls)
    monkeypatch.setattr(mod.time, "sleep", lambda seconds: sleeps.append(seconds))
    return sleeps


def test_main_once_uses_dynamo_launcher(monkeypatch):
    _patch_runner(monkeypatch)

    mod.main(["--user-id", "default", "--namespace", "koi", "--once"])

    assert FakeLauncher.instances[0].namespace == "koi"
    assert FakeOrca.instances[0].client == "client"
    assert FakeOrca.instances[0].launcher is FakeMultiClusterLauncher.instances[0]
    assert FakeMultiClusterLauncher.instances[0].default_cluster == "default"
    assert FakeOrca.instances[0].applied_users == ["default"]
    assert FakeRefresher.instances[0].client == "client"
    assert FakeRefresher.instances[0].regions == [
        "us-east-1",
        "us-east-2",
        "us-west-1",
        "us-west-2",
    ]
    assert FakeRefresher.instances[0].calls == [True, False]


def test_main_uses_env_defaults(monkeypatch):
    _patch_runner(monkeypatch)
    monkeypatch.setenv("TANDEMN_USER_ID", "env-user")
    monkeypatch.setenv("TANDEMN_K8S_NAMESPACE", "env-ns")
    monkeypatch.setenv("TANDEMN_ORCA_POLL_SECONDS", "7")
    monkeypatch.setenv("TANDEMN_AWS_REGIONS", "us-west-2,us-east-2")
    monkeypatch.setenv("TANDEMN_CAPACITY_REFRESH_SECONDS", "12")

    args = mod.parse_args([])

    assert args.user_id == "env-user"
    assert args.namespace == "env-ns"
    assert args.interval_seconds == 7
    assert args.aws_regions == "us-west-2,us-east-2"
    assert args.capacity_refresh_seconds == 12


def test_main_enables_local_router_config(monkeypatch):
    _patch_runner(monkeypatch)

    mod.main(
        [
            "--user-id",
            "default",
            "--namespace",
            "serving",
            "--router-config-dir",
            "/tmp/router-configs",
            "--router-port-base",
            "20000",
            "--once",
        ]
    )

    assert FakeLauncher.instances[0].namespace == "serving"
    multi = FakeMultiClusterLauncher.instances[0]
    assert multi.router_config_dir == "/tmp/router-configs"
    assert multi.router_port_base == 20000
    assert multi.model_catalogs == ("catalogs", "client")
    assert multi.tunnels == "tunnels"


def test_main_maps_cloud_regions_to_kube_contexts(monkeypatch, tmp_path):
    _patch_runner(monkeypatch)
    contexts = tmp_path / "contexts.json"
    contexts.write_text('{"aws|us-east-1":"eks-east","gcp|us-central1":"gke-central"}')

    mod.main(
        [
            "--user-id",
            "default",
            "--cluster-contexts",
            str(contexts),
            "--once",
        ]
    )

    multi = FakeMultiClusterLauncher.instances[0]
    assert set(multi.launchers) == {"aws|us-east-1", "gcp|us-central1"}
    assert [launcher.context for launcher in FakeLauncher.instances] == [
        "eks-east",
        "gke-central",
    ]
    assert FakeLauncher.instances[0].k8s == (("default",), {"context": "eks-east"})
    assert multi.default_cluster is None


def test_main_requires_user_id(monkeypatch):
    _patch_runner(monkeypatch)
    monkeypatch.delenv("TANDEMN_USER_ID", raising=False)

    with pytest.raises(SystemExit, match="--user-id"):
        mod.main(["--once"])


def test_main_can_skip_capacity_refresh(monkeypatch):
    _patch_runner(monkeypatch)

    mod.main(["--user-id", "default", "--skip-capacity-refresh", "--once"])

    assert FakeRefresher.instances[0].calls == []


def test_runner_logs_error_and_retries(monkeypatch):
    sleeps = _patch_runner(monkeypatch, FailingOnceOrca)

    def stop_after_sleep(seconds):
        sleeps.append(seconds)
        raise KeyboardInterrupt

    monkeypatch.setattr(mod.time, "sleep", stop_after_sleep)

    with pytest.raises(KeyboardInterrupt):
        mod.main(["--user-id", "default", "--interval-seconds", "3"])

    assert FakeOrca.instances[0].applied_users == ["default"]
    assert sleeps == [3]

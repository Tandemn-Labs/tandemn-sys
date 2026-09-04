from __future__ import annotations

from typing import ClassVar

import pytest

import tandemn_orca.orca as mod


class FakeOrca:
    instances: ClassVar[list[FakeOrca]] = []

    def __init__(self, client, launcher, **kwargs) -> None:
        self.client = client
        self.launcher = launcher
        self.kwargs = kwargs
        self.applied_users: list[str] = []
        self.finished_users: list[str] = []
        self.chunk_users: list[str] = []
        self.reconciled_users: list[str] = []
        self.health_users: list[str] = []
        FakeOrca.instances.append(self)

    def apply_pending(self, user_id: str) -> int:
        self.applied_users.append(user_id)
        return 1

    def reconcile_running(self, user_id: str) -> int:
        self.reconciled_users.append(user_id)
        return 0

    def reconcile_finished(self, user_id: str) -> int:
        self.finished_users.append(user_id)
        return 0

    def reconcile_chunk_jobs(self, user_id: str) -> int:
        self.chunk_users.append(user_id)
        return 0

    def reconcile_rank_health(self, user_id: str) -> list:
        self.health_users.append(user_id)
        return []


class FailingOnceOrca(FakeOrca):
    def apply_pending(self, user_id: str) -> int:
        self.applied_users.append(user_id)
        if len(self.applied_users) == 1:
            raise RuntimeError("boom")
        return 2


class FakeLauncher:
    instances: ClassVar[list[FakeLauncher]] = []

    def __init__(
        self,
        namespace: str,
        k8s=None,
        context=None,
        batch_chunk_manager_address=None,
        batch_namespace=None,
        batch_k8s=None,
        online_worker_secret=None,
        batch_worker_secret=None,
        batch_aws_region=None,
    ) -> None:
        self.namespace = namespace
        self.k8s = k8s
        self.context = context
        self.batch_chunk_manager_address = batch_chunk_manager_address
        self.batch_namespace = batch_namespace
        self.batch_k8s = batch_k8s
        self.online_worker_secret = online_worker_secret
        self.batch_worker_secret = batch_worker_secret
        self.batch_aws_region = batch_aws_region
        FakeLauncher.instances.append(self)


class FakeMultiClusterLauncher:
    instances: ClassVar[list[FakeMultiClusterLauncher]] = []

    def __init__(self, launchers, **kwargs) -> None:
        self.launchers = launchers
        for key, value in kwargs.items():
            setattr(self, key, value)
        FakeMultiClusterLauncher.instances.append(self)


class FakeTunnels:
    def __init__(self) -> None:
        self.closed = False

    def close(self):
        self.closed = True


def _patch_runner(monkeypatch, orca_cls=FakeOrca):
    FakeOrca.instances.clear()
    FakeLauncher.instances.clear()
    FakeMultiClusterLauncher.instances.clear()
    sleeps: list[float] = []
    monkeypatch.setattr(mod, "PostgresClient", lambda: "client")
    monkeypatch.setattr(mod, "DynamoLauncher", FakeLauncher)
    monkeypatch.setattr(mod, "MultiClusterLauncher", FakeMultiClusterLauncher)
    monkeypatch.setattr(mod, "load_kube_client", lambda *args, **kwargs: (args, kwargs))
    monkeypatch.setattr(mod, "ModelCatalogStore", lambda client: ("catalogs", client))
    monkeypatch.setattr(mod, "PortForwardManager", FakeTunnels)
    monkeypatch.setattr(mod, "Orca", orca_cls)
    monkeypatch.setattr(mod.time, "sleep", lambda seconds: sleeps.append(seconds))
    return sleeps


def test_main_once_uses_dynamo_launcher(monkeypatch):
    _patch_runner(monkeypatch)

    mod.main(
        [
            "--user-id",
            "default",
            "--namespace",
            "online",
            "--batch-namespace",
            "batch",
            "--batch-worker-secret",
            "tandemn-worker-secrets",
            "--online-worker-secret",
            "dynamo-worker-secrets",
            "--batch-aws-region",
            "us-east-2",
            "--once",
        ]
    )

    assert FakeLauncher.instances[0].namespace == "online"
    assert FakeLauncher.instances[0].batch_namespace == "batch"
    assert FakeLauncher.instances[0].online_worker_secret == "dynamo-worker-secrets"
    assert FakeLauncher.instances[0].batch_worker_secret == "tandemn-worker-secrets"
    assert FakeLauncher.instances[0].batch_aws_region == "us-east-2"
    assert FakeOrca.instances[0].client == "client"
    assert FakeOrca.instances[0].launcher is FakeMultiClusterLauncher.instances[0]
    assert FakeMultiClusterLauncher.instances[0].default_cluster == "default"
    assert FakeOrca.instances[0].applied_users == ["default"]
    assert FakeOrca.instances[0].finished_users == ["default"]
    assert FakeOrca.instances[0].chunk_users == ["default"]
    assert FakeOrca.instances[0].reconciled_users == ["default"]


def test_main_uses_env_defaults(monkeypatch):
    _patch_runner(monkeypatch)
    monkeypatch.setenv("TANDEMN_USER_ID", "env-user")
    monkeypatch.setenv("TANDEMN_K8S_NAMESPACE", "env-ns")
    monkeypatch.setenv("TANDEMN_BATCH_K8S_NAMESPACE", "env-batch-ns")
    monkeypatch.setenv("TANDEMN_ORCA_POLL_SECONDS", "7")

    args = mod.parse_args([])

    assert args.user_id == "env-user"
    assert args.namespace == "env-ns"
    assert args.batch_namespace == "env-batch-ns"
    assert args.interval_seconds == 7


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
    assert isinstance(multi.tunnels, FakeTunnels)
    assert multi.tunnels.closed


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
    assert FakeLauncher.instances[0].k8s == (("dynamo-system",), {"context": "eks-east"})
    assert multi.default_cluster is None


def test_main_requires_user_id(monkeypatch):
    _patch_runner(monkeypatch)
    monkeypatch.delenv("TANDEMN_USER_ID", raising=False)

    with pytest.raises(SystemExit, match="--user-id"):
        mod.main(["--once"])


def test_main_accepts_legacy_skip_capacity_refresh_flag(monkeypatch):
    _patch_runner(monkeypatch)

    mod.main(["--user-id", "default", "--skip-capacity-refresh", "--once"])

    assert mod.parse_args(["--skip-capacity-refresh"]).skip_capacity_refresh


def test_runner_logs_error_and_retries(monkeypatch):
    sleeps = _patch_runner(monkeypatch, FailingOnceOrca)

    def stop_after_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(mod.time, "sleep", stop_after_sleep)

    with pytest.raises(KeyboardInterrupt):
        mod.main(["--user-id", "default", "--interval-seconds", "3"])

    assert FakeOrca.instances[0].applied_users == ["default", "default"]
    assert FakeOrca.instances[0].reconciled_users == ["default", "default"]
    assert sleeps == [3, 3]

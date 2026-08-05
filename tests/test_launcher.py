from __future__ import annotations

import json

import pytest
from tandemn_system_data.models import ModelCatalog
from tandemn_system_data.models.enums import RankRole
from tandemn_system_data.models.rank import Rank

from tandemn_orca.launcher import (
    DynamoLauncher,
    ModelCatalogError,
    MultiClusterLauncher,
    ReconcileError,
)

RANK_ID = "rank_01JBM2YQYZ1KQ9C8GZP1XB6V5T"


def _rank() -> Rank:
    return Rank(
        rank_id=RANK_ID,
        job_id="job_online_001",
        plan_id="plan_1",
        role=RankRole.AGGREGATE,
        n_replicas=1,
        shape_json={
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "engine_name": "vllm",
            "instance_type": "g6e.12xlarge",
            "gpu_type": "L40S",
            "gpu_count": 1,
            "count": 1,
            "target_p99_ttft_ms": 500.0,
            "target_p99_tpot_ms": 50.0,
            "env": ["reserved", "aws", "us-east-2", "use2-az3", "L40S"],
        },
    )


class FakeK8s:
    def __init__(self, *, namespace="default", apply_error=None, delete_error=None) -> None:
        self.namespace = namespace
        self.apply_error = apply_error
        self.delete_error = delete_error
        self.applied: list[list[dict]] = []
        self.deleted: list[set[tuple[str, str]]] = []
        self.deleted_jobs: list[str] = []

    def list_job_objects(self, job_id):
        self.job_id = job_id
        return {("DynamoGraphDeployment", "job-online-001-aggregate-old")}

    def apply_many(self, objects):
        self.applied.append(objects)
        if self.apply_error:
            raise self.apply_error

    def delete_many(self, keys):
        self.deleted.append(keys)
        if self.delete_error:
            raise self.delete_error

    def delete_all_for_job(self, job_id):
        self.deleted_jobs.append(job_id)


class FakeCatalogs:
    def __init__(self, catalog=None) -> None:
        self.catalog = catalog

    def get(self, model_id):
        self.model_id = model_id
        return self.catalog


def _catalog() -> FakeCatalogs:
    return FakeCatalogs(
        ModelCatalog(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            max_num_seq=[{"gpu_type": "L40S", "value": 256}],
        )
    )


def test_dynamo_launcher_applies_desired_and_deletes_stale():
    k8s = FakeK8s()
    DynamoLauncher(k8s=k8s).reconcile("job_online_001", [_rank()])

    applied_names = {obj["metadata"]["name"] for obj in k8s.applied[0]}
    assert k8s.job_id == "job_online_001"
    assert len(applied_names) == 1
    assert k8s.deleted == [{("DynamoGraphDeployment", "job-online-001-aggregate-old")}]


def test_dynamo_launcher_keeps_stale_when_apply_fails():
    k8s = FakeK8s(apply_error=RuntimeError("boom"))

    with pytest.raises(ReconcileError) as error:
        DynamoLauncher(k8s=k8s).reconcile("job_online_001", [_rank()])

    assert isinstance(error.value.apply_error, RuntimeError)
    assert error.value.delete_error is None
    assert k8s.deleted == []


def test_dynamo_launcher_reports_delete_error_after_apply():
    k8s = FakeK8s(delete_error=RuntimeError("delete"))

    with pytest.raises(ReconcileError) as error:
        DynamoLauncher(k8s=k8s).reconcile("job_online_001", [_rank()])

    assert error.value.apply_error is None
    assert str(error.value) == "delete failed: delete"


def test_dynamo_launcher_teardown_deletes_job_objects():
    k8s = FakeK8s()
    DynamoLauncher(k8s=k8s).teardown_job("job_online_001")

    assert k8s.deleted_jobs == ["job_online_001"]


def test_dynamo_launcher_writes_and_deletes_local_router_config(tmp_path):
    workload = FakeK8s()
    rank = _rank()
    launcher = MultiClusterLauncher(
        {"default": DynamoLauncher(k8s=workload)},
        router_config_dir=str(tmp_path),
        model_catalogs=_catalog(),
        default_cluster="default",
    )

    launcher.reconcile("job_online_001", [rank])

    assert [obj["kind"] for obj in workload.applied[0]] == ["DynamoGraphDeployment"]
    config_path = tmp_path / "job_online_001.json"
    router_config = json.loads(config_path.read_text())
    deployment = router_config["deployments"][0]
    assert deployment["max_num_seq"] == 256
    assert deployment["maximum_requests"] == 256
    assert deployment["endpoint"].startswith("http://127.0.0.1:")
    worker_args = workload.applied[0][0]["spec"]["services"]["VllmDecodeWorker"]["extraPodSpec"][
        "mainContainer"
    ]["args"]
    assert "--max-num-seqs" not in worker_args
    launcher.teardown_job("job_online_001")
    assert workload.deleted_jobs == ["job_online_001"]
    assert not config_path.exists()


def test_dynamo_launcher_does_not_write_config_when_apply_fails(tmp_path):
    workload = FakeK8s(apply_error=RuntimeError("publish"))
    rank = _rank()

    with pytest.raises(ReconcileError, match="publish"):
        MultiClusterLauncher(
            {"default": DynamoLauncher(k8s=workload)},
            router_config_dir=str(tmp_path),
            model_catalogs=_catalog(),
            default_cluster="default",
        ).reconcile("job_online_001", [rank])

    assert workload.deleted == []
    assert not (tmp_path / "job_online_001.json").exists()


def test_dynamo_launcher_rejects_missing_catalog_before_workload_apply(tmp_path):
    workload = FakeK8s()
    rank = _rank()

    with pytest.raises(ModelCatalogError, match=r"ModelCatalog.*is missing"):
        MultiClusterLauncher(
            {"default": DynamoLauncher(k8s=workload)},
            router_config_dir=str(tmp_path),
            model_catalogs=FakeCatalogs(),
            default_cluster="default",
        ).reconcile("job_online_001", [rank])

    assert workload.applied == []


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([], "exactly one entry"),
        ([{"gpu_type": "L40S", "value": 0}], "positive integer"),
    ],
)
def test_dynamo_launcher_rejects_invalid_gpu_capacity(entries, message, tmp_path):
    workload = FakeK8s()
    rank = _rank()
    catalogs = FakeCatalogs(
        ModelCatalog(model_id="meta-llama/Llama-3.1-8B-Instruct", max_num_seq=entries)
    )

    with pytest.raises(ModelCatalogError, match=message):
        MultiClusterLauncher(
            {"default": DynamoLauncher(k8s=workload)},
            router_config_dir=str(tmp_path),
            model_catalogs=catalogs,
            default_cluster="default",
        ).reconcile("job_online_001", [rank])

    assert workload.applied == []


def test_multi_cluster_launcher_groups_ranks_by_cloud_region(tmp_path):
    aws = FakeK8s()
    gcp = FakeK8s()
    aws_rank = _rank()
    gcp_rank = aws_rank.model_copy(
        deep=True,
        update={"rank_id": "rank_01JBM30YQ7X3WQAR6HF8C2Q9T8"},
    )
    gcp_rank.shape_json["env"] = ["on_demand", "gcp", "us-central1", "a", "L40S"]

    class Tunnels:
        def __init__(self):
            self.calls = []

        def reconcile(self, job_id, specs):
            self.calls.append((job_id, specs))

    tunnels = Tunnels()
    launcher = MultiClusterLauncher(
        {
            "aws|us-east-2": DynamoLauncher(k8s=aws, context="aws-context"),
            "gcp|us-central1": DynamoLauncher(k8s=gcp, context="gcp-context"),
        },
        router_config_dir=str(tmp_path),
        model_catalogs=_catalog(),
        tunnels=tunnels,
    )

    launcher.reconcile("job_online_001", [aws_rank, gcp_rank])

    assert [obj["metadata"]["labels"]["tandemn.com/rank-id"] for obj in aws.applied[0]] == [
        aws_rank.rank_id
    ]
    assert [obj["metadata"]["labels"]["tandemn.com/rank-id"] for obj in gcp.applied[0]] == [
        gcp_rank.rank_id
    ]
    config = json.loads((tmp_path / "job_online_001.json").read_text())
    assert {deployment["rank_id"] for deployment in config["deployments"]} == {
        aws_rank.rank_id,
        gcp_rank.rank_id,
    }
    specs = tunnels.calls[0][1]
    assert {(spec.context, spec.service) for spec in specs} == {
        ("aws-context", "tdm-online-001-03a2a00c-frontend"),
        ("gcp-context", "tdm-online-001-efd9077e-frontend"),
    }

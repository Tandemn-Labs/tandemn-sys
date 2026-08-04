from __future__ import annotations

import pytest
from tandemn_system_data.models import ModelCatalog
from tandemn_system_data.models.enums import RankRole
from tandemn_system_data.models.rank import Rank

from tandemn_orca.launcher import DynamoLauncher, ModelCatalogError, ReconcileError

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
        return {
            ("ConfigMap", "stale-config"),
            ("DynamoGraphDeployment", "job-online-001-aggregate-old"),
        }

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
    assert k8s.deleted == [
        {("ConfigMap", "stale-config"), ("DynamoGraphDeployment", "job-online-001-aggregate-old")}
    ]


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


def test_dynamo_launcher_publishes_and_deletes_router_config():
    workload = FakeK8s()
    routing = FakeK8s(namespace="routing")
    rank = _rank()
    rank.shape_json["router_endpoint"] = "https://rank.example.internal"
    launcher = DynamoLauncher(
        k8s=workload,
        router_k8s=routing,
        router_image="registry.example/tandemn-router:test",
        model_catalogs=_catalog(),
    )

    launcher.reconcile("job_online_001", [rank])
    launcher.teardown_job("job_online_001")

    assert [obj["kind"] for obj in routing.applied[0]] == [
        "ConfigMap",
        "Deployment",
        "Service",
    ]
    assert all(obj["metadata"]["namespace"] == "routing" for obj in routing.applied[0])
    router_config = routing.applied[0][0]["data"]["router.json"]
    assert '"max_num_seq":256' in router_config
    assert '"maximum_requests":256' in router_config
    worker_args = workload.applied[0][0]["spec"]["services"]["VllmDecodeWorker"]["extraPodSpec"][
        "mainContainer"
    ]["args"]
    assert "--max-num-seqs" not in worker_args
    assert routing.deleted == [
        {
            ("ConfigMap", "tdm-online-001-router-config"),
            ("Deployment", "tdm-online-001-router"),
            ("Service", "tdm-online-001-router"),
        }
    ]


def test_dynamo_launcher_keeps_stale_when_router_publish_fails():
    workload = FakeK8s()
    routing = FakeK8s(namespace="routing", apply_error=RuntimeError("publish"))
    rank = _rank()
    rank.shape_json["router_endpoint"] = "https://rank.example.internal"

    with pytest.raises(ReconcileError, match="publish"):
        DynamoLauncher(
            k8s=workload,
            router_k8s=routing,
            router_image="registry.example/tandemn-router:test",
            model_catalogs=_catalog(),
        ).reconcile("job_online_001", [rank])

    assert workload.deleted == []


def test_dynamo_launcher_rejects_missing_catalog_before_workload_apply():
    workload = FakeK8s()
    rank = _rank()
    rank.shape_json["router_endpoint"] = "https://rank.example.internal"

    with pytest.raises(ModelCatalogError, match=r"ModelCatalog.*is missing"):
        DynamoLauncher(
            k8s=workload,
            router_k8s=FakeK8s(namespace="routing"),
            router_image="registry.example/tandemn-router:test",
            model_catalogs=FakeCatalogs(),
        ).reconcile("job_online_001", [rank])

    assert workload.applied == []


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([], "exactly one entry"),
        ([{"gpu_type": "L40S", "value": 0}], "positive integer"),
    ],
)
def test_dynamo_launcher_rejects_invalid_gpu_capacity(entries, message):
    workload = FakeK8s()
    rank = _rank()
    rank.shape_json["router_endpoint"] = "https://rank.example.internal"
    catalogs = FakeCatalogs(
        ModelCatalog(model_id="meta-llama/Llama-3.1-8B-Instruct", max_num_seq=entries)
    )

    with pytest.raises(ModelCatalogError, match=message):
        DynamoLauncher(
            k8s=workload,
            router_k8s=FakeK8s(namespace="routing"),
            router_image="registry.example/tandemn-router:test",
            model_catalogs=catalogs,
        ).reconcile("job_online_001", [rank])

    assert workload.applied == []

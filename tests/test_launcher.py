from __future__ import annotations

import pytest
from tandemn_system_data.models.chain import Chain
from tandemn_system_data.models.enums import ChainRole

from tandemn_orca.launcher import DynamoLauncher, ReconcileError


def _chain() -> Chain:
    return Chain(
        job_id="job_online_001",
        plan_id="plan_1",
        role=ChainRole.AGGREGATE,
        shape_json={
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "engine_name": "vllm",
            "instance_type": "g6e.12xlarge",
            "gpu_type": "L40S",
            "gpu_count": 1,
            "count": 1,
            "rank_id": "rank_0",
            "target_p99_ttft_ms": 500.0,
            "target_p99_tpot_ms": 50.0,
            "env": ["reserved", "aws", "us-east-2", "use2-az3", "L40S"],
        },
    )


class FakeK8s:
    def __init__(self, *, apply_error=None, delete_error=None) -> None:
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


def test_dynamo_launcher_applies_desired_and_deletes_stale():
    k8s = FakeK8s()
    DynamoLauncher(k8s=k8s).reconcile("job_online_001", [_chain()])

    applied_names = {obj["metadata"]["name"] for obj in k8s.applied[0]}
    assert k8s.job_id == "job_online_001"
    assert applied_names == {"tdm-online-001-rank-0"}
    assert k8s.deleted == [
        {("ConfigMap", "stale-config"), ("DynamoGraphDeployment", "job-online-001-aggregate-old")}
    ]


def test_dynamo_launcher_deletes_stale_even_when_apply_fails():
    k8s = FakeK8s(apply_error=RuntimeError("boom"))

    with pytest.raises(ReconcileError) as error:
        DynamoLauncher(k8s=k8s).reconcile("job_online_001", [_chain()])

    assert isinstance(error.value.apply_error, RuntimeError)
    assert error.value.delete_error is None
    assert k8s.deleted


def test_dynamo_launcher_reports_apply_and_delete_errors():
    k8s = FakeK8s(apply_error=RuntimeError("apply"), delete_error=RuntimeError("delete"))

    with pytest.raises(ReconcileError) as error:
        DynamoLauncher(k8s=k8s).reconcile("job_online_001", [_chain()])

    assert str(error.value) == "apply failed: apply; delete failed: delete"


def test_dynamo_launcher_teardown_deletes_job_objects():
    k8s = FakeK8s()
    DynamoLauncher(k8s=k8s).teardown_job("job_online_001")

    assert k8s.deleted_jobs == ["job_online_001"]

import json

import pytest
from tandemn_system_data.models import JobKind

from tandemn_orca.scripts import submit_job


def test_online_spec_gets_defaults_and_requires_latency_targets():
    spec = submit_job.normalize_job_spec(
        {
            "model_id": "Qwen/Qwen2.5-72B-Instruct",
            "target_p99_ttft_ms": 800,
            "target_p99_tpot_ms": 80,
        },
        JobKind.ONLINE,
    )

    features = spec["job_features"]
    assert features["type"] == "online"
    assert features["deadline_hrs"] == 0
    assert features["isl_token_avg"] == 4000
    assert features["osl_token_avg"] == 1000
    assert features["pd_ratio"] == 4.0
    assert features["priority_class"] == "STANDARD"
    assert features["max_concurrent_streaming"] == 10


def test_online_requires_latency_targets():
    with pytest.raises(SystemExit, match="target_p99_ttft_ms"):
        submit_job.normalize_job_spec({"model_id": "m", "target_p99_tpot_ms": 80}, JobKind.ONLINE)


def test_batched_requires_deadline_but_not_latency_targets():
    spec = submit_job.normalize_job_spec({"model_id": "m", "deadline_hrs": 2}, JobKind.BATCH)

    features = spec["job_features"]
    assert features["type"] == "batch"
    assert features["deadline_hrs"] == 2
    assert "target_p99_ttft_ms" not in features
    assert "target_p99_tpot_ms" not in features


def test_batched_requires_positive_deadline():
    with pytest.raises(SystemExit, match="deadline_hrs"):
        submit_job.normalize_job_spec({"model_id": "m"}, JobKind.BATCH)


def test_nested_job_features_override_defaults():
    spec = submit_job.normalize_job_spec(
        {
            "model_id": "m",
            "target_p99_ttft_ms": 800,
            "target_p99_tpot_ms": 80,
            "job_features": {"isl_token_avg": 2000, "osl_token_avg": 500},
        },
        JobKind.ONLINE,
    )

    assert spec["job_features"]["pd_ratio"] == 4.0
    assert spec["job_features"]["isl_token_avg"] == 2000


def test_invalid_priority_is_rejected():
    with pytest.raises(SystemExit, match="priority_class"):
        submit_job.normalize_job_spec(
            {
                "model_id": "m",
                "target_p99_ttft_ms": 800,
                "target_p99_tpot_ms": 80,
                "priority_class": "URGENT",
            },
            JobKind.ONLINE,
        )


def test_main_submits_normalized_spec(monkeypatch, capsys):
    store = _Store()
    monkeypatch.setattr(submit_job, "PostgresClient", lambda: object())
    monkeypatch.setattr(submit_job, "JobStore", lambda _client: store)

    rc = submit_job.main(
        [
            "--user-id",
            "user_1",
            "--online",
            "--spec",
            json.dumps(
                {
                    "model_id": "m",
                    "target_p99_ttft_ms": 800,
                    "target_p99_tpot_ms": 80,
                }
            ),
        ]
    )

    assert rc == 0
    assert store.job is not None
    assert store.job.kind is JobKind.ONLINE
    assert store.job.spec_json["job_features"]["deadline_hrs"] == 0
    assert capsys.readouterr().out.strip() == store.job.job_id


class _Store:
    def __init__(self):
        self.job = None

    def submit(self, job):
        self.job = job
        return job

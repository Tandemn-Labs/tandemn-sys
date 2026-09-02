from types import SimpleNamespace

from tandemn.chunkmanager.v1 import chunk_manager_pb2
from tandemn_orca.scripts import init_chunk_job


class FakePlanner:
    def __init__(self) -> None:
        self.calls = []

    def CreateJob(self, request, *, timeout):  # noqa: N802
        self.calls.append(("create", request, timeout))

    def RegisterChunks(self, request, *, timeout):  # noqa: N802
        self.calls.append(("register", request, timeout))

    def FinalizeJobRegistration(self, request, *, timeout):  # noqa: N802
        self.calls.append(("finalize", request, timeout))
        return SimpleNamespace(job=chunk_manager_pb2.Job(job_id=request.job_id))


def test_initializes_job_from_existing_chunk_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "chunks.json"
    manifest.write_text(
        '[{"chunk_id": 0, "s3_input_path": "s3://bucket/0.jsonl"},'
        '{"chunk_id": 1, "input_ref": "s3://bucket/1.jsonl"}]'
    )
    chunks = init_chunk_job.load_chunks(manifest)
    planner = FakePlanner()
    monkeypatch.setattr(init_chunk_job, "REGISTRATION_BATCH_SIZE", 1)

    job = init_chunk_job.initialize_job(
        planner,
        "job_01JBM2YQYZ1KQ9C8GZP1XB6V5T",
        chunks,
        max_retries=2,
        retry_backoff_seconds=1,
        lease_duration_seconds=60,
    )

    assert job.job_id == "01JBM2YQYZ1KQ9C8GZP1XB6V5T"
    assert [call[0] for call in planner.calls] == ["create", "register", "register", "finalize"]
    create = planner.calls[0][1]
    assert create.total_chunk_count == 2
    assert create.max_retries == 2
    assert [call[1].chunks[0].input_ref for call in planner.calls[1:3]] == [
        "s3://bucket/0.jsonl",
        "s3://bucket/1.jsonl",
    ]

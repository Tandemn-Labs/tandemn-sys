from tandemn.chunkmanager.v1 import chunk_manager_pb2
from tandemn_orca.chunk_manager import ChunkManagerClient


class FakePlanner:
    def __init__(self) -> None:
        self.calls = []

    def AddChainAssociation(self, request, *, timeout):  # noqa: N802
        self.calls.append(("add", request, timeout))

    def DrainChainAssociation(self, request, *, timeout):  # noqa: N802
        self.calls.append(("drain", request, timeout))

    def CancelJob(self, request, *, timeout):  # noqa: N802
        self.calls.append(("cancel", request, timeout))

    def GetJob(self, request, *, timeout):  # noqa: N802
        self.calls.append(("get", request, timeout))
        return chunk_manager_pb2.GetJobResponse(
            job=chunk_manager_pb2.Job(
                job_id=request.job_id,
                state=chunk_manager_pb2.JOB_STATE_SUCCEEDED,
            )
        )


def test_planner_calls_strip_orca_id_prefixes():
    client = ChunkManagerClient.__new__(ChunkManagerClient)
    client._planner = FakePlanner()

    client.add_chain_association(
        "job_01JBM2YQYZ1KQ9C8GZP1XB6V5T", "rank_01JBM30YQ7X3WQAR6HF8C2Q9T8", 2
    )
    client.drain_chain_association(
        "job_01JBM2YQYZ1KQ9C8GZP1XB6V5T", "rank_01JBM30YQ7X3WQAR6HF8C2Q9T8", 2
    )
    client.cancel_job("job_01JBM2YQYZ1KQ9C8GZP1XB6V5T")
    job = client.get_job("job_01JBM2YQYZ1KQ9C8GZP1XB6V5T")

    add, drain, cancel, get = client._planner.calls
    assert add == (
        "add",
        chunk_manager_pb2.AddChainAssociationRequest(
            chain=chunk_manager_pb2.ChainIdentity(
                job_id="01JBM2YQYZ1KQ9C8GZP1XB6V5T",
                rank_id="01JBM30YQ7X3WQAR6HF8C2Q9T8",
                chain_id=2,
            )
        ),
        5,
    )
    assert drain[0] == "drain"
    assert drain[1].chain == add[1].chain
    assert drain[2] == 5
    assert cancel == (
        "cancel",
        chunk_manager_pb2.CancelJobRequest(job_id="01JBM2YQYZ1KQ9C8GZP1XB6V5T"),
        5,
    )
    assert get == (
        "get",
        chunk_manager_pb2.GetJobRequest(job_id="01JBM2YQYZ1KQ9C8GZP1XB6V5T"),
        5,
    )
    assert job.state == chunk_manager_pb2.JOB_STATE_SUCCEEDED

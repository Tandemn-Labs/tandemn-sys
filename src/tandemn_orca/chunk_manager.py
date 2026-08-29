import grpc

from tandemn.chunkmanager.v1 import chunk_manager_pb2, chunk_manager_pb2_grpc

RPC_TIMEOUT_SECONDS = 5


class ChunkManagerClient:
    def __init__(self, target: str) -> None:
        self._channel = grpc.insecure_channel(target)
        self._planner = chunk_manager_pb2_grpc.PlannerServiceStub(self._channel)

    def add_chain_association(self, job_id: str, rank_id: str, chain_id: int) -> None:
        self._planner.AddChainAssociation(
            chunk_manager_pb2.AddChainAssociationRequest(
                chain=chunk_manager_pb2.ChainIdentity(
                    job_id=job_id.removeprefix("job_"),
                    rank_id=rank_id.removeprefix("rank_"),
                    chain_id=chain_id,
                )
            ),
            timeout=RPC_TIMEOUT_SECONDS,
        )

    def drain_chain_association(self, job_id: str, rank_id: str, chain_id: int) -> None:
        self._planner.DrainChainAssociation(
            chunk_manager_pb2.DrainChainAssociationRequest(
                chain=chunk_manager_pb2.ChainIdentity(
                    job_id=job_id.removeprefix("job_"),
                    rank_id=rank_id.removeprefix("rank_"),
                    chain_id=chain_id,
                )
            ),
            timeout=RPC_TIMEOUT_SECONDS,
        )

    def cancel_job(self, job_id: str) -> None:
        self._planner.CancelJob(
            chunk_manager_pb2.CancelJobRequest(job_id=job_id.removeprefix("job_")),
            timeout=RPC_TIMEOUT_SECONDS,
        )

    def close(self) -> None:
        self._channel.close()

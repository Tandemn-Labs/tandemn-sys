"""Create, register, and finalize one chunk-manager job."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import grpc
from google.protobuf import duration_pb2

from tandemn.chunkmanager.v1 import chunk_manager_pb2, chunk_manager_pb2_grpc

RPC_TIMEOUT_SECONDS = 5
REGISTRATION_BATCH_SIZE = 4096


def load_chunks(path: Path) -> list[chunk_manager_pb2.ChunkRegistration]:
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read chunk manifest {path}: {error}") from error
    if not isinstance(raw, list) or not raw:
        raise SystemExit("chunk manifest must be a non-empty JSON array")

    chunks = []
    seen_ids = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise SystemExit(f"chunk {index} must be a JSON object")
        chunk_id = item.get("chunk_id")
        input_ref = item.get("input_ref", item.get("s3_input_path"))
        if isinstance(chunk_id, bool) or not isinstance(chunk_id, int) or chunk_id < 0:
            raise SystemExit(f"chunk {index} has invalid chunk_id {chunk_id!r}")
        if chunk_id in seen_ids:
            raise SystemExit(f"duplicate chunk_id {chunk_id}")
        if not isinstance(input_ref, str) or not input_ref.strip():
            raise SystemExit(f"chunk {index} must include input_ref or s3_input_path")
        seen_ids.add(chunk_id)
        chunks.append(chunk_manager_pb2.ChunkRegistration(chunk_id=chunk_id, input_ref=input_ref))
    return chunks


def initialize_job(
    planner: chunk_manager_pb2_grpc.PlannerServiceStub,
    job_id: str,
    chunks: list[chunk_manager_pb2.ChunkRegistration],
    *,
    max_retries: int,
    retry_backoff_seconds: int,
    lease_duration_seconds: int,
) -> chunk_manager_pb2.Job:
    job_id = job_id.removeprefix("job_")
    planner.CreateJob(
        chunk_manager_pb2.CreateJobRequest(
            job_id=job_id,
            total_chunk_count=len(chunks),
            max_retries=max_retries,
            retry_backoff=duration_pb2.Duration(seconds=retry_backoff_seconds),
            lease_duration=duration_pb2.Duration(seconds=lease_duration_seconds),
        ),
        timeout=RPC_TIMEOUT_SECONDS,
    )
    for start in range(0, len(chunks), REGISTRATION_BATCH_SIZE):
        planner.RegisterChunks(
            chunk_manager_pb2.RegisterChunksRequest(
                job_id=job_id,
                chunks=chunks[start : start + REGISTRATION_BATCH_SIZE],
            ),
            timeout=RPC_TIMEOUT_SECONDS,
        )
    return planner.FinalizeJobRegistration(
        chunk_manager_pb2.FinalizeJobRegistrationRequest(job_id=job_id),
        timeout=RPC_TIMEOUT_SECONDS,
    ).job


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True, help="ORCA job_<ULID> or bare ULID")
    parser.add_argument("--chunks", required=True, type=Path, help="JSON chunk manifest")
    parser.add_argument(
        "--target",
        default=os.getenv("TANDEMN_CHUNK_MANAGER_TARGET", "127.0.0.1:9090"),
        help="chunk-manager gRPC target",
    )
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-backoff-seconds", type=int, default=1)
    parser.add_argument("--lease-duration-seconds", type=int, default=60)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_retries < 0 or args.retry_backoff_seconds < 0:
        raise SystemExit("retry values cannot be negative")
    if args.lease_duration_seconds <= 0:
        raise SystemExit("lease duration must be positive")
    chunks = load_chunks(args.chunks)

    try:
        with grpc.insecure_channel(args.target) as channel:
            job = initialize_job(
                chunk_manager_pb2_grpc.PlannerServiceStub(channel),
                args.job_id,
                chunks,
                max_retries=args.max_retries,
                retry_backoff_seconds=args.retry_backoff_seconds,
                lease_duration_seconds=args.lease_duration_seconds,
            )
    except grpc.RpcError as error:
        raise SystemExit(
            f"chunk-manager RPC failed: {error.code().name}: {error.details()}"
        ) from error

    print(f"{args.job_id} {chunk_manager_pb2.JobState.Name(job.state)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

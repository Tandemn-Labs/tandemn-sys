"""Compile batch ranks into tandemn-worker Kubernetes workloads."""

from __future__ import annotations

from typing import Any

from tandemn_system_data.models.rank import Rank

from tandemn_orca.compiler_common import (
    labels,
    local_model_path,
    model_weights_mount,
    model_weights_volume,
    rank_node_count,
    required,
    validate_unique_ranks,
    worker_gpu_count,
    workload_name,
)

BATCH_WORKER_IMAGE = "us-docker.pkg.dev/tandemn/tandemn-worker/tandemn-worker:v0.0.1-vllm0.28.0"
VLLM_IMAGE = (
    "docker.io/vllm/vllm-openai:v0.28.0@"
    "sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14"
)


def compile_batch_job(
    job_id: str,
    ranks: list[Rank],
    namespace: str,
    chunk_manager_address: str | None,
    worker_secret: str | None = None,
    aws_region: str | None = None,
) -> list[dict[str, Any]]:
    if not chunk_manager_address:
        raise ValueError("batch jobs require a chunk-manager address")
    validate_unique_ranks(ranks)
    return [
        render_chain(
            job_id,
            rank,
            chain_id,
            namespace,
            chunk_manager_address,
            worker_secret,
            aws_region,
        )
        for rank in ranks
        for chain_id in range(rank.n_replicas)
    ]


def render_chain(
    job_id: str,
    rank: Rank,
    chain_id: int,
    namespace: str,
    chunk_manager_address: str,
    worker_secret: str | None,
    aws_region: str | None,
) -> dict[str, Any]:
    node_count = rank_node_count(rank)
    gpu_count = worker_gpu_count(rank)
    if gpu_count % node_count:
        raise ValueError(
            f"rank {rank.rank_id}: count={gpu_count} does not divide evenly "
            f"across node_count={node_count}"
        )
    name = workload_name(job_id, f"{rank.rank_id}-{chain_id}")
    pod_labels = {
        "tandemn.com/job-type": "batched-inference",
        "tandemn.com/pods-discovery": "batch-worker",
        "tandemn.com/job-id": job_id,
        "tandemn.com/rank-id": rank.rank_id,
        "tandemn.com/chain-id": str(chain_id),
    }
    metadata = {
        "name": name,
        "namespace": namespace,
        "labels": {
            **labels(job_id, rank.rank_id, "batch-worker"),
            "tandemn.com/chain-id": str(chain_id),
        },
    }
    common: dict[str, Any] = {
        "name": "tandemn-worker",
        "image": BATCH_WORKER_IMAGE,
        "imagePullPolicy": "IfNotPresent",
        "env": _leader_env(rank, chain_id, chunk_manager_address, node_count, aws_region),
        "ports": [
            {"name": "metrics", "containerPort": 9000},
            {"name": "vllm", "containerPort": 8000},
        ],
        "readinessProbe": {
            "httpGet": {"path": "/health", "port": "vllm"},
            "periodSeconds": 10,
            "timeoutSeconds": 2,
        },
        "volumeMounts": [
            {"name": "dshm", "mountPath": "/dev/shm"},
            model_weights_mount(),
        ],
    }
    if worker_secret:
        common["envFrom"] = [{"secretRef": {"name": worker_secret, "optional": False}}]
    gpus_per_node = gpu_count // node_count
    if node_count == 1:
        common["resources"] = _resources(gpus_per_node, memory=True)
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": metadata,
            "spec": {
                "ttlSecondsAfterFinished": 86400,
                "template": {
                    "metadata": {"labels": pod_labels},
                    "spec": {
                        "restartPolicy": "Never",
                        "terminationGracePeriodSeconds": 90,
                        "nodeSelector": _node_selector(rank),
                        "containers": [common],
                        "volumes": [_dshm("96Gi"), model_weights_volume()],
                    },
                },
            },
        }

    common["name"] = "tandemn-batched-leader"
    common["resources"] = _resources(gpus_per_node, memory=True)
    model = local_model_path(rank.shape_json)
    parallel_args = _parallel_args(rank)
    distributed_args = (
        f"{parallel_args} --nnodes $(LWS_GROUP_SIZE) --node-rank $(LWS_WORKER_INDEX) "
        "--master-addr $(LWS_LEADER_ADDRESS)"
    )
    return {
        "apiVersion": "leaderworkerset.x-k8s.io/v1",
        "kind": "LeaderWorkerSet",
        "metadata": metadata,
        "spec": {
            "replicas": 1,
            "leaderWorkerTemplate": {
                "size": node_count,
                "restartPolicy": "None",
                "leaderTemplate": {
                    "metadata": {"labels": pod_labels},
                    "spec": {
                        "terminationGracePeriodSeconds": 90,
                        "nodeSelector": _node_selector(rank),
                        "containers": [common],
                        "volumes": [_dshm("16Gi"), model_weights_volume()],
                    },
                },
                "workerTemplate": {
                    "metadata": {"labels": pod_labels},
                    "spec": {
                        "nodeSelector": _node_selector(rank),
                        "containers": [
                            {
                                "name": "tandemn-batched-worker",
                                "image": VLLM_IMAGE,
                                "command": ["sh", "-c"],
                                "args": [f"vllm serve {model} {distributed_args} --headless"],
                                "resources": _resources(gpus_per_node),
                                "volumeMounts": [
                                    {"name": "dshm", "mountPath": "/dev/shm"},
                                    model_weights_mount(),
                                ],
                            }
                        ],
                        "volumes": [_dshm("16Gi"), model_weights_volume()],
                    },
                },
            },
        },
    }


def _leader_env(
    rank: Rank,
    chain_id: int,
    chunk_manager_address: str,
    node_count: int,
    aws_region: str | None,
) -> list[dict[str, Any]]:
    args = _parallel_args(rank)
    if node_count > 1:
        args += (
            " --nnodes $(LWS_GROUP_SIZE) --node-rank $(LWS_WORKER_INDEX) "
            "--master-addr $(LWS_LEADER_ADDRESS)"
        )
    env = [
        {"name": "TD_VLLM_MODEL", "value": local_model_path(rank.shape_json)},
        {"name": "TD_VLLM_HOST", "value": "0.0.0.0"},
        {"name": "TD_VLLM_READY_TIMEOUT_SECONDS", "value": "1800"},
        {"name": "TD_VLLM_EXTRA_ARGS", "value": args},
        {"name": "TD_CHUNK_MANAGER_ADDRESS", "value": chunk_manager_address},
        {"name": "TD_JOB_ID", "value": rank.job_id.removeprefix("job_")},
        {"name": "TD_RANK_ID", "value": rank.rank_id.removeprefix("rank_")},
        {"name": "TD_CHAIN_ID", "value": str(chain_id)},
    ]
    if aws_region:
        env.append({"name": "AWS_DEFAULT_REGION", "value": aws_region})
    return env


def _parallel_args(rank: Rank) -> str:
    shape = rank.shape_json
    return (
        f"--pipeline-parallel-size={shape.get('pp', 1)} --tensor-parallel-size={shape.get('tp', 1)}"
    )


def _node_selector(rank: Rank) -> dict[str, str]:
    return {"cloud.google.com/gke-nodepool": required(rank.shape_json, "instance_type")}


def _resources(gpus: int, *, memory: bool = False) -> dict[str, dict[str, str]]:
    requests = {"nvidia.com/gpu": str(gpus)}
    if memory:
        requests.update({"cpu": "1", "memory": "16Gi"})
    return {"requests": requests, "limits": {"nvidia.com/gpu": str(gpus)}}


def _dshm(size: str) -> dict[str, Any]:
    return {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": size}}

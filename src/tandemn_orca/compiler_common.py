"""Shared helpers for Orca Kubernetes workload compilers."""

from __future__ import annotations

import hashlib
from typing import Any

from tandemn_system_data.models.rank import Rank


def validate_unique_ranks(ranks: list[Rank]) -> None:
    if len({rank.rank_id for rank in ranks}) != len(ranks):
        raise ValueError("duplicate rank_id")


def workload_name(job_id: str, suffix: str, max_length: int = 29) -> str:
    tag = k8s_name(job_id).removeprefix("job-")[-10:].strip("-")
    name = k8s_name(f"tdm-{tag}-{suffix}")
    if len(name) > max_length:
        name = k8s_name(f"tdm-{tag}-{hashlib.sha1(suffix.encode()).hexdigest()[:8]}")
    return name


def k8s_name(value: str) -> str:
    value = value.lower().replace("_", "-").replace(".", "-")
    return "".join(ch for ch in value if ch.isalnum() or ch == "-")[:63].strip("-")


def labels(
    job_id: str,
    rank_id: str,
    plan_id: str | None,
    resource_kind: str,
    pool: str | None = None,
) -> dict[str, str]:
    result = {
        "tandemn.com/managed-by": "orca",
        "tandemn.com/job-id": job_id,
        "tandemn.com/rank-id": rank_id,
        "tandemn.com/resource-kind": resource_kind,
    }
    if plan_id:
        result["tandemn.com/plan-id"] = plan_id
    if pool:
        result["tandemn.com/pool-key"] = pool
    return result


def required(shape: dict[str, Any], key: str) -> str:
    value = shape.get(key)
    if value is None:
        raise ValueError(f"shape missing {key!r}")
    return str(value)


def worker_gpu_count(rank: Rank) -> int:
    count = rank.shape_json.get("count")
    if type(count) is not int or count <= 0:
        raise ValueError(f"rank {rank.rank_id} missing positive int count")
    return count


def rank_node_count(rank: Rank) -> int:
    node_count = rank.shape_json.get("node_count", rank.shape_json.get("num_nodes_per_chain", 1))
    if type(node_count) is not int or node_count < 1:
        raise ValueError(f"rank {rank.rank_id} node_count must be a positive int")
    return node_count

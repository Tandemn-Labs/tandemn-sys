"""Compile Orca Rank rows into Dynamo Kubernetes objects.

No Kubernetes calls live here; this file only returns dicts ready for server-side apply.

``shape_json["count"]`` is always the chain's world size (total GPUs across
tp/pp); ``shape_json["node_count"]`` (default 1) splits that world size across
N physical nodes for multinode TP/PP, requiring Grove or LWS+Volcano installed
-- the operator hard-rejects multinode DGDs without one.

Each pool is a self-contained DGD with its own Frontend. Cross-pool selection belongs to
the external Tandemn Router rather than Dynamo's GlobalRouter.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from tandemn_system_data.models.rank import Rank

from tandemn_orca.compiler_common import (
    k8s_name,
    labels,
    rank_node_count,
    required,
    validate_unique_ranks,
    worker_gpu_count,
    workload_name,
)

FRONTEND_IMAGE = "nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.2.1"
RUNTIME_IMAGE = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.1"
PLANNER_IMAGE = "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.2.1"

# Per-job router listen ports live above the rank tunnel range
# (launcher: 18000 + span 10000) so a router never collides with a tunnel.
ROUTER_LISTEN_PORT_BASE = 28000
ROUTER_LISTEN_PORT_SPAN = 1000


def router_listen_port(job_id: str) -> int:
    """Deterministic user-facing port for one job's router process."""
    digest = hashlib.sha256(job_id.encode()).digest()
    return ROUTER_LISTEN_PORT_BASE + int.from_bytes(digest[:4], "big") % ROUTER_LISTEN_PORT_SPAN


def compile_job(job_id: str, ranks: list[Rank], namespace: str = "default") -> list[dict[str, Any]]:
    validate_unique_ranks(ranks)
    return [render_rank_dgd(job_id, rank, namespace) for rank in ranks]


def render_router_config(
    job_id: str,
    ranks: list[Rank],
    max_num_seq_by_rank: dict[str, int],
    ports_by_rank: dict[str, int],
) -> dict[str, Any]:
    """Render the laptop router's per-job deployment registry."""
    plan_ids = {rank.plan_id for rank in ranks if rank.plan_id}
    if len(plan_ids) != 1 or len(ranks) == 0:
        raise ValueError("router config requires ranks from exactly one plan")

    deployments = []
    for rank in sorted(ranks, key=lambda item: item.rank_id):
        shape = rank.shape_json
        env = shape.get("env")
        if not isinstance(env, (list, tuple)) or len(env) != 5 or not all(env):
            raise ValueError(f"rank {rank.rank_id}: router config requires a five-part env")
        max_num_seq = max_num_seq_by_rank.get(rank.rank_id)
        if not max_num_seq:
            raise ValueError(f"rank {rank.rank_id}: max_num_seq must be a positive int")
        port = ports_by_rank.get(rank.rank_id)
        if not port:
            raise ValueError(f"rank {rank.rank_id}: local tunnel port is required")
        deployments.append(
            {
                "id": pool_dgd_name(job_id, rank),
                "rank_id": rank.rank_id,
                "endpoint": f"http://127.0.0.1:{port}",
                "env": list(env),
                "enabled": True,
                "max_num_seq": max_num_seq,
                "maximum_requests": rank.n_replicas * max_num_seq,
                # Replica ceiling for the overflow gate: above the KV
                # threshold the router sheds new sessions only once
                # ready_replicas reaches this, leaving earlier pressure to
                # local Dynamo autoscaling.
                "max_replicas": rank.n_replicas,
            }
        )

    plan_id = plan_ids.pop()
    return {
        "version": plan_id,
        "job_id": job_id,
        "listen_port": router_listen_port(job_id),
        "overflow_threshold": 0.8,
        "deployments": deployments,
    }


def pool_key(rank: Rank) -> str:
    shape = rank.shape_json
    return k8s_name(
        f"{rank.rank_id}-{required(shape, 'instance_type')}-{required(shape, 'gpu_type')}"
    )


def pool_dgd_name(job_id: str, rank: Rank) -> str:
    """Pool DGD name using the canonical rank ID."""
    return workload_name(job_id, rank.rank_id)


def pod_metadata(
    job_id: str, rank_id: str, plan_id: str | None, *, worker: bool = False
) -> dict[str, Any]:
    pod_labels = {
        "tandemn.com/job-id": job_id,
        "tandemn.com/rank-id": rank_id,
    }
    if plan_id:
        pod_labels["tandemn.com/plan-id"] = plan_id
    if worker:
        pod_labels["tandemn.com/pods-discovery"] = "dynamo-worker"
    return {"labels": pod_labels}


def dynamo_namespace(namespace: str, name: str) -> str:
    return f"{namespace}-{name}"


def render_rank_dgd(job_id: str, rank: Rank, namespace: str) -> dict[str, Any]:
    key = pool_key(rank)
    name = pool_dgd_name(job_id, rank)
    shape = rank.shape_json
    rank_id = rank.rank_id
    plan_id = rank.plan_id
    # gpu_count is one replica's world size (tp * pp, ... total GPUs); split
    # across node_count physical nodes for multinode TP/PP (default: one node,
    # gpus_per_node == gpu_count, byte-identical to the single-node DGD).
    gpu_count = worker_gpu_count(rank)
    node_count = rank_node_count(rank)
    if gpu_count % node_count != 0:
        raise ValueError(
            f"rank {rank.rank_id}: count={gpu_count} does not divide "
            f"evenly across node_count={node_count}"
        )
    gpus_per_node = gpu_count // node_count
    return {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {
            "name": name,
            "labels": labels(job_id, rank_id, plan_id, "pool", key),
        },
        "spec": {
            "backendFramework": required(shape, "engine_name"),
            "services": {
                "Frontend": {
                    "componentType": "frontend",
                    "replicas": 1,
                    "extraPodMetadata": pod_metadata(job_id, rank_id, plan_id),
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": FRONTEND_IMAGE,
                            "command": ["python3", "-m", "dynamo.frontend"],
                            "args": [
                                "--router-mode",
                                "round-robin",
                                "--model-name",
                                required(shape, "model_id"),
                            ],
                        }
                    },
                },
                "LocalRouter": {
                    "componentType": "default",
                    "replicas": 1,
                    "extraPodMetadata": pod_metadata(job_id, rank_id, plan_id),
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": RUNTIME_IMAGE,
                            "env": [{"name": "DYN_SYSTEM_PORT", "value": "9090"}],
                            "command": ["python3", "-m", "dynamo.router"],
                            "args": [
                                "--endpoint",
                                f"{dynamo_namespace(namespace, name)}.router.generate",
                                "--router-block-size",
                                "16",
                            ],
                        }
                    },
                },
                "VllmDecodeWorker": {
                    "componentType": "worker",
                    "subComponentType": "decode",
                    # No "replicas": the scaling adapter (DGDSA) owns it once
                    # enabled -- the in-pool Planner scales DP width within
                    # [min_endpoint, max_gpu_budget]. Claiming replicas here
                    # would make every Orca re-apply fight the adapter (webhook
                    # rejection or a reset to the initial value).
                    "scalingAdapter": {"enabled": True},
                    "extraPodMetadata": pod_metadata(job_id, rank_id, plan_id, worker=True),
                    "resources": {
                        "requests": {"gpu": str(gpus_per_node)},
                        "limits": {"gpu": str(gpus_per_node)},
                    },
                    # Only present for multinode replicas; the operator rejects
                    # multinode without Grove/LWS installed and asks LWS/Grove
                    # to gang-schedule this many pods, one per node.
                    **({"multinode": {"nodeCount": node_count}} if node_count > 1 else {}),
                    "extraPodSpec": {
                        "nodeSelector": node_selector(shape),
                        "tolerations": [
                            {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
                        ],
                        "mainContainer": {
                            "image": RUNTIME_IMAGE,
                            "workingDir": "/workspace/examples/backends/vllm",
                            "command": ["python3", "-m", "dynamo.vllm"],
                            "args": worker_args(shape),
                        },
                    },
                },
                "Planner": {
                    "componentType": "planner",
                    "replicas": 1,
                    "extraPodMetadata": pod_metadata(job_id, rank_id, plan_id),
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": PLANNER_IMAGE,
                            "command": ["python3", "-m", "dynamo.planner"],
                            "args": ["--config", planner_config(rank)],
                        }
                    },
                },
            },
        },
    }


def node_selector(shape: dict[str, Any]) -> dict[str, str]:
    capacity_type = capacity_type_for(shape)
    selector = {
        # "tandemn.com/launch-class": "tdm-gpu-cr" if capacity_type == "reserved" else "tdm-gpu-flex",
        "node.kubernetes.io/instance-type": required(shape, "instance_type"),
    }
    if capacity_type == "reserved":
        selector["karpenter.sh/capacity-type"] = "reserved"
    elif capacity_type in {"spot", "on-demand"}:
        selector["karpenter.sh/capacity-type"] = capacity_type
    return selector


def worker_args(shape: dict[str, Any]) -> list[str]:
    args = ["--model", required(shape, "model_id")]
    optional = {
        "tp": "--tensor-parallel-size",
        "pp": "--pipeline-parallel-size",
        "max_num_seq": "--max-num-seqs",
        "max_num_batched_tokens": "--max-num-batched-tokens",
        "gpu_mem_util": "--gpu-memory-utilization",
        "max_model_len": "--max-model-len",
        "block_size": "--block-size",
        "kvcache_dtype": "--kv-cache-dtype",
        "scheduling_policy": "--scheduling-policy",
    }
    for key, flag in optional.items():
        if key in shape:
            args.extend([flag, str(shape[key])])
    if shape.get("prefix_cache_enabled") is True:
        args.append("--enable-prefix-caching")
    if shape.get("chunked_prefill_enable") is True:
        args.append("--enable-chunked-prefill")
    dtype = _vllm_dtype(shape)
    if dtype:
        args.extend(["--dtype", dtype])
    quantization = shape.get("weight_quantization_method")
    if quantization and str(quantization).lower() != "none":
        args.extend(["--quantization", str(quantization)])
    if shape.get("spec_decoding_enabled") is True:
        spec_method = shape.get("spec_decoding_method")
        draft_model = shape.get("draft_model_id")
        spec_tokens = shape.get("num_speculative_tokens")
        if spec_method and str(spec_method).lower() != "none":
            args.extend(["--spec-method", str(spec_method)])
        if draft_model:
            args.extend(["--spec-model", str(draft_model)])
        if spec_tokens:
            args.extend(["--spec-tokens", str(spec_tokens)])
    return args


def _vllm_dtype(shape: dict[str, Any]) -> str | None:
    weight = shape.get("weight_dtype")
    activation = shape.get("activation_dtype")
    if weight and activation and str(weight) != str(activation):
        raise ValueError("weight_dtype and activation_dtype must match for vLLM --dtype")
    return str(weight or activation) if weight or activation else None


def planner_config(rank: Rank) -> str:
    """PlannerConfig for one pool's standalone (non-hierarchical) Planner.

    Field names/values match the documented PlannerConfig reference
    (docs/components/planner/planner-guide.md) -- a previous version of this
    function used several fields that do not exist in that schema
    (``global_planner_namespace``, ``decode_engine_num_gpu``, ``model_name``,
    ``pool_key``) and the wrong ``environment`` (``global-planner`` is only
    for the hierarchical GlobalPlanner setup; standalone per-pool Planners use
    ``kubernetes``, the schema default).

    ponytail: ``ttft_ms``/``itl_ms`` are the Reference-table field names; the
    guide's own DGDR example uses the shorter ``ttft``/``itl`` for the same
    fields, which the docs don't reconcile. Runtime-unverified which the
    installed Planner binary actually accepts.
    """
    shape = rank.shape_json
    config = {
        "mode": "agg",
        "backend": required(shape, "engine_name"),
        "environment": "kubernetes",
        "optimization_target": "sla",
        "enable_throughput_scaling": False,
        "enable_load_scaling": True,
        "pre_deployment_sweeping_mode": "none",
        "load_predictor": "prophet",
        "load_adjustment_interval_seconds": 5,
        "min_endpoint": 1,
        "max_gpu_budget": max_gpu_budget(rank),
        "ttft_ms": required_float(shape, "target_p99_ttft_ms"),
        "itl_ms": required_float(shape, "target_p99_tpot_ms"),
    }
    return json.dumps(config, separators=(",", ":"))


def capacity_type_for(shape: dict[str, Any]) -> str:
    env = shape.get("env")
    market = env[0] if isinstance(env, (list, tuple)) and env else "reserved"
    return {"reserved": "reserved", "on_demand": "on-demand"}.get(str(market), str(market))


def max_gpu_budget(rank: Rank) -> int:
    return worker_gpu_count(rank) * rank.n_replicas


def required_float(shape: dict[str, Any], key: str) -> float:
    value = shape.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"shape missing positive number {key!r}")
    return float(value)

"""Compile Orca Chain rows into Dynamo Kubernetes objects.

No Kubernetes calls live here; this file only returns dicts ready for server-side apply.
"""

from __future__ import annotations

import json
from typing import Any

from tandemn_system_data.models.chain import Chain

FRONTEND_IMAGE = "nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.0.2"
RUNTIME_IMAGE = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.2-efa-amd64"


def compile_job(job_id: str, chains: list[Chain], namespace: str = "default") -> list[dict[str, Any]]:
    groups = group_chains(chains)
    if not groups:
        return []
    return [
        render_router_configmap(job_id, groups, namespace),
        render_control_dgd(job_id, groups, namespace),
        *(render_pool_dgd(job_id, key, group, namespace) for key, group in groups.items()),
    ]


def group_chains(chains: list[Chain]) -> dict[str, list[Chain]]:
    grouped: dict[str, list[Chain]] = {}
    for chain in chains:
        grouped.setdefault(pool_key(chain), []).append(chain)
    return dict(grouped)


def pool_key(chain: Chain) -> str:
    shape = chain.shape_json
    return k8s_name(
        f"{chain.role}-{required(shape, 'instance_type')}-{required(shape, 'gpu_type')}"
    )


def dgd_name(job_id: str, suffix: str) -> str:
    return k8s_name(f"{job_id}-{suffix}")


def k8s_name(value: str) -> str:
    value = value.lower().replace("_", "-").replace(".", "-")
    return "".join(ch for ch in value if ch.isalnum() or ch == "-")[:63].strip("-")


def labels(
    job_id: str,
    plan_id: str | None,
    resource_kind: str,
    pool: str | None = None,
) -> dict[str, str]:
    result = {
        "tandemn.ai/managed-by": "orca",
        "tandemn.ai/job-id": job_id,
        "tandemn.ai/resource-kind": resource_kind,
    }
    if plan_id:
        result["tandemn.ai/plan-id"] = plan_id
    if pool:
        result["tandemn.ai/pool-key"] = pool
    return result


def dynamo_namespace(namespace: str, name: str) -> str:
    return f"{namespace}-{name}"


def render_router_configmap(
    job_id: str, groups: dict[str, list[Chain]], namespace: str
) -> dict[str, Any]:
    name = dgd_name(job_id, "global-router-config")
    plan_id = first_chain(groups).plan_id
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": name, "labels": labels(job_id, plan_id, "control")},
        "data": {"global_router_config.json": render_router_config(job_id, groups, namespace)},
    }


def render_router_config(job_id: str, groups: dict[str, list[Chain]], namespace: str) -> str:
    pool_namespaces = [
        dynamo_namespace(namespace, dgd_name(job_id, key)) for key in groups
    ]
    shape = first_chain(groups).shape_json
    pool_count = len(pool_namespaces)
    config = {
        "mode": "agg",
        "num_agg_pools": pool_count,
        "agg_pool_dynamo_namespaces": pool_namespaces,
        "agg_pool_selection_strategy": {
            "ttft_min": 0,
            "ttft_max": required_float(shape, "target_p99_ttft_ms"),
            "ttft_resolution": pool_count,
            "itl_min": 0,
            "itl_max": required_float(shape, "target_p99_tpot_ms"),
            "itl_resolution": 1,
            "agg_pool_mapping": [[pool] for pool in range(pool_count)],
            "priority_overrides": [],
        },
    }
    return json.dumps(config, separators=(",", ":"))


def render_control_dgd(job_id: str, groups: dict[str, list[Chain]], namespace: str) -> dict[str, Any]:
    name = dgd_name(job_id, "ctrl")
    router_config_name = dgd_name(job_id, "global-router-config")
    shape = first_chain(groups).shape_json
    model = required(shape, "model_id")
    plan_id = first_chain(groups).plan_id
    managed_namespaces = [dynamo_namespace(namespace, dgd_name(job_id, key)) for key in groups]
    return {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": name, "labels": labels(job_id, plan_id, "control")},
        "spec": {
            "services": {
                "Frontend": {
                    "componentType": "frontend",
                    "replicas": 1,
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": FRONTEND_IMAGE,
                            "command": ["python3", "-m", "dynamo.frontend"],
                            "args": [
                                "--router-mode",
                                "round-robin",
                                "--namespace",
                                dynamo_namespace(namespace, name),
                                "--model-name",
                                model,
                            ],
                        }
                    },
                },
                "GlobalRouter": {
                    "componentType": "default",
                    "replicas": 1,
                    "extraPodSpec": {
                        "volumes": [
                            {"name": "global-router-config", "configMap": {"name": router_config_name}}
                        ],
                        "mainContainer": {
                            "image": RUNTIME_IMAGE,
                            "command": ["python3", "-m", "dynamo.global_router"],
                            "args": [
                                "--config",
                                "/config/global_router_config.json",
                                "--model-name",
                                model,
                                "--namespace",
                                dynamo_namespace(namespace, name),
                                "--default-ttft-target",
                                str(required_float(shape, "target_p99_ttft_ms")),
                                "--default-itl-target",
                                str(required_float(shape, "target_p99_tpot_ms")),
                            ],
                            "volumeMounts": [
                                {
                                    "name": "global-router-config",
                                    "mountPath": "/config",
                                    "readOnly": True,
                                }
                            ],
                        },
                    },
                },
                "GlobalPlanner": {
                    "componentType": "planner",
                    "replicas": 1,
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": RUNTIME_IMAGE,
                            "command": ["python3", "-m", "dynamo.global_planner"],
                            "args": [
                                "--managed-namespaces",
                                *managed_namespaces,
                                "--max-total-gpus",
                                str(sum(max_gpu_budget(group) for group in groups.values())),
                            ],
                        }
                    },
                },
            }
        },
    }


def render_pool_dgd(
    job_id: str, key: str, chains: list[Chain], namespace: str
) -> dict[str, Any]:
    name = dgd_name(job_id, key)
    shape = chains[0].shape_json
    plan_id = chains[0].plan_id
    gpu_count = worker_gpu_count(chains)
    return {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": name, "labels": labels(job_id, plan_id, "pool", key)},
        "spec": {
            "services": {
                "LocalRouter": {
                    "componentType": "default",
                    "replicas": 1,
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
                    "replicas": 1,
                    "scalingAdapter": {"enabled": True},
                    "resources": {
                        "requests": {"gpu": str(gpu_count)},
                        "limits": {"gpu": str(gpu_count)},
                    },
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
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": RUNTIME_IMAGE,
                            "command": ["python3", "-m", "dynamo.planner"],
                            "args": ["--config", planner_config(job_id, key, chains, namespace)],
                        }
                    },
                },
            }
        },
    }


def node_selector(shape: dict[str, Any]) -> dict[str, str]:
    capacity_type = capacity_type_for(shape)
    selector = {
        # "tandemn.ai/launch-class": "tdm-gpu-cr" if capacity_type == "reserved" else "tdm-gpu-flex",
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
        "max_num_seq": "--max-num-seqs",
        "max_num_batched_tokens": "--max-num-batched-tokens",
        "gpu_mem_util": "--gpu-memory-utilization",
    }
    for key, flag in optional.items():
        if key in shape:
            args.extend([flag, str(shape[key])])
    if shape.get("prefix_cache_enabled") is True:
        args.append("--enable-prefix-caching")
    return args


def planner_config(job_id: str, key: str, chains: list[Chain], namespace: str) -> str:
    shape = chains[0].shape_json
    config = {
        "environment": "global-planner",
        "global_planner_namespace": dynamo_namespace(namespace, dgd_name(job_id, "ctrl")),
        "backend": required(shape, "engine_name"),
        "mode": "agg",
        "optimization_target": "sla",
        "ttft": required_float(shape, "target_p99_ttft_ms"),
        "itl": required_float(shape, "target_p99_tpot_ms"),
        "enable_throughput_scaling": False,
        "enable_load_scaling": True,
        "pre_deployment_sweeping_mode": "none",
        "load_predictor": "prophet",
        "load_adjustment_interval": 5,
        "min_endpoint": 1,
        "max_gpu_budget": max_gpu_budget(chains),
        "decode_engine_num_gpu": worker_gpu_count(chains),
        "model_name": required(shape, "model_id"),
        "pool_key": key,
    }
    return json.dumps(config, separators=(",", ":"))


def capacity_type_for(shape: dict[str, Any]) -> str:
    env = shape.get("env")
    market = env[0] if isinstance(env, (list, tuple)) and env else "reserved"
    return {"reserved": "reserved", "on_demand": "on-demand"}.get(str(market), str(market))


def worker_gpu_count(chains: list[Chain]) -> int:
    count = chains[0].shape_json.get("count")
    if type(count) is not int or count <= 0:
        raise ValueError(f"chain {chains[0].chain_id} missing positive int count")
    return count


def max_gpu_budget(chains: list[Chain]) -> int:
    total = 0
    for chain in chains:
        count = chain.shape_json.get("count")
        if type(count) is not int or count <= 0:
            raise ValueError(f"chain {chain.chain_id} missing positive int count")
        total += count
    return total


def first_chain(groups: dict[str, list[Chain]]) -> Chain:
    return next(iter(groups.values()))[0]


def required(shape: dict[str, Any], key: str) -> str:
    value = shape.get(key)
    if value is None:
        raise ValueError(f"shape missing {key!r}")
    return str(value)


def required_float(shape: dict[str, Any], key: str) -> float:
    value = shape.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"shape missing positive number {key!r}")
    return float(value)

"""Compile Orca Chain rows into Dynamo Kubernetes objects.

No Kubernetes calls live here; this file only returns dicts ready for server-side apply.

ponytail: GlobalRouter/GlobalPlanner (and the router ConfigMap) are currently unused in
practice -- Koi owns cross-pool GPU budgeting at plan time, and single-pool jobs need no
cross-pool routing. Kept for when multi-pool SLA routing / fast-loop rebalancing is needed.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from tandemn_system_data.models.chain import Chain

FRONTEND_IMAGE = "nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.0.2"
RUNTIME_IMAGE = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.2-efa-amd64"

# The Dynamo operator's admission webhook requires len(DGD name) +
# len(service name) <= 45 for pod naming; our longest service is
# "VllmDecodeWorker" (16), so DGD names must stay within 29.
MAX_DGD_NAME = 29


def compile_job(
    job_id: str, chains: list[Chain], namespace: str = "default"
) -> list[dict[str, Any]]:
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
    # rank_id (Koi's ladder rank id, e.g. "rank_0") pools per rung so the
    # rung's pods can carry one tandemn.ai/rank-id label; without it,
    # same-shape rungs merge and rank identity is not attributable.
    if shape.get("rank_id"):
        return k8s_name(
            f"{shape['rank_id']}-{required(shape, 'instance_type')}-{required(shape, 'gpu_type')}"
        )
    return k8s_name(
        f"{chain.role}-{required(shape, 'instance_type')}-{required(shape, 'gpu_type')}"
    )


def dgd_name(job_id: str, suffix: str) -> str:
    """Short DGD name: ``tdm-{job tag}-{suffix}``, capped at MAX_DGD_NAME.

    The job tag is the ULID tail, not the full job id (a full
    ``job-01kwwez...`` already busts the operator's 45-char pod-naming
    budget); the full job_id lives in the tandemn.ai/job-id label. A suffix
    that would not fit is replaced by an 8-char hash of itself.
    """
    tag = k8s_name(job_id).removeprefix("job-")[-10:].strip("-")
    name = k8s_name(f"tdm-{tag}-{suffix}")
    if len(name) > MAX_DGD_NAME:
        digest = hashlib.sha1(suffix.encode()).hexdigest()[:8]
        name = k8s_name(f"tdm-{tag}-{digest}")
    return name


def k8s_name(value: str) -> str:
    value = value.lower().replace("_", "-").replace(".", "-")
    return "".join(ch for ch in value if ch.isalnum() or ch == "-")[:63].strip("-")


def pool_dgd_name(job_id: str, key: str, chains: list[Chain]) -> str:
    """Pool DGD name, preferring the short rank_id over the descriptive key."""
    return dgd_name(job_id, str(chains[0].shape_json.get("rank_id") or key))


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
    name = dgd_name(job_id, "grc")
    plan_id = first_chain(groups).plan_id
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": name, "labels": labels(job_id, plan_id, "control")},
        "data": {"global_router_config.json": render_router_config(job_id, groups, namespace)},
    }


def render_router_config(job_id: str, groups: dict[str, list[Chain]], namespace: str) -> str:
    pool_namespaces = [
        dynamo_namespace(namespace, pool_dgd_name(job_id, key, group))
        for key, group in groups.items()
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


def render_control_dgd(
    job_id: str, groups: dict[str, list[Chain]], namespace: str
) -> dict[str, Any]:
    name = dgd_name(job_id, "ctrl")
    router_config_name = dgd_name(job_id, "grc")
    shape = first_chain(groups).shape_json
    model = required(shape, "model_id")
    plan_id = first_chain(groups).plan_id
    managed_namespaces = [
        dynamo_namespace(namespace, pool_dgd_name(job_id, key, group))
        for key, group in groups.items()
    ]
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
                            # Pool DGDs are named {tdm-tag}-{rank}, so the
                            # {namespace}-{tdm-tag} prefix lets the Frontend
                            # discover workers in every pool namespace
                            # (runtime-verified; a plain --namespace only sees
                            # the ctrl namespace and serves no models).
                            "env": [
                                {
                                    "name": "DYN_NAMESPACE_PREFIX",
                                    "value": dynamo_namespace(namespace, dgd_name(job_id, "")),
                                }
                            ],
                            "command": ["python3", "-m", "dynamo.frontend"],
                            "args": [
                                "--router-mode",
                                "round-robin",
                                "--namespace-prefix",
                                dynamo_namespace(namespace, dgd_name(job_id, "")),
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
                            {
                                "name": "global-router-config",
                                "configMap": {"name": router_config_name},
                            }
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


def render_pool_dgd(job_id: str, key: str, chains: list[Chain], namespace: str) -> dict[str, Any]:
    name = pool_dgd_name(job_id, key, chains)
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
                    # No "replicas": the scaling adapter (DGDSA) owns it once
                    # enabled -- the in-pool Planner scales DP width within
                    # [min_endpoint, max_gpu_budget]. Claiming replicas here
                    # would make every Orca re-apply fight the adapter (webhook
                    # rejection or a reset to the initial value).
                    "scalingAdapter": {"enabled": True},
                    # The gpu-metrics collector lifts these pod labels: job-id
                    # keys the pod to its job's chain rows, rank-id groups the
                    # rung's chains/GPUs under one rank in gpu_metrics.
                    "extraPodMetadata": {
                        "labels": {
                            "tandemn.ai/job-id": job_id,
                            **(
                                {"tandemn.ai/rank-id": str(shape["rank_id"])}
                                if shape.get("rank_id")
                                else {}
                            ),
                        }
                    },
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

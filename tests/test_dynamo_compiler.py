from __future__ import annotations

import json

import pytest
from tandemn_system_data.models.enums import RankRole
from tandemn_system_data.models.rank import Rank

from tandemn_orca.dynamo_compiler import (
    compile_job,
    node_selector,
    pool_key,
    render_rank_dgd,
    render_router_configmap,
    render_router_objects,
    worker_args,
)

RANK_IDS = [
    "rank_01JBM2YQYZ1KQ9C8GZP1XB6V5T",
    "rank_01JBM30YQ7X3WQAR6HF8C2Q9T8",
    "rank_01JBM31YQ7X3WQAR6HF8C2Q9T8",
]


def _rank(
    instance_type: str,
    gpu_type: str,
    plan_id: str = "plan_1",
    rank_id: str = RANK_IDS[0],
    n_replicas: int = 1,
) -> Rank:
    return Rank(
        rank_id=rank_id,
        job_id="job_online_001",
        plan_id=plan_id,
        role=RankRole.AGGREGATE,
        n_replicas=n_replicas,
        shape_json={
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "engine_name": "vllm",
            "instance_type": instance_type,
            "gpu_type": gpu_type,
            "gpu_count": 1,
            "count": 1,
            "target_p99_ttft_ms": 500.0,
            "target_p99_tpot_ms": 50.0,
            "tp": 1,
            "max_num_seq": 64,
            "max_num_batched_tokens": 32768,
            "max_model_len": 8192,
            "block_size": 16,
            "kvcache_dtype": "auto",
            "gpu_mem_util": 0.9,
            "scheduling_policy": "fcfs",
            "prefix_cache_enabled": True,
            "chunked_prefill_enable": True,
            "weight_dtype": "bfloat16",
            "activation_dtype": "bfloat16",
            "weight_quantization_method": "none",
            "env": ["reserved", "aws", "us-east-2", "use2-az3", gpu_type],
        },
    )


def _objects_by_name(objects: list[dict]) -> dict[str, dict]:
    return {obj["metadata"]["name"]: obj for obj in objects}


def test_compile_job_renders_self_contained_dgd_for_each_gpu_pool():
    objects = compile_job(
        "job_online_001",
        [
            _rank("p5.48xlarge", "H100", rank_id=RANK_IDS[0]),
            _rank("g6e.12xlarge", "L40S", rank_id=RANK_IDS[1]),
            _rank("g5.4xlarge", "A10", rank_id=RANK_IDS[2]),
        ],
    )
    by_name = _objects_by_name(objects)

    assert len(by_name) == 3

    h100 = next(
        obj for obj in objects if obj["metadata"]["labels"]["tandemn.com/rank-id"] == RANK_IDS[0]
    )
    assert list(h100["spec"]["services"]) == [
        "Frontend",
        "LocalRouter",
        "VllmDecodeWorker",
        "Planner",
    ]
    frontend = h100["spec"]["services"]["Frontend"]
    assert frontend["extraPodSpec"]["mainContainer"]["args"] == [
        "--router-mode",
        "round-robin",
        "--model-name",
        "meta-llama/Llama-3.1-8B-Instruct",
    ]
    assert h100["metadata"]["labels"]["tandemn.com/job-id"] == "job_online_001"
    assert h100["metadata"]["labels"]["tandemn.com/rank-id"] == RANK_IDS[0]
    for service_name, service in h100["spec"]["services"].items():
        expected = {
            "tandemn.com/job-id": "job_online_001",
            "tandemn.com/rank-id": RANK_IDS[0],
            "tandemn.com/plan-id": "plan_1",
        }
        if service_name == "VllmDecodeWorker":
            expected["tandemn.com/pods-discovery"] = "dynamo-worker"
        assert service["extraPodMetadata"] == {"labels": expected}
    worker = h100["spec"]["services"]["VllmDecodeWorker"]
    assert worker["extraPodSpec"]["nodeSelector"] == {
        "node.kubernetes.io/instance-type": "p5.48xlarge",
        "karpenter.sh/capacity-type": "reserved",
    }
    assert worker["resources"] == {"requests": {"gpu": "1"}, "limits": {"gpu": "1"}}
    assert worker["extraPodSpec"]["mainContainer"]["args"] == [
        "--model",
        "meta-llama/Llama-3.1-8B-Instruct",
        "--tensor-parallel-size",
        "1",
        "--max-num-seqs",
        "64",
        "--max-num-batched-tokens",
        "32768",
        "--gpu-memory-utilization",
        "0.9",
        "--max-model-len",
        "8192",
        "--block-size",
        "16",
        "--kv-cache-dtype",
        "auto",
        "--scheduling-policy",
        "fcfs",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--dtype",
        "bfloat16",
    ]


def test_one_rank_compiles_to_one_pool_with_replica_budget():
    objects = compile_job("job_online_001", [_rank("g6e.12xlarge", "L40S", n_replicas=3)])
    by_name = _objects_by_name(objects)

    assert len(by_name) == 1
    pool = next(iter(by_name.values()))
    planner = pool["spec"]["services"]["Planner"]
    planner_config = json.loads(planner["extraPodSpec"]["mainContainer"]["args"][1])

    # The DGDSA owns worker replicas (scalingAdapter enabled); Orca must not
    # claim the field or every re-apply would fight the Planner's scaling.
    worker = pool["spec"]["services"]["VllmDecodeWorker"]
    assert "replicas" not in worker
    assert worker["scalingAdapter"] == {"enabled": True}
    assert planner_config["environment"] == "kubernetes"
    assert planner_config["load_predictor"] == "prophet"
    assert planner_config["ttft_ms"] == 500.0
    assert planner_config["itl_ms"] == 50.0
    assert planner_config["max_gpu_budget"] == 3
    assert "global_planner_namespace" not in planner_config
    assert "decode_engine_num_gpu" not in planner_config
    assert "model_name" not in planner_config
    assert "pool_key" not in planner_config


def test_router_configmap_matches_router_json_contract():
    rank = _rank("g6e.12xlarge", "L40S")
    rank.shape_json["router_endpoint"] = "https://rank.example.internal"

    configmap = render_router_configmap("job_online_001", [rank], {rank.rank_id: 256}, "routing")
    config = json.loads(configmap["data"]["router.json"])

    assert configmap["metadata"]["namespace"] == "routing"
    assert configmap["metadata"]["labels"]["tandemn.com/job-id"] == "job_online_001"
    assert config == {
        "version": "plan_1",
        "job_id": "job_online_001",
        "overflow_threshold": 0.8,
        "deployments": [
            {
                "id": "tdm-online-001-03a2a00c",
                "endpoint": "https://rank.example.internal",
                "env": ["reserved", "aws", "us-east-2", "use2-az3", "L40S"],
                "enabled": True,
                "max_num_seq": 256,
                "maximum_requests": 256,
            }
        ],
    }


def test_router_objects_mount_config_and_expose_service():
    rank = _rank("g6e.12xlarge", "L40S")
    rank.shape_json["router_endpoint"] = "https://rank.example.internal"

    configmap, deployment, service = render_router_objects(
        "job_online_001",
        [rank],
        {rank.rank_id: 256},
        "registry.example/tandemn-router:test",
        "routing",
    )

    assert configmap["metadata"]["name"] == "tdm-online-001-router-config"
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == "registry.example/tandemn-router:test"
    assert container["volumeMounts"] == [
        {"name": "config", "mountPath": "/config", "readOnly": True}
    ]
    assert deployment["spec"]["template"]["spec"]["volumes"] == [
        {"name": "config", "configMap": {"name": "tdm-online-001-router-config"}}
    ]
    assert service["spec"]["ports"] == [{"name": "http", "port": 80, "targetPort": "http"}]


def test_rank_id_pools_per_rung():
    first = _rank("g6e.12xlarge", "L40S", rank_id=RANK_IDS[0])
    second = _rank("g6e.12xlarge", "L40S", rank_id=RANK_IDS[1])

    objects = compile_job("job_online_001", [first, second])
    by_name = _objects_by_name(objects)

    # Same-shape rungs stay separate pools when rank_id is present.
    assert len(by_name) == 2


def test_compile_job_rejects_duplicate_rank_ids():
    with pytest.raises(ValueError, match="duplicate rank_id"):
        compile_job(
            "job_online_001",
            [_rank("g6e.12xlarge", "L40S"), _rank("g6e.12xlarge", "L40S")],
        )


def test_worker_args_maps_tp_and_pp():
    shape = {"model_id": "m", "tp": 2, "pp": 4}
    args = worker_args(shape)
    assert "--tensor-parallel-size" in args
    assert args[args.index("--tensor-parallel-size") + 1] == "2"
    assert "--pipeline-parallel-size" in args
    assert args[args.index("--pipeline-parallel-size") + 1] == "4"


def test_worker_args_omits_pp_when_absent():
    args = worker_args({"model_id": "m", "tp": 1})
    assert "--pipeline-parallel-size" not in args


def test_pool_key_and_selector_are_from_rank_shape():
    rank = _rank("g5.4xlarge", "A10")

    assert pool_key(rank) == f"{RANK_IDS[0].replace('_', '-')}-g5-4xlarge-a10".lower()
    assert node_selector(rank.shape_json) == {
        "node.kubernetes.io/instance-type": "g5.4xlarge",
        "karpenter.sh/capacity-type": "reserved",
    }


def test_worker_args_adds_quantization_and_spec_decoding_flags():
    shape = dict(_rank("p5.48xlarge", "H100").shape_json)
    shape.update(
        {
            "weight_quantization_method": "awq",
            "spec_decoding_enabled": True,
            "spec_decoding_method": "draft_model",
            "draft_model_id": "draft/model",
            "num_speculative_tokens": 4,
        }
    )

    args = worker_args(shape)

    assert "--quantization" in args
    assert args[args.index("--quantization") + 1] == "awq"
    assert "--spec-method" in args
    assert args[args.index("--spec-method") + 1] == "draft_model"
    assert args[args.index("--spec-model") + 1] == "draft/model"
    assert args[args.index("--spec-tokens") + 1] == "4"


def test_single_node_rank_has_no_multinode_block():
    rank = _rank("g6.xlarge", "L4")
    dgd = render_rank_dgd("job_1", rank, "default")
    worker = dgd["spec"]["services"]["VllmDecodeWorker"]

    assert "multinode" not in worker
    assert worker["resources"] == {"requests": {"gpu": "1"}, "limits": {"gpu": "1"}}


def test_multinode_rank_splits_gpus_per_node():
    rank = _rank("g6.xlarge", "L4")
    rank.shape_json.update({"count": 2, "node_count": 2, "tp": 1, "pp": 2})
    dgd = render_rank_dgd("job_1", rank, "default")
    worker = dgd["spec"]["services"]["VllmDecodeWorker"]

    assert worker["multinode"] == {"nodeCount": 2}
    # 2 total GPUs / 2 nodes = 1 GPU requested per node, not the world size.
    assert worker["resources"] == {"requests": {"gpu": "1"}, "limits": {"gpu": "1"}}
    args = worker["extraPodSpec"]["mainContainer"]["args"]
    assert args[args.index("--pipeline-parallel-size") + 1] == "2"


def test_multinode_uneven_gpu_split_raises():
    rank = _rank("g6.xlarge", "L4")
    rank.shape_json.update({"count": 3, "node_count": 2})
    with pytest.raises(ValueError, match="does not divide evenly"):
        render_rank_dgd("job_1", rank, "default")

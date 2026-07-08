from __future__ import annotations

import json

from tandemn_system_data.models.chain import Chain
from tandemn_system_data.models.enums import ChainRole

from tandemn_orca.dynamo_compiler import (
    compile_job,
    group_chains,
    node_selector,
    pool_key,
    worker_args,
)


def _chain(instance_type: str, gpu_type: str, plan_id: str = "plan_1") -> Chain:
    return Chain(
        job_id="job_online_001",
        plan_id=plan_id,
        role=ChainRole.AGGREGATE,
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


def test_compile_job_renders_global_topology_for_three_gpu_pools():
    objects = compile_job(
        "job_online_001",
        [
            _chain("p5.48xlarge", "H100"),
            _chain("g6e.12xlarge", "L40S"),
            _chain("g5.4xlarge", "A10"),
        ],
    )
    by_name = _objects_by_name(objects)

    assert list(by_name) == [
        "tdm-online-001-grc",
        "tdm-online-001-ctrl",
        "tdm-online-001-c1c7dc72",
        "tdm-online-001-0dfb3ae9",
        "tdm-online-001-9a5f4df3",
    ]

    router_config = json.loads(by_name["tdm-online-001-grc"]["data"]["global_router_config.json"])
    assert router_config["mode"] == "agg"
    assert router_config["agg_pool_dynamo_namespaces"] == [
        "default-tdm-online-001-c1c7dc72",
        "default-tdm-online-001-0dfb3ae9",
        "default-tdm-online-001-9a5f4df3",
    ]
    router_strategy = router_config["agg_pool_selection_strategy"]
    assert router_strategy["ttft_min"] == 0
    assert router_strategy["ttft_max"] == 500.0
    assert router_strategy["itl_min"] == 0
    assert router_strategy["itl_max"] == 50.0
    assert router_strategy["agg_pool_mapping"] == [[0], [1], [2]]
    assert router_strategy["priority_overrides"] == []

    global_router = by_name["tdm-online-001-ctrl"]["spec"]["services"]["GlobalRouter"]
    assert global_router["extraPodSpec"]["mainContainer"]["args"][-4:] == [
        "--default-ttft-target",
        "500.0",
        "--default-itl-target",
        "50.0",
    ]

    h100 = by_name["tdm-online-001-c1c7dc72"]
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


def test_duplicate_chains_compile_to_one_pool_with_max_budget():
    objects = compile_job("job_online_001", [_chain("g6e.12xlarge", "L40S") for _ in range(3)])
    by_name = _objects_by_name(objects)

    assert list(by_name) == [
        "tdm-online-001-grc",
        "tdm-online-001-ctrl",
        "tdm-online-001-0dfb3ae9",
    ]
    pool = by_name["tdm-online-001-0dfb3ae9"]
    planner = pool["spec"]["services"]["Planner"]
    planner_config = json.loads(planner["extraPodSpec"]["mainContainer"]["args"][1])

    # The DGDSA owns worker replicas (scalingAdapter enabled); Orca must not
    # claim the field or every re-apply would fight the Planner's scaling.
    worker = pool["spec"]["services"]["VllmDecodeWorker"]
    assert "replicas" not in worker
    assert worker["scalingAdapter"] == {"enabled": True}
    assert planner_config["load_predictor"] == "prophet"
    assert planner_config["ttft"] == 500.0
    assert planner_config["itl"] == 50.0
    assert planner_config["max_gpu_budget"] == 3
    assert planner_config["decode_engine_num_gpu"] == 1


def test_rank_id_pools_per_rung_and_labels_worker_pods():
    first = _chain("g6e.12xlarge", "L40S")
    second = _chain("g6e.12xlarge", "L40S")
    first.shape_json["rank_id"] = "rank_0"
    second.shape_json["rank_id"] = "rank_1"

    objects = compile_job("job_online_001", [first, second])
    by_name = _objects_by_name(objects)

    # Same-shape rungs stay separate pools when rank_id is present.
    assert list(by_name) == [
        "tdm-online-001-grc",
        "tdm-online-001-ctrl",
        "tdm-online-001-rank-0",
        "tdm-online-001-rank-1",
    ]
    worker = by_name["tdm-online-001-rank-0"]["spec"]["services"]["VllmDecodeWorker"]
    assert worker["extraPodMetadata"] == {
        "labels": {"tandemn.ai/job-id": "job_online_001", "tandemn.ai/rank-id": "rank_0"}
    }


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


def test_pool_key_and_selector_are_from_chain_shape():
    chain = _chain("g5.4xlarge", "A10")

    assert pool_key(chain) == "aggregate-g5-4xlarge-a10"
    assert list(group_chains([chain])) == ["aggregate-g5-4xlarge-a10"]
    assert node_selector(chain.shape_json) == {
        "node.kubernetes.io/instance-type": "g5.4xlarge",
        "karpenter.sh/capacity-type": "reserved",
    }


def test_worker_args_adds_quantization_and_spec_decoding_flags():
    shape = dict(_chain("p5.48xlarge", "H100").shape_json)
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

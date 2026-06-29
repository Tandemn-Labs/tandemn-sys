from __future__ import annotations

import json

from tandemn_system_data.models.chain import Chain
from tandemn_system_data.models.enums import ChainRole

from tandemn_orca.dynamo_compiler import compile_job, group_chains, node_selector, pool_key


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
            "gpu_mem_util": 0.9,
            "prefix_cache_enabled": True,
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
        "job-online-001-global-router-config",
        "job-online-001-ctrl",
        "job-online-001-aggregate-p5-48xlarge-h100",
        "job-online-001-aggregate-g6e-12xlarge-l40s",
        "job-online-001-aggregate-g5-4xlarge-a10",
    ]

    router_config = json.loads(
        by_name["job-online-001-global-router-config"]["data"]["global_router_config.json"]
    )
    assert router_config["mode"] == "agg"
    assert router_config["agg_pool_dynamo_namespaces"] == [
        "default-job-online-001-aggregate-p5-48xlarge-h100",
        "default-job-online-001-aggregate-g6e-12xlarge-l40s",
        "default-job-online-001-aggregate-g5-4xlarge-a10",
    ]
    router_strategy = router_config["agg_pool_selection_strategy"]
    assert router_strategy["ttft_min"] == 0
    assert router_strategy["ttft_max"] == 500.0
    assert router_strategy["itl_min"] == 0
    assert router_strategy["itl_max"] == 50.0
    assert router_strategy["agg_pool_mapping"] == [[0], [1], [2]]
    assert router_strategy["priority_overrides"] == []

    global_router = by_name["job-online-001-ctrl"]["spec"]["services"]["GlobalRouter"]
    assert global_router["extraPodSpec"]["mainContainer"]["args"][-4:] == [
        "--default-ttft-target",
        "500.0",
        "--default-itl-target",
        "50.0",
    ]

    h100 = by_name["job-online-001-aggregate-p5-48xlarge-h100"]
    worker = h100["spec"]["services"]["VllmDecodeWorker"]
    assert worker["extraPodSpec"]["nodeSelector"] == {
        "tandemn.ai/launch-class": "tdm-gpu-cr",
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
        "--enable-prefix-caching",
    ]


def test_duplicate_chains_compile_to_one_pool_with_max_budget():
    objects = compile_job("job_online_001", [_chain("g6e.12xlarge", "L40S") for _ in range(3)])
    by_name = _objects_by_name(objects)

    assert list(by_name) == [
        "job-online-001-global-router-config",
        "job-online-001-ctrl",
        "job-online-001-aggregate-g6e-12xlarge-l40s",
    ]
    pool = by_name["job-online-001-aggregate-g6e-12xlarge-l40s"]
    planner = pool["spec"]["services"]["Planner"]
    planner_config = json.loads(planner["extraPodSpec"]["mainContainer"]["args"][1])

    assert pool["spec"]["services"]["VllmDecodeWorker"]["replicas"] == 1
    assert planner_config["load_predictor"] == "prophet"
    assert planner_config["ttft"] == 500.0
    assert planner_config["itl"] == 50.0
    assert planner_config["max_gpu_budget"] == 3
    assert planner_config["decode_engine_num_gpu"] == 1


def test_pool_key_and_selector_are_from_chain_shape():
    chain = _chain("g5.4xlarge", "A10")

    assert pool_key(chain) == "aggregate-g5-4xlarge-a10"
    assert list(group_chains([chain])) == ["aggregate-g5-4xlarge-a10"]
    assert node_selector(chain.shape_json) == {
        "tandemn.ai/launch-class": "tdm-gpu-cr",
        "node.kubernetes.io/instance-type": "g5.4xlarge",
        "karpenter.sh/capacity-type": "reserved",
    }

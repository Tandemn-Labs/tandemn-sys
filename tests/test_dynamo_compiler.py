from __future__ import annotations

import json

import pytest
from tandemn_system_data.models.chain import Chain
from tandemn_system_data.models.enums import ChainRole

from tandemn_orca.dynamo_compiler import (
    compile_job,
    group_chains,
    node_selector,
    pool_key,
    render_pool_dgd,
    worker_args,
)


def _chain(
    instance_type: str, gpu_type: str, plan_id: str = "plan_1", rank_id: str = "rank_0"
) -> Chain:
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
            "rank_id": rank_id,
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
            _chain("p5.48xlarge", "H100", rank_id="rank_0"),
            _chain("g6e.12xlarge", "L40S", rank_id="rank_1"),
            _chain("g5.4xlarge", "A10", rank_id="rank_2"),
        ],
    )
    by_name = _objects_by_name(objects)

    assert list(by_name) == [
        "tdm-online-001-rank-0",
        "tdm-online-001-rank-1",
        "tdm-online-001-rank-2",
    ]

    h100 = by_name["tdm-online-001-rank-0"]
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
    assert h100["metadata"]["labels"]["tandemn.com/rank-id"] == "rank_0"
    for service_name, service in h100["spec"]["services"].items():
        expected = {
            "tandemn.com/job-id": "job_online_001",
            "tandemn.com/rank-id": "rank_0",
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


def test_duplicate_chains_compile_to_one_pool_with_max_budget():
    objects = compile_job("job_online_001", [_chain("g6e.12xlarge", "L40S") for _ in range(3)])
    by_name = _objects_by_name(objects)

    assert list(by_name) == [
        "tdm-online-001-rank-0",
    ]
    pool = by_name["tdm-online-001-rank-0"]
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


def test_rank_id_pools_per_rung():
    first = _chain("g6e.12xlarge", "L40S")
    second = _chain("g6e.12xlarge", "L40S")
    first.shape_json["rank_id"] = "rank_0"
    second.shape_json["rank_id"] = "rank_1"

    objects = compile_job("job_online_001", [first, second])
    by_name = _objects_by_name(objects)

    # Same-shape rungs stay separate pools when rank_id is present.
    assert list(by_name) == [
        "tdm-online-001-rank-0",
        "tdm-online-001-rank-1",
    ]


def test_compile_job_requires_rank_id():
    chain = _chain("g6e.12xlarge", "L40S")
    del chain.shape_json["rank_id"]

    with pytest.raises(ValueError, match="rank_id"):
        compile_job("job_online_001", [chain])


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

    assert pool_key(chain) == "rank-0-g5-4xlarge-a10"
    assert list(group_chains([chain])) == ["rank-0-g5-4xlarge-a10"]
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


def test_single_node_chain_has_no_multinode_block():
    chain = _chain("g6.xlarge", "L4")
    dgd = render_pool_dgd("job_1", pool_key(chain), [chain], "default")
    worker = dgd["spec"]["services"]["VllmDecodeWorker"]

    assert "multinode" not in worker
    assert worker["resources"] == {"requests": {"gpu": "1"}, "limits": {"gpu": "1"}}


def test_multinode_chain_splits_gpus_per_node():
    chain = _chain("g6.xlarge", "L4")
    chain.shape_json.update({"count": 2, "node_count": 2, "tp": 1, "pp": 2})
    dgd = render_pool_dgd("job_1", pool_key(chain), [chain], "default")
    worker = dgd["spec"]["services"]["VllmDecodeWorker"]

    assert worker["multinode"] == {"nodeCount": 2}
    # 2 total GPUs / 2 nodes = 1 GPU requested per node, not the world size.
    assert worker["resources"] == {"requests": {"gpu": "1"}, "limits": {"gpu": "1"}}
    args = worker["extraPodSpec"]["mainContainer"]["args"]
    assert args[args.index("--pipeline-parallel-size") + 1] == "2"


def test_multinode_uneven_gpu_split_raises():
    chain = _chain("g6.xlarge", "L4")
    chain.shape_json.update({"count": 3, "node_count": 2})
    with pytest.raises(ValueError, match="does not divide evenly"):
        render_pool_dgd("job_1", pool_key(chain), [chain], "default")

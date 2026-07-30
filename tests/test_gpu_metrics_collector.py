"""collect_once attribution: owned GPUs carry identity, unowned GPUs do not."""

from __future__ import annotations

from types import SimpleNamespace

from tandemn_system_data.models import Chain, ChainRole, ResourceMap

from tandemn_orca.scripts.gpu_metrics_collector import (
    KubeWorkerIndex,
    WorkerInfo,
    assign_canonical_chain_ids,
    collect_once,
    resolve_instance_price_per_hour,
)


def test_worker_index_uses_discovery_and_com_identity_labels():
    pod = SimpleNamespace(
        metadata=SimpleNamespace(
            name="worker-1",
            labels={
                "tandemn.com/job-id": "job_1",
                "tandemn.com/rank-id": "rank_0",
                "tandemn.com/pods-discovery": "dynamo-worker",
            },
        ),
        spec=SimpleNamespace(node_name="node-1", containers=[]),
    )

    class Core:
        def list_namespaced_pod(self, namespace, label_selector):
            self.query = namespace, label_selector
            return SimpleNamespace(items=[pod])

    core = Core()
    worker = KubeWorkerIndex(core=core).by_pod()["worker-1"]

    assert core.query == ("default", "tandemn.com/pods-discovery=dynamo-worker")
    assert (worker.job_id, worker.rank_id) == ("job_1", "rank_0")


class FakeProm:
    """Two GPUs owned by one TP=2 worker (node-local indexes 2,3), one idle."""

    def gpu_targets(self):
        return [
            {
                "gpu_uuid": "GPU-b",
                "uuid_label": "GPU-b",
                "instance": "10.0.0.1:9400",
                "gpu_index": "3",
                "owner_pod": "worker-pod-1",
            },
            {
                "gpu_uuid": "GPU-a",
                "uuid_label": "GPU-a",
                "instance": "10.0.0.1:9400",
                "gpu_index": "2",
                "owner_pod": "worker-pod-1",
            },
            {
                "gpu_uuid": "GPU-idle",
                "uuid_label": "GPU-idle",
                "instance": "10.0.0.1:9400",
                "gpu_index": "0",
                "owner_pod": "",
            },
        ]

    def query_scalar(self, promql: str):
        # Worker-scoped inference queries must never run without a pod selector.
        assert "{}" not in promql and 'pod=""' not in promql
        return 1.5


def test_collect_once_attributes_owned_and_unowned_gpus():
    worker = WorkerInfo(
        worker_id="worker-pod-1",
        node_name="node-1",
        dynamo_namespace="ns",
        rank_id="rank_0",
        chain_id="chain-1",
        role="decode",
        job_id="job_1",
        model_name="Qwen/Qwen3-0.6B",
        ttft_target_ms=500.0,
    )
    samples = collect_once(
        FakeProm(),
        {"worker-pod-1": worker},
        node_names_by_ip={"10.0.0.1": "node-1"},
        instance_types_by_node={"node-1": "g6.12xlarge"},
        prices_by_instance_type={"g6.12xlarge": 3.6},
    )
    by_uuid = {s.gpu_uuid: s for s in samples}
    assert set(by_uuid) == {"GPU-a", "GPU-b", "GPU-idle"}

    owned = by_uuid["GPU-a"]
    assert owned.job_id == "job_1"
    assert owned.rank_id == "rank_0"
    assert owned.chain_id == "chain-1"
    assert owned.role == "decode"
    assert owned.node_name == "node-1"
    assert owned.instance_type == "g6.12xlarge"
    assert owned.model_name == "Qwen/Qwen3-0.6B"
    # local_rank is worker-local: node indexes 2,3 become ranks 0,1.
    assert owned.local_rank == "0"
    assert by_uuid["GPU-b"].local_rank == "1"
    assert owned.throughput_token_per_sec == 1.5
    # Instance $/hr over the chain's throughput, repeated on each GPU row.
    assert owned.cost_per_token == 3.6 / (1.5 * 3600)
    assert by_uuid["GPU-b"].cost_per_token == owned.cost_per_token
    # slo_margin from the worker's chain-shape TTFT target.
    assert owned.slo_margin == (500.0 - 1.5) / 500.0

    idle = by_uuid["GPU-idle"]
    assert idle.job_id is None
    assert idle.rank_id is None
    assert idle.chain_id is None
    assert idle.local_rank is None
    assert idle.role is None
    assert idle.node_name == "node-1"
    assert idle.instance_type == "g6.12xlarge"
    assert idle.model_name is None
    # Hardware metrics still collected; inference metrics all None.
    assert idle.gpu_mem_used_fraction == 1.5
    assert idle.throughput_token_per_sec is None
    assert idle.cost_per_token is None


def _worker(pod: str, rank_id: str | None, chain_id: str | None = None) -> WorkerInfo:
    return WorkerInfo(
        worker_id=pod,
        node_name="node-1",
        dynamo_namespace="ns",
        rank_id=rank_id,
        chain_id=chain_id or pod,
        role="decode",
    )


def _chain(chain_id: str, rank_id: str | None) -> Chain:
    shape: dict = {"count": 1, "target_p99_ttft_ms": 500.0}
    if rank_id is not None:
        shape["rank_id"] = rank_id
    return Chain(chain_id=chain_id, job_id="job_1", role=ChainRole.DECODE, shape_json=shape)


def test_assign_canonical_chain_ids_maps_pods_within_rank():
    workers = {
        "pod-b": _worker("pod-b", "decode-0"),
        "pod-a": _worker("pod-a", "decode-0"),
        "pod-c": _worker("pod-c", "decode-1"),
        # Explicitly labelled chain_id must be kept.
        "pod-d": _worker("pod-d", "decode-1", chain_id="chain_explicit"),
    }
    chains = [
        _chain("chain_2", "decode-0"),
        _chain("chain_1", "decode-0"),
        _chain("chain_3", "decode-1"),
        _chain("chain_4", "decode-1"),
    ]
    assign_canonical_chain_ids(workers, chains)

    # Deterministic: sorted pods onto sorted chain ids, per rank.
    assert workers["pod-a"].chain_id == "chain_1"
    assert workers["pod-b"].chain_id == "chain_2"
    assert workers["pod-c"].chain_id == "chain_3"
    assert workers["pod-d"].chain_id == "chain_explicit"
    # The chain's SLA target rides along for slo_margin.
    assert workers["pod-a"].ttft_target_ms == 500.0


def test_assign_canonical_chain_ids_keeps_pod_name_when_rows_run_out():
    workers = {"pod-a": _worker("pod-a", "decode-0"), "pod-b": _worker("pod-b", "decode-0")}
    assign_canonical_chain_ids(workers, [_chain("chain_1", "decode-0")])
    assert workers["pod-a"].chain_id == "chain_1"
    assert workers["pod-b"].chain_id == "pod-b"


def test_resolve_instance_price_per_hour_from_resource_map():
    resource_map = ResourceMap.model_validate(
        {
            "clouds": {
                "aws": {
                    "regions": {
                        "us-east-1": {
                            "zones": {
                                "use1-az1": {
                                    "network_fabrics": {
                                        "default": {
                                            "fabric_type": "default",
                                            "machine_pools": {
                                                "g6.12xlarge": {
                                                    "gpu_type": "L4",
                                                    "gpus_per_instance": 4,
                                                    "total_instances": 2,
                                                    "price_per_instance_hour": 4.0,
                                                },
                                                "g6.xlarge": {
                                                    "gpu_type": "L4",
                                                    "gpus_per_instance": 1,
                                                    "total_instances": 1,
                                                },
                                            },
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    )
    assert resolve_instance_price_per_hour(resource_map, "g6.12xlarge") == 4.0
    # Present but unpriced, absent, and unspecified all resolve to None.
    assert resolve_instance_price_per_hour(resource_map, "g6.xlarge") is None
    assert resolve_instance_price_per_hour(resource_map, "p5.48xlarge") is None
    assert resolve_instance_price_per_hour(resource_map, None) is None

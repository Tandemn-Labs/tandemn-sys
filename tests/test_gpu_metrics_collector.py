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


def test_worker_index_uses_single_and_multinode_grove_indexes():
    def pod(name, labels):
        return SimpleNamespace(
            metadata=SimpleNamespace(
                name=name,
                labels={
                    "tandemn.com/job-id": "job_1",
                    "tandemn.com/rank-id": "rank_0",
                    "tandemn.com/plan-id": "plan_1",
                    "tandemn.com/pods-discovery": "dynamo-worker",
                    **labels,
                },
            ),
            spec=SimpleNamespace(node_name="node-1", containers=[]),
        )

    pods = [
        pod("single", {"grove.io/podclique-pod-index": "1"}),
        pod(
            "multi",
            {
                "grove.io/podcliquescalinggroup-replica-index": "2",
                "grove.io/podclique-pod-index": "0",
            },
        ),
    ]

    class Core:
        def list_namespaced_pod(self, namespace, label_selector):
            self.query = namespace, label_selector
            return SimpleNamespace(items=pods)

    core = Core()
    workers = KubeWorkerIndex(core=core).by_pod()

    assert core.query == ("default", "tandemn.com/pods-discovery=dynamo-worker")
    assert (workers["single"].plan_id, workers["single"].replica_index) == ("plan_1", 1)
    assert (workers["multi"].replica_index, workers["multi"].pod_index) == (2, 0)


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


def _worker(
    pod: str,
    replica_index: int | None,
    *,
    rank_id: str = "rank_0",
    plan_id: str = "plan_1",
) -> WorkerInfo:
    return WorkerInfo(
        worker_id=pod,
        node_name="node-1",
        dynamo_namespace="ns",
        rank_id=rank_id,
        role="decode",
        job_id="job_1",
        plan_id=plan_id,
        replica_index=replica_index,
    )


def _chain(
    chain_id: str,
    replica_index: int,
    *,
    rank_id: str = "rank_0",
    plan_id: str = "plan_1",
) -> Chain:
    return Chain(
        chain_id=chain_id,
        job_id="job_1",
        plan_id=plan_id,
        role=ChainRole.DECODE,
        shape_json={
            "count": 1,
            "rank_id": rank_id,
            "replica_index": replica_index,
            "target_p99_ttft_ms": 500.0,
        },
    )


def test_assign_canonical_chain_ids_maps_exact_grove_slots():
    workers = {
        "single-0": _worker("single-0", 0),
        "single-1": _worker("single-1", 1),
        "multi-leader": _worker("multi-leader", 2),
        "multi-worker": _worker("multi-worker", 2),
    }
    chains = [
        _chain("chain_2", 2),
        _chain("chain_0", 0),
        _chain("chain_1", 1),
    ]
    assign_canonical_chain_ids(workers, chains)

    assert workers["single-0"].chain_id == "chain_0"
    assert workers["single-1"].chain_id == "chain_1"
    assert workers["multi-leader"].chain_id == "chain_2"
    assert workers["multi-worker"].chain_id == "chain_2"
    assert workers["single-0"].ttft_target_ms == 500.0


def test_assign_canonical_chain_ids_fails_closed():
    workers = {
        "missing": _worker("missing", None),
        "outside": _worker("outside", 9),
        "old-plan": _worker("old-plan", 0, plan_id="plan_old"),
        "duplicate": _worker("duplicate", 0),
    }
    assign_canonical_chain_ids(workers, [_chain("chain_0", 0), _chain("chain_dup", 0)])
    assert all(worker.chain_id is None for worker in workers.values())


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

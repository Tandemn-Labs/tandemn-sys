"""collect_once attribution: owned GPUs carry identity, unowned GPUs do not."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from io import BytesIO
from types import SimpleNamespace

from tandemn_system_data.models import Rank, RankRole, ResourceMap

from tandemn_orca.dynamo_compiler import router_listen_port
from tandemn_orca.scripts.gpu_metrics_collector import (
    KubeWorkerIndex,
    PrometheusClient,
    RankTelemetrySnapshot,
    RouterTelemetryClient,
    WorkerInfo,
    _worker_inference_values,
    collect_once,
    collect_rank_telemetry,
    kv_pressure_score,
    local_ranks_for_workers,
    resolve_instance_price_per_hour,
    validate_rank_identity,
)


def test_worker_index_uses_single_and_multinode_grove_indexes():
    def pod(name, labels):
        return SimpleNamespace(
            metadata=SimpleNamespace(
                name=name,
                labels={
                    "tandemn.com/job-id": "job_1",
                    "tandemn.com/rank-id": "rank_0",
                    "tandemn.com/pods-discovery": "dynamo-worker",
                    **labels,
                },
            ),
            spec=SimpleNamespace(node_name="node-1", containers=[]),
        )

    pods = [
        pod("single", {"grove.io/podclique-pod-index": "1"}),
        pod(
            "multi-leader",
            {
                "grove.io/podcliquescalinggroup-replica-index": "2",
                "grove.io/podclique": "rank-0-vllmdecodeworker-2-vllmdecodeworker-ldr",
                "grove.io/podclique-pod-index": "0",
            },
        ),
        pod(
            "multi-worker",
            {
                "grove.io/podcliquescalinggroup-replica-index": "2",
                "grove.io/podclique": "rank-0-vllmdecodeworker-2-vllmdecodeworker-wkr",
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

    assert core.query == ("default", "tandemn.com/pods-discovery")
    assert workers["single"].chain_index == 1
    assert (
        workers["single"].member_index,
        workers["multi-leader"].member_index,
        workers["multi-worker"].member_index,
    ) == (None, 0, 1)
    assert (workers["multi-leader"].chain_index, workers["multi-worker"].chain_index) == (2, 2)


def test_worker_index_discovers_batch_jobs_and_lws_members():
    def pod(name, labels):
        return SimpleNamespace(
            metadata=SimpleNamespace(
                name=name,
                labels={
                    "tandemn.com/job-id": "job_1",
                    "tandemn.com/rank-id": "rank_0",
                    "tandemn.com/pods-discovery": "batch-worker",
                    "tandemn.com/chain-id": "3",
                    **labels,
                },
            ),
            spec=SimpleNamespace(node_name="node-1", containers=[]),
        )

    pods = [
        pod("leader", {"leaderworkerset.sigs.k8s.io/worker-index": "0"}),
        pod("worker", {"leaderworkerset.sigs.k8s.io/worker-index": "1"}),
    ]

    class Core:
        def list_namespaced_pod(self, namespace, label_selector):
            self.queries = [*getattr(self, "queries", []), (namespace, label_selector)]
            return SimpleNamespace(items=pods if namespace == "tandemn-system" else [])

    core = Core()
    workers = KubeWorkerIndex(namespace=["dynamo-system", "tandemn-system"], core=core).by_pod()

    assert core.queries == [
        ("dynamo-system", "tandemn.com/pods-discovery"),
        ("tandemn-system", "tandemn.com/pods-discovery"),
    ]
    assert workers["leader"].worker_kind == "batch-worker"
    assert workers["leader"].chain_index == workers["worker"].chain_index == 3
    assert (workers["leader"].member_index, workers["worker"].member_index) == (0, 1)


def test_worker_index_uses_explicit_kube_context(monkeypatch):
    api_client = object()
    core = object()
    monkeypatch.setattr(
        "kubernetes.config.new_client_from_config",
        lambda **kwargs: api_client if kwargs == {"context": "gke-central"} else None,
    )
    monkeypatch.setattr(
        "kubernetes.client.CoreV1Api",
        lambda loaded: core if loaded is api_client else None,
    )

    index = KubeWorkerIndex(context="gke-central")

    assert index._core is core


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
        chain_index=1,
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
    assert owned.chain_index == 1
    assert owned.role == "decode"
    assert owned.node_name == "node-1"
    assert owned.instance_type == "g6.12xlarge"
    assert owned.model_name == "Qwen/Qwen3-0.6B"
    # local_rank is worker-local: node indexes 2,3 become ranks 0,1.
    assert owned.local_rank == "0"
    assert by_uuid["GPU-b"].local_rank == "1"
    assert owned.throughput_token_per_sec == 1.5
    assert owned.batched_reqs_inflight is None
    # Instance $/hr over the replica's throughput, repeated on each GPU row.
    assert owned.cost_per_token == 3.6 / (1.5 * 3600)
    assert by_uuid["GPU-b"].cost_per_token == owned.cost_per_token
    # slo_margin from the rank-shape TTFT target.
    assert owned.slo_margin == (500.0 - 1.5) / 500.0

    idle = by_uuid["GPU-idle"]
    assert idle.job_id is None
    assert idle.rank_id is None
    assert idle.chain_index is None
    assert idle.local_rank is None
    assert idle.role is None
    assert idle.node_name == "node-1"
    assert idle.instance_type == "g6.12xlarge"
    assert idle.model_name is None
    # Hardware metrics still collected; inference metrics all None.
    assert idle.gpu_mem_used_fraction == 1.5
    assert idle.throughput_token_per_sec is None
    assert idle.cost_per_token is None


def test_collect_once_infers_owner_from_unique_worker_on_node():
    class NodeProm(FakeProm):
        def gpu_targets(self):
            return [
                {
                    "gpu_uuid": "GPU-1",
                    "uuid_label": "GPU-1",
                    "instance": "10.0.0.1:9400",
                    "hostname": "node-1",
                    "gpu_index": "0",
                    "owner_pod": "",
                }
            ]

    worker = WorkerInfo(
        worker_id="batch-worker",
        node_name="node-1",
        dynamo_namespace=None,
        rank_id="rank_0",
        chain_index=0,
        role=None,
        worker_kind="batch-worker",
        job_id="job_1",
    )

    sample = collect_once(NodeProm(), {worker.worker_id: worker})[0]

    assert sample.job_id == "job_1"
    assert sample.rank_id == "rank_0"
    assert sample.chain_index == 0


def test_multinode_local_rank_and_cost_cover_the_full_chain():
    class MultiNodeProm(FakeProm):
        def gpu_targets(self):
            return [
                {
                    "gpu_uuid": f"GPU-{member}",
                    "uuid_label": f"GPU-{member}",
                    "instance": f"10.0.0.{member + 1}:9400",
                    "gpu_index": "0",
                    "owner_pod": f"worker-{member}",
                }
                for member in range(2)
            ]

    workers = {
        f"worker-{member}": WorkerInfo(
            worker_id=f"worker-{member}",
            node_name=f"node-{member}",
            dynamo_namespace="ns",
            rank_id="rank_0",
            chain_index=0,
            member_index=member,
            role="decode",
            job_id="job_1",
        )
        for member in range(2)
    }
    rank = _rank(1)
    rank.shape_json.update({"count": 2, "node_count": 2})
    validate_rank_identity(workers, [rank])

    samples = collect_once(
        MultiNodeProm(),
        workers,
        instance_types_by_node={"node-0": "gpu.small", "node-1": "gpu.large"},
        prices_by_instance_type={"gpu.small": 2.0, "gpu.large": 3.0},
    )
    by_uuid = {sample.gpu_uuid: sample for sample in samples}

    assert by_uuid["GPU-0"].local_rank == "0"
    assert by_uuid["GPU-1"].local_rank == "1"
    assert by_uuid["GPU-0"].cost_per_token == 5.0 / (1.5 * 3600)
    assert by_uuid["GPU-1"].cost_per_token == by_uuid["GPU-0"].cost_per_token


def test_batch_metrics_use_the_lws_leader_for_every_member():
    class BatchProm:
        def __init__(self):
            self.queries = []

        def gpu_targets(self):
            return [
                {
                    "gpu_uuid": f"GPU-{member}",
                    "uuid_label": f"GPU-{member}",
                    "instance": f"10.0.0.{member + 1}:9400",
                    "gpu_index": "0",
                    "owner_pod": "batch-leader" if member == 0 else "batch-worker",
                }
                for member in range(2)
            ]

        def query_scalar(self, query):
            self.queries.append(query)
            assert 'pod="batch-worker"' not in query
            if "batched_reqs_inflight" in query:
                return 100.0
            if "batched_reqs_processed_total" in query:
                return 387.0
            if "batched_chunks_input_pulled_total" in query:
                return 7.0
            if "batched_chunks_output_written_total" in query:
                return 2.0
            return 1.0

    workers = {
        pod: WorkerInfo(
            worker_id=pod,
            node_name=f"node-{member}",
            dynamo_namespace=None,
            rank_id="rank_0",
            chain_index=0,
            member_index=member,
            role=None,
            worker_kind="batch-worker",
            job_id="job_1",
            model_name="microsoft/phi-4" if member == 0 else None,
        )
        for member, pod in enumerate(("batch-leader", "batch-worker"))
    }
    rank = _rank(1)
    rank.shape_json.update({"count": 2, "node_count": 2})
    validate_rank_identity(workers, [rank])
    prom = BatchProm()

    samples = collect_once(prom, workers)

    assert {sample.model_name for sample in samples} == {"microsoft/phi-4"}
    for sample in samples:
        assert sample.batched_reqs_inflight == 100.0
        assert sample.batched_reqs_processed_total == 387.0
        assert sample.batched_chunks_input_pulled_total == 7.0
        assert sample.batched_chunks_output_written_total == 2.0


def _worker(
    pod: str,
    chain_index: int | None,
    *,
    rank_id: str = "rank_0",
) -> WorkerInfo:
    return WorkerInfo(
        worker_id=pod,
        node_name="node-1",
        dynamo_namespace="ns",
        rank_id=rank_id,
        role="decode",
        job_id="job_1",
        chain_index=chain_index,
    )


def _rank(
    n_replicas: int,
    *,
    rank_id: str = "rank_0",
    plan_id: str = "plan_1",
) -> Rank:
    return Rank(
        rank_id=rank_id,
        job_id="job_1",
        plan_id=plan_id,
        role=RankRole.DECODE,
        n_replicas=n_replicas,
        shape_json={
            "count": 1,
            "target_p99_ttft_ms": 500.0,
        },
    )


def test_validate_rank_identity_accepts_grove_slots_in_replica_range():
    workers = {
        "single-0": _worker("single-0", 0),
        "single-1": _worker("single-1", 1),
        "multi-leader": _worker("multi-leader", 2),
        "multi-worker": _worker("multi-worker", 2),
    }
    validate_rank_identity(workers, [_rank(3)])

    assert workers["single-0"].chain_index == 0
    assert workers["single-1"].chain_index == 1
    assert workers["multi-leader"].chain_index == 2
    assert workers["multi-worker"].chain_index == 2
    assert workers["single-0"].ttft_target_ms == 500.0


def test_collect_rank_telemetry_deduplicates_multinode_members():
    class Prom:
        def query_scalar(self, query):
            if "num_requests_running" in query:
                return 3.0
            if "gpu_cache_usage_percent" in query:
                # Distinct per chain: the snapshot must carry the rank-level mean.
                return 0.3 if "chain-0" in query else 0.5
            return 1.0

    workers = {}
    for chain_index in range(3):
        for member_index in range(2):
            worker = _worker(f"chain-{chain_index}-member-{member_index}", chain_index)
            worker.member_index = member_index
            worker.node_count = 2
            worker.ready = chain_index < 2 or member_index == 0
            workers[worker.worker_id] = worker
    rank = _rank(3)
    batch = _worker("batch-chain", 0)
    batch.worker_kind = "batch-worker"
    workers[batch.worker_id] = batch
    observed_at = datetime(2026, 8, 4, tzinfo=UTC)

    snapshots = collect_rank_telemetry(Prom(), workers, [rank], observed_at)

    assert snapshots == [
        RankTelemetrySnapshot(
            job_id="job_1",
            rank_id="rank_0",
            active_requests=6,
            pending_requests=2,
            ready_replicas=2,
            observed_at=observed_at,
            kv_cache_util=0.4,
        )
    ]


def test_local_ranks_for_workers_excludes_remote_cluster_ranks():
    local = _rank(1, rank_id="rank_local")
    remote = _rank(1, rank_id="rank_remote")
    workers = {"local-pod": _worker("local-pod", 0, rank_id="rank_local")}

    assert local_ranks_for_workers([local, remote], workers) == [local]


def test_router_telemetry_client_posts_authenticated_json(monkeypatch):
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def urlopen(request, timeout):
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr(
        "tandemn_orca.scripts.gpu_metrics_collector.urllib.request.urlopen", urlopen
    )
    snapshot = RankTelemetrySnapshot(
        job_id="job_1",
        rank_id="rank_0",
        active_requests=2,
        pending_requests=1,
        ready_replicas=1,
        observed_at=datetime(2026, 8, 4, tzinfo=UTC),
    )

    RouterTelemetryClient("http://127.0.0.1:18080", "secret").push(snapshot)

    request, timeout = requests[0]
    assert request.full_url == "http://127.0.0.1:18080/internal/telemetry"
    assert request.headers["Authorization"] == "Bearer secret"
    assert json.loads(request.data)["pending_requests"] == 1
    assert json.loads(request.data)["kv_cache_util"] == 0.0
    assert timeout == 5.0

    RouterTelemetryClient(None, "secret").push(snapshot)
    derived, _ = requests[1]
    port = router_listen_port("job_1")
    assert derived.full_url == f"http://127.0.0.1:{port}/internal/telemetry"


def test_prometheus_client_treats_connection_reset_as_missing_data(monkeypatch):
    def reset(*args, **kwargs):
        raise ConnectionResetError

    monkeypatch.setattr(
        "tandemn_orca.scripts.gpu_metrics_collector.urllib.request.urlopen",
        reset,
    )
    client = PrometheusClient("http://prometheus")

    assert client.query_scalar("up") is None
    assert client.gpu_targets() == []


def test_gpu_discovery_accepts_unrenamed_owner_pod_and_hostname(monkeypatch):
    payload = {
        "data": {
            "result": [
                {
                    "metric": {
                        "UUID": "GPU-1",
                        "gpu": "0",
                        "pod": "batch-worker-0",
                        "Hostname": "gke-node-1",
                    }
                }
            ]
        }
    }
    monkeypatch.setattr(
        "tandemn_orca.scripts.gpu_metrics_collector.urllib.request.urlopen",
        lambda *args, **kwargs: BytesIO(json.dumps(payload).encode()),
    )

    targets = PrometheusClient("http://prometheus").gpu_targets()

    assert targets[0]["owner_pod"] == "batch-worker-0"
    assert targets[0]["hostname"] == "gke-node-1"


def test_validate_rank_identity_fails_closed():
    workers = {
        "missing": _worker("missing", None),
        "outside": _worker("outside", 9),
        "negative": _worker("negative", -1),
        "old-plan": _worker("old-plan", 0),
        "wrong-rank": _worker("wrong-rank", 0, rank_id="rank_other"),
        "bad-member": _worker("bad-member", 0),
    }
    workers["old-plan"].member_index = 0
    workers["bad-member"].member_index = 2
    rank = _rank(2)
    rank.shape_json.update({"count": 2, "node_count": 2})
    validate_rank_identity(workers, [rank])
    assert workers["old-plan"].chain_index == 0
    for name in ("missing", "outside", "negative", "wrong-rank", "bad-member"):
        assert workers[name].chain_index is None
        assert workers[name].job_id is None and workers[name].rank_id is None


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


def test_kv_pressure_score_matches_simulator_ratio():
    # 5 requests, each block-rounded to ceil((100 + 60) / 16) = 10 blocks,
    # against 200 total KV blocks: 50 / 200.
    assert kv_pressure_score(2.0, 3.0, 100.0, 60.0, 200.0) == 0.25


def test_kv_pressure_score_zero_requests_is_zero_without_lengths():
    assert kv_pressure_score(0.0, 0.0, None, None, 200.0) == 0.0


def test_kv_pressure_score_missing_inputs_return_none():
    assert kv_pressure_score(2.0, 3.0, 100.0, 60.0, None) is None
    assert kv_pressure_score(2.0, 3.0, 100.0, 60.0, 0.0) is None
    assert kv_pressure_score(2.0, 3.0, None, 60.0, 200.0) is None
    assert kv_pressure_score(2.0, 3.0, 100.0, None, 200.0) is None
    assert kv_pressure_score(None, 3.0, 100.0, 60.0, 200.0) is None


class _KvPressureProm:
    """Answers each derived-metric input by a distinctive query fragment."""

    def __init__(self, *, max_tokens: float | None = 60.0):
        self.max_tokens = max_tokens

    def query_scalar(self, query: str):
        if "request_params_max_tokens" in query:
            return self.max_tokens
        if "num_requests_running" in query:
            return 2.0
        if "num_requests_waiting" in query:
            return 3.0
        if "request_prompt_tokens" in query:
            return 100.0
        if "request_generation_tokens" in query:
            return 60.0
        if "dynamo_component_total_blocks" in query:
            return 200.0
        return None


def test_worker_inference_values_compute_simulator_style_kv_pressure():
    values = _worker_inference_values(
        _KvPressureProm(),
        _worker("worker-pod-1", 0),
        price_per_hour=None,
        ttft_target_ms=None,
    )
    assert values["kv_pressure_score"] == 0.25


def test_kv_pressure_falls_back_to_observed_generation_length():
    # Without request_params_max_tokens, output_length_observed (60) stands in.
    values = _worker_inference_values(
        _KvPressureProm(max_tokens=None),
        _worker("worker-pod-1", 0),
        price_per_hour=None,
        ttft_target_ms=None,
    )
    assert values["kv_pressure_score"] == 0.25

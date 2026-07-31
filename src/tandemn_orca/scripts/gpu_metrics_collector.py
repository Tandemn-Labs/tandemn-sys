"""Collect GPU/inference telemetry from Prometheus into Tandemn Store.

One collector per cluster ("fleet mode"): every Dynamo worker pod in
``--namespace`` is tracked, across all DGDs and jobs. Identity is discovered,
not configured: ``job_id`` + ``rank_id`` from the ``tandemn.com/*`` pod labels
Orca stamps, the served
model from the worker's ``--model`` arg, the node's instance type from its
``node.kubernetes.io/instance-type`` label.

Writes one ``GpuMetric`` row per physical GPU per tick. Granularity:

- GPU hardware metrics (DCGM) are scoped to that one GPU (by ``UUID``).
- Inference metrics (vLLM) are scoped to the chain/worker that owns the GPU (by
  the worker pod name), so a multi-replica deployment records each chain's own
  numbers instead of a deployment-wide sum.

Tandemn job model, coarse -> fine: a ``rank_id`` is a ladder rung (rank config)
realized by N chains / DP replicas; a ``chain_id`` is one serving unit == one
worker pod; a chain spans N GPUs, each with a ``local_rank`` (its index in the
chain). ``chain_id`` is the canonical ``chains.chain_id`` from the store:
chains within a rank are fungible DP replicas, so each job's worker pods are
mapped onto its chain rows deterministically (both sides sorted). Pods without
a job label (or beyond the job's chain rows) fall back to the pod name.

The GPU->chain join uses dcgm-exporter's pod attribution: with
``DCGM_EXPORTER_KUBERNETES=true`` (cloud-setup/EKS/dcgm-exporter.yaml) each DCGM
series carries the pod its GPU is allocated to (``exported_pod`` after the
Prometheus scrape relabels it). A GPU no worker owns still gets a row --
hardware metrics with all-null identity -- so aggregate utilization sees idle
capacity on tracked nodes.

Run once (``--once``) or loop on a fixed ``COLLECT_INTERVAL_SECONDS`` cadence.
Metrics that are topology- or config-gated (NVLink/comm/expert;
``sm_utilization``) return no series and are left ``None``. ``cost_per_token``
uses the ``--user-id`` resource map's ``price_per_instance_hour`` for the
worker node's instance type (the resource map is assumed accurate);
``slo_margin`` uses the chain's ``target_p99_ttft_ms`` from its shape.

Prometheus URL: ``--prometheus-url`` or ``TANDEMN_PROMETHEUS_URL``. Postgres:
``TANDEMN_POSTGRES_URL`` (see tandemn-store). Kubernetes config is loaded
in-cluster first, else from the local kubeconfig.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from typing import Any

from tandemn_system_data.clients import (
    GpuMetricStore,
    JobStore,
    PostgresClient,
    ResourceMapStore,
)
from tandemn_system_data.models import Chain, GpuMetric, ResourceMap

logger = logging.getLogger(__name__)

DEFAULT_PROMETHEUS_URL = "http://localhost:9090"

# Fixed poll cadence for the looping mode (matches the Prometheus scrape rate).
COLLECT_INTERVAL_SECONDS = 10

# Per-GPU instant queries. {gpu} expands to a DCGM label selector pinning one
# physical GPU. Expressions mirror cloud-setup/EKS/METRICS.md.
GPU_QUERIES: dict[str, str] = {
    "gpu_mem_used_fraction": (
        "DCGM_FI_DEV_FB_USED{{{gpu}}} / "
        "(DCGM_FI_DEV_FB_USED{{{gpu}}} + DCGM_FI_DEV_FB_FREE{{{gpu}}} "
        "+ DCGM_FI_DEV_FB_RESERVED{{{gpu}}})"
    ),
    "vram_headroom_gb": "DCGM_FI_DEV_FB_FREE{{{gpu}}} / 1024",
    "sm_utilization": "DCGM_FI_PROF_SM_ACTIVE{{{gpu}}}",
    "mem_bandwidth_utilization": "DCGM_FI_PROF_DRAM_ACTIVE{{{gpu}}}",
    "pcie_tput_observed": (
        "DCGM_FI_PROF_PCIE_TX_BYTES{{{gpu}}} + DCGM_FI_PROF_PCIE_RX_BYTES{{{gpu}}}"
    ),
    "nvlink_tput_observed": (
        "DCGM_FI_PROF_NVLINK_TX_BYTES{{{gpu}}} + DCGM_FI_PROF_NVLINK_RX_BYTES{{{gpu}}}"
    ),
}

# Per-worker inference queries. {worker} expands to a label selector scoping to
# one vLLM worker (its dynamo_namespace), so multi-replica deployments get each
# worker's own numbers instead of a deployment-wide sum. Expressions mirror
# cloud-setup/EKS/METRICS.md.
WORKER_QUERIES: dict[str, str] = {
    "p99_ttft_ms": (
        "histogram_quantile(0.99, sum(rate("
        "vllm:time_to_first_token_seconds_bucket{{{worker}}}[5m])) by (le)) * 1000"
    ),
    "p99_tpot_ms": (
        "histogram_quantile(0.99, sum(rate("
        "vllm:request_time_per_output_token_seconds_bucket{{{worker}}}[5m])) by (le)) * 1000"
    ),
    "throughput_token_per_sec": "sum(rate(vllm:generation_tokens_total{{{worker}}}[1m]))",
    "live_batch_size": "sum(vllm:num_requests_running{{{worker}}})",
    "depth_req_q": "sum(vllm:num_requests_waiting{{{worker}}})",
    "kv_cache_util": "max(vllm:kv_cache_usage_perc{{{worker}}})",
    "kvcache_hit_rate": (
        "sum(rate(vllm:prefix_cache_hits_total{{{worker}}}[5m])) / "
        "sum(rate(vllm:prefix_cache_queries_total{{{worker}}}[5m]))"
    ),
    "input_length_observed": (
        "sum(rate(vllm:request_prompt_tokens_sum{{{worker}}}[5m])) / "
        "sum(rate(vllm:request_prompt_tokens_count{{{worker}}}[5m]))"
    ),
    "output_length_observed": (
        "sum(rate(vllm:request_generation_tokens_sum{{{worker}}}[5m])) / "
        "sum(rate(vllm:request_generation_tokens_count{{{worker}}}[5m]))"
    ),
    "prefill_iteration_counts_per_second": (
        "sum(rate(vllm:request_prefill_time_seconds_count{{{worker}}}[1m]))"
    ),
    "decode_itr_counts_per_second": (
        "sum(rate(vllm:request_decode_time_seconds_count{{{worker}}}[1m]))"
    ),
    "kv_pressure_score": (
        "max(vllm:kv_cache_usage_perc{{{worker}}}) + sum(vllm:num_requests_waiting{{{worker}}}) "
        "+ sum(rate(vllm:num_preemptions_total{{{worker}}}[5m]))"
    ),
    "pd_inbalance": (
        "sum(rate(vllm:request_prefill_time_seconds_sum{{{worker}}}[5m])) / "
        "sum(rate(vllm:request_decode_time_seconds_sum{{{worker}}}[5m]))"
    ),
}


class PrometheusClient:
    """Minimal Prometheus HTTP API client (instant queries)."""

    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def query_scalar(self, promql: str) -> float | None:
        """Return the first sample value of an instant query, or None."""
        url = f"{self._base_url}/api/v1/query?" + urllib.parse.urlencode({"query": promql})
        try:
            with urllib.request.urlopen(url, timeout=self._timeout) as resp:
                payload = json.load(resp)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            logger.warning("prometheus query failed: %s", error)
            return None
        results = payload.get("data", {}).get("result", [])
        if not results:
            return None
        value = results[0].get("value", [None, None])[1]
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        # Prometheus encodes NaN/Inf as strings that float() parses; drop them.
        if parsed != parsed or parsed in (float("inf"), float("-inf")):
            return None
        return parsed

    def gpu_targets(self) -> list[dict[str, str]]:
        """One descriptor per physical GPU currently exporting DCGM metrics."""
        url = f"{self._base_url}/api/v1/query?" + urllib.parse.urlencode(
            {"query": "DCGM_FI_DEV_GPU_UTIL"}
        )
        try:
            with urllib.request.urlopen(url, timeout=self._timeout) as resp:
                payload = json.load(resp)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            logger.warning("prometheus gpu discovery failed: %s", error)
            return []
        targets = []
        for series in payload.get("data", {}).get("result", []):
            metric = series.get("metric", {})
            uuid = metric.get("UUID") or metric.get("gpu") or metric.get("instance")
            if uuid is None:
                continue
            targets.append(
                {
                    "gpu_uuid": str(uuid),
                    "uuid_label": metric.get("UUID", ""),
                    "instance": metric.get("instance", ""),
                    # DCGM's node-local GPU index (0,1,2,...) = the GPU's rank
                    # within its chain/worker for TP/PP parallelism.
                    "gpu_index": metric.get("gpu", ""),
                    # The pod this GPU is allocated to, from dcgm-exporter's
                    # kubelet PodResources mapping. The exporter emits it as
                    # "pod"; the scrape's own pod label (the exporter pod)
                    # wins that name, so the GPU owner arrives as
                    # "exported_pod". Empty = unallocated GPU or mapping off.
                    "owner_pod": metric.get("exported_pod", ""),
                }
            )
        return targets


def _gpu_selector(target: dict[str, str]) -> str:
    """DCGM label selector that pins one GPU; prefer UUID, else instance."""
    if target.get("uuid_label"):
        return f'UUID="{target["uuid_label"]}"'
    return f'instance="{target["instance"]}"'


# Dynamo/tandemn pod labels. Tandemn job model, coarse -> fine:
#   rank  = a ladder rung (rank config), realized by N chains / DP replicas
#   chain = one serving unit == one worker (a DP replica within the rank)
#   worker spans N GPUs (its local ranks) under TP/PP.
_LABEL_DYN_NS = "nvidia.com/dynamo-namespace"
# PD-disaggregation role of the worker: "prefill" | "decode" (absent = aggregated).
_LABEL_SUBCOMPONENT = "nvidia.com/dynamo-sub-component-type"
_LABEL_DISCOVERY = "tandemn.com/pods-discovery"
_LABEL_JOB = "tandemn.com/job-id"
_LABEL_RANK = "tandemn.com/rank-id"
_LABEL_PLAN = "tandemn.com/plan-id"
_LABEL_PCSG_INDEX = "grove.io/podcliquescalinggroup-replica-index"
_LABEL_POD_INDEX = "grove.io/podclique-pod-index"

# Node label carrying the EC2 instance type (standard on EKS/Karpenter nodes).
_NODE_LABEL_INSTANCE_TYPE = "node.kubernetes.io/instance-type"


class WorkerInfo:
    """Identity of one Dynamo worker pod (== one chain / DP replica).

    Joined to the node it runs on. ``rank_id`` is the ladder rung the chain
    belongs to (shared across the rank's chains); ``chain_id`` is the serving
    unit (canonical chains.chain_id after ``assign_canonical_chain_ids``);
    ``worker_id`` is the pod name, kept internally for the
    vLLM ``pod=`` selector but not stored. Per-GPU local ranks are resolved
    separately from the DCGM ``gpu`` index.
    """

    def __init__(
        self,
        *,
        worker_id: str,
        node_name: str,
        dynamo_namespace: str | None,
        rank_id: str | None,
        role: str | None,
        job_id: str | None = None,
        plan_id: str | None = None,
        replica_index: int | None = None,
        pod_index: int | None = None,
        chain_id: str | None = None,
        model_name: str | None = None,
        ttft_target_ms: float | None = None,
    ) -> None:
        self.worker_id = worker_id
        self.node_name = node_name
        self.dynamo_namespace = dynamo_namespace
        self.rank_id = rank_id
        self.chain_id = chain_id
        self.role = role
        self.job_id = job_id
        self.plan_id = plan_id
        self.replica_index = replica_index
        self.pod_index = pod_index
        self.model_name = model_name
        self.ttft_target_ms = ttft_target_ms


def _model_from_pod(pod: Any) -> str | None:
    """The worker's served model, from its container ``--model <id>`` arg."""
    for container in pod.spec.containers or []:
        args = list(container.args or [])
        for i, arg in enumerate(args[:-1]):
            if arg == "--model":
                return str(args[i + 1])
    return None


def _index(value: str | None) -> int | None:
    try:
        parsed = int(value) if value is not None else -1
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


class KubeWorkerIndex:
    """Indexes every Dynamo worker pod (chain) in the namespace by pod name.

    dcgm-exporter's PodResources mapping names the pod that owns each GPU
    (``exported_pod``); this index supplies that pod's identity labels. The
    node maps name the node (dcgm-exporter runs with hostNetwork, so its
    ``instance`` label carries the node's InternalIP) and its instance type.
    """

    def __init__(self, namespace: str = "default", core: Any = None) -> None:
        from kubernetes import client, config

        if core is None:
            try:
                config.load_incluster_config()
            except Exception:
                config.load_kube_config()
            core = client.CoreV1Api()
        self._core = core
        self._namespace = namespace

    def nodes(self) -> tuple[dict[str, str], dict[str, str]]:
        """(InternalIP -> node name, node name -> instance type)."""
        names_by_ip: dict[str, str] = {}
        instance_types: dict[str, str] = {}
        for node in self._core.list_node().items:
            name = node.metadata.name
            instance_type = (node.metadata.labels or {}).get(_NODE_LABEL_INSTANCE_TYPE)
            if instance_type:
                instance_types[name] = instance_type
            for addr in node.status.addresses or []:
                if addr.type == "InternalIP":
                    names_by_ip[addr.address] = name
        return names_by_ip, instance_types

    def by_pod(self) -> dict[str, WorkerInfo]:
        """All Dynamo worker pods in the namespace, keyed by pod name."""
        selector = f"{_LABEL_DISCOVERY}=dynamo-worker"
        pods = self._core.list_namespaced_pod(self._namespace, label_selector=selector).items
        index: dict[str, WorkerInfo] = {}
        for pod in pods:
            labels = pod.metadata.labels or {}
            pod_index = _index(labels.get(_LABEL_POD_INDEX))
            replica_index = _index(labels.get(_LABEL_PCSG_INDEX))
            if replica_index is None:
                replica_index = pod_index
            index[pod.metadata.name] = WorkerInfo(
                worker_id=pod.metadata.name,
                node_name=pod.spec.node_name or "",
                dynamo_namespace=labels.get(_LABEL_DYN_NS),
                rank_id=labels.get(_LABEL_RANK),
                role=labels.get(_LABEL_SUBCOMPONENT),
                job_id=labels.get(_LABEL_JOB),
                plan_id=labels.get(_LABEL_PLAN),
                replica_index=replica_index,
                pod_index=pod_index,
                model_name=_model_from_pod(pod),
            )
        return index


def assign_canonical_chain_ids(workers_by_pod: dict[str, WorkerInfo], chains: list[Chain]) -> None:
    """Resolve canonical chain IDs from plan, rank, and Grove replica index."""
    by_slot: dict[tuple[str, str, str, int], Chain | None] = {}
    for chain in chains:
        rank_id = chain.shape_json.get("rank_id")
        replica_index = chain.shape_json.get("replica_index")
        if (
            not chain.plan_id
            or not rank_id
            or not isinstance(replica_index, int)
            or isinstance(replica_index, bool)
            or replica_index < 0
        ):
            continue
        key = (chain.job_id, chain.plan_id, str(rank_id), replica_index)
        by_slot[key] = None if key in by_slot else chain

    for worker in workers_by_pod.values():
        worker.chain_id = None
        if not worker.job_id or not worker.plan_id or not worker.rank_id:
            continue
        if worker.replica_index is None:
            continue
        key = (worker.job_id, worker.plan_id, worker.rank_id, worker.replica_index)
        chain = by_slot.get(key)
        if chain is None:
            continue
        worker.chain_id = chain.chain_id
        target = chain.shape_json.get("target_p99_ttft_ms")
        if isinstance(target, (int, float)):
            worker.ttft_target_ms = float(target)


def _worker_inference_values(
    prom: PrometheusClient,
    worker: WorkerInfo,
    *,
    price_per_hour: float | None,
    ttft_target_ms: float | None,
) -> dict[str, float | None]:
    """Inference metrics scoped to one worker (pod), plus derived cost/SLO.

    The PodMonitor attaches a ``pod`` label equal to the worker pod name, so
    ``pod="<worker_id>"`` scopes vLLM series to that single worker.
    """
    selector = f'pod="{worker.worker_id}"'
    values: dict[str, float | None] = {
        field: prom.query_scalar(query.format(worker=selector))
        for field, query in WORKER_QUERIES.items()
    }

    throughput = values.get("throughput_token_per_sec")
    values["cost_per_token"] = (
        price_per_hour / (throughput * 3600) if price_per_hour is not None and throughput else None
    )

    p99_ttft = values.get("p99_ttft_ms")
    values["slo_margin"] = (
        (ttft_target_ms - p99_ttft) / ttft_target_ms
        if ttft_target_ms and p99_ttft is not None
        else None
    )
    return values


def collect_once(
    prom: PrometheusClient,
    workers_by_pod: dict[str, WorkerInfo],
    *,
    node_names_by_ip: dict[str, str] | None = None,
    instance_types_by_node: dict[str, str] | None = None,
    prices_by_instance_type: dict[str, float | None] | None = None,
) -> list[GpuMetric]:
    """One GpuMetric per GPU: GPU metrics per-GPU, inference metrics per-chain.

    A GPU's ``owner_pod`` (dcgm-exporter PodResources mapping) picks its
    chain/worker from ``workers_by_pod``, so inference metrics are scoped to
    that chain rather than summed across the whole deployment. A GPU no worker
    owns still gets a row -- hardware metrics only, identity
    (``job_id``/``rank_id``/``chain_id``/``local_rank``/``role``) and
    inference metrics all None -- so aggregate utilization sees idle capacity.

    ``prices_by_instance_type`` holds instance $/hour; a chain's
    ``cost_per_token`` uses its node's instance price as-is, repeated on each
    of its GPU rows.
    """
    # Inference metrics are per-worker; cache so GPUs sharing a worker (TP>1)
    # reuse one query pass.
    worker_cache: dict[str, dict[str, float | None]] = {}
    none_inference: dict[str, float | None] = dict.fromkeys(
        [*WORKER_QUERIES, "cost_per_token", "slo_margin"]
    )

    targets = prom.gpu_targets()
    if targets and workers_by_pod and not any(t.get("owner_pod") for t in targets):
        logger.warning(
            "no DCGM series carries exported_pod; is DCGM_EXPORTER_KUBERNETES enabled? "
            "all GPUs will be recorded as unowned"
        )

    # local_rank is the GPU's index within its worker. The DCGM ``gpu`` label
    # is node-local (a packed node numbers a worker's GPUs 2,3), so rank the
    # worker's owned GPUs by that index instead of using it directly.
    local_ranks: dict[str, str] = {}
    owned: dict[str, list[dict[str, str]]] = {}
    for target in targets:
        if target.get("owner_pod") in workers_by_pod:
            owned.setdefault(target["owner_pod"], []).append(target)
    for pod_targets in owned.values():
        pod_targets.sort(key=lambda t: (len(t.get("gpu_index", "")), t.get("gpu_index", "")))
        for rank, target in enumerate(pod_targets):
            local_ranks[target["gpu_uuid"]] = str(rank)

    samples = []
    for target in targets:
        worker = workers_by_pod.get(target.get("owner_pod", ""))

        if worker is not None:
            node_name: str | None = worker.node_name or None
            instance_type = (instance_types_by_node or {}).get(node_name or "")
            if worker.worker_id not in worker_cache:
                price = (prices_by_instance_type or {}).get(instance_type or "")
                worker_cache[worker.worker_id] = _worker_inference_values(
                    prom,
                    worker,
                    price_per_hour=price,
                    ttft_target_ms=worker.ttft_target_ms,
                )
            inference_values = worker_cache[worker.worker_id]
        else:
            # dcgm-exporter runs with hostNetwork, so the instance IP is the
            # node's InternalIP.
            node_ip = target.get("instance", "").split(":", 1)[0]
            node_name = (node_names_by_ip or {}).get(node_ip)
            instance_type = (instance_types_by_node or {}).get(node_name or "")
            inference_values = none_inference

        gpu_values = {
            field: prom.query_scalar(query.format(gpu=_gpu_selector(target)))
            for field, query in GPU_QUERIES.items()
        }
        samples.append(
            GpuMetric(
                job_id=worker.job_id if worker else None,
                gpu_uuid=target["gpu_uuid"],
                rank_id=worker.rank_id if worker else None,
                chain_id=worker.chain_id if worker else None,
                # An idle GPU has no local_rank.
                local_rank=local_ranks.get(target["gpu_uuid"]) if worker else None,
                role=worker.role if worker else None,
                node_name=node_name,
                instance_type=instance_type,
                model_name=worker.model_name if worker else None,
                **gpu_values,
                **inference_values,
            )
        )
    return samples


def resolve_instance_price_per_hour(
    resource_map: ResourceMap, instance_type: str | None
) -> float | None:
    """Instance USD/hour for one instance type, from the user's resource map.

    Searches the map's machine pools (Orca fills ``price_per_instance_hour``
    from the hardware catalog). First priced pool wins; None when the
    instance type is absent or unpriced.
    """
    if instance_type is None:
        return None
    for *_head, pool_instance_type, pool in resource_map.iter_machine_pools():
        if pool_instance_type != instance_type or pool.price_per_instance_hour is None:
            continue
        return pool.price_per_instance_hour
    return None


def _ticks(once: bool) -> Iterator[None]:
    if once:
        yield None
        return
    while True:
        yield None
        time.sleep(COLLECT_INTERVAL_SECONDS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prometheus-url",
        default=os.getenv("TANDEMN_PROMETHEUS_URL", DEFAULT_PROMETHEUS_URL),
        help="Prometheus base URL (or TANDEMN_PROMETHEUS_URL)",
    )
    parser.add_argument(
        "--namespace", default="default", help="Kubernetes namespace of the worker pods"
    )
    parser.add_argument(
        "--user-id",
        default=os.getenv("TANDEMN_USER_ID"),
        help="Resource map owner (or TANDEMN_USER_ID); prices instances for cost_per_token",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help=f"Collect a single tick and exit (else loops every {COLLECT_INTERVAL_SECONDS}s)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    args = parse_args(argv)

    prom = PrometheusClient(args.prometheus_url)
    client = PostgresClient()
    store = GpuMetricStore(client)
    jobs = JobStore(client)
    resource_maps = (
        ResourceMapStore(client, user_id=args.user_id) if args.user_id is not None else None
    )
    kube = KubeWorkerIndex(namespace=args.namespace)

    for _ in _ticks(args.once):
        workers_by_pod = kube.by_pod()
        # Canonical chain ids, one job at a time (job ids come from pod labels).
        job_ids = sorted({w.job_id for w in workers_by_pod.values() if w.job_id})
        for job_id in job_ids:
            job_workers = {pod: w for pod, w in workers_by_pod.items() if w.job_id == job_id}
            assign_canonical_chain_ids(job_workers, jobs.active_chains(job_id))

        node_names_by_ip, instance_types_by_node = kube.nodes()
        # Re-read per tick: the map is one row and Orca republishes prices.
        resource_map = resource_maps.get() if resource_maps is not None else None
        prices_by_instance_type: dict[str, float | None] = {}
        if resource_map is not None:
            prices_by_instance_type = {
                instance_type: resolve_instance_price_per_hour(resource_map, instance_type)
                for instance_type in set(instance_types_by_node.values())
            }

        samples = collect_once(
            prom,
            workers_by_pod,
            node_names_by_ip=node_names_by_ip,
            instance_types_by_node=instance_types_by_node,
            prices_by_instance_type=prices_by_instance_type,
        )
        if samples:
            store.put_many(samples)
            logger.info(
                "wrote %d gpu_metrics rows (%d workers, %d jobs)",
                len(samples),
                len(workers_by_pod),
                len(job_ids),
            )
        else:
            logger.info("no GPUs found; nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

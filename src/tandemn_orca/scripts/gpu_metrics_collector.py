"""Collect GPU/inference telemetry from Prometheus into Tandemn Store.

Writes one ``GpuMetric`` row per physical GPU per tick. Granularity:

- GPU hardware metrics (DCGM) are scoped to that one GPU (by ``UUID``).
- Inference metrics (vLLM) are scoped to the chain/worker that owns the GPU (by
  the worker pod name), so a multi-replica deployment records each chain's own
  numbers instead of a deployment-wide sum.

Tandemn job model, coarse -> fine: a ``rank_id`` is a ladder rung (rank config)
realized by N chains / DP replicas; a ``chain_id`` is one serving unit == one
worker; a worker spans N GPUs, each with a ``local_rank`` (its index in the
worker). Each row therefore carries rank > chain > worker > local_rank, plus the
physical ``gpu_uuid``.

Each row also records the PD-disaggregation ``role`` ("prefill"/"decode", from
the Dynamo ``sub-component-type`` label; ``None`` for an aggregated worker) so
prefill and decode GPUs can be grouped and compared.

The GPU->chain join uses the Kubernetes API (``KubeWorkerIndex``): dcgm-exporter
series carry the node but not the worker pod, so the collector lists the
deployment's worker pods and maps node -> worker, lifting ``rank_id`` /
``chain_id`` / ``role`` from pod labels when Orca launched the ladder.

Run once (``--once``) or loop on a fixed ``COLLECT_INTERVAL_SECONDS`` cadence.
Metrics that are topology- or config-gated (NVLink/comm/expert;
``sm_utilization``) return no series and are left ``None``. ``cost_per_token``
and ``slo_margin`` use the ``--price-per-hour`` and ``--ttft-target-ms`` inputs.

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

from tandemn_system_data.clients import GpuMetricStore, PostgresClient
from tandemn_system_data.models import GpuMetric

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
                    "hostname": metric.get("Hostname", ""),
                    "instance": metric.get("instance", ""),
                    # DCGM's node-local GPU index (0,1,2,...) = the GPU's rank
                    # within its chain/worker for TP/PP parallelism.
                    "gpu_index": metric.get("gpu", ""),
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
_LABEL_DGD = "nvidia.com/dynamo-graph-deployment-name"
_LABEL_DYN_NS = "nvidia.com/dynamo-namespace"
_LABEL_COMPONENT = "nvidia.com/dynamo-component-type"
# PD-disaggregation role of the worker: "prefill" | "decode" (absent = aggregated).
_LABEL_SUBCOMPONENT = "nvidia.com/dynamo-sub-component-type"
# Optional tandemn job-model labels (present when Orca launched the ladder).
_LABEL_RANK = "tandemn.ai/rank-id"
_LABEL_CHAIN = "tandemn.ai/chain-id"


class WorkerInfo:
    """Identity of one Dynamo worker pod (== one chain / DP replica).

    Joined to the node it runs on. ``rank_id`` is the ladder rung the chain
    belongs to (shared across the rank's chains); ``chain_id`` is the serving
    unit; ``worker_id`` is the pod name. Per-GPU local ranks are resolved
    separately from the DCGM ``gpu`` index.
    """

    def __init__(
        self,
        *,
        worker_id: str,
        node_name: str,
        dynamo_namespace: str | None,
        rank_id: str | None,
        chain_id: str | None,
        role: str | None,
    ) -> None:
        self.worker_id = worker_id
        self.node_name = node_name
        self.dynamo_namespace = dynamo_namespace
        self.rank_id = rank_id
        self.chain_id = chain_id
        self.role = role


class KubeWorkerIndex:
    """Maps a GPU's node to the Dynamo worker pod (chain) serving a deployment.

    dcgm-exporter metrics carry the node (Hostname) but not the worker pod, so
    the node->worker join happens through the Kubernetes API rather than
    Prometheus labels. Single worker per node is assumed (one GPU worker per
    g6.xlarge); with multiple workers per node the first match on the node wins.
    """

    def __init__(self, deployment_id: str, namespace: str = "default", core: Any = None) -> None:
        from kubernetes import client, config

        if core is None:
            try:
                config.load_incluster_config()
            except Exception:
                config.load_kube_config()
            core = client.CoreV1Api()
        self._core = core
        self._deployment_id = deployment_id
        self._namespace = namespace

    def _node_ips(self) -> dict[str, str]:
        """node name -> InternalIP, to join DCGM's ``instance`` IP to a node."""
        ips: dict[str, str] = {}
        for node in self._core.list_node().items:
            for addr in node.status.addresses or []:
                if addr.type == "InternalIP":
                    ips[node.metadata.name] = addr.address
        return ips

    def by_node(self) -> dict[str, WorkerInfo]:
        """Worker pods of this deployment, keyed by node name AND node IP.

        dcgm-exporter runs with hostNetwork so its ``Hostname`` is often
        ``localhost``; the usable node key is the IP in its ``instance`` label.
        Registering each worker under both the node name and the node IP lets the
        collector join on whichever the DCGM series exposes.
        """
        selector = f"{_LABEL_DGD}={self._deployment_id},{_LABEL_COMPONENT}=worker"
        pods = self._core.list_namespaced_pod(self._namespace, label_selector=selector).items
        node_ips = self._node_ips()
        index: dict[str, WorkerInfo] = {}
        for pod in pods:
            node_name = pod.spec.node_name
            if not node_name:
                continue
            labels = pod.metadata.labels or {}
            # A chain == a worker; fall back to the pod name when Orca did not
            # stamp an explicit chain-id label. rank_id (the ladder rung) is only
            # known when Orca launched the ladder, else None.
            chain_id = labels.get(_LABEL_CHAIN) or pod.metadata.name
            info = WorkerInfo(
                worker_id=pod.metadata.name,
                node_name=node_name,
                dynamo_namespace=labels.get(_LABEL_DYN_NS),
                rank_id=labels.get(_LABEL_RANK),
                chain_id=chain_id,
                role=labels.get(_LABEL_SUBCOMPONENT),
            )
            index[node_name] = info
            node_ip = node_ips.get(node_name)
            if node_ip:
                index[node_ip] = info
        return index


def _worker_inference_values(
    prom: PrometheusClient,
    worker: WorkerInfo | None,
    *,
    price_per_hour: float | None,
    ttft_target_ms: float | None,
) -> dict[str, float | None]:
    """Inference metrics scoped to one worker (pod), plus derived cost/SLO.

    The PodMonitor attaches a ``pod`` label equal to the worker pod name, so
    ``pod="<worker_id>"`` scopes vLLM series to that single worker.
    """
    selector = f'pod="{worker.worker_id}"' if worker is not None else ""
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
    workers_by_node: dict[str, WorkerInfo],
    *,
    deployment_id: str,
    model_name: str | None,
    instance_type: str | None,
    price_per_hour: float | None,
    ttft_target_ms: float | None,
) -> list[GpuMetric]:
    """One GpuMetric per GPU: GPU metrics per-GPU, inference metrics per-chain.

    ``workers_by_node`` maps node name -> worker (from KubeWorkerIndex). A GPU's
    node picks its chain/worker, so inference metrics are scoped to that chain
    rather than summed across the whole deployment. Identity is nested coarse ->
    fine: ``rank_id`` (ladder rung, shared across the rank's chains) >
    ``chain_id`` (== worker) > ``worker_id`` > ``local_rank`` (the GPU's index
    within its worker, from the DCGM ``gpu`` label).
    """
    # Inference metrics are per-worker; cache so GPUs sharing a worker (TP>1)
    # reuse one query pass.
    worker_cache: dict[str, dict[str, float | None]] = {}

    samples = []
    for target in prom.gpu_targets():
        # dcgm-exporter's Hostname is often "localhost" (hostNetwork), so fall
        # back to the node IP parsed from the DCGM "instance" label.
        node_key = target.get("hostname", "")
        worker = workers_by_node.get(node_key)
        if worker is None:
            node_ip = target.get("instance", "").split(":", 1)[0]
            worker = workers_by_node.get(node_ip)
        # Prefer the worker's real node name over a "localhost" Hostname.
        node_name = worker.node_name if worker is not None else (node_key or None)

        cache_key = worker.worker_id if worker is not None else ""
        if cache_key not in worker_cache:
            worker_cache[cache_key] = _worker_inference_values(
                prom, worker, price_per_hour=price_per_hour, ttft_target_ms=ttft_target_ms
            )
        inference_values = worker_cache[cache_key]

        gpu_values = {
            field: prom.query_scalar(query.format(gpu=_gpu_selector(target)))
            for field, query in GPU_QUERIES.items()
        }
        samples.append(
            GpuMetric(
                deployment_id=deployment_id,
                gpu_uuid=target["gpu_uuid"],
                rank_id=worker.rank_id if worker else None,
                chain_id=worker.chain_id if worker else None,
                worker_id=worker.worker_id if worker else None,
                local_rank=target.get("gpu_index") or None,
                role=worker.role if worker else None,
                node_name=node_name,
                instance_type=instance_type,
                model_name=model_name,
                **gpu_values,
                **inference_values,
            )
        )
    return samples


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
        "--deployment-id", required=True, help="Dynamo graph deployment name, e.g. qwen3-06b-l4"
    )
    parser.add_argument(
        "--namespace", default="default", help="Kubernetes namespace of the worker pods"
    )
    parser.add_argument("--model-name", default=None, help="Served model name for vLLM queries")
    parser.add_argument("--instance-type", default=None, help="EC2 instance type for the GPUs")
    parser.add_argument(
        "--price-per-hour", type=float, default=None, help="Instance $/hour for cost_per_token"
    )
    parser.add_argument(
        "--ttft-target-ms", type=float, default=None, help="TTFT SLO target (ms) for slo_margin"
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
    store = GpuMetricStore(PostgresClient())
    kube = KubeWorkerIndex(args.deployment_id, namespace=args.namespace)

    for _ in _ticks(args.once):
        workers_by_node = kube.by_node()
        samples = collect_once(
            prom,
            workers_by_node,
            deployment_id=args.deployment_id,
            model_name=args.model_name,
            instance_type=args.instance_type,
            price_per_hour=args.price_per_hour,
            ttft_target_ms=args.ttft_target_ms,
        )
        if samples:
            store.put_many(samples)
            logger.info("wrote %d gpu_metrics rows for %s", len(samples), args.deployment_id)
        else:
            logger.info("no GPUs found; nothing written for %s", args.deployment_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

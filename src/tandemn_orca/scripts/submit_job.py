"""Submit a job to Tandemn Store.

The one production entry point for creating ``jobs`` rows: Koi plans for
existing jobs and Orca transitions them, but neither creates them. New jobs
start ``waiting``; Koi's next pass may place them.

``--spec`` is the job's spec_json. It must carry ``model_id`` -- Koi's ladder
configs do not, and Orca backfills chain shapes from the job spec.

Online traffic intent is mutually exclusive. A user may submit either:

* ``requestRate`` / ``request_arrival_rate``: peak request arrivals per second.
  Orca stores this as ``request_arrival_rate`` and derives
  ``max_concurrent_streaming``. Downstream Koi keeps request-rate mode and the
  surrogate turns it into DynoSim ``arrival_interval_ms``.
* ``concurrency`` / ``max_concurrent_streaming``: target active streaming
  requests. Orca stores this as ``max_concurrent_streaming`` and derives
  ``request_arrival_rate``. Downstream Koi keeps concurrency mode and the
  surrogate turns it into DynoSim ``replay_concurrency``.

Do not submit both. The derived field is only for Koi's existing X schema and
throughput sizing; ``_traffic_mode`` records which user intent is authoritative.

Postgres: ``TANDEMN_POSTGRES_URL`` (see tandemn-store).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from typing import Any

from tandemn_system_data.clients import JobStore, PostgresClient
from tandemn_system_data.models import Job, JobKind

logger = logging.getLogger(__name__)

WORKLOAD_DEFAULTS = {
    "isl_token_avg": 4000,
    "isl_token_min": 1,
    "isl_token_max": 8000,
    "isl_distribution_type": "LogNormal",
    "osl_token_avg": 1000,
    "osl_token_min": 1,
    "osl_token_max": 4000,
    "osl_distribution_type": "LogNormal",
    "request_arrival_rate": 1.0,
    "request_arrival_pattern": "Poisson",
    "peak_to_mean_ratio": 2,
    "workload_prefix_concentration": 0,
    "multi_turn_ratio": 0.5,
    "shared_prefix_length_avg": 500,
    "sliding_window_attention": 0,
    "is_session_affinity": True,
    "total_token_budget": 1_000_000,
    "priority_class": "STANDARD",
    "max_concurrent_streaming": 10,
    "preemption_policy": "lifo",
    "router_policy": "kv_router",
}
TRAFFIC_ALIASES = {
    "requestRate": "request_arrival_rate",
    "concurrency": "max_concurrent_streaming",
}
# Metadata, not a Koi X: tells the surrogate whether the user meant
# request-rate replay (arrival_interval_ms) or closed-loop replay_concurrency.
TRAFFIC_MODE_KEY = "_traffic_mode"
TRAFFIC_MODE_REQUEST_RATE = "request_rate"
TRAFFIC_MODE_CONCURRENCY = "concurrency"
REQUEST_RATE_KEYS = frozenset({"requestRate", "request_arrival_rate"})
CONCURRENCY_KEYS = frozenset({"concurrency", "max_concurrent_streaming"})
USER_FEATURE_KEYS = frozenset(
    {
        *WORKLOAD_DEFAULTS,
        *TRAFFIC_ALIASES,
        "deadline_hrs",
        "target_p99_ttft_ms",
        "target_p99_tpot_ms",
    }
)
DIST_TYPES = {"Constant", "Uniform", "Normal", "LogNormal", "Exponential"}
ARRIVAL_PATTERNS = {"Poisson", "Constant", "Bursty", "Diurnal"}
PRIORITY_CLASSES = {"HIGH", "STANDARD", "LOW"}
PREEMPTION_POLICIES = {"lifo", "fifo"}
ROUTER_POLICIES = {"kv_router"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--user-id",
        default=os.getenv("TANDEMN_USER_ID"),
        help="Job owner (or TANDEMN_USER_ID)",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--online", action="store_true", help="Submit an online serving job")
    mode.add_argument("--batched", action="store_true", help="Submit a batched/offline job")
    parser.add_argument(
        "--spec",
        required=True,
        help='Job spec_json as a JSON object, e.g. \'{"model_id": "Qwen/Qwen3-0.6B"}\'',
    )
    return parser.parse_args(argv)


def normalize_job_spec(spec: dict[str, Any], kind: JobKind) -> dict[str, Any]:
    """Return canonical spec_json with Koi's user-owned X defaults."""
    model_id = spec.get("model_id")
    if not model_id:
        raise SystemExit("--spec must include model_id (Orca backfills chain shapes from it)")

    provided_request_rate = _has_any_user_key(spec, REQUEST_RATE_KEYS)
    provided_concurrency = _has_any_user_key(spec, CONCURRENCY_KEYS)
    if provided_request_rate and provided_concurrency:
        raise SystemExit(
            "--spec must provide only one of requestRate/request_arrival_rate or "
            "concurrency/max_concurrent_streaming"
        )

    features = dict(WORKLOAD_DEFAULTS)
    if isinstance(spec.get("job_features"), dict):
        features.update(_canonical_user_features(spec["job_features"]))
    features.update(_canonical_user_features(spec))
    features["model_id"] = model_id
    features["type"] = kind.value
    _canonicalize_runtime_policy(features)

    if kind is JobKind.ONLINE:
        for key in ("target_p99_ttft_ms", "target_p99_tpot_ms"):
            if features.get(key) is None:
                raise SystemExit(f"--spec missing required field {key}")
        features.setdefault("deadline_hrs", 0)
        _derive_online_traffic(features, provided_request_rate, provided_concurrency)
    else:
        deadline = features.get("deadline_hrs")
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or deadline <= 0:
            raise SystemExit("--spec field deadline_hrs must be a positive number")

    features["pd_ratio"] = float(features["isl_token_avg"]) / float(features["osl_token_avg"])
    _validate_features(features)
    out = dict(spec)
    out["model_id"] = model_id
    out["job_features"] = features
    return out


def _canonical_user_features(source: dict[str, Any]) -> dict[str, Any]:
    return {
        TRAFFIC_ALIASES.get(key, key): value
        for key, value in source.items()
        if key in USER_FEATURE_KEYS
    }


def _has_any_user_key(spec: dict[str, Any], keys: frozenset[str]) -> bool:
    if any(key in spec for key in keys):
        return True
    nested = spec.get("job_features")
    return isinstance(nested, dict) and any(key in nested for key in keys)


def _derive_online_traffic(
    features: dict[str, Any], provided_request_rate: bool, provided_concurrency: bool
) -> None:
    if not provided_request_rate and not provided_concurrency:
        features[TRAFFIC_MODE_KEY] = TRAFFIC_MODE_REQUEST_RATE
        return
    service_time_s = _target_service_time_seconds(features)
    if provided_request_rate:
        features[TRAFFIC_MODE_KEY] = TRAFFIC_MODE_REQUEST_RATE
        request_rate = _positive_float(features, "request_arrival_rate")
        features["request_arrival_rate"] = request_rate
        features["max_concurrent_streaming"] = max(1, math.ceil(request_rate * service_time_s))
        return
    features[TRAFFIC_MODE_KEY] = TRAFFIC_MODE_CONCURRENCY
    concurrency = max(1, math.ceil(_positive_float(features, "max_concurrent_streaming")))
    features["max_concurrent_streaming"] = concurrency
    features["request_arrival_rate"] = concurrency / service_time_s


def _target_service_time_seconds(features: dict[str, Any]) -> float:
    ttft_ms = _positive_float(features, "target_p99_ttft_ms")
    tpot_ms = _positive_float(features, "target_p99_tpot_ms")
    osl = _positive_float(features, "osl_token_avg")
    return (ttft_ms + tpot_ms * osl) / 1000.0


def _positive_float(features: dict[str, Any], key: str) -> float:
    value = features.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise SystemExit(f"--spec field {key} must be a positive number")
    return float(value)


def _canonicalize_runtime_policy(features: dict[str, Any]) -> None:
    _force_allowed(features, "preemption_policy", PREEMPTION_POLICIES, "lifo")
    _force_allowed(features, "router_policy", ROUTER_POLICIES, "kv_router")


def _force_allowed(features: dict[str, Any], key: str, allowed: set[str], default: str) -> None:
    value = features.get(key)
    if value in allowed:
        return
    message = f"--spec {key}={value!r} is unsupported; using {default!r}"
    print(f"warning: {message}", file=sys.stderr)
    logger.warning(message)
    features[key] = default


def _validate_features(features: dict[str, Any]) -> None:
    for key in (
        "isl_token_avg",
        "isl_token_min",
        "isl_token_max",
        "osl_token_avg",
        "osl_token_min",
        "osl_token_max",
        "request_arrival_rate",
        "max_concurrent_streaming",
    ):
        _positive_float(features, key)
    for key, allowed in (
        ("isl_distribution_type", DIST_TYPES),
        ("osl_distribution_type", DIST_TYPES),
        ("request_arrival_pattern", ARRIVAL_PATTERNS),
        ("priority_class", PRIORITY_CLASSES),
    ):
        if features[key] not in allowed:
            raise SystemExit(f"--spec {key} must be one of {sorted(allowed)}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    args = parse_args(argv)
    if not args.user_id:
        raise SystemExit("--user-id or TANDEMN_USER_ID is required")
    try:
        spec = json.loads(args.spec)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--spec must be valid JSON: {exc}") from exc
    if not isinstance(spec, dict):
        raise SystemExit("--spec must be a JSON object")
    kind = JobKind.ONLINE if args.online else JobKind.BATCH
    spec = normalize_job_spec(spec, kind)

    job = Job(user_id=args.user_id, kind=kind, spec_json=spec)
    JobStore(PostgresClient()).submit(job)
    logger.info("submitted job %s (%s) for user %s", job.job_id, job.kind, job.user_id)
    print(job.job_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

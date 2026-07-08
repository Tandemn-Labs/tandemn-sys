"""Submit a job to Tandemn Store.

The one production entry point for creating ``jobs`` rows: Koi plans for
existing jobs and Orca transitions them, but neither creates them. New jobs
start ``waiting``; Koi's next pass may place them.

``--spec`` is the job's spec_json. It must carry ``model_id`` -- Koi's ladder
configs do not, and Orca backfills chain shapes from the job spec.

Postgres: ``TANDEMN_POSTGRES_URL`` (see tandemn-store).
"""

from __future__ import annotations

import argparse
import json
import logging
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
    "is_session_affinity": True,
    "total_token_budget": 1_000_000,
    "priority_class": "STANDARD",
    "max_concurrent_streaming": 10,
}
USER_FEATURE_KEYS = frozenset(
    {
        *WORKLOAD_DEFAULTS,
        "deadline_hrs",
        "target_p99_ttft_ms",
        "target_p99_tpot_ms",
    }
)
DIST_TYPES = {"Constant", "Uniform", "Normal", "LogNormal", "Exponential"}
ARRIVAL_PATTERNS = {"Poisson", "Constant", "Bursty", "Diurnal"}
PRIORITY_CLASSES = {"HIGH", "STANDARD", "LOW"}


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

    features = dict(WORKLOAD_DEFAULTS)
    if isinstance(spec.get("job_features"), dict):
        features.update(
            {key: value for key, value in spec["job_features"].items() if key in USER_FEATURE_KEYS}
        )
    features.update({key: value for key, value in spec.items() if key in USER_FEATURE_KEYS})
    features["model_id"] = model_id
    features["type"] = kind.value

    if kind is JobKind.ONLINE:
        for key in ("target_p99_ttft_ms", "target_p99_tpot_ms"):
            if features.get(key) is None:
                raise SystemExit(f"--spec missing required field {key}")
        features.setdefault("deadline_hrs", 0)
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


def _validate_features(features: dict[str, Any]) -> None:
    for key in (
        "isl_token_avg",
        "isl_token_min",
        "isl_token_max",
        "osl_token_avg",
        "osl_token_min",
        "osl_token_max",
        "request_arrival_rate",
    ):
        value = features.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise SystemExit(f"--spec field {key} must be a positive number")
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

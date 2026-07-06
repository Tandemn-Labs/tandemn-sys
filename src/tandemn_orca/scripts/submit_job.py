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

from tandemn_system_data.clients import JobStore, PostgresClient
from tandemn_system_data.models import Job, JobKind

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--user-id",
        default=os.getenv("TANDEMN_USER_ID"),
        help="Job owner (or TANDEMN_USER_ID)",
    )
    parser.add_argument(
        "--kind", choices=[kind.value for kind in JobKind], default=JobKind.ONLINE.value
    )
    parser.add_argument(
        "--spec",
        required=True,
        help='Job spec_json as a JSON object, e.g. \'{"model_id": "Qwen/Qwen3-0.6B"}\'',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    args = parse_args(argv)
    if not args.user_id:
        raise SystemExit("--user-id or TANDEMN_USER_ID is required")
    spec = json.loads(args.spec)
    if not isinstance(spec, dict):
        raise SystemExit("--spec must be a JSON object")
    if not spec.get("model_id"):
        raise SystemExit("--spec must include model_id (Orca backfills chain shapes from it)")

    job = Job(user_id=args.user_id, kind=JobKind(args.kind), spec_json=spec)
    JobStore(PostgresClient()).submit(job)
    logger.info("submitted job %s (%s) for user %s", job.job_id, job.kind, job.user_id)
    print(job.job_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

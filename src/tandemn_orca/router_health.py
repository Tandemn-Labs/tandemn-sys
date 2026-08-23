"""Publish rank serving health to each job's router process.

The router decides which rank a session lands on and must not route to a rank
whose workers are gone. It is a Go process with no database access, so Orca --
which already holds the Kubernetes credentials and the Postgres connection --
pushes the verdicts it derived from DGD status.

Level-triggered: every poll sends the full set of active ranks for a job. A rank
that drops out of the payload has left the active set (failed, stopped, or
replaced by a new plan), and the router ages it out rather than pinning a stale
verdict forever.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime

from tandemn_orca.dynamo_compiler import router_listen_port
from tandemn_orca.rank_health import RankHealth

logger = logging.getLogger(__name__)

RANK_STATUS_PATH = "/internal/rank-status"


def rank_status_payload(
    job_id: str,
    ranks: Sequence[RankHealth],
    observed_at: datetime,
) -> dict[str, object]:
    """Build the router's rank-status body.

    Wire contract with tandemn-router's RankStatusReport. That decoder sets
    DisallowUnknownFields, so an extra key here is a 400 rather than a warning:
    the two sides must change together.
    """
    return {
        "job_id": job_id,
        "observed_at": observed_at.isoformat(),
        "ranks": [
            {
                "rank_id": item.rank_id,
                "verdict": item.verdict.value,
                "serving_replicas": item.serving_replicas,
                "reason": item.detail,
            }
            for item in ranks
        ],
    }


class RankHealthPublisher:
    """POST rank health to the per-job router that owns those ranks."""

    def __init__(
        self,
        token: str,
        url_template: str | None = None,
        timeout: float = 5.0,
    ) -> None:
        self.token = token
        self.url_template = url_template
        self.timeout = timeout

    def publish(self, health: Sequence[RankHealth]) -> int:
        """Send one batch per job. Returns the number of jobs published."""
        by_job: dict[str, list[RankHealth]] = defaultdict(list)
        for item in health:
            by_job[item.job_id].append(item)

        observed_at = datetime.now(UTC)
        published = 0
        for job_id, ranks in by_job.items():
            payload = rank_status_payload(job_id, ranks, observed_at)
            try:
                self._post(job_id, payload)
                published += 1
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                # A router that is down or not yet started is not an Orca
                # failure; the next poll republishes the full set.
                logger.warning("rank health push failed for job %s: %s", job_id, error)
        return published

    def _post(self, job_id: str, payload: dict[str, object]) -> None:
        if self.url_template:
            base_url = self.url_template.format(job_id=job_id).rstrip("/")
        else:
            base_url = f"http://127.0.0.1:{router_listen_port(job_id)}"
        request = urllib.request.Request(
            f"{base_url}{RANK_STATUS_PATH}",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout):
            pass

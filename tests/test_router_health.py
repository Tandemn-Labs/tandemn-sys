"""Publishing rank health to the per-job router.

The payload shape is a cross-repository contract: tandemn-router decodes it with
DisallowUnknownFields, and its config-reload goroutine logs and continues on a
rejected request rather than exiting, so a mismatch fails silently. The Go side
holds the matching test over testdata/rank_status_orca.json.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

import tandemn_orca.router_health as router_health_mod
from tandemn_orca.dynamo_compiler import router_listen_port
from tandemn_orca.rank_health import RankHealth, Verdict
from tandemn_orca.router_health import RankHealthPublisher, rank_status_payload

JOB_ID = "job-01JBM2YQYZ1KQ9C8GZP1XB6V5T"
RANK_ID = "rank_01JBM2YQYZ1KQ9C8GZP1XB6V5T"
OTHER_RANK_ID = "rank_01JBM30YQ7X3WQAR6HF8C2Q9T8"


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _capture(monkeypatch) -> list:
    sent: list = []

    def urlopen(request, timeout=None):
        sent.append((request.full_url, json.loads(request.data), dict(request.headers)))
        return FakeResponse()

    monkeypatch.setattr(router_health_mod.urllib.request, "urlopen", urlopen)
    return sent


def test_payload_uses_the_wire_vocabulary():
    payload = rank_status_payload(
        JOB_ID,
        [
            RankHealth(RANK_ID, JOB_ID, Verdict.SERVING, 2, None, ""),
            RankHealth(OTHER_RANK_ID, JOB_ID, Verdict.UNKNOWN, None, None, "0 workers (1/2)"),
        ],
        datetime(2026, 8, 21, 12, 0, 0, tzinfo=UTC),
    )

    assert payload == {
        "job_id": JOB_ID,
        "observed_at": "2026-08-21T12:00:00+00:00",
        "ranks": [
            {"rank_id": RANK_ID, "verdict": "serving", "serving_replicas": 2, "reason": ""},
            {
                "rank_id": OTHER_RANK_ID,
                "verdict": "unknown",
                "serving_replicas": None,
                "reason": "0 workers (1/2)",
            },
        ],
    }


def test_payload_keeps_zero_distinct_from_no_count():
    payload = rank_status_payload(
        JOB_ID,
        [RankHealth(RANK_ID, JOB_ID, Verdict.DOWN, 0, "OOM", "worker died")],
        datetime.now(UTC),
    )
    assert payload["ranks"][0]["serving_replicas"] == 0


def test_publish_posts_to_the_jobs_deterministic_router_port(monkeypatch):
    sent = _capture(monkeypatch)
    publisher = RankHealthPublisher("secret-token")

    assert publisher.publish([RankHealth(RANK_ID, JOB_ID, Verdict.SERVING, 1, None, "")]) == 1

    url, body, headers = sent[0]
    assert url == f"http://127.0.0.1:{router_listen_port(JOB_ID)}/internal/rank-status"
    assert body["job_id"] == JOB_ID
    assert headers["Authorization"] == "Bearer secret-token"


def test_publish_sends_one_batch_per_job(monkeypatch):
    sent = _capture(monkeypatch)
    other_job = "job-01JBM30YQ7X3WQAR6HF8C2Q9T8"
    health = [
        RankHealth(RANK_ID, JOB_ID, Verdict.SERVING, 1, None, ""),
        RankHealth(OTHER_RANK_ID, JOB_ID, Verdict.DOWN, 0, "OOM", "died"),
        RankHealth(RANK_ID, other_job, Verdict.SERVING, 1, None, ""),
    ]

    assert RankHealthPublisher("t").publish(health) == 2

    by_job = {body["job_id"]: body for _, body, _ in sent}
    assert len(by_job[JOB_ID]["ranks"]) == 2
    assert len(by_job[other_job]["ranks"]) == 1


def test_a_router_that_is_down_does_not_break_the_poll(monkeypatch):
    def urlopen(request, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(router_health_mod.urllib.request, "urlopen", urlopen)

    assert (
        RankHealthPublisher("t").publish([RankHealth(RANK_ID, JOB_ID, Verdict.DOWN, 0, None, "")])
        == 0
    )


@pytest.mark.parametrize("template", ["http://router.local:9000", "http://router.local:9000/"])
def test_url_template_overrides_the_derived_port(monkeypatch, template):
    sent = _capture(monkeypatch)

    RankHealthPublisher("t", url_template=template).publish(
        [RankHealth(RANK_ID, JOB_ID, Verdict.SERVING, 1, None, "")]
    )

    assert sent[0][0] == "http://router.local:9000/internal/rank-status"

from __future__ import annotations

import json

import pytest

from tandemn_orca.dynamo_compiler import (
    ROUTER_LISTEN_PORT_BASE,
    ROUTER_LISTEN_PORT_SPAN,
    router_listen_port,
)
from tandemn_orca.scripts.endpoint import resolve_endpoint


def test_router_listen_port_is_stable_and_in_range():
    first = router_listen_port("job_01ABC")
    assert first == router_listen_port("job_01ABC")
    assert ROUTER_LISTEN_PORT_BASE <= first < ROUTER_LISTEN_PORT_BASE + ROUTER_LISTEN_PORT_SPAN
    assert first != router_listen_port("job_01XYZ")


def test_resolve_endpoint_reads_listen_port(tmp_path):
    (tmp_path / "job_1.json").write_text(json.dumps({"job_id": "job_1", "listen_port": 28123}))

    assert resolve_endpoint("job_1", tmp_path) == "http://127.0.0.1:28123"


def test_resolve_endpoint_derives_port_for_old_configs(tmp_path):
    (tmp_path / "job_1.json").write_text(json.dumps({"job_id": "job_1"}))

    assert resolve_endpoint("job_1", tmp_path) == f"http://127.0.0.1:{router_listen_port('job_1')}"


def test_resolve_endpoint_fails_without_config(tmp_path):
    with pytest.raises(SystemExit, match="no router config"):
        resolve_endpoint("job_missing", tmp_path)


def test_resolve_endpoint_rejects_invalid_port(tmp_path):
    (tmp_path / "job_1.json").write_text(json.dumps({"job_id": "job_1", "listen_port": -5}))

    with pytest.raises(SystemExit, match="invalid listen_port"):
        resolve_endpoint("job_1", tmp_path)

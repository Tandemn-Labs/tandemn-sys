"""Unit test for the AWS capacity-reservation -> ResourceMap mapping.

No AWS calls: fetch_reservations is monkeypatched to return canned
reservation dicts shaped like describe_capacity_reservations output.
"""

from __future__ import annotations

import json

import tandemn_orca.scripts.resource_map_from_aws as mod
from tandemn_orca.scripts.resource_map_from_aws import (
    build_resource_map,
    main,
    region_of_az,
    resource_map_to_pools_json,
)


def _cr(instance_type, az, count, cr_id="cr-x"):
    return {
        "CapacityReservationId": cr_id,
        "InstanceType": instance_type,
        "AvailabilityZone": az,
        "TotalInstanceCount": count,
        "State": "active",
    }


def test_region_of_az():
    assert region_of_az("us-east-1a") == "us-east-1"
    assert region_of_az("us-west-2c") == "us-west-2"
    assert region_of_az("") == ""


def test_build_maps_gpu_reservation_into_hierarchy(monkeypatch):
    monkeypatch.setattr(
        mod,
        "fetch_reservations",
        lambda region: [_cr("p5.48xlarge", "us-east-1a", 4)],
    )
    rm = build_resource_map(["us-east-1"])

    assert rm.market == ["reserved"]
    pool = (
        rm.clouds["aws"]
        .regions["us-east-1"]
        .zones["us-east-1a"]
        .network_fabrics["default"]
        .machine_pools["p5.48xlarge"]
    )
    assert pool.gpu_type == "H100"
    assert pool.gpus_per_instance == 8
    assert pool.total_instances == 4
    assert pool.total_gpus == 32


def test_build_sums_duplicate_reservations(monkeypatch):
    monkeypatch.setattr(
        mod,
        "fetch_reservations",
        lambda region: [
            _cr("p5.48xlarge", "us-east-1a", 4, "cr-1"),
            _cr("p5.48xlarge", "us-east-1a", 2, "cr-2"),
        ],
    )
    rm = build_resource_map(["us-east-1"])
    pool = (
        rm.clouds["aws"]
        .regions["us-east-1"]
        .zones["us-east-1a"]
        .network_fabrics["default"]
        .machine_pools["p5.48xlarge"]
    )
    assert pool.total_instances == 6


def test_build_skips_unknown_instance_types(monkeypatch):
    monkeypatch.setattr(
        mod,
        "fetch_reservations",
        lambda region: [_cr("m5.large", "us-east-1a", 10)],
    )
    rm = build_resource_map(["us-east-1"])
    assert rm.clouds == {}


def test_fetch_all_regions(monkeypatch):
    class _FakeEC2:
        def describe_regions(self):
            return {"Regions": [{"RegionName": "us-west-2"}, {"RegionName": "us-east-1"}]}

    monkeypatch.setattr(mod.boto3, "client", lambda *a, **k: _FakeEC2())
    assert mod.fetch_all_regions() == ["us-east-1", "us-west-2"]


def test_main_without_regions_scans_all_enabled(monkeypatch, capsys):
    monkeypatch.setattr(mod, "fetch_all_regions", lambda: ["us-east-1", "us-west-2"])
    scanned: list[str] = []

    def _fake_fetch(region):
        scanned.append(region)
        if region == "us-west-2":
            return [_cr("p4d.24xlarge", "us-west-2c", 2)]
        return []

    monkeypatch.setattr(mod, "fetch_reservations", _fake_fetch)
    assert main([]) == 0

    assert scanned == ["us-east-1", "us-west-2"]
    body = json.loads(capsys.readouterr().out)
    pool = body["clouds"]["aws"]["regions"]["us-west-2"]["zones"]["us-west-2c"][
        "network_fabrics"
    ]["default"]["machine_pools"]["p4d.24xlarge"]
    assert pool["gpu_type"] == "A100"


def test_pools_json_wire_shape(monkeypatch):
    monkeypatch.setattr(
        mod,
        "fetch_reservations",
        lambda region: [_cr("g6e.12xlarge", "us-east-2b", 8)],
    )
    rm = build_resource_map(["us-east-2"])
    body = resource_map_to_pools_json(rm)
    assert set(body) == {"market", "clouds"}
    assert body["market"] == ["reserved"]
    pool = body["clouds"]["aws"]["regions"]["us-east-2"]["zones"]["us-east-2b"]["network_fabrics"][
        "default"
    ]["machine_pools"]["g6e.12xlarge"]
    assert pool["gpu_type"] == "L40S"
    assert pool["total_instances"] == 8

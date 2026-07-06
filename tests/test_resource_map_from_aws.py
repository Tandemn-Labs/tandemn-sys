"""Unit test for the AWS capacity-reservation -> ResourceMap mapping.

No AWS calls: fetch_reservations is monkeypatched to return canned
reservation dicts shaped like describe_capacity_reservations output.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from tandemn_system_data.models.resource_map import (
    Cloud,
    MachinePool,
    NetworkFabric,
    Region,
    ResourceMap,
    Zone,
)

import tandemn_orca.scripts.resource_map_from_aws as mod
from tandemn_orca.scripts.resource_map_from_aws import (
    CapacityRefresher,
    build_resource_map,
    main,
    parse_region_csv,
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


def _resource_map() -> ResourceMap:
    return ResourceMap(
        clouds={
            "aws": Cloud(
                regions={
                    "us-east-2": Region(
                        zones={
                            "us-east-2a": Zone(
                                network_fabrics={
                                    "default": NetworkFabric(
                                        fabric_type="default",
                                        machine_pools={
                                            "g6e.12xlarge": MachinePool(
                                                gpu_type="L40S",
                                                gpu_memory_gb=48,
                                                gpus_per_instance=4,
                                                total_instances=2,
                                            )
                                        },
                                    )
                                }
                            )
                        }
                    )
                }
            )
        }
    )


def _catalog() -> dict:
    return {
        "regions": [
            {
                "region": "us-east-1",
                "instance_types": [
                    _catalog_instance("p5.48xlarge", "p5", "H100", 8, 81920, 98.32),
                ],
            },
            {
                "region": "us-east-2",
                "instance_types": [
                    _catalog_instance("g6e.12xlarge", "g6e", "L40S", 4, 49152, 3.21),
                ],
            },
            {
                "region": "us-west-2",
                "instance_types": [
                    _catalog_instance("p4d.24xlarge", "p4d", "A100", 8, 40960, 32.77),
                ],
            },
        ]
    }


def _catalog_instance(instance_type, family, gpu_type, count, memory_mib_each, price):
    return {
        "instance_type": instance_type,
        "family": family,
        "accelerators": [
            {
                "kind": "gpu",
                "name": gpu_type,
                "count": count,
                "memory_mib_each": memory_mib_each,
            }
        ],
        "pricing": {"capacity_reservation_usd_per_hour": price},
    }


def test_region_of_az():
    assert region_of_az("us-east-1a") == "us-east-1"
    assert region_of_az("us-west-2c") == "us-west-2"
    assert region_of_az("") == ""


def test_parse_region_csv():
    assert parse_region_csv("us-east-1, us-east-2") == ["us-east-1", "us-east-2"]
    with pytest.raises(ValueError):
        parse_region_csv("")


def test_build_maps_gpu_reservation_into_hierarchy(monkeypatch):
    monkeypatch.setattr(
        mod,
        "fetch_reservations",
        lambda region: [_cr("p5.48xlarge", "us-east-1a", 4)],
    )
    rm = build_resource_map(["us-east-1"], _catalog())

    assert rm.market == ["reserved"]
    pool = (
        rm.clouds["aws"]
        .regions["us-east-1"]
        .zones["us-east-1a"]
        .network_fabrics["default"]
        .machine_pools["p5.48xlarge"]
    )
    assert pool.gpu_type == "H100"
    assert pool.instance_family == "p5"
    assert pool.gpus_per_instance == 8
    assert pool.gpu_memory_gb == 80
    assert pool.price_per_instance_hour == 98.32
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
    rm = build_resource_map(["us-east-1"], _catalog())
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
    rm = build_resource_map(["us-east-1"], _catalog())
    assert rm.clouds == {}


def test_fetch_all_regions(monkeypatch):
    class _FakeEC2:
        def describe_regions(self):
            return {"Regions": [{"RegionName": "us-west-2"}, {"RegionName": "us-east-1"}]}

    monkeypatch.setattr(mod.boto3, "client", lambda *a, **k: _FakeEC2())
    assert mod.fetch_all_regions() == ["us-east-1", "us-west-2"]


def test_main_without_regions_scans_all_enabled(monkeypatch, capsys):
    monkeypatch.setattr(mod, "fetch_all_regions", lambda: ["us-east-1", "us-west-2"])
    monkeypatch.setattr(mod, "build_catalogs", lambda regions: _catalog())
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
    pool = body["clouds"]["aws"]["regions"]["us-west-2"]["zones"]["us-west-2c"]["network_fabrics"][
        "default"
    ]["machine_pools"]["p4d.24xlarge"]
    assert pool["gpu_type"] == "A100"


def test_pools_json_wire_shape(monkeypatch):
    monkeypatch.setattr(
        mod,
        "fetch_reservations",
        lambda region: [_cr("g6e.12xlarge", "us-east-2b", 8)],
    )
    rm = build_resource_map(["us-east-2"], _catalog())
    body = resource_map_to_pools_json(rm)
    assert set(body) == {"market", "clouds"}
    assert body["market"] == ["reserved"]
    pool = body["clouds"]["aws"]["regions"]["us-east-2"]["zones"]["us-east-2b"]["network_fabrics"][
        "default"
    ]["machine_pools"]["g6e.12xlarge"]
    assert pool["gpu_type"] == "L40S"
    assert pool["total_instances"] == 8


def test_refresh_capacity_stores_enriched_resource_map(monkeypatch):
    replaced: list[ResourceMap] = []
    stored_catalogs: list[dict] = []

    class FakeResourceMapStore:
        def __init__(self, client, *, user_id):
            self.client = client
            self.user_id = user_id

        def replace(self, resource_map):
            replaced.append(resource_map)

    class FakeCatalogStore:
        def __init__(self, client):
            self.client = client

        def get(self):
            return None

        def replace(self, catalog):
            stored_catalogs.append(catalog)
            return mod.CatalogSnapshot(catalog=catalog, updated_at=datetime.now(UTC))

    monkeypatch.setattr(mod, "build_catalogs", lambda regions: _catalog())
    monkeypatch.setattr(
        mod, "fetch_reservations", lambda region: [_cr("g6e.12xlarge", "us-east-2a", 2)]
    )
    monkeypatch.setattr(mod, "ResourceMapStore", FakeResourceMapStore)
    monkeypatch.setattr(mod, "HardwareCatalogStore", FakeCatalogStore)

    snapshot = mod.refresh_capacity("client", "user_1", ["us-east-2"])

    assert snapshot.catalog == _catalog()
    assert stored_catalogs == [_catalog()]
    assert next(replaced[0].iter_machine_pools())[-1].price_per_instance_hour == 3.21


def test_refresh_capacity_reuses_fresh_stored_catalog(monkeypatch):
    built = []
    replaced: list[ResourceMap] = []

    class FakeResourceMapStore:
        def __init__(self, client, *, user_id):
            pass

        def replace(self, resource_map):
            replaced.append(resource_map)

    class FakeCatalogStore:
        def __init__(self, client):
            pass

        def get(self):
            return mod.CatalogSnapshot(catalog=_catalog(), updated_at=datetime.now(UTC))

    monkeypatch.setattr(mod, "build_catalogs", lambda regions: built.append(regions) or _catalog())
    monkeypatch.setattr(
        mod, "fetch_reservations", lambda region: [_cr("g6e.12xlarge", "us-east-2a", 2)]
    )
    monkeypatch.setattr(mod, "ResourceMapStore", FakeResourceMapStore)
    monkeypatch.setattr(mod, "HardwareCatalogStore", FakeCatalogStore)

    snapshot = mod.refresh_capacity("client", "user_1", ["us-east-2"])

    assert snapshot.catalog == _catalog()
    assert built == []
    assert next(replaced[0].iter_machine_pools())[-1].price_per_instance_hour == 3.21


def test_capacity_refresher_runs_only_when_due(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mod,
        "refresh_capacity",
        lambda client, user_id, regions, **kwargs: (
            calls.append((client, user_id, regions, kwargs))
            or mod.CatalogSnapshot(catalog={}, updated_at=datetime.now(UTC))
        ),
    )
    refresher = CapacityRefresher("client", "user_1", ["us-east-2"], refresh_seconds=10)
    start = datetime(2026, 1, 1, tzinfo=UTC)

    assert refresher.refresh_if_due(now=start) is True
    assert refresher.refresh_if_due(now=start + timedelta(seconds=9)) is False
    assert refresher.refresh_if_due(now=start + timedelta(seconds=10)) is True
    assert len(calls) == 2
    assert calls[0][3] == {"catalog_max_age_seconds": 10}

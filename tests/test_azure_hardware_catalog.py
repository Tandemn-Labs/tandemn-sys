from tandemn_orca.scripts.azure_hardware_catalog import build_catalog


def test_build_catalog_preserves_azure_capabilities_and_shared_schema() -> None:
    sku = {
        "resourceType": "virtualMachines",
        "name": "Standard_NC24ads_A100_v4",
        "family": "standardNCadsA100v4Family",
        "locations": ["eastus"],
        "locationInfo": [{"location": "eastus", "zones": ["1", "2"]}],
        "capabilities": [
            {"name": "GPUs", "value": "1"},
            {"name": "GpuMemoryGB", "value": "80"},
            {"name": "vCPUs", "value": "24"},
            {"name": "MemoryGB", "value": "220"},
            {"name": "RdmaEnabled", "value": "True"},
            {"name": "UnmappedCapability", "value": "kept"},
        ],
    }

    catalog = build_catalog([sku, sku], ["eastus"])
    instance = catalog["instance_types"][0]

    assert instance["vcpu"] == 24
    assert instance["memory_mib"] == 225280
    assert instance["accelerators"][0]["name"] == "A100"
    assert instance["accelerators"][0]["gpu_bandwidth_gbps"] == 2039
    assert instance["network"]["rdma_supported"] is True
    assert catalog["instance_type_count"] == 1
    assert instance["offerings"] == [
        {
            "region": "eastus",
            "zone_name": "1",
            "zone_id": "1",
            "location_type": "availability-zone",
        },
        {
            "region": "eastus",
            "zone_name": "2",
            "zone_id": "2",
            "location_type": "availability-zone",
        },
    ]
    assert catalog["source_skus"][0]["capabilities"][-1] == {
        "name": "UnmappedCapability",
        "value": "kept",
    }

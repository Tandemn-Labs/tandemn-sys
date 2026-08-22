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
            {"name": "NetworkBandwidthMbps", "value": "40000"},
            {"name": "UnmappedCapability", "value": "kept"},
        ],
    }

    prices = {
        ("Standard_NC24ads_A100_v4", "eastus"): {
            "AcceleratorName": "A100",
            "Price": "3.21",
            "SpotPrice": "1.23",
        }
    }
    catalog = build_catalog([sku, sku], ["eastus"], prices)
    instance = catalog["instance_types"][0]

    assert instance["vcpu"] == 24
    assert instance["memory_mib"] == 225280
    assert instance["accelerators"][0]["name"] == "A100"
    assert instance["accelerators"][0]["gpu_bandwidth_gbps"] == 2039
    assert instance["network"]["rdma_supported"] is True
    assert instance["network"]["network_performance"] == "40000"
    assert catalog["instance_type_count"] == 1
    assert instance["offerings"] == [
        {
            "region": "eastus",
            "zone_name": "1",
            "zone_id": "1",
            "location_type": "availability-zone",
            "on_demand_usd_per_hour": 3.21,
            "capacity_reservation_usd_per_hour": 3.21,
            "spot_usd_per_hour": 1.23,
        },
        {
            "region": "eastus",
            "zone_name": "2",
            "zone_id": "2",
            "location_type": "availability-zone",
            "on_demand_usd_per_hour": 3.21,
            "capacity_reservation_usd_per_hour": 3.21,
            "spot_usd_per_hour": 1.23,
        },
    ]
    assert catalog["source_skus"][0]["capabilities"][-1] == {
        "name": "UnmappedCapability",
        "value": "kept",
    }


def test_build_catalog_uses_skypilot_amd_accelerator_name() -> None:
    sku = {
        "resourceType": "virtualMachines",
        "name": "Standard_NV8as_v4",
        "locations": ["eastus"],
        "locationInfo": [{"location": "eastus", "zones": []}],
        "capabilities": [{"name": "GPUs", "value": "1"}, {"name": "GpuMemoryGB", "value": "16"}],
    }

    catalog = build_catalog(
        [sku], ["eastus"], {("Standard_NV8as_v4", "eastus"): {"AcceleratorName": "MI25"}}
    )
    gpu = catalog["instance_types"][0]["accelerators"][0]

    assert gpu["vendor"] == "amd"
    assert gpu["name"] == "MI25"
    assert gpu["k8s_resource_name"] == "amd.com/gpu"


def test_build_catalog_normalizes_mi300x_with_amd_facts() -> None:
    sku = {
        "resourceType": "virtualMachines",
        "name": "Standard_ND96isr_MI300X_v5",
        "locations": ["westus"],
        "locationInfo": [{"location": "westus", "zones": []}],
        "capabilities": [{"name": "GPUs", "value": "8"}],
    }

    instance = build_catalog(
        [sku],
        ["westus"],
        {("Standard_ND96isr_MI300X_v5", "westus"): {"Price": "62.853"}},
    )["instance_types"][0]
    gpu = instance["accelerators"][0]

    assert gpu["vendor"] == "amd"
    assert gpu["name"] == "MI300X"
    assert gpu["canonical_gpu_name"] == "MI300X"
    assert gpu["memory_mib_each"] == 192 * 1024
    assert gpu["gpu_bandwidth_gbps"] == 5300
    assert gpu["infinity_fabric_bandwidth_gbps"] == 1024
    assert instance["offerings"][0]["zone_name"] == "default"
    assert instance["offerings"][0]["on_demand_usd_per_hour"] == 62.853


def test_build_catalog_keeps_sku_locations_missing_location_info() -> None:
    sku = {
        "resourceType": "virtualMachines",
        "name": "Standard_ND96isr_MI300X_v5",
        "locations": ["westus", "francecentral"],
        "locationInfo": [{"location": "westus", "zones": []}],
        "capabilities": [{"name": "GPUs", "value": "8"}],
    }

    offerings = build_catalog(
        [sku],
        ["westus", "francecentral"],
        {("Standard_ND96isr_MI300X_v5", "westus"): {"Price": "62.853"}},
    )["instance_types"][0]["offerings"]

    assert {offering["region"] for offering in offerings} == {"westus", "francecentral"}


def test_build_catalog_uses_verified_a100_sku_facts_when_azure_omits_memory() -> None:
    skus = [
        {
            "resourceType": "virtualMachines",
            "name": "Standard_ND96amsr_A100_v4",
            "locations": ["westus2"],
            "locationInfo": [{"location": "westus2", "zones": ["1"]}],
            "capabilities": [{"name": "GPUs", "value": "8"}],
        },
        {
            "resourceType": "virtualMachines",
            "name": "Standard_NC96ads_A100_v4",
            "locations": ["westus3"],
            "locationInfo": [{"location": "westus3", "zones": ["1"]}],
            "capabilities": [{"name": "GPUs", "value": "4"}],
        },
        {
            "resourceType": "virtualMachines",
            "name": "Standard_ND96asr_v4",
            "locations": ["westus2"],
            "locationInfo": [{"location": "westus2", "zones": ["1"]}],
            "capabilities": [{"name": "GPUs", "value": "8"}],
        },
    ]
    prices = {
        ("Standard_ND96amsr_A100_v4", "westus2"): {"AcceleratorName": "A100-80GB"},
        ("Standard_NC96ads_A100_v4", "westus3"): {"AcceleratorName": "A100-80GB"},
        ("Standard_ND96asr_v4", "westus2"): {"AcceleratorName": "A100"},
    }
    instances = {
        instance["instance_type"]: instance
        for instance in build_catalog(skus, ["westus2", "westus3"], prices)["instance_types"]
    }

    nd = instances["Standard_ND96amsr_A100_v4"]
    nd_gpu = nd["accelerators"][0]
    assert nd_gpu["memory_mib_each"] == 80 * 1024
    assert nd_gpu["memory_mib_total"] == 8 * 80 * 1024
    assert nd_gpu["canonical_gpu_name"] == "A100-80GB"
    assert nd_gpu["nvlink_bandwidth_gbps"] == 600
    assert nd["network"]["network_cards"] == [{"peak_bandwidth_gbps": 1600}]

    nc = instances["Standard_NC96ads_A100_v4"]
    nc_gpu = nc["accelerators"][0]
    assert nc_gpu["memory_mib_each"] == 80 * 1024
    assert nc_gpu["canonical_gpu_name"] == "A100-80GB"
    assert nc_gpu["nvlink_bandwidth_gbps"] == 0
    assert nc["network"]["network_cards"] == [{"peak_bandwidth_gbps": 200}]

    legacy_nd = instances["Standard_ND96asr_v4"]["accelerators"][0]
    assert legacy_nd["memory_mib_each"] == 40 * 1024
    assert legacy_nd["canonical_gpu_name"] == "A100-40GB"

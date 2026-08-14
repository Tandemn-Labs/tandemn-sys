"""Import Azure GPU VM SKUs into Tandemn Store's hardware-catalog schema.

Uses the signed-in Azure CLI account because the Resource SKUs API is scoped to
the subscription and exposes the exact SKU availability for that subscription.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import delete
from tandemn_system_data.clients import HardwareCatalogStore, PostgresClient
from tandemn_system_data.db.orm import HardwareCatalogRow

from tandemn_orca.scripts.aws_hardware_catalog import (
    UNKNOWN_GPU_SPEC,
    accelerator_resource_name,
    gpu_spec,
)

JsonDict = dict[str, Any]
AZURE_CATALOG_KEY = "azure-accelerated-hardware-v1"
API_VERSION = "2021-07-01"


def az(*args: str) -> str:
    """Run one Azure CLI query without adding an Azure SDK dependency."""
    return subprocess.check_output(["az", *args], text=True).strip()


def azure_skus() -> list[JsonDict]:
    """Fetch every VM SKU visible to the signed-in subscription."""
    subscription_id = az("account", "show", "--query", "id", "-o", "tsv")
    token = az(
        "account",
        "get-access-token",
        "--resource",
        "https://management.azure.com/",
        "--query",
        "accessToken",
        "-o",
        "tsv",
    )
    url = (
        f"https://management.azure.com/subscriptions/{subscription_id}"
        f"/providers/Microsoft.Compute/skus?api-version={API_VERSION}"
    )
    skus: list[JsonDict] = []
    while url:
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
        skus.extend(payload.get("value", []))
        url = payload.get("nextLink")
    return skus


def capabilities(sku: JsonDict) -> JsonDict:
    """Keep Azure's complete capability map alongside normalized fields."""
    return {
        str(capability["name"]): capability.get("value")
        for capability in sku.get("capabilities", [])
        if capability.get("name")
    }


def as_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def as_bool(value: object) -> bool:
    return str(value).lower() == "true"


def gpu_name(sku_name: str) -> str:
    """Infer the physical GPU where Azure's Resource SKUs API omits it."""
    name = sku_name.lower()
    for fragment, gpu in (
        ("gb200", "B200"),
        ("h200", "H200"),
        ("h100", "H100"),
        ("a100", "A100"),
        ("a10", "A10G"),
        ("l40s", "L40S"),
        ("l4", "L4"),
        ("t4", "T4"),
        ("v100", "V100"),
        ("m60", "M60"),
    ):
        if fragment in name:
            return gpu
    return sku_name


def normalize_gpu(sku: JsonDict, caps: JsonDict) -> JsonDict | None:
    count = as_int(caps.get("GPUs"))
    if count is None or count <= 0:
        return None
    name = gpu_name(str(sku["name"]))
    memory_gb = as_int(caps.get("GpuMemoryGB"))
    memory_mib = memory_gb * 1024 if memory_gb is not None else None
    if name == "A100" and memory_mib is None:
        memory_mib = 80 * 1024
    return {
        "kind": "gpu",
        "vendor": "nvidia" if name != str(sku["name"]) else None,
        "name": name,
        "count": count,
        "memory_mib_each": memory_mib,
        "memory_mib_total": memory_mib * count if memory_mib is not None else None,
        "k8s_resource_name": accelerator_resource_name("gpu", "nvidia"),
        **(gpu_spec(name, memory_mib) if name != str(sku["name"]) else UNKNOWN_GPU_SPEC),
    }


def normalize_sku(sku: JsonDict, regions: set[str]) -> JsonDict | None:
    if sku.get("resourceType") != "virtualMachines":
        return None
    caps = capabilities(sku)
    gpu = normalize_gpu(sku, caps)
    if gpu is None:
        return None
    locations = {
        str(item["location"]): sorted({str(zone) for zone in item.get("zones", [])})
        for item in sku.get("locationInfo", [])
        if item.get("location") in regions
    }
    network = {
        "network_performance": None,
        "max_enis": as_int(caps.get("MaxNetworkInterfaces")),
        "max_network_cards": None,
        "efa_supported": False,
        "efa_max_interfaces": 0,
        "ena_support": None,
        "ena_srd_supported": False,
        "encryption_in_transit_supported": False,
        "network_cards": [],
        "accelerated_networking_supported": as_bool(caps.get("AcceleratedNetworkingEnabled")),
        "rdma_supported": as_bool(caps.get("RdmaEnabled")),
    }
    storage = {
        "instance_storage_supported": False,
        "instance_storage_total_gb": 0,
        "instance_storage_nvme_support": None,
        "instance_storage_encryption_support": None,
        "instance_storage_disks": [],
        "ebs_optimized_support": None,
        "ebs_nvme_support": None,
        "ebs_bandwidth_mbps": None,
        "ebs_iops": None,
        "ebs_throughput_mbps": None,
        "max_data_disks": as_int(caps.get("MaxDataDiskCount")),
        "os_disk_size_mib": as_int(caps.get("OSVhdSizeMB")),
        "premium_io_supported": as_bool(caps.get("PremiumIO")),
    }
    name = str(sku["name"])
    return {
        "cloud": "azure",
        "instance_type": name,
        "family": sku.get("family"),
        "bare_metal": False,
        "architecture": [caps["CpuArchitectureType"]] if caps.get("CpuArchitectureType") else [],
        "cpu_manufacturer": None,
        "vcpu": as_int(caps.get("vCPUs")),
        "memory_mib": round(float(caps["MemoryGB"]) * 1024) if caps.get("MemoryGB") else None,
        "accelerators": [gpu],
        "network": network,
        "storage": storage,
        "supported_usage_classes": ["on-demand", "spot"]
        if as_bool(caps.get("LowPriorityCapable"))
        else ["on-demand"],
        "supported_boot_modes": [],
        "launch_capabilities": {
            "supports_karpenter": False,
            "supports_efa_nodeclass": False,
            "recommended_efa_interfaces": 0,
            "efa_network_interface_profile": None,
            "requires_nvidia_device_plugin": True,
            "requires_amd_device_plugin": False,
            "requires_neuron_device_plugin": False,
            "k8s_accelerator_resource_names": ["nvidia.com/gpu"],
            "instance_store_policy": None,
        },
        "offerings": [
            {
                "region": region,
                "zone_name": zone,
                "zone_id": zone,
                "location_type": "availability-zone" if zone else "region",
            }
            for region, zones in locations.items()
            for zone in zones or [None]
        ],
        "pricing": {"on_demand_usd_per_hour": None, "capacity_reservation_usd_per_hour": None},
    }


def build_catalog(skus: list[JsonDict], regions: list[str] | None = None) -> JsonDict:
    """Normalize one unique record per Azure VM SKU."""
    selected_regions = set(regions or (str(location) for sku in skus for location in sku.get("locations", [])))
    source_skus = {
        str(sku["name"]): sku
        for sku in skus
        if selected_regions.intersection(str(location) for location in sku.get("locations", []))
    }
    normalized = [
        (sku, instance)
        for sku in source_skus.values()
        if (instance := normalize_sku(sku, selected_regions)) is not None
    ]
    normalized.sort(key=lambda pair: str(pair[1]["instance_type"]))
    return {
        "catalog_version": AZURE_CATALOG_KEY,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "cloud": "azure",
        "regions": sorted(selected_regions),
        "instance_type_count": len(normalized),
        "instance_types": [instance for _, instance in normalized],
        "source_skus": [sku for sku, _ in normalized],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", help="Comma-separated Azure regions; defaults to all visible regions")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--store", action="store_true", help="Write the catalog to Tandemn Store")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    skus = azure_skus()
    regions = (
        [region.strip() for region in args.regions.split(",")]
        if args.regions
        else None
    )
    catalog = build_catalog(skus, regions)
    rendered = json.dumps(catalog, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    elif not args.store:
        print(rendered)
    if args.store:
        client = PostgresClient()
        with client.begin() as session:
            session.execute(
                delete(HardwareCatalogRow).where(
                    HardwareCatalogRow.catalog_key.like(f"{AZURE_CATALOG_KEY}-%")
                )
            )
        HardwareCatalogStore(client).replace(catalog, AZURE_CATALOG_KEY)
        print(f"stored {catalog['instance_type_count']} unique Azure GPU SKUs")


if __name__ == "__main__":
    main()

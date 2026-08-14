"""Static RunPod MI300X catalog from SkyPilot v8 and AMD MI300X specifications."""

from __future__ import annotations

from typing import Any

RUNPOD_CATALOG_KEY = "runpod-accelerated-hardware-v1"
_SKYPILOT_VMS = "https://raw.githubusercontent.com/skypilot-org/skypilot-catalog/master/catalogs/v8/runpod/vms.csv"
_AMD_MI300X = "https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html"


def _instance(region: str, zones: list[str]) -> dict[str, Any]:
    return {
        "cloud": "runpod",
        "instance_type": "8x_MI300X_SECURE",
        "family": "MI300X_SECURE",
        "bare_metal": False,
        "architecture": [],
        "cpu_manufacturer": None,
        "vcpu": 192,
        "memory_mib": 2264 * 1024,
        "accelerators": [{
            "kind": "gpu",
            "vendor": "amd",
            "name": "MI300",
            "canonical_gpu_name": "MI300X",
            "count": 8,
            "memory_mib_each": 192 * 1024,
            "memory_mib_total": 8 * 192 * 1024,
            "gpu_generation": "CDNA3",
            "gpu_bandwidth_gbps": 5300,
            "gpu_tflops_fp16": 1300,
            "infinity_fabric_bandwidth_gbps": 8 * 128,
            "pcie_bandwidth_gbps": 128,
            "gpu_watts": 750,
            "k8s_resource_name": "amd.com/gpu",
        }],
        "network": {"network_cards": [], "rdma_supported": None},
        "storage": {"instance_storage_supported": None},
        "supported_usage_classes": ["on-demand"],
        "supported_boot_modes": [],
        "launch_capabilities": {
            "supports_karpenter": False,
            "supports_efa_nodeclass": False,
            "requires_nvidia_device_plugin": False,
            "requires_amd_device_plugin": True,
            "k8s_accelerator_resource_names": ["amd.com/gpu"],
        },
        "offerings": [
            {"region": region, "zone_name": zone, "zone_id": zone, "location_type": "availability-zone"}
            for zone in zones
        ],
        "pricing": {"on_demand_usd_per_hour": 19.12, "capacity_reservation_usd_per_hour": None},
        "source": {"catalog": _SKYPILOT_VMS, "gpu_spec": _AMD_MI300X},
    }


def build_catalog() -> dict[str, Any]:
    """Return every benchmarked RunPod MI300X offering with sourced facts."""
    return {
        "catalog_version": RUNPOD_CATALOG_KEY,
        "cloud": "runpod",
        "schema_note": "SkyPilot v8 supplies instance, price, CPU, RAM, GPU count, memory, and zones; RunPod NIC, RDMA, storage, CPU architecture, and verified host topology are unavailable. GPU bandwidth and Infinity Fabric facts come from AMD MI300X specifications.",
        "regions": [
            {"cloud": "runpod", "region": "US", "instance_types": [_instance("US", ["US-CA-1"])]},
            {"cloud": "runpod", "region": "AU", "instance_types": [_instance("AU", ["OC-AU-1"])]},
        ],
    }

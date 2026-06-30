from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

JsonDict = dict[str, Any]
DEFAULT_US_REGIONS = ("us-east-1", "us-east-2", "us-west-1", "us-west-2")
PRICING_REGION = "us-east-1"
_AWS_CONFIG = Config(connect_timeout=5, read_timeout=20, retries={"max_attempts": 2})


def warn_skipped_aws_call(action: str, error: Exception) -> None:
    print(f"warning: skipped {action}: {error}", file=sys.stderr)


def fetch_instance_types(region: str) -> list[JsonDict]:
    client = boto3.client("ec2", region_name=region, config=_AWS_CONFIG)
    return [
        item
        for page in client.get_paginator("describe_instance_types").paginate()
        for item in cast(list[JsonDict], page.get("InstanceTypes", []))
    ]


def fetch_instance_type_offerings(region: str) -> list[JsonDict]:
    client = boto3.client("ec2", region_name=region, config=_AWS_CONFIG)
    return [
        item
        for page in client.get_paginator("describe_instance_type_offerings").paginate(
            LocationType="availability-zone"
        )
        for item in cast(list[JsonDict], page.get("InstanceTypeOfferings", []))
    ]


def fetch_availability_zones(region: str) -> dict[str, str]:
    client = boto3.client("ec2", region_name=region, config=_AWS_CONFIG)
    payload = client.describe_availability_zones()
    zones = cast(list[JsonDict], payload.get("AvailabilityZones", []))
    return {str(zone["ZoneName"]): str(zone["ZoneId"]) for zone in zones}


def parse_usd(value: object) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def on_demand_hourly_usd(product: JsonDict) -> float | None:
    terms = cast(JsonDict, product.get("terms", {}))
    for term in cast(JsonDict, terms.get("OnDemand", {})).values():
        price_dimensions = cast(JsonDict, cast(JsonDict, term).get("priceDimensions", {}))
        for dimension in price_dimensions.values():
            price_dimension = cast(JsonDict, dimension)
            if price_dimension.get("unit") != "Hrs":
                continue
            price_per_unit = cast(JsonDict, price_dimension.get("pricePerUnit", {}))
            return parse_usd(price_per_unit.get("USD"))
    return None


def fetch_on_demand_prices(region: str, instance_types: list[str]) -> dict[str, float]:
    wanted = set(instance_types)
    if not wanted:
        return {}

    try:
        pages = boto3.client("pricing", region_name=PRICING_REGION, config=_AWS_CONFIG).get_paginator(
            "get_products"
        ).paginate(
            ServiceCode="AmazonEC2",
            Filters=[
                {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region},
                {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
                {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
                {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
                {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
                {"Type": "TERM_MATCH", "Field": "licenseModel", "Value": "No License required"},
            ],
        )
    except (BotoCoreError, ClientError) as error:
        warn_skipped_aws_call(f"on-demand prices for {region}", error)
        return {}
    prices = {}

    for page in pages:
        for raw_product in cast(list[str], page.get("PriceList", [])):
            product = cast(JsonDict, json.loads(raw_product))
            product_info = cast(JsonDict, product.get("product", {}))
            attributes = cast(JsonDict, product_info.get("attributes", {}))
            instance_type = str(attributes.get("instanceType", ""))
            if instance_type not in wanted:
                continue

            price = on_demand_hourly_usd(product)
            if price is not None:
                prices[instance_type] = price

    return prices


def fetch_spot_prices(region: str, instance_types: list[str]) -> dict[str, dict[str, float]]:
    if not instance_types:
        return {}

    try:
        pages = boto3.client("ec2", region_name=region, config=_AWS_CONFIG).get_paginator(
            "describe_spot_price_history"
        ).paginate(ProductDescriptions=["Linux/UNIX"], InstanceTypes=instance_types)
    except (BotoCoreError, ClientError) as error:
        warn_skipped_aws_call(f"spot prices for {region}", error)
        return {}
    latest: dict[tuple[str, str], tuple[str, float]] = {}

    for page in pages:
        for item in cast(list[JsonDict], page.get("SpotPriceHistory", [])):
            price = parse_usd(item.get("SpotPrice"))
            if price is None:
                continue

            key = (str(item["InstanceType"]), str(item["AvailabilityZone"]))
            timestamp = str(item.get("Timestamp", ""))
            previous = latest.get(key)
            if previous is None or timestamp > previous[0]:
                latest[key] = (timestamp, price)

    prices: dict[str, dict[str, float]] = {}
    for (instance_type, zone_name), (_, price) in latest.items():
        prices.setdefault(instance_type, {})[zone_name] = price

    return prices


def normalize_vendor(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.lower()
    if normalized == "amazon web services":
        return "aws"
    return normalized


def accelerator_resource_name(kind: str, vendor: str | None) -> str | None:
    if kind == "gpu" and vendor == "nvidia":
        return "nvidia.com/gpu"
    if kind == "gpu" and vendor == "amd":
        # TODO(amd): verify the exact device plugin resource name before Orca
        # renders pod requests for AMD-backed workloads.
        return "amd.com/gpu"
    if kind == "neuron":
        # TODO(neuron): verify whether workloads should request neuron devices,
        # neuron cores, or both for each Neuron runtime integration.
        return "aws.amazon.com/neuron"
    if kind == "fpga":
        # TODO(fpga): verify the Kubernetes resource exposed by the FPGA plugin
        # before using FPGA-backed instances in ResourceMap planning.
        return "aws.amazon.com/fpga"
    return None


def normalize_gpu_accelerators(instance_type: JsonDict) -> list[JsonDict]:
    gpu_info = cast(JsonDict, instance_type.get("GpuInfo", {}))
    total_memory = gpu_info.get("TotalGpuMemoryInMiB")
    accelerators = []

    for gpu in cast(list[JsonDict], gpu_info.get("Gpus", [])):
        vendor = normalize_vendor(cast(str | None, gpu.get("Manufacturer")))
        memory_info = cast(JsonDict, gpu.get("MemoryInfo", {}))
        count = cast(int, gpu.get("Count", 0))
        memory_each = cast(int | None, memory_info.get("SizeInMiB"))
        accelerators.append(
            {
                "kind": "gpu",
                "vendor": vendor,
                "name": gpu.get("Name"),
                "count": count,
                "memory_mib_each": memory_each,
                "memory_mib_total": total_memory,
                "k8s_resource_name": accelerator_resource_name("gpu", vendor),
            }
        )

    return accelerators


def normalize_neuron_accelerators(instance_type: JsonDict) -> list[JsonDict]:
    neuron_info = cast(JsonDict, instance_type.get("NeuronInfo", {}))
    total_memory = neuron_info.get("TotalNeuronDeviceMemoryInMiB")
    accelerators = []

    for device in cast(list[JsonDict], neuron_info.get("NeuronDevices", [])):
        core_info = cast(JsonDict, device.get("CoreInfo", {}))
        memory_info = cast(JsonDict, device.get("MemoryInfo", {}))
        accelerators.append(
            {
                "kind": "neuron",
                "vendor": "aws",
                "name": device.get("Name"),
                "count": device.get("Count"),
                "cores_per_device": core_info.get("Count"),
                "core_version": core_info.get("Version"),
                "memory_mib_each": memory_info.get("SizeInMiB"),
                "memory_mib_total": total_memory,
                "k8s_resource_name": accelerator_resource_name("neuron", "aws"),
            }
        )

    return accelerators


def normalize_inference_accelerators(instance_type: JsonDict) -> list[JsonDict]:
    if "NeuronInfo" in instance_type:
        return []

    inference_info = cast(JsonDict, instance_type.get("InferenceAcceleratorInfo", {}))
    total_memory = inference_info.get("TotalInferenceMemoryInMiB")
    accelerators = []

    for accelerator in cast(list[JsonDict], inference_info.get("Accelerators", [])):
        vendor = normalize_vendor(cast(str | None, accelerator.get("Manufacturer")))
        memory_info = cast(JsonDict, accelerator.get("MemoryInfo", {}))
        accelerators.append(
            {
                "kind": "inference_accelerator",
                "vendor": vendor,
                "name": accelerator.get("Name"),
                "count": accelerator.get("Count"),
                "memory_mib_each": memory_info.get("SizeInMiB"),
                "memory_mib_total": total_memory,
                "k8s_resource_name": None,
            }
        )

    return accelerators


def normalize_fpga_accelerators(instance_type: JsonDict) -> list[JsonDict]:
    fpga_info = cast(JsonDict, instance_type.get("FpgaInfo", {}))
    total_memory = fpga_info.get("TotalFpgaMemoryInMiB")
    accelerators = []

    for fpga in cast(list[JsonDict], fpga_info.get("Fpgas", [])):
        vendor = normalize_vendor(cast(str | None, fpga.get("Manufacturer")))
        memory_info = cast(JsonDict, fpga.get("MemoryInfo", {}))
        accelerators.append(
            {
                "kind": "fpga",
                "vendor": vendor,
                "name": fpga.get("Name"),
                "count": fpga.get("Count"),
                "memory_mib_each": memory_info.get("SizeInMiB"),
                "memory_mib_total": total_memory,
                "k8s_resource_name": accelerator_resource_name("fpga", vendor),
            }
        )

    return accelerators


def normalize_accelerators(instance_type: JsonDict) -> list[JsonDict]:
    return [
        *normalize_gpu_accelerators(instance_type),
        *normalize_neuron_accelerators(instance_type),
        *normalize_inference_accelerators(instance_type),
        *normalize_fpga_accelerators(instance_type),
    ]


def normalize_network(instance_type: JsonDict) -> JsonDict:
    network_info = cast(JsonDict, instance_type.get("NetworkInfo", {}))
    efa_info = cast(JsonDict, network_info.get("EfaInfo", {}))
    network_cards = []

    for card in cast(list[JsonDict], network_info.get("NetworkCards", [])):
        network_cards.append(
            {
                "index": card.get("NetworkCardIndex"),
                "network_performance": card.get("NetworkPerformance"),
                "max_enis": card.get("MaximumNetworkInterfaces"),
                "baseline_bandwidth_gbps": card.get("BaselineBandwidthInGbps"),
                "peak_bandwidth_gbps": card.get("PeakBandwidthInGbps"),
            }
        )

    return {
        "network_performance": network_info.get("NetworkPerformance"),
        "max_enis": network_info.get("MaximumNetworkInterfaces"),
        "max_network_cards": network_info.get("MaximumNetworkCards"),
        "efa_supported": network_info.get("EfaSupported", False),
        "efa_max_interfaces": efa_info.get("MaximumEfaInterfaces", 0),
        "ena_support": network_info.get("EnaSupport"),
        "ena_srd_supported": network_info.get("EnaSrdSupported", False),
        "encryption_in_transit_supported": network_info.get("EncryptionInTransitSupported", False),
        "network_cards": network_cards,
    }


def normalize_storage(instance_type: JsonDict) -> JsonDict:
    instance_storage_info = cast(JsonDict, instance_type.get("InstanceStorageInfo", {}))
    ebs_info = cast(JsonDict, instance_type.get("EbsInfo", {}))
    ebs_optimized_info = cast(JsonDict, ebs_info.get("EbsOptimizedInfo", {}))
    disks = []

    for disk in cast(list[JsonDict], instance_storage_info.get("Disks", [])):
        disks.append(
            {
                "size_gb": disk.get("SizeInGB"),
                "count": disk.get("Count"),
                "type": disk.get("Type"),
            }
        )

    return {
        "instance_storage_supported": instance_type.get("InstanceStorageSupported", False),
        "instance_storage_total_gb": instance_storage_info.get("TotalSizeInGB", 0),
        "instance_storage_nvme_support": instance_storage_info.get("NvmeSupport"),
        "instance_storage_encryption_support": instance_storage_info.get("EncryptionSupport"),
        "instance_storage_disks": disks,
        "ebs_optimized_support": ebs_info.get("EbsOptimizedSupport"),
        "ebs_nvme_support": ebs_info.get("NvmeSupport"),
        "ebs_bandwidth_mbps": ebs_optimized_info.get("BaselineBandwidthInMbps"),
        "ebs_iops": ebs_optimized_info.get("BaselineIops"),
        "ebs_throughput_mbps": ebs_optimized_info.get("BaselineThroughputInMBps"),
    }


def derive_efa_profile(network: JsonDict) -> str | None:
    if not network["efa_supported"]:
        return None
    if cast(int, network["efa_max_interfaces"]) <= 1:
        return "single-interface"
    return "efa-only-per-network-card"


def derive_launch_capabilities(
    accelerators: list[JsonDict], network: JsonDict, storage: JsonDict
) -> JsonDict:
    resource_names = sorted(
        {
            str(accelerator["k8s_resource_name"])
            for accelerator in accelerators
            if accelerator.get("k8s_resource_name")
        }
    )
    vendors = {accelerator.get("vendor") for accelerator in accelerators}
    kinds = {accelerator.get("kind") for accelerator in accelerators}

    return {
        "supports_karpenter": True,
        "supports_efa_nodeclass": network["efa_supported"],
        "recommended_efa_interfaces": network["efa_max_interfaces"]
        if network["efa_supported"]
        else 0,
        "efa_network_interface_profile": derive_efa_profile(network),
        "requires_nvidia_device_plugin": "nvidia" in vendors,
        "requires_amd_device_plugin": "amd" in vendors,
        "requires_neuron_device_plugin": "neuron" in kinds,
        "k8s_accelerator_resource_names": resource_names,
        "instance_store_policy": "RAID0" if storage["instance_storage_supported"] else None,
    }


def family_from_instance_type(instance_type: str) -> str:
    return instance_type.split(".", maxsplit=1)[0]


def normalize_instance_type(
    instance_type: JsonDict, offerings_by_type: dict[str, list[JsonDict]]
) -> JsonDict | None:
    accelerators = normalize_accelerators(instance_type)
    if not accelerators:
        return None

    name = str(instance_type["InstanceType"])
    processor_info = cast(JsonDict, instance_type.get("ProcessorInfo", {}))
    vcpu_info = cast(JsonDict, instance_type.get("VCpuInfo", {}))
    memory_info = cast(JsonDict, instance_type.get("MemoryInfo", {}))
    network = normalize_network(instance_type)
    storage = normalize_storage(instance_type)

    return {
        "cloud": "aws",
        "instance_type": name,
        "family": family_from_instance_type(name),
        "bare_metal": instance_type.get("BareMetal", False),
        "architecture": processor_info.get("SupportedArchitectures", []),
        "cpu_manufacturer": processor_info.get("Manufacturer"),
        "vcpu": vcpu_info.get("DefaultVCpus"),
        "memory_mib": memory_info.get("SizeInMiB"),
        "accelerators": accelerators,
        "network": network,
        "storage": storage,
        "supported_usage_classes": instance_type.get("SupportedUsageClasses", []),
        "supported_boot_modes": instance_type.get("SupportedBootModes", []),
        "launch_capabilities": derive_launch_capabilities(accelerators, network, storage),
        "offerings": offerings_by_type.get(name, []),
    }


def group_offerings_by_instance_type(
    offerings: list[JsonDict], zone_ids_by_name: dict[str, str]
) -> dict[str, list[JsonDict]]:
    grouped: dict[str, list[JsonDict]] = {}

    for offering in offerings:
        instance_type = str(offering["InstanceType"])
        zone_name = str(offering["Location"])
        grouped.setdefault(instance_type, []).append(
            {
                "zone_name": zone_name,
                "zone_id": zone_ids_by_name.get(zone_name),
                "location_type": offering.get("LocationType"),
            }
        )

    for instance_offerings in grouped.values():
        instance_offerings.sort(key=lambda item: str(item["zone_name"]))

    return grouped


def add_pricing(instance_types: list[JsonDict], region: str) -> None:
    names = [str(instance_type["instance_type"]) for instance_type in instance_types]
    on_demand_prices = fetch_on_demand_prices(region, names)
    spot_prices = fetch_spot_prices(region, names)

    for instance_type in instance_types:
        name = str(instance_type["instance_type"])
        on_demand_price = on_demand_prices.get(name)
        instance_type["pricing"] = {
            "on_demand_usd_per_hour": on_demand_price,
            "capacity_reservation_usd_per_hour": on_demand_price,
        }

        for offering in cast(list[JsonDict], instance_type.get("offerings", [])):
            zone_name = str(offering["zone_name"])
            offering["spot_usd_per_hour"] = spot_prices.get(name, {}).get(zone_name)


def build_catalog(region: str) -> JsonDict:
    zone_ids_by_name = fetch_availability_zones(region)
    offerings_by_type = group_offerings_by_instance_type(
        fetch_instance_type_offerings(region), zone_ids_by_name
    )
    raw_instance_types = fetch_instance_types(region)
    instance_types = []

    for raw_instance_type in raw_instance_types:
        normalized = normalize_instance_type(raw_instance_type, offerings_by_type)
        if normalized is not None:
            instance_types.append(normalized)

    instance_types.sort(key=lambda item: str(item["instance_type"]))
    add_pricing(instance_types, region)

    return {
        "catalog_version": "aws-accelerated-hardware-v2",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "cloud": "aws",
        "region": region,
        "schema_note": "Hardware facts plus public on-demand and spot prices; Reserved Instances, account discounts, and Capacity Blocks are intentionally excluded.",
        "instance_type_count": len(instance_types),
        "instance_types": instance_types,
    }


def build_catalogs(regions: list[str]) -> JsonDict:
    catalogs = [build_catalog(region) for region in regions]
    return {
        "catalog_version": "aws-accelerated-hardware-v2",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "cloud": "aws",
        "region_count": len(catalogs),
        "instance_type_count": sum(int(catalog["instance_type_count"]) for catalog in catalogs),
        "regions": catalogs,
    }


def parse_regions(value: str) -> list[str]:
    regions = [region.strip() for region in value.split(",") if region.strip()]
    if not regions:
        raise ValueError("At least one AWS region is required")
    return regions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an AWS accelerated hardware catalog.")
    parser.add_argument("--region", help="Single AWS region to inspect; overrides --regions")
    parser.add_argument(
        "--regions",
        default=",".join(DEFAULT_US_REGIONS),
        help="Comma-separated AWS regions to inspect",
    )
    parser.add_argument("--output", help="Path to write the JSON catalog")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    regions = [str(args.region)] if args.region else parse_regions(str(args.regions))
    catalog = build_catalog(regions[0]) if len(regions) == 1 else build_catalogs(regions)
    rendered = json.dumps(catalog, indent=args.indent, sort_keys=True)

    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)

    if len(regions) == 1:
        message = (
            f"wrote {catalog['instance_type_count']} accelerated instance types for {regions[0]}"
        )
    else:
        message = (
            f"wrote {catalog['region_count']} regions / "
            f"{catalog['instance_type_count']} accelerated instance types"
        )
    print(message, file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()

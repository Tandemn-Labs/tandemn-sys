"""Build a ResourceMap from AWS EC2 capacity reservations.

Reads active capacity reservations across one or more regions and maps
each onto the canonical hierarchical ResourceMap (cloud -> region -> zone
-> network_fabric -> machine_pool). These are reservations the account
already holds, so the market is "reserved".

Capacity reservations do not report GPU details, so GPU type / count per
instance come from a static lookup (GPU_INSTANCES). Reservations for
instance types not in that table are skipped (logged), since the
ResourceMap is a GPU capacity view.

Output is the ResourceMap wire shape (``pools_json``: ``market`` +
``clouds``) printed as JSON. No database writes.

Usage:
    uv run python -m tandemn_orca.scripts.resource_map_from_aws \\
        --regions us-east-1 us-east-2 us-west-2

    # Omit --regions to scan every region enabled for the account:
    uv run python -m tandemn_orca.scripts.resource_map_from_aws
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from tandemn_system_data.models.resource_map import (
    Cloud,
    MachinePool,
    NetworkFabric,
    Region,
    ResourceMap,
    Zone,
)

logger = logging.getLogger(__name__)

CLOUD_ID = "aws"
# Capacity reservations carry no fabric concept; group all pools under one.
DEFAULT_FABRIC_ID = "default"
DEFAULT_FABRIC_TYPE = "default"

# Bounded timeouts so a slow or unreachable region (e.g. a flaky opt-in
# region) fails fast instead of stalling a full-fleet scan.
_EC2_CONFIG = Config(connect_timeout=5, read_timeout=10, retries={"max_attempts": 2})

# Capacity reservations don't report GPU info. Map the GPU instance types
# we care about to (gpu_type, gpus_per_instance, gpu_memory_gb). Extend as
# needed; unknown instance types are skipped.
GPU_INSTANCES: dict[str, tuple[str, int, int]] = {
    "p5.48xlarge": ("H100", 8, 80),
    "p5e.48xlarge": ("H200", 8, 141),
    "p4d.24xlarge": ("A100", 8, 40),
    "p4de.24xlarge": ("A100", 8, 80),
    "g6.12xlarge": ("L4", 4, 24),
    "g6.48xlarge": ("L4", 8, 24),
    "g6e.12xlarge": ("L40S", 4, 48),
    "g6e.48xlarge": ("L40S", 8, 48),
    "g5.12xlarge": ("A10G", 4, 24),
    "g5.48xlarge": ("A10G", 8, 24),
}


def region_of_az(az: str) -> str:
    """``us-east-1a`` -> ``us-east-1``. AZ is region + a trailing letter."""
    return az[:-1] if az and az[-1].isalpha() else az


def fetch_all_regions() -> list[str]:
    """Region names enabled for this account (excludes disabled opt-in regions)."""
    client = boto3.client("ec2", region_name="us-east-1")
    resp = client.describe_regions()
    return sorted(r["RegionName"] for r in resp.get("Regions", []))


def fetch_reservations(region: str) -> list[dict[str, Any]]:
    """Active capacity reservations for one region.

    A region that is slow or unreachable is logged and treated as empty so
    one bad region cannot stall a full-fleet scan.
    """
    client = boto3.client("ec2", region_name=region, config=_EC2_CONFIG)
    reservations: list[dict[str, Any]] = []
    paginator = client.get_paginator("describe_capacity_reservations")
    try:
        for page in paginator.paginate(Filters=[{"Name": "state", "Values": ["active"]}]):
            reservations.extend(page.get("CapacityReservations", []))
    except (BotoCoreError, ClientError) as exc:
        logger.warning("skipping region %s: %s", region, exc)
        return []
    return reservations


def build_resource_map(regions: list[str]) -> ResourceMap:
    clouds: dict[str, Cloud] = {}

    # Fetch concurrently (network-bound, one call per region), then merge
    # serially so the tree mutation stays single-threaded and race-free.
    with ThreadPoolExecutor(max_workers=min(16, len(regions) or 1)) as pool:
        reservations_by_region = list(
            zip(regions, pool.map(fetch_reservations, regions), strict=True)
        )

    for region, reservations in reservations_by_region:
        for cr in reservations:
            instance_type = cr["InstanceType"]
            gpu = GPU_INSTANCES.get(instance_type)
            if gpu is None:
                logger.info(
                    "skipping non-GPU / unknown instance type %s (reservation %s)",
                    instance_type,
                    cr.get("CapacityReservationId"),
                )
                continue

            gpu_type, gpus_per_instance, gpu_memory_gb = gpu
            az = cr.get("AvailabilityZone", "")
            zone_id = az or "unknown"
            region_id = region_of_az(az) or region
            total_instances = int(cr.get("TotalInstanceCount", 0))
            if total_instances <= 0:
                continue

            cloud = clouds.setdefault(CLOUD_ID, Cloud())
            region_obj = cloud.regions.setdefault(region_id, Region())
            zone_obj = region_obj.zones.setdefault(zone_id, Zone())
            fabric = zone_obj.network_fabrics.setdefault(
                DEFAULT_FABRIC_ID, NetworkFabric(fabric_type=DEFAULT_FABRIC_TYPE)
            )

            existing = fabric.machine_pools.get(instance_type)
            if existing is None:
                fabric.machine_pools[instance_type] = MachinePool(
                    gpu_type=gpu_type,
                    gpu_memory_gb=gpu_memory_gb,
                    gpus_per_instance=gpus_per_instance,
                    total_instances=total_instances,
                )
            else:
                # Multiple reservations, same instance type + AZ: sum them.
                existing.total_instances += total_instances

    return ResourceMap(market=["reserved"], clouds=clouds)


def resource_map_to_pools_json(resource_map: ResourceMap) -> dict[str, Any]:
    """The ``pools_json`` wire shape: ``market`` + ``clouds``."""
    return {
        "market": list(resource_map.market),
        "clouds": {
            cloud_id: cloud.model_dump(mode="json")
            for cloud_id, cloud in resource_map.clouds.items()
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regions",
        nargs="+",
        default=None,
        help="AWS regions to scan. Omit to scan all account-enabled regions.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    args = parse_args(argv)
    regions = args.regions if args.regions is not None else fetch_all_regions()
    resource_map = build_resource_map(regions)
    print(json.dumps(resource_map_to_pools_json(resource_map), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

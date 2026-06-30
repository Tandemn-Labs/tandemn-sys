"""Build a ResourceMap from AWS EC2 capacity reservations.

Reads active capacity reservations across one or more regions and maps
each onto the canonical hierarchical ResourceMap (cloud -> region -> zone
-> network_fabric -> machine_pool). These are reservations the account
already holds, so the market is "reserved".

Capacity reservations do not report GPU details, so GPU type / count per
instance come from the AWS hardware catalog. Reservations missing from the
catalog are skipped (logged), since the ResourceMap is a GPU capacity view.

CLI output is the ResourceMap wire shape (``pools_json``: ``market`` +
``clouds``) printed as JSON. Runner refresh writes to Tandemn Store.

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
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from tandemn_system_data.clients import HardwareCatalogStore, PostgresClient, ResourceMapStore
from tandemn_system_data.models.resource_map import (
    Cloud,
    MachinePool,
    NetworkFabric,
    Region,
    ResourceMap,
    Zone,
)

from tandemn_orca.scripts.aws_hardware_catalog import build_catalogs

logger = logging.getLogger(__name__)

CLOUD_ID = "aws"
# Capacity reservations carry no fabric concept; group all pools under one.
DEFAULT_FABRIC_ID = "default"
DEFAULT_FABRIC_TYPE = "default"

# Bounded timeouts so a slow or unreachable region (e.g. a flaky opt-in
# region) fails fast instead of stalling a full-fleet scan.
_EC2_CONFIG = Config(connect_timeout=5, read_timeout=10, retries={"max_attempts": 2})

@dataclass
class CatalogSnapshot:
    catalog: dict[str, Any]
    updated_at: datetime


def parse_region_csv(value: str | None) -> list[str]:
    regions = [region.strip() for region in (value or "").split(",") if region.strip()]
    if not regions:
        raise ValueError("at least one AWS region is required")
    return regions


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


def build_resource_map(regions: list[str], catalog: dict[str, Any]) -> ResourceMap:
    hardware_by_region_type = catalog_instance_index(catalog)
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
            hardware = hardware_by_region_type.get((region, instance_type))
            gpu = gpu_specs_from_catalog(hardware)
            if gpu is None:
                logger.info(
                    "skipping non-GPU / uncataloged instance type %s (reservation %s)",
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
                    instance_family=hardware.get("family") if hardware else None,
                    gpu_type=gpu_type,
                    gpu_memory_gb=gpu_memory_gb,
                    gpus_per_instance=gpus_per_instance,
                    total_instances=total_instances,
                    price_per_instance_hour=price_from_catalog(hardware),
                )
            else:
                # Multiple reservations, same instance type + AZ: sum them.
                existing.total_instances += total_instances

    return ResourceMap(market=["reserved"], clouds=clouds)


def catalog_instance_index(catalog: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    regions = catalog.get("regions") or [catalog]
    return {
        (str(region_catalog.get("region")), str(instance["instance_type"])): instance
        for region_catalog in regions
        for instance in region_catalog.get("instance_types", [])
    }


def gpu_specs_from_catalog(hardware: dict[str, Any] | None) -> tuple[str, int, int | None] | None:
    if hardware is None:
        return None
    for accelerator in hardware.get("accelerators", []):
        if accelerator.get("kind") != "gpu":
            continue
        count = accelerator.get("count")
        if type(count) is not int or count <= 0:
            continue
        memory_mib = accelerator.get("memory_mib_each")
        memory_gb = int(memory_mib) // 1024 if type(memory_mib) is int else None
        return str(accelerator.get("name") or hardware["instance_type"]), count, memory_gb
    return None


def price_from_catalog(hardware: dict[str, Any] | None) -> float | None:
    if hardware is None:
        return None
    price = (hardware.get("pricing") or {}).get("capacity_reservation_usd_per_hour")
    return float(price) if price is not None else None


def resource_map_to_pools_json(resource_map: ResourceMap) -> dict[str, Any]:
    """The ``pools_json`` wire shape: ``market`` + ``clouds``."""
    return {
        "market": list(resource_map.market),
        "clouds": {
            cloud_id: cloud.model_dump(mode="json")
            for cloud_id, cloud in resource_map.clouds.items()
        },
    }


def catalog_snapshot(
    client: PostgresClient,
    regions: list[str],
    max_age_seconds: float,
    now: datetime | None = None,
) -> CatalogSnapshot:
    now = now or datetime.now(UTC)
    store = HardwareCatalogStore(client)
    stored = store.get()
    if stored is not None and (now - stored.updated_at).total_seconds() < max_age_seconds:
        return CatalogSnapshot(catalog=stored.catalog, updated_at=stored.updated_at)

    stored = store.replace(build_catalogs(regions))
    return CatalogSnapshot(catalog=stored.catalog, updated_at=stored.updated_at)


def refresh_capacity(
    client: PostgresClient,
    user_id: str,
    regions: list[str],
    catalog_max_age_seconds: float = 86400,
) -> CatalogSnapshot:
    snapshot = catalog_snapshot(client, regions, catalog_max_age_seconds)
    catalog = snapshot.catalog
    resource_map = build_resource_map(regions, catalog)
    ResourceMapStore(client, user_id=user_id).replace(resource_map)
    return snapshot


class CapacityRefresher:
    def __init__(
        self,
        client: PostgresClient,
        user_id: str,
        regions: list[str],
        refresh_seconds: float = 86400,
    ) -> None:
        self.client = client
        self.user_id = user_id
        self.regions = regions
        self.refresh_seconds = refresh_seconds
        self.last_refresh: datetime | None = None
        self.snapshot: CatalogSnapshot | None = None

    def refresh_if_due(self, *, force: bool = False, now: datetime | None = None) -> bool:
        now = now or datetime.now(UTC)
        if not force and self.last_refresh is not None:
            elapsed = (now - self.last_refresh).total_seconds()
            if elapsed < self.refresh_seconds:
                return False

        self.last_refresh = now
        self.snapshot = refresh_capacity(
            self.client,
            self.user_id,
            self.regions,
            catalog_max_age_seconds=self.refresh_seconds,
        )
        return True


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
    resource_map = build_resource_map(regions, build_catalogs(regions))
    print(json.dumps(resource_map_to_pools_json(resource_map), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

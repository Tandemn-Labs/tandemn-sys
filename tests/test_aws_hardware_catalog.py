from __future__ import annotations

import json

import tandemn_orca.scripts.aws_hardware_catalog as mod


class FakePaginator:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def paginate(self, **kwargs):
        self.calls.append(kwargs)
        return self.pages


class FakeEC2:
    def __init__(self) -> None:
        self.paginators = {
            "describe_instance_types": FakePaginator([{"InstanceTypes": [_g6e()]}]),
            "describe_instance_type_offerings": FakePaginator(
                [
                    {
                        "InstanceTypeOfferings": [
                            {
                                "InstanceType": "g6e.12xlarge",
                                "Location": "us-east-2a",
                                "LocationType": "availability-zone",
                            }
                        ]
                    }
                ]
            ),
            "describe_spot_price_history": FakePaginator(
                [
                    {
                        "SpotPriceHistory": [
                            {
                                "InstanceType": "g6e.12xlarge",
                                "AvailabilityZone": "us-east-2a",
                                "SpotPrice": "1.23",
                                "Timestamp": "2026-01-01T00:00:00Z",
                            }
                        ]
                    }
                ]
            ),
        }

    def get_paginator(self, name):
        return self.paginators[name]

    def describe_availability_zones(self):
        return {"AvailabilityZones": [{"ZoneName": "us-east-2a", "ZoneId": "use2-az1"}]}


class FakePricing:
    def __init__(self) -> None:
        self.paginator = FakePaginator([{"PriceList": [json.dumps(_price_product())]}])

    def get_paginator(self, name):
        assert name == "get_products"
        return self.paginator


def _g6e():
    return {
        "InstanceType": "g6e.12xlarge",
        "ProcessorInfo": {"SupportedArchitectures": ["x86_64"], "Manufacturer": "Intel"},
        "VCpuInfo": {"DefaultVCpus": 48},
        "MemoryInfo": {"SizeInMiB": 393216},
        "GpuInfo": {
            "TotalGpuMemoryInMiB": 196608,
            "Gpus": [
                {
                    "Manufacturer": "NVIDIA",
                    "Name": "L40S",
                    "Count": 4,
                    "MemoryInfo": {"SizeInMiB": 49152},
                }
            ],
        },
        "NetworkInfo": {
            "NetworkPerformance": "100 Gigabit",
            "MaximumNetworkInterfaces": 15,
            "MaximumNetworkCards": 1,
            "EfaSupported": False,
            "EnaSupport": "required",
            "NetworkCards": [{"NetworkCardIndex": 0, "MaximumNetworkInterfaces": 15}],
        },
        "InstanceStorageSupported": True,
        "InstanceStorageInfo": {
            "TotalSizeInGB": 3800,
            "Disks": [{"SizeInGB": 3800, "Count": 1, "Type": "ssd"}],
        },
        "EbsInfo": {"EbsOptimizedInfo": {"BaselineBandwidthInMbps": 10000}},
        "SupportedUsageClasses": ["on-demand", "spot"],
        "SupportedBootModes": ["uefi"],
    }


def _price_product():
    return {
        "product": {"attributes": {"instanceType": "g6e.12xlarge"}},
        "terms": {
            "OnDemand": {
                "term": {
                    "priceDimensions": {"dim": {"unit": "Hrs", "pricePerUnit": {"USD": "3.21"}}}
                }
            }
        },
    }


def test_build_catalog_uses_boto3_clients(monkeypatch):
    ec2 = FakeEC2()
    pricing = FakePricing()

    def client(service, region_name=None, config=None):
        assert config is not None
        return pricing if service == "pricing" else ec2

    monkeypatch.setattr(mod.boto3, "client", client)

    catalog = mod.build_catalog("us-east-2")
    instance = catalog["instance_types"][0]

    assert catalog["region"] == "us-east-2"
    assert instance["instance_type"] == "g6e.12xlarge"
    assert instance["accelerators"][0]["name"] == "L40S"
    assert instance["accelerators"][0]["canonical_gpu_name"] == "L40S"
    assert instance["accelerators"][0]["gpu_tflops_fp16"] == 362.05
    assert instance["pricing"]["capacity_reservation_usd_per_hour"] == 3.21
    assert instance["offerings"][0]["zone_id"] == "use2-az1"
    assert instance["offerings"][0]["spot_usd_per_hour"] == 1.23


def test_gpu_specs_split_a100_memory_and_use_na_for_unknown():
    assert mod.gpu_spec("A100", 40960)["canonical_gpu_name"] == "A100-40GB"
    assert mod.gpu_spec("A100", 81920)["canonical_gpu_name"] == "A100-80GB"
    assert mod.gpu_spec("A100", None)["canonical_gpu_name"] == "NA"
    assert mod.gpu_spec("Mystery GPU", None)["gpu_bandwidth_gbps"] == "NA"

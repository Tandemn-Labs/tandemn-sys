from tandemn_orca.scripts.runpod_hardware_catalog import build_catalog


def test_build_catalog_has_priced_mi300x_offerings() -> None:
    catalog = build_catalog()

    us = catalog["regions"][0]["instance_types"][0]
    assert us["instance_type"] == "8x_MI300X_SECURE"
    assert us["accelerators"][0]["memory_mib_each"] == 192 * 1024
    assert us["accelerators"][0]["gpu_bandwidth_gbps"] == 5300
    assert us["pricing"]["on_demand_usd_per_hour"] == 19.12

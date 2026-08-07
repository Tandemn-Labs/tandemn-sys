"""Resolve a job's user-facing inference endpoint.

Orca renders each job's router config with a deterministic ``listen_port``
(hash of the job ID); the router binds it when started without ``-listen``.
This CLI closes the loop: job ID in, ``http://127.0.0.1:<port>`` out, straight
from the config file — the same source of truth the router reads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from tandemn_orca.dynamo_compiler import router_listen_port

DEFAULT_CONFIG_DIR = "~/.tandemn/router-configs"


def resolve_endpoint(job_id: str, config_dir: Path) -> str:
    """The job's router URL; falls back to the derived port for old configs."""
    path = config_dir / f"{job_id}.json"
    try:
        config = json.loads(path.read_text())
    except FileNotFoundError:
        raise SystemExit(
            f"no router config for {job_id!r} in {config_dir} — is the job placed and Orca running?"
        ) from None
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"unreadable router config {path}: {error}") from None
    port = config.get("listen_port") or router_listen_port(job_id)
    if not isinstance(port, int) or not 0 < port < 65536:
        raise SystemExit(f"router config {path} has invalid listen_port {port!r}")
    return f"http://127.0.0.1:{port}"


def check_ready(endpoint: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{endpoint}/readyz", timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", help="job ID, e.g. job_01KZ...")
    parser.add_argument(
        "--router-config-dir",
        default=os.environ.get("TANDEMN_ROUTER_CONFIG_DIR", DEFAULT_CONFIG_DIR),
        help=f"where Orca writes per-job router configs (default {DEFAULT_CONFIG_DIR})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="probe the router's /readyz and fail if it is not serving",
    )
    args = parser.parse_args()

    endpoint = resolve_endpoint(args.job_id, Path(args.router_config_dir).expanduser())
    print(endpoint)
    if args.check and not check_ready(endpoint):
        print(f"router at {endpoint} is not ready", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

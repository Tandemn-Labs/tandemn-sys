from __future__ import annotations

import threading

from tandemn_orca.tunnels import PortForwardManager, TunnelSpec


class FakeProcess:
    def __init__(self) -> None:
        self.running = True
        self.terminated = False

    def poll(self):
        return None if self.running else 0

    def terminate(self):
        self.terminated = True
        self.running = False

    def wait(self, timeout):
        return 0

    def kill(self):
        self.running = False


def test_port_forward_manager_starts_and_stops_rank_tunnel():
    started = threading.Event()
    calls = []

    def popen(command, **kwargs):
        process = FakeProcess()
        calls.append((command, kwargs, process))
        started.set()
        return process

    manager = PortForwardManager(retry_seconds=0.01, popen=popen)
    spec = TunnelSpec(
        job_id="job_1",
        rank_id="rank_1",
        context="eks-east",
        namespace="default",
        service="dgd-frontend",
        local_port=18042,
    )
    try:
        manager.reconcile("job_1", [spec])
        assert started.wait(timeout=1)
        assert calls[0][0] == [
            "kubectl",
            "--context",
            "eks-east",
            "--namespace",
            "default",
            "port-forward",
            "service/dgd-frontend",
            "18042:8000",
            "--address",
            "127.0.0.1",
        ]

        manager.reconcile("job_1", [])
        assert calls[0][2].terminated
    finally:
        manager.close()

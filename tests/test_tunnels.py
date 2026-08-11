from __future__ import annotations

import threading

import pytest

from tandemn_orca.tunnels import PortForwardManager, RouterProcessManager, TunnelSpec


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


def _spec(job_id: str, rank_id: str, port: int) -> TunnelSpec:
    return TunnelSpec(
        job_id=job_id,
        rank_id=rank_id,
        context=None,
        namespace="default",
        service=f"{rank_id}-frontend",
        local_port=port,
    )


def test_port_forward_manager_rejects_cross_job_port_conflict():
    manager = PortForwardManager(retry_seconds=0.01, popen=lambda *a, **k: FakeProcess())
    try:
        manager.reconcile("job_1", [_spec("job_1", "rank_1", 18042)])

        with pytest.raises(ValueError, match="already assigned to job 'job_1'"):
            manager.reconcile("job_2", [_spec("job_2", "rank_2", 18042)])

        # The same job re-claiming its own port is a no-op, not a conflict.
        manager.reconcile("job_1", [_spec("job_1", "rank_1", 18042)])
        # And the port frees up once the first job releases it.
        manager.reconcile("job_1", [])
        manager.reconcile("job_2", [_spec("job_2", "rank_2", 18042)])
    finally:
        manager.close()


def test_router_process_manager_starts_and_stops_job_router():
    started = threading.Event()
    calls = []

    def popen(command, **kwargs):
        process = FakeProcess()
        calls.append((command, process))
        started.set()
        return process

    manager = RouterProcessManager("/usr/local/bin/tandemn-router", retry_seconds=0.01, popen=popen)
    try:
        manager.reconcile("job_1", "/tmp/router-configs/job_1.json")
        assert started.wait(timeout=1)
        assert calls[0][0] == [
            "/usr/local/bin/tandemn-router",
            "-config",
            "/tmp/router-configs/job_1.json",
        ]

        manager.reconcile("job_1", None)
        assert calls[0][1].terminated
    finally:
        manager.close()

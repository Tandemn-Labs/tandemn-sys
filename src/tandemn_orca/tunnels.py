from __future__ import annotations

import atexit
import logging
import subprocess
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TunnelSpec:
    job_id: str
    rank_id: str
    context: str | None
    namespace: str
    service: str
    local_port: int


@dataclass(frozen=True)
class RouterSpec:
    job_id: str
    binary: str
    config_path: str


class _ProcessSupervisor:
    """Keep desired subprocesses alive on the control-plane laptop.

    Subclasses provide the desired-state key space and the command for one
    spec; the supervisor thread restarts anything that exits.
    """

    def __init__(
        self, *, retry_seconds: float = 2.0, popen: Any = subprocess.Popen, thread_name: str
    ) -> None:
        self.retry_seconds = retry_seconds
        self.popen = popen
        self._desired: dict[Any, Any] = {}
        self._processes: dict[Any, Any] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=thread_name)
        self._thread.start()
        atexit.register(self.close)

    def _command(self, spec: Any) -> list[str]:
        raise NotImplementedError

    def _describe(self, spec: Any) -> str:
        raise NotImplementedError

    def _replace_job(self, job_id: str, desired: dict[Any, Any]) -> None:
        """Swap one job's desired specs; stale keys for that job are stopped."""
        with self._lock:
            stale = [key for key in self._desired if key[0] == job_id and key not in desired]
            for key in stale:
                self._desired.pop(key, None)
                self._stop_process(key)
            self._desired.update(desired)
        self._wake.set()

    def close(self) -> None:
        if self._stopped.is_set():
            return
        self._stopped.set()
        self._wake.set()
        with self._lock:
            for key in list(self._processes):
                self._stop_process(key)
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stopped.is_set():
            with self._lock:
                for key, spec in self._desired.items():
                    process = self._processes.get(key)
                    if process is None or process.poll() is not None:
                        self._processes[key] = self._start(spec)
            self._wake.wait(self.retry_seconds)
            self._wake.clear()

    def _start(self, spec: Any) -> Any:
        logger.info("starting %s", self._describe(spec))
        return self.popen(self._command(spec), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _stop_process(self, key: Any) -> None:
        process = self._processes.pop(key, None)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()


class PortForwardManager(_ProcessSupervisor):
    """Keep desired kubectl port-forwards alive, one per (job, rank)."""

    def __init__(self, *, retry_seconds: float = 2.0, popen: Any = subprocess.Popen) -> None:
        super().__init__(retry_seconds=retry_seconds, popen=popen, thread_name="router-tunnels")

    def reconcile(self, job_id: str, specs: list[TunnelSpec]) -> None:
        desired = {(spec.job_id, spec.rank_id): spec for spec in specs}
        with self._lock:
            # Rank tunnel ports are hashed per rank with no cross-job
            # coordination; refuse a port another job already holds so its
            # router config can never point at this job's DGD.
            claimed = {
                other.local_port: key[0] for key, other in self._desired.items() if key[0] != job_id
            }
            for spec in specs:
                owner = claimed.get(spec.local_port)
                if owner is not None:
                    raise ValueError(
                        f"tunnel port {spec.local_port} for job {job_id!r} "
                        f"is already assigned to job {owner!r}"
                    )
        self._replace_job(job_id, desired)

    def _command(self, spec: TunnelSpec) -> list[str]:
        command = ["kubectl"]
        if spec.context:
            command.extend(["--context", spec.context])
        command.extend(
            [
                "--namespace",
                spec.namespace,
                "port-forward",
                f"service/{spec.service}",
                f"{spec.local_port}:8000",
                "--address",
                "127.0.0.1",
            ]
        )
        return command

    def _describe(self, spec: TunnelSpec) -> str:
        return f"router tunnel for rank {spec.rank_id} on port {spec.local_port}"


class RouterProcessManager(_ProcessSupervisor):
    """Keep one tandemn-router process alive per active job.

    The router binds the listen_port rendered into its config and reads
    TANDEMN_ROUTER_TELEMETRY_TOKEN from the inherited environment.
    """

    def __init__(
        self, binary: str, *, retry_seconds: float = 2.0, popen: Any = subprocess.Popen
    ) -> None:
        self.binary = binary
        super().__init__(retry_seconds=retry_seconds, popen=popen, thread_name="job-routers")

    def reconcile(self, job_id: str, config_path: str | None) -> None:
        desired: dict[Any, Any] = {}
        if config_path is not None:
            desired[(job_id,)] = RouterSpec(
                job_id=job_id, binary=self.binary, config_path=config_path
            )
        self._replace_job(job_id, desired)

    def _command(self, spec: RouterSpec) -> list[str]:
        return [spec.binary, "-config", spec.config_path]

    def _describe(self, spec: RouterSpec) -> str:
        return f"router process for job {spec.job_id}"

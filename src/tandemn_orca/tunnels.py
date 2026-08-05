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


class PortForwardManager:
    """Keep desired kubectl port-forwards alive on the control-plane laptop."""

    def __init__(self, *, retry_seconds: float = 2.0, popen: Any = subprocess.Popen) -> None:
        self.retry_seconds = retry_seconds
        self.popen = popen
        self._desired: dict[tuple[str, str], TunnelSpec] = {}
        self._processes: dict[tuple[str, str], Any] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="router-tunnels")
        self._thread.start()
        atexit.register(self.close)

    def reconcile(self, job_id: str, specs: list[TunnelSpec]) -> None:
        desired = {(spec.job_id, spec.rank_id): spec for spec in specs}
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

    def _start(self, spec: TunnelSpec) -> Any:
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
        logger.info("starting router tunnel for rank %s on port %s", spec.rank_id, spec.local_port)
        return self.popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _stop_process(self, key: tuple[str, str]) -> None:
        process = self._processes.pop(key, None)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()

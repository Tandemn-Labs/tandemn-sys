"""Launcher seam — "make the chain real".

Orca records canonical chain rows in the store, but bringing up the
actual GPU workers (SkyPilot today; Dynamo / td_operator k8s later) is a
swappable implementation. ca and live in Orca, not the store.

A Launcher operates on the canonical ``Chain`` rows Orca has produced: it
brings the workers up (``launch``) and tears them down (``teardown``).
Recording rows and updating status/events stays in Orca; the launcher
only touches infrastructure.
"""

from __future__ import annotations

import logging
from typing import Protocol

from tandemn_system_data.models.chain import Chain

logger = logging.getLogger(__name__)


class Launcher(Protocol):
    """Brings chains up and tears them down on real infrastructure."""

    def launch(self, chains: list[Chain]) -> None:
        """Bring up workers for each chain (gang: all at once)."""
        ...

    def teardown(self, chain_ids: list[str]) -> None:
        """Stop the workers for the given chains."""
        ...


class NoopLauncher:
    """Records intent only — no infrastructure is touched.

    The default in the MVP: Orca persists chain rows but does not yet
    bring up real workers. Swap in a SkyPilot / Dynamo launcher later.
    """

    def launch(self, chains: list[Chain]) -> None:
        for chain in chains:
            logger.info(
                "noop launch: chain %s role=%s shape=%s",
                chain.chain_id,
                chain.role,
                chain.shape_json,
            )

    def teardown(self, chain_ids: list[str]) -> None:
        for chain_id in chain_ids:
            logger.info("noop teardown: chain %s", chain_id)

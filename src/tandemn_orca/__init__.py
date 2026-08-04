"""tandemn_orca — the executor.

Orca polls Koi's plans from the canonical store, applies the per-job
actions (launching ranks, transitioning jobs), and records what
happened. It is the only writer of the job/chain lifecycle; Koi reads.

See tandemn-store DATA_ARCHITECTURE.md and docs/KOI_INTEGRATION.md for
the contract this implements.
"""

__version__ = "0.1.0"

"""Transparent target-weight producers for the ETF Allocation Workbench."""

from strategies.allocation import (
    AllocationResult,
    StrategyPolicy,
    generate_allocation_targets,
    position_cap,
)

__all__ = [
    "AllocationResult",
    "StrategyPolicy",
    "generate_allocation_targets",
    "position_cap",
]

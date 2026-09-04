"""Transparent target-weight producers for the ETF Allocation Workbench."""

from strategies.allocation import (
    AllocationResult,
    StrategyPolicy,
    allocate_inverse_volatility,
    generate_allocation_targets,
    position_cap,
    validate_price_selection,
)
from strategies.forecast import (
    ForecastAllocationResult,
    generate_forecast_allocation_targets,
)

__all__ = [
    "AllocationResult",
    "ForecastAllocationResult",
    "StrategyPolicy",
    "allocate_inverse_volatility",
    "generate_allocation_targets",
    "generate_forecast_allocation_targets",
    "position_cap",
    "validate_price_selection",
]

"""Offline TimesFM research helpers.

This package is imported by the artifact builder and tests.  The public
Streamlit process deliberately does not import it or the optional TimesFM
dependency.
"""

from forecasting.timesfm import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    CONTEXT_SESSIONS,
    LAST_VALUE_MODE,
    MULTIVARIATE_MODE,
    QUANTILE_LEVELS,
    TIMESFM_SOURCE_REVISION,
    UNIVARIATE_MODE,
    ForecastWindow,
    TimesFM3Runner,
    build_forecast_windows,
    last_value_forecast,
    model_policy,
    model_policy_sha256,
)

__all__ = [
    "CHECKPOINT_ID",
    "CHECKPOINT_REVISION",
    "CONTEXT_SESSIONS",
    "LAST_VALUE_MODE",
    "MULTIVARIATE_MODE",
    "QUANTILE_LEVELS",
    "TIMESFM_SOURCE_REVISION",
    "UNIVARIATE_MODE",
    "ForecastWindow",
    "TimesFM3Runner",
    "build_forecast_windows",
    "last_value_forecast",
    "model_policy",
    "model_policy_sha256",
]

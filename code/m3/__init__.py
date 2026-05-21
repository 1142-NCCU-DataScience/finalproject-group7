"""M3 — feature engineering, ring buffer, labels, training-dataset assembly."""

from m3.build_features import (
    FEATURE_COLUMNS,
    NON_NULLABLE_FEATURES,
    STATION_STATIC_CSV,
    build_features,
    load_station_static,
    validate_features,
)
from m3.labels import add_shortage_label
from m3.lag_buffer import LagBuffer

__all__ = [
    "FEATURE_COLUMNS",
    "NON_NULLABLE_FEATURES",
    "STATION_STATIC_CSV",
    "LagBuffer",
    "add_shortage_label",
    "build_features",
    "load_station_static",
    "validate_features",
]

"""
tests/test_feature_parity.py
-----------------------------
Verifies that the offline (build_features) and online (LagBuffer) paths
produce numerically identical feature vectors, and that structural
invariants (column order, NaN leakage) are upheld.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from m3.build_features import FEATURE_COLUMNS, build_features
from m3.lag_buffer import LagBuffer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SNOS = ["500101001", "500101002", "500101003"]
N_TICKS = 10  # time steps per station

TZ = "Asia/Taipei"


def _make_mini_df() -> pd.DataFrame:
    """Create a 10-tick × 3-station mini DataFrame (sorted by sno, timestamp)."""
    rng = np.random.default_rng(42)
    rows = []
    base_ts = pd.Timestamp("2024-01-08 07:00:00", tz=TZ)  # Monday, rush hour

    for sno in SNOS:
        total_capacity = int(rng.integers(10, 30))
        for tick in range(N_TICKS):
            available = int(rng.integers(0, total_capacity + 1))
            shortage_rate = round((total_capacity - available) / total_capacity, 6)
            rows.append(
                {
                    "sno": sno,
                    "timestamp": base_ts + pd.Timedelta(minutes=10 * tick),
                    "shortage_rate": shortage_rate,
                    "available_bikes": available,
                    "total_capacity": total_capacity,
                }
            )

    df = pd.DataFrame(rows).sort_values(["sno", "timestamp"]).reset_index(drop=True)
    return df


def _make_station_static() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sno": SNOS,
            "distance_to_mrt": [312.5, 87.3, 1450.0],
        }
    )


def _make_spatial_lag(df: pd.DataFrame) -> pd.Series:
    """Synthetic spatial_lag_shortage aligned to df's index."""
    rng = np.random.default_rng(7)
    return pd.Series(
        rng.uniform(0.0, 1.0, size=len(df)),
        index=df.index,
        name="spatial_lag_shortage",
    )


# ---------------------------------------------------------------------------
# Test 1: column order
# ---------------------------------------------------------------------------

def test_feature_columns_order():
    """build_features output columns must exactly match FEATURE_COLUMNS."""
    df = _make_mini_df()
    station_static = _make_station_static()
    spatial_lag = _make_spatial_lag(df)

    X = build_features(df, station_static, spatial_lag)

    assert X.columns.tolist() == FEATURE_COLUMNS, (
        f"Expected {FEATURE_COLUMNS}, got {X.columns.tolist()}"
    )


# ---------------------------------------------------------------------------
# Test 2: no leakage — first 6 rows of lag_60min must be NaN per station
# ---------------------------------------------------------------------------

def test_no_leakage_lag60min_first_6_rows():
    """Offline path: lag_60min must be NaN for the first 6 ticks of each station."""
    df = _make_mini_df()
    station_static = _make_station_static()
    spatial_lag = _make_spatial_lag(df)

    X = build_features(df, station_static, spatial_lag)

    # Re-attach sno for grouping (build_features returns only FEATURE_COLUMNS)
    X_with_sno = X.copy()
    X_with_sno["sno"] = df["sno"].values

    for sno in SNOS:
        station_rows = X_with_sno[X_with_sno["sno"] == sno]["lag_60min"].values
        # First 6 ticks (indices 0-5) must be NaN
        nan_rows = station_rows[:6]
        non_nan_rows = station_rows[6:]
        assert np.all(np.isnan(nan_rows)), (
            f"sno={sno}: expected NaN in first 6 lag_60min rows, got {nan_rows}"
        )
        assert np.all(~np.isnan(non_nan_rows)), (
            f"sno={sno}: expected non-NaN after tick 6, got {non_nan_rows}"
        )


# ---------------------------------------------------------------------------
# Test 3: offline vs online parity
# ---------------------------------------------------------------------------

def test_offline_online_parity():
    """
    Simulate the online path tick-by-tick and verify every FEATURE_COLUMN
    matches the offline build_features output to atol=1e-9.

    The online path appends one reading at a time to LagBuffer, then calls
    build_online_features for that single tick's snapshot.  We compare only
    the ticks where the offline path has no NaN (i.e. after the warm-up
    period for each station) to avoid comparing NaN vs NaN mismatches across
    independent NaN-handling paths — equality of NaN placement is separately
    covered by test_no_leakage_lag60min_first_6_rows.

    For ticks that DO have NaN in both paths (warm-up), we verify both agree
    on which values are NaN (equal_nan=True).
    """
    df = _make_mini_df()
    station_static = _make_station_static()
    spatial_lag = _make_spatial_lag(df)

    # --- Offline reference ---
    X_offline = build_features(df, station_static, spatial_lag)

    # --- Online simulation ---
    buf = LagBuffer()
    online_rows: list[pd.DataFrame] = []

    # Group by tick index (all stations share the same tick schedule)
    # df is sorted by (sno, timestamp); reshape to (N_TICKS, n_stations)
    n_stations = len(SNOS)
    for tick in range(N_TICKS):
        # Rows for this tick across all stations
        tick_indices = [tick + s * N_TICKS for s in range(n_stations)]
        snapshot = df.iloc[tick_indices].copy().reset_index(drop=True)
        spatial_snap = spatial_lag.iloc[tick_indices].reset_index(drop=True)

        # Append current values to buffer BEFORE building features
        for _, row in snapshot.iterrows():
            buf.append(str(row["sno"]), float(row["shortage_rate"]))

        X_online_tick = buf.build_online_features(snapshot, station_static, spatial_snap)
        online_rows.append(X_online_tick)

    X_online = pd.concat(online_rows, ignore_index=True)

    # The online output rows are ordered differently from offline
    # (online: tick-major, offline: station-major).
    # Re-align by re-attaching (sno, timestamp) and sorting.
    X_online["sno"] = [
        df.iloc[tick + s * N_TICKS]["sno"]
        for tick in range(N_TICKS)
        for s in range(n_stations)
    ]
    X_online["timestamp"] = [
        df.iloc[tick + s * N_TICKS]["timestamp"]
        for tick in range(N_TICKS)
        for s in range(n_stations)
    ]
    X_online = X_online.sort_values(["sno", "timestamp"]).reset_index(drop=True)
    X_online = X_online[FEATURE_COLUMNS]  # drop helper cols

    # Compare column by column with equal_nan=True
    for col in FEATURE_COLUMNS:
        a = X_offline[col].values.astype(float)
        b = X_online[col].values.astype(float)
        match = np.isclose(a, b, atol=1e-9, equal_nan=True)
        assert match.all(), (
            f"Parity failure in column '{col}':\n"
            f"  offline : {a[~match]}\n"
            f"  online  : {b[~match]}\n"
            f"  at indices: {np.where(~match)[0].tolist()}"
        )


# ---------------------------------------------------------------------------
# Test 4 / 5: TimeSeriesSplit cut-point leakage (期末架構_v2.md §5.7.4)
# ---------------------------------------------------------------------------

_BOUNDARY_N_TICKS = 20  # need > 6 ticks each side to exercise lag_60min
_VAL_START_TICK = 14    # train: ticks 0..13, val: ticks 14..19


def _make_boundary_df() -> pd.DataFrame:
    """20-tick × 3-station frame with a `tick` column for splitting."""
    rng = np.random.default_rng(42)
    rows = []
    base_ts = pd.Timestamp("2024-01-08 07:00:00", tz=TZ)

    for sno in SNOS:
        total_capacity = int(rng.integers(10, 30))
        for tick in range(_BOUNDARY_N_TICKS):
            available = int(rng.integers(0, total_capacity + 1))
            shortage_rate = round((total_capacity - available) / total_capacity, 6)
            rows.append(
                {
                    "sno": sno,
                    "timestamp": base_ts + pd.Timedelta(minutes=10 * tick),
                    "shortage_rate": shortage_rate,
                    "available_bikes": available,
                    "total_capacity": total_capacity,
                    "tick": tick,
                }
            )

    return pd.DataFrame(rows).sort_values(["sno", "timestamp"]).reset_index(drop=True)


def test_no_leakage_at_split_boundary():
    """
    Correct anti-leakage pattern: split *then* build features.

    After a TimeSeriesSplit-style cut at val_start, the first 6 ticks of
    each station in the validation slice must have NaN `lag_60min` —
    because no within-split history exists for those rows. If a future
    refactor lets cross-station or cross-split values bleed in, this
    test fails.
    """
    df = _make_boundary_df()
    station_static = _make_station_static()

    val = (
        df[df["tick"] >= _VAL_START_TICK]
        .drop(columns="tick")
        .reset_index(drop=True)
    )
    val_spatial = _make_spatial_lag(val)

    val_X = build_features(val, station_static, val_spatial)
    val_X["sno"] = val["sno"].values

    for sno in SNOS:
        lag60 = val_X[val_X["sno"] == sno]["lag_60min"].values
        first_6 = lag60[:6]
        assert np.all(np.isnan(first_6)), (
            f"sno={sno}: first 6 lag_60min in val must be NaN after splitting first, "
            f"got {first_6}"
        )


def test_leakage_when_features_built_before_split():
    """
    Anti-pattern guard: if build_features runs on the *full* DataFrame
    before the split, lag values from the train tail leak into val's
    first rows. This test proves the leakage path is real, which is
    why test_no_leakage_at_split_boundary's contract ("split first")
    is required, not optional.
    """
    df = _make_boundary_df()
    station_static = _make_station_static()
    spatial = _make_spatial_lag(df)

    full_X = build_features(df, station_static, spatial)
    full_X["sno"] = df["sno"].values
    full_X["tick"] = df["tick"].values

    val_X = full_X[full_X["tick"] >= _VAL_START_TICK]

    for sno in SNOS:
        lag60 = val_X[val_X["sno"] == sno]["lag_60min"].values
        first_6 = lag60[:6]
        assert not np.any(np.isnan(first_6)), (
            f"sno={sno}: leakage check inverted — first 6 lag_60min in val were NaN "
            f"even though features were built before split. build_features may have "
            f"changed semantics; re-evaluate the splitting contract."
        )

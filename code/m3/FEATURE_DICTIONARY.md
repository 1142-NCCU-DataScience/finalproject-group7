# Feature Dictionary — YouBike Shortage Prediction (M3)

Aligned with 期末架構_v2.md Appendix A.
Column order matches `FEATURE_COLUMNS` in `features/build_features.py` (locked).

| # | Feature | Source | Description | Unit | Range / Notes |
|---|---------|--------|-------------|------|---------------|
| 1 | `shortage_rate` | M1 (API) | Fraction of capacity that is empty: `(total_capacity - available_bikes) / total_capacity` | ratio | [0.0, 1.0]; 1.0 = fully empty |
| 2 | `available_bikes` | M1 (API) | Number of bikes currently docked at the station | count | ≥ 0 integer |
| 3 | `total_capacity` | M1 (API) | Total number of docks at the station (static per station) | count | ≥ 1 integer |
| 4 | `lag_10min` | M3 (offline: groupby shift 1; online: LagBuffer) | `shortage_rate` 10 minutes ago (1 polling tick) | ratio | [0.0, 1.0]; NaN for first tick per station |
| 5 | `lag_20min` | M3 | `shortage_rate` 20 minutes ago (2 ticks) | ratio | NaN for first 2 ticks |
| 6 | `lag_30min` | M3 | `shortage_rate` 30 minutes ago (3 ticks) | ratio | NaN for first 3 ticks |
| 7 | `lag_60min` | M3 | `shortage_rate` 60 minutes ago (6 ticks) | ratio | NaN for first 6 ticks |
| 8 | `delta_10min` | M3 | `shortage_rate - lag_10min`; positive = filling up faster | ratio diff | [-1.0, 1.0]; NaN when lag_10min is NaN |
| 9 | `delta_30min` | M3 | `shortage_rate - lag_30min` | ratio diff | [-1.0, 1.0]; NaN when lag_30min is NaN |
| 10 | `hour_sin` | M3 | Cyclical encoding of hour-of-day: `sin(2π·hour/24)` | dimensionless | [-1.0, 1.0] |
| 11 | `hour_cos` | M3 | Cyclical encoding of hour-of-day: `cos(2π·hour/24)` | dimensionless | [-1.0, 1.0] |
| 12 | `dow_sin` | M3 | Cyclical encoding of day-of-week: `sin(2π·dow/7)`, Monday=0 | dimensionless | [-1.0, 1.0] |
| 13 | `dow_cos` | M3 | Cyclical encoding of day-of-week: `cos(2π·dow/7)` | dimensionless | [-1.0, 1.0] |
| 14 | `is_rush_hour` | M3 | 1 if weekday AND (07:00–08:59 OR 17:00–18:59 Asia/Taipei), else 0 | binary | {0, 1} |
| 15 | `is_weekend` | M3 | 1 if Saturday or Sunday (dayofweek ≥ 5), else 0 | binary | {0, 1} |
| 16 | `spatial_lag_shortage` | M2 | Weighted average `shortage_rate` of neighbouring stations (spatial weight matrix defined in M2) | ratio | [0.0, 1.0] |
| 17 | `distance_to_mrt` | M3 (`data/youbike_station.csv`) | Haversine distance from station centroid to nearest MRT exit | metres | ≥ 0; precomputed by `scripts/add_distance_to_mrt.py` |

## Notes

- **Time zone**: all time features are computed from timestamps assumed to be in `Asia/Taipei` (UTC+8). No tz conversion is performed inside M3; M1 is responsible for ensuring correct tz.
- **NaN handling**: lag NaNs at the start of each station's history are intentional and expected. The training pipeline (M4) uses `TimeSeriesSplit` and drops NaN rows before fitting. The online pipeline should not encounter NaN lags after the warm-up period of 6 ticks (60 min).
- **Regenerating `distance_to_mrt`**: if either `data/youbike_station.csv` (station list) or `data/mrt_station.csv` (MRT exit coordinates, Big5-encoded with swapped lat/lng labels) is refreshed, re-run `python scripts/add_distance_to_mrt.py` to recompute and overwrite the `distance_to_mrt` column.

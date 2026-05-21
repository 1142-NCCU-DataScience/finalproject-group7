"""
Add distance_to_mrt column to data/youbike_station.csv.

For each YouBike station, computes the Haversine distance (metres) to the
nearest MRT exit listed in data/mrt_station.csv, then writes the result
back to data/youbike_station.csv with a new `distance_to_mrt` column.

Note: mrt_station.csv has its `latitude` / `longitude` column labels
swapped (latitude column contains 121.x, longitude column contains 25.x).
This script corrects that on read.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
YOUBIKE_CSV = DATA_DIR / "youbike_station.csv"
MRT_CSV     = DATA_DIR / "mrt_station.csv"


def haversine_vec(lat1: np.ndarray, lng1: np.ndarray,
                  lat2: np.ndarray, lng2: np.ndarray) -> np.ndarray:
    """Vectorised great-circle distance in metres between point arrays.

    Broadcasts: lat1/lng1 shape (n, 1), lat2/lng2 shape (m,) → output (n, m).
    """
    R = 6_371_000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lng2 - lng1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def main() -> None:
    youbike = pd.read_csv(YOUBIKE_CSV)
    # mrt_station.csv is Big5-encoded and has lat/lng column labels swapped.
    mrt_raw = pd.read_csv(MRT_CSV, encoding="big5")
    mrt = pd.DataFrame({
        "lat": mrt_raw["longitude"].astype(float),  # real latitude (25.x)
        "lng": mrt_raw["latitude"].astype(float),   # real longitude (121.x)
    })

    yb_lat = youbike["latitude"].to_numpy().reshape(-1, 1)
    yb_lng = youbike["longitude"].to_numpy().reshape(-1, 1)
    mrt_lat = mrt["lat"].to_numpy()
    mrt_lng = mrt["lng"].to_numpy()

    dist_matrix = haversine_vec(yb_lat, yb_lng, mrt_lat, mrt_lng)
    youbike["distance_to_mrt"] = np.round(dist_matrix.min(axis=1), 1)

    youbike.to_csv(YOUBIKE_CSV, index=False)
    print(f"Updated {YOUBIKE_CSV} with distance_to_mrt for {len(youbike)} stations.")
    print(f"  min = {youbike['distance_to_mrt'].min():.1f} m")
    print(f"  max = {youbike['distance_to_mrt'].max():.1f} m")
    print(f"  mean = {youbike['distance_to_mrt'].mean():.1f} m")


if __name__ == "__main__":
    main()
